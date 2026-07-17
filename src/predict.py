import gc
import json
import logging
import os
import tomllib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub.utils import disable_progress_bars
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.blender import ThresholdDecision
from src.config_utils import (
    apply_runtime_settings,
    fail_on_model_error,
    resolve_quantization_mode,
    resolve_runtime,
    resolve_section,
    use_llm_verifier,
    validate_config,
)
from src.llm_verifier import GemmaVerifier
from src.rag import BanglaRAG
from src.xlmr_encoder import predict_test

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

console = Console()


def _prediction_checkpoint_path(config, key, default_filename):
    predict_config = config.get("predict", {})
    configured_path = predict_config.get(key)
    if configured_path:
        return configured_path
    return os.path.join(config["data"]["processed_dir"], default_filename)


def _load_prediction_checkpoint(
    path,
    expected_len,
    column,
    expected_ids=None,
    expected_metadata=None,
):
    if not os.path.exists(path):
        return None
    try:
        checkpoint_df = pd.read_csv(path)
    except Exception as e:
        console.print(f"[yellow]Could not read prediction checkpoint {path}: {e}[/yellow]")
        return None

    if column not in checkpoint_df.columns:
        console.print(f"[yellow]Ignoring {path}; missing column '{column}'.[/yellow]")
        return None
    if len(checkpoint_df) != expected_len:
        console.print(
            f"[yellow]Ignoring {path}; expected {expected_len} rows, found "
            f"{len(checkpoint_df)}.[/yellow]"
        )
        return None
    if expected_ids is not None and "id" in checkpoint_df.columns:
        checkpoint_ids = checkpoint_df["id"].astype(str).tolist()
        current_ids = pd.Series(expected_ids).astype(str).tolist()
        if checkpoint_ids != current_ids:
            console.print(
                f"[yellow]Ignoring {path}; checkpoint ids do not match test ids.[/yellow]"
            )
            return None
    for meta_key, expected_value in (expected_metadata or {}).items():
        if meta_key not in checkpoint_df.columns:
            console.print(f"[yellow]Ignoring {path}; missing metadata '{meta_key}'.[/yellow]")
            return None
        actual_values = checkpoint_df[meta_key].astype(str).unique().tolist()
        if actual_values != [str(expected_value)]:
            console.print(
                f"[yellow]Ignoring {path}; metadata '{meta_key}' is {actual_values}, "
                f"expected {expected_value}.[/yellow]"
            )
            return None

    values = checkpoint_df[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        console.print(f"[yellow]Ignoring {path}; found non-finite predictions.[/yellow]")
        return None
    console.print(f"[bold green]Loaded checkpointed {column} from {path}.[/bold green]")
    return values


def _save_prediction_checkpoint(path, ids, column, values, metadata=None):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_df = pd.DataFrame({column: values})
    if ids is not None:
        checkpoint_df.insert(0, "id", ids)
    for meta_key, meta_value in (metadata or {}).items():
        checkpoint_df[meta_key] = meta_value
    checkpoint_df.to_csv(path, index=False)
    console.print(f"[green]Saved {column} checkpoint to {path}.[/green]")


def build_rag_query(row, query_mode):
    if query_mode == "prompt_response":
        return f"{row['prompt_bn']} {row['response_bn']}"
    return str(row["prompt_bn"])


def _retrieve_many_fast(rag, queries, progress=None, task=None, chunk_size=256):
    """Same result as rag.retrieve_many, scored as batched float32 matmuls.

    The index is stored float16 (the encoder is halved on CUDA), and NumPy has no
    BLAS kernel for float16 — it falls back to an unvectorized loop. Casting the
    index to float32 once and scoring a whole query chunk in one matmul is ~200x
    faster than a float16 matvec per query.
    """
    queries = list(queries)
    if not queries:
        return []

    rag.load_model()
    index = np.ascontiguousarray(rag.embeddings, dtype=np.float32)
    top_k = rag.top_k
    threshold = rag.similarity_threshold

    all_results = []
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        query_embeddings = np.asarray(
            rag.model.encode(
                chunk,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=rag.query_batch_size,
            ),
            dtype=np.float32,
        )

        sims = query_embeddings @ index.T  # (chunk, n_passages)
        k = min(top_k, sims.shape[1])
        candidates = np.argpartition(sims, -k, axis=1)[:, -k:]

        for row_idx in range(sims.shape[0]):
            row_sims = sims[row_idx]
            idx = candidates[row_idx]
            idx = idx[np.argsort(row_sims[idx])[::-1]]
            all_results.append(
                [
                    {"text": rag.passages[i], "score": float(row_sims[i])}
                    for i in idx
                    if row_sims[i] >= threshold
                ]
            )

        if progress is not None and task is not None:
            progress.advance(task, len(chunk))

    return all_results


def load_verifier_debug_map(log_path="logs/debug_llm_verifier.jsonl"):
    """Map (prompt, response) -> latest verifier debug fields."""
    cache = {}
    if not os.path.exists(log_path):
        return cache
    try:
        with open(log_path, "r", encoding="utf-8") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = (str(entry.get("prompt", "")), str(entry.get("response", "")))
                    reasons = entry.get("think_reasons", [])
                    if isinstance(reasons, list):
                        reasons = "|".join(str(r) for r in reasons)
                    cache[key] = {
                        "p_llm_no_think": entry.get("p_llm_no_think"),
                        "p_llm_final": entry.get("p_llm_final"),
                        "triggered_think": entry.get("triggered_think"),
                        "verdict_parsed": entry.get("verdict_parsed"),
                        "think_reasons": reasons,
                        "thinking_cot": entry.get("thinking_cot"),
                        "is_c0": entry.get("is_c0"),
                        "is_c1": entry.get("is_c1"),
                        "is_c2": entry.get("is_c2"),
                    }
                except Exception:
                    continue
    except Exception as e:
        console.print(f"[yellow]Could not merge verifier debug log: {e}[/yellow]")
    return cache


def build_debug_df(test_df, p_llm, decision, ctx):
    """Build the full debug frame for `test_df` scored with `p_llm`.

    Shared by the end-of-run debug CSV and the periodic partial flush, so a
    partial file has exactly the same schema as the final one. `test_df` may be
    a head-slice of the full set; every column is derived per-row or from ctx.
    """
    p_llm = np.asarray(p_llm, dtype=float)
    p_final, preds = decision.predict(p_llm)
    debug_df = test_df.copy()
    debug_df["p_llm"] = p_llm
    debug_df["p_final"] = p_final
    debug_df["label"] = preds
    debug_df["threshold"] = decision.threshold
    debug_df["threshold_metric"] = decision.threshold_metric
    debug_df["threshold_margin"] = debug_df["p_final"] - float(decision.threshold)
    debug_df["threshold_abs_margin"] = debug_df["threshold_margin"].abs()
    debug_df["used_llm_verifier"] = use_llm_verifier(ctx["config"])
    debug_df["encoder_disagree"] = (debug_df["p_xlmr"] - debug_df["p_llm"]).abs()
    debug_df["llm_minus_xlmr"] = debug_df["p_llm"] - debug_df["p_xlmr"]
    debug_df["xlmr_label_at_threshold"] = (debug_df["p_xlmr"] >= decision.threshold).astype(int)
    debug_df["llm_label_at_threshold"] = (debug_df["p_llm"] >= decision.threshold).astype(int)
    debug_df["xlmr_llm_label_disagree"] = (
        debug_df["xlmr_label_at_threshold"] != debug_df["llm_label_at_threshold"]
    )
    debug_df["rag_filled"] = (debug_df["context_original"] == "[NULL]") & (
        debug_df["context"] != "[NULL]"
    )
    debug_df["evidence_is_null"] = (
        debug_df["context"].astype(str).str.strip().isin(("[NULL]", "", "None", "nan"))
    )
    debug_df["context_original_is_null"] = (
        debug_df["context_original"].astype(str).str.strip().isin(("[NULL]", "", "None", "nan"))
    )
    debug_df["context_char_len"] = debug_df["context"].astype(str).str.len()
    debug_df["context_word_len"] = debug_df["context"].astype(str).str.split().str.len()
    debug_df["prompt_char_len"] = debug_df["prompt_bn"].astype(str).str.len()
    debug_df["response_char_len"] = debug_df["response_bn"].astype(str).str.len()
    debug_df["prompt_response_char_len"] = (
        debug_df["prompt_char_len"] + debug_df["response_char_len"]
    )

    xlmr_config = resolve_section(ctx["config"], "xlmr")
    gemma_config = resolve_section(ctx["config"], "gemma")
    rag_config = resolve_section(ctx["config"], "rag")
    debug_df["run_timestamp"] = ctx["run_ts"]
    debug_df["hardware_profile"] = ctx["hardware_profile"]
    debug_df["num_folds"] = ctx["config"].get("num_folds")
    debug_df["seed"] = ctx["config"].get("seed")
    debug_df["use_checkpoints"] = ctx["use_checkpoints"]
    debug_df["force_recompute"] = ctx["force_recompute"]
    debug_df["xlmr_from_checkpoint"] = ctx["xlmr_from_checkpoint"]
    debug_df["llm_from_checkpoint"] = ctx["llm_from_checkpoint"]
    debug_df["llm_checkpoint_source"] = ctx["llm_checkpoint_source"]
    debug_df["xlmr_checkpoint_path"] = ctx["xlmr_checkpoint_path"]
    debug_df["llm_checkpoint_path"] = ctx["llm_checkpoint_path"]
    debug_df["verifier_debug_log_path"] = ctx["verifier_debug_log_path"]
    debug_df["threshold_path"] = ctx["threshold_path"]
    debug_df["submission_path"] = ctx["submission_path"]
    debug_df["debug_path"] = ctx["debug_path"]
    debug_df["xlmr_model_name"] = xlmr_config.get("model_name")
    debug_df["xlmr_max_length"] = xlmr_config.get("max_length")
    debug_df["xlmr_batch_size"] = xlmr_config.get("batch_size")
    debug_df["xlmr_use_amp"] = xlmr_config.get("use_amp")
    debug_df["xlmr_num_workers"] = xlmr_config.get("num_workers")
    debug_df["xlmr_pin_memory"] = xlmr_config.get("pin_memory")
    debug_df["gemma_model_name"] = gemma_config.get("model_name")
    debug_df["gemma_model_loader"] = gemma_config.get("model_loader")
    debug_df["gemma_device_map"] = gemma_config.get("device_map")
    debug_df["gemma_cuda_max_memory"] = gemma_config.get("cuda_max_memory")
    debug_df["gemma_max_input_tokens"] = gemma_config.get("max_input_tokens")
    debug_df["gemma_max_think_tokens"] = gemma_config.get("max_think_tokens")
    debug_df["gemma_exemplar_top_k"] = gemma_config.get("exemplar_top_k")
    debug_df["gemma_load_in"] = resolve_quantization_mode(gemma_config)
    debug_df["gemma_int8_cpu_offload"] = gemma_config.get("llm_int8_enable_fp32_cpu_offload")
    debug_df["gemma_use_inputs_embeds_for_forward"] = gemma_config.get(
        "use_inputs_embeds_for_forward"
    )
    debug_df["gemma_use_language_model_direct"] = gemma_config.get("use_language_model_direct")
    debug_df["gemma_classify_cultural_band"] = gemma_config.get("classify_cultural_band")
    debug_df["gemma_enable_think_pass"] = gemma_config.get("enable_think_pass")
    debug_df["gemma_think_reasoning_language"] = gemma_config.get("think_reasoning_language")
    debug_df["rag_query_mode"] = rag_config.get("query_mode")
    debug_df["rag_top_k"] = rag_config.get("top_k")
    debug_df["rag_query_batch_size"] = rag_config.get("query_batch_size")
    debug_df["rag_similarity_threshold"] = rag_config.get("similarity_threshold")
    debug_df["rag_max_evidence_tokens"] = rag_config.get("max_evidence_tokens")

    debug_df["p_llm_no_think"] = np.nan
    debug_df["p_llm_from_log"] = np.nan
    debug_df["p_llm_log_delta"] = np.nan
    debug_df["triggered_think"] = False
    # Object dtype, not float: this holds True/False/None, and pandas 3 refuses to
    # assign a bool into a float64 column.
    debug_df["verdict_parsed"] = pd.Series(None, index=debug_df.index, dtype="object")
    debug_df["think_reasons"] = ""
    debug_df["thinking_cot"] = ""
    debug_df["is_c0"] = np.nan
    debug_df["is_c1"] = np.nan
    debug_df["is_c2"] = np.nan
    debug_df["think_changed_probability"] = np.nan
    debug_df["think_changed_label"] = False

    verifier_map = load_verifier_debug_map(ctx["verifier_debug_log_path"])
    if verifier_map:
        for i, row in debug_df.iterrows():
            key = (str(row["prompt_bn"]), str(row["response_bn"]))
            if key in verifier_map:
                meta = verifier_map[key]
                debug_df.at[i, "p_llm_no_think"] = meta.get("p_llm_no_think")
                debug_df.at[i, "p_llm_from_log"] = meta.get("p_llm_final")
                debug_df.at[i, "triggered_think"] = bool(meta.get("triggered_think"))
                debug_df.at[i, "verdict_parsed"] = meta.get("verdict_parsed")
                debug_df.at[i, "think_reasons"] = meta.get("think_reasons", "")
                debug_df.at[i, "thinking_cot"] = meta.get("thinking_cot") or ""
                debug_df.at[i, "is_c0"] = meta.get("is_c0")
                debug_df.at[i, "is_c1"] = meta.get("is_c1")
                debug_df.at[i, "is_c2"] = meta.get("is_c2")
        has_no_think = debug_df["p_llm_no_think"].notna()
        has_log_prob = debug_df["p_llm_from_log"].notna()
        debug_df.loc[has_log_prob, "p_llm_log_delta"] = (
            debug_df.loc[has_log_prob, "p_llm"] - debug_df.loc[has_log_prob, "p_llm_from_log"]
        )
        debug_df.loc[has_no_think, "think_changed_probability"] = (
            debug_df.loc[has_no_think, "p_llm"] - debug_df.loc[has_no_think, "p_llm_no_think"]
        )
        debug_df.loc[has_no_think, "think_changed_label"] = (
            debug_df.loc[has_no_think, "p_llm"] >= decision.threshold
        ) != (debug_df.loc[has_no_think, "p_llm_no_think"] >= decision.threshold)

    preferred_cols = [
        "id",
        "label",
        "p_final",
        "threshold",
        "threshold_margin",
        "threshold_abs_margin",
        "threshold_metric",
        "p_llm",
        "p_llm_no_think",
        "p_llm_from_log",
        "p_llm_log_delta",
        "p_xlmr",
        "llm_minus_xlmr",
        "encoder_disagree",
        "xlmr_label_at_threshold",
        "llm_label_at_threshold",
        "xlmr_llm_label_disagree",
        "triggered_think",
        "verdict_parsed",
        "think_reasons",
        "think_changed_probability",
        "think_changed_label",
        "is_c0",
        "is_c1",
        "is_c2",
        "has_context",
        "evidence_is_null",
        "context_original_is_null",
        "rag_filled",
        "n_retrieved",
        "retrieval_sim_max",
        "retrieval_sim_mean",
        "context_char_len",
        "context_word_len",
        "prompt_char_len",
        "response_char_len",
        "prompt_response_char_len",
        "context_original",
        "context",
        "prompt_bn",
        "response_bn",
        "thinking_cot",
        "run_timestamp",
        "hardware_profile",
        "used_llm_verifier",
        "llm_checkpoint_source",
        "xlmr_from_checkpoint",
        "llm_from_checkpoint",
        "use_checkpoints",
        "force_recompute",
        "num_folds",
        "seed",
        "xlmr_model_name",
        "xlmr_max_length",
        "xlmr_batch_size",
        "xlmr_use_amp",
        "xlmr_num_workers",
        "xlmr_pin_memory",
        "gemma_model_name",
        "gemma_model_loader",
        "gemma_device_map",
        "gemma_cuda_max_memory",
        "gemma_max_input_tokens",
        "gemma_max_think_tokens",
        "gemma_exemplar_top_k",
        "gemma_load_in",
        "gemma_int8_cpu_offload",
        "gemma_use_inputs_embeds_for_forward",
        "gemma_use_language_model_direct",
        "gemma_classify_cultural_band",
        "gemma_enable_think_pass",
        "gemma_think_reasoning_language",
        "rag_query_mode",
        "rag_top_k",
        "rag_query_batch_size",
        "rag_similarity_threshold",
        "rag_max_evidence_tokens",
        "xlmr_checkpoint_path",
        "llm_checkpoint_path",
        "verifier_debug_log_path",
        "threshold_path",
        "submission_path",
        "debug_path",
    ]
    ordered = [c for c in preferred_cols if c in debug_df.columns]
    ordered += [c for c in debug_df.columns if c not in ordered]
    debug_df = debug_df[ordered]
    return debug_df


def main():
    console.print(
        Panel(
            "[bold yellow]Test Set Inference & Prediction Pipeline[/bold yellow]",
            border_style="bold yellow",
        )
    )

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)
    validate_config(config)
    apply_runtime_settings(config)
    predict_config = config.get("predict", {})
    use_checkpoints = bool(predict_config.get("use_checkpoints", True))
    force_recompute = bool(predict_config.get("force_recompute", False))
    runtime_config = resolve_runtime(config)
    hardware_profile = runtime_config.get("hardware_profile", "default")

    # 1. Load preprocessed test dataset
    test_processed_path = os.path.join(config["data"]["processed_dir"], "test.csv")
    test_evidence_path = os.path.join(config["data"]["processed_dir"], "test_with_evidence.csv")

    if os.path.exists(test_evidence_path):
        console.print(
            f"[bold green]Found existing {test_evidence_path}. Loading cached contexts...[/bold green]"
        )
        test_df = pd.read_csv(test_evidence_path)
        if "context_original" not in test_df.columns:
            test_df["context_original"] = test_df["context"]
        for col, default in (
            ("n_retrieved", 0),
            ("retrieval_sim_max", np.nan),
            ("retrieval_sim_mean", np.nan),
        ):
            if col not in test_df.columns:
                test_df[col] = default
    else:
        if not os.path.exists(test_processed_path):
            console.print(
                f"[bold red]Error: Processed test file not found at {test_processed_path}. Run preprocessing first.[/bold red]"
            )
            return

        test_df = pd.read_csv(test_processed_path)
        test_df["context_original"] = test_df["context"]
        test_df["n_retrieved"] = 0
        test_df["retrieval_sim_max"] = np.nan
        test_df["retrieval_sim_mean"] = np.nan

        # 2. Retrieve evidence for test rows if context is NULL
        console.print("\n[bold cyan]Step 1: Retrieve context facts for NULL test rows[/bold cyan]")
        null_mask = test_df["context"] == "[NULL]"
        num_nulls = null_mask.sum()

        if num_nulls > 0:
            index_path = config["rag"]["index_path"]
            query_mode = config["rag"].get("query_mode", "prompt")
            if os.path.exists(index_path):
                console.print(
                    f"Dense RAG index found. Retrieving evidence for [bold yellow]{num_nulls}[/bold yellow] test rows..."
                )
                rag = BanglaRAG()
                rag.load_index()

                null_rows = test_df[null_mask]
                queries = [build_rag_query(row, query_mode) for _, row in null_rows.iterrows()]
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(description="Retrieving facts...", total=num_nulls)
                    hits_by_query = _retrieve_many_fast(rag, queries, progress, task)

                retrieved_contexts = []
                n_retrieved = []
                sim_max = []
                sim_mean = []
                for hits in hits_by_query:
                    evidence, n_hits, max_score, mean_score = rag.format_evidence(hits)
                    retrieved_contexts.append(evidence)
                    n_retrieved.append(n_hits)
                    sim_max.append(max_score)
                    sim_mean.append(mean_score)

                test_df.loc[null_mask, "context"] = retrieved_contexts
                test_df.loc[null_mask, "n_retrieved"] = n_retrieved
                test_df.loc[null_mask, "retrieval_sim_max"] = sim_max
                test_df.loc[null_mask, "retrieval_sim_mean"] = sim_mean
                console.print("[green]✔ Evidence retrieval complete.[/green]")
            else:
                console.print(
                    f"[bold red]WARNING: Dense RAG index not found at {index_path}.[/bold red]"
                )
                console.print("NULL-context test rows will remain ungrounded.")

        test_df.to_csv(test_evidence_path, index=False)

    # 3. Predict with XLM-RoBERTa Cross-Encoder
    console.print("\n[bold cyan]Step 2: XLM-RoBERTa Ensemble Inference[/bold cyan]")
    ids = test_df["id"] if "id" in test_df.columns else None
    xlmr_checkpoint_path = _prediction_checkpoint_path(
        config, "xlmr_predictions_path", "test_xlmr_preds.csv"
    )
    p_xlmr = None
    xlmr_from_checkpoint = False
    xlmr_metadata = {"checkpoint_source": "xlmr", "hardware_profile": hardware_profile}
    if use_checkpoints and not force_recompute:
        p_xlmr = _load_prediction_checkpoint(
            xlmr_checkpoint_path,
            len(test_df),
            "p_xlmr",
            expected_ids=ids,
            expected_metadata=xlmr_metadata,
        )
        xlmr_from_checkpoint = p_xlmr is not None
    if p_xlmr is None:
        p_xlmr = predict_test(test_df, config)
        if use_checkpoints:
            _save_prediction_checkpoint(
                xlmr_checkpoint_path,
                ids,
                "p_xlmr",
                p_xlmr,
                metadata=xlmr_metadata,
            )

    # Clean up GPU memory after Cross-Encoder prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 4. Set up the run folder and threshold before Gemma, so the verifier can
    # flush partial submissions as it goes instead of only at the end.
    base_submission_path = config["data"]["submission_output_path"]
    submissions_dir = os.path.dirname(base_submission_path)
    basename = os.path.basename(base_submission_path)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(submissions_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    submission_path = os.path.join(run_dir, basename)
    partial_path = os.path.join(run_dir, "submission_partial.csv")

    decision = ThresholdDecision()
    threshold_path = "models/blender_config.pkl"
    if os.path.exists(threshold_path):
        decision.load(threshold_path)
    else:
        console.print(
            f"[bold red]Warning: Threshold config not found at {threshold_path}. "
            f"Using default threshold=0.5.[/bold red]"
        )

    partial_flush_every = int(predict_config.get("partial_flush_every", 0))

    test_df["p_xlmr"] = p_xlmr
    verifier = GemmaVerifier()

    # Shared by the partial flush and the final debug CSV so both have one schema.
    # Mutated as the run learns its own provenance (llm_checkpoint_source, etc.).
    ctx = {
        "config": config,
        "run_ts": run_ts,
        "hardware_profile": hardware_profile,
        "use_checkpoints": use_checkpoints,
        "force_recompute": force_recompute,
        "xlmr_from_checkpoint": xlmr_from_checkpoint,
        "llm_from_checkpoint": False,
        "llm_checkpoint_source": "gemma",
        "xlmr_checkpoint_path": xlmr_checkpoint_path,
        "llm_checkpoint_path": _prediction_checkpoint_path(
            config, "llm_predictions_path", "test_llm_preds.csv"
        ),
        "verifier_debug_log_path": verifier.debug_log_path,
        "threshold_path": threshold_path,
        "submission_path": submission_path,
        "debug_path": submission_path.replace(".csv", "_debug.csv"),
    }

    def _write_partial(n_done, preds_so_far):
        partial_df = build_debug_df(
            test_df.iloc[:n_done].copy(), preds_so_far, decision, ctx
        )
        partial_df.to_csv(partial_path, index=False)

    # 5. Predict with Gemma 4 Verifier (encoder prior from XLM-R)
    console.print("\n[bold cyan]Step 3: Gemma 4 Verifier Inference[/bold cyan]")
    if use_llm_verifier(config):
        llm_checkpoint_path = ctx["llm_checkpoint_path"]
        p_llm = None
        llm_from_checkpoint = False
        llm_checkpoint_source = "gemma"
        checkpoint_gemma_config = resolve_section(config, "gemma")
        llm_metadata = {
            "checkpoint_source": "gemma",
            "hardware_profile": hardware_profile,
            "gemma_model_name": checkpoint_gemma_config.get("model_name"),
            "gemma_model_loader": checkpoint_gemma_config.get("model_loader"),
            "gemma_load_in": resolve_quantization_mode(checkpoint_gemma_config),
        }
        if use_checkpoints and not force_recompute:
            p_llm = _load_prediction_checkpoint(
                llm_checkpoint_path,
                len(test_df),
                "p_llm",
                expected_ids=ids,
                expected_metadata=llm_metadata,
            )
            llm_from_checkpoint = p_llm is not None
        if p_llm is not None:
            console.print(
                "[bold green]Skipping Gemma; verifier checkpoint is complete.[/bold green]"
            )
        else:
            try:
                verifier.load_model()
                if verifier.exemplar_top_k > 0 and not verifier.exemplar_retriever.load_index():
                    train_evidence_path = os.path.join(
                        config["data"]["processed_dir"], "train_with_evidence.csv"
                    )
                    if os.path.exists(train_evidence_path):
                        train_df = pd.read_csv(train_evidence_path)
                        if "label" in train_df.columns:
                            console.print(
                                "[yellow]Exemplar index missing; rebuilding from labeled train data.[/yellow]"
                            )
                            verifier.exemplar_retriever.build_index(train_df)
                p_llm = verifier.predict_dataset(
                    test_df,
                    p_xlmr=p_xlmr,
                    use_cache=True,
                    on_partial=_write_partial,
                    partial_every=partial_flush_every,
                )
                if use_checkpoints:
                    _save_prediction_checkpoint(
                        llm_checkpoint_path,
                        ids,
                        "p_llm",
                        p_llm,
                        metadata=llm_metadata,
                    )
            except Exception as e:
                if fail_on_model_error(config):
                    raise RuntimeError(
                        "Gemma verifier failed; refusing to submit fake scores."
                    ) from e
                console.print(f"[bold red]Failed to run Gemma verifier on test set: {e}[/bold red]")
                console.print(
                    "[yellow]Falling back to XLM-R-only scores because fail_on_model_error=false.[/yellow]"
                )
                p_llm = p_xlmr.copy()
                llm_checkpoint_source = "xlmr_fallback_error"
    else:
        console.print(
            "[yellow]LLM verifier disabled. Using XLM-R scores as Gemma fallback.[/yellow]"
        )
        p_llm = p_xlmr.copy()
        llm_from_checkpoint = False
        llm_checkpoint_source = "xlmr_fallback_disabled"
        llm_checkpoint_path = _prediction_checkpoint_path(
            config, "llm_predictions_path", "test_llm_preds.csv"
        )
        if use_checkpoints:
            _save_prediction_checkpoint(
                llm_checkpoint_path,
                ids,
                "p_llm",
                p_llm,
                metadata={
                    "checkpoint_source": "xlmr_fallback",
                    "hardware_profile": hardware_profile,
                },
            )

    test_df["p_llm"] = p_llm
    test_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "test_with_preds.csv"), index=False
    )

    # 6. Apply tuned threshold to Gemma verdicts (threshold loaded before Gemma)
    console.print("\n[bold cyan]Step 4: Final Threshold Decision[/bold cyan]")
    p_final, preds = decision.predict(p_llm)

    # 7. Create submission file in the run folder created before Gemma
    submission_df = pd.DataFrame(
        {
            "id": test_df["id"] if "id" in test_df.columns else range(len(preds)),
            "label": preds,
        }
    )

    submission_df.to_csv(submission_path, index=False)

    # The complete submission supersedes the partial one.
    if os.path.exists(partial_path):
        os.remove(partial_path)

    debug_path = ctx["debug_path"]
    ctx["llm_from_checkpoint"] = llm_from_checkpoint
    ctx["llm_checkpoint_source"] = llm_checkpoint_source
    debug_df = build_debug_df(test_df, p_llm, decision, ctx)
    debug_df.to_csv(debug_path, index=False)
    console.print(f"Saved detailed debug submission to: [bold white]{debug_path}[/bold white]")

    # Create/update 'latest' symlink for convenience  (submissions/latest -> 20250716_123456/)
    latest_link = os.path.join(submissions_dir, "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.abspath(run_dir), latest_link)

    console.print(
        Panel(
            f"[bold green]✔ Prediction Pipeline Complete![/bold green]\n"
            f"Saved final submission to: [bold white]{submission_path}[/bold white]\n"
            f"Label distribution: [bold cyan]0 (Hallucinated): {sum(preds == 0)}[/bold cyan] | [bold green]1 (Faithful): {sum(preds == 1)}[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
