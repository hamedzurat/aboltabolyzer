import hashlib
import json
import logging
import os
import tomllib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, message=".*_check_is_size.*")


import numpy as np
import pandas as pd
import transformers
from huggingface_hub.utils import disable_progress_bars

from src.config_utils import (
    apply_runtime_settings,
    resolve_quantization_mode,
    resolve_runtime,
    resolve_section,
    validate_config,
)
from src.evidence_policy import (
    rag_fallback_source,
    rag_skip_reason,
    rag_source_for_task,
    should_use_rag,
)
from src.llm_verifier import (
    GemmaVerifier,
    verifier_case_key,
    verifier_log_matches_metadata,
)
from src.nli import nli_cache_tag
from src.rag import BanglaRAG, resolve_rag_sources, source_paths
from src.router import route_dataframe
from src.tui import (
    banner,
    count_table,
    done_panel,
    info,
    kv_table,
    ok,
    pipeline_progress,
    step,
    warn,
)

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)


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
        warn(f"Could not read prediction checkpoint {path}: {e}")
        return None

    if column not in checkpoint_df.columns:
        warn(f"Ignoring {path}; missing column '{column}'")
        return None
    if len(checkpoint_df) != expected_len:
        warn(f"Ignoring {path}; expected {expected_len} rows, found {len(checkpoint_df)}")
        return None
    if expected_ids is not None and "id" in checkpoint_df.columns:
        checkpoint_ids = checkpoint_df["id"].astype(str).tolist()
        current_ids = pd.Series(expected_ids).astype(str).tolist()
        if checkpoint_ids != current_ids:
            warn(f"Ignoring {path}; checkpoint ids do not match test ids")
            return None
    for meta_key, expected_value in (expected_metadata or {}).items():
        if meta_key not in checkpoint_df.columns:
            warn(f"Ignoring {path}; missing metadata '{meta_key}'")
            return None
        actual_values = checkpoint_df[meta_key].astype(str).unique().tolist()
        if actual_values != [str(expected_value)]:
            warn(
                f"Ignoring {path}; metadata '{meta_key}' is {actual_values}, "
                f"expected {expected_value}"
            )
            return None

    values = checkpoint_df[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        warn(f"Ignoring {path}; found non-finite predictions")
        return None
    ok(f"Loaded checkpointed {column} from {path}")
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
    ok(f"Saved {column} checkpoint → {path}")


def dataframe_cache_fingerprint(df):
    """Fingerprint fields that can change verifier outputs."""
    fields = [
        "id",
        "context",
        "context_original",
        "prompt_bn",
        "response_bn",
        "task_type",
        "rag_source",
        "rag_skipped_reason",
        "evidence_source",
    ]
    available = [field for field in fields if field in df.columns]
    payload = df[available].astype(str).to_json(orient="split", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rag_query(row, query_mode):
    if query_mode == "prompt_response":
        return f"{row['prompt_bn']} {row['response_bn']}"
    return str(row["prompt_bn"])


def apply_threshold(p_llm, threshold=0.5):
    p_llm = np.asarray(p_llm, dtype=float)
    labels = (p_llm >= float(threshold)).astype(int)
    return p_llm, labels


def validate_submission_df(submission_df):
    cols = list(submission_df.columns)
    if cols != ["id", "label"]:
        raise ValueError(f"submission must have columns ['id', 'label'], got {cols}")
    if submission_df["id"].duplicated().any():
        raise ValueError("submission ids must be unique")
    bad = ~submission_df["label"].isin([0, 1])
    if bad.any():
        raise ValueError("submission labels must be 0 or 1")
    return True


def _retrieve_many_fast(rag, queries, progress=None, task=None, chunk_size=256):
    """Batch-encode queries, then retrieve hits for each query scored as batched float32 matmuls."""
    queries = list(queries)
    if not queries:
        return []

    rag.load_model()
    index = rag.prepare_search_embeddings()
    if index is None:
        return [[] for _ in queries]
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

        sims = query_embeddings @ index.T
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


def load_verifier_debug_map(log_path="logs/debug_llm_verifier.jsonl", expected_metadata=None):
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
                    if expected_metadata and not verifier_log_matches_metadata(
                        entry, expected_metadata
                    ):
                        continue
                    key = verifier_case_key(
                        evidence=entry.get("evidence", ""),
                        prompt=entry.get("prompt", ""),
                        response=entry.get("response", ""),
                        task_type=entry.get("task_type", ""),
                        context_original=entry.get("context_original", ""),
                        metadata=expected_metadata,
                    )
                    reasons = entry.get("think_reasons", [])
                    if isinstance(reasons, list):
                        reasons = "|".join(str(r) for r in reasons)
                    cache[key] = {
                        "p_llm_no_think": entry.get("p_llm_no_think"),
                        "p_fast": entry.get("p_fast"),
                        "p_think": entry.get("p_think"),
                        "p_llm_final": entry.get("p_llm_final"),
                        "triggered_think": entry.get("triggered_think"),
                        "verdict_parsed": entry.get("verdict_parsed"),
                        "confidence_parsed": entry.get("confidence_parsed"),
                        "think_max_tokens": entry.get("think_max_tokens"),
                        "think_reasons": reasons,
                        "thinking_cot": entry.get("thinking_cot"),
                        "task_type": entry.get("task_type"),
                    }
                except Exception:
                    continue
    except Exception as e:
        warn(f"Could not merge verifier debug log: {e}")
    return cache


def apply_router_disabled_policy(test_df):
    """Baseline evidence policy: no task router and no typed RAG."""
    test_df = test_df.copy()
    if "context_original" not in test_df.columns:
        test_df["context_original"] = test_df["context"]

    test_df["context"] = test_df["context_original"]
    context_present = ~test_df["context_original"].astype(str).str.strip().isin(
        ["", "[NULL]", "None", "nan"]
    )
    test_df["task_type"] = np.where(context_present, "context_grounded_other", "other_null")
    test_df["n_retrieved"] = 0
    test_df["retrieval_sim_max"] = np.nan
    test_df["retrieval_sim_mean"] = np.nan
    test_df["rag_used"] = False
    test_df["rag_source"] = ""
    test_df["rag_skipped_reason"] = "router_disabled"
    test_df["evidence_source"] = np.where(context_present, "original_context", "none")
    test_df["evidence_relevance"] = np.where(context_present, "provided", "no_evidence")
    count_table(
        "Router disabled task baseline",
        test_df["task_type"].value_counts().to_dict(),
        key_header="task_type",
    )
    return test_df


def _resolve_available_source(config, task_type):
    """Pick preferred typed source, then fallback, if its index exists."""
    sources = resolve_rag_sources(config)
    candidates = []
    preferred = rag_source_for_task(task_type)
    if preferred:
        candidates.append(preferred)
    fallback = rag_fallback_source(task_type)
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    for name in candidates:
        if name not in sources:
            continue
        index_path = sources[name]["index_path"]
        if os.path.exists(index_path):
            return name, index_path
    if candidates:
        return candidates[0], sources.get(candidates[0], {}).get("index_path")
    return None, None


def apply_task_evidence_policy(test_df, config, verifier=None):
    """Route rows and apply typed RAG only where the task policy allows it."""
    test_df = test_df.copy()
    if "context_original" not in test_df.columns:
        test_df["context_original"] = test_df["context"]

    # Always start from original context; old evidence caches may have wrong RAG fills.
    test_df["context"] = test_df["context_original"]

    llm_routing = bool(config.get("router", {}).get("llm_routing", False))
    if llm_routing and verifier is not None:
        from src.router import route_dataframe_llm

        test_df["task_type"] = route_dataframe_llm(test_df, verifier)

        static_task_types = route_dataframe(test_df)
        disagreements = test_df["task_type"] != static_task_types
        n_disagree = int(disagreements.sum())
        total = len(test_df)

        info(
            f"LLM Router vs Static Router Disagreement: {n_disagree}/{total} ({100.0 * n_disagree / total:.1f}%)"
        )
        if n_disagree > 0:
            from collections import Counter
            from src.tui import kv_table

            mismatches = Counter()
            for l_type, s_type in zip(
                test_df["task_type"].tolist(), static_task_types.tolist(), strict=True
            ):
                if l_type != s_type:
                    mismatches[(s_type, l_type)] += 1
            kv_table(
                "Router mismatches (Static → LLM)",
                {f"{s} → {l}": str(count) for (s, l), count in mismatches.most_common(10)},
            )
    else:
        test_df["task_type"] = route_dataframe(test_df)

    test_df["n_retrieved"] = 0
    test_df["retrieval_sim_max"] = np.nan
    test_df["retrieval_sim_mean"] = np.nan
    test_df["rag_used"] = False
    test_df["rag_skipped_reason"] = ""
    test_df["rag_source"] = ""
    test_df["evidence_source"] = "original_context"
    test_df["evidence_relevance"] = "n/a"

    for i, row in test_df.iterrows():
        reason = rag_skip_reason(row["task_type"], row["context_original"])
        if reason is not None:
            test_df.at[i, "rag_skipped_reason"] = reason
            if str(row["context_original"]).strip() in ("[NULL]", "", "None", "nan"):
                test_df.at[i, "evidence_source"] = "none"
                test_df.at[i, "evidence_relevance"] = "no_evidence"
            else:
                test_df.at[i, "evidence_source"] = "original_context"
                test_df.at[i, "evidence_relevance"] = "provided"
        elif not should_use_rag(row["task_type"], row["context_original"], row["prompt_bn"]):
            test_df.at[i, "rag_skipped_reason"] = "other_null_not_factual"
            test_df.at[i, "evidence_source"] = "none"
            test_df.at[i, "evidence_relevance"] = "no_evidence"
        else:
            source_name, _ = _resolve_available_source(config, row["task_type"])
            test_df.at[i, "rag_source"] = source_name or ""

    rag_mask = test_df.apply(
        lambda r: should_use_rag(r["task_type"], r["context_original"], r["prompt_bn"]),
        axis=1,
    )
    num_rag = int(rag_mask.sum())
    task_counts = test_df["task_type"].value_counts().to_dict()
    count_table(
        "Routed task types",
        {str(k): int(v) for k, v in task_counts.items()},
        key_header="task_type",
    )
    info(f"RAG candidates this run: {num_rag}/{len(test_df)}")

    if num_rag == 0:
        warn("No rows need retrieval — all evidence is original context or policy-skipped.")
        return test_df

    query_mode = config["rag"].get("query_mode", "prompt")
    rag_rows = test_df.loc[rag_mask]
    by_source = {}
    for idx, row in rag_rows.iterrows():
        source_name, index_path = _resolve_available_source(config, row["task_type"])
        if not source_name or not index_path or not os.path.exists(index_path):
            test_df.at[idx, "rag_skipped_reason"] = (
                f"index_missing:{source_name or rag_source_for_task(row['task_type'])}"
            )
            test_df.at[idx, "evidence_source"] = "none"
            test_df.at[idx, "evidence_relevance"] = "no_evidence"
            test_df.at[idx, "rag_source"] = source_name or ""
            continue
        by_source.setdefault(source_name, []).append(idx)

    if not by_source:
        warn("No typed RAG indexes available for candidate rows (all index_missing).")
        skip_counts = (
            test_df.loc[rag_mask, "rag_skipped_reason"].value_counts().to_dict() if num_rag else {}
        )
        if skip_counts:
            count_table("RAG skip reasons", {str(k): int(v) for k, v in skip_counts.items()})
        return test_df

    count_table(
        "Retrieval by source",
        {name: len(idxs) for name, idxs in by_source.items()},
        key_header="rag_source",
    )

    for source_name, idxs in by_source.items():
        paths = source_paths(config, source_name)
        info(f"Loading index '{source_name}' ← {paths['index_path']}")
        rag = BanglaRAG(
            config=config,
            source_name=source_name,
            corpus_dir=paths["corpus_dir"],
            index_path=paths["index_path"],
        )
        if not rag.load_index():
            for idx in idxs:
                test_df.at[idx, "rag_skipped_reason"] = f"index_missing:{source_name}"
                test_df.at[idx, "evidence_source"] = "none"
                test_df.at[idx, "evidence_relevance"] = "no_evidence"
            continue

        subset = test_df.loc[idxs]
        queries = [build_rag_query(row, query_mode) for _, row in subset.iterrows()]
        with pipeline_progress() as progress:
            task = progress.add_task(f"Retrieving ({source_name})", total=len(queries))
            hits_by_query = _retrieve_many_fast(rag, queries, progress, task)

        filled = 0
        for idx, hits in zip(idxs, hits_by_query, strict=True):
            evidence, n_hits, max_score, mean_score = rag.format_evidence(hits)
            test_df.at[idx, "context"] = evidence
            test_df.at[idx, "n_retrieved"] = n_hits
            test_df.at[idx, "retrieval_sim_max"] = max_score
            test_df.at[idx, "retrieval_sim_mean"] = mean_score
            test_df.at[idx, "rag_used"] = True
            test_df.at[idx, "rag_skipped_reason"] = ""
            test_df.at[idx, "rag_source"] = source_name
            test_df.at[idx, "evidence_source"] = f"rag:{source_name}"
            if str(evidence).strip() in ("[NULL]", "", "None", "nan"):
                test_df.at[idx, "evidence_relevance"] = "retrieval_empty"
            else:
                test_df.at[idx, "evidence_relevance"] = "retrieved"
                filled += 1
        ok(f"{source_name}: retrieved for {len(idxs)} rows · non-empty evidence {filled}")

    ok("Task-aware typed evidence selection complete")
    return test_df


def build_debug_df(test_df, p_llm, threshold, ctx):
    """Build a compact debug frame for error analysis and threshold tuning."""
    p_llm = np.asarray(p_llm, dtype=float)
    _, preds = apply_threshold(p_llm, threshold)

    debug_df = pd.DataFrame(
        {
            "id": test_df["id"] if "id" in test_df.columns else range(len(test_df)),
            "label": preds,
            "p_llm": p_llm,
            "threshold": float(threshold),
            "threshold_margin": p_llm - float(threshold),
            "task_type": test_df["task_type"] if "task_type" in test_df.columns else "",
            "p_fast": np.nan,
            "p_think": np.nan,
            "triggered_think": False,
            "think_max_tokens": np.nan,
            "think_reasons": "",
            "verdict_parsed": pd.Series(None, index=test_df.index, dtype="object"),
            "confidence_parsed": pd.Series(None, index=test_df.index, dtype="object"),
            "think_changed_label": False,
            "thinking_cot": "",
            "nli_eligible": False,
            "nli_applied": False,
            "nli_skip_reason": "",
            "nli_p_entail": np.nan,
            "nli_p_contradict": np.nan,
            "nli_p_neutral": np.nan,
            "nli_margin": np.nan,
            "p_nli": np.nan,
            "rag_used": test_df["rag_used"] if "rag_used" in test_df.columns else False,
            "rag_source": test_df["rag_source"] if "rag_source" in test_df.columns else "",
            "rag_skipped_reason": (
                test_df["rag_skipped_reason"] if "rag_skipped_reason" in test_df.columns else ""
            ),
            "evidence_source": (
                test_df["evidence_source"] if "evidence_source" in test_df.columns else ""
            ),
            "evidence_relevance": (
                test_df["evidence_relevance"] if "evidence_relevance" in test_df.columns else ""
            ),
            "n_retrieved": test_df["n_retrieved"] if "n_retrieved" in test_df.columns else 0,
            "retrieval_sim_max": (
                test_df["retrieval_sim_max"] if "retrieval_sim_max" in test_df.columns else np.nan
            ),
            "retrieval_sim_mean": (
                test_df["retrieval_sim_mean"] if "retrieval_sim_mean" in test_df.columns else np.nan
            ),
            "context_original": (
                test_df["context_original"]
                if "context_original" in test_df.columns
                else test_df["context"]
            ),
            "context": test_df["context"],
            "prompt_bn": test_df["prompt_bn"],
            "response_bn": test_df["response_bn"],
            "run_timestamp": ctx["run_ts"],
            "hardware_profile": ctx["hardware_profile"],
            "gemma_model_name": resolve_section(ctx["config"], "gemma").get("fast_model_name")
            or resolve_section(ctx["config"], "gemma").get("model_name"),
            "gemma_load_in": resolve_quantization_mode(resolve_section(ctx["config"], "gemma")),
        }
    )

    verifier_map = load_verifier_debug_map(
        ctx["verifier_debug_log_path"],
        expected_metadata=ctx.get("verifier_cache_metadata"),
    )
    if verifier_map:
        for row_pos, (i, row) in enumerate(debug_df.iterrows()):
            key = verifier_case_key(
                evidence=row["context"],
                prompt=row["prompt_bn"],
                response=row["response_bn"],
                task_type=row["task_type"],
                context_original=row["context_original"],
                metadata=ctx.get("verifier_cache_metadata"),
            )
            if key not in verifier_map:
                continue
            meta = verifier_map[key]
            p_fast = meta.get("p_fast", meta.get("p_llm_no_think"))
            debug_df.at[i, "p_fast"] = p_fast
            debug_df.at[i, "p_think"] = meta.get("p_think")
            debug_df.at[i, "triggered_think"] = bool(meta.get("triggered_think"))
            debug_df.at[i, "think_max_tokens"] = meta.get("think_max_tokens")
            debug_df.at[i, "think_reasons"] = meta.get("think_reasons", "")
            debug_df.at[i, "verdict_parsed"] = meta.get("verdict_parsed")
            debug_df.at[i, "confidence_parsed"] = meta.get("confidence_parsed")
            debug_df.at[i, "thinking_cot"] = meta.get("thinking_cot") or ""
            if p_fast is not None and np.isfinite(float(p_fast)):
                debug_df.at[i, "think_changed_label"] = (float(p_llm[row_pos]) >= threshold) != (
                    float(p_fast) >= threshold
                )
            for ncol in (
                "nli_eligible",
                "nli_applied",
                "nli_skip_reason",
                "nli_p_entail",
                "nli_p_contradict",
                "nli_p_neutral",
                "nli_margin",
                "p_nli",
            ):
                if ncol in meta:
                    debug_df.at[i, ncol] = meta.get(ncol)

    # Prefer live NLI frame from this run when present
    for col in (
        "nli_eligible",
        "nli_applied",
        "nli_skip_reason",
        "nli_p_entail",
        "nli_p_contradict",
        "nli_p_neutral",
        "nli_margin",
        "p_nli",
    ):
        if col in test_df.columns:
            debug_df[col] = test_df[col].values

    return debug_df


def main():
    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)
    validate_config(config)
    apply_runtime_settings(config)
    predict_config = config.get("predict", {})
    use_checkpoints = bool(predict_config.get("use_checkpoints", True))
    force_recompute = bool(predict_config.get("force_recompute", False))
    runtime_config = resolve_runtime(config)
    hardware_profile = runtime_config.get("hardware_profile", "default")
    decision_config = config.get("decision", {})
    threshold = float(decision_config.get("threshold", 0.5))
    router_enabled = bool(config.get("router", {}).get("enabled", True))
    gemma_config = resolve_section(config, "gemma")

    banner(
        "Routed Gemma / Qwen Inference",
        "router → typed RAG → fast → NLI-first → think fallback → submission",
    )
    nli_config_preview = config.get("nli", {})
    kv_table(
        "Run config",
        {
            "hardware_profile": hardware_profile,
            "fast_verifier": gemma_config.get("fast_model_name"),
            "think_verifier": gemma_config.get("think_model_name"),
            "load_in": resolve_quantization_mode(gemma_config),
            "think_pass": gemma_config.get("enable_think_pass"),
            "nli_first": bool(nli_config_preview.get("enabled", False)),
            "threshold": threshold,
            "router": router_enabled,
            "checkpoints": use_checkpoints,
            "force_recompute": force_recompute,
        },
    )

    test_processed_path = os.path.join(config["data"]["processed_dir"], "test.csv")
    test_evidence_path = os.path.join(config["data"]["processed_dir"], "test_with_evidence.csv")

    if not os.path.exists(test_processed_path):
        warn(
            f"Processed test file not found at {test_processed_path}. Run `just preprocess` first."
        )
        return

    test_df = pd.read_csv(test_processed_path)
    if "context_original" not in test_df.columns:
        test_df["context_original"] = test_df["context"]
    info(f"Loaded test set: {len(test_df)} rows from {test_processed_path}")

    # Check if checkpoint exists and we should use it
    resume_routing = use_checkpoints and not force_recompute and os.path.exists(test_evidence_path)

    # Instantiate verifier early
    verifier = GemmaVerifier()
    nli_config = config.get("nli", {})
    nli_enabled = bool(nli_config.get("enabled", False))
    # Fingerprint NLI into verifier cache metadata before any log matching
    verifier._nli_cache_tag = nli_cache_tag(nli_config if nli_enabled else None)
    verifier_cache_metadata = verifier.cache_metadata()

    router_config = config.get("router", {})
    router_enabled = bool(router_config.get("enabled", True))
    llm_routing = bool(router_config.get("llm_routing", False))
    if router_enabled and llm_routing and not resume_routing:
        info("LLM-based routing enabled — loading verifier model early...")
        verifier.load_model()

    total_steps = 3
    step(1, total_steps, "Route tasks + select evidence")
    if resume_routing:
        info(f"Resuming routing and RAG evidence from cached file: {test_evidence_path}")
        test_df = pd.read_csv(test_evidence_path)
    else:
        if not router_enabled:
            warn("Router disabled in config — using original context only, no typed RAG.")
            test_df = apply_router_disabled_policy(test_df)
        else:
            test_df = apply_task_evidence_policy(test_df, config, verifier=verifier)
        test_df.to_csv(test_evidence_path, index=False)
        ok(f"Cached evidence frame → {test_evidence_path}")

    base_submission_path = config["data"]["submission_output_path"]
    submissions_dir = os.path.dirname(base_submission_path)
    basename = os.path.basename(base_submission_path)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(submissions_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    submission_path = os.path.join(run_dir, basename)
    partial_debug_path = os.path.join(run_dir, "submission_partial_debug.csv")
    info(f"Run folder: {run_dir}")

    partial_flush_seconds = float(predict_config.get("partial_flush_seconds", 60))
    write_debug = bool(config.get("debug", {}).get("write_debug", True))
    ids = test_df["id"] if "id" in test_df.columns else None

    ctx = {
        "config": config,
        "run_ts": run_ts,
        "hardware_profile": hardware_profile,
        "use_checkpoints": use_checkpoints,
        "force_recompute": force_recompute,
        "llm_from_checkpoint": False,
        "llm_checkpoint_source": "gemma",
        "llm_checkpoint_path": _prediction_checkpoint_path(
            config, "llm_predictions_path", "test_llm_preds.csv"
        ),
        "verifier_debug_log_path": verifier.debug_log_path,
        "verifier_cache_metadata": verifier_cache_metadata,
        "submission_path": submission_path,
        "debug_path": submission_path.replace(".csv", "_debug.csv"),
    }

    def _write_partial(n_done, preds_so_far):
        if not write_debug:
            return
        vals = np.asarray(
            [np.nan if p is None else float(p) for p in preds_so_far],
            dtype=float,
        )
        if len(vals) != len(test_df):
            # legacy: compact list of completed scores only
            partial_df = build_debug_df(
                test_df.iloc[: len(vals)].copy(), vals, threshold, ctx
            )
        else:
            partial_df = build_debug_df(test_df.copy(), vals, threshold, ctx)
            partial_df = partial_df[np.isfinite(partial_df["p_llm"])]
        if partial_df.empty:
            return
        partial_df.to_csv(partial_debug_path, index=False)
        info(f"Partial debug flushed · {len(partial_df)} rows → {partial_debug_path}")

    llm_checkpoint_path = ctx["llm_checkpoint_path"]
    p_llm = None
    llm_from_checkpoint = False
    llm_checkpoint_source = "gemma"
    checkpoint_gemma_config = resolve_section(config, "gemma")
    llm_metadata = {
        "checkpoint_source": "gemma",
        "hardware_profile": hardware_profile,
        "gemma_model_name": checkpoint_gemma_config.get("fast_model_name")
        or checkpoint_gemma_config.get("model_name"),
        "gemma_model_loader": checkpoint_gemma_config.get("model_loader"),
        "gemma_load_in": resolve_quantization_mode(checkpoint_gemma_config),
        "verifier_case_fingerprint": dataframe_cache_fingerprint(test_df),
        "pipeline": "routed_gemma",
    }
    llm_metadata.update({f"verifier_{k}": v for k, v in verifier_cache_metadata.items()})
    llm_metadata["nli_cache_tag"] = nli_cache_tag(nli_config)
    step(2, total_steps, "Verifier · fast → NLI-first → think fallback")
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
        ok("Using complete verifier checkpoint — skipping model load")
    else:
        info(
            f"Loading verifier: "
            f"{checkpoint_gemma_config.get('fast_model_name') or checkpoint_gemma_config.get('model_name')}"
        )
        verifier.load_model()
        if verifier.exemplar_top_k > 0 and not verifier.exemplar_retriever.load_index():
            train_evidence_path = os.path.join(
                config["data"]["processed_dir"], "train_with_evidence.csv"
            )
            train_path = os.path.join(config["data"]["processed_dir"], "train.csv")
            exemplar_source = None
            if os.path.exists(train_evidence_path):
                exemplar_source = train_evidence_path
            elif os.path.exists(train_path):
                exemplar_source = train_path
            if exemplar_source is not None:
                train_df = pd.read_csv(exemplar_source)
                if "label" in train_df.columns:
                    warn(f"Exemplar index missing — rebuilding from {exemplar_source}")
                    verifier.exemplar_retriever.build_index(train_df)
        if partial_flush_seconds > 0:
            info(f"Partial debug flush every {partial_flush_seconds:.0f}s")
        p_llm = verifier.predict_dataset(
            test_df,
            use_cache=use_checkpoints and not force_recompute,
            on_partial=_write_partial,
            partial_every_seconds=partial_flush_seconds,
            nli_config=nli_config if nli_enabled else None,
        )
        # Attach NLI debug columns for submission_debug.csv
        nli_debug = getattr(verifier, "last_nli_debug", None)
        if nli_debug is not None:
            for col in nli_debug.columns:
                test_df[col] = nli_debug[col].values
        if use_checkpoints:
            _save_prediction_checkpoint(
                llm_checkpoint_path,
                ids,
                "p_llm",
                p_llm,
                metadata=llm_metadata,
            )

    test_df["p_llm"] = p_llm
    test_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "test_with_preds.csv"), index=False
    )

    step(3, total_steps, "Threshold + write submission")
    info(f"Applying threshold={threshold}")
    p_final, preds = apply_threshold(p_llm, threshold)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"] if "id" in test_df.columns else range(len(preds)),
            "label": preds,
        }
    )
    validate_submission_df(submission_df)
    submission_df.to_csv(submission_path, index=False)
    ok(f"Wrote submission → {submission_path}")

    if os.path.exists(partial_debug_path):
        os.remove(partial_debug_path)

    ctx["llm_from_checkpoint"] = llm_from_checkpoint
    ctx["llm_checkpoint_source"] = llm_checkpoint_source
    if write_debug:
        debug_df = build_debug_df(test_df, p_llm, threshold, ctx)
        debug_df.to_csv(ctx["debug_path"], index=False)
        ok(f"Wrote debug → {ctx['debug_path']}")

    latest_link = os.path.join(submissions_dir, "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.abspath(run_dir), latest_link)

    n0 = int(sum(preds == 0))
    n1 = int(sum(preds == 1))
    rag_used_n = int(test_df["rag_used"].sum()) if "rag_used" in test_df.columns else 0
    think_n = 0
    nli_n = 0
    if write_debug and os.path.exists(ctx["debug_path"]):
        try:
            _dbg = pd.read_csv(ctx["debug_path"])
            think_n = int(_dbg["triggered_think"].fillna(False).sum())
            if "nli_applied" in _dbg.columns:
                nli_n = int(_dbg["nli_applied"].fillna(False).sum())
        except Exception:
            think_n = 0
            nli_n = 0

    count_table(
        "Label distribution",
        {"0 Hallucinated": n0, "1 Faithful": n1},
        key_header="label",
    )

    done_lines = [
        f"Rows: [bold]{len(preds)}[/bold]",
        f"Labels: [cyan]0={n0}[/cyan] · [green]1={n1}[/green]",
        f"RAG filled: [bold]{rag_used_n}[/bold]",
        f"Think triggered: [bold]{think_n}[/bold]",
    ]
    if nli_enabled:
        done_lines.append(f"NLI skip-think: [bold]{nli_n}[/bold]")
    done_lines.extend(
        [
            f"Submission: [bold white]{submission_path}[/bold white]",
            f"Latest link: [bold white]{latest_link}[/bold white]",
        ]
    )
    done_panel("Prediction complete", done_lines)


if __name__ == "__main__":
    main()
