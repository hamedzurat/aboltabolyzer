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
from src.config_utils import fail_on_model_error
from src.llm_verifier import GemmaVerifier
from src.rag import BanglaRAG
from src.xlmr_encoder import predict_test

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

console = Console()


def build_rag_query(row, query_mode):
    if query_mode == "prompt_response":
        return f"{row['prompt_bn']} {row['response_bn']}"
    return str(row["prompt_bn"])


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
                        "triggered_think": entry.get("triggered_think"),
                        "think_reasons": reasons,
                        "is_c0": entry.get("is_c0"),
                        "is_c1": entry.get("is_c1"),
                        "is_c2": entry.get("is_c2"),
                    }
                except Exception:
                    continue
    except Exception as e:
        console.print(f"[yellow]Could not merge verifier debug log: {e}[/yellow]")
    return cache


def main():
    console.print(
        Panel(
            "[bold yellow]Test Set Inference & Prediction Pipeline[/bold yellow]",
            border_style="bold yellow",
        )
    )

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)

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

                retrieved_contexts = []
                n_retrieved = []
                sim_max = []
                sim_mean = []
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(description="Retrieving facts...", total=num_nulls)
                    for idx, row in test_df[null_mask].iterrows():
                        query = build_rag_query(row, query_mode)
                        hits = rag.retrieve(query)
                        evidence, n_hits, max_score, mean_score = rag.format_evidence(hits)
                        retrieved_contexts.append(evidence)
                        n_retrieved.append(n_hits)
                        sim_max.append(max_score)
                        sim_mean.append(mean_score)
                        progress.advance(task)

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
    p_xlmr = predict_test(test_df, config)

    # Clean up GPU memory after Cross-Encoder prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 4. Predict with Gemma 4 Verifier (encoder prior from XLM-R)
    console.print("\n[bold cyan]Step 3: Gemma 4 Verifier Inference[/bold cyan]")
    test_df["p_xlmr"] = p_xlmr
    verifier = GemmaVerifier()
    if config.get("runtime", {}).get("use_llm_verifier", True):
        try:
            verifier.load_model()
            if not verifier.exemplar_retriever.load_index():
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
            )
        except Exception as e:
            if fail_on_model_error(config):
                raise RuntimeError("Gemma verifier failed; refusing to submit fake scores.") from e
            console.print(f"[bold red]Failed to run Gemma verifier on test set: {e}[/bold red]")
            console.print(
                "[yellow]Falling back to XLM-R-only scores because fail_on_model_error=false.[/yellow]"
            )
            p_llm = p_xlmr.copy()
    else:
        console.print(
            "[yellow]LLM verifier disabled. Using XLM-R scores as Gemma fallback.[/yellow]"
        )
        p_llm = p_xlmr.copy()

    test_df["p_llm"] = p_llm
    test_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "test_with_preds.csv"), index=False
    )

    # 5. Load threshold decision config
    decision = ThresholdDecision()
    threshold_path = "models/blender_config.pkl"
    if os.path.exists(threshold_path):
        decision.load(threshold_path)
    else:
        console.print(
            f"[bold red]Warning: Threshold config not found at {threshold_path}. "
            f"Using default threshold=0.5.[/bold red]"
        )

    # 6. Apply tuned threshold to Gemma verdicts
    console.print("\n[bold cyan]Step 4: Final Threshold Decision[/bold cyan]")
    p_final, preds = decision.predict(p_llm)

    # 7. Create submission file — timestamped run folder
    base_submission_path = config["data"][
        "submission_output_path"
    ]  # e.g. submissions/submission.csv
    submissions_dir = os.path.dirname(base_submission_path)  # e.g. submissions/
    basename = os.path.basename(base_submission_path)  # e.g. submission.csv

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(submissions_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    submission_path = os.path.join(run_dir, basename)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"] if "id" in test_df.columns else range(len(preds)),
            "label": preds,
        }
    )

    submission_df.to_csv(submission_path, index=False)

    # 8. Rich debug CSV for error analysis
    debug_path = submission_path.replace(".csv", "_debug.csv")
    debug_df = test_df.copy()
    debug_df["p_final"] = p_final
    debug_df["label"] = preds
    debug_df["threshold"] = decision.threshold
    debug_df["threshold_metric"] = decision.threshold_metric
    debug_df["used_llm_verifier"] = bool(config.get("runtime", {}).get("use_llm_verifier", True))
    debug_df["encoder_disagree"] = (debug_df["p_xlmr"] - debug_df["p_llm"]).abs()
    debug_df["rag_filled"] = (debug_df["context_original"] == "[NULL]") & (
        debug_df["context"] != "[NULL]"
    )

    verifier_map = load_verifier_debug_map()
    if verifier_map:
        debug_df["p_llm_no_think"] = np.nan
        debug_df["triggered_think"] = False
        debug_df["think_reasons"] = ""
        debug_df["is_c0"] = np.nan
        debug_df["is_c1"] = np.nan
        debug_df["is_c2"] = np.nan
        for i, row in debug_df.iterrows():
            key = (str(row["prompt_bn"]), str(row["response_bn"]))
            if key in verifier_map:
                meta = verifier_map[key]
                debug_df.at[i, "p_llm_no_think"] = meta.get("p_llm_no_think")
                debug_df.at[i, "triggered_think"] = bool(meta.get("triggered_think"))
                debug_df.at[i, "think_reasons"] = meta.get("think_reasons", "")
                debug_df.at[i, "is_c0"] = meta.get("is_c0")
                debug_df.at[i, "is_c1"] = meta.get("is_c1")
                debug_df.at[i, "is_c2"] = meta.get("is_c2")

    preferred_cols = [
        "id",
        "has_context",
        "context_original",
        "context",
        "prompt_bn",
        "response_bn",
        "n_retrieved",
        "retrieval_sim_max",
        "retrieval_sim_mean",
        "rag_filled",
        "p_xlmr",
        "p_llm_no_think",
        "p_llm",
        "p_final",
        "threshold",
        "threshold_metric",
        "encoder_disagree",
        "triggered_think",
        "think_reasons",
        "is_c0",
        "is_c1",
        "is_c2",
        "used_llm_verifier",
        "label",
    ]
    ordered = [c for c in preferred_cols if c in debug_df.columns]
    ordered += [c for c in debug_df.columns if c not in ordered]
    debug_df = debug_df[ordered]
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
