import gc
import logging
import os
import tomllib

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub.utils import disable_progress_bars
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sklearn.model_selection import StratifiedKFold

from src.blender import ThresholdDecision
from src.config_utils import apply_runtime_settings, fail_on_model_error
from src.evaluate import compute_metrics
from src.llm_verifier import GemmaVerifier
from src.preprocess import main as run_preprocess
from src.rag import BanglaRAG
from src.xlmr_encoder import train_cross_validation

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


def main():
    console.print(
        Panel(
            "[bold yellow]Start Training & Optimization Pipeline[/bold yellow]",
            border_style="bold yellow",
        )
    )

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)
    apply_runtime_settings(config)

    # 1. Preprocess raw data
    train_processed_path = os.path.join(config["data"]["processed_dir"], "train.csv")
    train_evidence_path = os.path.join(config["data"]["processed_dir"], "train_with_evidence.csv")

    if os.path.exists(train_evidence_path):
        console.print(
            f"[bold green]Found existing {train_evidence_path}. Loading cached contexts...[/bold green]"
        )
        train_df = pd.read_csv(train_evidence_path)
        if "context_original" not in train_df.columns:
            train_df["context_original"] = train_df["context"]
        for col, default in (
            ("n_retrieved", 0),
            ("retrieval_sim_max", np.nan),
            ("retrieval_sim_mean", np.nan),
        ):
            if col not in train_df.columns:
                train_df[col] = default
    else:
        if not os.path.exists(train_processed_path):
            console.print("[yellow]Processed files not found. Executing preprocessing...[/yellow]")
            run_preprocess()

        train_df = pd.read_csv(train_processed_path)
        train_df["context_original"] = train_df["context"]
        train_df["n_retrieved"] = 0
        train_df["retrieval_sim_max"] = np.nan
        train_df["retrieval_sim_mean"] = np.nan

        # 2. RAG retrieval for NULL-context rows
        console.print("\n[bold cyan]Step 1: Context Evidence Retrieval[/bold cyan]")
        null_mask = train_df["context"] == "[NULL]"
        num_nulls = null_mask.sum()

        if num_nulls > 0:
            index_path = config["rag"]["index_path"]
            query_mode = config["rag"].get("query_mode", "prompt")
            if os.path.exists(index_path):
                console.print(
                    f"Dense RAG index found. Retrieving evidence for [bold yellow]{num_nulls}[/bold yellow] NULL-context rows..."
                )
                rag = BanglaRAG()
                rag.load_index()

                null_rows = train_df[null_mask]
                queries = [build_rag_query(row, query_mode) for _, row in null_rows.iterrows()]
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(description="Retrieving facts...", total=num_nulls)
                    hits_by_query = rag.retrieve_many(queries)
                    progress.advance(task, num_nulls)

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

                train_df.loc[null_mask, "context"] = retrieved_contexts
                train_df.loc[null_mask, "n_retrieved"] = n_retrieved
                train_df.loc[null_mask, "retrieval_sim_max"] = sim_max
                train_df.loc[null_mask, "retrieval_sim_mean"] = sim_mean
                console.print("[green]✔ Context evidence retrieval complete.[/green]")
            else:
                console.print(
                    f"[bold red]WARNING: Dense RAG index not found at {index_path}.[/bold red]"
                )
                console.print(
                    "NULL-context rows will remain ungrounded. Build the RAG index if a corpus is available."
                )

        os.makedirs(config["data"]["processed_dir"], exist_ok=True)
        train_df.to_csv(train_evidence_path, index=False)

    # 3. Train XLM-RoBERTa Cross-Encoder
    console.print("\n[bold cyan]Step 2: Training XLM-RoBERTa Cross-Encoder[/bold cyan]")
    p_xlmr = train_cross_validation(train_df, config)
    train_df["p_xlmr"] = p_xlmr

    # Clean up GPU memory after Cross-Encoder training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 4. Fold-isolated OOF Gemma verifier scores (encoder prior from OOF p_xlmr)
    console.print("\n[bold cyan]Step 3: Generating OOF Gemma Verifier Scores[/bold cyan]")

    if config.get("runtime", {}).get("use_llm_verifier", True):
        verifier = GemmaVerifier()
        oof_p_llm = np.zeros(len(train_df))

        try:
            verifier.load_model()
            skf = StratifiedKFold(
                n_splits=config["num_folds"], shuffle=True, random_state=config["seed"]
            )

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(train_df, train_df["label"]), start=1
            ):
                console.print(
                    f"\n[bold yellow]Gemma OOF fold {fold}/{config['num_folds']}[/bold yellow]"
                )
                fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
                fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

                verifier.exemplar_retriever.build_index(fold_train_df)
                fold_p_llm = verifier.predict_dataset(
                    fold_val_df,
                    p_xlmr=fold_val_df["p_xlmr"].values,
                    use_cache=False,
                    debug_log_path=f"logs/debug_llm_verifier_oof_fold_{fold}.jsonl",
                )

                oof_p_llm[val_idx] = fold_p_llm

            console.print("\n[bold cyan]Building full exemplar index for inference...[/bold cyan]")
            verifier.exemplar_retriever.build_index(train_df)

            p_llm = oof_p_llm
        except Exception as e:
            if fail_on_model_error(config):
                raise RuntimeError(
                    "Gemma verifier failed; refusing to train on fake scores."
                ) from e
            console.print(f"[bold red]Failed to run Gemma verifier: {e}[/bold red]")
            console.print(
                "[yellow]Falling back to XLM-R-only scores because fail_on_model_error=false.[/yellow]"
            )
            p_llm = p_xlmr.copy()
    else:
        console.print(
            "[yellow]LLM verifier disabled. Using XLM-R scores as Gemma fallback.[/yellow]"
        )
        p_llm = p_xlmr.copy()

    train_df["p_llm"] = p_llm
    train_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "train_with_preds.csv"), index=False
    )

    # 5. Fit OOF threshold on Gemma verdicts
    console.print("\n[bold cyan]Step 4: Tuning Threshold on OOF Gemma Scores[/bold cyan]")
    y_true = train_df["label"].values
    p_llm = train_df["p_llm"].values

    decision = ThresholdDecision()

    with Console().status("Tuning threshold...", spinner="simpleDots"):
        decision.fit(
            y_true,
            p_llm,
            threshold_metric=config.get("blender", {}).get("threshold_metric", "macro_f1"),
        )
    decision.save()

    # 6. Evaluation metrics on OOF Gemma + tuned threshold
    console.print("\n[bold cyan]Step 5: Overall Pipeline Performance (OOF)[/bold cyan]")
    _, preds = decision.predict(p_llm)
    compute_metrics(y_true, preds)

    console.print(
        Panel(
            "[bold green]Training Pipeline Successfully Finished![/bold green]",
            border_style="bold green",
        )
    )


if __name__ == "__main__":
    main()
