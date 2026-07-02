import gc
import os
import tomllib

import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.blender import ScoreBlender
from src.config_utils import fail_on_model_error
from src.llm_verifier import GemmaVerifier
from src.rag import BanglaRAG
from src.xlmr_encoder import predict_test

import logging
import transformers
from huggingface_hub.utils import disable_progress_bars

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
            "[bold yellow]🔮 Test Set Inference & Prediction Pipeline[/bold yellow]",
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
    else:
        if not os.path.exists(test_processed_path):
            console.print(
                f"[bold red]Error: Processed test file not found at {test_processed_path}. Run preprocessing first.[/bold red]"
            )
            return

        test_df = pd.read_csv(test_processed_path)

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
                        chunks = rag.retrieve(query)
                        if chunks:
                            retrieved_contexts.append(" ".join(chunks))
                        else:
                            retrieved_contexts.append("[NULL]")
                        progress.advance(task)

                test_df.loc[null_mask, "context"] = retrieved_contexts
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

    # 4. Predict with Gemma 4 Verifier
    console.print("\n[bold cyan]Step 3: Gemma 4 Verifier Inference[/bold cyan]")
    verifier = GemmaVerifier()
    if config.get("runtime", {}).get("use_llm_verifier", True):
        try:
            verifier.load_model()
            p_llm, is_c0, is_c1, is_c2 = verifier.predict_dataset(test_df)
        except Exception as e:
            if fail_on_model_error(config):
                raise RuntimeError("Gemma verifier failed; refusing to submit fake scores.") from e
            console.print(f"[bold red]Failed to run Gemma verifier on test set: {e}[/bold red]")
            console.print(
                "[yellow]Falling back to XLM-R-only scores because fail_on_model_error=false.[/yellow]"
            )
            p_llm = p_xlmr.copy()
            is_c0 = pd.Series(0.0, index=test_df.index).values
            is_c1 = pd.Series(0.0, index=test_df.index).values
            is_c2 = pd.Series(0.0, index=test_df.index).values
    else:
        console.print("[yellow]LLM verifier disabled. Using XLM-R-only blend features.[/yellow]")
        p_llm = p_xlmr.copy()
        is_c0 = pd.Series(0.0, index=test_df.index).values
        is_c1 = pd.Series(0.0, index=test_df.index).values
        is_c2 = pd.Series(0.0, index=test_df.index).values

    test_df["p_xlmr"] = p_xlmr
    test_df["p_llm"] = p_llm
    test_df["is_c0"] = is_c0
    test_df["is_c1"] = is_c1
    test_df["is_c2"] = is_c2
    test_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "test_with_preds.csv"), index=False
    )

    # 5. Load blender parameters
    blender = ScoreBlender()
    blender_path = "models/blender_config.pkl"
    if os.path.exists(blender_path):
        blender.load(blender_path)
    else:
        console.print(
            f"[bold red]Warning: Meta-classifier config not found at {blender_path}. Using 50/50 blend fallback.[/bold red]"
        )

    # 6. Score Blending
    console.print("\n[bold cyan]Step 4: Blending Predictions[/bold cyan]")
    p_blend, preds = blender.predict(
        p_xlmr,
        p_llm,
        has_context=test_df["has_context"].values,
        is_c0=test_df["is_c0"].values,
        is_c1=test_df["is_c1"].values,
        is_c2=test_df["is_c2"].values,
    )

    # 7. Create submission file
    submission_path = config["data"]["submission_output_path"]
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"] if "id" in test_df.columns else range(len(preds)),
            "label": preds,
        }
    )

    submission_df.to_csv(submission_path, index=False)

    # Save a detailed debug CSV with all inputs, intermediate probabilities, and final decisions
    debug_path = submission_path.replace(".csv", "_debug.csv")
    debug_df = test_df.copy()
    debug_df["p_blend"] = p_blend
    debug_df["label"] = preds
    debug_df.to_csv(debug_path, index=False)
    console.print(f"Saved detailed debug submission to: [bold white]{debug_path}[/bold white]")

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
