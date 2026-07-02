import json
import os
import re
import unicodedata

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def clean_text(text):
    if text is None:
        return "[NULL]"
    if not isinstance(text, str):
        text = str(text)

    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters and soft hyphens
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xad]", "", text)

    # Normalize multiple whitespaces
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    if text == "" or text.lower() == "null" or text == "[NULL]":
        return "[NULL]"

    return text


def preprocess_dataset(input_path, is_test=False):
    console.print(f"[bold cyan]Reading file:[/bold cyan] {input_path}")
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(input_path)

    total_rows = len(df)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        task = progress.add_task(description=f"Cleaning {total_rows} rows...", total=total_rows)

        # We can clean columns in vectorized form, but let's show visual feedback
        df["context"] = df["context"].apply(clean_text)
        df["prompt_bn"] = df["prompt_bn"].apply(clean_text)
        df["response_bn"] = df["response_bn"].apply(clean_text)

        progress.advance(task, total_rows)

    df["has_context"] = df["context"] != "[NULL]"

    if not is_test:
        df["label"] = df["label"].astype(int)

    return df


def main():
    console.print(Panel("[bold yellow]🧹 Preprocessing Phase[/bold yellow]", border_style="yellow"))

    import tomllib

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)

    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # Preprocess sample/train set
    sample_df = preprocess_dataset(config["data"]["sample_path"], is_test=False)
    out_train_path = os.path.join(processed_dir, "train.csv")
    sample_df.to_csv(out_train_path, index=False)
    console.print(f"[green]✔ Saved preprocessed train data to {out_train_path}[/green]")

    # Preprocess test set
    test_df = preprocess_dataset(config["data"]["test_path"], is_test=True)
    out_test_path = os.path.join(processed_dir, "test.csv")
    test_df.to_csv(out_test_path, index=False)
    console.print(f"[green]✔ Saved preprocessed test data to {out_test_path}[/green]")

    console.print("\n[bold green]★ Preprocessing Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    main()
