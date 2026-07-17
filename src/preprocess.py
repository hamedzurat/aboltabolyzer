import json
import os
import re
import unicodedata

import pandas as pd

from src.tui import banner, console, count_table, done_panel, info, ok, step


def clean_text(text):
    if text is None:
        return "[NULL]"
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xad]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if text == "" or text.lower() == "null" or text == "[NULL]":
        return "[NULL]"

    return text


def preprocess_dataset(input_path, is_test=False):
    info(f"Reading {'test' if is_test else 'train'}: {input_path}")
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(input_path)

    total_rows = len(df)
    with console.status(f"Cleaning {total_rows} rows...", spinner="dots"):
        df["context"] = df["context"].apply(clean_text)
        df["prompt_bn"] = df["prompt_bn"].apply(clean_text)
        df["response_bn"] = df["response_bn"].apply(clean_text)

    df["has_context"] = df["context"] != "[NULL]"
    n_null = int((~df["has_context"]).sum())
    ok(f"Cleaned {total_rows} rows · context present {total_rows - n_null} · [NULL] {n_null}")

    if not is_test:
        df["label"] = df["label"].astype(int)
        label_counts = df["label"].value_counts().to_dict()
        count_table(
            "Train label distribution",
            {f"label {k}": int(v) for k, v in sorted(label_counts.items())},
            key_header="Label",
        )

    return df


def main():
    banner("Preprocessing", "Clean competition CSVs into generated/processed/")

    import tomllib

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)

    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    step(1, 2, "Train / sample set")
    sample_df = preprocess_dataset(config["data"]["sample_path"], is_test=False)
    out_train_path = os.path.join(processed_dir, "train.csv")
    sample_df.to_csv(out_train_path, index=False)
    ok(f"Wrote {out_train_path}")

    step(2, 2, "Test set")
    test_df = preprocess_dataset(config["data"]["test_path"], is_test=True)
    out_test_path = os.path.join(processed_dir, "test.csv")
    test_df.to_csv(out_test_path, index=False)
    ok(f"Wrote {out_test_path}")

    done_panel(
        "Preprocessing complete",
        [
            f"Train rows: [bold]{len(sample_df)}[/bold] → {out_train_path}",
            f"Test rows:  [bold]{len(test_df)}[/bold] → {out_test_path}",
            f"Test [NULL] context: [bold]{int((test_df['context'] == '[NULL]').sum())}[/bold]",
        ],
    )


if __name__ == "__main__":
    main()
