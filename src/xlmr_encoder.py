import gc
import os

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.config_utils import resolve_section

console = Console()


class BanglaDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, has_labels=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

        self.inputs = []
        self.labels = []

        contexts = df["context"].astype(str).tolist()
        prompts = df["prompt_bn"].astype(str).tolist()
        responses = df["response_bn"].astype(str).tolist()

        self.inputs = [(ctx, f"{p} </s> {r}") for ctx, p, r in zip(contexts, prompts, responses)]
        if has_labels:
            self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text, text_pair = self.inputs[idx]
        encoding = self.tokenizer(
            text,
            text_pair,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_model(model_name, lora_r, lora_alpha, lora_dropout, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, peft_config)
    model.to(device)
    return model


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    progress,
    task_id,
    scheduler=None,
    grad_accum_steps=1,
    use_amp=True,
):
    model.train()
    total_loss = 0
    total_steps = len(dataloader)
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            scaled_loss = loss / grad_accum_steps

        scaled_loss.backward()

        should_step = step % grad_accum_steps == 0 or step == total_steps
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item()

        progress.advance(task_id)

    return total_loss / total_steps


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_binary = (all_preds >= 0.5).astype(int)
    macro_f1 = f1_score(all_labels, preds_binary, average="macro")
    f1_class_0 = f1_score(all_labels, preds_binary, pos_label=0)

    return macro_f1, f1_class_0, all_preds


def train_cross_validation(train_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xlmr_config = resolve_section(config, "xlmr")
    console.print(f"[bold cyan]Selected training hardware device:[/bold cyan] {device}")
    console.print(
        "[bold cyan]XLM-R settings:[/bold cyan] "
        f"model={xlmr_config['model_name']}, max_length={xlmr_config['max_length']}, "
        f"batch_size={xlmr_config['batch_size']}, grad_accum={xlmr_config['grad_accum_steps']}"
    )

    with Console().status("Loading XLM-R Tokenizer...", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(xlmr_config["model_name"])

    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True, random_state=config["seed"])
    oof_preds = np.zeros(len(train_df))

    os.makedirs("models/xlmr", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        console.print(
            Panel(
                f"[bold yellow]Training Fold {fold + 1}/{config['num_folds']}[/bold yellow]",
                border_style="yellow",
            )
        )

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = BanglaDataset(fold_train_df, tokenizer, xlmr_config["max_length"])
        val_dataset = BanglaDataset(fold_val_df, tokenizer, xlmr_config["max_length"])

        train_loader = DataLoader(train_dataset, batch_size=xlmr_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=xlmr_config["batch_size"], shuffle=False)

        with Console().status(
            "Initializing base model with LoRA configuration...", spinner="aesthetic"
        ):
            model = get_model(
                xlmr_config["model_name"],
                xlmr_config["lora_r"],
                xlmr_config["lora_alpha"],
                xlmr_config["lora_dropout"],
                device,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(xlmr_config["lr"]),
            weight_decay=xlmr_config["weight_decay"],
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        epochs = xlmr_config["epochs"]
        grad_accum_steps = max(1, int(xlmr_config.get("grad_accum_steps", 1)))
        total_steps = epochs * max(
            1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
        )
        warmup_steps = max(1, total_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_f1 = 0
        best_fold_preds = None
        patience_counter = 0

        for epoch in range(epochs):
            steps_per_epoch = len(train_loader)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=True,
            ) as progress:
                epoch_task = progress.add_task(
                    description=f"Epoch {epoch + 1}/{epochs} training...", total=steps_per_epoch
                )
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    progress,
                    epoch_task,
                    scheduler=scheduler,
                    grad_accum_steps=grad_accum_steps,
                    use_amp=bool(xlmr_config.get("use_amp", True)),
                )

            val_macro_f1, val_f1_0, val_probs = evaluate(model, val_loader, device)

            # Print epoch report
            console.print(
                f"[bold cyan]Epoch {epoch + 1:02d}[/bold cyan] | "
                f"Loss: [bold white]{train_loss:.4f}[/bold white] | "
                f"Val Macro-F1: [bold green]{val_macro_f1:.4f}[/bold green] | "
                f"Val F1(0): [bold green]{val_f1_0:.4f}[/bold green]"
            )

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_fold_preds = val_probs
                patience_counter = 0
                torch.save(model.state_dict(), f"models/xlmr/best_fold_{fold}.pt")
            else:
                patience_counter += 1

            if patience_counter >= xlmr_config["early_stopping_patience"]:
                console.print(f"[bold red]Early stopping triggered at epoch {epoch + 1}[/bold red]")
                break

        oof_preds[val_idx] = best_fold_preds

        # Cleanup memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    train_df["p_xlmr"] = oof_preds
    train_df.to_csv(os.path.join(config["data"]["processed_dir"], "oof_xlmr.csv"), index=False)

    overall_f1 = f1_score(train_df["label"], (oof_preds >= 0.5).astype(int), average="macro")
    console.print(
        Panel(
            f"[bold green]✔ XLM-R CV Completed! Overall OOF Macro-F1: {overall_f1:.4f}[/bold green]",
            border_style="green",
        )
    )
    return oof_preds


def predict_test(test_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xlmr_config = resolve_section(config, "xlmr")
    tokenizer = AutoTokenizer.from_pretrained(xlmr_config["model_name"])

    test_dataset = BanglaDataset(test_df, tokenizer, xlmr_config["max_length"], has_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=xlmr_config["batch_size"], shuffle=False)

    all_fold_preds = []
    num_folds = config["num_folds"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        fold_task = progress.add_task(
            description="Running cross-encoder ensemble inference...", total=num_folds
        )

        for fold in range(num_folds):
            progress.update(fold_task, description=f"Predicting Fold {fold + 1}/{num_folds}...")
            model = get_model(
                xlmr_config["model_name"],
                xlmr_config["lora_r"],
                xlmr_config["lora_alpha"],
                xlmr_config["lora_dropout"],
                device,
            )

            state_dict_path = f"models/xlmr/best_fold_{fold}.pt"
            if os.path.exists(state_dict_path):
                model.load_state_dict(
                    torch.load(state_dict_path, map_location=device, weights_only=True)
                )

            model.eval()
            fold_preds = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_preds.extend(probs)

            all_fold_preds.append(fold_preds)

            del model
            torch.cuda.empty_cache()
            gc.collect()
            progress.advance(fold_task)

    mean_preds = np.mean(all_fold_preds, axis=0)
    test_df["p_xlmr"] = mean_preds
    test_df.to_csv(
        os.path.join(config["data"]["processed_dir"], "preds_test_xlmr.csv"), index=False
    )
    console.print("[green]✔ Cross-encoder test set inference complete and saved.[/green]")
    return mean_preds
