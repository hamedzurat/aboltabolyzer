import os
import pickle

import numpy as np
from sklearn.metrics import f1_score


class ThresholdDecision:
    """Gemma-led final decision via a single OOF-tuned threshold on p_llm.

    p_llm represents P(label=1 faithful). XLM-R influences Gemma upstream as an
    encoder prior; it is not blended here.
    """

    def __init__(self):
        self.is_fitted = False
        self.threshold = 0.5
        self.threshold_metric = "macro_f1"

    def fit(self, y_true, p_llm, threshold_metric="macro_f1"):
        """Grid-search threshold on out-of-fold Gemma probabilities."""
        self.threshold_metric = threshold_metric
        p_llm = np.asarray(p_llm, dtype=float)
        self.threshold, _ = self._find_best_threshold(y_true, p_llm, threshold_metric)
        self.is_fitted = True

        preds = (p_llm >= self.threshold).astype(int)
        overall_f1 = f1_score(y_true, preds, average="macro")
        f1_class_0 = f1_score(y_true, preds, pos_label=0)

        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(
            Panel(
                f"[bold cyan]Decision source:[/]  Gemma p_llm (encoder prior upstream)\n"
                f"[bold cyan]Optimal threshold:[/] [bold yellow]{self.threshold:.3f}[/] "
                f"(optimized for: {threshold_metric})\n"
                f"[bold cyan]OOF Macro-F1:[/]     [bold green]{overall_f1:.4f}[/]\n"
                f"[bold cyan]OOF F1(0):[/]        [bold red]{f1_class_0:.4f}[/]",
                title="[bold green]✔ Threshold Decision Fitted[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        return overall_f1

    def _find_best_threshold(self, y_true, probs, metric):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.05, 0.95, 181):
            preds = (probs >= threshold).astype(int)
            if metric == "f1_class_0":
                score = f1_score(y_true, preds, pos_label=0)
            else:
                score = f1_score(y_true, preds, average="macro")
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold, best_score

    def predict(self, p_llm):
        """Apply tuned threshold to Gemma probabilities."""
        p_llm = np.asarray(p_llm, dtype=float)
        threshold = self.threshold if self.is_fitted else 0.5
        if not self.is_fitted:
            print(
                "Warning: ThresholdDecision has not been fitted yet! Using default threshold=0.5."
            )
        preds = (p_llm >= threshold).astype(int)
        return p_llm, preds

    def save(self, filepath="models/blender_config.pkl"):
        """Save threshold configuration to disk."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "threshold": self.threshold,
                    "threshold_metric": self.threshold_metric,
                    "decision_source": "p_llm",
                },
                f,
            )
        print(f"Saved threshold decision config to {filepath}")

    def load(self, filepath="models/blender_config.pkl"):
        """Load threshold configuration from disk."""
        if filepath.endswith(".json"):
            filepath = filepath.replace(".json", ".pkl")

        if not os.path.exists(filepath):
            print(f"Threshold config file {filepath} not found. Using default threshold=0.5.")
            return False

        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self.threshold = float(payload.get("threshold", 0.5))
            self.threshold_metric = payload.get("threshold_metric", "macro_f1")
        else:
            self.threshold = 0.5
        self.is_fitted = True
        print(f"Loaded threshold decision from {filepath} with threshold={self.threshold:.3f}")
        return True
