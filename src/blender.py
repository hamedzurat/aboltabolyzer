import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class ScoreBlender:
    """Ensembles cross-encoder and LLM predictions using a Meta-Classifier (RandomForestClassifier).

    Both p_xlmr and p_llm represent Class 1 (Faithful) probabilities.
    """

    def __init__(self):
        # We use a RandomForestClassifier as our meta-classifier.
        # This is a robust tree-based model that captures non-linear interactions
        # (e.g., trust LLM more if has_context is False) and works on any dataset size.
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=3, min_samples_leaf=1, random_state=42
        )
        self.is_fitted = False
        self.threshold = 0.5
        self.threshold_metric = "macro_f1"

    def fit(
        self,
        y_true,
        p_xlmr,
        p_llm,
        has_context,
        is_c0=None,
        is_c1=None,
        is_c2=None,
        threshold_metric="macro_f1",
        **kwargs,
    ):
        """Fits the meta-classifier on the Out-Of-Fold predictions."""
        self.threshold_metric = threshold_metric
        # Ensure min_samples_leaf is smaller than the dataset size (important for small unit tests)
        min_samples = min(20, max(1, len(y_true) // 3))
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=min_samples,
            random_state=42,
        )

        # Fallbacks if categories are not provided (e.g. legacy tests)
        if is_c0 is None:
            is_c0 = np.zeros(len(p_xlmr))
        if is_c1 is None:
            is_c1 = np.zeros(len(p_xlmr))
        if is_c2 is None:
            is_c2 = np.zeros(len(p_xlmr))

        # Stack inputs into feature matrix
        x = np.stack(
            [
                p_xlmr,
                p_llm,
                has_context.astype(float),
                is_c0.astype(float),
                is_c1.astype(float),
                is_c2.astype(float),
            ],
            axis=1,
        )

        self.model.fit(x, y_true)
        self.is_fitted = True

        p_train = self.model.predict_proba(x)[:, 1]
        self.threshold, overall_f1 = self._find_best_threshold(y_true, p_train, threshold_metric)
        preds = (p_train >= self.threshold).astype(int)
        overall_f1 = f1_score(y_true, preds, average="macro")
        f1_class_0 = f1_score(y_true, preds, pos_label=0)
        print(
            "Meta-Classifier Fitted. "
            f"Train Macro-F1 (in-sample): {overall_f1:.4f}, "
            f"F1(0): {f1_class_0:.4f}, threshold: {self.threshold:.3f}"
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

    def predict(self, p_xlmr, p_llm, has_context, is_c0=None, is_c1=None, is_c2=None):
        """Calculates blended probability and final binary predictions using the meta-classifier."""
        if not self.is_fitted:
            print("Warning: Meta-classifier has not been fitted yet! Returning 50/50 fallback.")
            p_blend = 0.5 * p_xlmr + 0.5 * p_llm
            preds = (p_blend >= 0.5).astype(int)
            return p_blend, preds

        # Fallbacks if categories are not provided
        if is_c0 is None:
            is_c0 = np.zeros(len(p_xlmr))
        if is_c1 is None:
            is_c1 = np.zeros(len(p_xlmr))
        if is_c2 is None:
            is_c2 = np.zeros(len(p_xlmr))

        x = np.stack(
            [
                p_xlmr,
                p_llm,
                has_context.astype(float),
                is_c0.astype(float),
                is_c1.astype(float),
                is_c2.astype(float),
            ],
            axis=1,
        )

        p_blend = self.model.predict_proba(x)[:, 1]
        preds = (p_blend >= self.threshold).astype(int)
        return p_blend, preds

    def save(self, filepath="models/blender_config.pkl"):
        """Saves the meta-classifier state to disk using pickle."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "threshold": self.threshold,
                    "threshold_metric": self.threshold_metric,
                },
                f,
            )
        print(f"Saved meta-classifier to {filepath}")

    def load(self, filepath="models/blender_config.pkl"):
        """Loads the meta-classifier state from disk."""
        # Check for legacy JSON file path and adapt
        if filepath.endswith(".json"):
            filepath = filepath.replace(".json", ".pkl")

        if not os.path.exists(filepath):
            print(f"Meta-classifier config file {filepath} not found. Fallback to default.")
            return False

        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.threshold = float(payload.get("threshold", 0.5))
            self.threshold_metric = payload.get("threshold_metric", "macro_f1")
        else:
            self.model = payload
            self.threshold = 0.5
        self.is_fitted = True
        print(f"Loaded meta-classifier from {filepath} with threshold={self.threshold:.3f}")
        return True
