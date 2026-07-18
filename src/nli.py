"""NLI-first gate for entailment-friendly task types.

Runs after the fast pass and before the think pass. For configured tasks with
non-empty premise evidence, if NLI is confident (|P(entail)-P(contradict)| >=
margin) the row uses the NLI score and skips the expensive think pass.
Otherwise the normal think triggers apply.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config_utils import resolve_model_path
from src.tui import info, ok, warn

DEFAULT_NLI_TASKS = (
    "context_grounded_fact",
    "context_grounded_other",
    "famous_bn_fact_context",
    "general_fact_null",
)

NLI_DEBUG_COLUMNS = (
    "nli_eligible",
    "nli_applied",
    "nli_skip_reason",
    "nli_p_entail",
    "nli_p_contradict",
    "nli_p_neutral",
    "nli_margin",
    "p_nli",
)


def _empty_premise(text: Any) -> bool:
    s = "" if text is None else str(text).strip()
    return s in ("", "[NULL]", "None", "nan")


def premise_for_row(row: pd.Series) -> str:
    """Prefer original context; fall back to retrieved evidence."""
    for col in ("context_original", "context"):
        if col in row.index and not _empty_premise(row[col]):
            return str(row[col]).strip()
    return ""


def build_hypothesis(prompt_bn: str, response_bn: str, style: str = "qa_en") -> str:
    q = str(prompt_bn)
    a = str(response_bn)
    if style == "answer_only":
        return a
    if style == "claim_bn":
        return f"প্রশ্ন: {q}\nউত্তর: {a}"
    if style == "qa_bn":
        return f"প্রশ্নের উত্তর: {a}"
    return f"The answer to the question '{q}' is: {a}"


def hybrid_override_mask(
    p_entail: np.ndarray,
    p_contradict: np.ndarray,
    margin: float,
) -> np.ndarray:
    """True where NLI is confident enough to skip think / override."""
    return np.abs(p_entail - p_contradict) >= float(margin)


def nli_scores_to_p(
    p_entail: np.ndarray,
    p_contradict: np.ndarray,
    *,
    faithful_score: float = 0.90,
    hallucinated_score: float = 0.10,
) -> np.ndarray:
    return np.where(
        p_entail > p_contradict,
        float(faithful_score),
        float(hallucinated_score),
    )


def nli_cache_tag(nli_config: dict | None) -> str:
    """Stable fingerprint for verifier cache invalidation."""
    if not nli_config or not nli_config.get("enabled"):
        return "nli_off"
    tasks = ",".join(sorted(str(t) for t in nli_config.get("tasks", DEFAULT_NLI_TASKS)))
    return (
        f"nli_first|"
        f"m={nli_config.get('margin', 0.35)}|"
        f"h={nli_config.get('hypothesis_style', 'qa_en')}|"
        f"t={tasks}"
    )


def empty_nli_debug(index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nli_eligible": False,
            "nli_applied": False,
            "nli_skip_reason": "nli_disabled",
            "nli_p_entail": np.nan,
            "nli_p_contradict": np.nan,
            "nli_p_neutral": np.nan,
            "nli_margin": np.nan,
            "p_nli": np.nan,
        },
        index=index,
    )


class NLIRefiner:
    """Batched multilingual NLI gate (NLI-first + think fallback)."""

    def __init__(self, nli_config: dict):
        self.enabled = bool(nli_config.get("enabled", False))
        self.model_name = nli_config.get(
            "model_name",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )
        self.batch_size = int(nli_config.get("batch_size", 16))
        self.margin = float(nli_config.get("margin", 0.35))
        self.hypothesis_style = str(nli_config.get("hypothesis_style", "qa_en"))
        self.faithful_score = float(nli_config.get("faithful_score", 0.90))
        self.hallucinated_score = float(nli_config.get("hallucinated_score", 0.10))
        tasks = nli_config.get("tasks", list(DEFAULT_NLI_TASKS))
        self.tasks = frozenset(str(t) for t in tasks)
        self.max_length = int(nli_config.get("max_length", 512))
        self.model = None
        self.tokenizer = None
        self._label_idx = {}

    def load(self):
        if self.model is not None:
            return
        resolved = resolve_model_path(self.model_name)
        info(f"Loading NLI model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device).eval()
        for idx, name in self.model.config.id2label.items():
            key = str(name).lower()
            i = int(idx) if not isinstance(idx, int) else idx
            if "entail" in key:
                self._label_idx["entailment"] = i
            elif "contradict" in key:
                self._label_idx["contradiction"] = i
            elif "neutral" in key:
                self._label_idx["neutral"] = i
        missing = {"entailment", "contradiction", "neutral"} - set(self._label_idx)
        if missing:
            raise ValueError(f"NLI model missing labels: {sorted(missing)}")
        ok(f"NLI loaded · device={device} · tasks={sorted(self.tasks)}")

    def unload(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def score_pairs(
        self,
        premises: list[str],
        hypotheses: list[str],
        on_batch=None,
    ) -> list[dict]:
        if not premises:
            return []
        self.load()
        device = next(self.model.parameters()).device
        rows = []
        for start in range(0, len(premises), self.batch_size):
            p = premises[start : start + self.batch_size]
            h = hypotheses[start : start + self.batch_size]
            enc = self.tokenizer(
                p,
                h,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            for i in range(len(p)):
                rows.append(
                    {
                        "p_entail": float(probs[i, self._label_idx["entailment"]]),
                        "p_contradict": float(probs[i, self._label_idx["contradiction"]]),
                        "p_neutral": float(probs[i, self._label_idx["neutral"]]),
                    }
                )
            if on_batch is not None:
                on_batch(len(p))
        return rows

    def gate(self, df: pd.DataFrame, on_batch=None) -> pd.DataFrame:
        """Score eligible rows; set nli_applied when confident enough to skip think."""
        debug = empty_nli_debug(df.index)
        debug["nli_skip_reason"] = ""

        if not self.enabled:
            debug["nli_skip_reason"] = "nli_disabled"
            return debug

        if "task_type" not in df.columns:
            debug["nli_skip_reason"] = "no_task_type"
            warn("NLI enabled but task_type missing — skipping gate")
            return debug

        candidates = []
        for i, (idx, row) in enumerate(df.iterrows()):
            task = str(row["task_type"])
            if task not in self.tasks:
                debug.at[idx, "nli_skip_reason"] = "task_not_in_nli_list"
                continue
            premise = premise_for_row(row)
            if not premise:
                debug.at[idx, "nli_skip_reason"] = "empty_premise"
                continue
            hyp = build_hypothesis(
                row.get("prompt_bn", ""),
                row.get("response_bn", ""),
                self.hypothesis_style,
            )
            debug.at[idx, "nli_eligible"] = True
            candidates.append((i, idx, premise, hyp))

        if not candidates:
            info("NLI-first: no eligible rows")
            return debug

        info(
            f"NLI-first gate · {len(candidates)}/{len(df)} eligible · "
            f"margin≥{self.margin} → skip think"
        )
        scores = self.score_pairs(
            [c[2] for c in candidates],
            [c[3] for c in candidates],
            on_batch=on_batch,
        )

        n_applied = 0
        for (_row_i, idx, _p, _h), sc in zip(candidates, scores):
            e, c, n = sc["p_entail"], sc["p_contradict"], sc["p_neutral"]
            margin = abs(e - c)
            debug.at[idx, "nli_p_entail"] = e
            debug.at[idx, "nli_p_contradict"] = c
            debug.at[idx, "nli_p_neutral"] = n
            debug.at[idx, "nli_margin"] = margin
            if margin < self.margin:
                debug.at[idx, "nli_skip_reason"] = "margin_too_low"
                continue
            p_nli = float(
                nli_scores_to_p(
                    np.array([e]),
                    np.array([c]),
                    faithful_score=self.faithful_score,
                    hallucinated_score=self.hallucinated_score,
                )[0]
            )
            debug.at[idx, "p_nli"] = p_nli
            debug.at[idx, "nli_applied"] = True
            debug.at[idx, "nli_skip_reason"] = ""
            n_applied += 1

        ok(
            f"NLI-first · confident {n_applied}/{len(candidates)} "
            f"(skip think) · fallback {len(candidates) - n_applied}"
        )
        return debug


def release_cuda_for_nli():
    """Free GPU memory before loading the NLI encoder."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
