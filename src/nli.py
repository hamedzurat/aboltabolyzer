"""NLI-first gate for entailment-friendly task types.

Runs after the fast pass and before the think pass. For configured tasks with
non-empty premise evidence, if NLI is confident the row uses the NLI score and
skips the expensive think pass. Otherwise the normal think triggers apply.

Policy is asymmetric:
  - Hallucinated (contradict): lower margin — easier to auto-accept
  - Faithful (entail): higher margin + answer↔premise overlap + optional
    fast-disagreement / entail>neutral guards — harder to auto-accept
"""

from __future__ import annotations

import re
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


def answer_premise_overlap(
    premise: str,
    response_bn: str,
    *,
    min_ratio: float = 0.3,
) -> bool:
    """True when enough answer tokens appear in the premise (Faithful guard)."""
    premise = "" if premise is None else str(premise)
    ans = "" if response_bn is None else str(response_bn)
    ans = ans.replace("*", "").replace("_", "").strip()
    if len(ans) < 2 or not premise.strip():
        return False
    if ans in premise:
        return True
    toks = [t for t in re.split(r"\s+", ans) if len(t) >= 3]
    if not toks:
        # Short answers (names): require full string or single token ≥2 chars in premise
        short = [t for t in re.split(r"\s+", ans) if len(t) >= 2]
        if not short:
            return False
        return any(t in premise for t in short)
    hits = sum(1 for t in toks if t in premise)
    return (hits / len(toks)) >= float(min_ratio)


def hybrid_override_mask(
    p_entail: np.ndarray,
    p_contradict: np.ndarray,
    margin: float,
    margin_faithful: float | None = None,
    margin_hallucinated: float | None = None,
) -> np.ndarray:
    """True where NLI is confident enough to skip think / override.

    Asymmetric margins: Faithful uses margin_faithful (default: margin),
    Hallucinated uses margin_hallucinated (default: margin).
    """
    mf = float(margin if margin_faithful is None else margin_faithful)
    mh = float(margin if margin_hallucinated is None else margin_hallucinated)
    delta = np.abs(p_entail - p_contradict)
    needed = np.where(p_entail > p_contradict, mf, mh)
    return delta >= needed


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
    base_m = nli_config.get("margin", 0.35)
    return (
        f"nli_first|"
        f"m={base_m}|"
        f"mf={nli_config.get('margin_faithful', base_m)}|"
        f"mh={nli_config.get('margin_hallucinated', base_m)}|"
        f"h={nli_config.get('hypothesis_style', 'qa_en')}|"
        f"ov={int(bool(nli_config.get('require_faithful_overlap', True)))}|"
        f"ovr={nli_config.get('faithful_overlap_min', 0.3)}|"
        f"rsim={nli_config.get('min_rag_sim_for_nli', 0.0)}|"
        f"bfh={int(bool(nli_config.get('block_faithful_on_fast_h', True)))}|"
        f"fh={nli_config.get('fast_h_max_for_nli_faithful', 0.40)}|"
        f"en={int(bool(nli_config.get('require_entail_gt_neutral', True)))}|"
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


def _row_p_fast(row: pd.Series) -> float | None:
    if "p_fast" not in row.index:
        return None
    val = row.get("p_fast")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


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
        # Asymmetric: Faithful harder, Hallucinated easier (fall back to margin).
        self.margin_faithful = float(nli_config.get("margin_faithful", max(self.margin, 0.45)))
        self.margin_hallucinated = float(
            nli_config.get("margin_hallucinated", min(self.margin, 0.30))
        )
        self.hypothesis_style = str(nli_config.get("hypothesis_style", "qa_en"))
        self.faithful_score = float(nli_config.get("faithful_score", 0.90))
        self.hallucinated_score = float(nli_config.get("hallucinated_score", 0.10))
        self.require_faithful_overlap = bool(nli_config.get("require_faithful_overlap", True))
        self.faithful_overlap_min = float(nli_config.get("faithful_overlap_min", 0.4))
        # When RAG evidence is the premise, skip NLI if sim_max is below this (0 = off).
        self.min_rag_sim_for_nli = float(nli_config.get("min_rag_sim_for_nli", 0.55))
        # Faithful + fast strongly Hallucinated → fall through to think (not hard agree).
        self.block_faithful_on_fast_h = bool(nli_config.get("block_faithful_on_fast_h", True))
        self.fast_h_max_for_nli_faithful = float(
            nli_config.get("fast_h_max_for_nli_faithful", 0.40)
        )
        # Faithful only when entailment beats neutral (topic-overlap FP filter).
        self.require_entail_gt_neutral = bool(nli_config.get("require_entail_gt_neutral", True))
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
            # Weak RAG premise → don't let NLI auto-accept / skip think.
            rag_used = bool(row.get("rag_used", False)) if "rag_used" in row.index else False
            if rag_used and self.min_rag_sim_for_nli > 0:
                sim = row.get("retrieval_sim_max", np.nan)
                try:
                    sim_f = float(sim)
                except (TypeError, ValueError):
                    sim_f = float("nan")
                if np.isnan(sim_f) or sim_f < self.min_rag_sim_for_nli:
                    debug.at[idx, "nli_skip_reason"] = "weak_rag_premise"
                    continue
            hyp = build_hypothesis(
                row.get("prompt_bn", ""),
                row.get("response_bn", ""),
                self.hypothesis_style,
            )
            debug.at[idx, "nli_eligible"] = True
            candidates.append(
                (
                    i,
                    idx,
                    premise,
                    hyp,
                    str(row.get("response_bn", "")),
                    _row_p_fast(row),
                )
            )

        if not candidates:
            info("NLI-first: no eligible rows")
            return debug

        info(
            f"NLI-first gate · {len(candidates)}/{len(df)} eligible · "
            f"F≥{self.margin_faithful} / H≥{self.margin_hallucinated} → skip think"
        )
        scores = self.score_pairs(
            [c[2] for c in candidates],
            [c[3] for c in candidates],
            on_batch=on_batch,
        )

        n_applied = 0
        n_block = {"overlap": 0, "fast_h": 0, "neutral": 0, "margin": 0}
        for (_row_i, idx, premise, _h, response, p_fast), sc in zip(candidates, scores):
            e, c, n = sc["p_entail"], sc["p_contradict"], sc["p_neutral"]
            margin = abs(e - c)
            debug.at[idx, "nli_p_entail"] = e
            debug.at[idx, "nli_p_contradict"] = c
            debug.at[idx, "nli_p_neutral"] = n
            debug.at[idx, "nli_margin"] = margin

            is_faithful = e > c
            needed = self.margin_faithful if is_faithful else self.margin_hallucinated
            if margin < needed:
                debug.at[idx, "nli_skip_reason"] = "margin_too_low"
                n_block["margin"] += 1
                continue

            p_nli = float(
                nli_scores_to_p(
                    np.array([e]),
                    np.array([c]),
                    faithful_score=self.faithful_score,
                    hallucinated_score=self.hallucinated_score,
                )[0]
            )

            if is_faithful:
                if self.require_entail_gt_neutral and e <= n:
                    debug.at[idx, "p_nli"] = p_nli
                    debug.at[idx, "nli_applied"] = False
                    debug.at[idx, "nli_skip_reason"] = "entail_le_neutral"
                    n_block["neutral"] += 1
                    continue
                if self.require_faithful_overlap and not answer_premise_overlap(
                    premise,
                    response,
                    min_ratio=self.faithful_overlap_min,
                ):
                    debug.at[idx, "p_nli"] = p_nli
                    debug.at[idx, "nli_applied"] = False
                    debug.at[idx, "nli_skip_reason"] = "faithful_low_overlap"
                    n_block["overlap"] += 1
                    continue
                if (
                    self.block_faithful_on_fast_h
                    and p_fast is not None
                    and p_fast < self.fast_h_max_for_nli_faithful
                ):
                    debug.at[idx, "p_nli"] = p_nli
                    debug.at[idx, "nli_applied"] = False
                    debug.at[idx, "nli_skip_reason"] = "faithful_fast_disagrees"
                    n_block["fast_h"] += 1
                    continue

            debug.at[idx, "p_nli"] = p_nli
            debug.at[idx, "nli_applied"] = True
            debug.at[idx, "nli_skip_reason"] = ""
            n_applied += 1

        blocked = sum(n_block.values()) - n_block["margin"]
        ok(
            f"NLI-first · confident {n_applied}/{len(candidates)} "
            f"(skip think) · blocked F "
            f"overlap={n_block['overlap']} fastH={n_block['fast_h']} "
            f"neutral={n_block['neutral']} · "
            f"fallback {len(candidates) - n_applied - blocked}"
        )
        return debug


def release_cuda_for_nli():
    """Free GPU memory before loading the NLI encoder."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
