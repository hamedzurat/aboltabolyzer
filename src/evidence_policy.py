"""Task-specific evidence and prompt policies for the routed Gemma verifier."""

from __future__ import annotations

import re

# Task type → typed corpus source under corpus/<source>/ and indexes/<source>.pkl.
# None means no retrieval for that task.
TASK_RAG_SOURCE = {
    "general_fact_null": "wiki",
    "other_null": "wiki",
    "famous_bn_fact_null": "wiki",
    "idiom_meaning_null": "idioms",
    "literal_meaning_null": "literal",
    "bangla_grammar": "grammar",
    "context_grounded_fact": None,
    "context_grounded_other": None,
    "famous_bn_fact_context": None,
    "translation_or_bilingual": None,
    "math_work_rate": None,
    "math_speed_distance": None,
    "math_profit_loss": None,
    "math_average": None,
    "calendar_arithmetic": None,
}

# If the preferred source index is missing, try this fallback (still typed).
TASK_RAG_FALLBACK = {}

# Never retrieve for these even if a source mapping exists.
RAG_SKIP_TASKS = frozenset(
    {
        "context_grounded_fact",
        "context_grounded_other",
        "famous_bn_fact_context",
        "translation_or_bilingual",
        "math_work_rate",
        "math_speed_distance",
        "math_profit_loss",
        "math_average",
        "calendar_arithmetic",
    }
)

TASK_INSTRUCTIONS = {
    "context_grounded_fact": (
        "Use only the evidence. Mark H for contradiction, unsupported claims, wrong "
        "date/person/place, or missing required qualifier."
    ),
    "context_grounded_other": (
        "Use only the evidence. Mark H for contradiction, unsupported claims, wrong "
        "date/person/place, or missing required qualifier."
    ),
    "famous_bn_fact_context": (
        "Use the evidence first. Watch for Bangladesh/literature swaps that contradict "
        "well-known facts."
    ),
    "idiom_meaning_null": (
        "Judge the Bengali ভাবার্থ. Missing evidence alone is not H. Mark F only for "
        "the correct figurative meaning."
    ),
    "literal_meaning_null": (
        "Judge the শাব্দিক/compositional meaning. Missing evidence alone is not H."
    ),
    "general_fact_null": (
        "Check the fact carefully. Watch for swapped people, dates, places, nearby "
        "events, and total-vs-part numbers."
    ),
    "famous_bn_fact_null": (
        "Check the Bangladesh/literature fact carefully. Watch for swapped people, "
        "dates, places, and nearby events."
    ),
    "bangla_grammar": (
        "Judge by Bangla grammar rules. Use evidence if helpful, but missing evidence "
        "alone is not H. Accept minor spelling variants when the category is clear."
    ),
    "translation_or_bilingual": (
        "Judge the English/Bengali translation or term. Watch for antonyms, wrong "
        "technical terms, and title/person confusion."
    ),
    "math_work_rate": (
        "Calculate internally. Mark F only if the computed answer matches the response."
    ),
    "math_speed_distance": (
        "Calculate internally. Mark F only if the computed answer matches the response."
    ),
    "math_profit_loss": (
        "Calculate internally. Mark F only if the computed answer matches the response."
    ),
    "math_average": (
        "Calculate internally. Mark F only if the computed answer matches the response."
    ),
    "calendar_arithmetic": (
        "Calculate internally. Mark F only if the computed answer matches the response."
    ),
    "other_null": (
        "No context is provided. Use general knowledge only when confident. If not "
        "clearly correct, mark H."
    ),
}

MATH_TASKS = frozenset(
    {
        "math_work_rate",
        "math_speed_distance",
        "math_profit_loss",
        "math_average",
        "calendar_arithmetic",
    }
)

_DATE_OR_ENTITY = re.compile(
    r"\d{4}|\d{1,2}/\d{1,2}|জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|"
    r"জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|"
    r"সালে|তারিখে"
)


def rag_source_for_task(task_type: str) -> str | None:
    """Return the preferred typed corpus source for a task, or None."""
    if task_type in RAG_SKIP_TASKS:
        return None
    return TASK_RAG_SOURCE.get(task_type)


def rag_fallback_source(task_type: str) -> str | None:
    return TASK_RAG_FALLBACK.get(task_type)


def rag_skip_reason(task_type: str, context_original: str) -> str | None:
    """Return why RAG should be skipped, or None if RAG is allowed."""
    context_original = "" if context_original is None else str(context_original).strip()
    if context_original not in ("", "[NULL]", "None", "nan"):
        return "original_context_present"
    if task_type in RAG_SKIP_TASKS:
        return f"task_policy:{task_type}"
    if rag_source_for_task(task_type) is None:
        return f"no_rag_source:{task_type}"
    if task_type == "other_null":
        return None
    return None


def should_use_rag(task_type: str, context_original: str, prompt_bn: str = "") -> bool:
    reason = rag_skip_reason(task_type, context_original)
    if reason is not None:
        return False
    if task_type == "other_null":
        from src.router import is_factual_prompt

        return is_factual_prompt(prompt_bn or "")
    return rag_source_for_task(task_type) is not None


def task_instruction(task_type: str) -> str:
    return TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["other_null"])


def evidence_looks_irrelevant(prompt_bn: str, evidence: str) -> bool:
    evidence = "" if evidence is None else str(evidence).strip()
    if evidence in ("", "[NULL]", "None", "nan"):
        return True
    tokens = [t for t in re.split(r"\s+", prompt_bn) if len(t) >= 3]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in evidence)
    return hits == 0


def context_has_multiple_entities(context: str) -> bool:
    context = "" if context is None else str(context)
    matches = _DATE_OR_ENTITY.findall(context)
    return len(matches) >= 2


def should_trigger_think(
    *,
    p_fast: float,
    task_type: str,
    evidence: str,
    context_original: str,
    prompt_bn: str,
    conf_low: float = 0.35,
    conf_high: float = 0.65,
    think_reasons: list[str] | None = None,
) -> bool:
    """Decide whether to run the think pass for a routed row."""
    reasons = think_reasons if think_reasons is not None else []
    triggered = False

    if conf_low <= float(p_fast) <= conf_high:
        reasons.append("near_threshold")
        triggered = True

    if task_type == "famous_bn_fact_null":
        reasons.append("famous_bn_fact_null")
        triggered = True

    if task_type == "context_grounded_fact" and context_has_multiple_entities(context_original):
        reasons.append("multi_entity_context")
        triggered = True

    if task_type in MATH_TASKS:
        reasons.append("math_needs_check")
        triggered = True

    if task_type in ("idiom_meaning_null", "literal_meaning_null") and evidence_looks_irrelevant(
        prompt_bn, evidence
    ):
        reasons.append("lexical_missing_evidence")
        triggered = True

    if should_use_rag(task_type, context_original, prompt_bn) and evidence_looks_irrelevant(
        prompt_bn, evidence
    ):
        reasons.append("evidence_missing_keyphrase")
        triggered = True

    return triggered


THINK_SCORE_MAP = {
    ("faithful", "strong"): 0.90,
    ("faithful", "likely"): 0.75,
    ("faithful", "uncertain"): 0.50,
    ("hallucinated", "uncertain"): 0.50,
    ("hallucinated", "likely"): 0.25,
    ("hallucinated", "strong"): 0.10,
}


def map_think_verdict(verdict: str, confidence: str | None) -> float | None:
    if not verdict:
        return None
    verdict_key = verdict.strip().lower()
    conf_key = (confidence or "likely").strip().lower()
    if conf_key not in ("strong", "likely", "uncertain"):
        conf_key = "likely"
    return THINK_SCORE_MAP.get((verdict_key, conf_key))
