"""Task-specific evidence and prompt policies for the routed verifier."""

from __future__ import annotations

import re

# Task type → typed corpus source under corpus/<source>/ and indexes/<source>.pkl.
# None means no retrieval for that task.
# Fill corpus/{idioms,literal,grammar} then `just make-rag --source <name>`.
# Empty/weak hits fall back via TASK_RAG_FALLBACK or stay [NULL].
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
    "math_other": None,
    "calendar_arithmetic": None,
}

# Prefer typed corpus; if index missing or all hits empty, try fallback.
TASK_RAG_FALLBACK = {
    "idiom_meaning_null": "wiki",
    "literal_meaning_null": "wiki",
}

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
        "math_other",
        "calendar_arithmetic",
    }
)

TASK_INSTRUCTIONS = {
    "context_grounded_fact": (
        "Use only the evidence. Mark H for clear factual contradictions, wrong "
        "date/person/place, or claims the evidence directly refutes. "
        "Accept correct partial answers — a response does not need to list every detail "
        "the evidence mentions; it only needs to be factually correct for what it states."
    ),
    "context_grounded_other": (
        "Use only the evidence. Mark H for clear contradictions or factually wrong claims. "
        "Accept correct partial answers — do not reject an answer just because it omits "
        "details the evidence mentions."
    ),
    "famous_bn_fact_context": (
        "Use the evidence first. Watch for Bangladesh/literature swaps that contradict "
        "well-known facts."
    ),
    "idiom_meaning_null": (
        "Judge the Bengali ভাবার্থ / বাগধারা. Missing or irrelevant evidence alone is "
        "NOT H — use your knowledge of the figurative meaning. Mark F only if the "
        "response matches the true ভাবার্থ."
    ),
    "literal_meaning_null": (
        "Judge the শাব্দিক/compositional meaning. Missing or irrelevant evidence alone "
        "is NOT H — use compositional knowledge. Mark F only if the literal gloss is correct."
    ),
    "general_fact_null": (
        "Check the fact carefully. Watch for swapped people, dates, places, nearby "
        "events, and total-vs-part numbers. If the evidence is missing or silent on "
        "the fact, rely on your general knowledge to verify if the statement is correct."
    ),
    "famous_bn_fact_null": (
        "Check the Bangladesh/literature fact carefully. Watch for swapped people, "
        "dates, places, and nearby events. If the evidence is missing or silent on "
        "the fact, rely on your general knowledge to verify if the statement is correct."
    ),
    "bangla_grammar": (
        "Judge strictly by Bangla grammar rules (সমাস, সন্ধি, কারক, বিভক্তি, etc.). "
        "Use evidence when it states the rule or correct form; missing evidence alone "
        "is NOT hallucination — use your internal grammar knowledge. "
        "Accept minor spelling variants when the grammatical category is unambiguously correct. "
        "For সন্ধি: apply the actual phonological rule (e.g. আ+ঈ→এ, giving মহেশ not মহাঈশ; "
        "আ+আ→া, বিদ্যালয় not বিদ্যাআলয়). Reject naive concatenation."
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
    "math_other": (
        "Calculate internally (ratio, interest, area, mixture, etc.). "
        "Mark F only if the computed answer matches the response."
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
        "math_other",
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
    force_think_all: bool = False,
) -> bool:
    """Decide whether to run the think pass for a routed row.

    Bias toward fewer thinks (speed): most extra triggers only fire inside the
    near-threshold band. Math/grammar still get a slightly wider band.
    """
    reasons = think_reasons if think_reasons is not None else []
    prompt_bn = "" if prompt_bn is None else str(prompt_bn)
    p = float(p_fast)

    if force_think_all:
        reasons.append("force_think_all")
        return True

    triggered = False
    near = conf_low <= p <= conf_high

    if near:
        reasons.append("near_threshold")
        triggered = True

    if task_type in ("famous_bn_fact_null", "famous_bn_fact_context") and near:
        reasons.append("famous_bn_fact")
        triggered = True

    # Multi-entity context: only when uncertain (was always-on → slow + mixed).
    if (
        task_type == "context_grounded_fact"
        and near
        and context_has_multiple_entities(context_original)
    ):
        reasons.append("multi_entity_context")
        triggered = True

    # Math/calendar: only near-threshold (always-think flipped confident-H → F).
    if task_type in MATH_TASKS and near:
        reasons.append("math_needs_check")
        triggered = True

    # Grammar: slightly wider band; সন্ধি/সমাস inside that band.
    if task_type == "bangla_grammar":
        grammar_near = not (p < 0.2 or p > 0.8)
        if ("সন্ধি" in prompt_bn or "সমাস" in prompt_bn or "ব্যাসবাক্য" in prompt_bn) and grammar_near:
            reasons.append("grammar_rule_check")
            triggered = True
        elif grammar_near:
            reasons.append("bangla_grammar_wide_window")
            triggered = True

    # Translation: routing is fixed — no wide window (saves thinks).
    if task_type == "translation_or_bilingual" and near:
        reasons.append("translation_check")
        triggered = True

    if (
        should_use_rag(task_type, context_original, prompt_bn)
        and evidence_looks_irrelevant(prompt_bn, evidence)
        and near
    ):
        reasons.append("evidence_missing_keyphrase")
        triggered = True

    return triggered


THINK_SCORE_MAP = {
    ("faithful", "strong"): 0.90,
    ("faithful", "likely"): 0.75,
    ("faithful", "uncertain"): 0.51,
    ("hallucinated", "uncertain"): 0.49,
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
