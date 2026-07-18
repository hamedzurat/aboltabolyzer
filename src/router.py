"""Deterministic + hybrid task router for the inference pipeline."""

from __future__ import annotations

import re

import pandas as pd

FACTUAL_PATTERNS = (
    "কত সালে",
    "কত তারিখে",
    "কতটি",
    "কতজন",
    "কোথায়",
    "কোথায়",
    "কোন",
    "কবে",
    "কত",
    "কে",
    "কী",
    "কি",
)

# High-precision grammar cues (safe for static veto).
GRAMMAR_CORE = (
    "সমাস",
    "সন্ধি",
    "কারক",
    "বিভক্তি",
    "ব্যাসবাক্য",
    "ক্রিয়া পদের",
    "ক্রিয়া পদের",
    "স্বরধ্বনি",
    "ব্যঞ্জনধ্বনি",
    "উপসর্গ",
    "প্রত্যয়",
    "প্রত্যয়",
    "প্রকৃতি",
    "বাচ্য",
    "বচন",
    "স্বরবর্ণ",
    "ব্যঞ্জনবর্ণ",
    "বর্ণমালা",
    "বর্ণনানুক্রম",
)

MATH_WORK_RATE = (
    "একসাথে কাজ",
    "যৌথভাবে কাজ",
    "একত্রে কাজ",
    "নির্মাণ প্রকল্প",
)
MATH_SPEED = ("গতিবেগ", "ঘণ্টায়", "ঘণ্টায়", "দূরত্ব")
# Avoid bare লাভ/ক্ষতি — they match "স্বীকৃতি লাভ করে" (received recognition).
MATH_PROFIT = (
    "ক্রয়মূল্য",
    "ক্রয়মূল্য",
    "বিক্রয়মূল্য",
    "বিক্রয়মূল্য",
    "শতকরা লাভ",
    "শতকরা ক্ষতি",
    "লাভের হার",
    "ক্ষতির হার",
    "লাভ হয়",
    "লাভ হয়",
    "ক্ষতি হয়",
    "ক্ষতি হয়",
    "মুনাফা",
)
# Avoid bare গড়/গড় — they match "গড় উচ্চতা" geography facts.
MATH_AVERAGE = ("গড় মান", "গড় মান", "গড় নম্বর", "গড় নম্বর", "গড়মান", "অ্যাভারেজ", "গড় কত", "গড় কত")
MATH_OTHER = (
    "অনুপাত",
    "সুদ",
    "মূলধন",
    "সরল সুদ",
    "ক্ষেত্রফল",
    "পরিসীমা",
    "মিশ্রণ",
    "নল দিয়ে",
    "নল দিয়ে",
    "বর্গমূল",
    "√",
)

# Weekday / date-offset arithmetic (avoid bare "বার" which hits ভাবার্থ text).
CALENDAR = (
    "বার হলে",
    "সপ্তাহের কোন দিন",
    "সপ্তাহের কোন বার",
    "সপ্তাহের কোন",
    "কোন বার হবে",
    "কী বার",
    "কি বার",
    "কোন বার ছিল",
    "দিন পরবর্তী",
    "দিন পরে",
)

TRANSLATION_PATTERNS = (
    "ইংরেজি",
    "english",
    "translation",
    "অনুবাদ",
    "terminology",
    "term",
)

FAMOUS_ENTITY_PATTERNS = (
    "রবীন্দ্রনাথ",
    "নজরুল",
    "শেখ মুজিব",
    "মুজিবনগর",
    "স্বাধীনতা দিবস",
    "বিজয় দিবস",
    "বিজয় দিবস",
    "অপারেশন সার্চলাইট",
    "ঢাকা বিশ্ববিদ্যালয়",
    "ঢাকা বিশ্ববিদ্যালয়",
    "সুন্দরবন",
    "পদ্মা",
    "মেঘনা",
    "যমুনা",
    "অভ্র",
)

# Static owns these even when LLM disagrees (hybrid mode).
STATIC_VETO_TASKS = frozenset(
    {
        "idiom_meaning_null",
        "literal_meaning_null",
        "calendar_arithmetic",
        "math_work_rate",
        "math_speed_distance",
        "math_profit_loss",
        "math_average",
        "math_other",
        "bangla_grammar",
    }
)

MATH_TASK_TYPES = frozenset(
    {
        "math_work_rate",
        "math_speed_distance",
        "math_profit_loss",
        "math_average",
        "math_other",
        "calendar_arithmetic",
    }
)

STICKY_FACT_TASKS = frozenset(
    {
        "context_grounded_fact",
        "context_grounded_other",
        "famous_bn_fact_context",
        "famous_bn_fact_null",
        "general_fact_null",
    }
)

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def is_factual_prompt(prompt_bn: str) -> bool:
    return _is_factual(prompt_bn)


def _is_factual(prompt_bn: str) -> bool:
    return _contains_any(prompt_bn, FACTUAL_PATTERNS)


def _is_bangla_grammar(prompt_bn: str) -> bool:
    """True for real grammar items; avoids স্বর্ণ/chemistry ধাতু false positives."""
    if _contains_any(prompt_bn, GRAMMAR_CORE):
        return True
    if "ধাতু" in prompt_bn and ("ক্রিয়া" in prompt_bn or "ক্রিয়া" in prompt_bn or "মূল" in prompt_bn):
        return True
    if "শুদ্ধ" in prompt_bn and ("বানান" in prompt_bn or "বিপরীত" in prompt_bn or "শব্দ" in prompt_bn):
        return True
    if "উচ্চারণ" in prompt_bn and ("ধ্বনি" in prompt_bn or "বর্ণ" in prompt_bn):
        return True
    return False


def _is_translation_or_bilingual(prompt_bn: str, response_bn: str) -> bool:
    if _contains_any(prompt_bn.lower(), TRANSLATION_PATTERNS):
        return True
    prompt_has_latin = bool(_LATIN_WORD.search(prompt_bn))
    if prompt_has_latin:
        return True
    # Latin only in the answer (acronyms, English names) is not enough when the
    # question is a normal factual WH prompt — that was flooding "translation".
    response_has_latin = bool(_LATIN_WORD.search(response_bn))
    if response_has_latin and not _is_factual(prompt_bn):
        return True
    return False


def _route_math(prompt_bn: str) -> str | None:
    if _contains_any(prompt_bn, CALENDAR):
        return "calendar_arithmetic"
    if (
        _contains_any(prompt_bn, MATH_WORK_RATE)
        and ("কাজ" in prompt_bn or "দিন" in prompt_bn or "ঘণ্টা" in prompt_bn or "ঘন্টা" in prompt_bn)
    ) or (
        ("লোক" in prompt_bn or "জন" in prompt_bn)
        and ("দিন" in prompt_bn or "ঘণ্টা" in prompt_bn or "ঘন্টা" in prompt_bn)
        and ("কাজ" in prompt_bn or "করতে" in prompt_bn or "সময়" in prompt_bn or "সময়" in prompt_bn)
    ):
        return "math_work_rate"
    if _contains_any(prompt_bn, MATH_SPEED):
        return "math_speed_distance"
    if _contains_any(prompt_bn, MATH_PROFIT):
        return "math_profit_loss"
    if _contains_any(prompt_bn, MATH_AVERAGE):
        return "math_average"
    if ("বয়স" in prompt_bn or "বয়স" in prompt_bn) and (
        "অনুপাত" in prompt_bn or "গুণ" in prompt_bn or ":" in prompt_bn
    ):
        return "math_other"
    if _contains_any(prompt_bn, MATH_OTHER):
        return "math_other"
    return None


def route_row(context: str, prompt_bn: str, response_bn: str = "") -> str:
    """Return a task_type label for one row (full static policy)."""
    context = "" if context is None else str(context).strip()
    prompt_bn = "" if prompt_bn is None else str(prompt_bn)
    response_bn = "" if response_bn is None else str(response_bn)
    context_is_null = context in ("", "[NULL]", "None", "nan")

    if "ভাবার্থ" in prompt_bn or "বাগধারা" in prompt_bn or "প্রবাদ" in prompt_bn:
        return "idiom_meaning_null"
    if "শাব্দিক অর্থ" in prompt_bn:
        return "literal_meaning_null"

    # Grammar before math: "লাভ শব্দের শুদ্ধ বিপরীত" must not hit profit keywords.
    if _is_bangla_grammar(prompt_bn):
        return "bangla_grammar"

    math_task = _route_math(prompt_bn)
    if math_task is not None:
        return math_task

    is_trans = _is_translation_or_bilingual(prompt_bn, response_bn)
    if is_trans and not (
        _is_factual(prompt_bn) and not _contains_any(prompt_bn.lower(), TRANSLATION_PATTERNS)
    ):
        return "translation_or_bilingual"

    if not context_is_null:
        if _contains_any(prompt_bn, FAMOUS_ENTITY_PATTERNS) and _is_factual(prompt_bn):
            return "famous_bn_fact_context"
        if _is_factual(prompt_bn):
            return "context_grounded_fact"
        return "context_grounded_other"

    if _contains_any(prompt_bn, FAMOUS_ENTITY_PATTERNS):
        return "famous_bn_fact_null"
    if _is_factual(prompt_bn):
        return "general_fact_null"
    return "other_null"


def route_dataframe(df: pd.DataFrame, context_col: str = "context") -> pd.Series:
    """Route every row; prefer context_original when present."""
    if "context_original" in df.columns:
        contexts = df["context_original"]
    else:
        contexts = df[context_col]
    responses = df["response_bn"] if "response_bn" in df.columns else pd.Series([""] * len(df))
    return pd.Series(
        [
            route_row(ctx, prompt, response)
            for ctx, prompt, response in zip(
                contexts.tolist(),
                df["prompt_bn"].tolist(),
                responses.tolist(),
                strict=True,
            )
        ],
        index=df.index,
        name="task_type",
    )


def map_llm_category_to_task_type(
    best_char: str, context: str, prompt_bn: str, response_bn: str = ""
) -> str:
    context = "" if context is None else str(context).strip()
    prompt_bn = "" if prompt_bn is None else str(prompt_bn)
    response_bn = "" if response_bn is None else str(response_bn)
    context_is_null = context in ("", "[NULL]", "None", "nan")

    if best_char == "G":
        return "bangla_grammar"
    if best_char == "M":
        math_task = _route_math(prompt_bn)
        return math_task if math_task is not None else "math_other"
    if best_char == "I":
        if "ভাবার্থ" in prompt_bn or "বাগধারা" in prompt_bn or "প্রবাদ" in prompt_bn:
            return "idiom_meaning_null"
        if "শাব্দিক অর্থ" in prompt_bn:
            return "literal_meaning_null"
        return "literal_meaning_null"
    if best_char == "T":
        return "translation_or_bilingual"
    if best_char == "F":
        if not context_is_null:
            if _contains_any(prompt_bn, FAMOUS_ENTITY_PATTERNS) and _is_factual(prompt_bn):
                return "famous_bn_fact_context"
            return "context_grounded_fact"
        if _contains_any(prompt_bn, FAMOUS_ENTITY_PATTERNS):
            return "famous_bn_fact_null"
        return "general_fact_null"
    if not context_is_null:
        return "context_grounded_other"
    return "other_null"


def resolve_routing_mode(router_config: dict | None) -> str:
    """Return static | llm | hybrid from config."""
    cfg = router_config or {}
    mode = str(cfg.get("routing_mode", "") or "").strip().lower()
    if mode in ("static", "llm", "hybrid"):
        return mode
    # Backward compat: llm_routing bool
    if bool(cfg.get("llm_routing", False)):
        return "hybrid"
    return "static"


def needs_llm_router(router_config: dict | None) -> bool:
    return resolve_routing_mode(router_config) in ("llm", "hybrid")


def route_dataframe_llm(df: pd.DataFrame, verifier, context_col: str = "context") -> pd.Series:
    """Route every row using LLM classification only."""
    if "context_original" in df.columns:
        contexts = df["context_original"]
    else:
        contexts = df[context_col]
    responses = df["response_bn"] if "response_bn" in df.columns else pd.Series([""] * len(df))

    from src.tui import pipeline_progress

    task_types = []
    with pipeline_progress() as progress:
        route_task = progress.add_task("LLM Router (Classifying)", total=len(df))
        for ctx, prompt, response in zip(
            contexts.tolist(), df["prompt_bn"].tolist(), responses.tolist(), strict=True
        ):
            best_char = verifier.route_single_llm(prompt, response)
            task_types.append(map_llm_category_to_task_type(best_char, ctx, prompt, response))
            progress.advance(route_task)

    return pd.Series(task_types, index=df.index, name="task_type")


def route_dataframe_hybrid(df: pd.DataFrame, verifier, context_col: str = "context") -> pd.Series:
    """Static veto for high-precision domains; LLM only on residual rows.

    Guards: LLM must not demote factual/context rows into translation, or invent
    idiom/literal/grammar without lexical cues.
    """
    static = route_dataframe(df, context_col=context_col)
    if "context_original" in df.columns:
        contexts = df["context_original"]
    else:
        contexts = df[context_col]
    responses = df["response_bn"] if "response_bn" in df.columns else pd.Series([""] * len(df))

    from src.tui import pipeline_progress

    out = static.copy()
    residual_idx = [i for i, t in enumerate(static.tolist()) if t not in STATIC_VETO_TASKS]

    if not residual_idx:
        return out

    with pipeline_progress() as progress:
        route_task = progress.add_task("Hybrid Router (LLM residual)", total=len(residual_idx))
        for i in residual_idx:
            ctx = contexts.iloc[i]
            prompt = str(df["prompt_bn"].iloc[i])
            response = responses.iloc[i]
            best_char = verifier.route_single_llm(prompt, response)
            llm_task = map_llm_category_to_task_type(best_char, ctx, prompt, response)
            static_task = static.iloc[i]

            # Never demote grounded/null facts into translation.
            if llm_task == "translation_or_bilingual" and static_task in STICKY_FACT_TASKS:
                out.iloc[i] = static_task
            # Idiom/literal/grammar only with cues (static already owns true hits).
            elif llm_task in (
                "idiom_meaning_null",
                "literal_meaning_null",
                "bangla_grammar",
            ):
                out.iloc[i] = static_task
            # Don't invent math without cues — keep static.
            elif llm_task in MATH_TASK_TYPES and _route_math(prompt) is None:
                out.iloc[i] = static_task
            else:
                out.iloc[i] = llm_task
            progress.advance(route_task)

    return out
