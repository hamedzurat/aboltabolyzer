"""Deterministic task router for the Gemma-only inference pipeline."""

from __future__ import annotations

import re

import pandas as pd

FACTUAL_PATTERNS = (
    "কত সালে",
    "কত তারিখে",
    "কতটি",
    "কতজন",
    "কোথায়",
    "কোন",
    "কবে",
    "কত",
    "কে",
)

GRAMMAR_PATTERNS = (
    "সমাস",
    "সন্ধি",
    "কারক",
    "বিভক্তি",
    "ব্যাসবাক্য",
    "ধাতু",
    "ক্রিয়া পদের",
    "ক্রিয়া পদের",
    "স্বরধ্বনি",
    "ব্যঞ্জনধ্বনি",
    "উচ্চারণ",
    "বর্ণ",
    "বানান",
    "ধ্বনি",
    "প্রত্যয়",
    "প্রকৃতি",
    "শুদ্ধ",
)

MATH_WORK_RATE = (
    "একসাথে কাজ",
    "যৌথভাবে কাজ",
    "একত্রে কাজ",
    "রহিম",
    "করিম",
    "নির্মাণ প্রকল্প",
)
MATH_SPEED = ("গতিবেগ", "ঘণ্টায়", "ঘণ্টায়", "দূরত্ব")
MATH_PROFIT = ("ক্রয়মূল্য", "ক্রয়মূল্য", "বিক্রয়মূল্য", "বিক্রয়মূল্য", "লাভ", "ক্ষতি", "শতকরা লাভ", "শতকরা ক্ষতি")
MATH_AVERAGE = ("গড়", "গড়মান", "অ্যাভারেজ")
CALENDAR = ("বার হলে", "সপ্তাহের কোন দিন", "সপ্তাহের কোন")

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

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def is_factual_prompt(prompt_bn: str) -> bool:
    return _is_factual(prompt_bn)


def _is_factual(prompt_bn: str) -> bool:
    return _contains_any(prompt_bn, FACTUAL_PATTERNS)


def _is_translation_or_bilingual(prompt_bn: str, response_bn: str) -> bool:
    if _contains_any(prompt_bn.lower(), TRANSLATION_PATTERNS):
        return True
    prompt_has_latin = bool(_LATIN_WORD.search(prompt_bn))
    response_has_latin = bool(_LATIN_WORD.search(response_bn))
    return prompt_has_latin or response_has_latin


def route_row(context: str, prompt_bn: str, response_bn: str = "") -> str:
    """Return a task_type label for one row."""
    context = "" if context is None else str(context).strip()
    prompt_bn = "" if prompt_bn is None else str(prompt_bn)
    response_bn = "" if response_bn is None else str(response_bn)
    context_is_null = context in ("", "[NULL]", "None", "nan")

    if not context_is_null:
        if _contains_any(prompt_bn, FAMOUS_ENTITY_PATTERNS) and _is_factual(prompt_bn):
            return "famous_bn_fact_context"
        if _is_factual(prompt_bn):
            return "context_grounded_fact"
        return "context_grounded_other"

    if "ভাবার্থ" in prompt_bn:
        return "idiom_meaning_null"
    if "শাব্দিক অর্থ" in prompt_bn:
        return "literal_meaning_null"

    if (
        _contains_any(prompt_bn, MATH_WORK_RATE)
        and ("কাজ" in prompt_bn or "দিন" in prompt_bn or "ঘণ্টা" in prompt_bn or "ঘন্টা" in prompt_bn)
    ) or (
        ("লোক" in prompt_bn or "জন" in prompt_bn)
        and ("দিন" in prompt_bn or "ঘণ্টা" in prompt_bn or "ঘন্টা" in prompt_bn)
        and ("কাজ" in prompt_bn or "করতে" in prompt_bn or "সময়" in prompt_bn)
    ):
        return "math_work_rate"
    if _contains_any(prompt_bn, MATH_SPEED):
        return "math_speed_distance"
    if _contains_any(prompt_bn, MATH_PROFIT):
        return "math_profit_loss"
    if _contains_any(prompt_bn, MATH_AVERAGE):
        return "math_average"
    if _contains_any(prompt_bn, CALENDAR):
        return "calendar_arithmetic"
    if _contains_any(prompt_bn, GRAMMAR_PATTERNS):
        return "bangla_grammar"
    # Prioritize factual questions over translation/bilingual if prompt is factual and lacks explicit translation keywords
    is_trans = _is_translation_or_bilingual(prompt_bn, response_bn)
    if is_trans and not (
        _is_factual(prompt_bn) and not _contains_any(prompt_bn.lower(), TRANSLATION_PATTERNS)
    ):
        return "translation_or_bilingual"
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
