import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.sort_corpus import parse_bucket
from src.config_utils import resolve_quantization_mode, resolve_section, validate_config
from src.evidence_policy import (
    map_think_verdict,
    rag_source_for_task,
    should_trigger_think,
    should_use_rag,
    task_instruction,
)
from src.llm_verifier import GemmaVerifier, _resolve_torch_dtype
from src.predict import (
    _load_prediction_checkpoint,
    _save_prediction_checkpoint,
    apply_router_disabled_policy,
    apply_threshold,
    build_debug_df,
    dataframe_cache_fingerprint,
    load_verifier_debug_map,
    validate_submission_df,
)
from src.preprocess import clean_text
from src.rag import resolve_rag_sources
from src.router import route_row


def test_clean_text():
    assert clean_text(None) == "[NULL]"
    dirty_text = "উইন্ডোজে\u200b ইউনিকোড\u200c ভিত্তিক"
    assert clean_text(dirty_text) == "উইন্ডোজে ইউনিকোড ভিত্তিক"
    assert clean_text(12345) == "12345"
    assert clean_text("   ") == "[NULL]"
    assert clean_text("[NULL]") == "[NULL]"


def test_sort_corpus_bucket_parser():
    assert parse_bucket("bucket: famous_bn") == "wiki"
    assert parse_bucket("IDIOM") == "idioms"
    assert parse_bucket("discard this") == "skip"


def test_router_classifies_core_task_types():
    assert route_row("[NULL]", "‘লাঠালাঠি’ শব্দটির সমাস –", "কর্মধারায়") == "bangla_grammar"
    assert route_row("[NULL]", '"জো-হুকুমের দল" এর ভাবার্থ কী?', "আজ্ঞাবহ") == "idiom_meaning_null"
    assert route_row("[NULL]", '"ফ্ল্যাট" এর শাব্দিক অর্থ কী?', "চ্যাপ্টা") == "literal_meaning_null"
    assert route_row("কিছু প্রসঙ্গ আছে", "ঢাকায় কে জন্মগ্রহণ করেন?", "কেউ") == "context_grounded_fact"
    assert route_row("[NULL]", "বাংলাদেশের রাজধানী কোনটি?", "ঢাকা") == "general_fact_null"
    assert route_row("[NULL]", "রবীন্দ্রনাথ কবে জন্মগ্রহণ করেন?", "১৮৬১") == "famous_bn_fact_null"
    assert (
        route_row("[NULL]", "একটি গাড়ির গতিবেগ ৬০ কিমি/ঘণ্টায়, দূরত্ব কত?", "১২০")
        == "math_speed_distance"
    )
    assert (
        route_row(
            "[NULL]",
            "কোনো একদিন বুধবার বার হলে, তার ৫১ দিন পরবর্তী দিনটি সপ্তাহের কোন বার হবে?",
            "বৃহস্পতিবার",
        )
        == "calendar_arithmetic"
    )
    assert route_row("[NULL]", "দুই সংখ্যার অনুপাত ৩:৫ হলে…", "১৫") == "math_other"
    # Chemistry ধাতু / স্বর্ণ must not become grammar
    assert (
        route_row("[NULL]", "পারমাণবিক চুল্লিতে তাপ পরিবাহক হিসেবে কোন ধাতু ব্যবহৃত হয়?", "সোডিয়াম")
        == "general_fact_null"
    )
    # Latin-only answer on factual Q is not translation
    assert (
        route_row("[NULL]", "জাতিসংঘের কোন সংস্থা জনসংখ্যা বিষয়ে কাজ করে?", "UNFPA")
        == "general_fact_null"
    )


def test_rag_skip_policy_by_task_type():
    assert should_use_rag("math_average", "[NULL]") is False
    assert should_use_rag("math_other", "[NULL]") is False
    assert should_use_rag("context_grounded_fact", "some context") is False
    assert should_use_rag("translation_or_bilingual", "[NULL]") is False
    assert should_use_rag("general_fact_null", "[NULL]") is True
    assert should_use_rag("famous_bn_fact_null", "[NULL]") is True
    assert should_use_rag("idiom_meaning_null", "[NULL]") is True
    assert should_use_rag("literal_meaning_null", "[NULL]") is True
    assert should_use_rag("bangla_grammar", "[NULL]") is True
    assert should_use_rag("other_null", "[NULL]", "বাংলাদেশের রাজধানী কোনটি?") is True
    assert should_use_rag("other_null", "[NULL]", "শুধু একটি অদ্ভুত বাক্য") is False


def test_rag_source_mapping_by_task_type():
    assert rag_source_for_task("general_fact_null") == "wiki"
    assert rag_source_for_task("other_null") == "wiki"
    assert rag_source_for_task("famous_bn_fact_null") == "wiki"
    assert rag_source_for_task("idiom_meaning_null") == "idioms"
    assert rag_source_for_task("literal_meaning_null") == "literal"
    assert rag_source_for_task("bangla_grammar") == "grammar"
    assert rag_source_for_task("math_speed_distance") is None
    assert rag_source_for_task("context_grounded_fact") is None


def test_resolve_rag_sources_from_config():
    config = {
        "rag": {
            "corpus_root": "corpus",
            "index_root": "indexes",
            "sources": ["wiki", "idioms"],
        }
    }
    sources = resolve_rag_sources(config)
    assert set(sources) == {"wiki", "idioms"}
    assert sources["wiki"]["corpus_dir"] == "corpus/wiki"
    assert sources["idioms"]["index_path"] == "indexes/idioms.pkl"


def test_think_pass_verdict_confidence_parser():
    assert map_think_verdict("Faithful", "strong") == 0.90
    assert map_think_verdict("Faithful", "likely") == 0.75
    assert map_think_verdict("Hallucinated", "strong") == 0.10
    text = "reason: mismatch\nverdict: Hallucinated\n"
    verdict, confidence, score = GemmaVerifier._parse_think_output(text)
    assert verdict.lower() == "hallucinated"
    assert confidence is None
    assert score == 0.10


def test_think_prompt_puts_parseable_fields_first():
    instruction = GemmaVerifier()._think_instruction("general_fact_null")
    assert "verdict: Faithful|Hallucinated" in instruction


def test_grammar_instruction_allows_helpful_evidence_without_requiring_it():
    instruction = task_instruction("bangla_grammar")
    assert "not RAG" not in instruction
    assert "missing evidence alone is not" in instruction.lower()


def test_think_trigger_near_threshold_and_task():
    reasons = []
    assert should_trigger_think(
        p_fast=0.5,
        task_type="general_fact_null",
        evidence="evidence",
        context_original="[NULL]",
        prompt_bn="কোন",
        think_reasons=reasons,
    )
    assert "near_threshold" in reasons

    reasons = []
    assert should_trigger_think(
        p_fast=0.5,
        task_type="famous_bn_fact_null",
        evidence="[NULL]",
        context_original="[NULL]",
        prompt_bn="রবীন্দ্রনাথ",
        think_reasons=reasons,
    )
    assert "famous_bn_fact" in reasons

    # Overconfident famous: do not always-think
    reasons = []
    assert not should_trigger_think(
        p_fast=0.95,
        task_type="famous_bn_fact_null",
        evidence="[NULL]",
        context_original="[NULL]",
        prompt_bn="রবীন্দ্রনাথ",
        think_reasons=reasons,
    )

    # Sandhi forces think when not extremely confident
    reasons = []
    assert should_trigger_think(
        p_fast=0.7,
        task_type="bangla_grammar",
        evidence="[NULL]",
        context_original="[NULL]",
        prompt_bn="মহা' ও 'ঈশ' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?",
        think_reasons=reasons,
    )
    assert "grammar_rule_check" in reasons

    # Extremely confident sandhi: skip forced grammar think
    reasons = []
    assert not should_trigger_think(
        p_fast=0.99,
        task_type="bangla_grammar",
        evidence="evidence about সন্ধি",
        context_original="[NULL]",
        prompt_bn="মহা' ও 'ঈশ' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?",
        think_reasons=reasons,
    )

    # Math: only near-threshold, not when confident H
    reasons = []
    assert not should_trigger_think(
        p_fast=0.05,
        task_type="math_work_rate",
        evidence="[NULL]",
        context_original="[NULL]",
        prompt_bn="কাজ",
        think_reasons=reasons,
    )


def test_nli_answer_overlap_guard():
    from src.nli import answer_premise_overlap

    premise = "পাহাড়পুর বৌদ্ধ বিহার পাল রাজবংশের দ্বিতীয় রাজা ধর্মপাল নির্মাণ করেন।"
    assert answer_premise_overlap(premise, "ধর্মপাল") is True
    assert answer_premise_overlap(premise, "দেবপাল") is False


def test_nli_asymmetric_override_mask():
    from src.nli import hybrid_override_mask

    e = np.array([0.70, 0.70, 0.20])
    c = np.array([0.30, 0.30, 0.70])
    # Same |delta|=0.40: Faithful needs 0.45 → False; Hallucinated needs 0.30 → True
    mask = hybrid_override_mask(e, c, margin=0.35, margin_faithful=0.45, margin_hallucinated=0.30)
    assert list(mask) == [False, False, True]


def test_nli_gate_policy_without_model(monkeypatch):
    """Exercise asymmetric Faithful guards with mocked score_pairs (no GPU/model)."""
    from src.nli import NLIRefiner

    premise = "পাহাড়পুর বৌদ্ধ বিহার পাল রাজবংশের দ্বিতীয় রাজা ধর্মপাল নির্মাণ করেন।"
    # One row per policy branch; scores returned in the same candidate order.
    df = pd.DataFrame(
        {
            "task_type": ["context_grounded_fact"] * 6,
            "context_original": [premise] * 6,
            "context": [premise] * 6,
            "prompt_bn": ["কে নির্মাণ করেন?"] * 6,
            "response_bn": [
                "ধর্মপাল",  # 0: Faithful OK
                "ধর্মপাল",  # 1: margin too low for Faithful
                "দেবপাল",  # 2: low overlap
                "ধর্মপাল",  # 3: fast strongly H
                "ধর্মপাল",  # 4: entail <= neutral
                "ভুল নাম",  # 5: Hallucinated OK (asymmetric margin)
            ],
            "p_fast": [0.80, 0.80, 0.80, 0.25, 0.80, 0.20],
            "rag_used": [False] * 6,
        }
    )
    scores = [
        {"p_entail": 0.75, "p_contradict": 0.20, "p_neutral": 0.05},  # Δ0.55 F apply
        {"p_entail": 0.60, "p_contradict": 0.25, "p_neutral": 0.15},  # Δ0.35 < 0.45
        {"p_entail": 0.75, "p_contradict": 0.20, "p_neutral": 0.05},  # F low overlap
        {"p_entail": 0.75, "p_contradict": 0.20, "p_neutral": 0.05},  # F fast H
        {"p_entail": 0.48, "p_contradict": 0.02, "p_neutral": 0.50},  # Δ0.46 F but e≤n
        {"p_entail": 0.20, "p_contradict": 0.55, "p_neutral": 0.25},  # Δ0.35 H apply
    ]

    refiner = NLIRefiner(
        {
            "enabled": True,
            "tasks": ["context_grounded_fact"],
            "margin": 0.35,
            "margin_faithful": 0.45,
            "margin_hallucinated": 0.30,
            "require_faithful_overlap": True,
            "faithful_overlap_min": 0.4,
            "require_entail_gt_neutral": True,
            "block_faithful_on_fast_h": True,
            "fast_h_max_for_nli_faithful": 0.40,
            "min_rag_sim_for_nli": 0.55,
        }
    )
    monkeypatch.setattr(refiner, "score_pairs", lambda premises, hyps, on_batch=None: scores)

    debug = refiner.gate(df)
    reasons = list(debug["nli_skip_reason"])
    applied = list(debug["nli_applied"])

    assert applied == [True, False, False, False, False, True]
    assert reasons[0] == ""
    assert reasons[1] == "margin_too_low"
    assert reasons[2] == "faithful_low_overlap"
    assert reasons[3] == "faithful_fast_disagrees"
    assert reasons[4] == "entail_le_neutral"
    assert reasons[5] == ""
    assert debug.loc[0, "p_nli"] == pytest.approx(0.90)
    assert debug.loc[5, "p_nli"] == pytest.approx(0.10)


def test_nli_gate_skips_weak_rag_without_model():
    from src.nli import NLIRefiner

    refiner = NLIRefiner(
        {
            "enabled": True,
            "tasks": ["general_fact_null"],
            "min_rag_sim_for_nli": 0.55,
        }
    )
    df = pd.DataFrame(
        {
            "task_type": ["general_fact_null"],
            "context_original": ["[NULL]"],
            "context": ["weak wiki hit"],
            "prompt_bn": ["q"],
            "response_bn": ["a"],
            "rag_used": [True],
            "retrieval_sim_max": [0.40],
        }
    )
    debug = refiner.gate(df)
    assert not debug["nli_applied"].any()
    assert debug.iloc[0]["nli_skip_reason"] == "weak_rag_premise"
    assert not debug.iloc[0]["nli_eligible"]


def test_apply_threshold_and_submission_validator():
    p_final, preds = apply_threshold(np.array([0.9, 0.1, 0.5]), threshold=0.5)
    assert list(preds) == [1, 0, 1]
    assert np.allclose(p_final, [0.9, 0.1, 0.5])

    ok = pd.DataFrame({"id": [1, 2], "label": [0, 1]})
    assert validate_submission_df(ok) is True

    with pytest.raises(ValueError):
        validate_submission_df(pd.DataFrame({"id": [1], "label": [0], "extra": [1]}))
    with pytest.raises(ValueError):
        validate_submission_df(pd.DataFrame({"id": [1, 1], "label": [0, 1]}))


def test_debug_df_has_compact_tuning_schema():
    test_df = pd.DataFrame(
        {
            "id": [1, 2],
            "task_type": ["general_fact_null", "idiom_meaning_null"],
            "context_original": ["[NULL]", "[NULL]"],
            "context": ["evidence A", "[NULL]"],
            "prompt_bn": ["প্রশ্ন ১", "ভাবার্থ"],
            "response_bn": ["উত্তর ১", "অর্থ"],
            "rag_used": [True, False],
            "rag_source": ["wiki", ""],
            "rag_skipped_reason": ["", "index_missing:idioms"],
            "evidence_source": ["rag:wiki", "none"],
            "evidence_relevance": ["retrieved", "no_evidence"],
            "n_retrieved": [2, 0],
            "retrieval_sim_max": [0.8, np.nan],
            "retrieval_sim_mean": [0.7, np.nan],
        }
    )
    ctx = {
        "config": {
            "gemma": {
                "model_name": "google/gemma-4-E4B-it",
                "load_in": "4bit",
                "model_loader": "multimodal_lm",
                "device_map": "cuda:0",
            }
        },
        "run_ts": "20260101_000000",
        "hardware_profile": "16gb",
        "verifier_debug_log_path": "/tmp/does-not-exist.jsonl",
    }
    debug_df = build_debug_df(test_df, np.array([0.9, 0.2]), threshold=0.5, ctx=ctx)
    expected = [
        "id",
        "label",
        "p_llm",
        "threshold",
        "threshold_margin",
        "task_type",
        "p_fast",
        "p_think",
        "triggered_think",
        "think_max_tokens",
        "think_reasons",
        "verdict_parsed",
        "confidence_parsed",
        "think_changed_label",
        "thinking_cot",
        "nli_eligible",
        "nli_applied",
        "nli_skip_reason",
        "nli_p_entail",
        "nli_p_contradict",
        "nli_p_neutral",
        "nli_margin",
        "p_nli",
        "rag_used",
        "rag_source",
        "rag_skipped_reason",
        "evidence_source",
        "evidence_relevance",
        "n_retrieved",
        "retrieval_sim_max",
        "retrieval_sim_mean",
        "context_original",
        "context",
        "prompt_bn",
        "response_bn",
        "run_timestamp",
        "hardware_profile",
        "gemma_model_name",
        "gemma_load_in",
    ]
    assert list(debug_df.columns) == expected
    assert list(debug_df["label"]) == [1, 0]
    assert "p_final" not in debug_df.columns
    assert "gemma_device_map" not in debug_df.columns
    assert "context_char_len" not in debug_df.columns


def test_debug_log_cache_key_includes_evidence_and_task(tmp_path):
    log_path = tmp_path / "debug.jsonl"
    metadata = {
        "model_name": "test/model",
        "model_loader": "causal_lm",
        "load_in": "4bit",
        "max_input_tokens": 128,
        "enable_think_pass": False,
        "exemplar_top_k": 0,
        "think_conf_low": 0.35,
        "think_conf_high": 0.65,
    }
    from src.llm_verifier import CACHE_VERSION

    base = {
        "cache_version": CACHE_VERSION,
        "prompt": "একই প্রশ্ন",
        "response": "একই উত্তর",
        "context_original": "[NULL]",
        "task_type": "general_fact_null",
        "p_llm_final": 0.1,
        "triggered_think": False,
        **metadata,
    }
    rows = [
        {**base, "evidence": "প্রমাণ ক", "p_fast": 0.1},
        {**base, "evidence": "প্রমাণ খ", "p_fast": 0.9, "p_llm_final": 0.9},
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cache = load_verifier_debug_map(str(log_path), expected_metadata=metadata)

    assert len(cache) == 2
    assert {entry["p_fast"] for entry in cache.values()} == {0.1, 0.9}


def test_nli_hybrid_override_mask_and_scores():
    from src.nli import (
        build_hypothesis,
        hybrid_override_mask,
        nli_cache_tag,
        nli_scores_to_p,
        premise_for_row,
    )

    e = np.array([0.8, 0.55, 0.1])
    c = np.array([0.1, 0.45, 0.8])
    mask = hybrid_override_mask(e, c, margin=0.35)
    assert list(mask) == [True, False, True]

    p = nli_scores_to_p(e, c)
    assert np.allclose(p, [0.90, 0.90, 0.10])

    assert "The answer to the question" in build_hypothesis("Q?", "A", "qa_en")
    assert build_hypothesis("Q?", "A only", "answer_only") == "A only"

    row = pd.Series(
        {
            "context_original": "[NULL]",
            "context": "retrieved wiki blurb",
            "prompt_bn": "প্রশ্ন",
            "response_bn": "উত্তর",
        }
    )
    assert premise_for_row(row) == "retrieved wiki blurb"
    row2 = pd.Series({"context_original": "orig", "context": "rag"})
    assert premise_for_row(row2) == "orig"
    assert nli_cache_tag(None) == "nli_off"
    tag = nli_cache_tag(
        {
            "enabled": True,
            "margin": 0.35,
            "margin_faithful": 0.45,
            "margin_hallucinated": 0.30,
            "tasks": ["a"],
        }
    )
    assert "nli_first" in tag
    assert "mf=0.45" in tag
    assert "mh=0.3" in tag
    assert "bfh=1" in tag


def test_nli_gate_skips_non_configured_tasks_without_model():
    from src.nli import NLIRefiner

    refiner = NLIRefiner(
        {
            "enabled": True,
            "tasks": ["context_grounded_fact"],
            "margin": 0.35,
        }
    )
    df = pd.DataFrame(
        {
            "task_type": ["bangla_grammar", "idiom_meaning_null"],
            "context_original": ["[NULL]", "[NULL]"],
            "context": ["[NULL]", "[NULL]"],
            "prompt_bn": ["q1", "q2"],
            "response_bn": ["a1", "a2"],
        }
    )
    debug = refiner.gate(df)
    assert not debug["nli_applied"].any()
    assert set(debug["nli_skip_reason"]) == {"task_not_in_nli_list"}


def test_think_token_budget_uses_per_task_config():
    verifier = GemmaVerifier()
    verifier.max_think_tokens = 512
    verifier.max_think_tokens_by_task = {
        "math_speed_distance": 256,
        "bangla_grammar": 768,
        "context_grounded_fact": 384,
    }

    assert verifier._think_token_budget("math_speed_distance", ["math_needs_check"]) == 256
    assert verifier._think_token_budget("bangla_grammar", ["bangla_grammar_wide_window"]) == 768
    assert verifier._think_token_budget("context_grounded_fact", ["multi_entity_context"]) == 384
    # Unlisted tasks fall back to max_think_tokens
    assert verifier._think_token_budget("famous_bn_fact_null", ["famous_bn_fact"]) == 512
    assert verifier._think_token_budget("general_fact_null", ["near_threshold"]) == 512
    assert verifier._think_token_budget("idiom_meaning_null", ["lexical_missing_evidence"]) == 512


def test_position_ids_from_mask_left_padding():
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    pos = GemmaVerifier._position_ids_from_mask(mask)
    assert pos.tolist() == [[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]]


def test_router_disabled_policy_uses_original_context_without_rag():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "context": ["প্রসঙ্গ", "[NULL]"],
            "prompt_bn": ["বাংলাদেশের রাজধানী কোনটি?", "বাংলাদেশের রাজধানী কোনটি?"],
            "response_bn": ["ঢাকা", "ঢাকা"],
        }
    )

    routed = apply_router_disabled_policy(df)

    assert routed["task_type"].tolist() == ["context_grounded_other", "other_null"]
    assert routed["context"].tolist() == ["প্রসঙ্গ", "[NULL]"]
    assert routed["rag_used"].tolist() == [False, False]
    assert routed["rag_skipped_reason"].tolist() == ["router_disabled", "router_disabled"]


def test_dataframe_cache_fingerprint_changes_when_evidence_changes():
    df = pd.DataFrame(
        {
            "id": [1],
            "context": ["evidence A"],
            "context_original": ["[NULL]"],
            "prompt_bn": ["prompt"],
            "response_bn": ["response"],
            "task_type": ["general_fact_null"],
        }
    )
    changed = df.copy()
    changed.loc[0, "context"] = "evidence B"

    assert dataframe_cache_fingerprint(df) != dataframe_cache_fingerprint(changed)


def test_testset_audit_200_shape():
    path = os.path.join(os.path.dirname(__file__), "..", "dataset", "testset_audit_200.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "context", "prompt_bn", "response_bn"]
    assert len(df) == 200
    assert df["id"].is_unique


def test_hardware_profile_overlays_gemma_memory_settings():
    config = {
        "runtime": {"hardware_profile": "8gb"},
        "gemma": {
            "device_map": "cuda:0",
            "cuda_max_memory": "14GiB",
            "exemplar_top_k": 3,
            "load_in": "4bit",
        },
        "hardware_profiles": {
            "8gb": {
                "gemma": {
                    "device_map": "auto",
                    "cuda_max_memory": "6GiB",
                    "exemplar_top_k": 0,
                    "load_in": "8bit",
                    "llm_int8_enable_fp32_cpu_offload": True,
                }
            }
        },
    }

    gemma_config = resolve_section(config, "gemma")
    assert gemma_config["device_map"] == "auto"
    assert gemma_config["cuda_max_memory"] == "6GiB"
    assert gemma_config["exemplar_top_k"] == 0
    assert gemma_config["load_in"] == "8bit"
    assert gemma_config["llm_int8_enable_fp32_cpu_offload"] is True


def test_prediction_checkpoint_round_trip(tmp_path):
    checkpoint_path = tmp_path / "test_llm_preds.csv"
    ids = pd.Series([101, 102, 103])
    values = np.array([0.1, 0.9, 0.4])
    metadata = {"checkpoint_source": "gemma", "hardware_profile": "8gb"}

    _save_prediction_checkpoint(str(checkpoint_path), ids, "p_llm", values, metadata=metadata)
    loaded = _load_prediction_checkpoint(
        str(checkpoint_path),
        3,
        "p_llm",
        expected_ids=ids,
        expected_metadata=metadata,
    )

    assert np.allclose(loaded, values)
    assert _load_prediction_checkpoint(str(checkpoint_path), 4, "p_llm") is None
    assert (
        _load_prediction_checkpoint(
            str(checkpoint_path),
            3,
            "p_llm",
            expected_ids=ids,
            expected_metadata={"checkpoint_source": "wrong"},
        )
        is None
    )


def test_resolve_torch_dtype_rejects_unknown_dtype():
    import torch

    assert _resolve_torch_dtype("bf16", torch.float32) is torch.bfloat16
    with pytest.raises(ValueError):
        _resolve_torch_dtype("tinyfloat", torch.float32)


def test_resolve_quantization_mode_accepts_single_load_in():
    assert resolve_quantization_mode({"load_in": "4bit"}) == "4bit"
    assert resolve_quantization_mode({"load_in": "4"}) == "4bit"
    assert resolve_quantization_mode({"load_in": "8bit"}) == "8bit"
    assert resolve_quantization_mode({"load_in": "none"}) == "none"
    assert resolve_quantization_mode({"load_in_4bit": True}) == "4bit"
    with pytest.raises(ValueError):
        resolve_quantization_mode({"load_in": "2bit"})


def test_validate_config_rejects_4bit_auto_offload():
    config = {
        "seed": 42,
        "runtime": {"hardware_profile": "bad"},
        "rag": {"query_mode": "prompt"},
        "decision": {"threshold": 0.5},
        "gemma": {
            "model_name": "test/model",
            "load_in": "4bit",
            "device_map": "auto",
            "llm_int8_enable_fp32_cpu_offload": False,
        },
        "hardware_profiles": {"bad": {}},
    }

    with pytest.raises(ValueError, match='load_in="4bit"'):
        validate_config(config)


def test_validate_config_accepts_8bit_auto_offload():
    config = {
        "seed": 42,
        "runtime": {"hardware_profile": "8gb"},
        "rag": {"query_mode": "prompt"},
        "decision": {"threshold": 0.5},
        "gemma": {
            "model_name": "test/model",
            "load_in": "8bit",
            "device_map": "auto",
            "llm_int8_enable_fp32_cpu_offload": True,
        },
        "hardware_profiles": {"8gb": {}},
    }

    validate_config(config)


def test_validate_config_minimal():
    config = {
        "seed": 42,
        "runtime": {"hardware_profile": "16gb"},
        "rag": {"query_mode": "prompt"},
        "decision": {"threshold": 0.5},
        "gemma": {
            "model_name": "test/model",
            "load_in": "4bit",
            "device_map": "cuda:0",
            "model_loader": "multimodal_lm",
        },
        "hardware_profiles": {"16gb": {}},
    }
    validate_config(config)


def test_repo_config_profiles_resolve_complete_settings():
    import tomllib
    from pathlib import Path

    from src.config_utils import describe_active_profile

    path = Path(__file__).resolve().parents[1] / "configs" / "config.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    validate_config(config)
    active_profile = config["runtime"]["hardware_profile"]
    snap = describe_active_profile(config)
    expected_8gb_model = config["hardware_profiles"]["8gb"]["gemma"]["fast_model_name"]
    if active_profile == "16gb":
        assert snap["fast_verifier_model"] == "google/gemma-4-E4B-it"
    elif active_profile == "8gb":
        assert snap["fast_verifier_model"] == expected_8gb_model
    assert config["data"]["processed_dir"] == "generated/processed"
    assert config["predict"]["llm_predictions_path"].startswith("generated/processed/")

    config["runtime"]["hardware_profile"] = "8gb"
    validate_config(config)
    snap = describe_active_profile(config)
    assert snap["fast_verifier_model"] == expected_8gb_model
    assert snap["enable_think_pass"] is True
    assert snap["rag_batch_size"] == 32


def test_dense_rag():
    import json
    import tempfile

    os.environ["ABOLTABOLYZER_FORCE_CPU"] = "1"
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "corpus")
        os.makedirs(corpus_dir)

        doc1 = {
            "text": "উইন্ডোজে ইউনিকোড ভিত্তিক বাংলা লেখার জন্য ২০০৩ সালের ২৬শে মার্চ অভ্র কীবোর্ড সফটওয়্যারটি আবির্ভূত হয়।"
        }
        doc2 = {"text": "বাংলাদেশের রাজধানী ঢাকা এবং এটি একটি প্রাচীন শহর।"}

        with open(os.path.join(corpus_dir, "corpus_test.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(doc1, ensure_ascii=False) + "\n")
            f.write(json.dumps(doc2, ensure_ascii=False) + "\n")

        index_path = os.path.join(tmpdir, "indexes", "test_dense.pkl")

        from src.rag import BanglaRAG

        rag = BanglaRAG()
        rag.corpus_dir = corpus_dir
        rag.index_path = index_path
        rag.top_k = 2
        rag.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        try:
            assert rag.build_index() is True
        except Exception as exc:
            pytest.skip(f"Embedding model unavailable in this environment: {exc}")
        assert os.path.exists(index_path)

        rag2 = BanglaRAG()
        rag2.index_path = index_path
        rag2.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert rag2.load_index() is True
        assert len(rag2.passages) == 2
        assert rag2.embeddings.dtype == np.float16
        assert rag2.search_embeddings is None

        results = rag2.retrieve("অভ্র কীবোর্ড সফটওয়্যার", similarity_threshold=0.3)
        assert len(results) > 0
        assert "text" in results[0] and "score" in results[0]
        assert "উইন্ডোজে" in results[0]["text"]
        assert rag2.search_embeddings.dtype == np.float32

        evidence, n_hits, sim_max, sim_mean = rag2.format_evidence(results)
        assert n_hits == len(results)
        assert "উইন্ডোজে" in evidence
        assert sim_max >= sim_mean

        rag2.max_evidence_tokens = 3
        short_evidence, _, _, _ = rag2.format_evidence(results)
        assert len(short_evidence.split()) <= 3

        results_irrelevant = rag2.retrieve("সম্পূর্ণ অপ্রাসঙ্গিক প্রশ্ন", similarity_threshold=0.9)
        assert len(results_irrelevant) == 0
