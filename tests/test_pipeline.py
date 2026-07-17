import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

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


def test_rag_skip_policy_by_task_type():
    assert should_use_rag("math_average", "[NULL]") is False
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
    text = "reason: mismatch\nverdict: Hallucinated\nconfidence: likely\n"
    verdict, confidence, score = GemmaVerifier._parse_think_output(text)
    assert verdict.lower() == "hallucinated"
    assert confidence.lower() == "likely"
    assert score == 0.25


def test_think_prompt_puts_parseable_fields_first():
    instruction = GemmaVerifier()._think_instruction("general_fact_null")
    assert "verdict: Faithful|Hallucinated" in instruction
    assert "confidence: strong|likely|uncertain" in instruction


def test_grammar_instruction_allows_helpful_evidence_without_requiring_it():
    instruction = task_instruction("bangla_grammar")
    assert "not RAG" not in instruction
    assert "missing evidence alone is not H" in instruction


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
        p_fast=0.9,
        task_type="famous_bn_fact_null",
        evidence="[NULL]",
        context_original="[NULL]",
        prompt_bn="রবীন্দ্রনাথ",
        think_reasons=reasons,
    )
    assert "famous_bn_fact" in reasons


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
    base = {
        "cache_version": "verifier-cache-v2",
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


def test_think_token_budget_uses_full_cap_only_for_harder_tasks():
    verifier = GemmaVerifier()
    verifier.max_think_tokens = 512

    assert verifier._think_token_budget("math_speed_distance", ["math_needs_check"]) == 512
    assert verifier._think_token_budget("famous_bn_fact_null", ["famous_bn_fact_null"]) == 512
    assert verifier._think_token_budget("context_grounded_fact", ["multi_entity_context"]) == 512
    assert verifier._think_token_budget("idiom_meaning_null", ["lexical_missing_evidence"]) == 512
    assert verifier._think_token_budget("general_fact_null", ["near_threshold"]) == 512
    assert verifier._think_token_budget("general_fact_null", ["evidence_missing_keyphrase"]) == 512


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
    expected_8gb_model = config["hardware_profiles"]["8gb"]["gemma"]["model_name"]
    if active_profile == "16gb":
        assert snap["verifier_model"] == "google/gemma-4-E4B-it"
    elif active_profile == "8gb":
        assert snap["verifier_model"] == expected_8gb_model
    assert config["data"]["processed_dir"] == "generated/processed"
    assert config["predict"]["llm_predictions_path"].startswith("generated/processed/")

    config["runtime"]["hardware_profile"] = "8gb"
    validate_config(config)
    snap = describe_active_profile(config)
    assert snap["verifier_model"] == expected_8gb_model
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
