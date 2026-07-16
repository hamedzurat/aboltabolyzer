import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.blender import ThresholdDecision
from src.config_utils import resolve_quantization_mode, resolve_section, validate_config
from src.llm_verifier import _resolve_torch_dtype
from src.predict import _load_prediction_checkpoint, _save_prediction_checkpoint
from src.preprocess import clean_text


def test_clean_text():
    # Test None input
    assert clean_text(None) == "[NULL]"

    # Test zero-width space removal
    dirty_text = "উইন্ডোজে\u200b ইউনিকোড\u200c ভিত্তিক"
    assert clean_text(dirty_text) == "উইন্ডোজে ইউনিকোড ভিত্তিক"

    # Test non-string conversion
    assert clean_text(12345) == "12345"

    # Test empty string conversion
    assert clean_text("   ") == "[NULL]"
    assert clean_text("[NULL]") == "[NULL]"


def test_threshold_decision():
    y_true = np.array([1, 0, 1, 0, 1])
    p_llm = np.array([0.9, 0.1, 0.85, 0.15, 0.95])

    decision = ThresholdDecision()
    best_f1 = decision.fit(y_true, p_llm)

    assert best_f1 > 0.9
    assert decision.is_fitted is True

    p_final, preds = decision.predict(p_llm)
    assert len(p_final) == 5
    assert len(preds) == 5
    assert (preds == y_true).all()


def test_threshold_decision_flips_with_threshold():
    y_true = np.array([1, 0, 1, 0])
    p_llm = np.array([0.55, 0.45, 0.6, 0.4])

    decision = ThresholdDecision()
    decision.fit(y_true, p_llm, threshold_metric="macro_f1")

    _, preds_default = decision.predict(p_llm)
    decision.threshold = 0.7
    _, preds_strict = decision.predict(p_llm)

    assert not np.array_equal(preds_default, preds_strict)


def test_threshold_decision_unfitted_default():
    decision = ThresholdDecision()
    p_llm = np.array([0.8, 0.2, 0.6])
    p_final, preds = decision.predict(p_llm)

    assert np.allclose(p_final, p_llm)
    assert list(preds) == [1, 0, 1]


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
            expected_metadata={"checkpoint_source": "xlmr_fallback"},
        )
        is None
    )


def test_resolve_torch_dtype_rejects_unknown_dtype():
    import pytest
    import torch

    assert _resolve_torch_dtype("bf16", torch.float32) is torch.bfloat16
    with pytest.raises(ValueError):
        _resolve_torch_dtype("tinyfloat", torch.float32)


def test_resolve_quantization_mode_accepts_single_load_in():
    import pytest

    assert resolve_quantization_mode({"load_in": "4bit"}) == "4bit"
    assert resolve_quantization_mode({"load_in": "4"}) == "4bit"
    assert resolve_quantization_mode({"load_in": "8bit"}) == "8bit"
    assert resolve_quantization_mode({"load_in": "none"}) == "none"
    assert resolve_quantization_mode({"load_in_4bit": True}) == "4bit"
    with pytest.raises(ValueError):
        resolve_quantization_mode({"load_in": "2bit"})


def test_validate_config_rejects_4bit_auto_offload():
    import pytest

    config = {
        "seed": 42,
        "num_folds": 2,
        "runtime": {"hardware_profile": "bad"},
        "rag": {"query_mode": "prompt"},
        "xlmr": {"batch_size": 1, "max_length": 16},
        "gemma": {
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
        "num_folds": 2,
        "runtime": {"hardware_profile": "8gb"},
        "rag": {"query_mode": "prompt"},
        "xlmr": {"batch_size": 1, "max_length": 16},
        "gemma": {
            "load_in": "8bit",
            "device_map": "auto",
            "llm_int8_enable_fp32_cpu_offload": True,
        },
        "hardware_profiles": {"8gb": {}},
    }

    validate_config(config)


def test_dense_rag():
    import json
    import tempfile

    # Create a temporary corpus
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "corpus")
        os.makedirs(corpus_dir)

        # Write dummy corpus documents
        doc1 = {
            "text": "উইন্ডোজে ইউনিকোড ভিত্তিক বাংলা লেখার জন্য ২০০৩ সালের ২৬শে মার্চ অভ্র কীবোর্ড সফটওয়্যারটি আবির্ভূত হয়।"
        }
        doc2 = {"text": "বাংলাদেশের রাজধানী ঢাকা এবং এটি একটি প্রাচীন শহর।"}

        with open(os.path.join(corpus_dir, "corpus_test.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(doc1, ensure_ascii=False) + "\n")
            f.write(json.dumps(doc2, ensure_ascii=False) + "\n")

        index_path = os.path.join(tmpdir, "indexes", "test_dense.pkl")

        # Instantiate BanglaRAG and manually override configs for testing
        from src.rag import BanglaRAG

        rag = BanglaRAG()
        rag.corpus_dir = corpus_dir
        rag.index_path = index_path
        rag.top_k = 2
        # Use a lightweight fast embedding model for testing
        rag.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Test index building
        assert rag.build_index() is True
        assert os.path.exists(index_path)

        # Test loading index
        rag2 = BanglaRAG()
        rag2.index_path = index_path
        rag2.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert rag2.load_index() is True
        assert len(rag2.passages) == 2

        # Test retrieval (scored hits)
        results = rag2.retrieve("অভ্র কীবোর্ড সফটওয়্যার", similarity_threshold=0.3)
        assert len(results) > 0
        assert "text" in results[0] and "score" in results[0]
        assert "উইন্ডোজে" in results[0]["text"]

        evidence, n_hits, sim_max, sim_mean = rag2.format_evidence(results)
        assert n_hits == len(results)
        assert "উইন্ডোজে" in evidence
        assert sim_max >= sim_mean

        # Truncation respects max_evidence_tokens
        rag2.max_evidence_tokens = 3
        short_evidence, _, _, _ = rag2.format_evidence(results)
        assert len(short_evidence.split()) <= 3

        # Test threshold filtering (should be empty for highly irrelevant queries)
        results_irrelevant = rag2.retrieve("সম্পূর্ণ অপ্রাসঙ্গিক প্রশ্ন", similarity_threshold=0.9)
        assert len(results_irrelevant) == 0
