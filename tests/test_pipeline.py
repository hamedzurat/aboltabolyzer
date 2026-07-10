import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.blender import ThresholdDecision
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
