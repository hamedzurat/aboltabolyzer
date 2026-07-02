import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.blender import ScoreBlender
from src.preprocess import clean_text
from src.rag import bengali_tokenize


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


def test_bengali_tokenize():
    text = "অভ্র কিবোর্ড কে উদ্ভাবন করেন ?"
    tokens = bengali_tokenize(text)

    # Ensure punctuation is ignored and tokens are returned
    assert "?" not in tokens
    assert "অভ্র" in tokens
    assert "কিবোর্ড" in tokens


def test_score_blender():
    y_true = np.array([1, 0, 1, 0, 1])

    # Perfect alignment with different models
    p_xlmr = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    p_llm = np.array([0.85, 0.15, 0.9, 0.1, 0.8])

    has_context = np.array([True, False, True, False, True])
    is_c0 = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    is_c1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    is_c2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    blender = ScoreBlender()
    best_f1 = blender.fit(
        y_true,
        p_xlmr,
        p_llm,
        has_context=has_context,
        is_c0=is_c0,
        is_c1=is_c1,
        is_c2=is_c2,
        alpha_step=0.2,
        threshold_step=0.1,
    )

    assert best_f1 > 0.9

    p_blend, preds = blender.predict(
        p_xlmr, p_llm, has_context=has_context, is_c0=is_c0, is_c1=is_c1, is_c2=is_c2
    )
    assert len(p_blend) == 5
    assert len(preds) == 5
    assert (preds == y_true).all()


def test_score_blender_edge_cases():
    y_true = np.array([1, 0, 1])
    p_xlmr = np.array([0.9, 0.1, 0.8])
    p_llm = np.array([0.8, 0.2, 0.7])

    # Case 1: All context present
    has_context_all = np.array([True, True, True])
    blender = ScoreBlender()
    best_f1 = blender.fit(
        y_true, p_xlmr, p_llm, has_context=has_context_all, alpha_step=0.2, threshold_step=0.1
    )
    assert best_f1 > 0.9
    assert blender.is_fitted is True

    # Case 2: All context NULL
    has_context_none = np.array([False, False, False])
    blender2 = ScoreBlender()
    best_f1_none = blender2.fit(
        y_true, p_xlmr, p_llm, has_context=has_context_none, alpha_step=0.2, threshold_step=0.1
    )
    assert best_f1_none > 0.9
    assert blender2.is_fitted is True


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

        # Test retrieval
        results = rag2.retrieve("অভ্র কীবোর্ড সফটওয়্যার", similarity_threshold=0.3)
        assert len(results) > 0
        assert "উইন্ডোজে" in results[0]

        # Test threshold filtering (should be empty for highly irrelevant queries)
        results_irrelevant = rag2.retrieve("সম্পূর্ণ অপ্রাসঙ্গিক প্রশ্ন", similarity_threshold=0.9)
        assert len(results_irrelevant) == 0
