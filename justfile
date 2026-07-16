# Aboltabolyzer — command runner
# Run `just` to list recipes by group. Config: configs/config.toml

export PYTHONPATH := "."

default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

[doc('Install Python dependencies (uv sync)')]
[group('setup')]
sync:
    uv sync

[doc('Download XLM-R for the active hardware profile + BGE-M3 embedder')]
[group('setup')]
download-models:
    uv run python scripts/download_models.py

[doc('Download XLM-R for both 8gb/16gb profiles + BGE-M3')]
[group('setup')]
download-models-all:
    uv run python scripts/download_models.py --all-profiles

[doc('Download all XLM-R profiles + BGE-M3 + Gemma (HF gated access may be required)')]
[group('setup')]
download-models-gemma:
    uv run python scripts/download_models.py --all-profiles --include-gemma

[doc('Download full Bengali Wikipedia chunks into corpus/')]
[group('setup')]
download-corpus:
    uv run python scripts/download_corpus.py

[doc('Download 200 wiki articles for quick RAG smoke tests')]
[group('setup')]
download-corpus-small:
    uv run python scripts/download_corpus.py --max-articles 200

[doc('Build indexes/dense_index.pkl from corpus/*.jsonl')]
[group('setup')]
build-index:
    uv run python src/rag.py --build-index

[doc('download-corpus + build-index')]
[group('setup')]
prepare-rag: download-corpus build-index

[doc('download-models-all + full wiki corpus + RAG index (no Gemma)')]
[group('setup')]
prepare-assets: download-models-all download-corpus build-index

[doc('Everything for the 16GB full pipeline: models (incl. Gemma) + wiki + RAG index')]
[group('setup')]
prepare-full: download-models-gemma download-corpus build-index

[doc('Minimal assets for 8GB XLM-R-only debug (uses active profile in config)')]
[group('setup')]
prepare-lite: download-models

# ── Workflows (match README hardware profiles) ───────────────────────────────

[doc('16GB first run: sync → all assets → preprocess → train → predict')]
[group('workflows')]
first-run-16gb: sync prepare-full preprocess train predict

[doc('8GB first run: sync → lite assets → preprocess → train → predict (no Gemma)')]
[group('workflows')]
first-run-8gb: sync prepare-lite preprocess train predict

[doc('Full pipeline: preprocess → train → predict')]
[group('workflows')]
run: preprocess train predict

[doc('Retrain + submit when data is already preprocessed')]
[group('workflows')]
submit: train predict

[doc('RAG smoke test: small corpus → index → train → predict')]
[group('workflows')]
smoke-rag: download-corpus-small build-index train predict

# ── Pipeline steps ───────────────────────────────────────────────────────────

[doc('Clean raw JSON/CSV → dataset/processed/train.csv and test.csv')]
[group('pipeline')]
preprocess:
    uv run python src/preprocess.py

[doc('XLM-R OOF + fold-isolated Gemma OOF + threshold tune')]
[group('pipeline')]
train:
    uv run python src/train.py

[doc('Test inference → submissions/submission.csv + submission_debug.csv')]
[group('pipeline')]
predict:
    uv run python src/predict.py

# ── Cache / artifacts ────────────────────────────────────────────────────────

[doc('Remove cached RAG evidence CSVs (re-run train/predict to rebuild retrieval scores)')]
[group('cache')]
clean-rag-cache:
    rm -f dataset/processed/train_with_evidence.csv dataset/processed/test_with_evidence.csv
    rm -rf indexes/dense_index.pkl indexes/exemplar_index.pkl indexes/chunks/

[doc('Remove all dataset/processed intermediates (keeps raw data in dataset/)')]
[group('cache')]
clean-processed:
    rm -rf dataset/processed/*

[doc('Remove verifier debug logs')]
[group('cache')]
clean-logs:
    rm -f logs/debug_llm_verifier.jsonl logs/debug_llm_verifier_oof_fold_*.jsonl

[doc('clean-rag-cache + clean-processed + clean-logs')]
[group('cache')]
clean-all: clean-rag-cache clean-processed clean-logs

# ── Development ──────────────────────────────────────────────────────────────

[doc('Run pytest (RAG test may download a small embedding model)')]
[group('dev')]
test:
    uv run pytest tests/

[doc('Ruff linter')]
[group('dev')]
lint:
    uv run ruff check .

[doc('Ruff formatter')]
[group('dev')]
format:
    uv run ruff format .

[doc('lint + test')]
[group('dev')]
check: lint test

[doc('Export locked deps to requirements.txt')]
[group('dev')]
export:
    uv export --format requirements-txt -o requirements.txt
