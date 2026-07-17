# Aboltabolyzer — command runner
# Config: configs/config.toml  →  set [runtime].hardware_profile once ("16gb" | "8gb")
# All recipes below read that profile automatically (models, RAG batch sizes, verifier).

export PYTHONPATH := "."
export PYTORCH_CUDA_ALLOC_CONF := "expandable_segments:True"

default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

[doc('Show active hardware_profile and resolved verifier / RAG settings')]
[group('setup')]
show-profile:
    uv run python -c "import tomllib; from src.config_utils import describe_active_profile, validate_config; from src.tui import banner, kv_table; c=tomllib.load(open('configs/config.toml','rb')); validate_config(c); d=describe_active_profile(c); banner('Active profile', str(d.get('hardware_profile'))); kv_table('Resolved from hardware_profile', d)"

[doc('Install Python dependencies')]
[group('setup')]
sync:
    uv sync

[doc('Download BGE-M3 embedder')]
[group('setup')]
download-models:
    uv run python scripts/download_models.py

[doc('Download BGE-M3 + verifier for the active hardware_profile')]
[group('setup')]
download-models-gemma:
    uv run python scripts/download_models.py --include-gemma

[doc('Download Bengali Wikipedia into corpus/wiki/')]
[group('setup')]
download-corpus *args:
    uv run python scripts/download_corpus.py {{args}}

[doc('Download English counterparts for the wiki corpus')]
[group('setup')]
download-english-corpus:
    uv run python scripts/download_english_corpus.py


[doc('LLM-sort a JSONL file into corpus/<source>/ folders')]
[group('setup')]
sort-corpus input *args:
    uv run python scripts/sort_corpus.py {{input}} {{args}}

[doc('Build typed RAG indexes from corpus/<source>/ → indexes/<source>.pkl')]
[group('setup')]
make-rag *args:
    uv run python src/rag.py --make-rag {{args}}

[doc('Models + wiki corpus + RAG indexes (uses active hardware_profile)')]
[group('setup')]
setup: sync download-models-gemma download-corpus make-rag

# ── Pipeline ─────────────────────────────────────────────────────────────────

[doc('Clean raw data → generated/processed/')]
[group('pipeline')]
preprocess:
    uv run python src/preprocess.py

[doc('Routed Gemma/Qwen inference → submissions/<timestamp>/')]
[group('pipeline')]
predict:
    uv run python src/predict.py

[doc('preprocess → predict')]
[group('pipeline')]
run: preprocess predict

[doc('Analyze predictions accuracy against ground truth: just analyze [path_to_submission]')]
[group('pipeline')]
analyze *args:
    uv run python scripts/analyze_submission.py {{args}}

[doc('Full first run for the active hardware_profile: setup → preprocess → predict')]
[group('pipeline')]
first-run: setup preprocess predict

# Back-compat aliases (profile comes from config.toml, not the recipe name)
[doc('Alias of first-run (set hardware_profile=16gb in config.toml)')]
[group('pipeline')]
first-run-16gb: first-run

[doc('Alias of first-run (set hardware_profile=8gb in config.toml)')]
[group('pipeline')]
first-run-8gb: first-run

# ── Cache ────────────────────────────────────────────────────────────────────


[doc('Drop evidence caches + RAG indexes')]
[group('cache')]
clean-rag:
    rm -f generated/processed/train_with_evidence.csv generated/processed/test_with_evidence.csv
    rm -rf indexes/*.pkl indexes/chunks/ indexes/exemplar_index.pkl

[doc('Drop generated/processed/')]
[group('cache')]
clean-processed:
    rm -rf generated/processed/*

[doc('Drop verifier debug logs')]
[group('cache')]
clean-logs:
    rm -f logs/debug_llm_verifier.jsonl logs/debug_llm_verifier_oof_fold_*.jsonl

[doc('clean-rag + clean-processed + clean-logs')]
[group('cache')]
clean-all: clean-rag clean-processed clean-logs

# ── Dev ──────────────────────────────────────────────────────────────────────

[doc('Run pytest')]
[group('dev')]
test:
    uv run pytest tests/

[doc('Ruff lint')]
[group('dev')]
lint:
    uv run ruff check .

[doc('Ruff format')]
[group('dev')]
format:
    uv run ruff format .

[doc('lint + test')]
[group('dev')]
check: lint test
