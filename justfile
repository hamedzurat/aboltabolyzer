# List all available commands
export PYTHONPATH := "."
export HF_HUB_ENABLE_HF_TRANSFER := "1"

default:
    @just --list

# Sync virtual environment and dependencies
sync:
    uv sync

# Download the active-profile XLM-R model and RAG embedding model into models/hf/
download-models:
    uv run python scripts/download_models.py

# Download models for both 8GB and 16GB profiles, plus the RAG embedding model
download-models-all:
    uv run python scripts/download_models.py --all-profiles

# Download Gemma too. This may require Hugging Face gated-model access.
download-models-gemma:
    uv run python scripts/download_models.py --all-profiles --include-gemma

# Download and chunk Bengali Wikipedia into corpus/wiki_bn.jsonl
download-corpus:
    uv run python scripts/download_corpus.py

# Small corpus download for smoke testing the RAG path without pulling the full wiki
download-corpus-small:
    uv run python scripts/download_corpus.py --max-articles 200

# Download corpus and build the dense RAG index
prepare-rag: download-corpus build-index

# Download all non-gated assets needed for normal experiments
prepare-assets: download-models-all download-corpus build-index

# Run data cleaning and preprocessing
preprocess:
    uv run python src/preprocess.py

# Build Dense RAG index from corpus
build-index:
    uv run python src/rag.py --build-index

# Run the 5-fold training loop (XLM-R + LLM OOF predictions + Blender)
train:
    uv run python src/train.py

# Run test set inference and generate final submission file
predict:
    uv run python src/predict.py

# Run the test suite
test:
    uv run pytest tests/

# Lint the codebase using ruff
lint:
    uv run ruff check .

# Format the codebase using ruff
format:
    uv run ruff format .

# Export project dependencies to standard requirements.txt file
export:
    uv export --format requirements-txt -o requirements.txt

