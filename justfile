# List all available commands
default:
    @just --list

# Sync virtual environment and dependencies
sync:
    uv sync

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


