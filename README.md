# Aboltabolyzer

Bangla hallucination detection for competition submission.

| Field         | Meaning                             |
| ------------- | ----------------------------------- |
| `context`     | Supporting passage, or `[NULL]`     |
| `prompt_bn`   | Bengali question / instruction      |
| `response_bn` | Candidate Bengali answer            |
| **label 0**   | Hallucinated, unsupported, or wrong |
| **label 1**   | Faithful, supported, correct        |

**Architecture:** XLM-R (LoRA) produces an encoder prior → Gemma gives the final verdict → one OOF-tuned threshold on `p_llm` decides the label. No RandomForest blender.

**Config:** choose the machine profile in [`configs/config.toml`](configs/config.toml),
then run the matching `just` workflow.

## Quick Start

1. Put the competition files in `dataset/`:

```text
dataset/sample_dataset.json
dataset/.3_testset.csv
dataset/sample_submission.csv
```

2. Pick a profile in `configs/config.toml`:

```toml
[runtime]
hardware_profile = "8gb"   # RTX 5060 mobile 8GB: int8 Gemma with CPU offload
# hardware_profile = "16gb" # RTX 5060 16GB: faster all-GPU Gemma
use_llm_verifier = true
```

3. Run one command:

```bash
just first-run-8gb   # 8GB, slower but designed to fit
# or
just first-run-16gb  # 16GB, faster full pipeline
```

4. Upload:

```text
submissions/latest/submission.csv
```

Inspect:

```text
submissions/latest/submission_debug.csv
```

Gemma may require Hugging Face access. If the Gemma repository is gated, log in
with `huggingface-cli login` before `just prepare-full`.

---

## Pipeline diagram

```mermaid
flowchart TD
    A["Raw data<br/>context, prompt_bn, response_bn, optional label"] --> B["Preprocess<br/>NFC normalize · strip zero-width chars<br/>whitespace cleanup · empty → [NULL]"]
    B --> C["has_context = context != [NULL]"]

    C --> D{"Provided context?"}
    D -->|"Yes"| E["Keep original context"]
    D -->|"No"| F{"RAG index exists?"}

    F -->|"No"| G["context stays [NULL]"]
    F -->|"Yes"| H["Query = prompt_bn only<br/>(see query_mode in config)"]

    H --> I["Embed query · BGE-M3"]
    I --> J["Dense cosine search · top_k passages"]
    J --> K{"score >= similarity_threshold?"}
    K -->|"Yes"| L["Join passages · truncate max_evidence_tokens<br/>log n_retrieved · sim_max · sim_mean"]
    K -->|"No"| G

    E --> M["Cross-encoder input"]
    G --> M
    L --> M

    M --> N["text = context/evidence<br/>text_pair = prompt_bn + response_bn"]
    N --> O["XLM-R + LoRA · configured CV folds<br/>8gb: roberta-base · 16gb: roberta-large"]

    O --> P["p_xlmr = P(faithful)"]
    P --> Q["OOF → dataset/processed/oof_xlmr.csv"]

    M --> R{"use_llm_verifier?"}
    R -->|"No"| S["p_llm = p_xlmr"]
    R -->|"Yes"| T["Fold-isolated exemplar index<br/>few-shot from train fold only"]

    T --> U["Retrieve similar train exemplars<br/>8gb default: 0 · 16gb default: 3"]
    U --> V["Cultural band C0/C1/C2<br/>routes think-pass only"]

    Q --> X["Gemma prompt<br/>encoder prior + evidence + Q/A + exemplars"]
    U --> X
    V --> X
    X --> Y["Fast pass · F/H next-token logits<br/>p_fast = P(Faithful)"]

    Y --> AA{"Think trigger?<br/>uncertain · encoder disagree<br/>C0+NULL · C2"}
    AA -->|"No"| AB["p_llm = p_fast"]
    AA -->|"Yes"| AC["CoT think pass<br/>verdict: Faithful / Hallucinated"]
    AC --> AD["Parse verdict → p_llm ∈ {0.0, 1.0}"]

    AB --> AE["OOF p_llm · train only"]
    AD --> AE
    S --> AE

    AE --> AI["Grid-search threshold on OOF p_llm<br/>macro_f1 or f1_class_0"]
    AI --> AJ["label = 1 if p_llm >= threshold else 0"]

    AJ --> AK["submissions/<timestamp>/submission.csv"]
    AJ --> AL["submissions/<timestamp>/submission_debug.csv<br/>RAG scores · p_xlmr · p_llm · think meta · label"]
    AK --> AM["submissions/latest → <timestamp>/ (symlink)"]
```

---

## Installation

Requires [uv](https://github.com/astral-sh/uv) and [just](https://github.com/casey/just).

```bash
just sync      # install dependencies
just           # list all commands
```

---

## Hardware profiles

Pick one profile, edit **`configs/config.toml` `[runtime]`**, then run the command block below.

### Profile A — 16GB full pipeline (recommended for submission)

**Machine:** RTX 5060 16GB, Kaggle P100/T4, or similar.

Use this when Gemma can fit fully on GPU. It is the faster path and keeps
dynamic Gemma exemplars enabled.

**Set in `configs/config.toml`:**

```toml
[runtime]
hardware_profile = "16gb"
use_llm_verifier = true
fail_on_model_error = true

[predict]
use_checkpoints = true
force_recompute = false

[hardware_profiles.16gb.gemma]
load_in = "4bit"
device_map = "cuda:0"
exemplar_top_k = 3
max_input_tokens = 3072

[hardware_profiles.16gb.xlmr]
use_amp = true
batch_size = 4

[hardware_profiles.16gb.rag]
batch_size = 64
query_batch_size = 64
max_seq_length = 512
```

**Commands (first time):**

```bash
just first-run-16gb
```

Or step by step:

```bash
just sync
just prepare-full             # models (incl. Gemma) + wiki + RAG index
just preprocess
just train
just predict
```

**Commands (after assets exist):**

```bash
just submit                   # train (fresh) + predict
just submit-continue          # resume from existing checkpoints + predict
# or, if raw data changed:
just run                      # preprocess + fresh train + predict
```

**Notes:** Training runs Gemma once per CV fold for OOF scoring, then builds a
full exemplar index for inference. Inference is one Gemma fast pass per test row,
plus a think pass on triggered rows.

---

### Profile B — 8GB low-VRAM Gemma

**Machine:** RTX 5060 mobile 8GB or any GPU too small for all-GPU Gemma.

Use this when avoiding OOM matters more than raw speed. Gemma uses 8-bit
quantization with fp32 CPU offload, capped GPU placement, shorter prompts, and
no dynamic exemplars. It also disables the extra cultural-band and think-pass
Gemma calls by default. This is slower and less feature-complete than 16GB, but
it is the fit-first profile.

**Set in `configs/config.toml`:**

```toml
[runtime]
hardware_profile = "8gb"
use_llm_verifier = true
fail_on_model_error = true

[predict]
use_checkpoints = true
force_recompute = false

[hardware_profiles.8gb.gemma]
load_in = "8bit"
llm_int8_enable_fp32_cpu_offload = true
device_map = "auto"
cuda_max_memory = "6GiB"
max_input_tokens = 1536
use_inputs_embeds_for_forward = true
use_language_model_direct = true
classify_cultural_band = false
enable_think_pass = false
exemplar_top_k = 0

[hardware_profiles.8gb.xlmr]
model_name = "FacebookAI/xlm-roberta-base"
batch_size = 2
use_amp = false

[hardware_profiles.8gb.rag]
batch_size = 32
query_batch_size = 32
max_seq_length = 512
```

**Commands:**

```bash
just first-run-8gb
```

Or step by step: `just sync` → `just prepare-full` → `just preprocess` → `just train` → `just predict`

This profile uses the bitsandbytes int8 CPU-offload path because 4-bit
quantization cannot be auto-dispatched to CPU/disk on this stack. It uses
Gemma's underlying text language model directly for fast-pass scoring, bypassing
the multimodal wrapper path that can touch meta tensors under CPU offload. It
should resume cleanly if the run is interrupted.

For a pure XLM-R debug run without loading Gemma, set:

```toml
[runtime]
use_llm_verifier = false
```

> **Tip:** `just train` always wipes `models/xlmr/` and retrains from scratch.
> Use `just train-continue` to resume — it skips folds whose checkpoint already exists.

Use this to debug preprocessing, RAG, and cross-encoder training without loading Gemma.

---

### Profile C — RAG smoke test

Add to either profile after first setup:

```bash
just smoke-rag
```

Or: `just download-corpus-small` → `just build-index` → `just submit`

Compare `submission_debug.csv` columns `n_retrieved`, `retrieval_sim_max`, `rag_filled` with and without the index.

---

### OOM / stability

| Symptom          | Fix                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| XLM-R OOM        | Lower `batch_size` or `max_length` in active hardware profile (`configs/config.toml`)                               |
| Gemma OOM        | Use the `8gb` profile, lower `cuda_max_memory`, `max_input_tokens`, `max_think_tokens`, or set `exemplar_top_k = 0` |
| RAG Indexing OOM | Lower `batch_size` or `max_seq_length` in `[rag]` or profile-specific `[hardware_profiles.<profile>.rag]` section   |
| Stale RAG scores | `just clean-rag-cache` then re-run `just train` / `just predict`                                                    |

### Prediction resume

`just predict` writes stage checkpoints:

| File                                       | Stage                             |
| ------------------------------------------ | --------------------------------- |
| `dataset/processed/test_with_evidence.csv` | RAG-filled test contexts          |
| `dataset/processed/test_xlmr_preds.csv`    | XLM-R ensemble probabilities      |
| `logs/debug_llm_verifier.jsonl`            | Row-level Gemma verifier cache    |
| `dataset/processed/test_llm_preds.csv`     | Complete Gemma probability vector |

Leave `[predict].use_checkpoints = true` to resume after an OOM or interruption.
Set `[predict].force_recompute = true` for one clean rerun without deleting files.

### Daily workflow

Use these once assets and preprocessing already exist:

| Goal                      | Command                                                     | Notes                                                              |
| ------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ |
| Resume fold training      | `just train-continue`                                       | Keeps `models/xlmr/best_fold_*.pt` and skips completed folds       |
| Fresh train, then predict | `just submit`                                               | Deletes XLM-R fold checkpoints first because `just train` is fresh |
| Resume training + predict | `just submit-continue`                                      | Best after an interrupted training run                             |
| Re-run prediction only    | `just predict`                                              | Reuses RAG, XLM-R, and Gemma prediction checkpoints when valid     |
| Force clean prediction    | Set `[predict].force_recompute = true`, then `just predict` | Does not delete files; ignores checkpoint reuse for that run       |
| Rebuild RAG evidence      | `just clean-rag-cache`                                      | Use after changing corpus, RAG index, or RAG knobs                 |

Checkpoint CSVs include metadata for `hardware_profile` and checkpoint source.
If you switch between `8gb` and `16gb`, incompatible prediction checkpoints are
ignored instead of silently reused.

### Performance tuning

The defaults are conservative. Tune in `configs/config.toml`:

| Knob                                           | Where                         | Effect                                                          |
| ---------------------------------------------- | ----------------------------- | --------------------------------------------------------------- |
| `batch_size`                                   | `[hardware_profiles.*.xlmr]`  | Bigger is faster until XLM-R OOMs                               |
| `use_amp`                                      | `[hardware_profiles.*.xlmr]`  | Faster/lower VRAM on newer GPUs; enabled for 16GB by default    |
| `num_workers`, `pin_memory`, `prefetch_factor` | `[hardware_profiles.*.xlmr]`  | Improves DataLoader throughput                                  |
| `query_batch_size`                             | `[hardware_profiles.*.rag]`   | Batch-encodes RAG queries; bigger is faster until embedding OOM |
| `device_map`                                   | `[hardware_profiles.*.gemma]` | `"cuda:0"` is fastest; `"auto"` allows CPU offload              |
| `load_in`                                      | `[hardware_profiles.*.gemma]` | `"4bit"` for 16GB all-GPU; `"8bit"` for 8GB CPU offload         |
| `cuda_max_memory`                              | `[hardware_profiles.*.gemma]` | Lower to avoid OOM on 8GB; raise on larger cards                |
| `max_input_tokens`                             | `[hardware_profiles.*.gemma]` | Lower saves memory/time but may truncate evidence               |
| `use_language_model_direct`                    | `[hardware_profiles.*.gemma]` | Bypasses Gemma 4 multimodal wrapper for text-only offload runs  |
| `use_inputs_embeds_for_forward`                | `[hardware_profiles.*.gemma]` | Optional text-embedding workaround for non-direct forwards      |
| `classify_cultural_band`, `enable_think_pass`  | `[hardware_profiles.*.gemma]` | Extra Gemma calls; enabled on 16GB, disabled on 8GB by default  |
| `exemplar_top_k`                               | `[hardware_profiles.*.gemma]` | More examples may help quality but increases prompt cost        |

For 8GB, do not expect Gemma to saturate the GPU: CPU offload is a fit-first
mode. For 16GB, keep `device_map = "cuda:0"` for best throughput.
The config is validated before training/prediction, so unsupported combinations
such as `load_in = "4bit"` with `device_map = "auto"` fail early with a clear
message.

---

## Command reference

Run `just` to see recipes grouped by **setup**, **workflows**, **pipeline**, **cache**, **dev**.

### Workflows (start here)

| Command                | What it does                                                                  |
| ---------------------- | ----------------------------------------------------------------------------- |
| `just first-run-16gb`  | sync → prepare-full → preprocess → **fresh train** → predict                  |
| `just first-run-8gb`   | sync → prepare-full → preprocess → **fresh train** → predict (low-VRAM Gemma) |
| `just run`             | preprocess → fresh train → predict                                            |
| `just submit`          | **fresh train** → predict (data already preprocessed)                         |
| `just submit-continue` | resume existing checkpoints → predict                                         |
| `just smoke-rag`       | small wiki → index → fresh train → predict                                    |

### Setup

| Command                      | What it does                                            |
| ---------------------------- | ------------------------------------------------------- |
| `just sync`                  | Install Python deps via uv                              |
| `just prepare-full`          | Gemma + both XLM-R profiles + BGE-M3 + wiki + RAG index |
| `just prepare-assets`        | Both XLM-R profiles + BGE-M3 + wiki + index (no Gemma)  |
| `just prepare-lite`          | Active-profile XLM-R + BGE-M3 only                      |
| `just prepare-rag`           | Full wiki download + build index                        |
| `just download-models`       | XLM-R for active profile + BGE-M3                       |
| `just download-models-all`   | XLM-R for both profiles + BGE-M3                        |
| `just download-models-gemma` | Above + Gemma                                           |
| `just download-corpus`       | Full Bengali Wikipedia → `corpus/`                      |
| `just download-corpus-small` | 200-article wiki sample                                 |
| `just build-index`           | Build `indexes/dense_index.pkl`                         |

### Pipeline

| Command               | What it does                                               |
| --------------------- | ---------------------------------------------------------- |
| `just preprocess`     | Clean data → `dataset/processed/train.csv`, `test.csv`     |
| `just train`          | Wipe `models/xlmr/` + full training pipeline (fresh start) |
| `just train-continue` | Resume training — keeps existing fold checkpoints          |
| `just predict`        | Test inference → `submissions/<timestamp>/`                |

### Cache

| Command                | What it does                           |
| ---------------------- | -------------------------------------- |
| `just clean-rag-cache` | Drop cached `*_with_evidence.csv` only |
| `just clean-processed` | Remove all of `dataset/processed/`     |
| `just clean-logs`      | Remove Gemma JSONL debug logs          |
| `just clean-all`       | All of the above                       |

### Dev

| Command                     | What it does             |
| --------------------------- | ------------------------ |
| `just test`                 | pytest                   |
| `just lint` / `just format` | Ruff                     |
| `just check`                | lint + test              |
| `just export`               | Write `requirements.txt` |

---

## Outputs

### Training (`just train`)

| Path                                        | Contents                                         |
| ------------------------------------------- | ------------------------------------------------ |
| `dataset/processed/train_with_evidence.csv` | Train rows after RAG (+ retrieval score columns) |
| `dataset/processed/oof_xlmr.csv`            | OOF XLM-R probabilities                          |
| `dataset/processed/train_with_preds.csv`    | OOF `p_xlmr` + OOF `p_llm`                       |
| `models/xlmr/best_fold_*.pt`                | LoRA checkpoints, one per configured CV fold     |
| `models/blender_config.pkl`                 | Tuned threshold (legacy filename)                |
| `indexes/exemplar_index.pkl`                | Full train exemplars for inference               |
| `logs/debug_llm_verifier_oof_fold_*.jsonl`  | Per-fold Gemma debug during training             |

### Prediction (`just predict`)

| Path                                           | Contents                                    |
| ---------------------------------------------- | ------------------------------------------- |
| `submissions/<timestamp>/submission.csv`       | `id, label` — upload this                   |
| `submissions/<timestamp>/submission_debug.csv` | Full trace for error analysis               |
| `submissions/latest`                           | Symlink → most recent timestamped run dir   |
| `dataset/processed/test_with_evidence.csv`     | Test after RAG                              |
| `dataset/processed/test_xlmr_preds.csv`        | Resumable XLM-R test probabilities          |
| `dataset/processed/test_llm_preds.csv`         | Resumable Gemma/fallback test probabilities |
| `dataset/processed/test_with_preds.csv`        | Test with `p_xlmr`, `p_llm`                 |
| `logs/debug_llm_verifier.jsonl`                | Per-row Gemma debug at inference            |

### `submission_debug.csv` columns

The debug CSV is intentionally wide. Key groups:

| Group            | Useful columns                                                                                                                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Final decision   | `label`, `p_final`, `threshold`, `threshold_margin`, `threshold_abs_margin`, `threshold_metric`                                                                                                                                        |
| Model scores     | `p_xlmr`, `p_llm`, `llm_minus_xlmr`, `encoder_disagree`, `xlmr_llm_label_disagree`                                                                                                                                                     |
| Gemma think pass | `p_llm_no_think`, `p_llm_from_log`, `p_llm_log_delta`, `triggered_think`, `think_reasons`, `thinking_cot`, `think_changed_label`                                                                                                       |
| Cultural routing | `is_c0`, `is_c1`, `is_c2`                                                                                                                                                                                                              |
| RAG/evidence     | `has_context`, `evidence_is_null`, `rag_filled`, `n_retrieved`, `retrieval_sim_max`, `retrieval_sim_mean`, `context_word_len`                                                                                                          |
| Run provenance   | `run_timestamp`, `hardware_profile`, `used_llm_verifier`, `llm_checkpoint_source`, `xlmr_from_checkpoint`, `llm_from_checkpoint`                                                                                                       |
| Config snapshot  | `xlmr_model_name`, `xlmr_batch_size`, `xlmr_use_amp`, `gemma_load_in`, `gemma_device_map`, `gemma_cuda_max_memory`, `gemma_max_input_tokens`, `gemma_use_language_model_direct`, `gemma_enable_think_pass`, `rag_query_batch_size` |
| Artifact paths   | `submission_path`, `debug_path`, `xlmr_checkpoint_path`, `llm_checkpoint_path`, `verifier_debug_log_path`                                                                                                                              |

Start with `threshold_abs_margin` to find borderline decisions, then inspect
`encoder_disagree`, `triggered_think`, and `rag_filled` for likely error modes.

---

## Data files

Place competition files here:

```text
dataset/sample_dataset.json    # labeled train (299 rows)
dataset/.3_testset.csv         # test (2516 rows)
dataset/sample_submission.csv  # format example
```

Corpus for RAG (optional, built via `just download-corpus`):

```text
corpus/*.jsonl                 # { "text": "..." } or { "passage": "..." }
```

---

## File-by-file guide

### Root

| File                         | Role                                                     |
| ---------------------------- | -------------------------------------------------------- |
| `README.md`                  | This document — architecture, hardware recipes, commands |
| `configs/config.toml`        | **All configuration knobs with inline docs**             |
| `justfile`                   | Command runner                                           |
| `pyproject.toml` / `uv.lock` | Dependencies (uv)                                        |
| `requirements.txt`           | Exported deps for non-uv environments                    |
| `main.py`                    | Placeholder; use `just train` / `just predict`           |

### `src/`

| File              | Role                                                                   |
| ----------------- | ---------------------------------------------------------------------- |
| `preprocess.py`   | Bengali text cleanup, `[NULL]` handling, `has_context`                 |
| `rag.py`          | BGE-M3 dense index, scored retrieval, `max_evidence_tokens` truncation |
| `xlmr_encoder.py` | LoRA fine-tune, configured-fold OOF, test ensemble                     |
| `llm_verifier.py` | Gemma verifier — encoder prior, fast/think passes, C0/C1/C2 routing    |
| `blender.py`      | `ThresholdDecision` — OOF threshold on `p_llm` only                    |
| `train.py`        | Orchestrates preprocess → RAG → XLM-R → OOF Gemma → threshold          |
| `predict.py`      | Orchestrates RAG → XLM-R → Gemma → threshold → submission + debug CSV  |
| `evaluate.py`     | Macro F1, per-class F1, confusion matrix                               |
| `config_utils.py` | Hardware profile resolution, model path cache, runtime torch settings  |

### `scripts/`

| File                 | Role                                              |
| -------------------- | ------------------------------------------------- |
| `download_models.py` | Hugging Face snapshots → `models/hf/`             |
| `download_corpus.py` | Bengali Wikipedia chunks → `corpus/wiki_bn.jsonl` |

### Generated (gitignored)

| Path                 | Role                                     |
| -------------------- | ---------------------------------------- |
| `dataset/processed/` | Intermediate CSVs                        |
| `corpus/`            | RAG source documents                     |
| `indexes/`           | `dense_index.pkl`, `exemplar_index.pkl`  |
| `models/`            | Fold weights, threshold pickle, HF cache |
| `submissions/`       | Final + debug CSVs                       |
| `logs/`              | Gemma JSONL debug logs                   |

---

## Known weaknesses

1. **Small labeled set (299)** — threshold and LoRA both fit on little data; treat train metrics as noisy.
2. **C0/C1/C2 bands** — LLM-guessed, only used to trigger think; can misfire on edge cases.
3. **Wiki-only RAG** — weak on math, spelling MCQs, recent news; think-pass is the fallback.
4. **Expensive OOF Gemma training** — one verifier pass per configured CV fold; plan GPU time accordingly.
5. **8GB Gemma is fit-first** — int8 CPU offload avoids OOM but is much slower than all-GPU 16GB inference.
6. **Cached evidence CSVs** — if you change corpus/index/config RAG knobs, delete `*_with_evidence.csv` before re-running.
7. **Kaggle packaging not automated** — fold checkpoints, index, and exemplars must be bundled manually as a Dataset for offline submit.

---

## Roadmap

**Done**

- [x] Gemma-led verifier with XLM-R encoder prior
- [x] Fold-isolated OOF Gemma + OOF threshold (no RandomForest)
- [x] Think triggers: uncertainty, encoder disagreement, C0+NULL, C2
- [x] `max_evidence_tokens` + retrieval scores in debug CSV
- [x] 8GB / 16GB hardware profiles
- [x] Resumable prediction checkpoints with profile/source metadata
- [x] Wide debug submission CSV with score, RAG, checkpoint, and config provenance
- [x] Batched RAG query retrieval and faster XLM-R inference settings
- [x] `just` recipes for assets, train, predict

**Next**

- [ ] Compare `macro_f1` vs `f1_class_0` threshold on real OOF runs
- [ ] Kaggle Dataset bundle: index + fold checkpoints + exemplars + threshold
- [ ] Optional curated corpus packs (BD history, grammar) beyond wiki
- [ ] Submission notebook template (internet-off inference)
