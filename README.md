# Aboltabolyzer

Bangla hallucination detection for competition submission.

| Field         | Meaning                             |
| ------------- | ----------------------------------- |
| `context`     | Supporting passage, or `[NULL]`     |
| `prompt_bn`   | Bengali question / instruction      |
| `response_bn` | Candidate Bengali answer            |
| **label 0**   | Hallucinated, unsupported, or wrong |
| **label 1**   | Faithful, supported, correct        |

**Architecture:** deterministic task router → typed evidence policy (per-corpus RAG) → fast pass → **NLI-first gate** (skip think when confident) → think fallback → fixed threshold on `p_llm` (`decision.threshold`, default `0.5`).

No training. Inference only.

**Config:** set `hardware_profile` once in [`configs/config.toml`](configs/config.toml). Every `just` recipe (setup, predict, downloads) follows that profile.

---

## Quick start

Requires [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just), and a CUDA GPU.

1. Put competition files in `dataset/`:

```text
dataset/sample_dataset.json    # labeled train (few-shot exemplars on 16GB)
dataset/testset.csv            # full test → submission
dataset/sample_submission.csv  # id,label format example
```

2. Pick a profile in `configs/config.toml`:

```toml
[runtime]
hardware_profile = "16gb"  # RTX 5060 Ti 16GB → Gemma 4
# hardware_profile = "8gb" # RTX 5060 mobile 8GB → ungated Qwen
```

```bash
just show-profile   # confirm resolved verifier / VRAM / RAG batch sizes
```

3. Run on a machine with a real GPU:

```bash
just first-run   # sync → models for this profile + wiki + indexes → preprocess → predict
```

For a 200-row dry run, point `[data].test_path` at `dataset/testset_audit_200.csv` (same columns as the full test file), then `just run`.

4. Upload only:

```text
submissions/latest/submission.csv
```

Inspect:

```text
submissions/latest/submission_debug.csv
```

The 8GB profile uses ungated `Qwen/Qwen3-1.7B`. The 16GB profile uses Gemma 4, which may require Hugging Face access.

---

## Pipeline diagram

```mermaid
flowchart TD
    A["Raw test CSV<br/>id · context · prompt_bn · response_bn"] --> B["preprocess.py<br/>NFC · strip ZW chars · empty → [NULL]<br/>has_context = context ≠ [NULL]"]

    TRAIN["sample_dataset.json<br/>→ processed/train.csv"] -.->|16GB: build exemplar index<br/>exemplar_top_k &gt; 0| EX["indexes/exemplar_index.pkl<br/>few-shot F/H neighbors"]

    B --> C["router.py<br/>deterministic task_type"]

    C --> D{"Evidence policy<br/>evidence_policy.py"}

    D -->|"original context present<br/>context_grounded_* / famous_bn_fact_context"| E["Keep original context<br/>rag_used = false<br/>evidence_source = original_context"]

    D -->|"math_* / calendar / translation<br/>RAG_SKIP_TASKS"| F["No RAG<br/>LLM judges via task prompt<br/>context stays [NULL]<br/>evidence_source = none"]

    D -->|"other_null + not factual prompt"| F2["No RAG<br/>rag_skipped_reason = other_null_not_factual"]

    D -->|"NULL + RAG allowed<br/>general_fact_null · factual other_null · famous_bn_fact_null<br/>idiom · literal · grammar"| G{"Resolve typed source<br/>TASK_RAG_SOURCE<br/>only if indexes/*.pkl exists"}

    G -->|"general_fact_null / factual other_null / famous_bn_fact_null"| H1["source = wiki"]
    G -->|"idiom_meaning_null"| H3["source = idioms"]
    G -->|"literal_meaning_null"| H4["source = literal"]
    G -->|"bangla_grammar"| H5["source = grammar"]
    G -->|"index missing"| K["Skip retrieval<br/>rag_skipped_reason = index_missing:source<br/>context stays [NULL]"]

    H1 --> I["BGE-M3 dense retrieve<br/>query = prompt_bn default<br/>top_k · similarity_threshold<br/>truncate max_evidence_tokens"]
    H3 --> I
    H4 --> I
    H5 --> I

    I --> L["Overwrite context with evidence<br/>n_retrieved · sim_max · sim_mean<br/>rag_used = true<br/>evidence_source = rag:source"]

    E --> M["Task-specific verifier prompt<br/>English scaffolding · TASK_INSTRUCTIONS"]
    F --> M
    F2 --> M
    K --> M
    L --> M
    EX -.-> M

    M --> N["Fast pass<br/>next-token logits F / H<br/>p_fast = P Faithful"]

    N --> NLI{"NLI-first gate?<br/>task in [nli].tasks<br/>+ non-empty premise<br/>+ |entail−contradict| ≥ margin"}

    NLI -->|"Yes · confident"| NLI_OUT["p_llm = NLI score<br/>skip think"]
    NLI -->|"No / uncertain / other tasks"| O{"Think? OR of triggers<br/>and enable_think_pass<br/>near threshold · famous_bn<br/>multi-entity context<br/>math / grammar / …"}

    O -->|"No triggers or think disabled"| P["p_llm = p_fast"]
    O -->|"Yes"| Q["Think pass CoT<br/>verdict: Faithful|Hallucinated"]

    Q --> R{"Parse verdict?"}
    R -->|"Faithful / Hallucinated"| S["Map to soft score<br/>0.90 / 0.10"]
    R -->|"unparsed"| T["Keep p_fast<br/>think_reasons += verdict_unparsed"]

    NLI_OUT --> U["decision.threshold default 0.5<br/>label = 1 if p_llm ≥ threshold else 0"]
    P --> U
    S --> U
    T --> U

    U --> V["submissions/timestamp/submission.csv<br/>id, label only"]
    U --> W["submission_debug.csv<br/>task_type · rag_source · p_fast · p_think · p_llm<br/>think meta · evidence fields"]
    V --> X["submissions/latest → timestamp/"]

    CHK["Optional resume<br/>test_with_evidence.csv<br/>test_llm_preds.csv<br/>debug_llm_verifier.jsonl"] -.->|use_checkpoints| U
```

### Task → corpus source

| `task_type`                | Evidence                    | Corpus source |
| -------------------------- | --------------------------- | ------------- |
| `context_grounded_*`       | Original context only       | —             |
| `famous_bn_fact_context`   | Original context only       | —             |
| `general_fact_null`        | Typed RAG                   | `wiki`        |
| `other_null` (factual)     | Typed RAG                   | `wiki`        |
| `other_null` (not factual) | No RAG                      | —             |
| `famous_bn_fact_null`      | Typed RAG                   | `wiki`        |
| `idiom_meaning_null`       | Typed RAG when index exists | `idioms`      |
| `literal_meaning_null`     | Typed RAG when index exists | `literal`     |
| `bangla_grammar`           | Typed RAG when index exists | `grammar`     |
| `math_*` / `calendar_*`    | No RAG — LLM calculates     | —             |
| `translation_or_bilingual` | No RAG — bilingual judge    | —             |

Empty corpus folders are fine: `just make-rag` skips them, and predict records `index_missing:<source>`. Wiki is filled by `just download-corpus`; idiom / literal / grammar need curated `*.jsonl`.

---

## Installation

```bash
just sync      # install dependencies
just           # list all commands
```

---

## Hardware profiles

Set **`[runtime].hardware_profile`** once. Shared knobs stay in `[gemma]` / `[rag]`; machine-specific model, VRAM, think, and RAG batch sizes live under `[hardware_profiles.<name>.*]`. `resolve_section()` overlays the active profile for every command.

```bash
just show-profile   # print resolved verifier + RAG settings
just first-run      # setup → preprocess → predict for that profile
```

### Profile A — 16GB full pipeline (recommended)

**Machine:** RTX 5060 16GB, Kaggle P100/T4, or similar.

```toml
[runtime]
hardware_profile = "16gb"

[hardware_profiles.16gb.gemma]
fast_model_name = "google/gemma-4-E4B-it"
think_model_name = "google/gemma-4-E4B-it"
model_loader = "multimodal_lm"
load_in = "4bit"
device_map = "cuda:0"
cuda_max_memory = "14GiB"
exemplar_top_k = 3
max_input_tokens = 3072
enable_think_pass = true
fast_pass_batch_size = 16

[hardware_profiles.16gb.rag]
batch_size = 128
query_batch_size = 128
```

```bash
just first-run
```

Or step by step:

```bash
just setup             # sync + models for active profile + wiki + make-rag
just preprocess
just predict
```

After assets exist:

```bash
just run               # preprocess → predict
just predict           # prediction only (resumes checkpoints when valid)
just analyze           # evaluate latest prediction run against test ground truth
```

---

### Profile B — 8GB dual-model (Qwen + DeepSeek thinking verifier)

**Machine:** RTX 5060 mobile 8GB or any GPU too small for Gemma 4 E4B.

```toml
[runtime]
hardware_profile = "8gb"

[hardware_profiles.8gb.gemma]
fast_model_name = "Qwen/Qwen2.5-3B-Instruct"
think_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_loader = "causal_lm"
load_in = "4bit"
device_map = "cuda:0"
cuda_max_memory = "7GiB"
max_input_tokens = 1536
exemplar_top_k = 0
enable_think_pass = true
chat_template_enable_thinking_fast = false
chat_template_enable_thinking_think = true
fast_pass_batch_size = 8

[hardware_profiles.8gb.rag]
batch_size = 32
query_batch_size = 32
```

```bash
just first-run
```

This profile uses `Qwen2.5-3B-Instruct` for the fast pass and `DeepSeek-R1-Distill-Qwen-7B` for the thinking pass. It sequentially unloads the fast model from VRAM to make room before loading the thinking model. The fast F/H pass uses non-thinking chat-template mode so Qwen does not start with `<think>` when the code needs a single F/H token.

---

### OOM / stability

| Symptom          | Fix                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| Gemma / Qwen OOM | Use `8gb`, lower `cuda_max_memory` / `max_input_tokens` / `max_think_tokens`, or set `exemplar_top_k = 0` |
| RAG indexing OOM | Lower `batch_size` / `max_seq_length` in `[rag]` or profile RAG overrides                                 |
| Stale RAG scores | `just clean-rag` then `just predict`                                                                      |
| Missing indexes  | `rag_skipped_reason=index_missing:<source>` → fill `corpus/<source>/` then `just make-rag`                |

### Prediction resume

| File                                         | Stage                                   |
| -------------------------------------------- | --------------------------------------- |
| `generated/processed/test_with_evidence.csv` | Routed + typed-RAG-filled test contexts |
| `logs/debug_llm_verifier.jsonl`              | Row-level verifier cache                |
| `generated/processed/test_llm_preds.csv`     | Complete verifier probability vector    |

`[predict].use_checkpoints = true` resumes after OOM. `force_recompute = true` ignores checkpoints for one run.

### Daily workflow

| Goal                   | Command                                           |
| ---------------------- | ------------------------------------------------- |
| Predict only           | `just predict`                                    |
| Force clean prediction | set `force_recompute = true`, then `just predict` |
| Rebuild RAG indexes    | `just make-rag` / `just make-rag --source wiki`   |
| Drop RAG caches        | `just clean-rag`                                  |
| Full refresh           | `just clean-all` → `just setup` → `just run`      |

### Performance tuning

| Knob                                   | Where                         | Effect                                    |
| -------------------------------------- | ----------------------------- | ----------------------------------------- |
| `query_batch_size`                     | `[hardware_profiles.*.rag]`   | Faster RAG queries until embed OOM        |
| `index_dtype`                          | `[rag]`                       | Compact RAG indexes; default `float16`    |
| `load_in` / `device_map`               | `[hardware_profiles.*.gemma]` | Quantization and placement                |
| `max_input_tokens`                     | `[hardware_profiles.*.gemma]` | Memory vs truncation                      |
| `enable_think_pass`                    | `[hardware_profiles.*.gemma]` | Explicit think pass toggle                |
| `think_pass_batch_size`                | `[hardware_profiles.*.gemma]` | Batched think generate (1 on 8GB)         |
| `max_think_tokens_by_task`             | `[gemma]`                     | Per-task CoT token budgets                |
| `think_conf_low` / `think_conf_high`   | `[gemma]`                     | Near-threshold think band                 |
| `nli.enabled` / `nli.tasks` / `margin` | `[nli]`                       | NLI-first gate; skip think when confident |
| `exemplar_top_k`                       | `[hardware_profiles.*.gemma]` | Few-shot; `0` skips exemplar embedder     |
| `decision.threshold`                   | `[decision]`                  | Label cutoff on `p_llm` (default `0.5`)   |

---

## Verifier Prompt

The prompt is intentionally short and blunt. Fast pass asks for one next token only:

```text
Task: <task_type>
Rule: <task-specific rule>
<evidence>
...
</evidence>
Q: <prompt_bn>
A: <response_bn>
Return one token only: F = faithful/correct/label 1; H = hallucinated/wrong/label 0.
V:
```

The model does not generate a sentence for the fast pass. The code reads next-token logits for F vs H.

Think pass uses verdict-first output so truncation is less likely to lose the parseable answer:

```text
verdict: Faithful|Hallucinated
confidence: strong|likely|uncertain
reason: <one short English sentence>
```

Token budget is per-task via `[gemma.max_think_tokens_by_task]` (math short, grammar longer). Think is skipped when the NLI-first gate is confident on configured tasks.

---

## Typed RAG corpora

Four typed sources: `wiki`, `idioms`, `literal`, `grammar`. Empty folders are fine (`index_missing:<source>`).

```bash
just download-corpus                 # → generated/wiki/ (categorized JSONLs and titles)
just download-english-corpus         # → generated/wiki_en/ (English counterparts for places/people)
just sort-corpus data.jsonl          # LLM-sort rows into corpus/<source>/data.jsonl
just sort-corpus data.jsonl -- --dry-run --limit 20
uv run python scripts/sort_corpus.py --tui
just make-rag                        # all non-empty sources
just make-rag --source wiki
```

`sort-corpus` uses the active verifier model from `configs/config.toml`. It writes useful rows under `corpus/wiki`, `corpus/idioms`, `corpus/literal`, or `corpus/grammar`; skipped/noisy rows go under `generated/corpus_sort_skipped/`.

Layout, JSONL examples, writing guidance, and starter filenames: [`corpus/README.md`](corpus/README.md).

---

## Command reference

Run `just` to list recipes.

| Command                        | What it does                                       |
| ------------------------------ | -------------------------------------------------- |
| `just sync`                    | Install deps                                       |
| `just show-profile`            | Print active `hardware_profile` + resolved knobs   |
| `just download-models`         | BGE-M3                                             |
| `just download-models-gemma`   | BGE-M3 + verifier for active profile               |
| `just download-corpus`         | Wiki → `generated/wiki/` (downloads & categorizes) |
| `just download-english-corpus` | Fetch English counterparts to `generated/wiki_en/` |
| `just sort-corpus file.jsonl`  | LLM-sort JSONL rows into typed corpus folders      |
| `just make-rag`                | Build `indexes/<source>.pkl` from corpus folders   |
| `just setup`                   | sync + models + corpus + make-rag                  |
| `just preprocess`              | Clean → `generated/processed/`                     |
| `just predict`                 | Routed inference → `submissions/<timestamp>/`      |
| `just run`                     | preprocess → predict                               |
| `just analyze`                 | Evaluate predictions against test ground truth     |
| `just first-run`               | setup → preprocess → predict (uses profile)        |
| `just first-run-16gb` / `8gb`  | Aliases of `first-run` (set profile in config)     |
| `just clean-rag`               | Drop evidence CSVs + indexes                       |
| `just clean-processed`         | Drop `generated/processed/`                        |
| `just clean-logs`              | Drop verifier JSONL logs                           |
| `just clean-all`               | All cleans                                         |
| `just test` / `lint` / `check` | Dev helpers                                        |

---

## Outputs

### Prediction (`just predict`)

| Path                                           | Contents                                  |
| ---------------------------------------------- | ----------------------------------------- |
| `submissions/<timestamp>/submission.csv`       | `id, label` — **upload this only**        |
| `submissions/<timestamp>/submission_debug.csv` | Full trace for error analysis             |
| `submissions/latest`                           | Symlink → most recent timestamped run dir |
| `generated/processed/test_with_evidence.csv`   | Test after routing + typed RAG            |
| `generated/processed/test_llm_preds.csv`       | Resumable verifier probabilities          |
| `generated/processed/test_with_preds.csv`      | Test with `p_llm` + `task_type`           |
| `logs/debug_llm_verifier.jsonl`                | Per-row verifier debug at inference       |

Partial mid-run debug (if enabled) is written as `submission_partial_debug.csv` and removed when the final submission is complete. Never upload a partial/debug file.

### `submission_debug.csv` columns

Fixed schema for checking mistakes and tuning thresholds:

| Group      | Columns                                                                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decision   | `id`, `label`, `p_llm`, `threshold`, `threshold_margin`                                                                                                   |
| Routing    | `task_type`                                                                                                                                               |
| Verifier   | `p_fast`, `p_think`, `triggered_think`, `think_max_tokens`, `think_reasons`, `verdict_parsed`, `confidence_parsed`, `think_changed_label`, `thinking_cot` |
| NLI-first  | `nli_eligible`, `nli_applied`, `nli_skip_reason`, `nli_p_entail`, `nli_p_contradict`, `nli_p_neutral`, `nli_margin`, `p_nli`                              |
| Evidence   | `rag_used`, `rag_source`, `rag_skipped_reason`, `evidence_source`, `evidence_relevance`, `n_retrieved`, `retrieval_sim_max`, `retrieval_sim_mean`         |
| Text       | `context_original`, `context`, `prompt_bn`, `response_bn`                                                                                                 |
| Provenance | `run_timestamp`, `hardware_profile`, `gemma_model_name`, `gemma_load_in`                                                                                  |

Sort by `abs(threshold_margin)` for borderline rows; filter by `task_type` / `rag_source` / `think_changed_label` to find weak categories.

---

## Data files

```text
dataset/sample_dataset.json              # labeled train (exemplars for 16GB few-shot)
dataset/testset.csv                      # competition-like test (2516 rows) → submission
dataset/sample_submission.csv            # id,label format example
dataset/testset_audit_200.csv            # runnable 200-row dry run
dataset/analysis/testset_audit_200.csv   # same 200 rows + gold_label columns to fill
```

### Full test set (`dataset/testset.csv`)

2516 rows · 1155 `[NULL]` context · 1361 with context.

| `task_type`                | Count |
| -------------------------- | ----: |
| `context_grounded_fact`    |   885 |
| `general_fact_null`        |   575 |
| `context_grounded_other`   |   342 |
| `other_null`               |   254 |
| `bangla_grammar`           |   106 |
| `idiom_meaning_null`       |    75 |
| `literal_meaning_null`     |    75 |
| `famous_bn_fact_null`      |    59 |
| `translation_or_bilingual` |    59 |
| `math_speed_distance`      |    23 |
| `famous_bn_fact_context`   |    17 |
| `calendar_arithmetic`      |    13 |
| `math_profit_loss`         |    12 |
| `math_average`             |    11 |
| `math_work_rate`           |    10 |

This is a mixed benchmark: context entailment, null facts, idioms/literal meanings, grammar, arithmetic, famous BN facts, and translation. One blunt “RAG must support the answer” rule fails on idioms/lexicon rows.

### 200-row audit set

Stratified slice for labeling and threshold tuning:

| `task_type`                | Count |
| -------------------------- | ----: |
| `context_grounded_fact`    |    40 |
| `general_fact_null`        |    30 |
| `bangla_grammar`           |    18 |
| `context_grounded_other`   |    18 |
| `idiom_meaning_null`       |    15 |
| `literal_meaning_null`     |    15 |
| `famous_bn_fact_null`      |    14 |
| `other_null`               |    12 |
| `translation_or_bilingual` |     8 |
| `math_*` / `calendar_*`    |    25 |
| `famous_bn_fact_context`   |     5 |

Runnable inference file: `dataset/testset_audit_200.csv`  
Label file: `dataset/analysis/testset_audit_200.csv` — fill `gold_label` (`1` faithful / `0` hallucinated), `auditor_confidence`, `needs_human_review`, `audit_reason`.

**Labeling rules:**

1. Judge `response_bn` for `prompt_bn`; use original context when present.
2. Idiom (`ভাবার্থ`) / literal (`শাব্দিক অর্থ`): use language knowledge — do not mark wrong only because RAG is empty.
3. Math/calendar: calculate the answer.
4. Watch common swaps: Mujib ↔ Nazrul ↔ Tagore; Independence Day ↔ Victory Day; Searchlight ↔ Mujibnagar; total vs Bangladesh-only numbers; birth year vs later event year.
5. After labeling, score **by `task_type`**, not only global accuracy.

---

## File-by-file guide

### Root

| File                         | Role                                         |
| ---------------------------- | -------------------------------------------- |
| `README.md`                  | Architecture, hardware recipes, commands     |
| `configs/config.toml`        | **All configuration knobs with inline docs** |
| `justfile`                   | Command runner                               |
| `pyproject.toml` / `uv.lock` | Dependencies (uv)                            |
| `requirements.txt`           | Exported deps for non-uv environments        |

### `src/`

| File                 | Role                                                                  |
| -------------------- | --------------------------------------------------------------------- |
| `preprocess.py`      | Bengali text cleanup, `[NULL]` handling, `has_context`                |
| `router.py`          | Deterministic `task_type` classification                              |
| `evidence_policy.py` | Task→corpus map, prompts, think triggers, think score map             |
| `rag.py`             | Typed corpora, per-source BGE-M3 indexes, scored retrieval            |
| `llm_verifier.py`    | Gemma/Qwen verifier — fast pass, NLI-first gate, think fallback       |
| `nli.py`             | Multilingual NLI gate (confident → skip think)                        |
| `predict.py`         | Route → typed evidence → verifier → threshold → submission + debug    |
| `tui.py`             | Shared Rich banners / progress / tables                               |
| `config_utils.py`    | Hardware profile resolution, model path cache, runtime torch settings |

### `scripts/`

| File                 | Role                                                   |
| -------------------- | ------------------------------------------------------ |
| `download_models.py` | Hugging Face snapshots → `models/hf/`                  |
| `download_corpus.py` | Bengali Wikipedia chunks → `corpus/wiki/wiki_bn.jsonl` |

### Generated (gitignored)

| Path                   | Role                                            |
| ---------------------- | ----------------------------------------------- |
| `generated/processed/` | Intermediate CSVs                               |
| `corpus/<source>/`     | Typed RAG documents (`*.jsonl`)                 |
| `indexes/<source>.pkl` | Per-source dense indexes (+ optional exemplars) |
| `models/`              | HF cache / offload                              |
| `submissions/`         | Final + debug CSVs                              |
| `logs/`                | Verifier JSONL debug logs                       |

---

## Known weaknesses

1. **No labeled gold for the 200-row audit set yet** — fill `dataset/analysis/testset_audit_200.csv` before tuning thresholds.
2. **Typed lexical corpora are empty by default** — idiom / literal / grammar folders need curated `*.jsonl` before those indexes help.
3. **Math/calendar rows are LLM-only** — no separate symbolic solver; Gemma/Qwen judges via task prompts (RAG skipped).
4. **8GB uses a smaller verifier** — `Qwen/Qwen3-1.7B` instead of Gemma 4; fast F/H uses non-thinking chat mode and the explicit think pass uses thinking mode.
5. **Evidence cache stickiness** — after corpus/index/config RAG changes, run `just clean-rag` before `just predict`.
6. **Kaggle packaging not automated**.

---

## Roadmap

- [ ] Label the 200-row audit set and tune per-task thresholds if needed
- [ ] Fill idiom / literal / grammar corpus tables
- [ ] Kaggle Dataset bundle + offline submit notebook
