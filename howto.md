# How to Use

Operator cheat sheet. Architecture / NLI policy / diagram: [`README.md`](README.md). Per-row examples: [`how-it-works.md`](how-it-works.md).

## Machines

| GPU                           | `hardware_profile` | Verifier                                            |
| ----------------------------- | ------------------ | --------------------------------------------------- |
| RTX 5060 Ti 16GB (submission) | `16gb`             | Gemma 4 E4B                                         |
| RTX 5060 mobile 8GB (debug)   | `8gb`              | Qwen2.5-3B fast + DeepSeek-R1-Distill-Qwen-7B think |

```toml
[runtime]
hardware_profile = "16gb"   # or "8gb"
```

```bash
just show-profile   # confirm resolved model / VRAM / RAG batches
```

Routing, RAG, and NLI knobs are shared. Only the verifier model changes with the profile. Do not treat 8GB audit accuracy as final.

## First Run

```bash
just first-run
```

That is: sync → models for this profile → wiki corpus → RAG indexes → preprocess → predict.

## Audit vs full test

```toml
[data]
# Dry run (200 rows):
test_path = "dataset/testset_audit_200.csv"
# Competition submission:
# test_path = "dataset/testset.csv"
```

Then `just run` or `just predict`. Evaluate with `just analyze` (uses `dataset/sample_submission.csv` as labels when present).

## Main Corpus Loop

After editing `corpus/<source>/*.jsonl` (wiki / idioms / literal / grammar):

```bash
just clean-rag
just make-rag                    # or: just make-rag --source idioms
just predict
```

Empty idiom/literal indexes fall back to wiki at predict time; still prefer filling the typed corpora.

## Dataset Changed

After editing files in `dataset/`:

```bash
just preprocess
just predict
```

If corpus changed too:

```bash
just preprocess
just clean-rag
just make-rag
just predict
```

## Force Fresh Prediction

Needed after NLI / router / verifier policy changes (cache fingerprint may not cover every knob), or when you distrust checkpoints:

```toml
[predict]
force_recompute = true
```

```bash
just predict
```

Set `force_recompute` back to `false` afterward.

## Evaluate Accuracy

```bash
just analyze
just analyze submissions/20260718_222014
```

## Outputs

Upload **only**:

```text
submissions/latest/submission.csv
```

Inspect:

```text
submissions/latest/submission_debug.csv
```

Key debug columns:

```text
task_type,
rag_used, rag_source, rag_skipped_reason, retrieval_sim_max,
p_fast, nli_applied, nli_skip_reason, p_nli,
triggered_think, think_reasons, p_think, p_llm, label
```

NLI skip reasons worth filtering: `faithful_low_overlap`, `entail_le_neutral`, `faithful_fast_disagrees`, `margin_too_low`, `weak_rag_premise`.

## Config knobs you will touch most

| Knob                      | File section | Typical use                                                               |
| ------------------------- | ------------ | ------------------------------------------------------------------------- |
| `hardware_profile`        | `[runtime]`  | Switch 8gb ↔ 16gb                                                         |
| `test_path`               | `[data]`     | Audit vs full test                                                        |
| `force_recompute`         | `[predict]`  | Clean predict                                                             |
| `routing_mode`            | `[router]`   | `hybrid` (default)                                                        |
| NLI margins / guards      | `[nli]`      | See README design note — keep `block_faithful_on_fast_h = true` for Gemma |
| `think_conf_low` / `high` | `[gemma]`    | Near-threshold think band                                                 |
| `similarity_threshold`    | `[rag]`      | Retrieval cutoff (default `0.55`)                                         |
