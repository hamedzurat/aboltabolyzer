# How to Use

## First Run

Set profile in `configs/config.toml`:

```toml
[runtime]
hardware_profile = "16gb" # or "8gb"
```

Run:

```bash
just first-run
```

## Main Corpus Loop

After editing `corpus/<source>/*.jsonl`:

```bash
just clean-rag
just make-rag
just predict
```

This is the main loop right now.

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

In `configs/config.toml`:

```toml
[predict]
force_recompute = true
```

Then:

```bash
just predict
```

Set it back to `false` afterward.

## Outputs

Upload:

```text
submissions/latest/submission.csv
```

Inspect:

```text
submissions/latest/submission_debug.csv
```

Key debug columns:

```text
task_type, rag_used, rag_source, rag_skipped_reason,
n_retrieved, retrieval_sim_max, p_fast, triggered_think,
think_reasons, p_llm, label
```
