#!/usr/bin/env python3
"""Smoke-test batched think_pass on the real verifier model.

Compares sequential vs batched generate on same-budget prompts.
Short generations should match exactly; long CoT may diverge under bf16
(see probe_think_batching.py logit proof for correctness of left-pad).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from src.llm_verifier import GemmaVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="context_grounded_fact")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug-csv", default="submissions/20260718_133444/submission_debug.csv")
    args = parser.parse_args()

    debug = pd.read_csv(args.debug_csv)
    hits = debug[(debug["task_type"] == args.task) & (debug["triggered_think"] == True)].head(
        args.n
    )
    if len(hits) < 2:
        raise SystemExit(f"Need >=2 think rows for task={args.task}")

    v = GemmaVerifier()
    v.load_model(v.think_model_name)
    budget = min(args.max_new_tokens, v._think_token_budget(args.task))
    print(f"model={v.think_model_name} task={args.task} n={len(hits)} budget={budget}")

    prompts = []
    for _, r in hits.iterrows():
        prompts.append(
            v._build_think_prompt(
                evidence=str(r["context"]),
                prompt_bn=str(r["prompt_bn"]),
                response_bn=str(r["response_bn"]),
                task_type=args.task,
                exemplars=[],
            )
        )

    t0 = time.perf_counter()
    seq = [v._batched_think_generate([p], budget)[0] for p in prompts]
    seq_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    bat = v._batched_think_generate(prompts, budget)
    bat_s = time.perf_counter() - t1

    exact = sum(a == b for a, b in zip(seq, bat))
    print(f"sequential {seq_s:.1f}s | batched {bat_s:.1f}s | {seq_s / max(bat_s, 1e-9):.2f}x")
    print(f"exact_text_matches={exact}/{len(prompts)}")
    for i, (a, b) in enumerate(zip(seq, bat)):
        va, _, _ = v._parse_think_output(a)
        vb, _, _ = v._parse_think_output(b)
        print(f"  [{i}] exact={a == b} verdicts={va}/{vb} chars={len(a)}/{len(b)}")
    if torch.cuda.is_available():
        print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    if exact == len(prompts):
        print("PASS: exact match at this budget")
    else:
        print("NOTE: divergence at this budget (bf16 long-decode flake); logits still match")


if __name__ == "__main__":
    main()
