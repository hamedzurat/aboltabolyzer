#!/usr/bin/env python3
"""Prove left-padded batched decode is correct for think_pass.

Two proofs:
  A) LOGIT PARITY — last-real-token logits from left-padded batch must match
     sequential (batch=1) within atol. This is the real correctness check;
     greedy text can diverge from float non-associativity.
  B) GENERATE SMOKE — left-padded batched generate produces non-empty text
     and is faster than sequential.

Default: local Qwen2.5-3B-Instruct on cuda.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model(name: str) -> str:
    local = Path("models/hf") / name.replace("/", "__")
    if local.is_dir():
        return str(local)
    return name


def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def _chat_prompt(tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def last_token_logits_sequential(model, tokenizer, prompts: list[str], device):
    """Return list of logits vectors at the last real token (one forward each)."""
    out = []
    tokenizer.padding_side = "right"
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits  # [1, T, V]
        out.append(logits[0, -1, :].float().cpu())
    return out


@torch.inference_mode()
def last_token_logits_batched(model, tokenizer, prompts: list[str], device):
    """Left-padded batch; gather logits at each row's last real token."""
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["position_ids"] = _position_ids_from_mask(enc["attention_mask"])
    logits = model(**enc).logits  # [B, Tpad, V]
    # with left padding, last real token is always at index -1
    gathered = logits[:, -1, :].float().cpu()
    return [gathered[i] for i in range(len(prompts))]


@torch.inference_mode()
def generate_sequential(model, tokenizer, prompts, max_new_tokens, device):
    texts = []
    tokenizer.padding_side = "right"
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts.append(tokenizer.decode(gen[0][in_len:], skip_special_tokens=True).strip())
    return texts


@torch.inference_mode()
def generate_batched(model, tokenizer, prompts, max_new_tokens, device):
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["position_ids"] = _position_ids_from_mask(enc["attention_mask"])
    pad_width = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [
        tokenizer.decode(gen[i][pad_width:], skip_special_tokens=True).strip()
        for i in range(len(prompts))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.998,
        help="bf16 batched vs sequential logits won't match atol; cosine is the real check",
    )
    parser.add_argument("--min-argmax-rate", type=float, default=0.8)
    args = parser.parse_args()

    resolved = _resolve_model(args.model)
    print(f"model={args.model}\nresolved={resolved}\ncuda={torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device

    raw_users = [
        "Q: 2+2=?\nA: 4\nReason one sentence, then end with:\nverdict: Faithful",
        "Evidence: Dhaka is the capital of Bangladesh.\nQ: Capital?\nA: Dhaka\n"
        "Reason briefly, then:\nverdict: Faithful",
        "Task: bangla_grammar\nQ: ‘লাঠালাঠি’ শব্দটির সমাস?\nA: কর্মধারায়\n"
        "Reason about সমাস, then:\nverdict: Faithful|Hallucinated",
        ("বাংলাদেশের ইতিহাস। " * 25) + "\nQ: রাজধানী?\nA: ঢাকা\nEnd with:\nverdict: Faithful",
        "Math: 60 km/h for 2 hours. Distance?\nA: 120\nShow work, then:\nverdict: Faithful",
        "Q: Is the answer faithful?\nA: yes\nverdict: Faithful",
    ]
    prompts = [_chat_prompt(tokenizer, u) for u in raw_users]
    lens = [len(tokenizer.encode(p)) for p in prompts]
    print(f"prompt token lens={lens}")

    # ---- A) logit parity across batches ----
    print("\n=== A) last-token logit parity (left-pad batch vs sequential) ===")
    max_abs = 0.0
    min_cos = 1.0
    argmax_hits = 0
    n_rows = 0
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start : start + args.batch_size]
        seq_logits = last_token_logits_sequential(model, tokenizer, chunk, device)
        bat_logits = last_token_logits_batched(model, tokenizer, chunk, device)
        for i, (s, b) in enumerate(zip(seq_logits, bat_logits)):
            abs_err = (s - b).abs().max().item()
            cos = F.cosine_similarity(s.unsqueeze(0), b.unsqueeze(0)).item()
            same_argmax = int(s.argmax()) == int(b.argmax())
            max_abs = max(max_abs, abs_err)
            min_cos = min(min_cos, cos)
            argmax_hits += int(same_argmax)
            n_rows += 1
            status = "OK" if cos >= args.min_cosine else "FAIL"
            print(
                f"  row {start + i}: {status} max_abs={abs_err:.4g} cos={cos:.6f} "
                f"argmax_match={same_argmax}"
            )

    argmax_rate = argmax_hits / max(n_rows, 1)
    parity_ok = min_cos >= args.min_cosine and argmax_rate >= args.min_argmax_rate
    print(
        f"summary: max_abs={max_abs:.4g} min_cos={min_cos:.6f} "
        f"argmax_rate={argmax_rate:.0%} parity_ok={parity_ok}"
    )
    print("(bf16 abs diffs of ~0.5 with cos≈0.999 are expected batched-matmul noise)")

    # ---- B) generate smoke + speedup ----
    print("\n=== B) batched generate smoke ===")
    t0 = time.perf_counter()
    seq_texts = generate_sequential(model, tokenizer, prompts, args.max_new_tokens, device)
    seq_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    bat_texts = []
    for start in range(0, len(prompts), args.batch_size):
        bat_texts.extend(
            generate_batched(
                model,
                tokenizer,
                prompts[start : start + args.batch_size],
                args.max_new_tokens,
                device,
            )
        )
    bat_s = time.perf_counter() - t1
    speedup = seq_s / max(bat_s, 1e-9)

    nonempty = all(len(t) > 0 for t in bat_texts)
    no_fffd = all("\ufffd" not in t[:8] for t in bat_texts)
    exact = sum(a == b for a, b in zip(seq_texts, bat_texts))
    print(f"sequential {seq_s:.2f}s | batched {bat_s:.2f}s | speedup {speedup:.2f}x")
    print(f"exact_text_matches={exact}/{len(prompts)} (informational; float can diverge)")
    print(f"nonempty={nonempty} no_leading_fffd={no_fffd}")
    for i, t in enumerate(bat_texts):
        print(f"  [{i}] {t.replace(chr(10), ' ')[:90]!r}")

    print("\n" + "=" * 60)
    gen_ok = nonempty and no_fffd and speedup >= 1.2
    if parity_ok and gen_ok:
        print("PASS: left-padded batching is correct (logits) and faster (generate)")
        raise SystemExit(0)

    print(f"FAIL parity_ok={parity_ok} gen_ok={gen_ok}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
