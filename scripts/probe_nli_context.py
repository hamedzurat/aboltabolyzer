#!/usr/bin/env python3
"""Probe multilingual NLI as a drop-in for context_grounded_fact.

Loads context_grounded_fact rows from a submission_debug.csv (or processed test),
scores premise=context / hypothesis=answer (and a few prompt variants), and
reports accuracy vs ground truth + agreement with the current verifier.

Does NOT wire NLI into the pipeline — evaluation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def _load_rows(debug_csv: Path, gt_csv: Path) -> pd.DataFrame:
    debug = pd.read_csv(debug_csv)
    gt = pd.read_csv(gt_csv).rename(columns={"label": "y_true"})
    m = debug.merge(gt, on="id")
    cg = m[m["task_type"] == "context_grounded_fact"].copy()
    if cg.empty:
        raise SystemExit("No context_grounded_fact rows found in debug CSV")
    return cg


def _hypothesis_variants(row) -> dict[str, str]:
    q = str(row["prompt_bn"])
    a = str(row["response_bn"])
    return {
        "answer_only": a,
        "qa_en": f"The answer to the question '{q}' is: {a}",
        "qa_bn": f"প্রশ্নের উত্তর: {a}",
        "claim_bn": f"প্রশ্ন: {q}\nউত্তর: {a}",
    }


@torch.inference_mode()
def score_nli(model, tokenizer, premises: list[str], hypotheses: list[str], batch_size: int = 8):
    """Return entailment / contradiction / neutral probs per pair."""
    device = next(model.parameters()).device
    id2label = {
        int(k) if not isinstance(k, int) else k: v for k, v in model.config.id2label.items()
    }
    # normalize label names
    label_to_idx = {}
    for idx, name in id2label.items():
        key = str(name).lower()
        if "entail" in key:
            label_to_idx["entailment"] = idx
        elif "contradict" in key:
            label_to_idx["contradiction"] = idx
        elif "neutral" in key:
            label_to_idx["neutral"] = idx

    rows = []
    for start in range(0, len(premises), batch_size):
        p = premises[start : start + batch_size]
        h = hypotheses[start : start + batch_size]
        enc = tokenizer(
            p,
            h,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        for i in range(len(p)):
            rows.append(
                {
                    "p_entail": float(probs[i, label_to_idx["entailment"]]),
                    "p_contradict": float(probs[i, label_to_idx["contradiction"]]),
                    "p_neutral": float(probs[i, label_to_idx["neutral"]]),
                }
            )
    return rows


def evaluate_variant(cg: pd.DataFrame, scores: list[dict], name: str) -> dict:
    df = cg.copy()
    df["p_entail"] = [s["p_entail"] for s in scores]
    df["p_contradict"] = [s["p_contradict"] for s in scores]
    df["p_neutral"] = [s["p_neutral"] for s in scores]

    # Decision rules to try
    rules = {}

    # A: entail > contradict → Faithful
    pred = (df["p_entail"] > df["p_contradict"]).astype(int)
    rules["entail_gt_contradict"] = (pred == df["y_true"]).mean()

    # B: entail > 0.5 → Faithful
    pred = (df["p_entail"] > 0.5).astype(int)
    rules["entail_gt_0.5"] = (pred == df["y_true"]).mean()

    # C: contradict > 0.5 → Hallucinated else Faithful
    pred = (df["p_contradict"] <= 0.5).astype(int)
    rules["not_contradict_gt_0.5"] = (pred == df["y_true"]).mean()

    # D: argmax label (neutral counts as Faithful — soft)
    arg = df[["p_entail", "p_contradict", "p_neutral"]].idxmax(axis=1)
    pred = arg.map({"p_entail": 1, "p_neutral": 1, "p_contradict": 0})
    rules["argmax_neutral_as_faithful"] = (pred == df["y_true"]).mean()

    # E: argmax (neutral → Hallucinated — strict unsupported)
    pred = arg.map({"p_entail": 1, "p_neutral": 0, "p_contradict": 0})
    rules["argmax_neutral_as_hallucinated"] = (pred == df["y_true"]).mean()

    # F: only trust when max(entail,contradict) clear; else abstain (use verifier label)
    margin = (df["p_entail"] - df["p_contradict"]).abs()
    nli_pred = (df["p_entail"] > df["p_contradict"]).astype(int)
    hybrid = df["label"].copy()  # current verifier
    hybrid = hybrid.where(margin < 0.2, nli_pred)  # override when NLI is confident
    # wait: where(cond, other) keeps self when cond True. We want override when margin >= 0.2
    hybrid = df["label"].where(margin < 0.2, nli_pred)
    rules["hybrid_margin_0.2"] = (hybrid == df["y_true"]).mean()

    hybrid2 = df["label"].where(margin < 0.35, nli_pred)
    rules["hybrid_margin_0.35"] = (hybrid2 == df["y_true"]).mean()

    best_rule = max(rules, key=rules.get)
    return {
        "variant": name,
        "rules": rules,
        "best_rule": best_rule,
        "best_acc": rules[best_rule],
        "df": df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-csv",
        default="submissions/20260718_133444/submission_debug.csv",
    )
    parser.add_argument("--gt-csv", default="dataset/sample_submission.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cg = _load_rows(Path(args.debug_csv), Path(args.gt_csv))
    verifier_acc = (cg["label"] == cg["y_true"]).mean()
    p_fast_acc = ((cg["p_fast"] >= 0.5).astype(int) == cg["y_true"]).mean()
    print(f"context_grounded_fact rows: {len(cg)}")
    print(f"verifier acc: {verifier_acc:.1%}")
    print(f"p_fast acc:   {p_fast_acc:.1%}")
    print(f"NLI model:    {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"labels: {model.config.id2label}")
    print(f"device: {device}")

    premises = cg["context_original"].fillna(cg["context"]).astype(str).tolist()
    results = []
    for vname in ["answer_only", "qa_en", "qa_bn", "claim_bn"]:
        hyps = [_hypothesis_variants(r)[vname] for _, r in cg.iterrows()]
        scores = score_nli(model, tokenizer, premises, hyps, batch_size=args.batch_size)
        ev = evaluate_variant(cg, scores, vname)
        results.append(ev)
        print(f"\n=== variant: {vname} ===")
        for rule, acc in sorted(ev["rules"].items(), key=lambda x: -x[1]):
            marker = " <-- best" if rule == ev["best_rule"] else ""
            print(f"  {rule:35s} {acc:.1%}{marker}")

    best = max(results, key=lambda r: r["best_acc"])
    print("\n" + "=" * 60)
    print(
        f"BEST NLI: variant={best['variant']} rule={best['best_rule']} acc={best['best_acc']:.1%}"
    )
    print(f"Verifier baseline: {verifier_acc:.1%}")
    delta = best["best_acc"] - verifier_acc
    if best["best_acc"] > verifier_acc + 0.01:
        print(f"VERDICT: NLI looks promising (+{delta:.1%} vs verifier) — worth wiring")
    elif abs(delta) <= 0.01:
        print("VERDICT: NLI ≈ verifier — not worth the extra dependency yet")
    else:
        print(f"VERDICT: NLI worse ({delta:.1%}) — keep think-pass for context_grounded_fact")

    # Show errors where best NLI disagrees with truth
    df = best["df"]
    rule = best["best_rule"]
    if rule == "entail_gt_contradict":
        pred = (df["p_entail"] > df["p_contradict"]).astype(int)
    elif rule == "entail_gt_0.5":
        pred = (df["p_entail"] > 0.5).astype(int)
    elif rule == "not_contradict_gt_0.5":
        pred = (df["p_contradict"] <= 0.5).astype(int)
    elif rule == "argmax_neutral_as_faithful":
        arg = df[["p_entail", "p_contradict", "p_neutral"]].idxmax(axis=1)
        pred = arg.map({"p_entail": 1, "p_neutral": 1, "p_contradict": 0})
    else:
        arg = df[["p_entail", "p_contradict", "p_neutral"]].idxmax(axis=1)
        pred = arg.map({"p_entail": 1, "p_neutral": 0, "p_contradict": 0})

    err = df[pred != df["y_true"]]
    print(f"\nNLI errors ({len(err)}):")
    for _, r in err.head(8).iterrows():
        print(
            f"  id={r['id']} true={r['y_true']} nli_pred={int(pred.loc[r.name])} "
            f"ver={r['label']} e={r['p_entail']:.2f} c={r['p_contradict']:.2f} "
            f"n={r['p_neutral']:.2f}"
        )
        print(f"    Q: {str(r['prompt_bn'])[:100]}")
        print(f"    A: {str(r['response_bn'])[:100]}")


if __name__ == "__main__":
    main()
