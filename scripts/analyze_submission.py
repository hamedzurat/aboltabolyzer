import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def analyze_submission(sub_path):
    gt_path = "dataset/sample_submission.csv"

    if os.path.isdir(sub_path):
        sub_csv = os.path.join(sub_path, "submission.csv")
        debug_csv = os.path.join(sub_path, "submission_debug.csv")
    elif os.path.exists(sub_path):
        if sub_path.endswith("submission_debug.csv"):
            debug_csv = sub_path
            sub_csv = sub_path.replace("submission_debug.csv", "submission.csv")
        else:
            sub_csv = sub_path
            debug_csv = sub_path.replace("submission.csv", "submission_debug.csv")
    else:
        print(f"Error: Submission path {sub_path} does not exist.")
        return

    if not os.path.exists(sub_csv):
        print(f"Error: {sub_csv} not found.")
        return

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth {gt_path} not found. Run generate_ground_truth.py first.")
        return

    print(f"Loading submission: {sub_csv}")
    sub_df = pd.read_csv(sub_csv)

    print(f"Loading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)

    merged = pd.merge(sub_df, gt_df, on="id", suffixes=("_pred", "_true"))

    if len(merged) < len(sub_df):
        print(f"Warning: Only {len(merged)} / {len(sub_df)} rows matched in the ground truth!")

    y_true = merged["label_true"]
    y_pred = merged["label_pred"]

    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"OVERALL ACCURACY: {accuracy:.1%}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Hallucinated (0)", "Faithful (1)"]))

    # If debug file is available, show extended analysis
    if not os.path.exists(debug_csv):
        print("\nNote: Debug submission file not found, skipping extended analysis.")
        return

    debug_df = pd.read_csv(debug_csv)
    merged_debug = pd.merge(merged, debug_df, on="id")
    merged_debug["correct"] = merged_debug["label_pred"] == merged_debug["label_true"]
    merged_debug["wrong"] = ~merged_debug["correct"]
    wrong = merged_debug[merged_debug["wrong"]]
    right = merged_debug[merged_debug["correct"]]

    # ── Task-type error breakdown ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ERROR BREAKDOWN BY TASK TYPE")
    print("=" * 50)
    incorrect = merged_debug[merged_debug["label_pred"] != merged_debug["label_true"]]
    counts = incorrect["task_type"].value_counts()
    total_by_type = merged_debug["task_type"].value_counts()
    error_rate = (counts / total_by_type).fillna(0) * 100
    breakdown_df = pd.DataFrame(
        {
            "Errors": counts.fillna(0).astype(int),
            "Total Rows": total_by_type,
            "Error Rate (%)": error_rate.round(1),
        }
    ).sort_values(by="Errors", ascending=False)
    print(breakdown_df)

    # ── Thinking / COT analysis ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("THINKING (COT) ANALYSIS")
    print("=" * 50)

    triggered = merged_debug["triggered_think"].fillna(False).astype(bool)
    triggered_sum = int(triggered.sum())
    has_cot = merged_debug["thinking_cot"].notna() & (
        merged_debug["thinking_cot"].fillna("").str.strip() != ""
    )
    triggered_with_cot = int((triggered & has_cot).sum())
    triggered_no_cot = triggered_sum - triggered_with_cot

    print(f"triggered_think=True:  {triggered_sum}")
    print(f"  with COT text:       {triggered_with_cot}")
    print(
        f"  WITHOUT COT text:    {triggered_no_cot}  {'⚠  think triggered but no output!' if triggered_no_cot else ''}"
    )

    # Classify think outcome
    md = merged_debug.copy()
    md["cot_len"] = md["thinking_cot"].fillna("").str.len()
    md["think_status"] = "no_think"
    md.loc[
        merged_debug["triggered_think"].fillna(False).astype(bool)
        & ~merged_debug["think_changed_label"].fillna(False).astype(bool),
        "think_status",
    ] = "think_no_change"
    md.loc[
        merged_debug["triggered_think"].fillna(False).astype(bool)
        & merged_debug["think_changed_label"].fillna(False).astype(bool)
        & md["correct"],
        "think_status",
    ] = "think_helped"
    md.loc[
        merged_debug["triggered_think"].fillna(False).astype(bool)
        & merged_debug["think_changed_label"].fillna(False).astype(bool)
        & ~md["correct"],
        "think_status",
    ] = "think_hurt"

    print("\n-- Think outcome summary --")
    ts = (
        md.groupby("think_status")
        .agg(count=("correct", "count"), accuracy=("correct", "mean"))
        .round(3)
    )
    ts["accuracy_%"] = (ts["accuracy"] * 100).round(1)
    order = ["no_think", "think_no_change", "think_helped", "think_hurt"]
    ts = ts.reindex([o for o in order if o in ts.index])
    print(ts[["count", "accuracy_%"]])

    print("\n-- COT length (chars): wrong vs correct think-triggered rows --")
    wrong_cot = md[triggered & ~md["correct"]]["cot_len"]
    right_cot = md[triggered & md["correct"]]["cot_len"]
    if len(wrong_cot) and len(right_cot):
        print(
            f"  wrong:  n={int(wrong_cot.count())}  mean={wrong_cot.mean():.0f}  max={wrong_cot.max()}"
        )
        print(
            f"  right:  n={int(right_cot.count())}  mean={right_cot.mean():.0f}  max={right_cot.max()}"
        )

    # think_max_tokens distribution in wrong cases
    if "think_max_tokens" in wrong.columns:
        tok_dist = wrong["think_max_tokens"].value_counts().to_dict()
        if tok_dist:
            print(f"\n-- think_max_tokens dist (wrong think-triggered): {tok_dist}")
            max_tok = wrong["think_max_tokens"].max()
            if not pd.isna(max_tok) and int(max_tok) <= 512:
                print(
                    "  ⚠  All wrong think cases hit <=512 token limit "
                    "— COT is truncated mid-reasoning!"
                )

    # think_reasons breakdown
    if "think_reasons" in wrong.columns:
        print("\n-- think_reasons (wrong cases) --")
        reasons_wrong = (
            wrong["think_reasons"]
            .fillna("")
            .astype(str)
            .str.split("|")
            .explode()
            .str.strip()
            .value_counts()
        )
        non_empty = reasons_wrong[reasons_wrong.index != ""]
        if len(non_empty):
            print(non_empty.head(10))

    # Cases where think HURT (flipped correct→wrong)
    hurt = md[md["think_status"] == "think_hurt"]
    if len(hurt) > 0:
        print(f"\n-- Think HURT ({len(hurt)} cases: think flipped correct→wrong) --")
        print(hurt[["id", "task_type", "p_fast", "p_think"]].to_string(index=False))

    # ── RAG analysis ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RAG EFFECTIVENESS ANALYSIS")
    print("=" * 50)

    if "rag_used" in merged_debug.columns:
        rag_yes = merged_debug[merged_debug["rag_used"].fillna(False).astype(bool)]
        rag_no = merged_debug[~merged_debug["rag_used"].fillna(False).astype(bool)]
        rag_acc = 100 * rag_yes["correct"].mean() if len(rag_yes) else float("nan")
        norag_acc = 100 * rag_no["correct"].mean() if len(rag_no) else float("nan")
        delta = rag_acc - norag_acc
        sign = "+" if delta >= 0 else ""
        print(f"RAG used:     {len(rag_yes)} rows · accuracy {rag_acc:.1f}%")
        print(f"RAG NOT used: {len(rag_no)} rows · accuracy {norag_acc:.1f}%")
        print(
            f"RAG delta:    {sign}{delta:.1f}pp  "
            f"{'✓ helpful' if delta >= 0 else '✗ HURTING accuracy'}"
        )

        print("\n-- RAG accuracy by task type --")
        for task in sorted(merged_debug["task_type"].unique()):
            sub = merged_debug[merged_debug["task_type"] == task]
            rag_s = sub[sub["rag_used"].fillna(False).astype(bool)]
            norag_s = sub[~sub["rag_used"].fillna(False).astype(bool)]
            rag_str = (
                f"{100 * rag_s['correct'].mean():.0f}% (n={len(rag_s)})" if len(rag_s) else "n/a"
            )
            norag_str = (
                f"{100 * norag_s['correct'].mean():.0f}% (n={len(norag_s)})"
                if len(norag_s)
                else "n/a"
            )
            overall_str = f"{100 * sub['correct'].mean():.0f}%"
            print(f"  {task:<35} overall={overall_str:<7} rag={rag_str:<18} no_rag={norag_str}")

        if "retrieval_sim_max" in wrong.columns:
            rag_wrong = wrong[wrong["rag_used"].fillna(False).astype(bool)]
            rag_right = right[right["rag_used"].fillna(False).astype(bool)]
            print(
                f"\n-- Retrieval sim_max (RAG rows): "
                f"wrong_mean={rag_wrong['retrieval_sim_max'].mean():.3f}  "
                f"right_mean={rag_right['retrieval_sim_max'].mean():.3f}"
            )

    # ── Overconfidence analysis ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("OVERCONFIDENCE ANALYSIS")
    print("=" * 50)

    if "p_fast" in merged_debug.columns:
        overconf_wrong = wrong[
            ((wrong["p_fast"] > 0.8) | (wrong["p_fast"] < 0.2))
            & ~wrong["triggered_think"].fillna(False).astype(bool)
        ]
        print(f"Overconfident wrong (|p_fast|>0.8, no think): {len(overconf_wrong)}")
        if len(overconf_wrong) > 0:
            print("  Task breakdown:", overconf_wrong["task_type"].value_counts().to_dict())

        print(f"\n  Wrong  p_llm: mean={wrong['p_llm'].mean():.3f}  std={wrong['p_llm'].std():.3f}")
        print(f"  Right  p_llm: mean={right['p_llm'].mean():.3f}  std={right['p_llm'].std():.3f}")

    # ── Sample wrong predictions per worst task ────────────────────────────────
    print("\n" + "=" * 50)
    print("SAMPLE WRONG PREDICTIONS (worst tasks)")
    print("=" * 50)

    eligible = breakdown_df[breakdown_df["Total Rows"] >= 5]
    top_bad_tasks = eligible.head(3).index.tolist()

    for task in top_bad_tasks:
        task_wrong = wrong[wrong["task_type"] == task]
        task_all = merged_debug[merged_debug["task_type"] == task]
        print(
            f"\n  [{task}]  {len(task_wrong)}/{len(task_all)} wrong "
            f"({100 * len(task_wrong) / len(task_all):.0f}% error rate)"
        )
        for _, row in task_wrong.head(3).iterrows():
            think_flag = "THINK" if row.get("triggered_think") else "FAST"
            rag_flag = "+RAG" if row.get("rag_used") else "    "
            cot_text = str(row.get("thinking_cot", "") or "")
            cot_snippet = ""
            if cot_text.strip():
                cot_snippet = "\n      COT: " + cot_text[:200].replace("\n", " ")
            pfas = row.get("p_fast", row.get("p_llm", float("nan")))
            try:
                pfas_str = f"{float(pfas):.3f}"
            except (TypeError, ValueError):
                pfas_str = str(pfas)
            print(
                f"    [{think_flag}][{rag_flag}] ID {row['id']}: "
                f"pred={row['label_pred']} truth={row['label_true']} "
                f"p_fast={pfas_str}"
            )
            print(f"      Q: {str(row.get('prompt_bn', ''))[:120]}")
            print(f"      A: {str(row.get('response_bn', ''))[:100]}")
            if cot_snippet:
                print(cot_snippet)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submissions/latest"
    analyze_submission(path)
