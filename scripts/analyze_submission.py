import sys
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def analyze_submission(sub_path):
    gt_path = "dataset/testset_audit_200_labeled.csv"

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

    # If debug file is available, show task type analysis
    if os.path.exists(debug_csv):
        print("\n" + "=" * 50)
        print("ERROR BREAKDOWN BY TASK TYPE")
        print("=" * 50)
        debug_df = pd.read_csv(debug_csv)
        merged_debug = pd.merge(merged, debug_df[["id", "task_type"]], on="id")

        incorrect = merged_debug[merged_debug["label_pred"] != merged_debug["label_true"]]
        counts = incorrect["task_type"].value_counts()
        total_by_type = merged_debug["task_type"].value_counts()

        error_rate = (counts / total_by_type).fillna(0) * 100
        breakdown_df = pd.DataFrame(
            {
                "Errors": counts.fillna(0).astype(int),
                "Total Rows": total_by_type,
                "Error Rate (%)": error_rate,
            }
        ).sort_values(by="Errors", ascending=False)
        print(breakdown_df)
    else:
        print("\nNote: Debug submission file not found, skipping task type breakdown.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submissions/latest"
    analyze_submission(path)
