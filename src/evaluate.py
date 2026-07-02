from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, probs=None):
    """Calculates all key metrics for evaluation."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    # Label=0 is the hallucinated class in our target metric
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)

    report = classification_report(
        y_true, y_pred, target_names=["Hallucinated (0)", "Faithful (1)"]
    )
    cm = confusion_matrix(y_true, y_pred)

    print("=== Evaluation Report ===")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Hallucinated F1 (Class 0): {f1_class_0:.4f}")
    print(f"Faithful F1 (Class 1): {f1_class_1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics = {
        "macro_f1": macro_f1,
        "f1_class_0": f1_class_0,
        "f1_class_1": f1_class_1,
        "confusion_matrix": cm.tolist(),
    }
    return metrics
