from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, probs=None):
    """Calculates and displays all key evaluation metrics using a rich console layout."""
    console = Console()

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, recall, f1, support per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )

    # 1. Performance Summary Panel
    summary_text = (
        f"[bold cyan]Macro F1 Score:[/]   [bold green]{macro_f1:.4f}[/]\n"
        f"[bold cyan]Accuracy:[/]         [bold green]{accuracy:.4f}[/]\n\n"
        f"[bold yellow]Hallucinated F1 (Class 0):[/] [bold red]{f1[0]:.4f}[/]\n"
        f"[bold yellow]Faithful F1 (Class 1):[/]     [bold green]{f1[1]:.4f}[/]"
    )

    console.print(
        Panel(
            summary_text,
            title="[bold magenta]Performance Summary[/]",
            border_style="magenta",
            expand=False,
        )
    )

    # 2. Detailed Classification Report Table
    table = Table(title="Detailed Classification Report", border_style="cyan", show_header=True)
    table.add_column("Class", style="bold white")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right", style="bold green")
    table.add_column("Support", justify="right")

    table.add_row(
        "Hallucinated (0)",
        f"{precision[0]:.4f}",
        f"{recall[0]:.4f}",
        f"{f1[0]:.4f}",
        f"{support[0]}",
    )
    table.add_row(
        "Faithful (1)", f"{precision[1]:.4f}", f"{recall[1]:.4f}", f"{f1[1]:.4f}", f"{support[1]}"
    )
    table.add_section()
    table.add_row(
        "Macro Average",
        f"{precision.mean():.4f}",
        f"{recall.mean():.4f}",
        f"{macro_f1:.4f}",
        f"{support.sum()}",
    )

    # 3. Confusion Matrix Table
    cm = confusion_matrix(y_true, y_pred)
    cm_table = Table(title="Confusion Matrix", border_style="yellow", show_header=True)
    cm_table.add_column("Actual / Predicted", style="bold white")
    cm_table.add_column("Predicted Hallucinated (0)", justify="center", style="bold red")
    cm_table.add_column("Predicted Faithful (1)", justify="center", style="bold green")

    cm_table.add_row(
        "Actual Hallucinated (0)",
        f"{cm[0, 0]} [dim](True Neg)[/dim]",
        f"{cm[0, 1]} [dim](False Pos)[/dim]",
    )
    cm_table.add_row(
        "Actual Faithful (1)",
        f"{cm[1, 0]} [dim](False Neg)[/dim]",
        f"{cm[1, 1]} [dim](True Pos)[/dim]",
    )

    console.print(table)
    console.print(cm_table)

    metrics = {
        "macro_f1": macro_f1,
        "f1_class_0": f1[0],
        "f1_class_1": f1[1],
        "confusion_matrix": cm.tolist(),
    }
    return metrics
