"""Shared Rich terminal UI helpers for pipeline commands."""

from __future__ import annotations

from collections.abc import Mapping

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table

console = Console()


def banner(title: str, subtitle: str | None = None, style: str = "bold yellow"):
    body = f"[{style}]{title}[/{style}]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(body, border_style="yellow", expand=False))


def step(n: int, total: int, title: str):
    console.print()
    console.print(Rule(f"[bold cyan]Step {n}/{total} · {title}[/bold cyan]", style="cyan"))


def ok(message: str):
    console.print(f"[green]✔ {message}[/green]")


def warn(message: str):
    console.print(f"[yellow]⚠ {message}[/yellow]")


def info(message: str):
    console.print(f"[cyan]→ {message}[/cyan]")


def kv_table(title: str, rows: Mapping[str, object], *, key_header: str = "Key"):
    table = Table(title=title, show_header=True, header_style="bold", expand=False)
    table.add_column(key_header, style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in rows.items():
        table.add_row(str(key), str(value))
    console.print(table)


def count_table(
    title: str, counts: Mapping[str, int], *, key_header: str = "Item", limit: int = 20
):
    table = Table(title=title, show_header=True, header_style="bold", expand=False)
    table.add_column(key_header, style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("%", justify="right", style="dim")
    total = sum(int(v) for v in counts.values()) or 1
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    for key, value in items[:limit]:
        table.add_row(str(key), str(int(value)), f"{100.0 * int(value) / total:.1f}")
    if len(items) > limit:
        table.add_row("…", str(sum(int(v) for _, v in items[limit:])), "")
    console.print(table)


def pipeline_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def done_panel(title: str, lines: list[str]):
    body = f"[bold green]✔ {title}[/bold green]\n" + "\n".join(lines)
    console.print(Panel(body, border_style="green", expand=False))
