"""Template selection score visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_score_breakdown(
    df: pd.DataFrame,
    *,
    score_cols: list[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Stacked horizontal bar chart of score components.

    Parameters
    ----------
    df:
        DataFrame returned by ``TemplateScorer.to_dataframe()``.
    score_cols:
        Columns to stack.  Defaults to ``["hbond_score", "stability_score"]``
        plus any ``bonus_*`` columns.
    """
    if score_cols is None:
        bonus_cols = [c for c in df.columns if c.startswith("bonus_")]
        score_cols = ["hbond_score", "stability_score"] + bonus_cols

    plot_df = df.set_index("sample")[score_cols].sort_values(
        "hbond_score", ascending=True
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.45)))

    colors = ["#4878CF", "#6ACC65"] + sns.color_palette("Set2", len(score_cols) - 2)
    plot_df.plot(kind="barh", stacked=True, ax=ax, color=colors[: len(score_cols)],
                 edgecolor="white", width=0.65)

    totals = df.sort_values("hbond_score", ascending=True)["total_score"]
    for i, val in enumerate(totals):
        ax.text(val + 0.5, i, f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Template Selection Score Breakdown", fontsize=13, fontweight="bold")
    ax.axvline(50, ls="--", color="gray", lw=1, alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax


def plot_hbond_quality_map(
    df: pd.DataFrame,
    *,
    group_col: str | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Bubble plot: occupancy vs distance, bubble ∝ total_score.

    Parameters
    ----------
    df:
        DataFrame from ``TemplateScorer.to_dataframe()``.
    group_col:
        Optional column used for colour grouping.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    palette = sns.color_palette("tab10", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row["occupancy_%"],
            row["mean_dist"],
            s=row["total_score"] * 8,
            color=palette[i],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            str(row["sample"]),
            (row["occupancy_%"], row["mean_dist"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
        )

    ax.set_xlabel("H-bond Occupancy (%)", fontsize=12)
    ax.set_ylabel("Mean D–A Distance (Å)", fontsize=12)
    ax.set_title(
        "H-bond Quality Map\n(bubble size ∝ total score)", fontsize=13, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax
