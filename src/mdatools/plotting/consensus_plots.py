"""Plotting functions for consensus ranking and Pareto optimisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..scoring.consensus import ConsensusResult


def plot_consensus_ranking(
    result: ConsensusResult,
    top_n: int = 20,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Horizontal bar chart of the top-N ranked samples.

    Parameters
    ----------
    result:
        :class:`ConsensusResult` from :class:`ConsensusRanker`.
    top_n:
        Number of top-ranked samples to display.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    df = result.ranking.sort_values("rank").head(top_n)
    names = df.index.tolist()
    scores = df["consensus_score"].values
    pareto = set(result.pareto_front)

    colors = ["darkorange" if n in pareto else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel("Consensus score")
    ax.set_title(f"Consensus ranking ({result.method}) — top {len(names)}")

    if pareto:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="darkorange", alpha=0.85, label="Pareto front"),
            Patch(facecolor="steelblue", alpha=0.85, label="Other"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_pareto_front(
    metrics_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    pareto_names: list[str],
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Scatter plot highlighting the Pareto front for two objectives.

    Parameters
    ----------
    metrics_df:
        Raw metrics DataFrame (rows = samples, columns = metrics).
    x_metric:
        Column name for the x-axis.
    y_metric:
        Column name for the y-axis.
    pareto_names:
        List of sample names on the Pareto front (from
        :meth:`ConsensusRanker.extract_pareto`).
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    pareto_set = set(pareto_names)
    is_pareto = metrics_df.index.isin(pareto_set)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        metrics_df.loc[~is_pareto, x_metric],
        metrics_df.loc[~is_pareto, y_metric],
        color="steelblue",
        alpha=0.6,
        s=40,
        label="Dominated",
        zorder=2,
    )
    ax.scatter(
        metrics_df.loc[is_pareto, x_metric],
        metrics_df.loc[is_pareto, y_metric],
        color="darkorange",
        alpha=0.9,
        s=70,
        marker="*",
        label="Pareto front",
        zorder=3,
    )

    # Annotate Pareto points
    for name in pareto_names:
        if name in metrics_df.index:
            ax.annotate(
                name,
                (metrics_df.loc[name, x_metric], metrics_df.loc[name, y_metric]),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 4),
            )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"Pareto front: {x_metric} vs {y_metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_metric_correlation(
    metrics_df: pd.DataFrame,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Heatmap of Pearson correlation between all metric columns.

    Parameters
    ----------
    metrics_df:
        Raw metrics DataFrame.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    corr = metrics_df.corr(numeric_only=True)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n - 1)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Metric correlation")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
