"""Interaction fingerprint visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_interaction_heatmap(
    result,  # InteractionResult
    top_n: int = 20,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Heatmap of interaction occupancy: residue × interaction type.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.interactions.InteractionResult`.
    top_n:
        Show only the top-*n* interactions by occupancy.
    save_path:
        If given, save figure to this path.
    dpi:
        Figure resolution.
    """
    summary: pd.DataFrame = result.summary
    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Interaction Heatmap — {result.sample_name} (no data)")
        return fig

    top = summary.head(top_n)

    # Parse residue and interaction type from label "RES-Chain-ResID_IType"
    rows_data: dict[str, dict[str, float]] = {}
    for _, row in top.iterrows():
        label: str = str(row["interaction"])
        if "_" in label:
            res_part, itype = label.rsplit("_", 1)
        else:
            res_part, itype = label, "Unknown"
        rows_data.setdefault(res_part, {})[itype] = float(row["occupancy_pct"])

    df_pivot = pd.DataFrame(rows_data).T.fillna(0)
    df_pivot = df_pivot.loc[df_pivot.max(axis=1).sort_values(ascending=False).index]

    fig_h = max(4, len(df_pivot) * 0.4)
    fig_w = max(6, len(df_pivot.columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(
        df_pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(df_pivot)))
    ax.set_yticklabels(df_pivot.index, fontsize=8)
    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, fontsize=8, rotation=45, ha="right")
    ax.set_title(f"Interaction Occupancy (%) — {result.sample_name}")
    plt.colorbar(im, ax=ax, label="Occupancy (%)", fraction=0.03, pad=0.04)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_interaction_timeline(
    result,  # InteractionResult
    interactions: list[str] | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Binary timeline of interaction presence over simulation frames.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.interactions.InteractionResult`.
    interactions:
        Column names to show. Defaults to top 10 by occupancy.
    save_path:
        If given, save figure to this path.
    """
    df: pd.DataFrame = result.df
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Interaction Timeline — {result.sample_name} (no data)")
        return fig

    if interactions is None:
        # Select top 10 by mean occupancy
        occ = (df > 0).mean().sort_values(ascending=False)
        cols = occ.head(10).index.tolist()
    else:
        cols = [c for c in interactions if c in df.columns]

    if not cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Interaction Timeline — {result.sample_name} (no columns)")
        return fig

    sub = (df[cols] > 0).astype(int)

    fig, ax = plt.subplots(
        figsize=(max(10, len(sub) * 0.05), max(4, len(cols) * 0.4)),
        constrained_layout=True,
    )
    cmap = matplotlib.colormaps.get_cmap("Blues")
    ax.imshow(
        sub.T.values,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_xlabel("Frame")
    ax.set_title(f"Interaction Timeline — {result.sample_name}")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_interaction_comparison(
    results: list,  # list[InteractionResult]
    top_n: int = 15,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Grouped bar chart comparing interaction occupancy across replicas.

    Parameters
    ----------
    results:
        List of :class:`~mdatools.analysis.interactions.InteractionResult`.
    top_n:
        Show only the top-*n* interactions (ranked by max occupancy across
        all replicas).
    save_path:
        If given, save figure to this path.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Interaction Comparison (no data)")
        return fig

    # Merge summaries
    frames = []
    for res in results:
        s = res.summary.copy()
        s["sample"] = res.sample_name
        frames.append(s)
    merged = pd.concat(frames, ignore_index=True)

    # Select top interactions by max occupancy
    max_occ = (
        merged.groupby("interaction")["occupancy_pct"]
        .max()
        .sort_values(ascending=False)
    )
    top_interactions = max_occ.head(top_n).index.tolist()
    sub = merged[merged["interaction"].isin(top_interactions)]

    pivot = sub.pivot_table(
        index="interaction", columns="sample", values="occupancy_pct", fill_value=0
    )
    pivot = pivot.loc[top_interactions]

    n_groups = len(pivot)
    n_bars = len(pivot.columns)
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    cmap = matplotlib.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.6), 5), constrained_layout=True)
    for i, col in enumerate(pivot.columns):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, pivot[col], width=width, label=col, color=cmap(i % 10))

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Occupancy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Interaction Fingerprint Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize="small")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
