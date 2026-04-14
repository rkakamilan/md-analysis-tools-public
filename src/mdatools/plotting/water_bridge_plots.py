"""Water bridge visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bridge_occupancy(
    result,  # WaterBridgeResult
    top_n: int = 10,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of water bridge occupancy (protein_atom ↔ ligand_atom pairs).

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.water_bridges.WaterBridgeResult`.
    top_n:
        Show only the top-*n* bridges by occupancy.
    save_path:
        If given, save figure to this path.
    dpi:
        Figure resolution.
    """
    summary: pd.DataFrame = result.summary
    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Water Bridge Occupancy — {result.sample_name} (no bridges)")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    top = summary.head(top_n).copy()
    labels = [
        f"{row.protein_atom}\n↕\n{row.ligand_atom}"
        for row in top.itertuples()
    ]

    cmap = matplotlib.colormaps.get_cmap("Blues")
    colors = [cmap(0.4 + 0.6 * v / 100) for v in top["occupancy_pct"]]

    fig, ax = plt.subplots(
        figsize=(max(6, len(top) * 1.2), 5), constrained_layout=True
    )
    bars = ax.bar(range(len(top)), top["occupancy_pct"], color=colors, edgecolor="grey", linewidth=0.5)
    for bar, row in zip(bars, top.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row.occupancy_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Occupancy (%)")
    ax.set_ylim(0, min(105, top["occupancy_pct"].max() + 12))
    ax.set_title(f"Water Bridge Occupancy — {result.sample_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_bridge_timeline(
    result,  # WaterBridgeResult
    top_n: int = 5,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Time series of water bridge formation/breaking events.

    Shows binary presence (1/0) for the top-*n* bridges over simulation
    frames.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.water_bridges.WaterBridgeResult`.
    top_n:
        Number of most-occupied bridges to display.
    save_path:
        If given, save figure to this path.
    """
    events: pd.DataFrame = result.events
    summary: pd.DataFrame = result.summary

    if events.empty or summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Water Bridge Timeline — {result.sample_name} (no bridges)")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    # Select top bridges
    top = summary.head(top_n)
    n_frames = result.n_frames
    all_frames = np.arange(n_frames)

    fig, axes = plt.subplots(
        len(top), 1,
        figsize=(max(10, n_frames * 0.04), max(3, len(top) * 1.2)),
        sharex=True,
        constrained_layout=True,
    )
    if len(top) == 1:
        axes = [axes]

    cmap = matplotlib.colormaps.get_cmap("tab10")

    for i, (ax, row) in enumerate(zip(axes, top.itertuples())):
        mask = (
            (events["protein_atom"] == row.protein_atom)
            & (events["ligand_atom"] == row.ligand_atom)
        )
        bridge_frames = set(events.loc[mask, "frame"].unique())
        presence = np.array([1 if f in bridge_frames else 0 for f in all_frames])

        ax.fill_between(all_frames, presence, alpha=0.6, color=cmap(i % 10))
        ax.set_ylabel(f"C{i}", fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["", ""], fontsize=6)
        ax.set_ylim(-0.1, 1.2)
        label = f"{row.protein_atom} ↔ {row.ligand_atom}  ({row.occupancy_pct:.1f}%)"
        ax.set_title(label, fontsize=7, loc="left", pad=2)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"Water Bridge Timeline — {result.sample_name}", fontsize=9)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
