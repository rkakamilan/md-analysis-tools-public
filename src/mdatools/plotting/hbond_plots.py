"""Hydrogen bond visualisation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_occupancy(
    summary: pd.DataFrame,
    sample_name: str,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Horizontal bar chart of H-bond occupancy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    if summary.empty or "hbond_id" not in summary.columns:
        ax.set_title(f"Protein–Ligand Hydrogen Bond Occupancy\n{sample_name} (no H-bonds detected)", fontsize=13)
        ax.text(0.5, 0.5, "No hydrogen bonds detected\n(topology may lack explicit hydrogens)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        return ax
    top = summary.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.45)))
    colors = sns.color_palette("Blues_d", len(top))[::-1]
    bars = ax.barh(top["hbond_id"], top["occupancy_%"], color=colors)
    ax.set_xlabel("Occupancy (%)", fontsize=12)
    ax.set_title(
        f"Protein–Ligand Hydrogen Bond Occupancy\n{sample_name} (Top {top_n})",
        fontsize=13,
    )
    ax.axvline(50, ls="--", color="red", lw=1, label="50%")
    ax.legend()
    ax.invert_yaxis()
    for bar, val in zip(bars, top["occupancy_%"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi)
    return ax


def plot_timeseries(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    sample_name: str,
    top_n: int = 5,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Raster plot of H-bond presence over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(2, top_n * 0.7)))
    if summary.empty or "hbond_id" not in summary.columns or events.empty:
        ax.set_title(f"Hydrogen Bond Presence Over Time\n{sample_name} (no data)", fontsize=13)
        return ax
    top_ids = summary.head(top_n)["hbond_id"].tolist()
    all_frames = sorted(events["frame"].unique())
    matrix = np.zeros((len(top_ids), len(all_frames)), dtype=int)
    frame_idx = {f: i for i, f in enumerate(all_frames)}
    for _, row in events[events["hbond_id"].isin(top_ids)].iterrows():
        i = top_ids.index(row["hbond_id"])
        j = frame_idx[row["frame"]]
        matrix[i, j] = 1

    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(2, top_n * 0.7)))
    ax.imshow(
        matrix,
        aspect="auto",
        cmap="Blues",
        interpolation="none",
        extent=[all_frames[0], all_frames[-1], -0.5, top_n - 0.5],
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_ids, fontsize=9)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_title(
        f"Hydrogen Bond Presence Over Time (Top {top_n})\n{sample_name}", fontsize=13
    )
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi)
    return ax


def plot_distance_distribution(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    sample_name: str,
    top_n: int = 5,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Histogram of donor–acceptor distances for top H-bonds."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if summary.empty or "hbond_id" not in summary.columns or events.empty:
        ax.set_title(f"Donor–Acceptor Distance Distribution\n{sample_name} (no data)", fontsize=13)
        return ax
    top_ids = summary.head(top_n)["hbond_id"].tolist()
    sub = events[events["hbond_id"].isin(top_ids)]
    for hid in top_ids:
        vals = sub[sub["hbond_id"] == hid]["distance"]
        ax.hist(vals, bins=40, alpha=0.6, label=hid, density=True)
    ax.set_xlabel("D–A Distance (Å)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Donor–Acceptor Distance Distribution\n{sample_name}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi)
    return ax


def plot_hbond_count_per_frame(
    events: pd.DataFrame,
    n_frames_total: int,
    sample_name: str,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Line plot of H-bond count per frame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    if events.empty:
        ax.set_title(f"Number of Protein–Ligand H-bonds per Frame\n{sample_name} (no data)", fontsize=13)
        return ax
    counts = events.groupby("frame").size().reindex(range(n_frames_total), fill_value=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(counts.index, counts.values, lw=0.8, color="steelblue", alpha=0.8)
    ax.fill_between(counts.index, counts.values, alpha=0.2, color="steelblue")
    ax.axhline(
        counts.mean(),
        ls="--",
        color="red",
        lw=1.2,
        label=f"Mean = {counts.mean():.2f}",
    )
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("# H-bonds", fontsize=11)
    ax.set_title(
        f"Number of Protein–Ligand H-bonds per Frame\n{sample_name}", fontsize=13
    )
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi)
    return ax
