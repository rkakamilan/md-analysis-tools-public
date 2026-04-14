"""Dihedral angle visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .validation_plots import shade_hbond


def plot_dihedral_timeseries(
    df: pd.DataFrame,
    *,
    dihedral_cols: list[str] | None = None,
    hbond_mask: np.ndarray | None = None,
    flip_time: float | None = None,
    rolling_window: int = 5,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """One subplot per dihedral with rolling mean and optional H-bond shading.

    Parameters
    ----------
    df:
        DataFrame with 'time_ns' column and one column per dihedral.
    dihedral_cols:
        Subset of columns to plot.  Defaults to all non-metadata columns.
    hbond_mask:
        Boolean array with length == len(df); True = H-bond forming frame.
    flip_time:
        Time (ns) at which a conformational flip occurred; draws a vertical
        dashed line.
    """
    if dihedral_cols is None:
        meta = {"frame", "time_ns", "has_hbond"}
        dihedral_cols = [c for c in df.columns if c not in meta]

    n = len(dihedral_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    time = df["time_ns"]

    for ax, col, color in zip(axes, dihedral_cols, colors):
        if hbond_mask is not None:
            shade_hbond(ax, time, hbond_mask)
        if flip_time is not None:
            ax.axvline(flip_time, color="purple", ls=":", lw=1.8,
                       label=f"Flip @ {flip_time:.0f} ns")
        ax.plot(time, df[col], color=color, lw=1.2)
        if rolling_window > 1:
            roll = df[col].rolling(rolling_window, center=True).mean()
            ax.plot(time, roll, color=color, lw=2.2, alpha=0.7, ls="--",
                    label=f"{rolling_window}-frame rolling mean")
        ax.set_ylabel(f"{col}\n(°)", fontsize=8)
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.grid(axis="y", lw=0.3, alpha=0.4)

    axes[-1].set_xlabel("Simulation time (ns)")
    axes[0].set_title("Ligand Rotatable Bond Dihedrals", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_dihedral_heatmap(
    df: pd.DataFrame,
    *,
    dihedral_cols: list[str] | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Heatmap of dihedral angles across frames."""
    if dihedral_cols is None:
        meta = {"frame", "time_ns", "has_hbond"}
        dihedral_cols = [c for c in df.columns if c not in meta]

    data = df[dihedral_cols].T
    fig, ax = plt.subplots(figsize=(14, max(3, len(dihedral_cols) * 0.55)))
    sns.heatmap(
        data,
        ax=ax,
        cmap="RdBu_r",
        vmin=-180,
        vmax=180,
        cbar_kws={"label": "Dihedral (°)"},
        linewidths=0,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Dihedral")
    ax.set_title("Ligand Dihedral Heatmap", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
