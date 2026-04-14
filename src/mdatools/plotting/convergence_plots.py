"""Convergence assessment visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rmsd_convergence(
    result,  # ConvergenceResult
    rmsd_df: pd.DataFrame,
    column: str = "Backbone",
    dt_factor: float = 1 / 1000,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """RMSD time series with first/second-half highlighted and KS annotation.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.convergence.ConvergenceResult`.
    rmsd_df:
        RMSD DataFrame (must contain *column* and optionally ``'Time'``).
    column:
        Column to plot (default ``'Backbone'``).
    dt_factor:
        Multiply ``Time`` column by this factor to get ns (default 1/1000
        assumes ps).
    save_path:
        If given, save figure to this path.
    dpi:
        Figure resolution.
    """
    values = rmsd_df[column].to_numpy()
    n = len(values)
    split = n // 2

    if "Time" in rmsd_df.columns:
        time = rmsd_df["Time"].to_numpy() * dt_factor
    else:
        time = np.arange(n, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Left: time series
    ax = axes[0]
    ax.plot(time[:split], values[:split], color="steelblue", alpha=0.7,
            linewidth=0.8, label="First half")
    ax.plot(time[split:], values[split:], color="darkorange", alpha=0.7,
            linewidth=0.8, label="Second half")
    ax.set_xlabel("Time (ns)" if "Time" in rmsd_df.columns else "Frame")
    ax.set_ylabel(f"{column} RMSD (Å)")
    ax.set_title("RMSD Time Series")
    ax.legend(fontsize="small")
    ax.grid(linestyle="--", alpha=0.4)
    status = "Converged" if result.is_converged else "NOT Converged"
    ax.text(0.02, 0.97, f"{status}\nKS p={result.ks_pvalue:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            color="green" if result.is_converged else "red")

    # Right: distribution comparison
    ax = axes[1]
    bins = np.linspace(values.min(), values.max(), 40)
    ax.hist(values[:split], bins=bins, alpha=0.5, color="steelblue",
            label=f"First half (μ={result.first_half_mean:.2f} Å)")
    ax.hist(values[split:], bins=bins, alpha=0.5, color="darkorange",
            label=f"Second half (μ={result.second_half_mean:.2f} Å)")
    ax.set_xlabel(f"{column} RMSD (Å)")
    ax.set_ylabel("Count")
    ax.set_title("RMSD Distribution Comparison")
    ax.legend(fontsize="small")
    ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle(f"Convergence Assessment — KS statistic={result.ks_statistic:.4f}",
                 fontsize=10)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_block_error(
    block_df: pd.DataFrame,
    observable_name: str = "observable",
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """SEM vs block size — plateau indicates uncorrelated sampling.

    Parameters
    ----------
    block_df:
        DataFrame from :meth:`ConvergenceAnalyzer.block_error`.
    observable_name:
        Label used in the plot title and y-axis.
    save_path:
        If given, save figure to this path.
    """
    if block_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"Block Error — {observable_name} (no data)")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(block_df["block_size"], block_df["sem"], marker="o", markersize=3,
            linewidth=1.2, color="steelblue")
    ax.set_xlabel("Block size (frames)")
    ax.set_ylabel("SEM")
    ax.set_title(f"Block Error Analysis — {observable_name}")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    # Annotate plateau region (last 20% of x-axis)
    n = len(block_df)
    plateau_start = int(n * 0.8)
    if plateau_start < n:
        plateau_sem = block_df["sem"].iloc[plateau_start:].mean()
        ax.axhline(plateau_sem, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"Plateau SEM ≈ {plateau_sem:.4f}")
        ax.legend(fontsize="small")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_replica_overlap(
    overlap_df: pd.DataFrame,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Heatmap of pairwise KS statistic across replicas.

    Parameters
    ----------
    overlap_df:
        DataFrame from :meth:`ConvergenceAnalyzer.replica_consistency`.
    save_path:
        If given, save figure to this path.
    """
    if overlap_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title("Replica Overlap (no data)")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    # Build symmetric matrix
    replicas = sorted(
        set(overlap_df["replica_a"].tolist() + overlap_df["replica_b"].tolist())
    )
    n = len(replicas)
    idx = {r: i for i, r in enumerate(replicas)}
    matrix = np.zeros((n, n))
    for row in overlap_df.itertuples():
        i, j = idx[row.replica_a], idx[row.replica_b]
        matrix[i, j] = row.ks_statistic
        matrix[j, i] = row.ks_statistic

    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="KS statistic (lower = more similar)",
                 fraction=0.04, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_xticklabels(replicas, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(replicas, fontsize=8)
    ax.set_title("Pairwise Replica KS Statistic")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
