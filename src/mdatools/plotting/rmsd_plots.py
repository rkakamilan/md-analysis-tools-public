"""RMSD visualisation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_rmsd_single(
    df: pd.DataFrame,
    sample_name: str,
    ax: plt.Axes | None = None,
    dt_factor: float = 1 / 1000,
    ylim: tuple[float, float] = (0, 10),
) -> plt.Axes:
    """Plot backbone + ligand RMSD for one replica.

    Parameters
    ----------
    df:
        DataFrame with columns Frame, Time, Backbone, Ligand.
    dt_factor:
        Multiply the Time column by this factor to convert to ns.
        Default 1/1000 assumes Time is in ps.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    time = df["Time"] * dt_factor
    ax.plot(time, df["Backbone"], label="Protein Backbone")
    ax.plot(time, df["Ligand"], label="Ligand")
    ax.set_title(sample_name)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize="small")
    return ax


def plot_rmsd_grid(
    results,  # dict[str, RMSDResult] or list[RMSDResult]
    n_cols: int = 3,
    dt_factor: float = 1 / 1000,
    ylim: tuple[float, float] = (0, 10),
    save_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot RMSD for all replicas in a grid.

    Parameters
    ----------
    results:
        Either a dict mapping ``sample_name → RMSDResult``, or a list of
        ``RMSDResult`` objects (uses ``sample_name`` attribute as label).
    """
    if isinstance(results, dict):
        results_dict = results
    else:
        results_dict = {r.sample_name: r for r in results}

    n_samples = len(results_dict)
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True
    )
    axes_flat = axes.flatten() if n_samples > 1 else [axes]

    for i, (name, res) in enumerate(results_dict.items()):
        plot_rmsd_single(
            res.df, name, ax=axes_flat[i], dt_factor=dt_factor, ylim=ylim
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi)
    return fig
