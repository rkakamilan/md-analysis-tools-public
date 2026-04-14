"""Plots for covalent bond distance monitoring results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..analysis.covalent import CovalentBondResult

# Reference distances (Å)
_IDEAL_CS_BOND = 1.82
_WARN_THRESHOLD = 2.5


def plot_covalent_bond_distance(
    result: CovalentBondResult,
    ideal_distance: float = _IDEAL_CS_BOND,
    warn_threshold: float = _WARN_THRESHOLD,
    color: str = "#2196F3",
) -> Figure:
    """Plot covalent bond distance time series and distribution side-by-side.

    Parameters
    ----------
    result :
        Output of :func:`~mdatools.analysis.covalent.covalent_bond_monitor`.
    ideal_distance :
        Expected bond length in Å (default: 1.82 Å for C–S).
    warn_threshold :
        Distance above which a warning line is drawn (default: 2.5 Å).
    color :
        Line / bar colour for this sample.

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Left: time series ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(
        result.time_ns,
        result.distances,
        lw=0.8,
        alpha=0.85,
        color=color,
        label=result.sample_name or "bond distance",
    )
    ax.axhline(
        ideal_distance,
        color="green",
        ls="--",
        lw=1.5,
        label=f"Ideal ({ideal_distance} \u00c5)",
    )
    ax.axhline(
        warn_threshold,
        color="orange",
        ls=":",
        lw=1.5,
        label=f"Warning ({warn_threshold} \u00c5)",
    )
    ax.axhline(
        result.mean_distance,
        color=color,
        ls="-",
        lw=1.0,
        alpha=0.5,
        label=f"Mean = {result.mean_distance:.2f} \u00c5",
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Bond distance (\u00c5)")
    ax.set_title(
        f"Covalent bond distance\n{result.sample_name}"
        if result.sample_name
        else "Covalent bond distance"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(warn_threshold * 1.4, result.distances.max() * 1.1))

    # ── Right: distribution ────────────────────────────────────────────────
    ax = axes[1]
    bins = np.linspace(
        max(0.5, result.distances.min() - 0.5),
        result.distances.max() + 0.5,
        40,
    )
    ax.hist(result.distances, bins=bins, alpha=0.75, color=color, density=True)
    ax.axvline(
        ideal_distance,
        color="green",
        ls="--",
        lw=1.5,
        label=f"Ideal ({ideal_distance} \u00c5)",
    )
    ax.axvline(
        result.mean_distance,
        color=color,
        ls="-",
        lw=1.5,
        alpha=0.8,
        label=f"Mean = {result.mean_distance:.2f} \u00c5",
    )
    ax.set_xlabel("Bond distance (\u00c5)")
    ax.set_ylabel("Density")
    ax.set_title("Distance distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig
