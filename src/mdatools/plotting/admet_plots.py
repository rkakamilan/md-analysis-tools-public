"""Plotting functions for ADMET / drug-likeness results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.admet import ADMETResult

# Radar axes: (display label, property name, max reference value for normalisation)
_RADAR_AXES = [
    ("MW",      "mw",       500.0),
    ("LogP",    "logp",     5.0),
    ("HBD",     "hbd",      5.0),
    ("HBA",     "hba",      10.0),
    ("TPSA",    "tpsa",     140.0),
    ("RotBonds","rotbonds", 10.0),
]


def plot_admet_radar(
    result: ADMETResult,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Radar chart of normalised physicochemical properties.

    Each axis is normalised by the Lipinski/Veber upper bound so that a fully
    drug-like molecule fits inside the unit circle.

    Parameters
    ----------
    result:
        :class:`ADMETResult` for a single molecule.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    labels = [a[0] for a in _RADAR_AXES]
    values = [getattr(result, a[1]) / a[2] for a in _RADAR_AXES]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Close the polygon
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    labels_closed = labels + [labels[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles_closed, values_closed, "o-", lw=1.5, color="steelblue")
    ax.fill(angles_closed, values_closed, alpha=0.2, color="steelblue")

    # Ro5 limit circle at 1.0
    ax.plot(angles_closed, [1.0] * (n + 1), "--", lw=0.8, color="gray", alpha=0.6)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(1.5, max(values) * 1.1))
    ax.set_title(f"ADMET radar — {result.name}", pad=15)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_admet_comparison(
    results: dict[str, ADMETResult],
    properties: list[str] | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Grouped bar chart comparing ADMET properties across multiple molecules.

    Parameters
    ----------
    results:
        Mapping of *name* → :class:`ADMETResult`.
    properties:
        Property names to include. Defaults to
        ``['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotbonds', 'qed', 'sa_score']``.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    if properties is None:
        properties = ["mw", "logp", "hbd", "hba", "tpsa", "rotbonds", "qed", "sa_score"]

    names = list(results.keys())
    n_mols = len(names)
    n_props = len(properties)

    data = np.array(
        [[getattr(results[n], p) for p in properties] for n in names],
        dtype=float,
    )

    x = np.arange(n_props)
    width = 0.8 / max(n_mols, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_props * 1.5), 5))
    for i, (name, row) in enumerate(zip(names, data)):
        offset = (i - n_mols / 2 + 0.5) * width
        ax.bar(x + offset, row, width=width * 0.9, label=name, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(properties, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("ADMET property comparison")
    if n_mols <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_qed_distribution(
    results_df: "pd.DataFrame",
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Histogram of QED scores from a batch results DataFrame.

    Parameters
    ----------
    results_df:
        DataFrame returned by :meth:`ADMETCalculator.batch`.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    qed_vals = results_df["qed"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(qed_vals, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="darkorange", ls="--", lw=1.2, label="QED = 0.5")
    ax.set_xlabel("QED score")
    ax.set_ylabel("Count")
    ax.set_title("QED distribution")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
