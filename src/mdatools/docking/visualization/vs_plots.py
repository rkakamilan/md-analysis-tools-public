"""Visualisation helpers for virtual screening evaluation metrics.

All functions follow the same conventions as the rest of the library:
- ``output_path`` saves the figure to disk when given.
- Functions return a ``plt.Figure`` so callers can customise further.

Example::

    from mdatools.docking.visualization.vs_plots import plot_roc_curves
    fig = plot_roc_curves(results)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdatools.docking.analysis.vs_metrics import VSMetricsResult


def plot_roc_curves(
    results: dict[str, VSMetricsResult],
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (6, 6),
    title: str = "ROC Curves",
) -> plt.Figure:
    """Overlay ROC curves for multiple scoring runs.

    Parameters
    ----------
    results:
        Mapping ``{name: VSMetricsResult}`` as returned by
        :meth:`~mdatools.docking.analysis.vs_metrics.VirtualScreeningEvaluator.run_multi`.
    output_path:
        If given, save the figure to this path (PNG/SVG/PDF).
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    title:
        Figure title.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, res in results.items():
        ax.plot(res.fpr, res.tpr, label=f"{name} (AUC={res.roc_auc:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig


def plot_enrichment_curves(
    results: dict[str, VSMetricsResult],
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (6, 6),
    title: str = "Enrichment Curves",
) -> plt.Figure:
    """Plot % library screened vs % actives recovered for each run.

    Parameters
    ----------
    results:
        Mapping ``{name: VSMetricsResult}``.
    output_path:
        If given, save the figure.
    figsize:
        Matplotlib figure size.
    title:
        Figure title.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, res in results.items():
        # Enrichment curve: x = % library, y = % actives recovered
        # Derived from the ROC data (tpr vs fpr), scaling fpr and tpr properly.
        # fpr = FP / (FP+TN) = fraction of decoys screened
        # tpr = TP / (TP+FN) = fraction of actives recovered
        # % library = (TP + FP) / N = tpr*n_act/N + fpr*(N-n_act)/N
        n = res.n_total
        n_act = res.n_actives
        pct_lib = (res.tpr * n_act + res.fpr * (n - n_act)) / n * 100.0
        pct_act = res.tpr * 100.0
        ax.plot(pct_lib, pct_act, label=name, linewidth=1.5)

    # Random diagonal
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("% Library Screened")
    ax.set_ylabel("% Actives Recovered")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig


def plot_ef_bars(
    results: dict[str, VSMetricsResult],
    pct: float = 5.0,
    normalised: bool = False,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str | None = None,
) -> plt.Figure:
    """Bar chart comparing EF at a given percentile across multiple runs.

    Parameters
    ----------
    results:
        Mapping ``{name: VSMetricsResult}``.
    pct:
        EF percentile to plot (must be one of the values used when
        :class:`~mdatools.docking.analysis.vs_metrics.VirtualScreeningEvaluator`
        was created, default 5.0).
    normalised:
        If ``True``, plot Normalised EF (NEF = EF / EF_max) instead of raw EF.
    output_path:
        If given, save the figure.
    figsize:
        Matplotlib figure size.
    title:
        Figure title.  Defaults to ``"EF{pct}%"`` or ``"NEF{pct}%"``.

    Returns
    -------
    plt.Figure
    """
    names = list(results.keys())
    if normalised:
        values = [results[n].ef_norm[pct] for n in names]
        ylabel = f"NEF{pct:g}% (0–1)"
        default_title = f"Normalised EF{pct:g}%"
    else:
        values = [results[n].ef[pct] for n in names]
        ylabel = f"EF{pct:g}%"
        default_title = f"Enrichment Factor at {pct:g}%"

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="Random (EF=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig


def plot_vs_summary_table(
    results: dict[str, VSMetricsResult],
    ef_percentiles: tuple[float, ...] = (1.0, 5.0, 10.0),
    include_bedroc: bool = True,
    include_roc_auc: bool = True,
) -> pd.DataFrame:
    """Build a summary DataFrame of key VS metrics.

    Parameters
    ----------
    results:
        Mapping ``{name: VSMetricsResult}``.
    ef_percentiles:
        Which EF columns to include in the table.
    include_bedroc:
        Add a BEDROC column.
    include_roc_auc:
        Add a ROC-AUC column.

    Returns
    -------
    pd.DataFrame
        Rows = runs, columns = metric values.  Sorted by ``EF5%`` descending
        (or the first percentile if 5.0 is not requested).
    """
    rows = []
    for name, res in results.items():
        row: dict[str, object] = {"name": name}
        for pct in ef_percentiles:
            if pct in res.ef:
                row[f"EF{pct:g}%"] = round(res.ef[pct], 3)
                row[f"NEF{pct:g}%"] = round(res.ef_norm[pct], 3)
        if include_roc_auc:
            row["ROC-AUC"] = round(res.roc_auc, 4)
        if include_bedroc:
            row["BEDROC"] = round(res.bedroc, 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("name")

    # Sort by first available EF column, descending
    sort_cols = [f"EF{pct:g}%" for pct in ef_percentiles if f"EF{pct:g}%" in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols[0], ascending=False)

    return df
