"""Residue-level contact heatmap visualisation.

Aggregates ProLIF interaction fingerprint columns into a
*group × residue* contact-rate matrix and renders it as a heatmap.
Supports dendrogram-ordered row clustering and per-cell annotation.

Example::

    rate_matrix = build_contact_rate_matrix(df, residue_cols, group_col="cluster_id")
    fig = plot_contact_heatmap(df, residue_cols, group_col="cluster_id",
                               cluster_rows=True, annotate=True,
                               output_path="results/contact_heatmap.png")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_contact_rate_matrix(
    df: pd.DataFrame,
    residue_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """Compute per-group contact rates for each residue column.

    Parameters
    ----------
    df:
        Poses DataFrame containing binary interaction flag columns and a
        group column.
    residue_cols:
        List of interaction flag column names (e.g., ``["GLN30_HBAcceptor",
        "ARG38_HBDonor"]``).  Columns absent from *df* are ignored.
    group_col:
        Column used for grouping (e.g., ``"cluster_id"`` or
        ``"receptor"``).

    Returns
    -------
    pd.DataFrame
        Shape ``(n_groups, n_residues)`` with contact rates in percent
        ``[0, 100]``.  Index = group labels, columns = residue column names.
    """
    present_cols = [c for c in residue_cols if c in df.columns]
    if not present_cols:
        return pd.DataFrame(index=df[group_col].unique(), columns=residue_cols,
                             dtype=float)

    matrix = (
        df.groupby(group_col)[present_cols]
        .mean()
        .multiply(100)
    )
    return matrix


def _shorten_label(col: str, max_width: int = 12) -> str:
    """Convert a ProLIF column name to a compact two-line label.

    ``GLN30_HBAcceptor`` → ``GLN30\\nHBAcceptor``
    """
    parts = col.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return col


def plot_contact_heatmap(
    df: pd.DataFrame,
    residue_cols: list[str],
    group_col: str,
    cluster_rows: bool = False,
    annotate: bool = False,
    cmap: str = "Blues",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Residue Contact Rates (%)",
) -> "plt.Figure":  # noqa: F821
    """Plot a group × residue contact-rate heatmap.

    Parameters
    ----------
    df:
        Poses DataFrame with interaction flag columns and *group_col*.
    residue_cols:
        Interaction flag column names to display on the x-axis.
    group_col:
        Column for row grouping (clusters, receptors, templates, …).
    cluster_rows:
        If ``True``, reorder rows by hierarchical clustering of the contact
        rate vectors (average linkage, Euclidean distance).
    annotate:
        If ``True``, draw contact-rate values inside each cell.
    cmap:
        Matplotlib colormap name (default ``"Blues"``).
    output_path:
        If given, save the figure to this path.  Parent directories are
        created automatically.
    figsize:
        Figure size ``(width, height)`` in inches.  Auto-sized when
        ``None``.
    title:
        Figure title.

    Returns
    -------
    plt.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = build_contact_rate_matrix(df, residue_cols, group_col)

    if matrix.empty:
        fig, ax = plt.subplots()
        ax.set_title(title)
        return fig

    present_cols = [c for c in residue_cols if c in matrix.columns]
    matrix = matrix[present_cols]

    # Row clustering
    if cluster_rows and len(matrix) > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        dists = pdist(matrix.values, metric="euclidean")
        Z = linkage(dists, method="average")
        order = leaves_list(Z)
        matrix = matrix.iloc[order]

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(6, n_cols * 0.9 + 2), max(4, n_rows * 0.6 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    # Axes labels
    xlabels = [_shorten_label(c) for c in present_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(g) for g in matrix.index], fontsize=9)
    ax.set_xlabel("Residue / Interaction", fontsize=9)
    ax.set_ylabel(group_col, fontsize=9)
    ax.set_title(title, fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Contact rate (%)", fontsize=8)

    # Cell annotations
    if annotate:
        for row_i in range(n_rows):
            for col_j in range(n_cols):
                val = matrix.values[row_i, col_j]
                if not np.isnan(val):
                    text_color = "white" if val > 60 else "black"
                    ax.text(col_j, row_i, f"{val:.0f}",
                            ha="center", va="center",
                            fontsize=7, color=text_color)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig
