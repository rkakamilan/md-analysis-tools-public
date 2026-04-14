"""Pose clustering visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colour palette for clusters (cycles for > 10 clusters)
_CLUSTER_CMAP = "tab10"


def plot_cluster_timeline(
    result,  # ClusterResult
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Scatter plot of cluster assignment over simulation frames.

    Each frame is coloured by its cluster ID.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.clustering.ClusterResult`.
    save_path:
        If given, save figure to this path.
    """
    labels = result.labels
    n_frames = len(labels)
    cmap = matplotlib.colormaps.get_cmap(_CLUSTER_CMAP)

    fig, ax = plt.subplots(figsize=(max(8, n_frames * 0.04), 3), constrained_layout=True)
    scatter = ax.scatter(
        range(n_frames),
        labels,
        c=labels,
        cmap=cmap,
        s=8,
        linewidths=0,
        vmin=0,
        vmax=max(result.n_clusters - 1, 1),
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cluster ID")
    ax.set_title(f"Cluster Assignment over Time — {result.sample_name}")
    ax.set_yticks(range(result.n_clusters))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Cluster ID", ticks=range(result.n_clusters))

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_cluster_population(
    result,  # ClusterResult
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of cluster population percentages.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.clustering.ClusterResult`.
    save_path:
        If given, save figure to this path.
    """
    summary: pd.DataFrame = result.summary
    if summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Cluster Population — {result.sample_name} (no data)")
        return fig

    cmap = matplotlib.colormaps.get_cmap(_CLUSTER_CMAP)
    colors = [cmap(int(cid) % 10) for cid in summary["cluster_id"]]

    fig, ax = plt.subplots(
        figsize=(max(5, len(summary) * 0.8), 4), constrained_layout=True
    )
    bars = ax.bar(
        [f"Cluster {int(c)}" for c in summary["cluster_id"]],
        summary["pct"],
        color=colors,
        edgecolor="grey",
        linewidth=0.5,
    )
    # Annotate bars with frame counts
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={row.n_frames}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Population (%)")
    ax.set_ylim(0, min(100, summary["pct"].max() + 10))
    ax.set_title(f"Cluster Population — {result.sample_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_rmsd_matrix(
    result,  # ClusterResult
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Heatmap of pairwise ligand RMSD matrix.

    Frames are ordered by cluster assignment so clusters appear as
    contiguous blocks.  Returns a figure with a note when ``rmsd_matrix``
    is ``None`` (e.g. for DIVINE results).

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.clustering.ClusterResult`.
    save_path:
        If given, save figure to this path.
    """
    if result.rmsd_matrix is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            f"RMSD matrix not available for method='{result.method}'",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title(f"Pairwise Ligand RMSD — {result.sample_name}")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    order = np.argsort(result.labels, kind="stable")
    reordered = result.rmsd_matrix[np.ix_(order, order)]

    size = max(5, len(order) * 0.04)
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    im = ax.imshow(reordered, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="RMSD (Å)", fraction=0.04, pad=0.04)

    # Draw cluster boundary lines
    boundaries = []
    prev = result.labels[order[0]]
    for i, lbl in enumerate(result.labels[order]):
        if lbl != prev:
            boundaries.append(i - 0.5)
            prev = lbl
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=0.8, alpha=0.7)
        ax.axvline(b, color="white", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Frame (ordered by cluster)")
    ax.set_ylabel("Frame (ordered by cluster)")
    ax.set_title(f"Pairwise Ligand RMSD — {result.sample_name}")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_dendrogram(
    result,  # ClusterResult
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Dendrogram of Ward hierarchical clustering.

    Only available when ``result.method == 'ward'``.  Returns an empty
    figure with a note for other methods.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.clustering.ClusterResult`.
    save_path:
        If given, save figure to this path.
    """
    if result.method != "ward":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5,
            f"Dendrogram not available for method='{result.method}'",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title(f"Dendrogram — {result.sample_name}")
        if save_path is not None:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    condensed = squareform(result.rmsd_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(max(8, len(result.labels) * 0.1), 5), constrained_layout=True)
    dendrogram(
        Z,
        ax=ax,
        no_labels=True,
        color_threshold=result.rmsd_matrix.max() / result.n_clusters,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ward Distance (Å)")
    ax.set_title(f"Ward Dendrogram — {result.sample_name} ({result.n_clusters} clusters)")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
