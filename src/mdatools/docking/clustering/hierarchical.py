"""RMSD-based hierarchical clustering of docked poses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from mdatools.docking.clustering.rmsd import compute_rmsd_matrix


@dataclass
class ClusteringResult:
    """Output of :func:`cluster_poses`.

    Attributes
    ----------
    mols:
        Input molecules with the ``cluster_id`` SDF property set on each.
    cluster_ids:
        1-D integer array of cluster assignments, parallel to *mols*.
    rmsd_matrix:
        Symmetric pairwise RMSD matrix used for clustering.
    linkage_matrix:
        SciPy linkage matrix (for dendrogram / re-clustering).
    n_clusters:
        Number of unique clusters found.
    threshold:
        RMSD distance threshold used to cut the dendrogram.
    """

    mols: list[Chem.Mol]
    cluster_ids: np.ndarray
    rmsd_matrix: np.ndarray
    linkage_matrix: np.ndarray
    n_clusters: int
    threshold: float


def cluster_poses(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 2.0,
    method: str = "average",
) -> ClusteringResult:
    """Perform hierarchical RMSD clustering on a set of docked poses.

    Parameters
    ----------
    mols:
        List of RDKit Mols (one conformer each, same atom ordering).
    rmsd_threshold:
        Distance threshold (Å) for cutting the dendrogram into clusters.
    method:
        SciPy linkage method: ``"single"``, ``"complete"``, ``"average"``
        (default), or ``"ward"``.

    Returns
    -------
    ClusteringResult
        Each mol has its ``cluster_id`` property set.
    """
    rmsd_matrix = compute_rmsd_matrix(mols, align=False)
    condensed = squareform(rmsd_matrix)
    Z = linkage(condensed, method=method)
    cluster_ids = fcluster(Z, rmsd_threshold, criterion="distance")

    for cid, mol in zip(cluster_ids, mols):
        mol.SetProp("cluster_id", str(int(cid)))

    return ClusteringResult(
        mols=mols,
        cluster_ids=cluster_ids,
        rmsd_matrix=rmsd_matrix,
        linkage_matrix=Z,
        n_clusters=int(np.unique(cluster_ids).size),
        threshold=rmsd_threshold,
    )


def find_optimal_threshold(
    linkage_matrix: np.ndarray,
    target_clusters: int = 25,
    search_range: tuple[float, float] = (1.0, 20.0),
    step: float = 0.5,
) -> float:
    """Grid-search for the smallest RMSD threshold yielding ≤ *target_clusters*.

    Parameters
    ----------
    linkage_matrix:
        SciPy linkage matrix from a previous :func:`cluster_poses` call.
    target_clusters:
        Maximum desired number of clusters.
    search_range:
        ``(min, max)`` RMSD range in Å to search over.
    step:
        Step size in Å.

    Returns
    -------
    float
        The first threshold (in ascending order) that yields ≤ *target_clusters*
        clusters.  Returns the maximum of *search_range* if the target is never
        reached.
    """
    thresholds = np.arange(search_range[0], search_range[1] + step, step)

    for thresh in thresholds:
        n = int(np.unique(fcluster(linkage_matrix, thresh, criterion="distance")).size)
        if n <= target_clusters:
            return float(thresh)

    return float(thresholds[-1])


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: float,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5),
    **matplotlib_kwargs,
) -> plt.Figure:
    """Plot a hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix:
        SciPy linkage matrix.
    threshold:
        RMSD threshold drawn as a horizontal reference line.
    output_path:
        If provided, the figure is saved to this path (PNG/SVG/PDF).
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    **matplotlib_kwargs:
        Additional keyword arguments forwarded to :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object (caller can customise further).
    """
    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(linkage_matrix, ax=ax, no_labels=True, **matplotlib_kwargs)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"threshold = {threshold:.1f} Å")
    ax.set_xlabel("Poses")
    ax.set_ylabel("RMSD distance (Å)")
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig
