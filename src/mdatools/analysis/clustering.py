"""Binding pose clustering by ligand RMSD matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Results from pose clustering analysis.

    Attributes
    ----------
    sample_name:
        Identifier for the replica/sample.
    labels:
        Cluster assignment for each frame, shape ``(n_frames,)``.
        Cluster IDs start from 0.
    centers:
        List of representative frame indices (one per cluster).
    rmsd_matrix:
        Symmetric pairwise RMSD matrix, shape ``(n_frames, n_frames)``.
        ``None`` for methods that do not compute the full matrix (e.g. DIVINE).
    summary:
        Per-cluster statistics. Columns: ``cluster_id``, ``n_frames``,
        ``pct``, ``center_frame``, ``mean_rmsd``.
    method:
        Clustering method used.
    n_clusters:
        Number of clusters detected.
    """

    sample_name: str
    labels: np.ndarray
    centers: list[int]
    rmsd_matrix: Optional[np.ndarray]
    summary: pd.DataFrame
    method: str
    n_clusters: int


class PoseClusterer:
    """Cluster trajectory frames by ligand RMSD matrix.

    Supported methods:
    - ``'ward'``: hierarchical clustering with Ward linkage (default).
    - ``'kmeans'``: k-means on flattened coordinate vectors.

    scipy (already a core dependency) provides both algorithms.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        method: str = "ward",
        n_clusters: int | None = None,
        rmsd_cutoff: float = 2.0,
    ) -> None:
        if method not in ("ward", "kmeans"):
            raise ValueError(f"Unknown clustering method: '{method}'. Use 'ward' or 'kmeans'.")
        self.cfg = cfg
        self.method = method
        self.n_clusters = n_clusters
        self.rmsd_cutoff = rmsd_cutoff

    def run(self, u: mda.Universe, sample_name: str = "") -> ClusterResult:
        """Cluster trajectory frames and return a :class:`ClusterResult`.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`ClusterResult`.
        """
        lig_sel = f"resname {self.cfg.ligand_resname} and not name H*"
        ligand = u.select_atoms(lig_sel)
        if len(ligand) == 0:
            raise ValueError(
                f"Ligand '{self.cfg.ligand_resname}' not found in universe."
            )

        # Collect ligand positions for all frames: (n_frames, n_atoms, 3)
        positions = np.array(
            [ligand.positions.copy() for _ in u.trajectory], dtype=np.float64
        )
        n_frames = len(positions)

        # Compute pairwise RMSD matrix
        rmsd_matrix = _compute_rmsd_matrix(positions)

        # Cluster
        if self.method == "ward":
            labels, n_clusters = _ward_cluster(
                rmsd_matrix, self.n_clusters, self.rmsd_cutoff
            )
        else:
            labels, n_clusters = _kmeans_cluster(
                positions, self.n_clusters or max(2, n_frames // 10)
            )

        # Find cluster medoids (frame closest to all others in same cluster)
        centers = _find_medoids(rmsd_matrix, labels, n_clusters)

        summary = _build_summary(labels, centers, n_clusters, rmsd_matrix=rmsd_matrix)

        logger.info(
            "Clustering done for %s: %d frames → %d clusters (%s)",
            sample_name,
            n_frames,
            n_clusters,
            self.method,
        )
        return ClusterResult(
            sample_name=sample_name,
            labels=labels,
            centers=centers,
            rmsd_matrix=rmsd_matrix,
            summary=summary,
            method=self.method,
            n_clusters=n_clusters,
        )

    def extract_representatives(
        self,
        u: mda.Universe,
        result: ClusterResult,
        output_dir: Path | str,
    ) -> list[Path]:
        """Write one clean PDB per cluster representative frame.

        Parameters
        ----------
        u:
            Universe (aligned trajectory).
        result:
            Clustering result from :meth:`run`.
        output_dir:
            Directory where PDB files are saved.

        Returns
        -------
        List of paths to written PDB files, one per cluster.
        """
        from ..io.writers import write_clean_pdb

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for cluster_id, frame_idx in enumerate(result.centers):
            u.trajectory[frame_idx]
            out = output_dir / f"cluster{cluster_id:02d}_frame{frame_idx}.pdb"
            write_clean_pdb(u, out, ligand_resname=self.cfg.ligand_resname)
            paths.append(out)
        return paths


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_rmsd_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute symmetric pairwise RMSD matrix.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)``.

    Returns
    -------
    Symmetric matrix of shape ``(n_frames, n_frames)``.
    """
    n_frames, n_atoms, _ = positions.shape
    matrix = np.zeros((n_frames, n_frames), dtype=np.float64)
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            diff = positions[i] - positions[j]
            rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
            matrix[i, j] = rmsd
            matrix[j, i] = rmsd
    return matrix


def _ward_cluster(
    rmsd_matrix: np.ndarray,
    n_clusters: int | None,
    rmsd_cutoff: float,
) -> tuple[np.ndarray, int]:
    """Apply Ward hierarchical clustering to a precomputed RMSD matrix."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    condensed = squareform(rmsd_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    if n_clusters is None:
        # Automatic: cut dendrogram at rmsd_cutoff
        raw_labels = fcluster(Z, t=rmsd_cutoff, criterion="distance")
    else:
        raw_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # fcluster returns 1-based labels; convert to 0-based
    labels = raw_labels - 1
    detected = int(labels.max()) + 1
    return labels, detected


def _kmeans_cluster(
    positions: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, int]:
    """Apply k-means clustering on flattened ligand coordinate vectors."""
    from scipy.cluster.vq import kmeans2, whiten

    n_frames, n_atoms, _ = positions.shape
    flat = positions.reshape(n_frames, -1).astype(np.float64)
    # Whiten (unit variance) before k-means
    flat_w = whiten(flat)
    n_clusters = min(n_clusters, n_frames)
    _, labels = kmeans2(flat_w, n_clusters, minit="points", seed=42)
    return labels.astype(np.intp), int(labels.max()) + 1


def _find_medoids(
    rmsd_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> list[int]:
    """For each cluster, find the frame with minimum mean RMSD to cluster peers."""
    centers: list[int] = []
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            centers.append(0)
            continue
        if len(members) == 1:
            centers.append(int(members[0]))
            continue
        sub = rmsd_matrix[np.ix_(members, members)]
        mean_dists = sub.mean(axis=1)
        centers.append(int(members[np.argmin(mean_dists)]))
    return centers


def _build_summary(
    labels: np.ndarray,
    centers: list[int],
    n_clusters: int,
    rmsd_matrix: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build per-cluster summary DataFrame."""
    n_frames = len(labels)
    rows = []
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        center = centers[cid]
        if rmsd_matrix is not None and len(members) > 1:
            sub = rmsd_matrix[np.ix_(members, members)]
            mean_rmsd = float(sub[sub > 0].mean())
        else:
            mean_rmsd = 0.0
        rows.append(
            {
                "cluster_id": cid,
                "n_frames": len(members),
                "pct": round(len(members) / n_frames * 100, 2),
                "center_frame": center,
                "mean_rmsd": round(mean_rmsd, 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# DIVINE clusterer
# ---------------------------------------------------------------------------


class DIVINEClusterer:
    """Deterministic top-down divisive clustering for large MD trajectories.

    DIVINE (DIVIsive N-ary Ensembles) avoids the O(n²) pairwise RMSD matrix
    required by hierarchical methods.  Instead it recursively bisects the
    largest cluster using two pivot frames chosen by maximum distance, giving
    O(n log n) overall complexity.

    Compatible with :class:`PoseClusterer`: both return a :class:`ClusterResult`
    that can be passed to :meth:`extract_representatives`.

    Parameters
    ----------
    cfg:
        Analysis configuration (``ligand_resname`` is used for atom selection).
    n_clusters:
        Target number of clusters.  Actual count may be lower if the
        trajectory has fewer distinct frames.
    atom_selection:
        MDAnalysis selection string for atoms used in RMSD calculations.
        Defaults to ligand heavy atoms (``"resname <LIG> and not name H*"``).
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        n_clusters: int = 10,
        atom_selection: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.n_clusters = n_clusters
        self._atom_selection = atom_selection

    def _get_selection(self) -> str:
        if self._atom_selection is not None:
            return self._atom_selection
        return f"resname {self.cfg.ligand_resname} and not name H*"

    def run(self, u: mda.Universe, sample_name: str = "") -> ClusterResult:
        """Cluster trajectory frames using the DIVINE algorithm.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`ClusterResult`.
        """
        sel_str = self._get_selection()
        atoms = u.select_atoms(sel_str)
        if len(atoms) == 0:
            raise ValueError(
                f"Selection '{sel_str}' matched no atoms in the universe."
            )

        # Collect positions: (n_frames, n_atoms, 3)
        positions = np.array(
            [atoms.positions.copy() for _ in u.trajectory], dtype=np.float64
        )
        n_frames = len(positions)

        labels, n_clusters = _divine_cluster(positions, self.n_clusters)
        centers = _find_medoids_local(positions, labels, n_clusters)
        summary = _build_summary(labels, centers, n_clusters, rmsd_matrix=None)

        logger.info(
            "DIVINE clustering done for %s: %d frames → %d clusters",
            sample_name,
            n_frames,
            n_clusters,
        )
        return ClusterResult(
            sample_name=sample_name,
            labels=labels,
            centers=centers,
            rmsd_matrix=None,
            summary=summary,
            method="divine",
            n_clusters=n_clusters,
        )

    def extract_representatives(
        self,
        u: mda.Universe,
        result: ClusterResult,
        output_dir: Path | str,
    ) -> list[Path]:
        """Write one clean PDB per cluster representative frame.

        Parameters
        ----------
        u:
            Universe (aligned trajectory).
        result:
            Clustering result from :meth:`run`.
        output_dir:
            Directory where PDB files are saved.
        """
        from ..io.writers import write_clean_pdb

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for cluster_id, frame_idx in enumerate(result.centers):
            u.trajectory[frame_idx]
            out = output_dir / f"cluster{cluster_id:02d}_frame{frame_idx}.pdb"
            write_clean_pdb(u, out, ligand_resname=self.cfg.ligand_resname)
            paths.append(out)
        return paths


# ---------------------------------------------------------------------------
# DIVINE private helpers
# ---------------------------------------------------------------------------


def _batch_rmsd(positions_array: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Compute RMSD from each frame to a single reference structure.

    Parameters
    ----------
    positions_array:
        Shape ``(n_frames, n_atoms, 3)``.
    ref_pos:
        Shape ``(n_atoms, 3)``.

    Returns
    -------
    Array of shape ``(n_frames,)`` with RMSD values.
    """
    diff = positions_array - ref_pos[np.newaxis]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=2), axis=1))


def _divine_split(
    positions: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisect a cluster using two maximally distant pivot frames.

    Parameters
    ----------
    positions:
        Full trajectory positions, shape ``(n_frames_total, n_atoms, 3)``.
    frame_indices:
        Indices into *positions* belonging to this cluster.

    Returns
    -------
    Two arrays of frame indices (sub-cluster 1, sub-cluster 2).
    """
    cluster_pos = positions[frame_indices]  # (k, n_atoms, 3)

    # Pivot 1: first frame in the group (deterministic)
    pivot1_pos = cluster_pos[0]

    # Pivot 2: frame farthest from pivot 1
    dists = _batch_rmsd(cluster_pos, pivot1_pos)
    pivot2_local = int(np.argmax(dists))
    pivot2_pos = cluster_pos[pivot2_local]

    # Edge case: all frames identical → no split possible
    if dists[pivot2_local] == 0.0:
        half = len(frame_indices) // 2
        return frame_indices[:half], frame_indices[half:]

    # Assign each frame to nearest pivot
    dists_p1 = _batch_rmsd(cluster_pos, pivot1_pos)
    dists_p2 = _batch_rmsd(cluster_pos, pivot2_pos)

    mask = dists_p1 <= dists_p2
    group1 = frame_indices[mask]
    group2 = frame_indices[~mask]

    # Ensure non-empty groups (degenerate case)
    if len(group1) == 0:
        return frame_indices[:1], frame_indices[1:]
    if len(group2) == 0:
        return frame_indices[:-1], frame_indices[-1:]
    return group1, group2


def _divine_cluster(
    positions: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, int]:
    """Run the DIVINE top-down divisive clustering.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)``.
    n_clusters:
        Target number of clusters.

    Returns
    -------
    labels:
        Shape ``(n_frames,)`` with 0-based cluster IDs.
    n_detected:
        Actual number of clusters (≤ n_clusters).
    """
    n_frames = len(positions)
    n_clusters = min(n_clusters, n_frames)

    # Start with one cluster containing all frames
    clusters: list[np.ndarray] = [np.arange(n_frames, dtype=np.intp)]

    while len(clusters) < n_clusters:
        # Split the largest cluster
        largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        if len(clusters[largest_idx]) <= 1:
            break  # No cluster can be split further

        g1, g2 = _divine_split(positions, clusters[largest_idx])
        clusters.pop(largest_idx)
        clusters.append(g1)
        clusters.append(g2)

    # Build labels array
    labels = np.zeros(n_frames, dtype=np.intp)
    for cid, members in enumerate(clusters):
        labels[members] = cid

    return labels, len(clusters)


def _find_medoids_local(
    positions: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> list[int]:
    """Find the medoid frame for each cluster without a full pairwise matrix.

    For each cluster, compute mean RMSD of every member to all other members
    and return the frame with the minimum value.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)``.
    labels:
        Shape ``(n_frames,)`` with 0-based cluster IDs.
    n_clusters:
        Number of clusters.
    """
    centers: list[int] = []
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            centers.append(0)
        elif len(members) == 1:
            centers.append(int(members[0]))
        else:
            cluster_pos = positions[members]
            mean_rmsds = np.array(
                [_batch_rmsd(cluster_pos, cluster_pos[i]).mean() for i in range(len(members))]
            )
            centers.append(int(members[np.argmin(mean_rmsds)]))
    return centers
