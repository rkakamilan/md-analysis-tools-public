"""Chemical diversity clustering for multi-compound virtual screening.

Clusters compounds by 2D molecular fingerprint similarity using either
KMeans or RDKit Butina algorithm.  Supports 5 fingerprint types.


Example::

    fp_matrix = compute_fp_matrix(mols, fingerprint="morgan")
    result = cluster_by_kmeans(mols, n_clusters=10, fingerprint="morgan")
    result = cluster_by_butina(mols, threshold=0.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.DataStructs import ExplicitBitVect


@dataclass
class ChemicalClusteringResult:
    """Output of chemical diversity clustering.

    Attributes
    ----------
    cluster_ids:
        1-D integer array of cluster assignments, parallel to input mols.
    n_clusters:
        Number of unique clusters.
    fp_matrix:
        Fingerprint bit matrix ``(n_mols × n_bits)``.
    method:
        Clustering method used (``"kmeans"`` or ``"butina"``).
    fingerprint:
        Fingerprint type used.

    Notes
    -----
    Supports array-like protocol so ``df["cluster"] = result`` works
    without unpacking ``.cluster_ids`` manually.
    """

    cluster_ids: np.ndarray
    n_clusters: int
    fp_matrix: np.ndarray
    method: str
    fingerprint: str

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self.cluster_ids, dtype=dtype)

    def __iter__(self):
        return iter(self.cluster_ids)

    def __len__(self) -> int:
        return len(self.cluster_ids)

    def __getitem__(self, idx: Any) -> Any:
        return self.cluster_ids[idx]


def _fp_to_array(fp: ExplicitBitVect) -> np.ndarray:
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def compute_fp_matrix(
    mols: list[Chem.Mol],
    fingerprint: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute a fingerprint bit matrix for a list of molecules.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    fingerprint:
        One of ``"morgan"``, ``"rdkit"``, ``"maccs"``,
        ``"topological_torsion"``, ``"atom_pair"``.
    radius:
        Morgan fingerprint radius (default 2).
    n_bits:
        Bit vector length for variable-length fingerprints.

    Returns
    -------
    np.ndarray
        Shape ``(n_mols, n_bits)``.
    """
    valid = {"morgan", "rdkit", "maccs", "topological_torsion", "atom_pair"}
    if fingerprint not in valid:
        raise ValueError(
            f"Unknown fingerprint {fingerprint!r}. Available: {sorted(valid)}"
        )

    fps = []
    for mol in mols:
        if fingerprint == "morgan":
            gen = rdFPGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = gen.GetFingerprint(mol)
        elif fingerprint == "rdkit":
            gen = rdFPGen.GetRDKitFPGenerator(fpSize=n_bits)
            fp = gen.GetFingerprint(mol)
        elif fingerprint == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fingerprint == "topological_torsion":
            gen = rdFPGen.GetTopologicalTorsionGenerator(fpSize=n_bits)
            fp = gen.GetFingerprint(mol)
        else:  # atom_pair
            gen = rdFPGen.GetAtomPairGenerator(fpSize=n_bits)
            fp = gen.GetFingerprint(mol)
        fps.append(_fp_to_array(fp))

    return np.array(fps)


def cluster_by_kmeans(
    mols: list[Chem.Mol],
    n_clusters: int = 10,
    fingerprint: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    random_seed: int = 42,
) -> ChemicalClusteringResult:
    """Cluster compounds using KMeans on molecular fingerprints.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    n_clusters:
        Number of clusters.
    fingerprint:
        Fingerprint type (see :func:`compute_fp_matrix`).
    radius:
        Morgan FP radius.
    n_bits:
        Bit vector length.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    ChemicalClusteringResult
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    fp_matrix = compute_fp_matrix(mols, fingerprint, radius, n_bits)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_seed)
    cluster_ids = km.fit_predict(fp_matrix)

    return ChemicalClusteringResult(
        cluster_ids=cluster_ids,
        n_clusters=int(np.unique(cluster_ids).size),
        fp_matrix=fp_matrix,
        method="kmeans",
        fingerprint=fingerprint,
    )


def cluster_by_butina(
    mols: list[Chem.Mol],
    threshold: float = 0.4,
    fingerprint: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> ChemicalClusteringResult:
    """Cluster compounds using the RDKit Butina (Taylor-Butina) algorithm.

    No scikit-learn dependency required.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    threshold:
        Tanimoto *distance* threshold (0 = identical, 1 = completely different).
        Typical values: 0.3–0.5.
    fingerprint:
        Fingerprint type (see :func:`compute_fp_matrix`).
    radius:
        Morgan FP radius.
    n_bits:
        Bit vector length.

    Returns
    -------
    ChemicalClusteringResult
    """
    from rdkit import DataStructs  # noqa: PLC0415
    from rdkit.ML.Cluster import Butina  # noqa: PLC0415

    fp_matrix = compute_fp_matrix(mols, fingerprint, radius, n_bits)
    n = len(fp_matrix)

    # Build RDKit ExplicitBitVect objects for Tanimoto calculation
    rdkit_fps = []
    for row in fp_matrix:
        bv = ExplicitBitVect(len(row))
        for idx in np.where(row > 0)[0]:
            bv.SetBit(int(idx))
        rdkit_fps.append(bv)

    # Condensed Tanimoto distance matrix (lower triangle)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, threshold, isDistData=True)

    cluster_ids = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        for member in members:
            cluster_ids[member] = cid

    return ChemicalClusteringResult(
        cluster_ids=cluster_ids,
        n_clusters=len(clusters),
        fp_matrix=fp_matrix,
        method="butina",
        fingerprint=fingerprint,
    )
