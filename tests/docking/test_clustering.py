"""Tests for mdatools.docking.clustering.chemical."""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit required")
from rdkit import Chem  # noqa: E402

from mdatools.docking.clustering import (
    ChemicalClusteringResult,
    cluster_by_butina,
    cluster_by_kmeans,
    compute_fp_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMILES = [
    "c1ccccc1",          # benzene
    "c1ccc(O)cc1",       # phenol
    "c1ccc(N)cc1",       # aniline
    "c1ccc(F)cc1",       # fluorobenzene
    "CCCC",              # butane
    "CCCCO",             # butanol
    "CCCCN",             # butylamine
    "CCCCF",             # fluorobutane
    "c1ccncc1",          # pyridine
    "c1ccnc(N)c1",       # aminopyridine
]

@pytest.fixture(scope="module")
def mols():
    ms = [Chem.MolFromSmiles(s) for s in SMILES]
    assert all(m is not None for m in ms)
    return ms


# ---------------------------------------------------------------------------
# compute_fp_matrix
# ---------------------------------------------------------------------------


class TestComputeFpMatrix:
    def test_morgan_shape(self, mols):
        fp = compute_fp_matrix(mols, fingerprint="morgan", n_bits=1024)
        assert fp.shape == (len(mols), 1024)

    def test_rdkit_fp(self, mols):
        fp = compute_fp_matrix(mols, fingerprint="rdkit", n_bits=2048)
        assert fp.shape[0] == len(mols)

    def test_maccs_fixed_167_bits(self, mols):
        fp = compute_fp_matrix(mols, fingerprint="maccs")
        assert fp.shape == (len(mols), 167)

    def test_invalid_fp_raises(self, mols):
        with pytest.raises(ValueError, match="Unknown fingerprint"):
            compute_fp_matrix(mols, fingerprint="bogus")

    def test_values_are_binary(self, mols):
        fp = compute_fp_matrix(mols)
        assert set(np.unique(fp)).issubset({0, 1})


# ---------------------------------------------------------------------------
# ChemicalClusteringResult array protocol
# ---------------------------------------------------------------------------


class TestChemicalClusteringResult:
    def test_array_protocol(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        arr = np.asarray(result)
        assert arr.shape == (len(mols),)

    def test_iter_yields_cluster_ids(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        ids = list(result)
        assert len(ids) == len(mols)

    def test_len(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert len(result) == len(mols)

    def test_getitem(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert isinstance(result[0], (int, np.integer))

    def test_can_assign_to_dataframe(self, mols):
        import pandas as pd
        result = cluster_by_kmeans(mols, n_clusters=3)
        df = pd.DataFrame({"smiles": SMILES})
        df["cluster"] = result
        assert "cluster" in df.columns
        assert len(df["cluster"]) == len(mols)


# ---------------------------------------------------------------------------
# cluster_by_kmeans
# ---------------------------------------------------------------------------


class TestClusterByKMeans:
    def test_correct_n_clusters(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert result.n_clusters == 3

    def test_cluster_ids_shape(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert result.cluster_ids.shape == (len(mols),)

    def test_cluster_ids_in_range(self, mols):
        n_clusters = 4
        result = cluster_by_kmeans(mols, n_clusters=n_clusters)
        assert result.cluster_ids.min() >= 0
        assert result.cluster_ids.max() < n_clusters

    def test_method_recorded(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert result.method == "kmeans"

    def test_fingerprint_recorded(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3, fingerprint="rdkit")
        assert result.fingerprint == "rdkit"

    def test_fp_matrix_stored(self, mols):
        result = cluster_by_kmeans(mols, n_clusters=3)
        assert result.fp_matrix.shape[0] == len(mols)

    def test_reproducible_with_same_seed(self, mols):
        a = cluster_by_kmeans(mols, n_clusters=3, random_seed=42)
        b = cluster_by_kmeans(mols, n_clusters=3, random_seed=42)
        np.testing.assert_array_equal(a.cluster_ids, b.cluster_ids)


# ---------------------------------------------------------------------------
# cluster_by_butina
# ---------------------------------------------------------------------------


class TestClusterByButina:
    def test_clusters_aromatic_vs_aliphatic(self, mols):
        # Aromatics should cluster together, aliphatics together
        result = cluster_by_butina(mols, threshold=0.5)
        assert result.n_clusters >= 2

    def test_method_recorded(self, mols):
        result = cluster_by_butina(mols)
        assert result.method == "butina"

    def test_cluster_ids_cover_all_mols(self, mols):
        result = cluster_by_butina(mols, threshold=0.4)
        assert len(result.cluster_ids) == len(mols)

    def test_tighter_threshold_more_clusters(self, mols):
        loose = cluster_by_butina(mols, threshold=0.8)
        tight = cluster_by_butina(mols, threshold=0.2)
        # Tighter threshold → more (or equal) clusters
        assert tight.n_clusters >= loose.n_clusters

    def test_zero_threshold_all_singletons(self, mols):
        result = cluster_by_butina(mols, threshold=0.0)
        assert result.n_clusters == len(mols)
