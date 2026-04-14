"""Tests for PoseClusterer and clustering plot functions."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.clustering import DIVINEClusterer, PoseClusterer, ClusterResult
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    return AnalysisConfig(ligand_resname="UNK")


@pytest.fixture()
def cluster_result(synthetic_universe, cfg):
    clusterer = PoseClusterer(cfg, method="ward", rmsd_cutoff=10.0)
    return clusterer.run(synthetic_universe, "test")


# ---------------------------------------------------------------------------
# ClusterResult structure
# ---------------------------------------------------------------------------


class TestClusterResult:
    def test_labels_shape(self, synthetic_universe, cluster_result):
        n_frames = len(synthetic_universe.trajectory)
        assert cluster_result.labels.shape == (n_frames,)

    def test_labels_values_in_range(self, cluster_result):
        assert cluster_result.labels.min() >= 0
        assert cluster_result.labels.max() < cluster_result.n_clusters

    def test_rmsd_matrix_symmetric(self, cluster_result):
        m = cluster_result.rmsd_matrix
        assert m.shape[0] == m.shape[1]
        np.testing.assert_allclose(m, m.T, atol=1e-10)

    def test_rmsd_matrix_diagonal_zero(self, cluster_result):
        np.testing.assert_allclose(np.diag(cluster_result.rmsd_matrix), 0.0, atol=1e-10)

    def test_rmsd_matrix_nonnegative(self, cluster_result):
        assert (cluster_result.rmsd_matrix >= 0).all()

    def test_centers_count(self, cluster_result):
        assert len(cluster_result.centers) == cluster_result.n_clusters

    def test_centers_valid_frame_indices(self, synthetic_universe, cluster_result):
        n_frames = len(synthetic_universe.trajectory)
        for c in cluster_result.centers:
            assert 0 <= c < n_frames

    def test_summary_columns(self, cluster_result):
        expected = {"cluster_id", "n_frames", "pct", "center_frame", "mean_rmsd"}
        assert set(cluster_result.summary.columns) == expected

    def test_summary_pct_sums_to_100(self, cluster_result):
        total = cluster_result.summary["pct"].sum()
        assert abs(total - 100.0) < 0.5  # rounding tolerance

    def test_sample_name_stored(self, synthetic_universe, cfg):
        clusterer = PoseClusterer(cfg, rmsd_cutoff=10.0)
        result = clusterer.run(synthetic_universe, "replica_03")
        assert result.sample_name == "replica_03"


# ---------------------------------------------------------------------------
# n_clusters parameter
# ---------------------------------------------------------------------------


class TestNClusters:
    def test_fixed_n_clusters(self, synthetic_universe, cfg):
        """When n_clusters=2, result should have exactly 2 clusters."""
        clusterer = PoseClusterer(cfg, method="ward", n_clusters=2)
        result = clusterer.run(synthetic_universe)
        assert result.n_clusters == 2

    def test_fixed_n_clusters_kmeans(self, synthetic_universe, cfg):
        clusterer = PoseClusterer(cfg, method="kmeans", n_clusters=3)
        result = clusterer.run(synthetic_universe)
        assert result.n_clusters == 3

    def test_auto_n_clusters_returns_at_least_one(self, synthetic_universe, cfg):
        clusterer = PoseClusterer(cfg, rmsd_cutoff=100.0)  # large cutoff → 1 cluster
        result = clusterer.run(synthetic_universe)
        assert result.n_clusters >= 1

    def test_invalid_method_raises(self, cfg):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            PoseClusterer(cfg, method="dbscan")


# ---------------------------------------------------------------------------
# extract_representatives
# ---------------------------------------------------------------------------


class TestExtractRepresentatives:
    def test_creates_pdb_per_cluster(self, tmp_path, synthetic_universe, cfg):
        clusterer = PoseClusterer(cfg, method="ward", n_clusters=2)
        result = clusterer.run(synthetic_universe)
        paths = clusterer.extract_representatives(synthetic_universe, result, tmp_path)
        assert len(paths) == result.n_clusters

    def test_pdb_files_exist(self, tmp_path, synthetic_universe, cfg):
        clusterer = PoseClusterer(cfg, method="ward", n_clusters=2)
        result = clusterer.run(synthetic_universe)
        paths = clusterer.extract_representatives(synthetic_universe, result, tmp_path)
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_creates_output_dir(self, tmp_path, synthetic_universe, cfg):
        output_dir = tmp_path / "clusters" / "run01"
        clusterer = PoseClusterer(cfg, method="ward", n_clusters=2)
        result = clusterer.run(synthetic_universe)
        paths = clusterer.extract_representatives(synthetic_universe, result, output_dir)
        assert output_dir.exists()
        assert len(paths) == result.n_clusters


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestPlotClusterTimeline:
    def test_returns_figure(self, cluster_result):
        from mdatools.plotting.clustering_plots import plot_cluster_timeline
        fig = plot_cluster_timeline(cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, cluster_result):
        from mdatools.plotting.clustering_plots import plot_cluster_timeline
        out = tmp_path / "timeline.png"
        fig = plot_cluster_timeline(cluster_result, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotClusterPopulation:
    def test_returns_figure(self, cluster_result):
        from mdatools.plotting.clustering_plots import plot_cluster_population
        fig = plot_cluster_population(cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, cluster_result):
        from mdatools.plotting.clustering_plots import plot_cluster_population
        out = tmp_path / "population.png"
        fig = plot_cluster_population(cluster_result, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotRmsdMatrix:
    def test_returns_figure(self, cluster_result):
        from mdatools.plotting.clustering_plots import plot_rmsd_matrix
        fig = plot_rmsd_matrix(cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, cluster_result):
        from mdatools.plotting.clustering_plots import plot_rmsd_matrix
        out = tmp_path / "rmsd_matrix.png"
        fig = plot_rmsd_matrix(cluster_result, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotDendrogram:
    def test_returns_figure_for_ward(self, cluster_result):
        from mdatools.plotting.clustering_plots import plot_dendrogram
        fig = plot_dendrogram(cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_for_kmeans(self, synthetic_universe, cfg):
        from mdatools.plotting.clustering_plots import plot_dendrogram
        clusterer = PoseClusterer(cfg, method="kmeans", n_clusters=2)
        result = clusterer.run(synthetic_universe)
        fig = plot_dendrogram(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, cluster_result):
        from mdatools.plotting.clustering_plots import plot_dendrogram
        out = tmp_path / "dendrogram.png"
        fig = plot_dendrogram(cluster_result, save_path=out)
        assert out.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# DIVINEClusterer
# ---------------------------------------------------------------------------


@pytest.fixture()
def divine_result(synthetic_universe, cfg):
    clusterer = DIVINEClusterer(cfg, n_clusters=3)
    return clusterer.run(synthetic_universe, "divine_test")


class TestDIVINEClusterer:
    def test_returns_cluster_result(self, divine_result):
        assert isinstance(divine_result, ClusterResult)

    def test_method_is_divine(self, divine_result):
        assert divine_result.method == "divine"

    def test_n_clusters_matches_request(self, synthetic_universe, cfg):
        clusterer = DIVINEClusterer(cfg, n_clusters=4)
        result = clusterer.run(synthetic_universe)
        assert result.n_clusters == 4

    def test_labels_shape(self, synthetic_universe, divine_result):
        n_frames = len(synthetic_universe.trajectory)
        assert divine_result.labels.shape == (n_frames,)

    def test_labels_values_in_range(self, divine_result):
        assert divine_result.labels.min() >= 0
        assert divine_result.labels.max() < divine_result.n_clusters

    def test_all_frames_assigned(self, synthetic_universe, divine_result):
        n_frames = len(synthetic_universe.trajectory)
        assert len(divine_result.labels) == n_frames

    def test_centers_count(self, divine_result):
        assert len(divine_result.centers) == divine_result.n_clusters

    def test_centers_are_valid_frame_indices(self, synthetic_universe, divine_result):
        n_frames = len(synthetic_universe.trajectory)
        for c in divine_result.centers:
            assert 0 <= c < n_frames

    def test_centers_belong_to_their_cluster(self, divine_result):
        for cid, center in enumerate(divine_result.centers):
            assert divine_result.labels[center] == cid

    def test_rmsd_matrix_is_none(self, divine_result):
        assert divine_result.rmsd_matrix is None

    def test_summary_columns(self, divine_result):
        expected = {"cluster_id", "n_frames", "pct", "center_frame", "mean_rmsd"}
        assert set(divine_result.summary.columns) == expected

    def test_summary_pct_sums_to_100(self, divine_result):
        total = divine_result.summary["pct"].sum()
        assert abs(total - 100.0) < 0.5

    def test_n_clusters_capped_at_n_frames(self, synthetic_universe, cfg):
        n_frames = len(synthetic_universe.trajectory)
        clusterer = DIVINEClusterer(cfg, n_clusters=n_frames + 100)
        result = clusterer.run(synthetic_universe)
        assert result.n_clusters <= n_frames

    def test_invalid_selection_raises(self, synthetic_universe, cfg):
        clusterer = DIVINEClusterer(cfg, atom_selection="resname NONEXISTENT")
        with pytest.raises(ValueError, match="matched no atoms"):
            clusterer.run(synthetic_universe)

    def test_sample_name_stored(self, synthetic_universe, cfg):
        clusterer = DIVINEClusterer(cfg, n_clusters=2)
        result = clusterer.run(synthetic_universe, "replica_07")
        assert result.sample_name == "replica_07"

    def test_extract_representatives_creates_pdbs(self, tmp_path, synthetic_universe, cfg):
        clusterer = DIVINEClusterer(cfg, n_clusters=3)
        result = clusterer.run(synthetic_universe)
        paths = clusterer.extract_representatives(synthetic_universe, result, tmp_path)
        assert len(paths) == result.n_clusters
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0


class TestDIVINEPlots:
    def test_rmsd_matrix_returns_note_figure(self, divine_result):
        from mdatools.plotting.clustering_plots import plot_rmsd_matrix
        fig = plot_rmsd_matrix(divine_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_returns_note_figure(self, divine_result):
        from mdatools.plotting.clustering_plots import plot_dendrogram
        fig = plot_dendrogram(divine_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cluster_timeline_works(self, divine_result):
        from mdatools.plotting.clustering_plots import plot_cluster_timeline
        fig = plot_cluster_timeline(divine_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
