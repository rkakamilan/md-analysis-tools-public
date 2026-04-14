"""Tests for TrajectoryPCA, TrajectoryTICA and dimensionality plot functions."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.dimensionality import (
    TrajectoryPCA,
    TrajectoryTICA,
    DimRedResult,
    _pca_numpy,
    _tica_numpy,
)
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    return AnalysisConfig(ligand_resname="UNK")


@pytest.fixture()
def pca_result(synthetic_universe, cfg):
    pca = TrajectoryPCA(cfg, n_components=3, selection="name CA")
    return pca.fit_transform(synthetic_universe, "test")


@pytest.fixture()
def tica_result(synthetic_universe, cfg):
    tica = TrajectoryTICA(cfg, lag=2, n_components=2, selection="name CA")
    return tica.fit_transform(synthetic_universe, "test")


# ---------------------------------------------------------------------------
# DimRedResult dataclass
# ---------------------------------------------------------------------------


class TestDimRedResult:
    def test_method_field_pca(self, pca_result):
        assert pca_result.method == "pca"

    def test_method_field_tica(self, tica_result):
        assert tica_result.method == "tica"

    def test_sample_name(self, pca_result):
        assert pca_result.sample_name == "test"


# ---------------------------------------------------------------------------
# TrajectoryPCA
# ---------------------------------------------------------------------------


class TestTrajectoryPCA:
    def test_projection_shape(self, synthetic_universe, pca_result):
        n_frames = len(synthetic_universe.trajectory)
        assert pca_result.projection.shape == (n_frames, 3)

    def test_explained_variance_ratio_length(self, pca_result):
        assert len(pca_result.explained_variance_ratio) == 3

    def test_explained_variance_ratio_nonnegative(self, pca_result):
        assert (pca_result.explained_variance_ratio >= 0).all()

    def test_explained_variance_ratio_sum_leq_one(self, pca_result):
        assert pca_result.explained_variance_ratio.sum() <= 1.0 + 1e-9

    def test_components_shape(self, synthetic_universe, pca_result, cfg):
        ag = synthetic_universe.select_atoms("name CA")
        n_features = len(ag) * 3
        assert pca_result.components.shape == (3, n_features)

    def test_feature_mean_shape(self, synthetic_universe, pca_result, cfg):
        ag = synthetic_universe.select_atoms("name CA")
        assert pca_result.feature_mean.shape == (len(ag) * 3,)

    def test_invalid_selection_raises(self, synthetic_universe, cfg):
        pca = TrajectoryPCA(cfg, selection="name NOTEXIST")
        with pytest.raises(ValueError, match="returned no atoms"):
            pca.fit_transform(synthetic_universe)

    def test_fewer_components_than_requested(self, synthetic_universe, cfg):
        # Request more components than n_frames → should clamp silently
        n_frames = len(synthetic_universe.trajectory)
        pca = TrajectoryPCA(cfg, n_components=n_frames + 100, selection="name CA")
        result = pca.fit_transform(synthetic_universe)
        # projection dim ≤ n_frames
        assert result.projection.shape[0] == n_frames
        assert result.projection.shape[1] <= n_frames

    def test_projection_variance_decreasing(self, pca_result):
        evr = pca_result.explained_variance_ratio
        # PCA components should be sorted by descending variance
        assert evr[0] >= evr[1] - 1e-9


# ---------------------------------------------------------------------------
# TrajectoryTICA
# ---------------------------------------------------------------------------


class TestTrajectoryTICA:
    def test_projection_shape(self, synthetic_universe, tica_result):
        n_frames = len(synthetic_universe.trajectory)
        # TICA removes `lag` frames
        lag = 2
        assert tica_result.projection.shape == (n_frames - lag, 2)

    def test_components_shape(self, synthetic_universe, tica_result, cfg):
        ag = synthetic_universe.select_atoms("name CA")
        n_features = len(ag) * 3
        assert tica_result.components.shape == (2, n_features)

    def test_explained_variance_ratio_empty(self, tica_result):
        assert len(tica_result.explained_variance_ratio) == 0

    def test_invalid_selection_raises(self, synthetic_universe, cfg):
        tica = TrajectoryTICA(cfg, lag=1, selection="name NOTEXIST")
        with pytest.raises(ValueError, match="returned no atoms"):
            tica.fit_transform(synthetic_universe)

    def test_lag_zero_raises(self, cfg):
        with pytest.raises(ValueError, match="lag must be >= 1"):
            TrajectoryTICA(cfg, lag=0)

    def test_lag_too_large_returns_empty(self, synthetic_universe, cfg):
        """lag >= n_frames should return an empty DimRedResult, not raise."""
        n_frames = len(synthetic_universe.trajectory)
        tica = TrajectoryTICA(cfg, lag=n_frames, selection="name CA")
        result = tica.fit_transform(synthetic_universe, "test_empty")
        assert result.method == "tica"
        assert result.projection.shape[0] == 0

    def test_lag_equal_n_frames_minus_one_still_empty(self, synthetic_universe, cfg):
        """lag == n_frames - 1 leaves 1 pair; result should still be non-empty."""
        n_frames = len(synthetic_universe.trajectory)
        tica = TrajectoryTICA(cfg, lag=n_frames - 1, n_components=1, selection="name CA")
        result = tica.fit_transform(synthetic_universe, "test_one_pair")
        # n_frames - lag = 1 frame pair → projection has 1 row
        assert result.projection.shape[0] == 1

    def test_projection_finite(self, tica_result):
        assert np.isfinite(tica_result.projection).all()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestPcaNumpy:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 10))
        proj, evr, comps, mean = _pca_numpy(X, n_components=3)
        assert proj.shape == (30, 3)
        assert evr.shape == (3,)
        assert comps.shape == (3, 10)
        assert mean.shape == (10,)

    def test_evr_sum_leq_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 20))
        _, evr, _, _ = _pca_numpy(X, n_components=5)
        assert evr.sum() <= 1.0 + 1e-9

    def test_constant_data_evr_zero(self):
        X = np.ones((20, 5))
        _, evr, _, _ = _pca_numpy(X, n_components=2)
        assert (evr == 0).all()


class TestTicaNumpy:
    def test_output_shapes(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 8))
        lag = 3
        proj, comps, mean = _tica_numpy(X, lag=lag, n_components=2)
        assert proj.shape == (40 - lag, 2)
        assert comps.shape == (2, 8)
        assert mean.shape == (8,)

    def test_projection_finite(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 6))
        proj, _, _ = _tica_numpy(X, lag=5, n_components=2)
        assert np.isfinite(proj).all()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestDimensionalityPlots:
    def test_plot_pca_landscape(self, pca_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_landscape
        fig = plot_pca_landscape(pca_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pca_landscape_with_color(self, synthetic_universe, pca_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_landscape
        n_frames = len(synthetic_universe.trajectory)
        color = np.arange(n_frames)
        fig = plot_pca_landscape(pca_result, color_by=color)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_explained_variance(self, pca_result):
        from mdatools.plotting.dimensionality_plots import plot_explained_variance
        fig = plot_explained_variance(pca_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_explained_variance_raises_for_tica(self, tica_result):
        from mdatools.plotting.dimensionality_plots import plot_explained_variance
        with pytest.raises(ValueError, match="empty"):
            plot_explained_variance(tica_result)

    def test_plot_pca_projection_time(self, pca_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_projection_time
        fig = plot_pca_projection_time(pca_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pca_projection_time_tica(self, tica_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_projection_time
        fig = plot_pca_projection_time(tica_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_landscape_axis_labels_pca(self, pca_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_landscape
        fig = plot_pca_landscape(pca_result)
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_landscape_axis_labels_tica(self, tica_result):
        from mdatools.plotting.dimensionality_plots import plot_pca_landscape
        fig = plot_pca_landscape(tica_result)
        ax = fig.axes[0]
        assert "IC1" in ax.get_xlabel()
        plt.close(fig)

    def test_plot_pca_landscape_one_component_raises(self, synthetic_universe, cfg):
        pca = TrajectoryPCA(cfg, n_components=1, selection="name CA")
        result = pca.fit_transform(synthetic_universe)
        from mdatools.plotting.dimensionality_plots import plot_pca_landscape
        with pytest.raises(ValueError, match="at least 2 components"):
            plot_pca_landscape(result)
