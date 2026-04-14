"""Tests for ConvergenceAnalyzer and convergence plot functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.convergence import ConvergenceAnalyzer, ConvergenceResult
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    return AnalysisConfig()


@pytest.fixture()
def analyzer(cfg):
    return ConvergenceAnalyzer(cfg)


def _make_rmsd_df(values: np.ndarray) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "Frame": np.arange(n),
        "Time": np.arange(n, dtype=float) * 2.0,  # ps
        "Backbone": values,
        "Ligand": values + 0.5,
    })


def _stationary_series(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(3.0, 0.3, n)


def _drifting_series(n: int = 200, seed: int = 1) -> np.ndarray:
    """First half near 2.0, second half near 5.0 → not converged."""
    rng = np.random.default_rng(seed)
    first = rng.normal(2.0, 0.2, n // 2)
    second = rng.normal(5.0, 0.2, n - n // 2)
    return np.concatenate([first, second])


# ---------------------------------------------------------------------------
# rmsd_convergence
# ---------------------------------------------------------------------------


class TestRmsdConvergence:
    def test_converged_series(self, analyzer):
        vals = _stationary_series()
        df = _make_rmsd_df(vals)
        result = analyzer.rmsd_convergence(df)
        assert isinstance(result, ConvergenceResult)
        assert result.is_converged is True
        assert result.ks_pvalue >= 0.05

    def test_not_converged_series(self, analyzer):
        vals = _drifting_series()
        df = _make_rmsd_df(vals)
        result = analyzer.rmsd_convergence(df)
        assert result.is_converged is False
        assert result.ks_pvalue < 0.05

    def test_result_fields_finite(self, analyzer):
        df = _make_rmsd_df(_stationary_series())
        result = analyzer.rmsd_convergence(df)
        assert np.isfinite(result.first_half_mean)
        assert np.isfinite(result.second_half_mean)
        assert np.isfinite(result.ks_statistic)
        assert 0 <= result.ks_pvalue <= 1

    def test_recommendation_is_string(self, analyzer):
        df = _make_rmsd_df(_stationary_series())
        result = analyzer.rmsd_convergence(df)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0

    def test_missing_column_raises(self, analyzer):
        df = pd.DataFrame({"Frame": [0, 1], "Time": [0.0, 2.0]})
        with pytest.raises(ValueError, match="Column"):
            analyzer.rmsd_convergence(df, column="Backbone")

    def test_too_short_raises(self, analyzer):
        df = _make_rmsd_df(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="Too few frames"):
            analyzer.rmsd_convergence(df)


# ---------------------------------------------------------------------------
# block_error
# ---------------------------------------------------------------------------


class TestBlockError:
    def test_returns_dataframe(self, analyzer):
        series = _stationary_series()
        df = analyzer.block_error(series)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, analyzer):
        df = analyzer.block_error(_stationary_series())
        assert set(df.columns) == {"block_size", "mean", "sem", "n_blocks"}

    def test_sem_positive(self, analyzer):
        df = analyzer.block_error(_stationary_series())
        assert (df["sem"] > 0).all()

    def test_sem_finite(self, analyzer):
        df = analyzer.block_error(_stationary_series())
        assert np.all(np.isfinite(df["sem"]))

    def test_block_sizes_increasing(self, analyzer):
        df = analyzer.block_error(_stationary_series())
        assert (df["block_size"].diff().dropna() > 0).all()

    def test_too_short_raises(self, analyzer):
        with pytest.raises(ValueError, match="too short"):
            analyzer.block_error(np.array([1.0, 2.0]))

    def test_max_block_size_respected(self, analyzer):
        df = analyzer.block_error(_stationary_series(), max_block_size=10)
        assert df["block_size"].max() <= 10

    def test_pandas_series_input(self, analyzer):
        s = pd.Series(_stationary_series())
        df = analyzer.block_error(s)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# autocorrelation_time
# ---------------------------------------------------------------------------


class TestAutocorrelationTime:
    def test_returns_float(self, analyzer):
        tau = analyzer.autocorrelation_time(_stationary_series())
        assert isinstance(tau, float)

    def test_nonnegative(self, analyzer):
        tau = analyzer.autocorrelation_time(_stationary_series())
        assert tau >= 0.5

    def test_white_noise_small_tau(self, analyzer):
        rng = np.random.default_rng(42)
        white = rng.normal(0, 1, 500)
        tau = analyzer.autocorrelation_time(white)
        assert tau < 5.0  # white noise should have small τ

    def test_correlated_series_larger_tau(self, analyzer):
        """Highly autocorrelated series should have larger τ than white noise."""
        rng = np.random.default_rng(0)
        # AR(1) process with high correlation
        ar = np.zeros(500)
        ar[0] = rng.normal()
        for t in range(1, 500):
            ar[t] = 0.95 * ar[t - 1] + rng.normal(0, 0.1)
        tau_ar = analyzer.autocorrelation_time(ar)
        tau_white = analyzer.autocorrelation_time(rng.normal(0, 1, 500))
        assert tau_ar > tau_white

    def test_constant_series_returns_half(self, analyzer):
        """Constant series: FFT gives zero variance → return 0.5."""
        tau = analyzer.autocorrelation_time(np.ones(100))
        assert tau == 0.5


# ---------------------------------------------------------------------------
# replica_consistency
# ---------------------------------------------------------------------------


class TestReplicaConsistency:
    def _make_rmsd_result(self, values, name):
        """Minimal mock of RMSDResult."""
        class FakeRMSDResult:
            def __init__(self, vals, n):
                self.sample_name = n
                self.df = pd.DataFrame({
                    "Frame": range(len(vals)),
                    "Time": np.arange(len(vals), dtype=float),
                    "Backbone": vals,
                    "Ligand": vals + 0.5,
                })
        return FakeRMSDResult(values, name)

    def test_returns_dataframe(self, analyzer):
        r1 = self._make_rmsd_result(_stationary_series(seed=0), "rep1")
        r2 = self._make_rmsd_result(_stationary_series(seed=1), "rep2")
        df = analyzer.replica_consistency([r1, r2])
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, analyzer):
        r1 = self._make_rmsd_result(_stationary_series(seed=0), "rep1")
        r2 = self._make_rmsd_result(_stationary_series(seed=1), "rep2")
        df = analyzer.replica_consistency([r1, r2])
        assert set(df.columns) == {"replica_a", "replica_b", "ks_statistic", "wasserstein_distance"}

    def test_n_pairs(self, analyzer):
        """3 replicas → 3 pairs."""
        results = [
            self._make_rmsd_result(_stationary_series(seed=i), f"rep{i}")
            for i in range(3)
        ]
        df = analyzer.replica_consistency(results)
        assert len(df) == 3

    def test_ks_statistic_range(self, analyzer):
        r1 = self._make_rmsd_result(_stationary_series(seed=0), "rep1")
        r2 = self._make_rmsd_result(_stationary_series(seed=1), "rep2")
        df = analyzer.replica_consistency([r1, r2])
        assert (df["ks_statistic"] >= 0).all()
        assert (df["ks_statistic"] <= 1).all()

    def test_wasserstein_nonneg(self, analyzer):
        r1 = self._make_rmsd_result(_stationary_series(seed=0), "rep1")
        r2 = self._make_rmsd_result(_stationary_series(seed=1), "rep2")
        df = analyzer.replica_consistency([r1, r2])
        assert (df["wasserstein_distance"] >= 0).all()

    def test_ndarray_input(self, analyzer):
        a = _stationary_series(seed=0)
        b = _stationary_series(seed=1)
        df = analyzer.replica_consistency([a, b])
        assert len(df) == 1

    def test_single_replica_returns_empty(self, analyzer):
        r1 = self._make_rmsd_result(_stationary_series(), "rep1")
        df = analyzer.replica_consistency([r1])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


@pytest.fixture()
def convergence_result(analyzer):
    df = _make_rmsd_df(_stationary_series())
    return analyzer.rmsd_convergence(df), df


class TestPlotRmsdConvergence:
    def test_returns_figure(self, analyzer, convergence_result):
        from mdatools.plotting.convergence_plots import plot_rmsd_convergence
        result, df = convergence_result
        fig = plot_rmsd_convergence(result, df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, analyzer, convergence_result):
        from mdatools.plotting.convergence_plots import plot_rmsd_convergence
        result, df = convergence_result
        out = tmp_path / "convergence.png"
        fig = plot_rmsd_convergence(result, df, save_path=out)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)


class TestPlotBlockError:
    def test_returns_figure(self, analyzer):
        from mdatools.plotting.convergence_plots import plot_block_error
        df = analyzer.block_error(_stationary_series())
        fig = plot_block_error(df, observable_name="Backbone RMSD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_returns_figure(self):
        from mdatools.plotting.convergence_plots import plot_block_error
        fig = plot_block_error(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, analyzer):
        from mdatools.plotting.convergence_plots import plot_block_error
        df = analyzer.block_error(_stationary_series())
        out = tmp_path / "block_error.png"
        fig = plot_block_error(df, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotReplicaOverlap:
    def _overlap_df(self, analyzer):
        a = _stationary_series(seed=0)
        b = _stationary_series(seed=1)
        return analyzer.replica_consistency([a, b])

    def test_returns_figure(self, analyzer):
        from mdatools.plotting.convergence_plots import plot_replica_overlap
        df = self._overlap_df(analyzer)
        fig = plot_replica_overlap(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_returns_figure(self):
        from mdatools.plotting.convergence_plots import plot_replica_overlap
        fig = plot_replica_overlap(pd.DataFrame(
            columns=["replica_a", "replica_b", "ks_statistic", "wasserstein_distance"]
        ))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, analyzer):
        from mdatools.plotting.convergence_plots import plot_replica_overlap
        df = self._overlap_df(analyzer)
        out = tmp_path / "overlap.png"
        fig = plot_replica_overlap(df, save_path=out)
        assert out.exists()
        plt.close(fig)
