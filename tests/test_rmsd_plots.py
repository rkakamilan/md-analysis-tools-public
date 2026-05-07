"""Tests for RMSD plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mdatools.plotting.rmsd_plots import plot_rmsd_grid, plot_rmsd_single


@dataclass
class _FakeRMSDResult:
    sample_name: str
    df: pd.DataFrame


def _make_result(sample_name: str, n_frames: int = 100) -> _FakeRMSDResult:
    rng = np.random.default_rng(seed=hash(sample_name) % (2**32))
    df = pd.DataFrame(
        {
            "Frame": np.arange(n_frames),
            "Time": np.arange(n_frames) * 10.0,  # ps
            "Backbone": rng.uniform(0.5, 3.0, n_frames),
            "Ligand": rng.uniform(0.5, 5.0, n_frames),
        }
    )
    return _FakeRMSDResult(sample_name=sample_name, df=df)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotRmsdSingle:
    def test_returns_axes_and_plots_two_lines(self):
        result = _make_result("rep_1")
        ax = plot_rmsd_single(result.df, result.sample_name)
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 2  # Backbone + Ligand
        assert ax.get_title() == "rep_1"


class TestPlotRmsdGrid:
    def test_single_replica_default_ncols_does_not_crash(self, tmp_path: Path):
        """Regression test for issue #114.

        n_samples=1 with default n_cols=3 used to raise
        AttributeError: 'numpy.ndarray' object has no attribute 'plot'
        because the previous fallback `[axes]` wrapped the ndarray
        returned by `plt.subplots(1, 3)` instead of the first Axes.
        """
        results = [_make_result("only_replica")]
        save_path = tmp_path / "rmsd.png"

        fig = plot_rmsd_grid(results, n_cols=3, save_path=save_path)

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        # 1 sample plotted, remaining 2 axes turned off
        plotted = [ax for ax in fig.axes if ax.lines]
        assert len(plotted) == 1
        assert plotted[0].get_title() == "only_replica"

    def test_single_replica_ncols_1(self, tmp_path: Path):
        results = [_make_result("only_replica")]
        fig = plot_rmsd_grid(results, n_cols=1, save_path=tmp_path / "out.png")
        plotted = [ax for ax in fig.axes if ax.lines]
        assert len(plotted) == 1

    def test_multi_replica_grid_shape(self, tmp_path: Path):
        results = [_make_result(f"rep_{i}") for i in range(4)]
        fig = plot_rmsd_grid(results, n_cols=3, save_path=tmp_path / "out.png")
        # 4 samples plotted, 2 axes (in a 2x3 grid) turned off
        plotted = [ax for ax in fig.axes if ax.lines]
        assert len(plotted) == 4
        assert {ax.get_title() for ax in plotted} == {f"rep_{i}" for i in range(4)}

    def test_dict_input_supported(self, tmp_path: Path):
        results_dict = {f"rep_{i}": _make_result(f"rep_{i}") for i in range(2)}
        fig = plot_rmsd_grid(results_dict, n_cols=2, save_path=tmp_path / "out.png")
        plotted = [ax for ax in fig.axes if ax.lines]
        assert len(plotted) == 2
