"""Tests for InteractionAnalyzer and interaction plot functions.

ProLIF is an optional dependency and may not be installed in CI.
All tests inject a mock prolif module via sys.modules so that no
actual ProLIF installation is required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.interactions import (
    InteractionAnalyzer,
    InteractionResult,
    _DEFAULT_INTERACTIONS,
)
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Helpers — build a fake ProLIF module and fake fingerprint DataFrame
# ---------------------------------------------------------------------------


def _make_fake_prolif(n_frames: int = 10) -> types.ModuleType:
    """Return a mock prolif module that mimics the real ProLIF API."""
    prolif_mod = types.ModuleType("prolif")

    class FakeResidue:
        def __init__(self, resname, chain, resid):
            self.resname = resname
            self.chain = chain
            self.resid = resid

    # Build a MultiIndex DataFrame similar to prolif.Fingerprint.to_dataframe()
    lig = FakeResidue("UNK", "A", 0)
    interactions_data = {
        (lig, FakeResidue("ASN", "A", 58), "HBAcceptor"): [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        (lig, FakeResidue("SER", "A", 35), "HBDonor"):    [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        (lig, FakeResidue("PHE", "A", 72), "Hydrophobic"): [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    }
    raw_df = pd.DataFrame(
        {k: v[:n_frames] for k, v in interactions_data.items()}
    )

    class FakeFingerprint:
        def __init__(self, interactions=None, count=False):
            self.interactions = interactions
            self.count = count

        def run(self, trajectory, ligand_ag, protein_ag):
            self._df = raw_df

        def to_dataframe(self):
            return self._df

    prolif_mod.Fingerprint = FakeFingerprint
    prolif_mod.Residue = FakeResidue
    return prolif_mod


@pytest.fixture()
def fake_prolif():
    """Inject fake prolif into sys.modules for the duration of the test."""
    mod = _make_fake_prolif()
    with patch.dict(sys.modules, {"prolif": mod}):
        yield mod


@pytest.fixture()
def cfg():
    return AnalysisConfig(ligand_resname="UNK")


@pytest.fixture()
def interaction_result(synthetic_universe, cfg, fake_prolif):
    analyzer = InteractionAnalyzer(cfg)
    return analyzer.run(synthetic_universe, "test")


# ---------------------------------------------------------------------------
# InteractionResult structure
# ---------------------------------------------------------------------------


class TestInteractionResult:
    def test_df_has_rows(self, interaction_result):
        assert len(interaction_result.df) > 0

    def test_df_columns_are_strings(self, interaction_result):
        for col in interaction_result.df.columns:
            assert isinstance(col, str), f"Expected str column, got {type(col)}: {col}"

    def test_df_column_format(self, interaction_result):
        """Columns should follow ResName-Chain-ResID_IType pattern."""
        for col in interaction_result.df.columns:
            assert "_" in col, f"Column '{col}' missing underscore separator"

    def test_df_values_binary(self, interaction_result):
        """df values should be 0 or 1 (count=False default)."""
        vals = interaction_result.df.values.ravel()
        assert set(np.unique(vals)).issubset({0, 1})

    def test_summary_has_correct_columns(self, interaction_result):
        assert set(interaction_result.summary.columns) == {"interaction", "occupancy_pct"}

    def test_summary_occupancy_range(self, interaction_result):
        occ = interaction_result.summary["occupancy_pct"]
        assert (occ >= 0).all()
        assert (occ <= 100).all()

    def test_summary_sorted_by_occupancy(self, interaction_result):
        occ = interaction_result.summary["occupancy_pct"].tolist()
        assert occ == sorted(occ, reverse=True)

    def test_sample_name_stored(self, synthetic_universe, cfg, fake_prolif):
        analyzer = InteractionAnalyzer(cfg)
        result = analyzer.run(synthetic_universe, "replica_02")
        assert result.sample_name == "replica_02"

    def test_summary_row_count_matches_df_columns(self, interaction_result):
        assert len(interaction_result.summary) == len(interaction_result.df.columns)


# ---------------------------------------------------------------------------
# ImportError when prolif is absent
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_import_error_without_prolif(self, synthetic_universe, cfg):
        """ImportError with helpful message when prolif is not installed."""
        with patch.dict(sys.modules, {"prolif": None}):
            analyzer = InteractionAnalyzer(cfg)
            with pytest.raises(ImportError, match="(?i)prolif"):
                analyzer.run(synthetic_universe)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestPlotInteractionHeatmap:
    def test_returns_figure(self, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_heatmap
        fig = plot_interaction_heatmap(interaction_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_heatmap
        out = tmp_path / "heatmap.png"
        fig = plot_interaction_heatmap(interaction_result, save_path=out)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    def test_empty_result_returns_figure(self):
        from mdatools.plotting.interaction_plots import plot_interaction_heatmap
        empty = InteractionResult(
            sample_name="empty",
            df=pd.DataFrame(),
            summary=pd.DataFrame(columns=["interaction", "occupancy_pct"]),
        )
        fig = plot_interaction_heatmap(empty)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotInteractionTimeline:
    def test_returns_figure(self, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_timeline
        fig = plot_interaction_timeline(interaction_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_specific_interactions(self, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_timeline
        cols = interaction_result.df.columns[:2].tolist()
        fig = plot_interaction_timeline(interaction_result, interactions=cols)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_timeline
        out = tmp_path / "timeline.png"
        fig = plot_interaction_timeline(interaction_result, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotInteractionComparison:
    def test_returns_figure_single(self, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_comparison
        fig = plot_interaction_comparison([interaction_result])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_multiple(self, synthetic_universe, cfg, fake_prolif):
        from mdatools.plotting.interaction_plots import plot_interaction_comparison
        analyzer = InteractionAnalyzer(cfg)
        r1 = analyzer.run(synthetic_universe, "rep1")
        r2 = analyzer.run(synthetic_universe, "rep2")
        fig = plot_interaction_comparison([r1, r2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_list_returns_figure(self):
        from mdatools.plotting.interaction_plots import plot_interaction_comparison
        fig = plot_interaction_comparison([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, interaction_result):
        from mdatools.plotting.interaction_plots import plot_interaction_comparison
        out = tmp_path / "comparison.png"
        fig = plot_interaction_comparison([interaction_result], save_path=out)
        assert out.exists()
        plt.close(fig)
