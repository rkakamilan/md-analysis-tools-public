"""Tests for RMSFAnalyzer and RMSF plot functions."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.rmsf import RMSFAnalyzer, RMSFResult
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    return AnalysisConfig(ligand_resname="UNK")


# ---------------------------------------------------------------------------
# RMSFResult structure
# ---------------------------------------------------------------------------


class TestRMSFResult:
    def test_protein_df_columns(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run(synthetic_universe, "s1")
        assert set(result.protein_df.columns) == {"resid", "resname", "rmsf"}

    def test_ligand_df_columns(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run(synthetic_universe, "s1")
        assert set(result.ligand_df.columns) == {"atom", "element", "rmsf"}

    def test_sample_name_stored(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run(synthetic_universe, "replica_01")
        assert result.sample_name == "replica_01"

    def test_protein_rows_match_ca_count(self, synthetic_universe, cfg):
        """Each unique Cα residue should produce one row."""
        n_ca = len(synthetic_universe.select_atoms("name CA"))
        result = RMSFAnalyzer(cfg).run(synthetic_universe)
        assert len(result.protein_df) == n_ca

    def test_ligand_rows_match_atom_count(self, synthetic_universe, cfg):
        n_lig = len(synthetic_universe.select_atoms("resname UNK"))
        result = RMSFAnalyzer(cfg).run(synthetic_universe)
        assert len(result.ligand_df) == n_lig

    def test_rmsf_values_nonnegative(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run(synthetic_universe)
        assert (result.protein_df["rmsf"] >= 0).all()
        assert (result.ligand_df["rmsf"] >= 0).all()

    def test_rmsf_values_finite(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run(synthetic_universe)
        assert np.all(np.isfinite(result.protein_df["rmsf"]))
        assert np.all(np.isfinite(result.ligand_df["rmsf"]))


# ---------------------------------------------------------------------------
# run_binding_site
# ---------------------------------------------------------------------------


class TestRunBindingSite:
    def test_restricts_to_pocket_resids(self, synthetic_universe, cfg):
        """Only residues in pocket_resids should appear in protein_df.

        In the synthetic universe: resids 2,3,4,5,0 correspond to
        ALA, GLY, ASN, SER, TRP (all have CA).  resid=1 is the UNK ligand.
        """
        # Use resids that actually contain CA atoms in the synthetic universe
        ca_resids = set(
            synthetic_universe.select_atoms("name CA").resids.tolist()
        )
        pocket = sorted(ca_resids)[:2]  # take any two CA-containing resids
        result = RMSFAnalyzer(cfg).run_binding_site(
            synthetic_universe, pocket_resids=pocket
        )
        assert set(result.protein_df["resid"].tolist()) == set(pocket)

    def test_ligand_df_still_populated(self, synthetic_universe, cfg):
        ca_resids = sorted(
            set(synthetic_universe.select_atoms("name CA").resids.tolist())
        )
        result = RMSFAnalyzer(cfg).run_binding_site(
            synthetic_universe, pocket_resids=ca_resids[:2]
        )
        assert len(result.ligand_df) > 0

    def test_empty_pocket_returns_empty_protein_df(self, synthetic_universe, cfg):
        result = RMSFAnalyzer(cfg).run_binding_site(
            synthetic_universe, pocket_resids=[]
        )
        assert result.protein_df.empty

    def test_rmsf_nonneg_binding_site(self, synthetic_universe, cfg):
        ca_resids = sorted(
            set(synthetic_universe.select_atoms("name CA").resids.tolist())
        )
        result = RMSFAnalyzer(cfg).run_binding_site(
            synthetic_universe, pocket_resids=ca_resids[:3]
        )
        assert (result.protein_df["rmsf"] >= 0).all()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


@pytest.fixture()
def rmsf_result(synthetic_universe, cfg):
    return RMSFAnalyzer(cfg).run(synthetic_universe, "test")


class TestPlotProteinRmsf:
    def test_returns_figure(self, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_protein_rmsf

        fig = plot_protein_rmsf(rmsf_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_pocket_resids(self, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_protein_rmsf

        fig = plot_protein_rmsf(rmsf_result, pocket_resids=[1, 2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_protein_rmsf

        out = tmp_path / "protein_rmsf.png"
        fig = plot_protein_rmsf(rmsf_result, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)


class TestPlotLigandRmsf:
    def test_returns_figure(self, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_ligand_rmsf

        fig = plot_ligand_rmsf(rmsf_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_ligand_rmsf

        out = tmp_path / "ligand_rmsf.png"
        fig = plot_ligand_rmsf(rmsf_result, save_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotRmsfComparison:
    def test_returns_figure_single(self, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_rmsf_comparison

        fig = plot_rmsf_comparison([rmsf_result])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_multiple(self, synthetic_universe, cfg):
        from mdatools.plotting.rmsf_plots import plot_rmsf_comparison

        analyzer = RMSFAnalyzer(cfg)
        r1 = analyzer.run(synthetic_universe, "rep1")
        r2 = analyzer.run(synthetic_universe, "rep2")
        fig = plot_rmsf_comparison([r1, r2], pocket_resids=[1, 3])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_list_returns_figure(self):
        from mdatools.plotting.rmsf_plots import plot_rmsf_comparison

        fig = plot_rmsf_comparison([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, rmsf_result):
        from mdatools.plotting.rmsf_plots import plot_rmsf_comparison

        out = tmp_path / "rmsf_comparison.png"
        fig = plot_rmsf_comparison([rmsf_result], save_path=out)
        assert out.exists()
        plt.close(fig)
