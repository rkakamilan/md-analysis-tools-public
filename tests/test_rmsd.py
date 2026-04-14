"""Tests for RMSDAnalyzer."""

import numpy as np
import pytest
from unittest.mock import patch

from mdatools.analysis.rmsd import RMSDAnalyzer, RMSDResult
from mdatools.config import AnalysisConfig, RMSDConfig


def test_rmsd_result_shape(synthetic_universe, simple_cfg):
    """RMSDAnalyzer should return a DataFrame with correct columns."""
    # Use backbone-like select; synthetic universe has N and CA atoms
    cfg = AnalysisConfig(
        ligand_resname="UNK",
        output_dir=simple_cfg.output_dir,
        rmsd=RMSDConfig(backbone_select="name CA", align_select="name CA"),
    )
    analyzer = RMSDAnalyzer(cfg)
    result = analyzer.run(synthetic_universe, "test")

    assert isinstance(result, RMSDResult)
    assert set(result.df.columns) == {"Frame", "Time", "Backbone", "Ligand"}
    assert len(result.df) > 0


def test_rmsd_ligand_column(synthetic_universe, simple_cfg):
    """Ligand RMSD column should contain finite values."""
    cfg = AnalysisConfig(
        ligand_resname="UNK",
        rmsd=RMSDConfig(backbone_select="name CA", align_select="name CA"),
    )
    analyzer = RMSDAnalyzer(cfg)
    result = analyzer.run(synthetic_universe, "test")
    assert np.all(np.isfinite(result.df["Backbone"]))


def test_rmsd_zero_masses_falls_back_to_equal_weights(synthetic_universe):
    """When all atom masses are zero, RMSD must not return NaN."""
    cfg = AnalysisConfig(
        ligand_resname="UNK",
        rmsd=RMSDConfig(backbone_select="name CA", align_select="name CA"),
    )
    zero_masses = np.zeros(synthetic_universe.atoms.n_atoms)
    with patch.object(
        type(synthetic_universe.atoms), "masses",
        new_callable=lambda: property(lambda self: zero_masses),
    ):
        result = RMSDAnalyzer(cfg).run(synthetic_universe, "zero_mass")
    assert np.all(np.isfinite(result.df["Backbone"])), "RMSD should be finite even with zero masses"
