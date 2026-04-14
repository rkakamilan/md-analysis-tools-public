"""Tests for config models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mdatools.config import (
    AnalysisConfig,
    HBondConfig,
    RMSDConfig,
    ResidueGroup,
    ScoringWeights,
)


def test_default_config():
    cfg = AnalysisConfig()
    assert cfg.ligand_resname == "UNK"
    assert cfg.hbond.d_a_cutoff == 3.5
    assert cfg.scoring.occ_weight == 50.0


def test_residue_group():
    grp = ResidueGroup(name="ECD", resids=[35, 36, 39], bonus=20.0)
    assert grp.bonus == 20.0
    assert 36 in grp.resids


def test_config_with_groups():
    cfg = AnalysisConfig(
        ligand_resname="LIG",
        residue_groups={
            "ECD": ResidueGroup(name="ECD", resids=[35, 36], bonus=25.0)
        },
    )
    assert "ECD" in cfg.residue_groups
    assert cfg.residue_groups["ECD"].bonus == 25.0


def test_make_dirs(tmp_path):
    cfg = AnalysisConfig(
        output_dir=tmp_path / "out",
        figures_dir=tmp_path / "figs",
    )
    cfg.make_dirs()
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "figs").is_dir()


def test_hbond_config_defaults():
    hb = HBondConfig()
    assert hb.angle_cutoff == 150.0
    assert hb.start is None


def test_scoring_weights():
    sw = ScoringWeights(occ_weight=60.0)
    assert sw.occ_weight == 60.0
    assert sw.dist_weight == 30.0  # unchanged default
