"""Tests for PoseBusters modules.

PoseBusters / RDKit are optional extras and may not be installed in the
basic dev environment.  Tests requiring them are automatically skipped.
FrameExporter (MDAnalysis only) is always tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# FrameExporter tests (no rdkit / posebusters needed)
# ---------------------------------------------------------------------------

def test_frame_exporter_creates_pdbs(synthetic_universe, tmp_path, simple_cfg):
    """FrameExporter should write PDB files for each frame."""
    from mdatools.posebusters.frame_exporter import FrameExporter

    exporter = FrameExporter(simple_cfg)
    pdb_files = exporter.export_from_universe(
        synthetic_universe,
        output_dir=tmp_path / "frames",
        prefix="test",
        step=5,
    )
    assert len(pdb_files) > 0
    existing = [p for p in pdb_files if p.exists()]
    assert len(existing) > 0


def test_frame_exporter_skip_existing(synthetic_universe, tmp_path, simple_cfg):
    """Second call with skip_existing=True should not overwrite files."""
    from mdatools.posebusters.frame_exporter import FrameExporter

    out_dir = tmp_path / "frames2"
    exporter = FrameExporter(simple_cfg)
    files1 = exporter.export_from_universe(
        synthetic_universe, output_dir=out_dir, prefix="t", step=10
    )
    mtimes1 = {p: p.stat().st_mtime for p in files1 if p.exists()}

    files2 = exporter.export_from_universe(
        synthetic_universe, output_dir=out_dir, prefix="t", step=10,
        skip_existing=True,
    )
    mtimes2 = {p: p.stat().st_mtime for p in files2 if p.exists()}
    # modification times should not change
    for p in mtimes1:
        assert mtimes1[p] == mtimes2[p], f"{p} was overwritten"


def test_frame_exporter_unknown_ligand_raises(synthetic_universe, tmp_path):
    """FrameExporter should raise if ligand resname yields 0 atoms."""
    from mdatools.posebusters.frame_exporter import FrameExporter

    cfg = AnalysisConfig(ligand_resname="DOESNOTEXIST")
    exporter = FrameExporter(cfg)
    with pytest.raises(ValueError, match="0 atoms for ligand resname"):
        exporter.export_from_universe(
            synthetic_universe, output_dir=tmp_path / "f", prefix="x"
        )


# ---------------------------------------------------------------------------
# PoseBustersValidator — unit tests that do NOT call posebusters itself
# ---------------------------------------------------------------------------

def test_validator_requires_smiles_or_sdf():
    """Constructing validator without SMILES/SDF should raise ValueError."""
    from mdatools.posebusters.batch_validator import PoseBustersValidator

    cfg = AnalysisConfig(ligand_resname="UNK")
    with pytest.raises(ValueError, match="ligand_smiles or ligand_sdf"):
        PoseBustersValidator(cfg)


def test_validator_accepts_smiles():
    """Validator should accept a SMILES string without error."""
    from mdatools.posebusters.batch_validator import PoseBustersValidator

    cfg = AnalysisConfig(ligand_resname="UNK")
    v = PoseBustersValidator(cfg, ligand_smiles="c1ccccc1")
    assert v.ligand_smiles == "c1ccccc1"


def test_validator_accepts_sdf(tmp_path):
    """Validator should accept a SDF path without error."""
    from mdatools.posebusters.batch_validator import PoseBustersValidator

    sdf = tmp_path / "lig.sdf"
    sdf.write_text("")  # existence check only at construction time
    cfg = AnalysisConfig(ligand_resname="UNK")
    v = PoseBustersValidator(cfg, ligand_sdf=sdf)
    assert v.ligand_sdf == sdf


def test_parse_frame_index():
    """_parse_frame_index should extract the numeric frame from filename."""
    from mdatools.posebusters.batch_validator import PoseBustersValidator

    assert PoseBustersValidator._parse_frame_index(Path("snap_frame0042.pdb")) == 42
    assert PoseBustersValidator._parse_frame_index(Path("snap_frame0000.pdb")) == 0


# ---------------------------------------------------------------------------
# PML builder test
# ---------------------------------------------------------------------------

def test_build_frame_export_pml_tokens():
    """build_frame_export_pml should contain key tokens."""
    from mdatools.pymol_bridge.pml_builder import build_frame_export_pml

    pml = build_frame_export_pml(
        pse_path=Path("/x/traj.pse"),
        output_dir=Path("/x/frames"),
        prefix="snap",
        max_frames=50,
        step=2,
    )
    assert "load /x/traj.pse" in pml
    assert "MAX_FRAMES   = 50" in pml
    assert "STEP         = 2" in pml
    assert "/x/frames" in pml
    assert "python end" in pml
    assert "quit" in pml
