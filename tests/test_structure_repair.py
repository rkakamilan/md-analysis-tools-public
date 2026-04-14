"""Tests for io/structure_repair.py — repair_pdb() and RepairReport.

All tests are skipped when pdbfixer / openmm is not installed.
Install with: pip install "mdatools[repair]"
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip entire module when pdbfixer is not installed
pdbfixer = pytest.importorskip("pdbfixer", reason="pdbfixer not installed; skip repair tests")

from mdatools.io.structure_repair import RepairReport, repair_pdb

# ---------------------------------------------------------------------------
# Minimal PDB fixtures
# ---------------------------------------------------------------------------

# A small but valid PDB: one ALA residue + one water + one ligand HETATM
_PDB_WITH_WATER_AND_LIG = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000   1.000   4.000  1.00  0.00           C
HETATM    6  O   HOH A   2       5.000   5.000   5.000  1.00  0.00           O
HETATM    7  C1  LIG A   3       8.000   8.000   8.000  1.00  0.00           C
END
"""

# A PDB missing CB of ALA (triggers addMissingAtoms)
_PDB_MISSING_CB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.00           O
END
"""


@pytest.fixture()
def pdb_with_water_and_lig(tmp_path: Path) -> Path:
    p = tmp_path / "input.pdb"
    p.write_text(_PDB_WITH_WATER_AND_LIG)
    return p


@pytest.fixture()
def pdb_missing_cb(tmp_path: Path) -> Path:
    p = tmp_path / "missing_cb.pdb"
    p.write_text(_PDB_MISSING_CB)
    return p


# ---------------------------------------------------------------------------
# RepairReport structure
# ---------------------------------------------------------------------------


class TestRepairReport:
    def test_returns_repair_report(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out)
        assert isinstance(report, RepairReport)

    def test_input_path_stored(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out)
        assert report.input_path == pdb_with_water_and_lig

    def test_output_path_stored(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out)
        assert report.output_path == out

    def test_list_fields_are_lists(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out)
        assert isinstance(report.missing_residues, list)
        assert isinstance(report.missing_atoms, list)
        assert isinstance(report.nonstandard_replaced, list)
        assert isinstance(report.removed_heterogens, list)


# ---------------------------------------------------------------------------
# Output file behaviour
# ---------------------------------------------------------------------------


class TestOutputFile:
    def test_explicit_output_path_created(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "repaired.pdb"
        repair_pdb(pdb_with_water_and_lig, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_auto_output_path_default_naming(self, pdb_with_water_and_lig):
        report = repair_pdb(pdb_with_water_and_lig)  # no output_path
        expected = pdb_with_water_and_lig.with_name("input_repaired.pdb")
        assert report.output_path == expected
        assert expected.exists()

    def test_output_contains_atom_records(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        repair_pdb(pdb_with_water_and_lig, out)
        content = out.read_text()
        assert "ATOM" in content

    def test_creates_output_subdirectory(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "subdir" / "repaired.pdb"
        repair_pdb(pdb_with_water_and_lig, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# FileNotFoundError
# ---------------------------------------------------------------------------


class TestFileNotFound:
    def test_missing_input_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.pdb"
        with pytest.raises(FileNotFoundError, match="PDB file not found"):
            repair_pdb(missing)


# ---------------------------------------------------------------------------
# Water removal
# ---------------------------------------------------------------------------


class TestWaterRemoval:
    def test_keep_water_false_removes_water(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out, keep_water=False)
        content = out.read_text()
        assert "HOH" not in content

    def test_keep_water_true_retains_water(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        repair_pdb(pdb_with_water_and_lig, out, keep_water=True, keep_heterogens=True)
        content = out.read_text()
        assert "HOH" in content

    def test_removed_heterogens_includes_water_when_removed(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        report = repair_pdb(pdb_with_water_and_lig, out, keep_water=False, keep_heterogens=False)
        assert "HOH" in report.removed_heterogens


# ---------------------------------------------------------------------------
# Heterogens (ligand) retention
# ---------------------------------------------------------------------------


class TestHeterogenRetention:
    def test_keep_heterogens_true_retains_ligand(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        repair_pdb(pdb_with_water_and_lig, out, keep_heterogens=True, keep_water=True)
        content = out.read_text()
        assert "LIG" in content

    def test_keep_heterogens_false_removes_ligand(self, pdb_with_water_and_lig, tmp_path):
        out = tmp_path / "out.pdb"
        repair_pdb(pdb_with_water_and_lig, out, keep_heterogens=False, keep_water=False)
        content = out.read_text()
        assert "LIG" not in content
