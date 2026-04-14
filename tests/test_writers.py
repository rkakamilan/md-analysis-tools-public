"""Tests for io/writers.py — write_clean_pdb and write_snapshot_pdb."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import MDAnalysis as mda


# ---------------------------------------------------------------------------
# Fixture: Universe with protein + ligand + solvent atoms
# ---------------------------------------------------------------------------

def _make_universe_with_solvent() -> mda.Universe:
    """Synthetic Universe: 5 protein atoms + 1 ligand + 3 solvent (HOH)."""
    atom_names    = ["N", "CA", "C", "O", "CB",   "C1",  "O1", "O2", "O3"]
    resnames      = ["ALA", "ALA", "ALA", "ALA", "ALA", "UNK", "HOH", "HOH", "HOH"]
    resids        = [1, 1, 1, 1, 1, 0, 99, 100, 101]
    atom_resindex = [0, 0, 0, 0, 0, 1, 2, 3, 4]
    n_residues    = 5

    u = mda.Universe.empty(
        len(atom_names),
        n_residues=n_residues,
        n_segments=1,
        atom_resindex=atom_resindex,
        residue_segindex=[0] * n_residues,
        trajectory=True,
    )
    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("resnames", ["ALA", "UNK", "HOH", "HOH", "HOH"])
    u.add_TopologyAttr("resids", [1, 0, 99, 100, 101])
    u.add_TopologyAttr("segids", ["A"])

    rng = np.random.default_rng(7)
    coords = rng.uniform(-5, 5, (len(atom_names), 3)).astype(np.float32)
    from MDAnalysis.coordinates.memory import MemoryReader
    u.load_new(coords[np.newaxis], format=MemoryReader)
    return u


@pytest.fixture(scope="module")
def universe_with_solvent():
    return _make_universe_with_solvent()


# ---------------------------------------------------------------------------
# write_clean_pdb
# ---------------------------------------------------------------------------

class TestWriteCleanPdb:

    def test_output_file_created(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "clean.pdb"
        result = write_clean_pdb(universe_with_solvent, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_path(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "clean.pdb"
        result = write_clean_pdb(universe_with_solvent, out)
        assert isinstance(result, Path)

    def test_solvent_not_in_output(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "clean.pdb"
        write_clean_pdb(universe_with_solvent, out, ligand_resname="UNK")
        text = out.read_text()
        assert "HOH" not in text

    def test_protein_atoms_in_output(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "clean.pdb"
        write_clean_pdb(universe_with_solvent, out, ligand_resname="UNK")
        text = out.read_text()
        assert "ALA" in text

    def test_ligand_in_output(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "clean.pdb"
        write_clean_pdb(universe_with_solvent, out, ligand_resname="UNK")
        text = out.read_text()
        assert "UNK" in text

    def test_custom_remove_resnames_empty_list_does_not_raise(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        # Empty list is a valid input — should not raise and should produce a file.
        # HOH atoms are not "protein" in MDAnalysis regardless of remove_resnames.
        out = tmp_path / "empty_remove.pdb"
        write_clean_pdb(
            universe_with_solvent, out,
            ligand_resname="UNK",
            remove_resnames=[],
        )
        assert out.exists()
        assert "ALA" in out.read_text()
        assert "UNK" in out.read_text()

    def test_creates_parent_directories(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_clean_pdb
        out = tmp_path / "a" / "b" / "clean.pdb"
        write_clean_pdb(universe_with_solvent, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# write_snapshot_pdb (existing function — regression guard)
# ---------------------------------------------------------------------------

class TestWriteSnapshotPdb:

    def test_creates_file(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_snapshot_pdb
        out = tmp_path / "snap.pdb"
        result = write_snapshot_pdb(universe_with_solvent, 0, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_atoms_written(self, universe_with_solvent, tmp_path):
        from mdatools.io.writers import write_snapshot_pdb
        out = tmp_path / "snap.pdb"
        write_snapshot_pdb(universe_with_solvent, 0, out)
        text = out.read_text()
        # Solvent should be present (write_snapshot_pdb does NOT strip)
        assert "HOH" in text
