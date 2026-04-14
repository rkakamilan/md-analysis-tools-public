"""Tests for pocket environment profiler.

RDKit / cairosvg / Pillow are optional extras and may not be installed.
Tests requiring them are automatically skipped.
The core metric computation (numpy only) is always tested.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prot_atoms(positions_and_elements):
    """Build prot_atoms list from [(x, y, z, elem), ...]."""
    return list(positions_and_elements)


def _make_lig_atoms(entries):
    """Build lig_atoms list from [(serial, name, x, y, z, elem), ...]."""
    return list(entries)


# ---------------------------------------------------------------------------
# compute_metrics (numpy only — always runs)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_single_atom_d_min(self):
        """d_min equals distance from ligand atom to its nearest protein atom."""
        from mdatools.pocket.profiler import compute_metrics

        prot = [(5.0, 0.0, 0.0, "C")]
        lig = [(1, "C1", 0.0, 0.0, 0.0, "C")]
        result = compute_metrics(prot, lig)

        assert len(result) == 1
        assert result[0]["atom"] == "C1"
        assert result[0]["d_min"] == pytest.approx(5.0, abs=0.01)

    def test_d_margin_is_d_min_minus_vdw(self):
        """d_margin = d_min - (vdw_lig + vdw_prot)."""
        from mdatools.pocket.profiler import compute_metrics, VDW_RADII

        prot = [(5.0, 0.0, 0.0, "C")]
        lig = [(1, "C1", 0.0, 0.0, 0.0, "C")]
        result = compute_metrics(prot, lig)

        expected_margin = 5.0 - (VDW_RADII["C"] + VDW_RADII["C"])
        assert result[0]["d_margin"] == pytest.approx(expected_margin, abs=0.01)

    def test_hydrophob_pure_carbon_env(self):
        """All-carbon environment should give hydrophob == 1.0."""
        from mdatools.pocket.profiler import compute_metrics

        # 8 carbon atoms surrounding the ligand within ENV_RADIUS=5Å
        prot = [(2.0, 0.0, 0.0, "C"), (-2.0, 0.0, 0.0, "C"),
                (0.0, 2.0, 0.0, "C"), (0.0, -2.0, 0.0, "C"),
                (0.0, 0.0, 2.0, "C"), (0.0, 0.0, -2.0, "C"),
                (3.0, 0.0, 0.0, "C"), (-3.0, 0.0, 0.0, "C")]
        lig = [(1, "N1", 0.0, 0.0, 0.0, "N")]
        result = compute_metrics(prot, lig)

        assert result[0]["hydrophob"] == pytest.approx(1.0, abs=0.01)
        assert result[0]["n_NO"] == 0

    def test_n_NO_counts_nitrogen_and_oxygen(self):
        """n_NO should count N and O atoms within ENV_RADIUS."""
        from mdatools.pocket.profiler import compute_metrics

        prot = [
            (2.0, 0.0, 0.0, "N"),
            (2.5, 0.0, 0.0, "O"),
            (3.0, 0.0, 0.0, "C"),
            (3.5, 0.0, 0.0, "N"),
        ]
        lig = [(1, "C1", 0.0, 0.0, 0.0, "C")]
        result = compute_metrics(prot, lig)

        assert result[0]["n_NO"] == 3  # 2×N + 1×O

    def test_empty_prot_returns_empty(self):
        """Empty protein atom list should return empty metrics."""
        from mdatools.pocket.profiler import compute_metrics

        result = compute_metrics([], [(1, "C1", 0.0, 0.0, 0.0, "C")])
        assert result == []

    def test_empty_lig_returns_empty(self):
        """Empty ligand atom list should return empty metrics."""
        from mdatools.pocket.profiler import compute_metrics

        result = compute_metrics([(0.0, 0.0, 0.0, "C")], [])
        assert result == []

    def test_multiple_lig_atoms(self):
        """Each ligand atom should produce one metric entry."""
        from mdatools.pocket.profiler import compute_metrics

        prot = [(10.0, 0.0, 0.0, "C")]
        lig = [
            (1, "C1", 0.0, 0.0, 0.0, "C"),
            (2, "N2", 1.0, 0.0, 0.0, "N"),
            (3, "O3", 2.0, 0.0, 0.0, "O"),
        ]
        result = compute_metrics(prot, lig)

        assert len(result) == 3
        atoms = [r["atom"] for r in result]
        assert "C1" in atoms
        assert "N2" in atoms
        assert "O3" in atoms

    def test_nearest_element_recorded(self):
        """nearest_e should reflect the element of the closest protein atom."""
        from mdatools.pocket.profiler import compute_metrics

        prot = [(5.0, 0.0, 0.0, "N"), (3.0, 0.0, 0.0, "O")]  # O is closer
        lig = [(1, "C1", 0.0, 0.0, 0.0, "C")]
        result = compute_metrics(prot, lig)

        assert result[0]["nearest_e"] == "O"


# ---------------------------------------------------------------------------
# parse_clean_pdb (file I/O)
# ---------------------------------------------------------------------------

class TestParseCleanPdb:
    def _write_mini_pdb(self, tmp_path: Path, ligand_resname: str = "UNK") -> Path:
        pdb = tmp_path / "mini.pdb"
        pdb.write_text(
            f"ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            f"ATOM      2  N   ALA A   1       4.000   5.000   6.000  1.00  0.00           N\n"
            f"HETATM    3  C1  {ligand_resname} A   2       7.000   8.000   9.000  1.00  0.00           C\n"
            f"HETATM    4  H1  {ligand_resname} A   2       7.100   8.100   9.100  1.00  0.00           H\n"
            f"END\n"
        )
        return pdb

    def test_prot_atoms_parsed(self, tmp_path):
        from mdatools.pocket.profiler import parse_clean_pdb

        pdb = self._write_mini_pdb(tmp_path)
        prot, lig = parse_clean_pdb(pdb)

        assert len(prot) == 2
        assert prot[0][3] == "C"   # element
        assert prot[1][3] == "N"

    def test_lig_atoms_parsed(self, tmp_path):
        from mdatools.pocket.profiler import parse_clean_pdb

        pdb = self._write_mini_pdb(tmp_path)
        prot, lig = parse_clean_pdb(pdb)

        assert len(lig) == 1        # H should be excluded
        assert lig[0][1] == "C1"    # canon_name
        assert lig[0][5] == "C"     # element

    def test_custom_resname(self, tmp_path):
        from mdatools.pocket.profiler import parse_clean_pdb

        pdb = self._write_mini_pdb(tmp_path, ligand_resname="LIG")
        prot, lig = parse_clean_pdb(pdb, ligand_resname="LIG")

        assert len(lig) == 1

    def test_wrong_resname_gives_empty_lig(self, tmp_path):
        from mdatools.pocket.profiler import parse_clean_pdb

        pdb = self._write_mini_pdb(tmp_path)
        prot, lig = parse_clean_pdb(pdb, ligand_resname="NOTEXIST")

        assert lig == []


# ---------------------------------------------------------------------------
# PocketProfiler high-level API
# ---------------------------------------------------------------------------

class TestPocketProfiler:
    def _write_simple_pdb(self, tmp_path: Path) -> Path:
        pdb = tmp_path / "snap.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   5.000  1.00  0.00           C\n"
            "HETATM    2  C1  UNK A   2       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        return pdb

    def test_profile_returns_metrics(self, tmp_path):
        from mdatools.pocket.profiler import PocketProfiler

        pdb = self._write_simple_pdb(tmp_path)
        profiler = PocketProfiler()
        metrics = profiler.profile(pdb)

        assert len(metrics) == 1
        assert metrics[0]["atom"] == "C1"
        assert metrics[0]["d_min"] == pytest.approx(5.0, abs=0.01)

    def test_profile_missing_ligand_raises(self, tmp_path):
        from mdatools.pocket.profiler import PocketProfiler

        pdb = tmp_path / "no_lig.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   5.000  1.00  0.00           C\n"
            "END\n"
        )
        profiler = PocketProfiler(ligand_resname="UNK")
        with pytest.raises(ValueError, match="No ligand atoms"):
            profiler.profile(pdb)

    def test_profile_batch_produces_csv(self, tmp_path):
        from mdatools.pocket.profiler import PocketProfiler

        pdb = self._write_simple_pdb(tmp_path)
        profiler = PocketProfiler()
        out_dir = tmp_path / "out"
        results = profiler.profile_batch([pdb], output_dir=out_dir)

        assert "snap" in results
        csv_path = out_dir / "pocket_snap.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Color map functions
# ---------------------------------------------------------------------------

class TestColorMaps:
    def test_dmin_color_tight(self):
        from mdatools.plotting.pocket_profile import dmin_color
        r, g, b = dmin_color(3.0)
        assert r > 150  # predominantly red for tight contacts

    def test_dmin_color_open(self):
        from mdatools.plotting.pocket_profile import dmin_color
        r, g, b = dmin_color(8.0)
        assert g > 150  # predominantly green for open contacts

    def test_hydrophob_border_hydrophobic(self):
        from mdatools.plotting.pocket_profile import hydrophob_border
        r, g, b = hydrophob_border(0.8)
        assert r > g and r > b  # orange-ish

    def test_hydrophob_border_polar(self):
        from mdatools.plotting.pocket_profile import hydrophob_border
        r, g, b = hydrophob_border(0.1)
        assert b > r and b > g  # blue-ish
