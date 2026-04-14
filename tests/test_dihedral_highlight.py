"""Tests for dihedral_highlight module (Issue #3).

PNG generation tests require rdkit + cairosvg (pocket extra).
They are skipped automatically when the extra is not installed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dihedral_df(n_frames: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "frame":   np.arange(n_frames),
        "time_ns": np.arange(n_frames) * 0.1,
        "chi1":    rng.uniform(-60, 60, n_frames),    # range ≈ 120°
        "chi2":    rng.uniform(-180, 180, n_frames),  # range ≈ 360°
        "chi3":    np.full(n_frames, 30.0),            # range = 0°
    })


# ---------------------------------------------------------------------------
# dihedral_range_df
# ---------------------------------------------------------------------------

class TestDihedralRangeDf:

    def test_returns_dataframe(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_dihedral(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        assert list(df["dihedral"]) == ["chi1", "chi2", "chi3"]

    def test_columns_present(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        for col in ("dihedral", "mean", "std", "min", "max", "range"):
            assert col in df.columns

    def test_range_equals_max_minus_min(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        for _, row in df.iterrows():
            assert math.isclose(row["range"], row["max"] - row["min"], rel_tol=1e-9)

    def test_constant_column_has_zero_range(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        chi3_row = df.loc[df["dihedral"] == "chi3"].iloc[0]
        assert chi3_row["range"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_column_zero_std(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        chi3_row = df.loc[df["dihedral"] == "chi3"].iloc[0]
        assert chi3_row["std"] == pytest.approx(0.0, abs=1e-9)

    def test_mean_within_min_max(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        for _, row in df.iterrows():
            assert row["min"] <= row["mean"] <= row["max"]

    def test_skips_metadata_columns(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        df = dihedral_range_df(_make_dihedral_df())
        assert "frame" not in df["dihedral"].values
        assert "time_ns" not in df["dihedral"].values

    def test_nan_values_are_ignored(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        raw = _make_dihedral_df(10)
        raw.loc[0, "chi1"] = float("nan")
        df = dihedral_range_df(raw)
        chi1 = df.loc[df["dihedral"] == "chi1"].iloc[0]
        assert not math.isnan(chi1["mean"])

    def test_empty_dataframe_returns_empty(self):
        from mdatools.plotting.dihedral_highlight import dihedral_range_df
        raw = pd.DataFrame({"frame": [], "time_ns": []})
        df = dihedral_range_df(raw)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# range_color
# ---------------------------------------------------------------------------

class TestRangeColor:

    def test_small_range_is_green(self):
        from mdatools.plotting.dihedral_highlight import range_color
        r, g, b = range_color(10)
        assert g > r and g > b

    def test_large_range_is_red(self):
        from mdatools.plotting.dihedral_highlight import range_color
        r, g, b = range_color(150)
        assert r > g and r > b

    def test_returns_rgb_tuple(self):
        from mdatools.plotting.dihedral_highlight import range_color
        c = range_color(60)
        assert len(c) == 3
        assert all(0 <= v <= 255 for v in c)


# ---------------------------------------------------------------------------
# plot_dihedral_highlight  (requires pocket extra)
# ---------------------------------------------------------------------------

def _pocket_available() -> bool:
    """Return True only if rdkit *and* cairosvg (with libcairo) are usable."""
    try:
        import rdkit  # noqa: F401
        import cairosvg  # noqa: F401 — triggers OSError when libcairo missing
        return True
    except Exception:
        return False


_skip_pocket = pytest.mark.skipif(
    not _pocket_available(),
    reason="pocket extra (rdkit/cairosvg/libcairo) not available",
)


def _write_minimal_pdb(path: Path, resname: str = "UNK") -> Path:
    """Write a small 6-atom ligand PDB (benzene-like ring)."""
    lines = [
        "REMARK  minimal ligand for testing",
        f"HETATM    1  C1  {resname} A   1       0.000   0.000   0.000  1.00  0.00           C",
        f"HETATM    2  C2  {resname} A   1       1.400   0.000   0.000  1.00  0.00           C",
        f"HETATM    3  C3  {resname} A   1       2.100   1.212   0.000  1.00  0.00           C",
        f"HETATM    4  C4  {resname} A   1       1.400   2.424   0.000  1.00  0.00           C",
        f"HETATM    5  C5  {resname} A   1       0.000   2.424   0.000  1.00  0.00           C",
        f"HETATM    6  C6  {resname} A   1      -0.700   1.212   0.000  1.00  0.00           C",
        "CONECT    1    2    6",
        "CONECT    2    3",
        "CONECT    3    4",
        "CONECT    4    5",
        "CONECT    5    6",
        "END",
    ]
    path.write_text("\n".join(lines))
    return path


@_skip_pocket
class TestPlotDihedralHighlight:

    def test_output_file_created(self, tmp_path):
        from mdatools.plotting.dihedral_highlight import plot_dihedral_highlight
        pdb = _write_minimal_pdb(tmp_path / "lig.pdb")
        out = tmp_path / "out.png"
        result = plot_dihedral_highlight(
            pdb,
            highlight_bonds=[{"atom1": "C1", "atom2": "C2", "label": "Δ45°", "color": "#E05C5C"}],
            output_png=out,
        )
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_returns_path(self, tmp_path):
        from mdatools.plotting.dihedral_highlight import plot_dihedral_highlight
        pdb = _write_minimal_pdb(tmp_path / "lig.pdb")
        out = tmp_path / "out.png"
        result = plot_dihedral_highlight(pdb, [], out)
        assert isinstance(result, Path)

    def test_missing_atom_skips_gracefully(self, tmp_path):
        from mdatools.plotting.dihedral_highlight import plot_dihedral_highlight
        pdb = _write_minimal_pdb(tmp_path / "lig.pdb")
        out = tmp_path / "out.png"
        # XY does not exist in the PDB — should not raise
        plot_dihedral_highlight(
            pdb,
            [{"atom1": "XY", "atom2": "XZ", "label": "Δ10°", "color": "#333"}],
            out,
        )
        assert out.exists()

    def test_hex_and_rgb_color_formats(self, tmp_path):
        from mdatools.plotting.dihedral_highlight import plot_dihedral_highlight
        pdb = _write_minimal_pdb(tmp_path / "lig.pdb")
        out = tmp_path / "out.png"
        bonds = [
            {"atom1": "C1", "atom2": "C2", "label": "hex",  "color": "#4488CC"},
            {"atom1": "C3", "atom2": "C4", "label": "rgb",  "color": (200, 50, 50)},
        ]
        plot_dihedral_highlight(pdb, bonds, out)
        assert out.exists()

    def test_output_in_nested_directory(self, tmp_path):
        from mdatools.plotting.dihedral_highlight import plot_dihedral_highlight
        pdb = _write_minimal_pdb(tmp_path / "lig.pdb")
        out = tmp_path / "a" / "b" / "out.png"
        plot_dihedral_highlight(pdb, [], out)
        assert out.exists()
