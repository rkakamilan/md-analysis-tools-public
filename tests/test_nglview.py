"""Tests for NGLView interactive viewer utilities.

All tests are skipped when nglview is not installed.
Install with: pip install "mdatools[viewer]"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Skip entire module when nglview is not installed
nglview = pytest.importorskip("nglview", reason="nglview not installed; skip viewer tests")

from mdatools.plotting.nglview_utils import (
    add_contact_map_overlay,
    show_cluster_representatives,
    show_snapshot,
    show_universe,
)

# Minimal syntactically valid PDB content for tests
_MINIMAL_PDB = (
    "ATOM      1  CA  ALA A   1       1.000   2.000   3.000"
    "  1.00  0.00           C\n"
    "END\n"
)


@pytest.fixture()
def minimal_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "minimal.pdb"
    p.write_text(_MINIMAL_PDB)
    return p


# ---------------------------------------------------------------------------
# show_universe
# ---------------------------------------------------------------------------


class TestShowUniverse:
    def test_returns_nglwidget(self, synthetic_universe):
        view = show_universe(synthetic_universe)
        assert isinstance(view, nglview.NGLWidget)

    def test_custom_selection(self, synthetic_universe):
        view = show_universe(synthetic_universe, selection="resname UNK")
        assert isinstance(view, nglview.NGLWidget)

    def test_trajectory_false(self, synthetic_universe):
        view = show_universe(synthetic_universe, trajectory=False)
        assert isinstance(view, nglview.NGLWidget)


# ---------------------------------------------------------------------------
# show_snapshot
# ---------------------------------------------------------------------------


class TestShowSnapshot:
    def test_returns_nglwidget(self, minimal_pdb):
        view = show_snapshot(minimal_pdb)
        assert isinstance(view, nglview.NGLWidget)

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.pdb"
        with pytest.raises(FileNotFoundError, match="PDB file not found"):
            show_snapshot(missing)

    def test_accepts_string_path(self, minimal_pdb):
        view = show_snapshot(str(minimal_pdb))
        assert isinstance(view, nglview.NGLWidget)

    def test_highlight_residues(self, minimal_pdb):
        view = show_snapshot(minimal_pdb, highlight_residues=[1, 2, 3])
        assert isinstance(view, nglview.NGLWidget)

    def test_no_highlight_residues(self, minimal_pdb):
        view = show_snapshot(minimal_pdb, highlight_residues=None)
        assert isinstance(view, nglview.NGLWidget)


# ---------------------------------------------------------------------------
# show_cluster_representatives
# ---------------------------------------------------------------------------


class TestShowClusterRepresentatives:
    def test_returns_nglwidget(self, tmp_path):
        pdbs = []
        for i in range(3):
            p = tmp_path / f"cluster{i:02d}.pdb"
            p.write_text(_MINIMAL_PDB)
            pdbs.append(p)
        view = show_cluster_representatives(pdbs)
        assert isinstance(view, nglview.NGLWidget)

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.pdb"
        with pytest.raises(FileNotFoundError):
            show_cluster_representatives([missing])

    def test_with_labels(self, tmp_path):
        p = tmp_path / "rep.pdb"
        p.write_text(_MINIMAL_PDB)
        view = show_cluster_representatives([p], labels=["conf_A"])
        assert isinstance(view, nglview.NGLWidget)

    def test_empty_list(self):
        view = show_cluster_representatives([])
        assert isinstance(view, nglview.NGLWidget)


# ---------------------------------------------------------------------------
# add_contact_map_overlay
# ---------------------------------------------------------------------------


class TestAddContactMapOverlay:
    @pytest.fixture()
    def base_widget(self, minimal_pdb):
        return show_snapshot(minimal_pdb)

    def test_returns_same_widget(self, base_widget):
        contacts = pd.DataFrame({"resid": [10, 20], "occupancy_%": [80.0, 30.0]})
        result = add_contact_map_overlay(base_widget, contacts, threshold=0.5)
        assert result is base_widget

    def test_high_occupancy_residues_processed(self, base_widget):
        contacts = pd.DataFrame({
            "resid": [10, 20, 30],
            "occupancy_%": [80.0, 40.0, 60.0],
        })
        # threshold=0.5 → only resid 10 (80%) and 30 (60%) qualify
        result = add_contact_map_overlay(base_widget, contacts, threshold=0.5)
        assert result is base_widget

    def test_all_below_threshold(self, base_widget):
        contacts = pd.DataFrame({
            "resid": [10, 20],
            "occupancy_%": [10.0, 20.0],
        })
        result = add_contact_map_overlay(base_widget, contacts, threshold=0.5)
        assert result is base_widget

    def test_threshold_zero_includes_all(self, base_widget):
        contacts = pd.DataFrame({
            "resid": [10, 20, 30],
            "occupancy_%": [1.0, 2.0, 3.0],
        })
        result = add_contact_map_overlay(base_widget, contacts, threshold=0.0)
        assert result is base_widget

    def test_empty_contacts_dataframe(self, base_widget):
        contacts = pd.DataFrame({"resid": [], "occupancy_%": []})
        result = add_contact_map_overlay(base_widget, contacts)
        assert result is base_widget
