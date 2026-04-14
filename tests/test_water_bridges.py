"""Tests for WaterBridgeAnalyzer and water bridge plot functions."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from mdatools.analysis.water_bridges import WaterBridgeAnalyzer, WaterBridgeResult
from mdatools.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Test fixtures — synthetic universes with/without water bridges
# ---------------------------------------------------------------------------

_MASS_MAP = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008}


def _element_mass(name: str) -> float:
    return _MASS_MAP.get(name[0], 12.0)


def _make_bridge_universe(bridge: bool = True, n_frames: int = 10) -> mda.Universe:
    """Build a minimal Universe with protein, ligand, and water atoms.

    When ``bridge=True``, the water O is placed between the protein O and
    the ligand O at distances ~3.0 Å (both within the 3.5 Å cutoff).
    When ``bridge=False``, water is placed far away (> 10 Å).
    """
    # Atoms: 1 protein O (ALA1), 1 ligand O (UNK0), 1 water O (HOH2)
    atom_names = ["O", "O", "O"]
    resnames = ["ALA", "UNK", "HOH"]
    resids = [1, 0, 2]
    n_atoms = 3
    n_residues = 3

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        n_segments=1,
        atom_resindex=[0, 1, 2],
        residue_segindex=[0, 0, 0],
        trajectory=True,
    )
    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("resids", resids)
    u.add_TopologyAttr("segids", ["A"])
    u.add_TopologyAttr("masses", [_element_mass(n) for n in atom_names])

    # Protein O at origin
    prot_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Ligand O 6 Å away
    lig_pos = np.array([6.0, 0.0, 0.0], dtype=np.float32)

    if bridge:
        # Water O at midpoint: 3.0 Å from each
        water_pos = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    else:
        # Water far away
        water_pos = np.array([50.0, 0.0, 0.0], dtype=np.float32)

    base_coords = np.stack([prot_pos, lig_pos, water_pos])

    rng = np.random.default_rng(0)
    traj = np.stack(
        [base_coords + rng.normal(0, 0.02, base_coords.shape) for _ in range(n_frames)]
    ).astype(np.float32)

    u.load_new(traj, format=MemoryReader)
    return u


@pytest.fixture()
def cfg():
    return AnalysisConfig(ligand_resname="UNK")


@pytest.fixture()
def bridge_universe():
    return _make_bridge_universe(bridge=True)


@pytest.fixture()
def no_bridge_universe():
    return _make_bridge_universe(bridge=False)


@pytest.fixture()
def bridge_result(bridge_universe, cfg):
    analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"], max_bridge_dist=3.5)
    return analyzer.run(bridge_universe, "test")


# ---------------------------------------------------------------------------
# WaterBridgeResult structure — with bridges
# ---------------------------------------------------------------------------


class TestWaterBridgeResult:
    def test_events_not_empty_when_bridge_present(self, bridge_result):
        assert not bridge_result.events.empty

    def test_events_columns(self, bridge_result):
        expected = {"frame", "protein_atom", "water_resid", "ligand_atom", "dist_pw", "dist_wl"}
        assert set(bridge_result.events.columns) == expected

    def test_events_distances_within_cutoff(self, bridge_result):
        assert (bridge_result.events["dist_pw"] <= 3.5).all()
        assert (bridge_result.events["dist_wl"] <= 3.5).all()

    def test_events_distances_positive(self, bridge_result):
        assert (bridge_result.events["dist_pw"] > 0).all()
        assert (bridge_result.events["dist_wl"] > 0).all()

    def test_summary_not_empty_when_bridge_present(self, bridge_result):
        assert not bridge_result.summary.empty

    def test_summary_columns(self, bridge_result):
        expected = {"protein_atom", "ligand_atom", "occupancy_pct", "mean_dist_pw", "mean_dist_wl"}
        assert set(bridge_result.summary.columns) == expected

    def test_summary_occupancy_range(self, bridge_result):
        occ = bridge_result.summary["occupancy_pct"]
        assert (occ > 0).all()
        assert (occ <= 100).all()

    def test_summary_sorted_by_occupancy(self, bridge_result):
        occ = bridge_result.summary["occupancy_pct"].tolist()
        assert occ == sorted(occ, reverse=True)

    def test_n_frames_correct(self, bridge_universe, bridge_result):
        assert bridge_result.n_frames == len(bridge_universe.trajectory)

    def test_sample_name_stored(self, bridge_universe, cfg):
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"])
        result = analyzer.run(bridge_universe, "replica_01")
        assert result.sample_name == "replica_01"


# ---------------------------------------------------------------------------
# No-bridge case
# ---------------------------------------------------------------------------


class TestNoBridge:
    def test_events_empty_when_no_bridge(self, no_bridge_universe, cfg):
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"], max_bridge_dist=3.5)
        result = analyzer.run(no_bridge_universe, "no_bridge")
        assert result.events.empty

    def test_summary_empty_when_no_bridge(self, no_bridge_universe, cfg):
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"], max_bridge_dist=3.5)
        result = analyzer.run(no_bridge_universe, "no_bridge")
        assert result.summary.empty

    def test_no_water_returns_empty(self, synthetic_universe, cfg):
        """The standard synthetic_universe has no HOH → empty result."""
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"])
        result = analyzer.run(synthetic_universe)
        assert result.events.empty
        assert result.summary.empty


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestPlotBridgeOccupancy:
    def test_returns_figure_with_data(self, bridge_result):
        from mdatools.plotting.water_bridge_plots import plot_bridge_occupancy
        fig = plot_bridge_occupancy(bridge_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_empty_result(self, no_bridge_universe, cfg):
        from mdatools.plotting.water_bridge_plots import plot_bridge_occupancy
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"])
        result = analyzer.run(no_bridge_universe)
        fig = plot_bridge_occupancy(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, bridge_result):
        from mdatools.plotting.water_bridge_plots import plot_bridge_occupancy
        out = tmp_path / "bridge_occupancy.png"
        fig = plot_bridge_occupancy(bridge_result, save_path=out)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)


class TestPlotBridgeTimeline:
    def test_returns_figure_with_data(self, bridge_result):
        from mdatools.plotting.water_bridge_plots import plot_bridge_timeline
        fig = plot_bridge_timeline(bridge_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_empty_result(self, no_bridge_universe, cfg):
        from mdatools.plotting.water_bridge_plots import plot_bridge_timeline
        analyzer = WaterBridgeAnalyzer(cfg, water_resnames=["HOH"])
        result = analyzer.run(no_bridge_universe)
        fig = plot_bridge_timeline(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, bridge_result):
        from mdatools.plotting.water_bridge_plots import plot_bridge_timeline
        out = tmp_path / "bridge_timeline.png"
        fig = plot_bridge_timeline(bridge_result, save_path=out)
        assert out.exists()
        plt.close(fig)
