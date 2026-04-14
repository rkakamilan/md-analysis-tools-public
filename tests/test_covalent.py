"""Tests for covalent_bond_monitor and CovalentBondResult."""

from __future__ import annotations

import numpy as np
import pytest

from mdatools.analysis.covalent import CovalentBondResult, covalent_bond_monitor, list_ligand_atoms

# Atom selections that work with the synthetic universe in conftest.py.
# OD1 (atom index 15) and N4 (atom index 29) are unique atom names, placed
# near each other in the conftest to simulate a protein–ligand contact.
_PROT_SEL = "name OD1"
_LIG_SEL = "name N4"


# ---------------------------------------------------------------------------
# covalent_bond_monitor — analysis tests
# ---------------------------------------------------------------------------

def test_distances_length_matches_frames(synthetic_universe):
    """Number of distance values equals number of trajectory frames."""
    n_frames = len(synthetic_universe.trajectory)
    result = covalent_bond_monitor(
        synthetic_universe,
        residue_sel=_PROT_SEL,
        ligand_sel=_LIG_SEL,
        sample_name="test",
        dt_ns=2.0,
    )
    assert len(result.distances) == n_frames


def test_mean_and_std_are_correct(synthetic_universe):
    """mean_distance and std_distance agree with numpy computations."""
    result = covalent_bond_monitor(
        synthetic_universe,
        residue_sel=_PROT_SEL,
        ligand_sel=_LIG_SEL,
        sample_name="test",
    )
    np.testing.assert_allclose(result.mean_distance, result.distances.mean(), rtol=1e-6)
    np.testing.assert_allclose(result.std_distance, result.distances.std(), rtol=1e-6)


def test_sample_name_stored(synthetic_universe):
    """sample_name is preserved in the result."""
    result = covalent_bond_monitor(
        synthetic_universe,
        residue_sel=_PROT_SEL,
        ligand_sel=_LIG_SEL,
        sample_name="my_sample",
    )
    assert result.sample_name == "my_sample"


def test_dt_ns_stored_and_time_axis(synthetic_universe):
    """dt_ns is stored and time_ns property returns the correct axis."""
    dt = 0.5
    result = covalent_bond_monitor(
        synthetic_universe,
        residue_sel=_PROT_SEL,
        ligand_sel=_LIG_SEL,
        dt_ns=dt,
    )
    assert result.dt_ns == dt
    expected = np.arange(result.n_frames, dtype=float) * dt
    np.testing.assert_array_equal(result.time_ns, expected)


def test_empty_protein_selection_raises(synthetic_universe):
    """ValueError when protein atom selection matches nothing."""
    with pytest.raises(ValueError, match="matched no atoms"):
        covalent_bond_monitor(
            synthetic_universe,
            residue_sel="name NOSUCHATOM",
            ligand_sel=_LIG_SEL,
        )


def test_empty_ligand_selection_raises(synthetic_universe):
    """ValueError when ligand atom selection matches nothing."""
    with pytest.raises(ValueError, match="matched no atoms"):
        covalent_bond_monitor(
            synthetic_universe,
            residue_sel=_PROT_SEL,
            ligand_sel="name NOSUCHATOM",
        )


def test_empty_ligand_selection_includes_available_atoms(synthetic_universe):
    """ValueError for bad ligand atom name includes available atom names in the error message."""
    # Discover actual UNK atom names first so the assertion is not brittle against conftest changes
    unk_atoms = list_ligand_atoms(synthetic_universe, "UNK")
    assert unk_atoms, "synthetic universe must have UNK atoms"

    with pytest.raises(ValueError) as exc_info:
        covalent_bond_monitor(
            synthetic_universe,
            residue_sel=_PROT_SEL,
            ligand_sel="resname UNK and name NOSUCHATOM",
        )
    msg = str(exc_info.value)
    assert "Available atoms" in msg
    assert unk_atoms[0] in msg  # at least the first available atom name is mentioned


# ---------------------------------------------------------------------------
# list_ligand_atoms
# ---------------------------------------------------------------------------

def test_list_ligand_atoms_returns_sorted_unique_names(synthetic_universe):
    """list_ligand_atoms returns sorted, unique atom names for the given resname."""
    names = list_ligand_atoms(synthetic_universe, "UNK")
    assert isinstance(names, list)
    assert len(names) > 0, "UNK residue must have atoms in the synthetic universe"
    assert names == sorted(set(names)), "names should be sorted and unique"


def test_list_ligand_atoms_unknown_resname_returns_empty(synthetic_universe):
    """list_ligand_atoms returns an empty list for a resname not present in the universe."""
    names = list_ligand_atoms(synthetic_universe, "NOSUCHRES")
    assert names == []


def test_distances_are_finite(synthetic_universe):
    """All computed distances are finite (no NaN / inf)."""
    result = covalent_bond_monitor(
        synthetic_universe,
        residue_sel=_PROT_SEL,
        ligand_sel=_LIG_SEL,
    )
    assert np.all(np.isfinite(result.distances))


# ---------------------------------------------------------------------------
# CovalentBondResult — property tests
# ---------------------------------------------------------------------------

def test_n_frames_property():
    result = CovalentBondResult(
        sample_name="s",
        residue_sel="A",
        ligand_sel="B",
        distances=np.array([1.8, 1.9, 1.82]),
        mean_distance=1.84,
        std_distance=0.04,
        dt_ns=1.0,
    )
    assert result.n_frames == 3
