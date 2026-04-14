"""Tests for GPCRmd universe loader and PBC wrapping utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mdatools.universe import apply_membrane_pbc, load_gpcrmd_universe


def _mock_universe(n_frames: int = 10, n_atoms: int = 100) -> MagicMock:
    """Return a MagicMock that mimics an mda.Universe just enough."""
    instance = MagicMock()
    instance.trajectory.__len__ = MagicMock(return_value=n_frames)
    instance.atoms.n_atoms = n_atoms
    return instance


@patch("mdatools.universe.mda.Universe")
def test_single_dcd_string(mock_u, tmp_path):
    """Single DCD path as string → Universe(psf, dcd)."""
    psf = str(tmp_path / "top.psf")
    dcd = str(tmp_path / "traj.dcd")

    mock_u.return_value = _mock_universe()
    result = load_gpcrmd_universe(psf, dcd)

    mock_u.assert_called_once_with(psf, dcd)
    assert result is mock_u.return_value


@patch("mdatools.universe.mda.Universe")
def test_single_dcd_path_object(mock_u, tmp_path):
    """Single DCD as Path object is accepted and converted to str."""
    psf = tmp_path / "top.psf"
    dcd = tmp_path / "traj.dcd"

    mock_u.return_value = _mock_universe()
    load_gpcrmd_universe(psf, dcd)

    mock_u.assert_called_once_with(str(psf), str(dcd))


@patch("mdatools.universe.mda.Universe")
def test_multi_dcd_list(mock_u, tmp_path):
    """List of DCD paths → Universe(psf, dcd1, dcd2) concatenated."""
    psf = str(tmp_path / "top.psf")
    dcd1 = str(tmp_path / "rep1.dcd")
    dcd2 = str(tmp_path / "rep2.dcd")

    mock_u.return_value = _mock_universe(n_frames=20)
    load_gpcrmd_universe(psf, [dcd1, dcd2])

    mock_u.assert_called_once_with(psf, dcd1, dcd2)


@patch("mdatools.universe.mda.Universe")
def test_prmtop_overrides_psf(mock_u, tmp_path):
    """When prmtop_path is given it is used as topology instead of PSF."""
    psf = str(tmp_path / "top.psf")
    dcd = str(tmp_path / "traj.dcd")
    prmtop = str(tmp_path / "top.prmtop")

    mock_u.return_value = _mock_universe()
    load_gpcrmd_universe(psf, dcd, prmtop_path=prmtop)

    mock_u.assert_called_once_with(prmtop, dcd)


@patch("mdatools.universe.mda.Universe")
def test_no_prmtop_uses_psf(mock_u, tmp_path):
    """Without prmtop_path the PSF is used as topology."""
    psf = str(tmp_path / "top.psf")
    dcd = str(tmp_path / "traj.dcd")

    mock_u.return_value = _mock_universe()
    load_gpcrmd_universe(psf, dcd, prmtop_path=None)

    mock_u.assert_called_once_with(psf, dcd)


# ---------------------------------------------------------------------------
# apply_membrane_pbc tests
# ---------------------------------------------------------------------------
# MDAnalysis trajectories only allow add_transformations() once per Universe,
# so each test below builds its own fresh Universe instance.

@pytest.fixture()
def fresh_universe():
    """Fresh synthetic Universe for each test (function scope)."""
    from conftest import _make_universe
    return _make_universe(n_frames=5)


def test_apply_membrane_pbc_returns_same_universe(fresh_universe):
    """apply_membrane_pbc returns the same Universe instance."""
    with (
        patch("mdatools.universe.wrap") as mock_wrap,
        patch("mdatools.universe.center_in_box") as mock_cib,
    ):
        mock_wrap.return_value = MagicMock()
        mock_cib.return_value = MagicMock()

        result = apply_membrane_pbc(fresh_universe)

    assert result is fresh_universe


def test_apply_membrane_pbc_registers_two_transformations(fresh_universe):
    """Exactly two transformations (wrap + center_in_box) are registered."""
    with (
        patch("mdatools.universe.wrap") as mock_wrap,
        patch("mdatools.universe.center_in_box") as mock_cib,
    ):
        mock_wrap.return_value = MagicMock()
        mock_cib.return_value = MagicMock()

        apply_membrane_pbc(fresh_universe)

    assert len(fresh_universe.trajectory._transformations) == 2


def test_apply_membrane_pbc_custom_sel(fresh_universe):
    """Custom protein_sel is forwarded to center_in_box."""
    with (
        patch("mdatools.universe.wrap") as mock_wrap,
        patch("mdatools.universe.center_in_box") as mock_cib,
    ):
        mock_wrap.return_value = MagicMock()
        mock_cib.return_value = MagicMock()

        apply_membrane_pbc(fresh_universe, protein_sel="name CA")

        mock_cib.assert_called_once()
        _, kwargs = mock_cib.call_args
        assert kwargs.get("wrap") is True
