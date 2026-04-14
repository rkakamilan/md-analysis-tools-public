"""Covalent bond distance monitoring for irreversible inhibitor MD analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import dist as mda_dist

logger = logging.getLogger(__name__)


@dataclass
class CovalentBondResult:
    """Result of a covalent bond distance analysis.

    Attributes
    ----------
    sample_name :
        User-supplied label for this trajectory / sample.
    residue_sel :
        Atom-selection string used for the protein atom (e.g. Cys SG).
    ligand_sel :
        Atom-selection string used for the ligand atom (e.g. Michael acceptor C).
    distances :
        Per-frame distances in Å, shape ``(n_frames,)``.
    mean_distance :
        Mean C–S (or equivalent) distance across all frames.
    std_distance :
        Standard deviation of the distance across all frames.
    dt_ns :
        Frame interval in nanoseconds (used to build the time axis).
    """

    sample_name: str
    residue_sel: str
    ligand_sel: str
    distances: np.ndarray
    mean_distance: float
    std_distance: float
    dt_ns: float = 1.0

    @property
    def n_frames(self) -> int:
        return len(self.distances)

    @property
    def time_ns(self) -> np.ndarray:
        """Time axis in nanoseconds, shape ``(n_frames,)``."""
        return np.arange(self.n_frames, dtype=float) * self.dt_ns


def list_ligand_atoms(universe: mda.Universe, ligand_resname: str) -> list[str]:
    """Return sorted unique atom names for *ligand_resname* in *universe*.

    Use this helper to discover the correct atom name for
    :func:`covalent_bond_monitor` when the topology uses a naming convention
    different from the one assumed in the notebook CONFIG.

    Example
    -------
    >>> names = list_ligand_atoms(u, "LIG")
    >>> print("Available ligand atoms:", names)
    Available ligand atoms: ['C01', 'C03', 'N02', 'O17', ...]
    """
    lig = universe.select_atoms(f"resname {ligand_resname}")
    return sorted(set(lig.names))


def covalent_bond_monitor(
    universe: mda.Universe,
    residue_sel: str,
    ligand_sel: str,
    sample_name: str = "",
    dt_ns: float = 1.0,
) -> CovalentBondResult:
    """Track the distance between a covalently bonded atom pair across all frames.

    Typical use cases:

    - EGFR Cys797 Sγ — osimertinib Michael-acceptor C (C–S bond, ~1.82 Å)
    - BTK  Cys481 Sγ — ibrutinib / zanubrutinib
    - Mpro Cys145 Sγ — nirmatrelvir / N3

    Parameters
    ----------
    universe :
        MDAnalysis Universe with the trajectory to analyse.
    residue_sel :
        Atom-selection string for the **protein** atom of the bond,
        e.g. ``"resname CYS and name SG and resid 797"``.
    ligand_sel :
        Atom-selection string for the **ligand** atom of the bond,
        e.g. ``"resname LIG and name C20"``.
    sample_name :
        Label for this trajectory, used in plots and result repr.
    dt_ns :
        Frame interval in nanoseconds (default: 1.0 ns).

    Returns
    -------
    CovalentBondResult

    Raises
    ------
    ValueError
        If either atom selection returns zero atoms.
    """
    prot_atoms = universe.select_atoms(residue_sel)
    lig_atoms = universe.select_atoms(ligand_sel)

    if len(prot_atoms) == 0:
        raise ValueError(
            f"Protein atom selection '{residue_sel}' matched no atoms."
        )
    if len(lig_atoms) == 0:
        # Extract resname from the ligand selection to list available atoms.
        # Heuristic: look for 'resname XXX' pattern in the selection string.
        import re
        m = re.search(r"resname\s+(\S+)", ligand_sel)
        if m:
            available = list_ligand_atoms(universe, m.group(1))
            hint = f" Available atoms for resname {m.group(1)}: {available}"
        else:
            hint = ""
        raise ValueError(
            f"Ligand atom selection '{ligand_sel}' matched no atoms.{hint}"
        )

    distances: list[float] = []
    for _ts in universe.trajectory:
        d = mda_dist(prot_atoms, lig_atoms)[2][0]
        distances.append(float(d))

    dist_arr = np.array(distances, dtype=np.float64)

    logger.info(
        "covalent_bond_monitor [%s]: %d frames, mean=%.3f Å, std=%.3f Å",
        sample_name,
        len(dist_arr),
        dist_arr.mean(),
        dist_arr.std(),
    )

    return CovalentBondResult(
        sample_name=sample_name,
        residue_sel=residue_sel,
        ligand_sel=ligand_sel,
        distances=dist_arr,
        mean_distance=float(dist_arr.mean()),
        std_distance=float(dist_arr.std()),
        dt_ns=dt_ns,
    )
