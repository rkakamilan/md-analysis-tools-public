"""Universe loading and alignment helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import center_in_box, wrap

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def load_universe(topology: Path | str, trajectory: Path | str) -> mda.Universe:
    """Load an MDAnalysis Universe and log basic info."""
    u = mda.Universe(str(topology), str(trajectory))
    logger.info(
        "Loaded universe: %d frames, %d atoms",
        len(u.trajectory),
        u.atoms.n_atoms,
    )
    return u


def align_trajectory(
    u: mda.Universe,
    select: str = "protein and name CA",
    in_memory: bool = True,
) -> mda.Universe:
    """Align *u* in-place using AlignTraj.

    Returns the same Universe (alignment is done in-place when
    ``in_memory=True``).
    """
    align.AlignTraj(u, u, select=select, in_memory=in_memory).run()
    logger.info("Trajectory aligned on '%s'", select)
    return u


def load_gpcrmd_universe(
    psf_path: Path | str,
    dcd_paths: Path | str | list[Path | str],
    prmtop_path: Path | str | None = None,
) -> mda.Universe:
    """Load a GPCRmd PSF + DCD universe with optional multi-replica support.

    Parameters
    ----------
    psf_path :
        Path to the CHARMM PSF topology file.
    dcd_paths :
        Path(s) to DCD trajectory file(s).  A single path or a list of paths
        are both accepted; multiple DCDs are concatenated in order.
    prmtop_path :
        Optional AMBER PRMTOP file.  When provided it is used as topology
        instead of *psf_path* (PRMTOP carries more complete mass/charge info).

    Returns
    -------
    mda.Universe
    """
    if isinstance(dcd_paths, (str, Path)):
        dcd_paths = [dcd_paths]
    dcd_str = [str(p) for p in dcd_paths]

    topology = str(prmtop_path) if prmtop_path is not None else str(psf_path)

    u = mda.Universe(topology, *dcd_str)
    logger.info(
        "Loaded GPCRmd universe: topology=%s, %d DCD(s), %d frames, %d atoms",
        Path(topology).name,
        len(dcd_str),
        len(u.trajectory),
        u.atoms.n_atoms,
    )
    return u


def apply_membrane_pbc(
    universe: mda.Universe,
    protein_sel: str = "protein",
) -> mda.Universe:
    """Apply PBC wrapping and protein centring to a membrane-protein trajectory.

    Registers two transformations on *universe.trajectory* in order:

    1. ``wrap(universe.atoms)`` — fold all atoms back into the primary box.
    2. ``center_in_box(protein, wrap=True)`` — centre the protein selection
       in the simulation box while keeping molecules whole.

    Parameters
    ----------
    universe :
        MDAnalysis Universe loaded from a GPCRmd (or similar) DCD trajectory.
    protein_sel :
        Atom-selection string for the protein (default: ``"protein"``).
        Adjust when your topology uses non-standard residue names.

    Returns
    -------
    mda.Universe
        The same *universe* with transformations registered (in-place).
    """
    protein = universe.select_atoms(protein_sel)
    transforms = [
        wrap(universe.atoms),
        center_in_box(protein, wrap=True),
    ]
    universe.trajectory.add_transformations(*transforms)
    logger.info(
        "PBC transformations registered: wrap + center_in_box(sel='%s')",
        protein_sel,
    )
    return universe


def load_and_align(
    topology: Path | str,
    trajectory: Path | str,
    cfg: AnalysisConfig | None = None,
) -> mda.Universe:
    """Convenience: load + align in one call.

    Uses ``cfg.rmsd.align_select`` if *cfg* is provided, otherwise
    ``"protein and name CA"``.
    """
    select = "protein and name CA"
    if cfg is not None:
        select = cfg.rmsd.align_select
    u = load_universe(topology, trajectory)
    return align_trajectory(u, select=select)
