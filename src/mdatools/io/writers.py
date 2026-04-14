"""File output helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default solvent / ion residue names to strip when writing clean PDBs.
_DEFAULT_REMOVE_RESNAMES: list[str] = [
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4",
    "NA", "CL", "K", "CA", "MG", "ZN", "FE", "MN", "CU", "CO", "NI",
    "SO4", "PO4", "GOL", "EDO", "PEG", "DMS", "ACE", "ACT",
]


def write_summary_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write *df* to *path* and return the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_snapshot_pdb(u, frame: int, output_path: Path | str) -> Path:
    """Write a single-frame PDB from *u* at *frame*.

    Parameters
    ----------
    u:
        MDAnalysis Universe (already aligned if needed).
    frame:
        Zero-based frame index.
    output_path:
        Destination .pdb file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    u.trajectory[frame]
    u.select_atoms("all").write(str(output_path))
    return output_path


def write_clean_pdb(
    u,
    output_path: Path | str,
    ligand_resname: str = "UNK",
    remove_resnames: list[str] | None = None,
) -> Path:
    """Write the current frame of *u* with solvent/ions removed.

    Selects ``protein`` atoms plus ligand residues and writes them to
    *output_path*.  All residues whose names appear in *remove_resnames*
    are excluded.

    Parameters
    ----------
    u:
        MDAnalysis Universe positioned at the desired frame.
    output_path:
        Destination .pdb path.
    ligand_resname:
        Residue name of the ligand to keep (default ``"UNK"``).
    remove_resnames:
        Residue names to exclude.  Defaults to water, common ions, and
        crystallographic additives.

    Returns
    -------
    Path to the written PDB.
    """
    remove = remove_resnames if remove_resnames is not None else _DEFAULT_REMOVE_RESNAMES
    base_sel = f"protein or resname {ligand_resname}"
    if remove:
        exclude_sel = " or ".join(f"resname {r}" for r in remove)
        sel = f"({base_sel}) and not ({exclude_sel})"
    else:
        sel = base_sel
    ag = u.select_atoms(sel)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ag.write(str(output_path))
    return output_path
