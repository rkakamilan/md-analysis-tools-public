"""Binding site definition and AutoDock Vina config generation.

Provides three ways to define the docking grid box:

1. :func:`gridbox_from_ligand` — from a co-crystal ligand structure
2. :func:`gridbox_from_residues` — from known binding-site residue IDs
3. :func:`gridbox_from_coords` — from manually specified coordinates

The resulting :class:`GridBox` can be written to a Vina config file with
:func:`write_vina_config`.

Example::

    box = gridbox_from_ligand(ref_mol, padding=4.0)
    write_vina_config(box, receptor.pdbqt, ligand.pdbqt, "config.txt")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem


@dataclass
class GridBox:
    """Rectangular grid box for AutoDock Vina.

    Parameters
    ----------
    center:
        (x, y, z) coordinates of the box centre in Å.
    size:
        (size_x, size_y, size_z) dimensions of the box in Å.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float] = field(default_factory=lambda: (20.0, 20.0, 20.0))


def gridbox_from_ligand(mol: Chem.Mol, padding: float = 4.0) -> GridBox:
    """Define a grid box from a co-crystal ligand structure.

    The centre is the geometric centroid of all heavy atoms.
    The box dimensions are the ligand bounding box extended by *padding* on
    each side, with a minimum size of 10 Å per axis.

    Parameters
    ----------
    mol:
        RDKit Mol with at least one 3D conformer.
    padding:
        Extra space (Å) added beyond the ligand bounding box on each side.

    Returns
    -------
    GridBox

    Raises
    ------
    ValueError
        If *mol* has no conformer.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no 3D conformer")

    positions = mol.GetConformer().GetPositions()  # shape (n_atoms, 3)

    center = tuple(float(v) for v in positions.mean(axis=0))

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = maxs - mins
    raw_size = span + 2 * padding
    size = tuple(float(max(v, 10.0)) for v in raw_size)

    return GridBox(center=center, size=size)  # type: ignore[arg-type]


def gridbox_from_residues(
    receptor_pdb: str | Path,
    residue_ids: list[str],
    padding: float = 4.0,
) -> GridBox:
    """Define a grid box from binding-site residue IDs.

    Parses residue identifiers of the form ``<RESNAME><RESID>``
    (e.g. ``["GLN30", "ARG38"]``) and selects the corresponding residues
    from the receptor PDB via MDAnalysis.  The box centre is the centroid
    of all Cα atoms of the selected residues.

    Parameters
    ----------
    receptor_pdb:
        Path to the receptor PDB file.
    residue_ids:
        Residue identifiers in ``<RESNAME><RESID>`` format.
    padding:
        Extra space (Å) added beyond the residue bounding box on each side.

    Returns
    -------
    GridBox

    Raises
    ------
    ValueError
        If no atoms are found for the given residue IDs.
    """
    import MDAnalysis as mda  # optional heavy dependency — imported lazily

    _RESID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

    selectors = []
    for rid in residue_ids:
        m = _RESID_RE.match(rid)
        if not m:
            raise ValueError(f"Cannot parse residue ID '{rid}' — expected <RESNAME><RESID>")
        resname, resid = m.group(1).upper(), m.group(2)
        selectors.append(f"(resname {resname} and resid {resid})")

    selection_str = " or ".join(selectors)

    u = mda.Universe(str(receptor_pdb))
    atoms = u.select_atoms(f"({selection_str}) and name CA")

    if len(atoms) == 0:
        # Fall back to all atoms of the selection if no Cα found
        atoms = u.select_atoms(selection_str)

    if len(atoms) == 0:
        raise ValueError(
            f"No atoms found for residues {residue_ids} in {receptor_pdb}"
        )

    positions = atoms.positions  # shape (n_atoms, 3)
    center = tuple(float(v) for v in positions.mean(axis=0))

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = maxs - mins
    raw_size = span + 2 * padding
    size = tuple(float(max(v, 10.0)) for v in raw_size)

    return GridBox(center=center, size=size)  # type: ignore[arg-type]


def gridbox_from_coords(
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> GridBox:
    """Define a grid box from manually specified coordinates.

    Parameters
    ----------
    center:
        (x, y, z) coordinates of the box centre in Å.
    size:
        (size_x, size_y, size_z) dimensions in Å.

    Returns
    -------
    GridBox
    """
    return GridBox(center=center, size=size)


def write_vina_config(
    grid: GridBox,
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    output_path: str | Path,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 3.0,
) -> Path:
    """Write an AutoDock Vina config file.

    Parameters
    ----------
    grid:
        Grid box definition.
    receptor_pdbqt:
        Path to the prepared receptor PDBQT.
    ligand_pdbqt:
        Path to the prepared ligand PDBQT.
    output_path:
        Destination path for the config file.
    exhaustiveness:
        Vina exhaustiveness parameter (default 8).
    num_modes:
        Maximum number of binding modes to generate (default 9).
    energy_range:
        Maximum energy difference between best and worst mode in kcal/mol
        (default 3.0).

    Returns
    -------
    Path
        Absolute path to the written config file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cx, cy, cz = grid.center
    sx, sy, sz = grid.size

    lines = [
        f"receptor = {Path(receptor_pdbqt).resolve()}",
        f"ligand   = {Path(ligand_pdbqt).resolve()}",
        "",
        f"center_x = {cx:.3f}",
        f"center_y = {cy:.3f}",
        f"center_z = {cz:.3f}",
        "",
        f"size_x = {sx:.3f}",
        f"size_y = {sy:.3f}",
        f"size_z = {sz:.3f}",
        "",
        f"exhaustiveness = {exhaustiveness}",
        f"num_modes      = {num_modes}",
        f"energy_range   = {energy_range:.1f}",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    return output_path.resolve()
