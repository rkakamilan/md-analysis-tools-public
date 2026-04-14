"""PDBFixer integration for protein structure auto-repair (optional extra: repair).

Install with::

    pip install "mdatools[repair]"

Wraps `PDBFixer <https://github.com/openmm/pdbfixer>`_ to fill missing
residues and heavy atoms in crystallographic PDB files before MD analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Water residue names recognised for removal
_WATER_NAMES: frozenset[str] = frozenset(
    {"HOH", "WAT", "TIP3P", "TIP4P", "TIP5P", "SPC", "SPCE"}
)


def _require_pdbfixer():
    """Return pdbfixer module or raise an informative ImportError."""
    try:
        import pdbfixer  # noqa: F401
        return pdbfixer
    except ImportError as exc:
        raise ImportError(
            "pdbfixer is required for structure repair. "
            "Install it with:  pip install 'mdatools[repair]'"
        ) from exc


@dataclass
class RepairReport:
    """Summary of repairs performed by :func:`repair_pdb`.

    Attributes
    ----------
    input_path:
        Path of the original PDB file.
    output_path:
        Path where the repaired PDB was written.
    missing_residues:
        Residue names that were modelled into sequence gaps
        (e.g. ``["GLY", "ALA"]``).
    missing_atoms:
        Heavy-atom names that were added to incomplete residues
        (e.g. ``["CB", "OG1"]``).
    nonstandard_replaced:
        Non-standard residue name → standard name pairs that were
        replaced (e.g. ``["MSE→MET"]``).
    removed_heterogens:
        Residue names that were removed (e.g. ``["HOH", "SO4"]``).
    """

    input_path: Path
    output_path: Path
    missing_residues: list[str] = field(default_factory=list)
    missing_atoms: list[str] = field(default_factory=list)
    nonstandard_replaced: list[str] = field(default_factory=list)
    removed_heterogens: list[str] = field(default_factory=list)


def repair_pdb(
    input_path: "Path | str",
    output_path: "Path | str | None" = None,
    *,
    ph: float = 7.4,
    keep_water: bool = False,
    keep_heterogens: bool = True,
    add_hydrogens: bool = False,
) -> RepairReport:
    """Repair a PDB file using PDBFixer and return a :class:`RepairReport`.

    Steps performed (in order):

    1. Detect and fill missing residues (loop modelling).
    2. Replace non-standard residues (e.g. selenomethionine → methionine).
    3. Add missing heavy atoms.
    4. Remove solvent / heterogens according to *keep_water* / *keep_heterogens*.
    5. Optionally add hydrogen atoms at *ph*.

    Parameters
    ----------
    input_path:
        Path to the input PDB file.
    output_path:
        Where to write the repaired PDB.  Defaults to
        ``<stem>_repaired.pdb`` in the same directory as *input_path*.
    ph:
        pH used when *add_hydrogens* is ``True``.
    keep_water:
        Retain crystallographic water molecules (HOH/WAT/…).
    keep_heterogens:
        Retain non-protein HETATM records (ligands, cofactors, ions).
        When ``False``, all heterogens are removed (respecting
        *keep_water*).
    add_hydrogens:
        Add missing hydrogen atoms at the given *ph*.  Normally
        ``False`` for GROMACS/AMBER workflows that protonate
        internally.

    Returns
    -------
    RepairReport
        Summary of what was changed.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ImportError
        If ``pdbfixer`` / ``openmm`` is not installed.
    """
    _require_pdbfixer()
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PDB file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_repaired.pdb")
    output_path = Path(output_path)

    fixer = PDBFixer(filename=str(input_path))

    # ── 1. Missing residues ──────────────────────────────────────────────────
    fixer.findMissingResidues()
    missing_res_names: list[str] = []
    for res_list in fixer.missingResidues.values():
        missing_res_names.extend(res_list)

    # ── 2. Non-standard residues ─────────────────────────────────────────────
    fixer.findNonstandardResidues()
    nonstandard: list[str] = [
        f"{res.name}→{std}" for res, std in fixer.nonstandardResidues
    ]
    fixer.replaceNonstandardResidues()

    # ── 3. Missing heavy atoms ───────────────────────────────────────────────
    fixer.findMissingAtoms()
    missing_atom_names: list[str] = []
    for atom_list in fixer.missingAtoms.values():
        missing_atom_names.extend(a.name for a in atom_list)

    fixer.addMissingAtoms()

    # ── 4. Heterogens / water ────────────────────────────────────────────────
    removed_het: list[str] = []

    if not keep_heterogens:
        # Remove all heterogens; water obeys keep_water
        removed_het = _collect_heterogens(fixer, include_water=True)
        fixer.removeHeterogens(keepWater=keep_water)
        if keep_water:
            # Water was kept; remove it from the removed list
            removed_het = [n for n in removed_het if n not in _WATER_NAMES]
    elif not keep_water:
        # Keep ligands but remove water
        removed_het = _remove_water_only(fixer)

    # ── 5. Hydrogens (optional) ──────────────────────────────────────────────
    if add_hydrogens:
        fixer.addMissingHydrogens(ph)

    # ── Write output ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    logger.info(
        "repair_pdb: %s → %s  (missing_res=%d, missing_atoms=%d, nonstandard=%d, removed=%d)",
        input_path.name,
        output_path.name,
        len(missing_res_names),
        len(missing_atom_names),
        len(nonstandard),
        len(removed_het),
    )

    return RepairReport(
        input_path=input_path,
        output_path=output_path,
        missing_residues=missing_res_names,
        missing_atoms=missing_atom_names,
        nonstandard_replaced=nonstandard,
        removed_heterogens=removed_het,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _collect_heterogens(fixer, *, include_water: bool) -> list[str]:
    """Return residue names of heterogens currently in fixer topology."""
    names: list[str] = []
    for res in fixer.topology.residues():
        is_water = res.name in _WATER_NAMES
        if include_water or not is_water:
            # Check if hetatm (non-standard chain atoms in PDBFixer are marked
            # via residue.chain; simpler: any non-protein/nucleic residue)
            if not _is_protein_or_nucleic(res.name):
                names.append(res.name)
    return names


def _remove_water_only(fixer) -> list[str]:
    """Remove water residues from fixer topology in-place.

    Returns the list of removed residue names.
    """
    from openmm.app import Modeller

    modeller = Modeller(fixer.topology, fixer.positions)
    waters = [
        res for res in modeller.topology.residues() if res.name in _WATER_NAMES
    ]
    removed = [res.name for res in waters]
    modeller.delete(waters)
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions
    return removed


# Standard amino-acid and nucleic-acid residue names (PDBFixer / IUPAC)
_PROTEIN_NAMES: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    # Protonation variants used by PDBFixer/OpenMM
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
})
_NUCLEIC_NAMES: frozenset[str] = frozenset({
    "DA", "DC", "DG", "DT", "DI",
    "A", "C", "G", "U", "I",
})


def _is_protein_or_nucleic(resname: str) -> bool:
    return resname in _PROTEIN_NAMES or resname in _NUCLEIC_NAMES
