"""Astex Rule-of-Three (Ro3) filter for fragment library curation.

Reuses :func:`mdatools.docking.analysis.properties.calculate_properties`
for the underlying physicochemical descriptors.  Distinct from the
Lipinski Ro5 / Veber filters in :mod:`mdatools.docking.selection.filters`
because Ro3 has stricter thresholds (MW ≤ 300, LogP ≤ 3, HBD/HBA ≤ 3,
RotBonds ≤ 3) suitable for the fragment-screening regime.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from mdatools.docking.analysis.properties import calculate_properties


@dataclass
class Ro3Thresholds:
    """Astex Rule-of-Three thresholds.

    Reference: Congreve et al., Drug Discov. Today 2003,
    doi:10.1016/S1359-6446(03)02831-9.

    The EU-OPENSCREEN EFSL paper uses the same numeric defaults but
    omits the optional TPSA criterion, which is why ``psa_max`` is
    ``None`` here by default.  Set ``psa_max=60.0`` to enforce the
    extended Astex form.
    """

    mw_max: float = 300.0
    clogp_max: float = 3.0
    hbd_max: int = 3
    hba_max: int = 3
    rotbonds_max: int = 3
    psa_max: float | None = None


def passes_ro3(mol: Chem.Mol, thresholds: Ro3Thresholds | None = None) -> bool:
    """Return ``True`` when *mol* satisfies all Ro3 thresholds."""
    thresholds = thresholds or Ro3Thresholds()
    if mol is None or mol.GetNumAtoms() == 0:
        return False

    props = calculate_properties(mol)
    if props.mw > thresholds.mw_max:
        return False
    if props.logp > thresholds.clogp_max:
        return False
    if props.hbd > thresholds.hbd_max:
        return False
    if props.hba > thresholds.hba_max:
        return False
    if props.rot_bonds > thresholds.rotbonds_max:
        return False
    if thresholds.psa_max is not None and props.tpsa > thresholds.psa_max:
        return False
    return True


def ro3_filter(
    mols: list[Chem.Mol],
    thresholds: Ro3Thresholds | None = None,
) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    """Split *mols* into Ro3-passing and Ro3-failing lists.

    Returns
    -------
    tuple[list[Chem.Mol], list[Chem.Mol]]
        ``(passed, failed)`` — concatenating the two reproduces the
        input order, partitioned into the Ro3 buckets.
    """
    thresholds = thresholds or Ro3Thresholds()
    passed: list[Chem.Mol] = []
    failed: list[Chem.Mol] = []
    for mol in mols:
        if passes_ro3(mol, thresholds):
            passed.append(mol)
        else:
            failed.append(mol)
    return passed, failed
