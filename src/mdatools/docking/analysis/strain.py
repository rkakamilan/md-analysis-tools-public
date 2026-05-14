"""Ligand conformational strain energy calculation.

Computes ``strain = E_bound - E_free`` using MMFF94 or UFF force fields.
A high strain energy suggests the bound conformation is geometrically
strained and may be a false-positive docking hit.

Two protocols are provided:

- **Standard** (``compute_strain_energy``): full conformer relaxation in
  the unbound state — can over-penalise large flexible scaffolds.
- **H-relaxed** (``compute_strain_energy_h_relaxed``): only hydrogens are
  moved in the free-state minimisation; heavy atoms are frozen.  Avoids
  excessive strain penalties for macrocycles and flexible linkers.

Ported from ``docking_analysis.analysis.strain``.

Example::

    from rdkit import Chem
    from mdatools.docking.analysis.strain import compute_strain_energy

    mol = next(Chem.SDMolSupplier("poses.sdf", removeHs=False))
    strain = compute_strain_energy(mol)  # kcal/mol or None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StrainThresholds:
    """Recommended strain energy thresholds for pose acceptance.

    Attributes
    ----------
    warn:
        Strain above this value deserves a warning (kcal/mol).
    reject:
        Strain above this value warrants pose rejection (kcal/mol).
    protocol:
        The minimisation protocol used to derive the thresholds.
    """

    warn: float
    reject: float
    protocol: str = "standard"


def compute_strain_energy(
    mol: object,
    force_field: str = "mmff94",
) -> float | None:
    """Compute ``E_bound - E_free`` for a docked pose.

    Parameters
    ----------
    mol:
        RDKit Mol with a 3-D conformer (the bound pose).
    force_field:
        ``"mmff94"`` (default) or ``"uff"``.

    Returns
    -------
    float or None
        Strain energy in kcal/mol, or ``None`` when the force field
        fails to assign parameters (common for unusual functional groups).
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import AllChem  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for strain energy calculation. "
            "Install it with:  pip install 'mdatools[docking]'"
        ) from exc

    mol = Chem.RWMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    # Energy of the bound conformation
    if force_field == "mmff94":
        ff_bound = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol)
        )
    else:
        ff_bound = AllChem.UFFGetMoleculeForceField(mol)

    if ff_bound is None:
        logger.debug(
            "Force field assignment failed for bound conformation; returning None."
        )
        return None

    e_bound = ff_bound.CalcEnergy()

    # Copy and minimise to get free-state energy
    mol_free = Chem.RWMol(mol)
    if force_field == "mmff94":
        ff_free = AllChem.MMFFGetMoleculeForceField(
            mol_free, AllChem.MMFFGetMoleculeProperties(mol_free)
        )
    else:
        ff_free = AllChem.UFFGetMoleculeForceField(mol_free)

    if ff_free is None:
        return None

    ff_free.Minimize(maxIts=2000)
    e_free = ff_free.CalcEnergy()

    return float(e_bound - e_free)


def compute_strain_energy_h_relaxed(
    mol: object,
    force_field: str = "mmff94",
) -> float | None:
    """Compute strain using the H-relaxation protocol.

    Heavy atoms are frozen at their bound-pose coordinates; only
    hydrogens are minimised in the free state.  This avoids excessive
    strain penalties for macrocycles and large scaffolds.

    Parameters
    ----------
    mol:
        RDKit Mol with 3-D conformer (bound pose).
    force_field:
        ``"mmff94"`` or ``"uff"``.

    Returns
    -------
    float or None
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import AllChem  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("RDKit required for strain energy.") from exc

    mol = Chem.RWMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    if force_field == "mmff94":
        ff_bound = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol)
        )
    else:
        ff_bound = AllChem.UFFGetMoleculeForceField(mol)

    if ff_bound is None:
        return None

    e_bound = ff_bound.CalcEnergy()

    mol_free = Chem.RWMol(mol)
    if force_field == "mmff94":
        ff_free = AllChem.MMFFGetMoleculeForceField(
            mol_free, AllChem.MMFFGetMoleculeProperties(mol_free)
        )
    else:
        ff_free = AllChem.UFFGetMoleculeForceField(mol_free)

    if ff_free is None:
        return None

    # Freeze all heavy atoms — only H positions are relaxed
    for atom in mol_free.GetAtoms():
        if atom.GetAtomicNum() != 1:
            ff_free.AddFixedPoint(atom.GetIdx())

    ff_free.Minimize(maxIts=2000)
    e_free = ff_free.CalcEnergy()

    return float(e_bound - e_free)


def recommend_strain_thresholds(
    mol: object,
    protocol: str = "standard",
) -> StrainThresholds:
    """Return recommended strain thresholds for a molecule.

    Large, flexible molecules (MW > 500 Da or rotatable bonds > 10)
    receive relaxed thresholds to avoid over-penalising them in the
    H-relaxed protocol.

    Parameters
    ----------
    mol:
        RDKit Mol (2D or 3D).
    protocol:
        ``"standard"`` or ``"h_relaxed"``.

    Returns
    -------
    StrainThresholds
    """
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: PLC0415
    except ImportError:
        return StrainThresholds(warn=10.0, reject=20.0, protocol=protocol)

    mw = Descriptors.MolWt(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    if protocol == "h_relaxed" and (mw > 500 or n_rot > 10):
        return StrainThresholds(warn=50.0, reject=80.0, protocol=protocol)
    return StrainThresholds(warn=10.0, reject=20.0, protocol=protocol)


def add_strain_to_df(
    df: object,
    mols: list,
    force_field: str = "mmff94",
    h_relaxed: bool = False,
    column: str = "strain_kcal",
) -> object:
    """Add a strain energy column to a poses DataFrame.

    Parameters
    ----------
    df:
        Poses DataFrame (rows parallel to *mols*).
    mols:
        RDKit Mol list.
    force_field:
        ``"mmff94"`` or ``"uff"``.
    h_relaxed:
        Use H-relaxation protocol when ``True``.
    column:
        Output column name (default ``"strain_kcal"``).

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with strain column appended (``NaN`` on FF failure).
    """

    func = compute_strain_energy_h_relaxed if h_relaxed else compute_strain_energy
    strains = []
    for mol in mols:
        try:
            s = func(mol, force_field=force_field)
        except Exception:  # noqa: BLE001
            s = None
        strains.append(s)

    result = df.copy()
    result[column] = strains
    return result
