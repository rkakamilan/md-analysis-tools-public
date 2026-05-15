"""Molecular standardization pipeline for VS library preprocessing.

Ensures consistent molecular representations by stripping salts,
neutralizing charges, and optionally canonicalizing tautomers.

Example::

    mol = standardize_mol(mol, strip_salts=True, neutralize=True)
    df = standardize_df(df, smiles_col="SMILES")
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def strip_salts(mol: Chem.Mol) -> Chem.Mol:
    """Remove counter-ions by keeping the largest fragment.

    Parameters
    ----------
    mol:
        RDKit Mol (may contain multiple fragments, e.g. ``"CC.[Na+]"``).

    Returns
    -------
    Chem.Mol
        Largest fragment by heavy atom count.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


# SMARTS rules for common charge neutralization
_NEUTRALIZE_REACTIONS = [
    ("[n+;H]", "n"),           # quaternary N in aromatic ring
    ("[N+;!H0]", "N"),         # protonated amine
    ("[$([O-]);!$([O-][#7])]", "O"),  # carboxylate / phenolate (not nitro)
    ("[S-;X1]", "S"),          # thiolate
    ("[$([N-;X2]S(=O)=O)]", "N"),  # sulfonamide anion
    ("[$([N-;X2][C,N]=C)]", "N"),  # amidine/guanidine anion
]


def neutralize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize common charged groups via SMARTS transformations.

    Removes protons from protonated amines and adds protons to
    deprotonated carboxylates / thiolates.

    Parameters
    ----------
    mol:
        RDKit Mol.

    Returns
    -------
    Chem.Mol
        Neutralized molecule.
    """
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def normalize_tautomer(mol: Chem.Mol, max_transforms: int = 1000) -> Chem.Mol:
    """Select the canonical tautomer using RDKit's TautomerEnumerator.

    Parameters
    ----------
    mol:
        RDKit Mol.
    max_transforms:
        Maximum number of tautomeric transforms to consider.

    Returns
    -------
    Chem.Mol
        Canonical tautomer.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTransforms(max_transforms)
    return enumerator.Canonicalize(mol)


def standardize_mol(
    mol: Chem.Mol,
    do_strip_salts: bool = True,
    do_neutralize: bool = True,
    do_normalize_tautomer: bool = False,
) -> Chem.Mol:
    """Run the full standardization pipeline: salt → neutralize → tautomer.

    Parameters
    ----------
    mol:
        RDKit Mol.
    do_strip_salts:
        Remove counter-ions (default ``True``).
    do_neutralize:
        Neutralize common charged groups (default ``True``).
    do_normalize_tautomer:
        Canonicalize tautomers (default ``False`` — expensive).

    Returns
    -------
    Chem.Mol
        Standardized molecule.
    """
    if do_strip_salts:
        mol = strip_salts(mol)
    if do_neutralize:
        mol = neutralize_mol(mol)
    if do_normalize_tautomer:
        mol = normalize_tautomer(mol)
    return mol


def standardize_df(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    output_col: str = "std_SMILES",
    do_strip_salts: bool = True,
    do_neutralize: bool = True,
    do_normalize_tautomer: bool = False,
) -> pd.DataFrame:
    """Standardize all molecules in a DataFrame.

    Parameters
    ----------
    df:
        DataFrame with a SMILES column.
    smiles_col:
        Name of the input SMILES column.
    output_col:
        Name of the output standardized SMILES column.
    do_strip_salts, do_neutralize, do_normalize_tautomer:
        Passed to :func:`standardize_mol`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *output_col* appended.  Failed rows get ``None``.
    """
    result = df.copy()
    std_smiles = []

    for smi in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                std_smiles.append(None)
                continue
            mol = standardize_mol(
                mol,
                do_strip_salts=do_strip_salts,
                do_neutralize=do_neutralize,
                do_normalize_tautomer=do_normalize_tautomer,
            )
            std_smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            std_smiles.append(None)

    result[output_col] = std_smiles
    return result
