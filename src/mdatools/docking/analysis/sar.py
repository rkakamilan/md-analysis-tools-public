"""MCS-based SAR analysis and Murcko scaffold clustering.

Provides maximum common substructure (MCS) analysis and Bemis-Murcko
scaffold grouping for structure-activity relationship exploration.


Example::

    mcs = find_mcs(mols)
    scaffolds = cluster_by_scaffold(mols)
    df = add_scaffold_to_df(df, mols)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class MCSResult:
    """Result of a maximum common substructure search.

    Attributes
    ----------
    smarts:
        SMARTS string of the MCS.
    n_atoms:
        Number of atoms in the MCS.
    mol:
        RDKit Mol representation of the MCS (for drawing or substructure
        matching).
    """

    smarts: str
    n_atoms: int
    mol: Chem.Mol | None


def find_mcs(
    mols: list[Chem.Mol],
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
    timeout: int = 60,
) -> MCSResult:
    """Find the maximum common substructure of a set of molecules.

    Parameters
    ----------
    mols:
        List of RDKit Mols (at least 2).
    ring_matches_ring_only:
        Ring atoms can only match other ring atoms.
    complete_rings_only:
        Partial ring matches are not allowed.
    timeout:
        Maximum computation time in seconds.

    Returns
    -------
    MCSResult

    Raises
    ------
    ValueError
        If fewer than 2 molecules are provided.
    """
    if len(mols) < 2:
        raise ValueError("At least 2 molecules are required for MCS")

    result = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=complete_rings_only,
        timeout=timeout,
    )

    mcs_mol = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    return MCSResult(
        smarts=result.smartsString,
        n_atoms=result.numAtoms,
        mol=mcs_mol,
    )


def compute_pairwise_mcs_size(
    mols: list[Chem.Mol],
    min_atoms: int = 10,
    timeout: int = 10,
) -> pd.DataFrame:
    """Compute pairwise MCS atom counts for all molecule pairs.

    .. warning::
        This is O(n²) and very slow for >100 molecules.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    min_atoms:
        Minimum MCS atom count to include.
    timeout:
        Per-pair timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Columns: ``mol_i``, ``mol_j``, ``n_atoms_mcs``, ``smarts``.
    """
    rows = []
    n = len(mols)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                res = rdFMCS.FindMCS([mols[i], mols[j]], timeout=timeout)
                if res.numAtoms >= min_atoms:
                    rows.append(
                        {
                            "mol_i": i,
                            "mol_j": j,
                            "n_atoms_mcs": res.numAtoms,
                            "smarts": res.smartsString,
                        }
                    )
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows, columns=["mol_i", "mol_j", "n_atoms_mcs", "smarts"])


def get_murcko_scaffold(mol: Chem.Mol, generic: bool = False) -> Chem.Mol:
    """Extract the Bemis-Murcko scaffold from a molecule.

    Parameters
    ----------
    mol:
        RDKit Mol.
    generic:
        If ``True``, return the generic scaffold (all atoms → carbon,
        all bonds → single).  Useful for grouping across heteroatom variants.

    Returns
    -------
    Chem.Mol
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return scaffold


def cluster_by_scaffold(
    mols: list[Chem.Mol],
    generic: bool = False,
) -> dict[str, list[int]]:
    """Group molecules by their Murcko scaffold.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    generic:
        Use generic scaffolds.

    Returns
    -------
    dict[str, list[int]]
        ``{scaffold_smiles: [mol_indices]}``, sorted by cluster size
        (largest first).
    """
    clusters: dict[str, list[int]] = {}
    for idx, mol in enumerate(mols):
        scaffold = get_murcko_scaffold(mol, generic=generic)
        smi = Chem.MolToSmiles(scaffold)
        clusters.setdefault(smi, []).append(idx)
    return dict(sorted(clusters.items(), key=lambda kv: -len(kv[1])))


def add_scaffold_to_df(
    df: pd.DataFrame,
    mols: list[Chem.Mol],
    generic: bool = False,
    scaffold_col: str = "scaffold",
) -> pd.DataFrame:
    """Add a scaffold SMILES column to a DataFrame.

    Parameters
    ----------
    df:
        DataFrame with rows parallel to *mols*.
    mols:
        List of RDKit Mols.
    generic:
        Use generic scaffolds.
    scaffold_col:
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the scaffold column appended.
    """
    scaffolds = []
    for mol in mols:
        scaffold = get_murcko_scaffold(mol, generic=generic)
        scaffolds.append(Chem.MolToSmiles(scaffold))
    result = df.copy()
    result[scaffold_col] = scaffolds
    return result
