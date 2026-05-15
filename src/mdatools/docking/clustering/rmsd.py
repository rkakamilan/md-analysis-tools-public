"""RMSD matrix computation for docked poses.

Design note: ``align=False`` is the default and appropriate for docked poses
because all poses already share the same coordinate frame (the receptor).
Aligning before computing RMSD would distort the comparison.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def compute_rmsd_matrix(
    mols: list[Chem.Mol],
    align: bool = False,
) -> np.ndarray:
    """Compute a pairwise RMSD matrix for a list of docked poses.

    Parameters
    ----------
    mols:
        List of RDKit Mols, each with exactly one conformer and the same
        atom ordering (all poses must come from the same compound).
    align:
        If ``False`` (default), RMSD is computed without any coordinate
        alignment.  This is correct for docked poses that already share the
        receptor's coordinate frame.  Set to ``True`` only if poses were
        generated independently and need alignment.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` matrix of pairwise RMSD values in Ångströms.
    """
    n = len(mols)
    matrix = np.zeros((n, n))

    if align:
        from rdkit.Chem import AllChem

    for i in range(n):
        coords_i = mols[i].GetConformer().GetPositions()
        for j in range(i + 1, n):
            if align:
                mol_j = Chem.Mol(mols[j])
                rmsd = AllChem.GetBestRMS(mols[i], mol_j)
            else:
                coords_j = mols[j].GetConformer().GetPositions()
                rmsd = float(np.sqrt(np.mean(np.sum((coords_i - coords_j) ** 2, axis=1))))
            matrix[i, j] = rmsd
            matrix[j, i] = rmsd

    return matrix
