"""Scaffold hopping and R-group decomposition for library design.

Provides two complementary tools:

- :class:`ScaffoldAnalyzer` — extract Murcko scaffolds, compute scaffold
  frequency, and group a compound library by scaffold.
- :class:`RGroupDecomposer` — decompose a set of molecules against a shared
  core SMARTS to produce an R-group table, and enumerate analogs from
  combinatorial R-group sets.

Inspired by TeachOpenCADD T022 (substructure-based library search) and T005
(compound clustering).

Example::

    from rdkit import Chem
    from mdatools.docking.library_design.scaffold_hopping import (
        ScaffoldAnalyzer,
        RGroupDecomposer,
    )

    smiles = ["c1ccc(NC(=O)c2cccc(F)c2)cc1", "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    analyzer = ScaffoldAnalyzer()
    scaffolds = analyzer.extract_murcko(mols)
    freq_df = analyzer.scaffold_frequency(mols)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScaffoldAnalyzer
# ---------------------------------------------------------------------------


class ScaffoldAnalyzer:
    """Extract and analyse Murcko scaffolds from a compound library.

    All methods accept lists of RDKit Mol objects and skip ``None`` entries.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_murcko(self, mols: list) -> list:
        """Return the Murcko scaffold for each molecule.

        Parameters
        ----------
        mols:
            List of RDKit Mols.  ``None`` entries yield ``None`` in output.

        Returns
        -------
        list[Mol | None]
            Murcko scaffold Mol objects (same length as *mols*).
        """
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for scaffold analysis.") from exc

        scaffolds = []
        for mol in mols:
            if mol is None:
                scaffolds.append(None)
                continue
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(scaf)
            except Exception:  # noqa: BLE001
                scaffolds.append(None)
        return scaffolds

    def scaffold_frequency(self, mols: list) -> Any:
        """Compute scaffold frequency across a library.

        Parameters
        ----------
        mols:
            List of RDKit Mols.

        Returns
        -------
        pandas.DataFrame
            Columns: ``scaffold_smiles``, ``count``, ``fraction``.
            Sorted by *count* descending.
        """
        import pandas as pd  # noqa: PLC0415

        try:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for scaffold frequency.") from exc

        freq: dict[str, int] = {}
        n_valid = 0
        for mol in mols:
            if mol is None:
                continue
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                smi = Chem.MolToSmiles(scaf)
                freq[smi] = freq.get(smi, 0) + 1
                n_valid += 1
            except Exception:  # noqa: BLE001
                pass

        if not freq:
            return pd.DataFrame(columns=["scaffold_smiles", "count", "fraction"])

        rows = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        df = pd.DataFrame(rows, columns=["scaffold_smiles", "count"])
        df["fraction"] = df["count"] / n_valid
        return df

    def group_by_scaffold(self, mols: list) -> dict[str, list]:
        """Group molecules by their Murcko scaffold SMILES.

        Parameters
        ----------
        mols:
            List of RDKit Mols.

        Returns
        -------
        dict[str, list[Mol]]
            Mapping ``{scaffold_smiles: [mol, ...]}``.
            Molecules whose scaffold cannot be computed are omitted.
        """
        try:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for scaffold grouping.") from exc

        groups: dict[str, list] = {}
        for mol in mols:
            if mol is None:
                continue
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                smi = Chem.MolToSmiles(scaf)
                groups.setdefault(smi, []).append(mol)
            except Exception:  # noqa: BLE001
                pass
        return groups


# ---------------------------------------------------------------------------
# RGroupDecomposer
# ---------------------------------------------------------------------------


class RGroupDecomposer:
    """R-group decomposition and analog enumeration.

    Uses RDKit's ``RGroupDecomposition`` (rdkit.Chem.rdRGroupDecomposition)
    to match a shared core SMARTS and extract substituents.

    Parameters
    ----------
    sanitize:
        Sanitize enumerated analog molecules (default ``True``).
    """

    def __init__(self, sanitize: bool = True) -> None:
        self.sanitize = sanitize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, mols: list, core_smarts: str) -> Any:
        """Match *core_smarts* against each molecule and return an R-group table.

        Parameters
        ----------
        mols:
            List of RDKit Mols.
        core_smarts:
            SMARTS pattern defining the shared core scaffold.

        Returns
        -------
        pandas.DataFrame
            Columns: ``mol_idx``, ``core``, ``R1``, ``R2``, … (SMILES strings).
            Rows where the core did not match are omitted.
        """
        import pandas as pd  # noqa: PLC0415

        try:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem import rdRGroupDecomposition  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for R-group decomposition.") from exc

        core_mol = Chem.MolFromSmarts(core_smarts)
        if core_mol is None:
            raise ValueError(f"Invalid core SMARTS: {core_smarts!r}")

        valid_mols = [(i, m) for i, m in enumerate(mols) if m is not None]
        if not valid_mols:
            return pd.DataFrame(columns=["mol_idx", "core"])

        indices, mol_list = zip(*valid_mols)

        params = rdRGroupDecomposition.RGroupDecompositionParameters()
        params.removeHydrogensPostMatch = True
        decomp = rdRGroupDecomposition.RGroupDecomposition([core_mol], params)

        matched_indices = []
        for idx, (orig_idx, mol) in enumerate(zip(indices, mol_list)):
            result = decomp.Add(mol)
            if result >= 0:
                matched_indices.append(orig_idx)

        decomp.Process()
        rows_dict = decomp.GetRGroupsAsColumns()

        if not rows_dict or not matched_indices:
            return pd.DataFrame(columns=["mol_idx", "core"])

        # Convert Mol objects in rows_dict to SMILES strings
        n_rows = len(matched_indices)
        records = []
        for row_i in range(n_rows):
            rec: dict[str, Any] = {"mol_idx": matched_indices[row_i]}
            for col_name, mol_col in rows_dict.items():
                if row_i < len(mol_col):
                    m = mol_col[row_i]
                    rec[col_name] = Chem.MolToSmiles(m) if m is not None else None
                else:
                    rec[col_name] = None
            records.append(rec)

        return pd.DataFrame(records)

    def enumerate_analogs(
        self,
        core_smiles: str,
        rgroup_sets: dict[str, list[str]],
    ) -> list:
        """Generate analogs by attaching R-group combinations to a core.

        Parameters
        ----------
        core_smiles:
            SMILES of the core scaffold.  Attachment points are marked with
            ``[*]`` or dummy atoms (``[*:1]``, ``[*:2]``, …).
        rgroup_sets:
            Mapping ``{"R1": [smiles1, smiles2, ...], "R2": [...], ...}``.
            All combinations are enumerated (Cartesian product).

        Returns
        -------
        list[Mol]
            Valid, sanitized analog Mol objects.
            Combinations that fail sanitization are silently skipped.

        Notes
        -----
        This is a simple SMILES-concatenation approach.  For cores with
        labelled attachment points (``[*:1]``) use the label-aware variant.
        For production use, consider RDKit's ``AllChem.ReplaceSubstructs``.
        """
        try:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem import AllChem  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for analog enumeration.") from exc

        import itertools  # noqa: PLC0415

        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            raise ValueError(f"Invalid core SMILES: {core_smiles!r}")

        # Gather attachment point dummy atoms in the core
        dummy_atoms = [atom for atom in core_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        rgroup_keys = sorted(rgroup_sets.keys())

        if not dummy_atoms or not rgroup_keys:
            # No attachment points — return core as single analog
            return [core_mol]

        # Map each dummy atom to an R-group key by map number or position
        attachment_map: list[tuple[int, str]] = []
        for key, dummy in zip(rgroup_keys, dummy_atoms):
            attachment_map.append((dummy.GetIdx(), key))

        analogs: list = []
        combos = list(itertools.product(*[rgroup_sets[k] for k in rgroup_keys]))
        for combo in combos:
            rgroup_mols = [Chem.MolFromSmiles(s) for s in combo]
            if any(m is None for m in rgroup_mols):
                continue
            try:
                # Replace dummy atoms one by one using ReplaceSubstructs
                working = core_mol
                dummy_smarts = Chem.MolFromSmarts("[*]")
                for rg_mol in rgroup_mols:
                    replaced = AllChem.ReplaceSubstructs(working, dummy_smarts, rg_mol)
                    if replaced:
                        working = replaced[0]

                if self.sanitize:
                    Chem.SanitizeMol(working)
                analogs.append(working)
            except Exception:  # noqa: BLE001
                pass

        return analogs
