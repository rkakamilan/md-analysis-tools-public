"""Ligand-based virtual screening via fingerprint similarity.

Screens a compound library against one or more reference (query) molecules
using Morgan circular fingerprints and Tanimoto similarity.  Also provides
a MaxMin diversity picker for selecting structurally diverse subsets.

Example::

    from rdkit import Chem
    from mdatools.docking.lbvs import LBVSScreener, diverse_subset

    query = [Chem.MolFromSmiles("c1ccc(NC(=O)c2cccc(F)c2)cc1")]
    library = [Chem.MolFromSmiles(s) for s in smiles_list]

    screener = LBVSScreener(radius=2, n_bits=2048, threshold=0.4)
    results = screener.screen(query, library)
    for r in results[:10]:
        print(r.rank, r.score, Chem.MolToSmiles(r.mol))

    diverse = diverse_subset(library, n=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LBVSResult:
    """A single screening hit.

    Attributes
    ----------
    mol:
        RDKit Mol of the library compound.
    score:
        Maximum Tanimoto similarity to any query molecule (0–1).
    rank:
        1-based rank by score (descending).
    query_idx:
        Index of the closest query molecule.
    smiles:
        SMILES string of the library compound (set when screening from a
        DataFrame via :meth:`LBVSScreener.screen_df`).
    """

    mol: Any
    score: float
    rank: int = 0
    query_idx: int = 0
    smiles: str = ""


# ---------------------------------------------------------------------------
# LBVSScreener
# ---------------------------------------------------------------------------


class LBVSScreener:
    """Screen a compound library by fingerprint similarity to query molecules.

    Parameters
    ----------
    radius:
        Morgan fingerprint radius (default ``2`` = ECFP4).
    n_bits:
        Fingerprint bit length (default ``2048``).
    threshold:
        Minimum Tanimoto similarity to include a hit (default ``0.4``).
    metric:
        Similarity metric — currently only ``"tanimoto"`` is supported.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        threshold: float = 0.4,
        metric: str = "tanimoto",
    ) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.threshold = threshold
        self.metric = metric

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fp(self, mol: Any) -> Any:
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )
        return gen.GetFingerprint(mol)

    def _similarity(self, fp1: Any, fp2: Any) -> float:
        from rdkit import DataStructs  # noqa: PLC0415

        return float(DataStructs.TanimotoSimilarity(fp1, fp2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(
        self,
        query_mols: list,
        library_mols: list,
    ) -> list[LBVSResult]:
        """Screen *library_mols* against *query_mols*.

        Each library molecule is scored as the **maximum** Tanimoto
        similarity across all query molecules.  Results above
        :attr:`threshold` are returned, sorted by score descending.

        Parameters
        ----------
        query_mols:
            Reference active molecules.  ``None`` entries are skipped.
        library_mols:
            Molecules to screen.  ``None`` entries are skipped.

        Returns
        -------
        list[LBVSResult]
            Hits sorted by score descending, with 1-based ranks assigned.
        """
        try:
            from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415,F401
        except ImportError as exc:
            raise ImportError("RDKit is required for LBVS screening.") from exc

        query_fps = []
        for mol in query_mols:
            if mol is None:
                continue
            try:
                query_fps.append(self._fp(mol))
            except Exception:  # noqa: BLE001
                pass

        if not query_fps:
            return []

        hits: list[LBVSResult] = []
        for lib_mol in library_mols:
            if lib_mol is None:
                continue
            try:
                lib_fp = self._fp(lib_mol)
            except Exception:  # noqa: BLE001
                continue

            best_score = 0.0
            best_q_idx = 0
            for q_idx, q_fp in enumerate(query_fps):
                sim = self._similarity(q_fp, lib_fp)
                if sim > best_score:
                    best_score = sim
                    best_q_idx = q_idx

            if best_score >= self.threshold:
                hits.append(
                    LBVSResult(
                        mol=lib_mol,
                        score=round(best_score, 6),
                        query_idx=best_q_idx,
                    )
                )

        hits.sort(key=lambda r: r.score, reverse=True)
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits

    def screen_df(
        self,
        query_mols: list,
        df: Any,
        smiles_col: str = "smiles",
    ) -> Any:
        """Screen a DataFrame of compounds by SMILES column.

        Parameters
        ----------
        query_mols:
            Reference active molecules.
        df:
            DataFrame with at least *smiles_col*.
        smiles_col:
            Column containing SMILES strings.

        Returns
        -------
        pandas.DataFrame
            Copy of *df* with ``lbvs_score``, ``lbvs_rank``, and
            ``lbvs_query_idx`` columns added, filtered to hits above
            :attr:`threshold`, sorted by score descending.
        """
        import pandas as pd  # noqa: PLC0415

        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("RDKit is required for LBVS screening.") from exc

        if smiles_col not in df.columns:
            raise ValueError(f"Column {smiles_col!r} not found in DataFrame.")

        mols = [
            Chem.MolFromSmiles(s) if isinstance(s, str) else None
            for s in df[smiles_col]
        ]
        results = self.screen(query_mols, mols)

        # Build index mapping from mol identity → result
        result_map: dict[int, LBVSResult] = {id(r.mol): r for r in results}

        scores, ranks, q_idxs = [], [], []
        kept_indices = []
        for i, mol in enumerate(mols):
            if mol is not None and id(mol) in result_map:
                r = result_map[id(mol)]
                scores.append(r.score)
                ranks.append(r.rank)
                q_idxs.append(r.query_idx)
                kept_indices.append(i)

        if not kept_indices:
            return pd.DataFrame(
                columns=list(df.columns) + ["lbvs_score", "lbvs_rank", "lbvs_query_idx"]
            )

        out = df.iloc[kept_indices].copy()
        out["lbvs_score"] = scores
        out["lbvs_rank"] = ranks
        out["lbvs_query_idx"] = q_idxs
        return out.sort_values("lbvs_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diversity picker
# ---------------------------------------------------------------------------


def diverse_subset(
    mols: list,
    n: int,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 42,
) -> list:
    """Select a maximally diverse subset using the MaxMin algorithm.

    Picks *n* molecules from *mols* such that the minimum pairwise
    Tanimoto similarity among selected compounds is maximised.

    Parameters
    ----------
    mols:
        Pool of RDKit Mols.  ``None`` entries are skipped.
    n:
        Number of molecules to select.
    radius:
        Morgan fingerprint radius.
    n_bits:
        Fingerprint bit length.
    seed:
        Random seed for initial pick (default ``42``).

    Returns
    -------
    list[Mol]
        Selected diverse molecules (length ≤ *n*).
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415
        from rdkit.SimDivFilters import rdSimDivPickers  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("RDKit is required for diverse subset selection.") from exc

    valid = [m for m in mols if m is not None]
    if not valid:
        return []
    n = min(n, len(valid))

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprint(m) for m in valid]

    picker = rdSimDivPickers.MaxMinPicker()
    indices = list(picker.LazyBitVectorPick(fps, len(fps), n, seed=seed))
    return [valid[i] for i in indices]
