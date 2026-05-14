"""Composable pose filters for post-docking selection.

Filters are chained with :func:`apply_filters`.  Each filter returns a
boolean mask over the poses DataFrame; masks are combined with AND or OR
logic.

Ported from ``docking_analysis.selection.filters``.

Example::

    from mdatools.docking.selection.filters import (
        ScoreFilter, InteractionFilter, apply_filters
    )

    filters = [
        ScoreFilter(threshold=-7.0),
        InteractionFilter(residues=["GLN30", "ARG38"]),
    ]
    filtered_df, filtered_mols = apply_filters(df, mols, filters)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PoseFilter(ABC):
    """Abstract base class for pose filters.

    Subclasses implement :meth:`apply` (DataFrame-only) and may override
    :meth:`apply_with_mols` when 3-D coordinates are needed.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask aligned with *df*.

        Parameters
        ----------
        df:
            Poses DataFrame.

        Returns
        -------
        pd.Series
            Boolean Series with the same index as *df*.
        """

    def apply_with_mols(
        self,
        df: pd.DataFrame,
        mols: list,
    ) -> pd.Series:
        """Return a boolean mask with access to 3-D mol objects.

        Default delegates to :meth:`apply`.  Override in filters that
        need conformer coordinates.
        """
        return self.apply(df)


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------


@dataclass
class ScoreFilter(PoseFilter):
    """Keep poses with docking score ≤ *threshold*.

    Parameters
    ----------
    threshold:
        Maximum (best) docking score to keep (kcal/mol).
    score_col:
        DataFrame column name for the docking score.
    """

    threshold: float = -7.0
    score_col: str = "docking_score"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.score_col not in df.columns:
            return pd.Series(True, index=df.index)
        return df[self.score_col] <= self.threshold


@dataclass
class InteractionFilter(PoseFilter):
    """Require interactions with specified residues.

    Parameters
    ----------
    residues:
        List of residue identifiers (e.g. ``["GLN30", "ARG38"]``).
    logic:
        ``"AND"`` — all residues must interact; ``"OR"`` — any suffices.
    interaction_col_prefix:
        Column prefix as in ProLIF output (``"<RESID>_*"``).
    """

    residues: list[str] = field(default_factory=list)
    logic: Literal["AND", "OR"] = "AND"
    interaction_col_prefix: str = ""

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if not self.residues:
            return pd.Series(True, index=df.index)

        masks: list[pd.Series] = []
        for res in self.residues:
            cols = [c for c in df.columns if c.startswith(res + "_")]
            if cols:
                masks.append(df[cols].any(axis=1).astype(bool))
            else:
                # Residue has no IFP column → treat as no interaction
                masks.append(pd.Series(False, index=df.index))

        if not masks:
            return pd.Series(True, index=df.index)

        combined = masks[0]
        for m in masks[1:]:
            combined = (combined & m) if self.logic == "AND" else (combined | m)
        return combined


@dataclass
class ClusterRepresentativeFilter(PoseFilter):
    """Keep the best-scoring pose per cluster.

    Parameters
    ----------
    cluster_col:
        DataFrame column holding cluster labels.
    score_col:
        DataFrame column for ranking within a cluster (lower = better).
    """

    cluster_col: str = "cluster"
    score_col: str = "docking_score"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.cluster_col not in df.columns or self.score_col not in df.columns:
            return pd.Series(True, index=df.index)
        best_idx = df.groupby(self.cluster_col)[self.score_col].idxmin()
        mask = pd.Series(False, index=df.index)
        mask.loc[best_idx.values] = True
        return mask


@dataclass
class PropertyFilter(PoseFilter):
    """Keep poses satisfying molecular property thresholds.

    All criteria are checked; any failure excludes the pose.
    Pass ``None`` to skip a threshold.

    Parameters
    ----------
    max_mw:
        Maximum molecular weight (Da).
    max_logp:
        Maximum LogP.
    min_le:
        Minimum Ligand Efficiency (|score| / HAC).
    require_ro5:
        Require Lipinski Ro5 compliance.
    require_veber:
        Require Veber oral-bioavailability compliance.
    """

    max_mw: float | None = None
    max_logp: float | None = None
    min_le: float | None = None
    require_ro5: bool = False
    require_veber: bool = False

    def apply(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if self.max_mw is not None and "mw" in df.columns:
            mask &= df["mw"] <= self.max_mw
        if self.max_logp is not None and "logp" in df.columns:
            mask &= df["logp"] <= self.max_logp
        if self.min_le is not None and "le" in df.columns:
            mask &= df["le"].fillna(0) >= self.min_le
        if self.require_ro5 and "lipinski_pass" in df.columns:
            mask &= df["lipinski_pass"].astype(bool)
        if self.require_veber and "veber_pass" in df.columns:
            mask &= df["veber_pass"].astype(bool)
        return mask


@dataclass
class SubstructureFilter(PoseFilter):
    """SMARTS-based inclusion/exclusion filter.

    Parameters
    ----------
    include_smarts:
        Poses must contain all (match_all=True) or any (match_all=False)
        of these SMARTS patterns.
    exclude_smarts:
        Poses containing any of these patterns are excluded.
    match_all:
        Require ALL include patterns when ``True`` (AND logic).
    """

    include_smarts: list[str] = field(default_factory=list)
    exclude_smarts: list[str] = field(default_factory=list)
    match_all: bool = True

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    def apply_with_mols(self, df: pd.DataFrame, mols: list) -> pd.Series:
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            return pd.Series(True, index=df.index)

        include_pats = [Chem.MolFromSmarts(s) for s in self.include_smarts if s]
        exclude_pats = [Chem.MolFromSmarts(s) for s in self.exclude_smarts if s]

        mask = []
        for mol in mols:
            if mol is None:
                mask.append(False)
                continue
            ok = True
            if exclude_pats:
                ok = not any(mol.HasSubstructMatch(p) for p in exclude_pats if p)
            if ok and include_pats:
                if self.match_all:
                    ok = all(mol.HasSubstructMatch(p) for p in include_pats if p)
                else:
                    ok = any(mol.HasSubstructMatch(p) for p in include_pats if p)
            mask.append(ok)
        return pd.Series(mask, index=df.index)


@dataclass
class StructuralAlertFilter(PoseFilter):
    """RDKit FilterCatalog-based structural alert filter.

    Parameters
    ----------
    catalogs:
        One or more of ``"PAINS"``, ``"BRENK"``, ``"NIH"``, ``"REOS"``.
    mode:
        ``"exclude"`` — remove flagged poses.
        ``"flag"`` — keep all but add ``has_alert`` Boolean column.
        ``"annotate"`` — keep all but add ``alert_names`` string column.
    """

    catalogs: list[str] = field(default_factory=lambda: ["PAINS"])
    mode: Literal["exclude", "flag", "annotate"] = "exclude"

    def _build_catalog(self):
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # noqa: PLC0415

        params = FilterCatalogParams()
        mapping = {
            "PAINS": FilterCatalogParams.FilterCatalogs.PAINS,
            "BRENK": FilterCatalogParams.FilterCatalogs.BRENK,
            "NIH": FilterCatalogParams.FilterCatalogs.NIH,
            "REOS": FilterCatalogParams.FilterCatalogs.RECON,
        }
        for cat in self.catalogs:
            if cat.upper() in mapping:
                params.AddCatalog(mapping[cat.upper()])
        return FilterCatalog(params)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    def apply_with_mols(self, df: pd.DataFrame, mols: list) -> pd.Series:
        try:
            catalog = self._build_catalog()
        except ImportError:
            return pd.Series(True, index=df.index)

        mask = []
        for mol in mols:
            if mol is None:
                mask.append(self.mode != "exclude")
                continue
            entries = catalog.GetMatches(mol)
            has_alert = len(entries) > 0
            if self.mode == "exclude":
                mask.append(not has_alert)
            else:
                mask.append(True)
        return pd.Series(mask, index=df.index)


@dataclass
class AggregatorFilter(PoseFilter):
    """Flag lipophilic compounds similar to known colloidal aggregators.

    Parameters
    ----------
    max_slogp:
        Compounds with SLogP > this value are candidates.
    similarity_threshold:
        Morgan FP Tanimoto similarity to any known aggregator.
    aggregator_smiles:
        Reference aggregator SMILES.  Defaults to a small curated list.
    """

    max_slogp: float = 3.5
    similarity_threshold: float = 0.4
    aggregator_smiles: list[str] = field(default_factory=list)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    def apply_with_mols(self, df: pd.DataFrame, mols: list) -> pd.Series:
        try:
            from rdkit import Chem, DataStructs  # noqa: PLC0415
            from rdkit.Chem import Descriptors, rdFingerprintGenerator  # noqa: PLC0415
        except ImportError:
            return pd.Series(True, index=df.index)

        ref_smiles = self.aggregator_smiles or [
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen-like
            "c1ccc(cc1)CC(=O)O",
        ]
        ref_mols = [Chem.MolFromSmiles(s) for s in ref_smiles if s]
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        ref_fps = [gen.GetFingerprint(m) for m in ref_mols if m]

        mask = []
        for mol in mols:
            if mol is None:
                mask.append(True)
                continue
            slogp = Descriptors.MolLogP(mol)
            if slogp <= self.max_slogp:
                mask.append(True)
                continue
            fp = gen.GetFingerprint(mol)
            if (
                ref_fps
                and max(DataStructs.TanimotoSimilarity(fp, r) for r in ref_fps)
                >= self.similarity_threshold
            ):
                mask.append(False)
            else:
                mask.append(True)
        return pd.Series(mask, index=df.index)


@dataclass
class StrainEnergyFilter(PoseFilter):
    """Keep poses below a conformational strain threshold.

    Delegates to :func:`~mdatools.docking.analysis.strain.compute_strain_energy`.

    Parameters
    ----------
    max_strain:
        Maximum acceptable strain energy (kcal/mol).
    h_relaxed:
        Use the H-relaxation protocol (recommended for macrocycles).
    pass_nan:
        When ``True`` (default), poses where strain cannot be computed
        (FF failure) are passed through rather than excluded.
    """

    max_strain: float = 20.0
    h_relaxed: bool = False
    pass_nan: bool = True

    def apply(self, df: pd.DataFrame) -> pd.Series:
        col = "strain_kcal"
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        strain = df[col]
        if self.pass_nan:
            return (strain <= self.max_strain) | strain.isna()
        return strain.fillna(float("inf")) <= self.max_strain

    def apply_with_mols(self, df: pd.DataFrame, mols: list) -> pd.Series:
        # If strain is already in df, use the DataFrame-only path
        if "strain_kcal" in df.columns:
            return self.apply(df)

        # Compute on-the-fly
        try:
            from mdatools.docking.analysis.strain import (  # noqa: PLC0415
                compute_strain_energy,
                compute_strain_energy_h_relaxed,
            )
        except ImportError:
            return pd.Series(True, index=df.index)

        func = (
            compute_strain_energy_h_relaxed if self.h_relaxed else compute_strain_energy
        )
        mask = []
        for mol in mols:
            if mol is None:
                mask.append(self.pass_nan)
                continue
            try:
                s = func(mol)
                if s is None:
                    mask.append(self.pass_nan)
                else:
                    mask.append(s <= self.max_strain)
            except Exception:  # noqa: BLE001
                mask.append(self.pass_nan)
        return pd.Series(mask, index=df.index)


@dataclass
class ParetoFilter(PoseFilter):
    """Keep compounds within the specified Pareto rank.

    Parameters
    ----------
    objective_cols:
        DataFrame columns to use as Pareto objectives.
    directions:
        ``"min"`` or ``"max"`` per column.
    max_rank:
        Keep only compounds with Pareto rank ≤ *max_rank*.
    """

    objective_cols: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    max_rank: int = 1

    def apply(self, df: pd.DataFrame) -> pd.Series:
        from mdatools.docking.selection.pareto import compute_pareto_rank  # noqa: PLC0415

        if not self.objective_cols:
            return pd.Series(True, index=df.index)

        available = [c for c in self.objective_cols if c in df.columns]
        if not available:
            return pd.Series(True, index=df.index)

        directions = self.directions if self.directions else ["min"] * len(available)
        matrix = df[available].fillna(0).values.astype(float)
        ranks = compute_pareto_rank(matrix, directions[: len(available)])
        return pd.Series(ranks <= self.max_rank, index=df.index)


@dataclass
class ECFilter(PoseFilter):
    """Keep poses with Electrostatic Complementarity ≥ cutoff.

    Parameters
    ----------
    min_ec:
        Minimum EC score (default 0.29 based on literature benchmarks).
    ec_col:
        DataFrame column name for the EC score.
    """

    min_ec: float = 0.29
    ec_col: str = "ec_score"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.ec_col not in df.columns:
            return pd.Series(True, index=df.index)
        return df[self.ec_col] >= self.min_ec


# ---------------------------------------------------------------------------
# apply_filters utility
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    mols: list,
    filters: list[PoseFilter],
    logic: Literal["AND", "OR"] = "AND",
) -> tuple[pd.DataFrame, list]:
    """Apply a list of PoseFilters to a poses DataFrame.

    Parameters
    ----------
    df:
        Poses DataFrame (rows parallel to *mols*).
    mols:
        RDKit Mol list.
    filters:
        Ordered list of :class:`PoseFilter` instances.
    logic:
        ``"AND"`` — all filters must pass; ``"OR"`` — any filter passes.

    Returns
    -------
    tuple[pd.DataFrame, list]
        Filtered ``(DataFrame, mols)`` pair.
    """
    if not filters:
        return df, mols

    combined_mask = pd.Series(logic == "AND", index=df.index)

    for f in filters:
        mask = f.apply_with_mols(df, mols)
        mask = mask.reindex(df.index, fill_value=True if logic == "OR" else False)
        if logic == "AND":
            combined_mask &= mask
        else:
            combined_mask |= mask

    # Use positional indices to keep mols aligned with DataFrame rows
    kept_positions = [i for i, (idx, keep) in enumerate(combined_mask.items()) if keep]
    filtered_df = df[combined_mask].copy()
    filtered_mols = [mols[i] for i in kept_positions]

    return filtered_df, filtered_mols
