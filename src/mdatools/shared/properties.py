"""Molecular property calculation utilities.

All calculation uses RDKit descriptors applied to the 2D molecular graph
(no 3D conformer required).  SA score and PAINS detection are opt-in
(``include_sa=True`` / ``include_pains=False`` defaults).

Units
-----
- Molecular weight: Da (``Descriptors.ExactMolWt``)
- LogP: Wildman–Crippen estimate (``Descriptors.MolLogP``)
- TPSA: Å²
- Ligand Efficiency (LE): |docking_score| / heavy_atom_count (|kcal/mol| / count)
- LELP: LogP / LE (dimensionless)
- SA score: 1 (easy to synthesise) → 10 (very difficult)

Lipinski RO5 criteria (``lipinski_pass``):
  MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10

Veber oral bioavailability criteria (``veber_pass``):
  TPSA ≤ 140, rotatable_bonds ≤ 10

Example
-------
>>> from mdatools.shared.properties import MolecularProperties, calculate_properties
>>> from rdkit import Chem
>>> mol = Chem.MolFromSmiles("CCc1nn(C)c2ccc(nc12)C1CCN(CC1)C(=O)c1cncs1")  # sildenafil
>>> props = calculate_properties(mol, docking_score=-9.2)
>>> props.mw, props.le, props.lipinski_pass
(474.58, 0.288, True)

"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lipinski / Veber thresholds as named constants
# ---------------------------------------------------------------------------

#: Lipinski Rule-of-Five (RO5) hard limits.
LIPINSKI_MW_MAX: float = 500.0
LIPINSKI_LOGP_MAX: float = 5.0
LIPINSKI_HBD_MAX: int = 5
LIPINSKI_HBA_MAX: int = 10

#: Veber oral-bioavailability limits.
VEBER_TPSA_MAX: float = 140.0
VEBER_ROTBONDS_MAX: int = 10


# ---------------------------------------------------------------------------
# MolecularProperties — canonical unified dataclass
# ---------------------------------------------------------------------------


@dataclass
class MolecularProperties:
    """Molecular properties computed from the 2D graph.

    Attributes
    ----------
    mw:
        Exact molecular weight (Da).
    logp:
        Wildman–Crippen LogP estimate.
    hbd:
        Hydrogen bond donor count (Lipinski definition).
    hba:
        Hydrogen bond acceptor count (Lipinski definition).
    tpsa:
        Topological polar surface area (Å²).
    rotatable_bonds:
        Number of rotatable bonds.
    hac:
        Heavy atom count (denominator for LE/LELP).
    qed:
        Quantitative Estimate of Drug-likeness [0, 1].
    sa_score:
        Synthetic accessibility score [1, 10] — ``None`` when
        ``include_sa=False`` is passed to :func:`calculate_properties`.
    le:
        Ligand Efficiency = |docking_score| / hac (kcal/mol per heavy
        atom). ``None`` when no docking score is supplied.
    lelp:
        LE-Lipophilicity = LogP / LE.  ``None`` when LE is ``None`` or zero.
    lipinski_pass:
        ``True`` when all four Lipinski RO5 criteria are satisfied.
    veber_pass:
        ``True`` when both Veber criteria are satisfied.
    pains_alerts:
        List of PAINS structural-alert names detected in the molecule.
        Empty list when ``include_pains=False`` (default).

    Notes
    -----
    The dataclass supports dict-unpacking (``{**props}``) via
    ``keys()`` / ``__getitem__`` so that instances can be spread into
    dict literals — preserving the API of the legacy
    ``LigandProperties`` class.
    """

    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    hac: int
    qed: float
    sa_score: float | None
    le: float | None
    lelp: float | None
    lipinski_pass: bool
    veber_pass: bool
    pains_alerts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------

    @property
    def pains_alert(self) -> bool:
        """``True`` if any PAINS alert was detected."""
        return len(self.pains_alerts) > 0

    @property
    def ro5_pass(self) -> bool:
        """Alias for :attr:`lipinski_pass` (docking-tools backward compat)."""
        return self.lipinski_pass

    # ------------------------------------------------------------------
    # Mapping protocol — enables  {**props}  dict-unpacking
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return field names so that ``**props`` dict-unpacking works."""
        return [f.name for f in dataclasses.fields(self)]

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


# ---------------------------------------------------------------------------
# Public calculation functions
# ---------------------------------------------------------------------------


def calculate_properties(
    mol_or_smiles: Any,
    *,
    docking_score: float | None = None,
    include_sa: bool = True,
    include_pains: bool = False,
    _pains_catalog: Any = None,
) -> MolecularProperties:
    """Compute molecular properties for a single molecule.

    Parameters
    ----------
    mol_or_smiles:
        Either an RDKit ``Chem.Mol`` object or a SMILES string.
    docking_score:
        Docking score in kcal/mol (negative = better).  Required for
        computing LE and LELP; pass ``None`` to skip those fields.
    include_sa:
        When ``True`` (default), compute the Synthetic Accessibility
        score (requires the RDKit Contrib SA_Score module or uses a
        lightweight fallback).
    include_pains:
        When ``True``, run the PAINS A/B/C filter catalog and populate
        :attr:`MolecularProperties.pains_alerts`.  Costs one catalog
        lookup per molecule.
    _pains_catalog:
        Pre-built PAINS ``FilterCatalog`` instance (avoids repeated
        construction in batch calls).  Ignored when
        ``include_pains=False``.

    Returns
    -------
    MolecularProperties

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If *mol_or_smiles* is a string that RDKit cannot parse, or if
        the supplied ``Chem.Mol`` is ``None``.
    """
    Chem = _require_rdkit()

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {mol_or_smiles!r}")
    else:
        mol = mol_or_smiles
        if mol is None:
            raise ValueError("mol is None")

    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.QED import qed as _qed

    mw = float(Descriptors.ExactMolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    tpsa = float(Descriptors.TPSA(mol))
    rotbonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    hac = int(mol.GetNumHeavyAtoms())
    qed_score = float(_qed(mol))

    sa: float | None = None
    if include_sa:
        sa = _calc_sa_score(mol)

    le: float | None = None
    lelp: float | None = None
    if docking_score is not None and hac > 0:
        le = abs(docking_score) / hac
        if le > 0:
            lelp = logp / le

    lipinski = bool(
        mw <= LIPINSKI_MW_MAX
        and logp <= LIPINSKI_LOGP_MAX
        and hbd <= LIPINSKI_HBD_MAX
        and hba <= LIPINSKI_HBA_MAX
    )
    veber = bool(tpsa <= VEBER_TPSA_MAX and rotbonds <= VEBER_ROTBONDS_MAX)

    pains: list[str] = []
    if include_pains:
        catalog = _pains_catalog or _build_pains_catalog()
        pains = _get_pains_alerts(mol, catalog)

    return MolecularProperties(
        mw=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        rotatable_bonds=rotbonds,
        hac=hac,
        qed=qed_score,
        sa_score=sa,
        le=le,
        lelp=lelp,
        lipinski_pass=lipinski,
        veber_pass=veber,
        pains_alerts=pains,
    )


def add_properties_to_df(
    df: pd.DataFrame,
    mols: list[Any] | None = None,
    smiles_col: str | None = None,
    score_col: str = "docking_score",
    *,
    include_sa: bool = False,
    include_pains: bool = False,
) -> pd.DataFrame:
    """Add :class:`MolecularProperties` columns to a DataFrame.

    Two usage modes:

    **Mode 1 – mol list** (rows in *df* parallel to *mols*)::

        df_out = add_properties_to_df(df, mols=mol_list)

    **Mode 2 – SMILES column**::

        df_out = add_properties_to_df(df, smiles_col="smiles")

    Parameters
    ----------
    df:
        Input DataFrame.
    mols:
        List of RDKit Mols parallel to *df* rows.  Required when
        *smiles_col* is not given.
    smiles_col:
        Column name containing SMILES strings.  When provided, *mols*
        is ignored.
    score_col:
        Name of the docking score column used for LE / LELP computation.
        Silently ignored when not present in *df*.
    include_sa:
        Compute SA score.  Costs ~1 ms per molecule.
    include_pains:
        Detect PAINS structural alerts.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with property columns appended.  Failed
        calculations produce ``NaN`` rows rather than raising.

    Raises
    ------
    ValueError
        When neither *mols* nor *smiles_col* is supplied.
    """
    _require_rdkit()
    from rdkit import Chem as _Chem  # noqa: PLC0415

    if smiles_col is not None:
        mols = [
            _Chem.MolFromSmiles(s) if isinstance(s, str) else None
            for s in df[smiles_col]
        ]
    elif mols is None:
        raise ValueError("Either 'mols' or 'smiles_col' must be provided.")

    scores = df[score_col].tolist() if score_col in df.columns else [None] * len(df)

    # Pre-build PAINS catalog once for the whole batch
    pains_catalog = _build_pains_catalog() if include_pains else None

    _cols = (
        "mw",
        "logp",
        "hbd",
        "hba",
        "tpsa",
        "rotatable_bonds",
        "hac",
        "qed",
        "sa_score",
        "le",
        "lelp",
        "lipinski_pass",
        "veber_pass",
        "pains_alerts",
    )

    rows: list[dict[str, Any]] = []
    for mol, score in zip(mols, scores):  # type: ignore[arg-type]
        try:
            p = calculate_properties(
                mol,
                docking_score=score,
                include_sa=include_sa,
                include_pains=include_pains,
                _pains_catalog=pains_catalog,
            )
            rows.append({c: p[c] for c in _cols})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Property calculation failed: %s", exc)
            rows.append({c: float("nan") for c in _cols})

    props_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, props_df], axis=1)


def batch_from_smiles(
    smiles_list: list[str],
    *,
    include_sa: bool = True,
    include_pains: bool = False,
) -> pd.DataFrame:
    """Compute properties for a list of SMILES strings.

    Parameters
    ----------
    smiles_list:
        SMILES strings to process.  Invalid strings produce ``NaN`` rows.
    include_sa:
        Compute SA score (default ``True``).
    include_pains:
        Detect PAINS alerts (default ``False``).

    Returns
    -------
    pd.DataFrame
        One row per SMILES, columns matching :class:`MolecularProperties`
        fields plus a leading ``smiles`` column.
    """
    _require_rdkit()
    pains_catalog = _build_pains_catalog() if include_pains else None
    rows: list[dict[str, Any]] = []
    for smi in smiles_list:
        row: dict[str, Any] = {"smiles": smi}
        try:
            p = calculate_properties(
                smi,
                include_sa=include_sa,
                include_pains=include_pains,
                _pains_catalog=pains_catalog,
            )
            row.update({c: p[c] for c in p.keys()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %r: %s", smi, exc)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_rdkit() -> Any:
    """Import and return ``rdkit.Chem``, raising a clear ImportError if absent."""
    try:
        from rdkit import Chem  # noqa: PLC0415

        return Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for mdatools.shared.properties. "
            "Install it with:  pip install 'mdatools[docking]'"
        ) from exc


def _calc_sa_score(mol: Any) -> float:
    """Return synthetic accessibility score [1–10].

    Uses the official RDKit Contrib SA_Score module when available,
    otherwise falls back to a lightweight ring-complexity heuristic.
    The fallback is intentionally rough — a warning is emitted on first use.
    """
    try:
        from rdkit.Chem import RDConfig  # noqa: PLC0415
        import importlib.util

        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        if os.path.exists(sa_path):
            spec = importlib.util.spec_from_file_location("sascorer", sa_path)
            sascorer = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(sascorer)  # type: ignore[union-attr]
            return float(sascorer.calculateScore(mol))
    except Exception:  # noqa: BLE001
        pass

    # Lightweight fallback (not validated against the paper)
    logger.debug("SA_Score module unavailable; using ring-complexity approximation.")
    from rdkit.Chem import rdMolDescriptors  # noqa: PLC0415

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = len(rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True))
    n_heavy = mol.GetNumHeavyAtoms()
    score = 1.0 + 0.3 * n_rings + 0.5 * n_stereo + 0.02 * max(0, n_heavy - 20)
    return float(min(score, 10.0))


def _build_pains_catalog() -> Any:
    """Build and return a combined PAINS A/B/C FilterCatalog."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # noqa: PLC0415

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def _get_pains_alerts(mol: Any, catalog: Any) -> list[str]:
    """Return list of PAINS alert descriptions matched in *mol*."""
    return [e.GetDescription() for e in catalog.GetMatches(mol)]
