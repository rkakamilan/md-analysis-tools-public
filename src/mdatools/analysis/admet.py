"""Ligand ADMET and drug-likeness property calculator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ADMETResult:
    """Drug-likeness and ADMET properties for a single molecule.

    Attributes
    ----------
    name:
        Molecule identifier (SMILES string or filename stem).
    mw:
        Molecular weight (Da).
    logp:
        Wildman-Crippen LogP estimate.
    hbd:
        Number of hydrogen bond donors.
    hba:
        Number of hydrogen bond acceptors.
    tpsa:
        Topological polar surface area (Å²).
    rotbonds:
        Number of rotatable bonds.
    qed:
        Quantitative Estimate of Drug-likeness (0–1; higher = more drug-like).
    sa_score:
        Synthetic Accessibility score (1 = easy, 10 = very difficult).
    lipinski_pass:
        True if all Lipinski Ro5 criteria are satisfied.
    veber_pass:
        True if Veber oral bioavailability criteria are satisfied.
    pains_alert:
        True if at least one PAINS structural alert is detected.
    pains_patterns:
        List of matched PAINS alert names.
    """

    name: str
    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotbonds: int
    qed: float
    sa_score: float
    lipinski_pass: bool
    veber_pass: bool
    pains_alert: bool
    pains_patterns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class ADMETCalculator:
    """Calculate ADMET and drug-likeness properties using RDKit.

    Requires ``rdkit`` (optional extra ``admet``). Raises :class:`ImportError`
    with an actionable message when RDKit is not installed.

    Parameters
    ----------
    check_pains:
        When ``True`` (default) run the PAINS A/B/C filter catalog.
    """

    def __init__(self, check_pains: bool = True) -> None:
        self._check_pains = check_pains
        self._pains_catalog = None  # lazily built on first use

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def from_smiles(self, smiles: str, name: str = "") -> ADMETResult:
        """Calculate properties from a SMILES string.

        Parameters
        ----------
        smiles:
            Valid SMILES representation.
        name:
            Optional label stored in the returned :class:`ADMETResult`.
        """
        Chem = _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        return self._calc(mol, name=name or smiles)

    def from_mol(self, mol: object, name: str = "") -> ADMETResult:
        """Calculate properties from an RDKit ``Mol`` object.

        Parameters
        ----------
        mol:
            RDKit ``Chem.Mol`` instance.
        name:
            Optional label stored in the returned :class:`ADMETResult`.
        """
        _require_rdkit()
        return self._calc(mol, name=name)

    def from_pdb(
        self,
        pdb_path: Path | str,
        resname: str = "",
        name: str = "",
    ) -> ADMETResult:
        """Calculate properties by reading a ligand from a PDB file.

        Parameters
        ----------
        pdb_path:
            Path to the PDB file.
        resname:
            Residue name to extract (unused if single-residue PDB). When
            empty, the first residue in the file is used.
        name:
            Optional label; defaults to the file stem.
        """
        Chem = _require_rdkit()
        pdb_path = Path(pdb_path)
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=True)
        if mol is None:
            raise ValueError(f"RDKit could not parse PDB file: {pdb_path}")
        label = name or pdb_path.stem
        return self._calc(mol, name=label)

    def batch(self, smiles_list: list[str]) -> "pd.DataFrame":
        """Calculate properties for a list of SMILES strings.

        Returns
        -------
        :class:`pandas.DataFrame` with one row per molecule and columns
        matching the fields of :class:`ADMETResult`.
        """
        import pandas as pd

        rows = []
        for smi in smiles_list:
            try:
                r = self.from_smiles(smi)
                rows.append(vars(r))
            except ValueError as exc:
                logger.warning("Skipping invalid SMILES %r: %s", smi, exc)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal calculation
    # ------------------------------------------------------------------

    def _calc(self, mol: object, name: str) -> ADMETResult:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.QED import qed as _qed

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        qed_score = float(_qed(mol))
        sa = _calc_sa_score(mol)

        lipinski_pass = bool(
            mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
        )
        veber_pass = bool(tpsa <= 140 and rotbonds <= 10)

        pains_names: list[str] = []
        if self._check_pains:
            pains_names = _check_pains(mol, self._get_pains_catalog())
        pains_alert = len(pains_names) > 0

        logger.debug(
            "ADMET for %s: MW=%.1f LogP=%.2f QED=%.3f SA=%.2f Ro5=%s",
            name,
            mw,
            logp,
            qed_score,
            sa,
            lipinski_pass,
        )
        return ADMETResult(
            name=name,
            mw=mw,
            logp=logp,
            hbd=hbd,
            hba=hba,
            tpsa=tpsa,
            rotbonds=rotbonds,
            qed=qed_score,
            sa_score=sa,
            lipinski_pass=lipinski_pass,
            veber_pass=veber_pass,
            pains_alert=pains_alert,
            pains_patterns=pains_names,
        )

    def _get_pains_catalog(self) -> object:
        if self._pains_catalog is None:
            self._pains_catalog = _build_pains_catalog()
        return self._pains_catalog


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_rdkit() -> object:
    """Import and return ``rdkit.Chem``, raising a clear ImportError if absent."""
    try:
        from rdkit import Chem  # noqa: PLC0415
        return Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for ADMETCalculator. "
            "Install it with: pip install mdatools[admet]"
        ) from exc


def _build_pains_catalog() -> object:
    """Build and return a combined PAINS A/B/C FilterCatalog."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def _check_pains(mol: object, catalog: object) -> list[str]:
    """Return list of PAINS alert description strings for *mol*."""
    matches = catalog.GetMatches(mol)
    return [m.GetDescription() for m in matches]


def _calc_sa_score(mol: object) -> float:
    """Return synthetic accessibility score (1–10).

    Attempts to use the RDKit Contrib SA_Score module. Falls back to a
    ring-complexity approximation when the Contrib module is unavailable.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import]
        return float(sascorer.calculateScore(mol))
    except ImportError:
        pass

    # Lightweight approximation: penalise ring complexity and stereocentres
    from rdkit.Chem import rdMolDescriptors

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = len(rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True))
    n_heavy = mol.GetNumHeavyAtoms()
    # Score in [1, 10]: simple heuristic, not a replacement for sascorer
    score = 1.0 + 0.3 * n_rings + 0.5 * n_stereo + 0.02 * max(0, n_heavy - 20)
    return float(min(score, 10.0))
