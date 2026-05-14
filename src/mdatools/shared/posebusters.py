"""Unified PoseBusters pose quality gate.

Replaces duplicate implementations in:
  - ``mdatools.posebusters.batch_validator.PoseBustersValidator``
  - ``docking_analysis.analysis.posebusters.validate_poses_posebusters``

This module supports two input modes:

**Mode A — trajectory frames (md-analysis-tools workflow)**::

    result = bust_trajectory_poses(
        pdb_dir=Path("frames/"),
        receptor_pdb=Path("receptor.pdb"),
        ligand_smiles="Cc1cc2c...",   # or ligand_sdf=Path("lig.sdf")
    )
    print(result.pass_rate)

**Mode B — RDKit Mol list (docking-analysis-tools workflow)**::

    valid_df, valid_mols = bust_rdkit_poses(
        mols=docked_mols,
        receptor_pdb=Path("receptor.pdb"),
    )

PoseBusters checks
------------------
The ``posebusters`` library runs up to ~22 checks (in ``"dock"`` mode).
The following are **always excluded** as known false-positives in
typical MD/docking contexts:

- ``non_aromatic_ring_non_flatness`` — macrocycle false-positives

All other checks are included by default; use the ``exclude_checks``
parameter to add further exclusions.

Required optional dependency: ``posebusters``
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Checks excluded from the "all-pass" gate because they fire too often
# on legitimate MD frames / docking poses.
_DEFAULT_EXCLUDED = frozenset({"non_aromatic_ring_non_flatness"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PoseBustersResult:
    """Results of a PoseBusters batch validation run.

    Attributes
    ----------
    results:
        DataFrame with one row per evaluated pose.  Boolean columns
        correspond to individual PoseBusters checks.  The
        ``all_pass`` column is ``True`` when all non-excluded checks pass.
    n_evaluated:
        Total number of poses evaluated (including failures).
    excluded_checks:
        Set of check names excluded from ``all_pass``.

    Notes
    -----
    The ``energy_ratio`` column contains the ratio of strain energy to
    reference when supplied by the ``posebusters`` library (may be absent
    when running in ``"mol"`` or older library versions).
    """

    results: pd.DataFrame
    n_evaluated: int
    excluded_checks: frozenset[str] = _DEFAULT_EXCLUDED

    @property
    def pass_rate(self) -> float:
        """Fraction of poses where ``all_pass == True``."""
        if self.results.empty or "all_pass" not in self.results.columns:
            return 0.0
        return float(self.results["all_pass"].mean())

    def top_frames(self, n: int = 10) -> pd.DataFrame:
        """Return the *n* highest-quality frames, sorted by ``all_pass`` first
        then by ``energy_ratio`` (ascending) when available.
        """
        sort_cols = ["all_pass"]
        if "energy_ratio" in self.results.columns:
            sort_cols.append("energy_ratio")
        return self.results.sort_values(
            sort_cols, ascending=[False, True] if len(sort_cols) == 2 else [False]
        ).head(n)

    def failed_checks_summary(self) -> pd.Series:
        """Per-check failure rate as a Series, sorted descending."""
        bool_cols = [
            c
            for c in self.results.columns
            if c not in ("all_pass", "pose_idx", "frame_index", "energy_ratio")
            and self.results[c].dtype == bool
        ]
        if not bool_cols:
            return pd.Series(dtype=float)
        failure_rates = 1.0 - self.results[bool_cols].mean()
        return failure_rates.sort_values(ascending=False)

    def passing_poses(self) -> pd.DataFrame:
        """Return only the rows where ``all_pass == True``."""
        return self.results[self.results["all_pass"]].copy()


# ---------------------------------------------------------------------------
# Mode A: trajectory frame PDB files
# ---------------------------------------------------------------------------


def bust_trajectory_poses(
    pdb_dir: Path,
    receptor_pdb: Path,
    *,
    ligand_smiles: str | None = None,
    ligand_sdf: Path | None = None,
    mode: str = "dock",
    pdb_pattern: str = "*.pdb",
    exclude_checks: frozenset[str] | None = None,
) -> PoseBustersResult:
    """Validate trajectory-derived PDB frames with PoseBusters.

    Reads all PDB files matching *pdb_pattern* from *pdb_dir*, assigns
    bond orders from *ligand_smiles* / *ligand_sdf* (required because
    PDB format lacks bond-order information), and runs the PoseBusters
    check battery.

    Parameters
    ----------
    pdb_dir:
        Directory containing per-frame PDB files.
    receptor_pdb:
        Receptor-only PDB file for clash and contact checks.
    ligand_smiles:
        SMILES string used to build the bond-order template.
    ligand_sdf:
        Path to an SDF file used as bond-order template (used when
        *ligand_smiles* is ``None``).
    mode:
        PoseBusters mode — ``"dock"`` or ``"redock"``.
    pdb_pattern:
        Glob pattern for frame PDB files inside *pdb_dir*.
    exclude_checks:
        Additional check names to exclude from ``all_pass`` on top of
        :data:`_DEFAULT_EXCLUDED`.

    Returns
    -------
    PoseBustersResult

    Raises
    ------
    ValueError
        When neither *ligand_smiles* nor *ligand_sdf* is supplied.
    ImportError
        When the ``posebusters`` library is not installed.
    """
    _require_posebusters()
    if ligand_smiles is None and ligand_sdf is None:
        raise ValueError(
            "bust_trajectory_poses requires either ligand_smiles or ligand_sdf "
            "to assign bond orders — PDB format has no bond-order information."
        )

    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    exclusions = _DEFAULT_EXCLUDED | (exclude_checks or frozenset())

    # Build bond-order template mol
    if ligand_smiles:
        template = Chem.MolFromSmiles(ligand_smiles)
        if template is None:
            raise ValueError(f"RDKit could not parse SMILES: {ligand_smiles!r}")
        AllChem.EmbedMolecule(template, AllChem.ETKDGv3())
    else:
        template = next(Chem.SDMolSupplier(str(ligand_sdf), removeHs=True))
        if template is None:
            raise ValueError(f"Could not read ligand from SDF: {ligand_sdf}")

    pdb_files = sorted(Path(pdb_dir).glob(pdb_pattern))
    if not pdb_files:
        logger.warning("No PDB files matching %r in %s", pdb_pattern, pdb_dir)

    rows: list[dict[str, Any]] = []
    for pdb_path in pdb_files:
        frame_idx = _parse_frame_index(pdb_path)
        try:
            pose_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
            if pose_mol is None:
                raise ValueError("MolFromPDBFile returned None")
            pose_mol = AllChem.AssignBondOrdersFromTemplate(template, pose_mol)
            Chem.SanitizeMol(pose_mol)
            row = _run_pb(pose_mol, receptor_pdb, mode)
        except Exception as exc:  # noqa: BLE001
            logger.debug("PoseBusters failed for %s: %s", pdb_path.name, exc)
            row = {}
        row["frame_index"] = frame_idx
        row["pdb_path"] = str(pdb_path)
        rows.append(row)

    df = _build_result_df(rows, exclusions)
    return PoseBustersResult(
        results=df, n_evaluated=len(pdb_files), excluded_checks=exclusions
    )


# ---------------------------------------------------------------------------
# Mode B: RDKit Mol list
# ---------------------------------------------------------------------------


def bust_rdkit_poses(
    mols: list[Any],
    receptor_pdb: Path,
    *,
    mode: str = "dock",
    exclude_checks: frozenset[str] | None = None,
) -> PoseBustersResult:
    """Validate a list of RDKit Mol objects with PoseBusters.

    Parameters
    ----------
    mols:
        Docked poses (RDKit Mols with 3D conformers).
    receptor_pdb:
        Receptor PDB for clash / contact checks.
    mode:
        PoseBusters mode — ``"dock"`` or ``"redock"``.
    exclude_checks:
        Additional check names to exclude from ``all_pass``.

    Returns
    -------
    PoseBustersResult
    """
    _require_posebusters()
    exclusions = _DEFAULT_EXCLUDED | (exclude_checks or frozenset())
    rows: list[dict[str, Any]] = []
    for idx, mol in enumerate(mols):
        try:
            row = _run_pb(mol, receptor_pdb, mode)
        except Exception as exc:  # noqa: BLE001
            logger.debug("PoseBusters failed for pose %d: %s", idx, exc)
            row = {}
        row["pose_idx"] = idx
        rows.append(row)

    df = _build_result_df(rows, exclusions)
    return PoseBustersResult(
        results=df, n_evaluated=len(mols), excluded_checks=exclusions
    )


def filter_rdkit_poses(
    df: pd.DataFrame,
    mols: list[Any],
    receptor_pdb: Path,
    *,
    mode: str = "dock",
    exclude_checks: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, list[Any]]:
    """Filter a DataFrame + Mol list, keeping only PoseBusters-passing poses.

    Parameters
    ----------
    df:
        Poses DataFrame (rows parallel to *mols*).
    mols:
        RDKit Mol list parallel to *df* rows.
    receptor_pdb:
        Receptor PDB file.
    mode:
        PoseBusters mode.
    exclude_checks:
        Additional checks to exclude from ``all_pass``.

    Returns
    -------
    tuple[pd.DataFrame, list[Any]]
        Filtered (DataFrame, molecules) pair.
    """
    result = bust_rdkit_poses(
        mols, receptor_pdb, mode=mode, exclude_checks=exclude_checks
    )
    if "all_pass" not in result.results.columns:
        return df, mols
    mask = result.results["all_pass"].values
    filtered_df = df[mask].copy()
    filtered_mols = [m for m, v in zip(mols, mask) if v]
    return filtered_df, filtered_mols


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_posebusters() -> None:
    try:
        import posebusters  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "posebusters is required for mdatools.shared.posebusters. "
            "Install it with:  pip install posebusters"
        ) from exc


def _run_pb(mol: Any, receptor_pdb: Path, mode: str) -> dict[str, Any]:
    """Run PoseBusters on a single RDKit Mol; return a dict of check results."""
    import posebusters  # noqa: PLC0415

    pb = posebusters.PoseBusters(config=mode)
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tf:
        sdf_path = Path(tf.name)

    try:
        from rdkit.Chem import SDWriter  # noqa: PLC0415

        with SDWriter(str(sdf_path)) as w:
            w.write(mol)
        df = pb.bust(
            mol_pred=str(sdf_path), mol_cond=str(receptor_pdb), full_report=True
        )
        return df.iloc[0].to_dict() if len(df) > 0 else {}
    finally:
        sdf_path.unlink(missing_ok=True)


def _build_result_df(
    rows: list[dict[str, Any]], exclusions: frozenset[str]
) -> pd.DataFrame:
    """Build result DataFrame and compute the ``all_pass`` gate column."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    bool_cols = [
        c
        for c in df.columns
        if c not in exclusions
        and c not in ("pose_idx", "frame_index", "pdb_path", "energy_ratio")
        and df[c].dtype == bool
    ]
    if bool_cols:
        df["all_pass"] = df[bool_cols].all(axis=1)
    else:
        df["all_pass"] = True
    return df


def _parse_frame_index(path: Path) -> int:
    """Extract the numeric frame index from a PDB filename (e.g. ``frame_042.pdb`` → 42)."""
    import re  # noqa: PLC0415

    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0
