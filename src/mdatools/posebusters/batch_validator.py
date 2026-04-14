"""PoseBusters batch validation for MD trajectory frames.

Prerequisites (see docs/POSEBUSTERS_PREREQUISITES.md):
- ``pip install 'mdatools[posebusters]'``  (installs posebusters + rdkit)
- Ligand SMILES string **or** SDF/MOL2 file with correct bond orders
  (PDB format does not encode bond orders; without a template the chemical
  validity checks will fail for most drug-like molecules)

Typical workflow
----------------
1. Export frames:  ``FrameExporter.export_from_universe()``
2. Validate:       ``PoseBustersValidator.run()``
3. Rank / filter:  ``PoseBustersResult.top_frames()``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)

# Checks that are known false-positives when using PDB input
# (PDB format forces all ring atoms to single bonds → always fails)
_FALSE_POSITIVE_CHECKS: frozenset[str] = frozenset(
    {"non_aromatic_ring_non_flatness"}
)


@dataclass
class PoseBustersResult:
    """Container for PoseBusters batch results."""

    sample_name: str
    results: pd.DataFrame       # one row per frame; bool columns + energy_ratio
    ligand_smiles: str | None   # SMILES used for bond-order template
    ligand_sdf: Path | None     # SDF path used for bond-order template

    @property
    def pass_rate(self) -> float:
        """Fraction of frames passing all (non-false-positive) checks."""
        if self.results.empty:
            return 0.0
        return float(self.results["all_pass"].mean())

    def top_frames(self, n: int = 10) -> pd.DataFrame:
        """Return the top-*n* frames sorted by energy_ratio (best geometry)."""
        col = "energy_ratio" if "energy_ratio" in self.results.columns else "all_pass"
        passing = self.results[self.results["all_pass"]]
        if passing.empty:
            logger.warning("No frames passed all checks; returning best frames overall.")
            passing = self.results
        return (
            passing.sort_values(col, ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def failed_checks_summary(self) -> pd.DataFrame:
        """Return per-check failure rates across all frames."""
        bool_cols = [
            c for c in self.results.columns
            if self.results[c].dtype == bool and c != "all_pass"
        ]
        rates = {}
        for col in bool_cols:
            rates[col] = 1.0 - self.results[col].mean()
        return (
            pd.DataFrame.from_dict(rates, orient="index", columns=["failure_rate"])
            .sort_values("failure_rate", ascending=False)
        )


class PoseBustersValidator:
    """Run PoseBusters on a directory of PDB frames.

    Parameters
    ----------
    cfg:
        Analysis configuration.
    ligand_smiles:
        SMILES string of the ligand **with correct bond orders**.
        Takes priority over *ligand_sdf*.
    ligand_sdf:
        Path to an SDF (or MOL2) file of the ligand with correct bond orders.
        Used when *ligand_smiles* is not provided.
    full_report:
        If True, run all 35 PoseBusters checks (slow).
        If False (default), run the 22 docking-mode checks (fast).
    exclude_checks:
        Additional check names to exclude from the ``all_pass`` column.
        The ``non_aromatic_ring_non_flatness`` false-positive is always excluded.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        *,
        ligand_smiles: str | None = None,
        ligand_sdf: Path | None = None,
        full_report: bool = False,
        exclude_checks: list[str] | None = None,
    ) -> None:
        if ligand_smiles is None and ligand_sdf is None:
            raise ValueError(
                "Either ligand_smiles or ligand_sdf must be provided. "
                "PDB format lacks bond-order information; without a SMILES or SDF "
                "template, PoseBusters chemical validity checks will fail. "
                "See docs/POSEBUSTERS_PREREQUISITES.md for details."
            )
        self.cfg = cfg
        self.ligand_smiles = ligand_smiles
        self.ligand_sdf = Path(ligand_sdf) if ligand_sdf else None
        self.full_report = full_report
        self._exclude = _FALSE_POSITIVE_CHECKS | set(exclude_checks or [])
        self._template_mol = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_template_mol(self):
        """Return an RDKit Mol with correct bond orders (lazy cached)."""
        if self._template_mol is not None:
            return self._template_mol

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise ImportError(
                "rdkit is required for PoseBusters validation. "
                "Install with: pip install 'mdatools[posebusters]'"
            ) from e

        if self.ligand_smiles:
            mol = Chem.MolFromSmiles(self.ligand_smiles)
            if mol is None:
                raise ValueError(
                    f"Could not parse SMILES: {self.ligand_smiles!r}\n"
                    "Please verify the SMILES string is correct."
                )
            mol = Chem.AddHs(mol)
            logger.info("Ligand template loaded from SMILES.")
        else:
            mol = Chem.MolFromMolFile(str(self.ligand_sdf), removeHs=False)
            if mol is None:
                raise ValueError(
                    f"Could not parse SDF/MOL file: {self.ligand_sdf}\n"
                    "Ensure the file contains a valid V2000/V3000 mol block."
                )
            logger.info("Ligand template loaded from SDF: %s", self.ligand_sdf)

        self._template_mol = mol
        return mol

    def _prepare_ligand_mol(self, pdb_path: Path):
        """Extract ligand from frame PDB and assign bond orders from template."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        template = self._get_template_mol()

        # Extract only the ligand HETATM records to a temporary PDB string
        hetatm_lines = [
            line for line in pdb_path.read_text().splitlines()
            if line.startswith(("HETATM", "CONECT"))
            and self.cfg.ligand_resname in line[:30]
        ]
        if not hetatm_lines:
            raise ValueError(
                f"No HETATM records for resname '{self.cfg.ligand_resname}' "
                f"found in {pdb_path.name}"
            )

        lig_pdb_block = "\n".join(hetatm_lines) + "\nEND\n"
        lig_mol = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=False, sanitize=False)
        if lig_mol is None:
            raise ValueError(f"RDKit could not parse ligand block in {pdb_path.name}")

        # Assign bond orders from the template mol (SMILES/SDF)
        try:
            mol_assigned = AllChem.AssignBondOrdersFromTemplate(
                Chem.RemoveHs(template), Chem.RemoveHs(lig_mol)
            )
        except Exception as exc:
            raise ValueError(
                f"Bond order assignment failed for {pdb_path.name}. "
                "Check that the SMILES/SDF matches the ligand in the trajectory "
                f"(same atom connectivity). Original error: {exc}"
            ) from exc

        try:
            Chem.SanitizeMol(mol_assigned)
        except Exception as exc:
            raise ValueError(
                f"RDKit sanitization failed for {pdb_path.name}: {exc}"
            ) from exc

        return mol_assigned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pdb_dir: Path,
        protein_path: Path,
        *,
        prefix: str = "",
        pdb_pattern: str = "*.pdb",
    ) -> PoseBustersResult:
        """Run PoseBusters on all PDB files in *pdb_dir*.

        Parameters
        ----------
        pdb_dir:
            Directory containing per-frame PDB files from :class:`FrameExporter`.
        protein_path:
            PDB of the receptor (protein only, no solvent/ligand) used by
            PoseBusters for inter-molecular clash and binding-pose checks.
        prefix:
            Only process PDB files whose name starts with *prefix*.
        pdb_pattern:
            Glob pattern for frame PDBs (default ``"*.pdb"``).

        Returns
        -------
        :class:`PoseBustersResult`
        """
        try:
            from posebusters import PoseBusters
        except ImportError as e:
            raise ImportError(
                "posebusters is required. "
                "Install with: pip install 'mdatools[posebusters]'"
            ) from e

        pdb_dir = Path(pdb_dir)
        protein_path = Path(protein_path)

        pdb_files = sorted(pdb_dir.glob(pdb_pattern))
        if prefix:
            pdb_files = [p for p in pdb_files if p.name.startswith(prefix)]
        if not pdb_files:
            raise FileNotFoundError(
                f"No PDB files matching '{pdb_pattern}' in {pdb_dir}"
            )

        logger.info(
            "Running PoseBusters on %d frames (full_report=%s)",
            len(pdb_files),
            self.full_report,
        )

        pb = PoseBusters(config="dock", full_report=self.full_report)
        rows: list[dict] = []

        for pdb_path in pdb_files:
            frame_idx = self._parse_frame_index(pdb_path)
            row: dict = {"frame": frame_idx, "pdb": pdb_path.name}
            try:
                lig_mol = self._prepare_ligand_mol(pdb_path)
                result_df = pb.bust(
                    mol_pred=lig_mol,
                    mol_cond=str(protein_path),
                    full_report=self.full_report,
                )
                # Flatten single-row DataFrame into dict
                flat = result_df.iloc[0].to_dict()
                row.update(flat)
            except Exception as exc:
                logger.warning("Frame %d skipped: %s", frame_idx, exc)
                row["error"] = str(exc)
            rows.append(row)

        results = pd.DataFrame(rows)

        # Compute all_pass excluding known false positives
        bool_check_cols = [
            c for c in results.columns
            if results[c].dtype == bool
            and c not in self._exclude
            and c != "all_pass"
        ]
        if bool_check_cols:
            results["all_pass"] = results[bool_check_cols].all(axis=1)
        else:
            results["all_pass"] = False

        logger.info(
            "PoseBusters done: %d/%d frames passed (%.1f%%)",
            results["all_pass"].sum(),
            len(results),
            results["all_pass"].mean() * 100,
        )

        sample_name = pdb_dir.name
        return PoseBustersResult(
            sample_name=sample_name,
            results=results,
            ligand_smiles=self.ligand_smiles,
            ligand_sdf=self.ligand_sdf,
        )

    @staticmethod
    def _parse_frame_index(pdb_path: Path) -> int:
        """Extract frame index from filename like ``prefix_frame0042.pdb``."""
        stem = pdb_path.stem
        if "frame" in stem:
            try:
                return int(stem.split("frame")[-1])
            except ValueError:
                pass
        return 0
