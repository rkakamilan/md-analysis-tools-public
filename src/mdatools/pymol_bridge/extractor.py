"""PyMOL-based metric extraction via subprocess."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


class PyMOLExtractor:
    """Extract per-frame metrics from a PyMOL PSE file."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def _run_pml(self, pml_path: Path) -> None:
        result = subprocess.run(
            [self.cfg.pymol_executable, "-c", str(pml_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("PyMOL stderr:\n%s", result.stderr[-3000:])
            raise RuntimeError(f"PyMOL failed (exit {result.returncode})")

    def extract_validation_metrics(
        self,
        pse_path: Path | str,
        pocket_resids: list[int],
        hbond_pairs: list[tuple[str, str]],
        output_csv: Path | str,
        extra_pairs: list[tuple[str, str]] | None = None,
        object_name: str = "traj",
        pair_labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract RMSD and distance metrics to CSV using PyMOL.

        Parameters
        ----------
        pse_path:
            Path to the trajectory PSE file.
        pocket_resids:
            Residue IDs for pocket RMSD calculation.
        hbond_pairs:
            List of ``(sel1, sel2)`` strings for distance measurement.
        output_csv:
            Where to write the result CSV.
        extra_pairs:
            Additional distance pairs.
        object_name:
            PyMOL trajectory object name.
        pair_labels:
            Column names for the distance columns.  Defaults to dist_0, dist_1...
        """
        from .pml_builder import build_validation_pml

        pse_path = Path(pse_path)
        output_csv = Path(output_csv)
        pml_path = output_csv.with_suffix(".pml")

        pml = build_validation_pml(
            pse_path=pse_path,
            output_csv=output_csv,
            pocket_resids=pocket_resids,
            hbond_pairs=hbond_pairs,
            extra_pairs=extra_pairs,
            object_name=object_name,
            ligand_resname=self.cfg.ligand_resname,
        )
        pml_path.write_text(pml, encoding="utf-8")
        self._run_pml(pml_path)

        df = pd.read_csv(output_csv)
        df["time_ns"] = df["frame"] * self.cfg.dt_ns

        # Rename distance columns if labels provided
        if pair_labels:
            all_pairs = hbond_pairs + (extra_pairs or [])
            for i, label in enumerate(pair_labels[: len(all_pairs)]):
                df = df.rename(columns={f"dist_{i}": label})

        return df

    def extract_dihedrals(
        self,
        pse_path: Path | str,
        dihedral_defs: list[tuple[str, str, str, str, str]],
        output_csv: Path | str,
        object_name: str = "traj",
    ) -> pd.DataFrame:
        """Extract dihedral angles to CSV using PyMOL.

        Parameters
        ----------
        pse_path:
            Path to the trajectory PSE file.
        dihedral_defs:
            List of ``(label, a1, a2, a3, a4)`` tuples.
        output_csv:
            Destination CSV.
        object_name:
            PyMOL trajectory object name.
        """
        from .pml_builder import build_dihedral_pml

        pse_path = Path(pse_path)
        output_csv = Path(output_csv)
        pml_path = output_csv.with_suffix(".pml")

        pml = build_dihedral_pml(
            pse_path=pse_path,
            output_csv=output_csv,
            ligand_resname=self.cfg.ligand_resname,
            dihedral_defs=dihedral_defs,
            object_name=object_name,
        )
        pml_path.write_text(pml, encoding="utf-8")
        self._run_pml(pml_path)

        df = pd.read_csv(output_csv)
        df["time_ns"] = df["frame"] * self.cfg.dt_ns
        return df
