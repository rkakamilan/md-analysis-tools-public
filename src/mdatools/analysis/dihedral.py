"""Ligand dihedral angle analysis.

Preferred backend: MDAnalysis (no external process required).
PyMOL fallback is provided for cases where a PSE file is needed.
"""

from __future__ import annotations

import logging
import math
import subprocess
import textwrap
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)

# A dihedral definition: (label, atom1, atom2, atom3, atom4)
DihedralDef = tuple[str, str, str, str, str]


def _calc_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Calculate dihedral angle in degrees (−180 to +180)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    ln1, ln2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if ln1 < 1e-6 or ln2 < 1e-6:
        return float("nan")
    n1 /= ln1
    n2 /= ln2
    cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_a))
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return angle


class DihedralAnalyzer:
    """Compute ligand rotatable bond dihedrals using MDAnalysis.

    Parameters
    ----------
    cfg:
        Analysis configuration.
    dihedral_defs:
        List of ``(label, a1, a2, a3, a4)`` tuples where each *a_n* is an
        atom name in the ligand residue.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        dihedral_defs: list[DihedralDef],
    ) -> None:
        self.cfg = cfg
        self.dihedral_defs = dihedral_defs

    def run(self, u: mda.Universe) -> pd.DataFrame:
        """Compute dihedrals for every frame.

        Returns DataFrame with columns: frame, time_ns, <label1>, <label2>, ...
        """
        lig = u.select_atoms(f"resname {self.cfg.ligand_resname}")
        if len(lig) == 0:
            raise ValueError(f"Ligand '{self.cfg.ligand_resname}' not found.")

        records = []
        for ts in u.trajectory:
            row: dict = {
                "frame": ts.frame,
                "time_ns": ts.frame * self.cfg.dt_ns,
            }
            for label, a1, a2, a3, a4 in self.dihedral_defs:
                atoms = []
                for aname in (a1, a2, a3, a4):
                    sel = lig.select_atoms(f"name {aname}")
                    if len(sel) == 0:
                        atoms = None
                        break
                    atoms.append(sel.positions[0])
                if atoms is None:
                    row[label] = float("nan")
                else:
                    row[label] = round(_calc_dihedral(*atoms), 3)
            records.append(row)
        return pd.DataFrame(records)

    def run_from_pse(
        self,
        pse_path: Path | str,
        output_csv: Path | str,
        object_name: str = "traj",
    ) -> pd.DataFrame:
        """Extract dihedrals via PyMOL subprocess (fallback).

        Writes a temporary PML, runs PyMOL, then reads the output CSV.
        """
        from ..pymol_bridge.pml_builder import build_dihedral_pml

        pse_path = Path(pse_path)
        output_csv = Path(output_csv)
        pml_path = output_csv.with_suffix(".pml")

        pml_content = build_dihedral_pml(
            pse_path=pse_path,
            output_csv=output_csv,
            ligand_resname=self.cfg.ligand_resname,
            dihedral_defs=self.dihedral_defs,
            object_name=object_name,
        )
        pml_path.write_text(pml_content, encoding="utf-8")

        result = subprocess.run(
            [self.cfg.pymol_executable, "-c", str(pml_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("PyMOL stderr: %s", result.stderr[-2000:])
            raise RuntimeError("PyMOL dihedral extraction failed.")

        df = pd.read_csv(output_csv)
        df["time_ns"] = df["frame"] * self.cfg.dt_ns
        return df
