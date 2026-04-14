"""Protein-ligand contact fingerprint analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import distances as mda_dist

from ..config import AnalysisConfig
from ..universe import load_and_align

logger = logging.getLogger(__name__)

_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET", "TYR"}
_POLAR = {"SER", "THR", "ASN", "GLN", "CYS"}
_CHARGED = {"ARG", "LYS", "HIS", "ASP", "GLU"}


@dataclass
class IFPResult:
    """Container for interaction fingerprint results."""

    sample_name: str
    df: pd.DataFrame  # columns: residue, resname, resid, occupancy_%, contact_type, is_ecd

    @property
    def summary(self) -> pd.DataFrame:
        """Return DataFrame with columns 'interaction' and 'occupancy_pct'."""
        return (
            self.df[["residue", "occupancy_%"]]
            .rename(columns={"residue": "interaction", "occupancy_%": "occupancy_pct"})
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.df)


def _classify(resname: str) -> str:
    if resname in _HYDROPHOBIC:
        return "hydrophobic"
    if resname in _POLAR:
        return "polar"
    if resname in _CHARGED:
        return "charged"
    return "other"


class ContactFingerprint:
    """Compute per-residue contact occupancy with the ligand."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        u: mda.Universe,
        sample_name: str = "",
        ecd_resids: list[int] | None = None,
    ) -> IFPResult:
        """Compute contact fingerprint.

        Parameters
        ----------
        u:
            Aligned Universe.
        sample_name:
            Label for this replica (stored in :class:`IFPResult`).
        ecd_resids:
            Optional list of residue IDs to flag as ECD (or binding-pocket)
            residues.

        Returns
        -------
        :class:`IFPResult` wrapping a DataFrame with columns: residue,
        resname, resid, occupancy_%, contact_type, is_ecd.
        """
        cutoff = self.cfg.contact_cutoff
        ligand = u.select_atoms(
            f"resname {self.cfg.ligand_resname} and not name H*"
        )
        if len(ligand) == 0:
            raise ValueError(
                f"Ligand '{self.cfg.ligand_resname}' not found in universe."
            )
        n_frames = len(u.trajectory)
        contact_count: dict[str, int] = {}
        for _ts in u.trajectory:
            protein = u.select_atoms(
                f"protein and not name H* and (around {cutoff} resname {self.cfg.ligand_resname})"
            )
            for res in protein.residues:
                key = f"{res.resname}{res.resid}"
                contact_count[key] = contact_count.get(key, 0) + 1

        ecd_set = set(ecd_resids or [])
        rows = []
        for key, cnt in contact_count.items():
            resname = key[:3]
            resid = int(key[3:])
            rows.append(
                {
                    "residue": key,
                    "resname": resname,
                    "resid": resid,
                    "occupancy_%": cnt / n_frames * 100,
                    "contact_type": _classify(resname),
                    "is_ecd": resid in ecd_set,
                }
            )
        df = pd.DataFrame(rows).sort_values("occupancy_%", ascending=False).reset_index(drop=True)
        return IFPResult(sample_name=sample_name, df=df)

    def run_per_resid(
        self,
        u: mda.Universe,
        target_resids: list[int],
    ) -> pd.DataFrame:
        """Compute per-frame minimum distance for *target_resids*.

        Returns DataFrame with columns: frame, resid, resname, min_dist.
        """
        ligand = u.select_atoms(
            f"resname {self.cfg.ligand_resname} and not name H*"
        )
        records = []
        for ts in u.trajectory:
            for resid in target_resids:
                res_atoms = u.select_atoms(
                    f"protein and resid {resid} and not name H*"
                )
                if len(res_atoms) == 0:
                    continue
                d = mda_dist.distance_array(res_atoms.positions, ligand.positions)
                records.append(
                    {
                        "frame": ts.frame,
                        "resid": resid,
                        "resname": res_atoms.residues[0].resname,
                        "min_dist": float(d.min()),
                    }
                )
        return pd.DataFrame(records)
