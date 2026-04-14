"""Best-frame snapshot selection and PDB extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

from ..analysis.hbonds import HBondResult
from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class SnapshotResult:
    sample_name: str
    hbond_id: str
    frame: int
    distance: float
    angle: float
    mean_dist: float
    mean_angle: float
    snap_score: float


class SnapshotSelector:
    """Select the best representative frame from a HBond trajectory.

    Selection strategy:
    - Filter to frames in the latter ``(1 - min_frame_percentile)`` portion
      (to prefer equilibrated conformations).
    - Among those, pick the frame whose distance is closest to the mean
      and whose angle is highest.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def select(
        self,
        result: HBondResult,
        primary_hbond_id: str | None = None,
    ) -> SnapshotResult:
        """Select the best snapshot frame for *result*.

        Parameters
        ----------
        result:
            HBondResult from one replica.
        primary_hbond_id:
            Specific H-bond to optimise for.  Defaults to the highest-occupancy
            H-bond in ``result.summary``.
        """
        ev = result.events
        sm = result.summary
        if ev.empty or sm.empty:
            raise ValueError(f"No H-bond data for {result.sample_name}")

        if primary_hbond_id is None:
            primary_hbond_id = sm.iloc[0]["hbond_id"]

        target = ev[ev["hbond_id"] == primary_hbond_id].copy()
        if target.empty:
            raise ValueError(
                f"H-bond '{primary_hbond_id}' not found in events for {result.sample_name}"
            )

        max_frame = target["frame"].max()
        cutoff = int(max_frame * self.cfg.scoring.min_frame_percentile)
        stable = target[target["frame"] >= cutoff]
        if stable.empty:
            stable = target

        sm_row = sm[sm["hbond_id"] == primary_hbond_id].iloc[0]
        mean_dist = float(sm_row["mean_dist"])
        mean_angle = float(sm_row["mean_angle"])

        stable = stable.copy()
        stable["snap_score"] = (
            -abs(stable["distance"] - mean_dist) * 10
            + (stable["angle"] - 150) / 30
        )
        best = stable.loc[stable["snap_score"].idxmax()]

        return SnapshotResult(
            sample_name=result.sample_name,
            hbond_id=primary_hbond_id,
            frame=int(best["frame"]),
            distance=round(float(best["distance"]), 3),
            angle=round(float(best["angle"]), 2),
            mean_dist=round(mean_dist, 3),
            mean_angle=round(mean_angle, 2),
            snap_score=round(float(best["snap_score"]), 4),
        )

    def extract_pdb(
        self,
        u: mda.Universe,
        snap: SnapshotResult,
        output_path: Path | str,
    ) -> Path:
        """Write the snapshot frame to a PDB file.

        Parameters
        ----------
        u:
            Universe (already aligned).
        snap:
            Result from :meth:`select`.
        output_path:
            Destination PDB path.
        """
        from ..io.writers import write_snapshot_pdb

        return write_snapshot_pdb(u, snap.frame, output_path)
