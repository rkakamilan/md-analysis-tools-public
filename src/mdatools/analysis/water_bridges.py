"""Water bridge analysis — protein–water–ligand indirect H-bonds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import distances as mda_dist

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)

_DEFAULT_WATER_RESNAMES = ["HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP"]


@dataclass
class WaterBridgeResult:
    """Results from water bridge analysis.

    Attributes
    ----------
    sample_name:
        Identifier for the replica/sample.
    events:
        Per-frame bridge records. Columns: ``frame``, ``protein_atom``,
        ``water_resid``, ``ligand_atom``, ``dist_pw``, ``dist_wl``.
    summary:
        Per-(protein_atom, ligand_atom) pair statistics. Columns:
        ``protein_atom``, ``ligand_atom``, ``occupancy_pct``,
        ``mean_dist_pw``, ``mean_dist_wl``.
    n_frames:
        Total number of trajectory frames analysed.
    """

    sample_name: str
    events: pd.DataFrame
    summary: pd.DataFrame
    n_frames: int


class WaterBridgeAnalyzer:
    """Detect protein–water–ligand hydrogen bond bridges.

    Algorithm
    ---------
    For each trajectory frame:

    1. Select water oxygen atoms within ``max_bridge_dist`` Å of **both** a
       protein H-bond eligible heavy atom (N*/O*/S*) and a ligand H-bond
       eligible heavy atom.
    2. Record every (protein_atom, water_O, ligand_atom) triple that
       satisfies the distance criterion for both legs.

    The analysis is intentionally distance-only (no angle filter) so that it
    works regardless of whether explicit water hydrogen coordinates are
    present in the topology.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        water_resnames: list[str] | None = None,
        max_bridge_dist: float = 3.5,
        min_angle: float = 120.0,
    ) -> None:
        self.cfg = cfg
        self.water_resnames = water_resnames or _DEFAULT_WATER_RESNAMES
        self.max_bridge_dist = max_bridge_dist
        self.min_angle = min_angle  # reserved for future angle filtering

    def run(self, u: mda.Universe, sample_name: str = "") -> WaterBridgeResult:
        """Detect water bridges across all trajectory frames.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`WaterBridgeResult`.

        Returns
        -------
        WaterBridgeResult
        """
        water_res_str = " ".join(self.water_resnames)
        protein_ag = u.select_atoms("protein and (name N* or name O* or name S*)")
        ligand_ag = u.select_atoms(
            f"resname {self.cfg.ligand_resname} and (name N* or name O* or name S*)"
        )
        water_ag = u.select_atoms(f"resname {water_res_str} and name O*")

        n_frames = len(u.trajectory)

        if len(protein_ag) == 0 or len(ligand_ag) == 0 or len(water_ag) == 0:
            logger.info(
                "Water bridge analysis skipped for %s: "
                "protein=%d lig=%d water=%d atoms",
                sample_name, len(protein_ag), len(ligand_ag), len(water_ag),
            )
            return WaterBridgeResult(
                sample_name=sample_name,
                events=_empty_events(),
                summary=_empty_summary(),
                n_frames=n_frames,
            )

        event_rows: list[dict] = []
        cutoff = self.max_bridge_dist

        for ts in u.trajectory:
            # Pairwise distances: shape (n_water, n_protein) and (n_water, n_ligand)
            d_wp = mda_dist.distance_array(
                water_ag.positions, protein_ag.positions
            )
            d_wl = mda_dist.distance_array(
                water_ag.positions, ligand_ag.positions
            )

            # For each water O, check if any protein atom AND any ligand atom
            # are within cutoff
            near_prot = d_wp <= cutoff   # (n_water, n_prot) bool
            near_lig = d_wl <= cutoff    # (n_water, n_lig)  bool

            # Waters that bridge: have ≥1 protein contact AND ≥1 ligand contact
            bridging_water_indices = np.where(
                near_prot.any(axis=1) & near_lig.any(axis=1)
            )[0]

            for wi in bridging_water_indices:
                water_atom = water_ag[wi]
                prot_contacts = np.where(near_prot[wi])[0]
                lig_contacts = np.where(near_lig[wi])[0]

                for pi in prot_contacts:
                    for li in lig_contacts:
                        pa = protein_ag[pi]
                        la = ligand_ag[li]
                        event_rows.append(
                            {
                                "frame": int(ts.frame),
                                "protein_atom": f"{pa.resname}{pa.resid}:{pa.name}",
                                "water_resid": int(water_atom.resid),
                                "ligand_atom": f"{la.resname}{la.resid}:{la.name}",
                                "dist_pw": round(float(d_wp[wi, pi]), 3),
                                "dist_wl": round(float(d_wl[wi, li]), 3),
                            }
                        )

        events = pd.DataFrame(event_rows) if event_rows else _empty_events()
        summary = _compute_summary(events, n_frames)

        logger.info(
            "Water bridge analysis done for %s: %d bridge events across %d frames",
            sample_name, len(events), n_frames,
        )
        return WaterBridgeResult(
            sample_name=sample_name,
            events=events,
            summary=summary,
            n_frames=n_frames,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frame", "protein_atom", "water_resid",
            "ligand_atom", "dist_pw", "dist_wl",
        ]
    )


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protein_atom", "ligand_atom", "occupancy_pct",
            "mean_dist_pw", "mean_dist_wl",
        ]
    )


def _compute_summary(events: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Aggregate per-bridge occupancy from event records."""
    if events.empty:
        return _empty_summary()

    # Deduplicate within each frame (same protein-water-ligand triple counted once)
    dedup = events.drop_duplicates(subset=["frame", "protein_atom", "ligand_atom"])

    grp = dedup.groupby(["protein_atom", "ligand_atom"])
    summary = grp.agg(
        n_frames_present=("frame", "count"),
        mean_dist_pw=("dist_pw", "mean"),
        mean_dist_wl=("dist_wl", "mean"),
    ).reset_index()
    summary["occupancy_pct"] = (summary["n_frames_present"] / n_frames * 100).round(2)
    summary["mean_dist_pw"] = summary["mean_dist_pw"].round(3)
    summary["mean_dist_wl"] = summary["mean_dist_wl"].round(3)
    summary = summary.drop(columns="n_frames_present")
    summary = summary.sort_values("occupancy_pct", ascending=False).reset_index(drop=True)
    return summary
