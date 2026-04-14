"""Protein-ligand hydrogen bond analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

from ..config import AnalysisConfig
from ..universe import load_and_align

logger = logging.getLogger(__name__)


@dataclass
class HBondResult:
    """Container for hydrogen bond analysis results of one replica."""

    sample_name: str
    events: pd.DataFrame   # per-event rows
    summary: pd.DataFrame  # per-hbond_id aggregation
    n_frames: int


def _build_selections(u: mda.Universe, ligand_resname: str) -> tuple[str, str, str]:
    """Build element-based donor/hydrogen/acceptor selections.

    Uses N*/O*/S* pattern so topology charge information is not required.
    """
    scope = f"(protein or resname {ligand_resname})"
    acceptors_sel = f"{scope} and (name N* or name O* or name S*)"
    hydrogens_sel = (
        f"{scope} and name H* and "
        f"(bonded (name N* or name O* or name S*))"
    )
    donors_sel = (
        f"{scope} and (name N* or name O* or name S*) and "
        f"(bonded name H*)"
    )
    n_h = u.select_atoms(hydrogens_sel).n_atoms
    if n_h == 0:
        logger.warning(
            "No hydrogen atoms found in topology. "
            "H-bond analysis requires explicit hydrogens (typical of MD trajectories). "
            "Returning empty result."
        )
        return None, None, None
    logger.debug(
        "selections: acc=%d  hyd=%d  don=%d",
        u.select_atoms(acceptors_sel).n_atoms,
        n_h,
        u.select_atoms(donors_sel).n_atoms,
    )
    return donors_sel, hydrogens_sel, acceptors_sel


def _atom_label(u: mda.Universe, idx: int) -> str:
    a = u.atoms[idx]
    return f"{a.resname}{a.resid}:{a.name}"


def _build_events_df(hba: HydrogenBondAnalysis, u: mda.Universe) -> pd.DataFrame:
    if len(hba.results.hbonds) == 0:
        logger.warning("No hydrogen bonds detected.")
        return pd.DataFrame(
            columns=["frame", "donor_idx", "hydrogen_idx", "acceptor_idx",
                     "distance", "angle", "donor", "hydrogen", "acceptor", "hbond_id"]
        )
    cols = ["frame", "donor_idx", "hydrogen_idx", "acceptor_idx", "distance", "angle"]
    df = pd.DataFrame(hba.results.hbonds, columns=cols)
    df = df.astype({"frame": int, "donor_idx": int, "hydrogen_idx": int, "acceptor_idx": int})
    df["donor"] = df["donor_idx"].map(lambda i: _atom_label(u, i))
    df["hydrogen"] = df["hydrogen_idx"].map(lambda i: _atom_label(u, i))
    df["acceptor"] = df["acceptor_idx"].map(lambda i: _atom_label(u, i))
    df["hbond_id"] = df["donor"] + "···" + df["acceptor"]
    return df


def _summarize(events: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    summary = (
        events.groupby("hbond_id")
        .agg(
            count=("frame", "count"),
            mean_dist=("distance", "mean"),
            std_dist=("distance", "std"),
            mean_angle=("angle", "mean"),
            donor=("donor", "first"),
            acceptor=("acceptor", "first"),
        )
        .reset_index()
    )
    summary["occupancy_%"] = (summary["count"] / n_frames * 100).round(1)
    return summary.sort_values("occupancy_%", ascending=False).reset_index(drop=True)


class HBondAnalyzer:
    """Run protein-ligand hydrogen bond analysis on a single Universe."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def run(self, u: mda.Universe, sample_name: str) -> HBondResult:
        c = self.cfg.hbond
        donors_sel, hydrogens_sel, acceptors_sel = _build_selections(
            u, self.cfg.ligand_resname
        )
        if donors_sel is None:
            # No hydrogens in topology — return empty result gracefully
            n_frames = len(u.trajectory[c.start : c.stop : c.step])
            empty_cols_events = [
                "frame", "donor_idx", "hydrogen_idx", "acceptor_idx",
                "distance", "angle", "donor", "hydrogen", "acceptor", "hbond_id",
            ]
            return HBondResult(
                sample_name=sample_name,
                events=pd.DataFrame(columns=empty_cols_events),
                summary=pd.DataFrame(),
                n_frames=n_frames,
            )
        hba = HydrogenBondAnalysis(
            universe=u,
            donors_sel=donors_sel,
            hydrogens_sel=hydrogens_sel,
            acceptors_sel=acceptors_sel,
            between=["protein", f"resname {self.cfg.ligand_resname}"],
            d_a_cutoff=c.d_a_cutoff,
            d_h_a_angle_cutoff=c.angle_cutoff,
            update_selections=False,
        )
        hba.run(start=c.start, stop=c.stop, step=c.step)
        logger.info(
            "%s: %d hbond events detected", sample_name, len(hba.results.hbonds)
        )
        n_frames = len(u.trajectory[c.start : c.stop : c.step])
        events = _build_events_df(hba, u)
        summary = _summarize(events, n_frames)
        return HBondResult(
            sample_name=sample_name,
            events=events,
            summary=summary,
            n_frames=n_frames,
        )


def run_hbond_batch(
    replica_dirs: list[Path | str],
    cfg: AnalysisConfig,
) -> dict[str, HBondResult]:
    """Run hydrogen bond analysis for all replicas.

    Returns dict mapping *sample_name* → :class:`HBondResult`.
    """
    from ..io.loaders import discover_replicas

    replicas = discover_replicas(replica_dirs, cfg)
    analyzer = HBondAnalyzer(cfg)
    results: dict[str, HBondResult] = {}
    for rep in replicas:
        u = load_and_align(rep["topology"], rep["trajectory"], cfg)
        result = analyzer.run(u, rep["name"])
        # Save CSVs immediately
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        result.events.to_csv(
            cfg.output_dir / f"hbond_all_events_{rep['name']}.csv", index=False
        )
        result.summary.to_csv(
            cfg.output_dir / f"hbond_summary_{rep['name']}.csv", index=False
        )
        results[rep["name"]] = result
    return results
