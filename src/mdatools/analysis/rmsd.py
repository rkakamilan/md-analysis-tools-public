"""RMSD calculation for protein backbone and ligand."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis import rms

from ..config import AnalysisConfig
from ..universe import load_and_align

logger = logging.getLogger(__name__)


@dataclass
class RMSDResult:
    sample_name: str
    df: pd.DataFrame  # columns: Frame, Time, Backbone, Ligand


class RMSDAnalyzer:
    """Compute backbone + ligand RMSD for a single Universe."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def run(self, u: mda.Universe, sample_name: str) -> RMSDResult:
        """Run RMSD analysis and return a :class:`RMSDResult`."""
        c = self.cfg.rmsd
        group_selections = [f"resname {self.cfg.ligand_resname}"]
        try:
            masses = u.atoms.masses
            weights = "mass" if masses.sum() > 0 else None
            if weights is None:
                logger.warning(
                    "%s: all atom masses are zero — using equal weights for RMSD.",
                    sample_name,
                )
        except Exception:
            weights = None
        R = rms.RMSD(
            u,
            select=c.backbone_select,
            groupselections=group_selections,
            ref_frame=c.ref_frame,
            weights=weights,
        )
        R.run(start=c.start, stop=c.stop, step=c.step)
        df = pd.DataFrame(
            R.results.rmsd,
            columns=["Frame", "Time", "Backbone", "Ligand"],
        )
        logger.info("RMSD done for %s: %d frames", sample_name, len(df))
        return RMSDResult(sample_name=sample_name, df=df)


def run_rmsd_batch(
    replica_dirs: list[Path | str],
    cfg: AnalysisConfig,
) -> dict[str, RMSDResult]:
    """Run RMSD analysis for all replicas.

    Parameters
    ----------
    replica_dirs:
        List of directories, each containing a topology and trajectory
        matched by ``cfg.topology_glob`` / ``cfg.trajectory_glob``.
    cfg:
        Analysis configuration.

    Returns
    -------
    dict mapping *sample_name* → :class:`RMSDResult`.
    """
    from ..io.loaders import discover_replicas

    replicas = discover_replicas(replica_dirs, cfg)
    analyzer = RMSDAnalyzer(cfg)
    results: dict[str, RMSDResult] = {}
    for rep in replicas:
        u = load_and_align(rep["topology"], rep["trajectory"], cfg)
        results[rep["name"]] = analyzer.run(u, rep["name"])
    return results
