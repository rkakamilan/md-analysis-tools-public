"""Filesystem discovery and CSV loading helpers."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


def discover_replicas(
    root_dirs: list[Path | str],
    cfg: AnalysisConfig,
) -> list[dict]:
    """Find topology + trajectory file pairs under *root_dirs*.

    Each *root_dir* is searched with ``cfg.topology_glob`` and
    ``cfg.trajectory_glob``.  Returns a list of dicts::

        [{"name": str, "topology": Path, "trajectory": Path}, ...]

    Raises ``FileNotFoundError`` if a directory yields no matches.
    """
    results: list[dict] = []
    for root in root_dirs:
        root = Path(root)
        tops = sorted(root.glob(cfg.topology_glob))
        trajs = sorted(root.glob(cfg.trajectory_glob))
        if not tops:
            raise FileNotFoundError(
                f"No topology matching '{cfg.topology_glob}' in {root}"
            )
        if not trajs:
            raise FileNotFoundError(
                f"No trajectory matching '{cfg.trajectory_glob}' in {root}"
            )
        results.append(
            {
                "name": root.name,
                "topology": tops[0],
                "trajectory": trajs[0],
            }
        )
        logger.info("Discovered replica: %s", root.name)
    return results


def load_hbond_summary(output_dir: Path | str, sample_name: str) -> pd.DataFrame:
    """Load hbond_summary_{sample_name}.csv from *output_dir*."""
    path = Path(output_dir) / f"hbond_summary_{sample_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_hbond_events(output_dir: Path | str, sample_name: str) -> pd.DataFrame:
    """Load hbond_all_events_{sample_name}.csv from *output_dir*."""
    path = Path(output_dir) / f"hbond_all_events_{sample_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_all_summaries(output_dir: Path | str) -> pd.DataFrame:
    """Load all hbond_summary_*.csv files and concatenate with a 'sample' column."""
    rows = []
    for fpath in sorted(Path(output_dir).glob("hbond_summary_*.csv")):
        sample = fpath.stem.replace("hbond_summary_", "")
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        best = df.sort_values("occupancy_%", ascending=False).iloc[0].copy()
        best["sample"] = sample
        best["n_hbonds"] = len(df)
        rows.append(best)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)
