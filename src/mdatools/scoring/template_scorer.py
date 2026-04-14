"""Template selection scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.hbonds import HBondResult
from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class ScoredTemplate:
    sample_name: str
    hbond_id: str
    occupancy_pct: float
    mean_dist: float
    mean_angle: float
    hbond_score: float
    group_bonuses: dict[str, float]  # group_name -> bonus earned
    stability_score: float
    total_score: float
    group_hits: dict[str, bool]

    @property
    def total_group_bonus(self) -> float:
        return sum(self.group_bonuses.values())


class TemplateScorer:
    """Score HBond results for template selection.

    Scoring formula (all per best H-bond):
        hbond_score   = occ_score + dist_score + angle_score
        group_bonuses = sum over residue_groups of bonus if hit
        stability_score = fraction of events in latter half * weight
        total_score   = hbond_score + group_bonuses + stability_score
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def _hbond_score(self, row: pd.Series) -> tuple[float, float, float, float]:
        w = self.cfg.scoring
        occ = (row["occupancy_%"] / 100.0) * w.occ_weight
        dist = np.clip((w.dist_ref - row["mean_dist"]) / w.dist_range, 0, 1) * w.dist_weight
        angle = np.clip((row["mean_angle"] - w.angle_ref) / w.angle_range, 0, 1) * w.angle_weight
        return occ, dist, angle, occ + dist + angle

    def _group_bonuses(self, row: pd.Series) -> tuple[dict[str, float], dict[str, bool]]:
        bonuses: dict[str, float] = {}
        hits: dict[str, bool] = {}
        donor = str(row.get("donor", ""))
        acceptor = str(row.get("acceptor", ""))
        for gname, grp in self.cfg.residue_groups.items():
            # Match by resname<resid> pattern from residue_groups
            patterns: list[str] = []
            for resid, resname in zip(grp.resids, grp.resnames or [""]*len(grp.resids)):
                patterns.append(str(resid))
                if resname:
                    patterns.append(resname)
            # Also accept bare resids
            bare_resids = [str(r) for r in grp.resids]
            hit = any(
                p in donor or p in acceptor
                for p in bare_resids + (grp.resnames or [])
            )
            hits[gname] = hit
            bonuses[gname] = grp.bonus if hit else 0.0
        return bonuses, hits

    def _stability_score(self, result: HBondResult, hbond_id: str) -> float:
        w = self.cfg.scoring
        ev = result.events
        if ev.empty:
            return 0.0
        target = ev[ev["hbond_id"] == hbond_id]
        if target.empty:
            return 0.0
        max_frame = ev["frame"].max()
        mid = max_frame / 2.0
        frac_latter = (target["frame"] >= mid).mean()
        return float(frac_latter) * w.stability_weight

    def score_all(
        self, results: dict[str, HBondResult]
    ) -> list[ScoredTemplate]:
        """Score all replicas and return sorted list (highest score first)."""
        scored: list[ScoredTemplate] = []
        for name, result in results.items():
            if result.summary.empty:
                logger.warning("No H-bonds for %s, skipping.", name)
                continue
            best = result.summary.iloc[0]
            occ, dist, angle, hbond_score = self._hbond_score(best)
            bonuses, hits = self._group_bonuses(best)
            stab = self._stability_score(result, best["hbond_id"])
            total = hbond_score + sum(bonuses.values()) + stab
            scored.append(
                ScoredTemplate(
                    sample_name=name,
                    hbond_id=best["hbond_id"],
                    occupancy_pct=float(best["occupancy_%"]),
                    mean_dist=float(best["mean_dist"]),
                    mean_angle=float(best["mean_angle"]),
                    hbond_score=round(hbond_score, 2),
                    group_bonuses={k: round(v, 2) for k, v in bonuses.items()},
                    stability_score=round(stab, 2),
                    total_score=round(total, 2),
                    group_hits=hits,
                )
            )
        return sorted(scored, key=lambda s: s.total_score, reverse=True)

    def to_dataframe(self, scored: list[ScoredTemplate]) -> pd.DataFrame:
        rows = []
        for s in scored:
            row = {
                "sample": s.sample_name,
                "hbond_id": s.hbond_id,
                "occupancy_%": s.occupancy_pct,
                "mean_dist": s.mean_dist,
                "mean_angle": s.mean_angle,
                "hbond_score": s.hbond_score,
                "stability_score": s.stability_score,
                "total_score": s.total_score,
            }
            for gname, bonus in s.group_bonuses.items():
                row[f"bonus_{gname}"] = bonus
                row[f"hit_{gname}"] = s.group_hits.get(gname, False)
            rows.append(row)
        return pd.DataFrame(rows)
