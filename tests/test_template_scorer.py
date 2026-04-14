"""Tests for TemplateScorer."""

import pandas as pd
import pytest

from mdatools.analysis.hbonds import HBondResult
from mdatools.config import AnalysisConfig, ResidueGroup, ScoringWeights
from mdatools.scoring.template_scorer import ScoredTemplate, TemplateScorer


def _make_result(
    sample: str,
    hbond_id: str = "UNK0:N4···ASN290:OD1",
    occupancy: float = 80.0,
    mean_dist: float = 3.1,
    mean_angle: float = 160.0,
    donor: str = "UNK0:N4",
    acceptor: str = "ASN290:OD1",
    n_frames: int = 100,
) -> HBondResult:
    frames = list(range(int(n_frames * occupancy / 100)))
    events = pd.DataFrame({
        "frame": frames,
        "distance": [mean_dist] * len(frames),
        "angle": [mean_angle] * len(frames),
        "hbond_id": [hbond_id] * len(frames),
        "donor": [donor] * len(frames),
        "acceptor": [acceptor] * len(frames),
    })
    summary = pd.DataFrame([{
        "hbond_id": hbond_id,
        "count": len(frames),
        "mean_dist": mean_dist,
        "std_dist": 0.1,
        "mean_angle": mean_angle,
        "donor": donor,
        "acceptor": acceptor,
        "occupancy_%": occupancy,
    }])
    return HBondResult(
        sample_name=sample,
        events=events,
        summary=summary,
        n_frames=n_frames,
    )


def test_scorer_returns_sorted_list():
    cfg = AnalysisConfig(ligand_resname="UNK")
    scorer = TemplateScorer(cfg)
    results = {
        "high": _make_result("high", occupancy=90.0, mean_dist=3.0),
        "low": _make_result("low", occupancy=30.0, mean_dist=3.4),
    }
    scored = scorer.score_all(results)
    assert len(scored) == 2
    assert scored[0].total_score >= scored[1].total_score
    assert scored[0].sample_name == "high"


def test_scorer_group_bonus():
    cfg = AnalysisConfig(
        ligand_resname="UNK",
        residue_groups={
            "ECD": ResidueGroup(
                name="ECD",
                resids=[290],
                resnames=["ASN290"],
                bonus=20.0,
            )
        },
    )
    scorer = TemplateScorer(cfg)
    results = {
        "hit": _make_result("hit", acceptor="ASN290:OD1"),
        "miss": _make_result("miss", acceptor="GLY100:O"),
    }
    scored = scorer.score_all(results)
    hit = next(s for s in scored if s.sample_name == "hit")
    miss = next(s for s in scored if s.sample_name == "miss")
    assert hit.group_bonuses.get("ECD", 0) == pytest.approx(20.0)
    assert miss.group_bonuses.get("ECD", 0) == pytest.approx(0.0)


def test_scorer_to_dataframe():
    cfg = AnalysisConfig(ligand_resname="UNK")
    scorer = TemplateScorer(cfg)
    results = {"s1": _make_result("s1")}
    scored = scorer.score_all(results)
    df = scorer.to_dataframe(scored)
    assert "sample" in df.columns
    assert "total_score" in df.columns
    assert len(df) == 1


def test_scorer_empty_results():
    cfg = AnalysisConfig(ligand_resname="UNK")
    scorer = TemplateScorer(cfg)
    # Empty HBondResult should be skipped gracefully
    empty = HBondResult(
        sample_name="empty",
        events=pd.DataFrame(),
        summary=pd.DataFrame(),
        n_frames=100,
    )
    scored = scorer.score_all({"empty": empty})
    assert scored == []
