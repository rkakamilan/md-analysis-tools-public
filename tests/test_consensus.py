"""Tests for ConsensusRanker, ConsensusResult and consensus plot functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.scoring.consensus import ConsensusRanker, ConsensusResult, _dominates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df():
    """3 samples × 3 metrics. Expected best: S2, worst: S1."""
    return pd.DataFrame(
        {
            "hbond_occ": [0.3, 0.9, 0.6],   # higher is better
            "rmsd":      [2.5, 0.8, 1.5],   # lower is better
            "qed":       [0.4, 0.8, 0.6],   # higher is better
        },
        index=["S1", "S2", "S3"],
    )


@pytest.fixture()
def ranker():
    return ConsensusRanker(
        higher_is_better={"hbond_occ": True, "rmsd": False, "qed": True}
    )


# ---------------------------------------------------------------------------
# _dominates helper
# ---------------------------------------------------------------------------


class TestDominates:
    def test_a_dominates_b(self):
        assert _dominates(np.array([2.0, 3.0]), np.array([1.0, 2.0])) is True

    def test_equal_does_not_dominate(self):
        assert _dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0])) is False

    def test_partial_better_does_not_dominate(self):
        # a is better on dim 0 but worse on dim 1
        assert _dominates(np.array([2.0, 0.0]), np.array([1.0, 2.0])) is False

    def test_all_better_dominates(self):
        assert _dominates(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])) is True


# ---------------------------------------------------------------------------
# Borda count
# ---------------------------------------------------------------------------


class TestBorda:
    def test_best_sample_rank_one(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        assert result.ranking.loc["S2", "rank"] == 1

    def test_worst_sample_rank_last(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        n = len(simple_df)
        assert result.ranking.loc["S1", "rank"] == n

    def test_method_label(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        assert result.method == "borda"

    def test_ranking_has_consensus_score(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        assert "consensus_score" in result.ranking.columns

    def test_ranking_has_rank(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        assert "rank" in result.ranking.columns

    def test_pareto_front_empty_by_default(self, simple_df, ranker):
        result = ranker.rank_borda(simple_df)
        assert result.pareto_front == []

    def test_zero_weight_excluded(self, simple_df):
        # Assign zero weight to rmsd — result should be driven by hbond + qed
        r_with = ConsensusRanker(
            weights={"rmsd": 0.0},
            higher_is_better={"hbond_occ": True, "rmsd": False, "qed": True},
        )
        r_without = ConsensusRanker(
            higher_is_better={"hbond_occ": True, "qed": True},
        )
        res_with = r_with.rank_borda(simple_df)
        res_without = r_without.rank_borda(simple_df[["hbond_occ", "qed"]])
        assert list(res_with.ranking["rank"]) == list(res_without.ranking["rank"])

    def test_lower_is_better_inverts_rank(self):
        df = pd.DataFrame({"rmsd": [3.0, 1.0, 2.0]}, index=["A", "B", "C"])
        ranker = ConsensusRanker(higher_is_better={"rmsd": False})
        result = ranker.rank_borda(df)
        assert result.ranking.loc["B", "rank"] == 1  # lowest RMSD → best

    def test_all_equal_same_score(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]}, index=["A", "B", "C"])
        ranker = ConsensusRanker()
        result = ranker.rank_borda(df)
        scores = result.ranking["consensus_score"].values
        assert np.allclose(scores, scores[0])


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


class TestZscore:
    def test_best_sample_rank_one(self, simple_df, ranker):
        result = ranker.rank_zscore(simple_df)
        assert result.ranking.loc["S2", "rank"] == 1

    def test_method_label(self, simple_df, ranker):
        result = ranker.rank_zscore(simple_df)
        assert result.method == "zscore"

    def test_constant_column_zero_zscore(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [3.0, 1.0, 2.0]}, index=list("ABC"))
        ranker = ConsensusRanker(higher_is_better={"y": False})
        result = ranker.rank_zscore(df)
        # A should be best (lowest y=1)
        assert result.ranking.loc["B", "rank"] == 1

    def test_borda_zscore_agree_on_top(self, simple_df, ranker):
        borda = ranker.rank_borda(simple_df)
        zscore = ranker.rank_zscore(simple_df)
        assert borda.ranking["rank"].idxmin() == zscore.ranking["rank"].idxmin()

    def test_zero_weight_excluded(self, simple_df):
        r = ConsensusRanker(
            weights={"rmsd": 0.0},
            higher_is_better={"hbond_occ": True, "rmsd": False, "qed": True},
        )
        r_base = ConsensusRanker(
            higher_is_better={"hbond_occ": True, "qed": True},
        )
        res = r.rank_zscore(simple_df)
        res_base = r_base.rank_zscore(simple_df[["hbond_occ", "qed"]])
        assert list(res.ranking["rank"]) == list(res_base.ranking["rank"])


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_dominated_sample_excluded(self, simple_df, ranker):
        # S1 is dominated by S2 on all objectives (worse hbond, worse rmsd, worse qed)
        pareto = ranker.extract_pareto(simple_df)
        assert "S1" not in pareto

    def test_pareto_samples_are_non_dominated(self, simple_df, ranker):
        pareto = ranker.extract_pareto(simple_df)
        signs = np.array([1, -1, 1])  # hbond+, rmsd-, qed+
        M = simple_df.values * signs
        pareto_idx = [simple_df.index.get_loc(n) for n in pareto]
        all_idx = list(range(len(simple_df)))
        for pi in pareto_idx:
            for j in all_idx:
                if j != pi:
                    assert not _dominates(M[j], M[pi]), (
                        f"{simple_df.index[j]} should not dominate {simple_df.index[pi]}"
                    )

    def test_subset_objectives(self, simple_df, ranker):
        # Only considering rmsd (lower is better): S2 has best RMSD
        pareto = ranker.extract_pareto(simple_df, objectives=["rmsd"])
        assert "S2" in pareto

    def test_single_sample_is_pareto(self):
        df = pd.DataFrame({"a": [1.0]}, index=["X"])
        ranker = ConsensusRanker()
        assert ranker.extract_pareto(df) == ["X"]

    def test_empty_objectives_returns_empty(self, simple_df, ranker):
        pareto = ranker.extract_pareto(simple_df, objectives=["nonexistent"])
        assert pareto == []


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------


class TestCombine:
    def test_combine_borda_populates_pareto(self, simple_df, ranker):
        result = ranker.combine(simple_df, method="borda")
        assert len(result.pareto_front) >= 1

    def test_combine_zscore(self, simple_df, ranker):
        result = ranker.combine(simple_df, method="zscore")
        assert result.method == "zscore"

    def test_combine_unknown_method_raises(self, simple_df, ranker):
        with pytest.raises(ValueError, match="Unknown method"):
            ranker.combine(simple_df, method="invalid")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestConsensusPlots:
    @pytest.fixture()
    def result(self, simple_df, ranker):
        return ranker.combine(simple_df)

    def test_plot_consensus_ranking(self, result):
        from mdatools.plotting.consensus_plots import plot_consensus_ranking
        fig = plot_consensus_ranking(result, top_n=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_consensus_ranking_axis_label(self, result):
        from mdatools.plotting.consensus_plots import plot_consensus_ranking
        fig = plot_consensus_ranking(result)
        ax = fig.axes[0]
        assert "consensus" in ax.get_xlabel().lower() or "score" in ax.get_xlabel().lower()
        plt.close(fig)

    def test_plot_pareto_front(self, simple_df, result):
        from mdatools.plotting.consensus_plots import plot_pareto_front
        fig = plot_pareto_front(
            simple_df, "hbond_occ", "rmsd", result.pareto_front
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pareto_front_axis_labels(self, simple_df, result):
        from mdatools.plotting.consensus_plots import plot_pareto_front
        fig = plot_pareto_front(
            simple_df, "hbond_occ", "rmsd", result.pareto_front
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "hbond_occ"
        assert ax.get_ylabel() == "rmsd"
        plt.close(fig)

    def test_plot_metric_correlation(self, simple_df):
        from mdatools.plotting.consensus_plots import plot_metric_correlation
        fig = plot_metric_correlation(simple_df)
        assert isinstance(fig, plt.Figure)
        # Should be a square heatmap (one axes)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_pareto_front_empty_pareto(self, simple_df):
        from mdatools.plotting.consensus_plots import plot_pareto_front
        fig = plot_pareto_front(simple_df, "hbond_occ", "qed", [])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
