"""Multi-metric consensus ranking and Pareto front extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Results from consensus ranking.

    Attributes
    ----------
    ranking:
        DataFrame with one row per sample. Columns include the original
        metric values, normalised scores, and final ranking columns:
        ``consensus_score`` and ``rank`` (1 = best).
    pareto_front:
        List of sample names (index values) that belong to the Pareto front
        (non-dominated solutions) for the objectives used when calling
        :meth:`ConsensusRanker.extract_pareto`.
    method:
        Ranking method used: ``'borda'`` or ``'zscore'``.
    """

    ranking: pd.DataFrame
    pareto_front: list[str]
    method: str


class ConsensusRanker:
    """Aggregate multiple MD/scoring metrics into a single consensus ranking.

    Parameters
    ----------
    weights:
        Optional mapping of *metric_name* → weight. Metrics absent from
        this dict receive weight 1.0. Metrics with weight 0 are excluded
        from the consensus score.
    higher_is_better:
        Mapping of *metric_name* → ``True``/``False``.  When ``True``
        (default for all metrics) a higher raw value is considered better
        (e.g. H-bond occupancy). Set to ``False`` for metrics where lower
        is better (e.g. RMSD, SA-score).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        self.weights = weights or {}
        self.higher_is_better = higher_is_better or {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def rank_borda(self, metrics_df: pd.DataFrame) -> ConsensusResult:
        """Weighted Borda count consensus ranking.

        Each metric column is converted to a rank (best = highest rank
        integer), then multiplied by the column's weight and summed.

        Parameters
        ----------
        metrics_df:
            DataFrame with shape *(n_samples, n_metrics)*. Row index
            contains sample names.

        Returns
        -------
        :class:`ConsensusResult` with ``method='borda'``.
        """
        cols = self._active_columns(metrics_df)
        if not cols:
            return self._empty_result(metrics_df, "borda")

        ranks = pd.DataFrame(index=metrics_df.index)
        for col in cols:
            # ascending=True  → rank 1 = min, rank N = max  (use for higher-is-better)
            # ascending=False → rank 1 = max, rank N = min  (use for lower-is-better)
            # We want rank N = best, so higher-is-better cols use ascending=True.
            ascending = self._higher_is_better(col)
            ranks[col] = metrics_df[col].rank(ascending=ascending, method="average")

        weighted_sum = sum(
            ranks[col] * self._weight(col) for col in cols
        )
        return self._build_result(metrics_df, weighted_sum, "borda")

    def rank_zscore(self, metrics_df: pd.DataFrame) -> ConsensusResult:
        """Weighted Z-score normalisation consensus ranking.

        Each metric column is Z-score normalised, then the sign is flipped
        for metrics where lower is better, and the weighted average is
        computed.

        Parameters
        ----------
        metrics_df:
            DataFrame with shape *(n_samples, n_metrics)*.

        Returns
        -------
        :class:`ConsensusResult` with ``method='zscore'``.
        """
        cols = self._active_columns(metrics_df)
        if not cols:
            return self._empty_result(metrics_df, "zscore")

        z_scores = pd.DataFrame(index=metrics_df.index)
        for col in cols:
            vals = metrics_df[col].astype(float)
            std = vals.std(ddof=1)
            if std == 0:
                z = pd.Series(0.0, index=vals.index)
            else:
                z = (vals - vals.mean()) / std
            if not self._higher_is_better(col):
                z = -z
            z_scores[col] = z

        total_weight = sum(self._weight(col) for col in cols)
        if total_weight == 0:
            return self._empty_result(metrics_df, "zscore")

        weighted_sum = sum(
            z_scores[col] * self._weight(col) for col in cols
        ) / total_weight
        return self._build_result(metrics_df, weighted_sum, "zscore")

    def extract_pareto(
        self,
        metrics_df: pd.DataFrame,
        objectives: list[str] | None = None,
    ) -> list[str]:
        """Extract the Pareto front from a metrics DataFrame.

        A sample is on the Pareto front if no other sample dominates it
        across all objectives simultaneously.

        Parameters
        ----------
        metrics_df:
            DataFrame with shape *(n_samples, n_metrics)*.
        objectives:
            Subset of column names to consider. Defaults to all columns.

        Returns
        -------
        List of sample names (index values) that are on the Pareto front.
        """
        if objectives is None:
            objectives = list(metrics_df.columns)

        objectives = [o for o in objectives if o in metrics_df.columns]
        if not objectives:
            return []

        # Build sign array: +1 if higher-is-better, -1 otherwise.
        # Multiplying values by sign converts every objective to
        # "higher normalised value = better".
        signs = np.array(
            [1.0 if self._higher_is_better(o) else -1.0 for o in objectives]
        )
        M = metrics_df[objectives].values.astype(float) * signs  # (n, k)

        n = len(M)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                if _dominates(M[j], M[i]):
                    is_pareto[i] = False
                    break

        pareto_names = list(metrics_df.index[is_pareto])
        logger.debug(
            "Pareto front: %d / %d samples", len(pareto_names), n
        )
        return pareto_names

    def combine(
        self,
        metrics_df: pd.DataFrame,
        method: str = "borda",
        pareto_objectives: list[str] | None = None,
    ) -> ConsensusResult:
        """Convenience entry point: rank + extract Pareto front.

        Parameters
        ----------
        metrics_df:
            DataFrame with shape *(n_samples, n_metrics)*.
        method:
            ``'borda'`` (default) or ``'zscore'``.
        pareto_objectives:
            Columns to use for Pareto front extraction. Defaults to all
            columns.

        Returns
        -------
        :class:`ConsensusResult` with ``pareto_front`` populated.
        """
        if method == "borda":
            result = self.rank_borda(metrics_df)
        elif method == "zscore":
            result = self.rank_zscore(metrics_df)
        else:
            raise ValueError(f"Unknown method: '{method}'. Use 'borda' or 'zscore'.")

        pareto = self.extract_pareto(metrics_df, objectives=pareto_objectives)
        result.pareto_front = pareto
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weight(self, col: str) -> float:
        return self.weights.get(col, 1.0)

    def _higher_is_better(self, col: str) -> bool:
        return self.higher_is_better.get(col, True)

    def _active_columns(self, df: pd.DataFrame) -> list[str]:
        """Return columns with non-zero weight present in df."""
        return [c for c in df.columns if self._weight(c) != 0]

    def _build_result(
        self,
        metrics_df: pd.DataFrame,
        score_series: pd.Series,
        method: str,
    ) -> ConsensusResult:
        out = metrics_df.copy()
        out["consensus_score"] = score_series
        out["rank"] = out["consensus_score"].rank(
            ascending=False, method="min"
        ).astype(int)
        out = out.sort_values("rank")
        return ConsensusResult(ranking=out, pareto_front=[], method=method)

    def _empty_result(self, metrics_df: pd.DataFrame, method: str) -> ConsensusResult:
        out = metrics_df.copy()
        out["consensus_score"] = float("nan")
        out["rank"] = range(1, len(out) + 1)
        return ConsensusResult(ranking=out, pareto_front=[], method=method)


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates solution *b*.

    *a* dominates *b* when *a* is at least as good on all objectives and
    strictly better on at least one.  Higher values are assumed to be better
    (call-site normalises the sign).
    """
    return bool(np.all(a >= b) and np.any(a > b))
