"""Virtual screening evaluation metrics.

Provides Enrichment Factor (EF), ROC-AUC, and BEDROC for evaluating
docking and scoring functions against known actives.

Ported from ``docking_analysis.analysis.vs_metrics``.

Example::

    import numpy as np
    from mdatools.docking.analysis.vs_metrics import VirtualScreeningEvaluator

    scores = np.array([-9.0, -7.5, -8.5, -6.0, -9.5, -7.0])
    labels = np.array([1, 0, 1, 0, 1, 0])

    ev = VirtualScreeningEvaluator()
    result = ev.run(scores, labels)
    print(result.ef[1.0])    # EF at 1 %
    print(result.roc_auc)
    print(result.bedroc)

References
----------
Truchon & Bayly (2007) J. Chem. Inf. Model. 47:488-508 (BEDROC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Default EF percentile cutpoints.
EF_PERCENTILES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass
class VSMetricsResult:
    """Metrics for one scoring vector.

    Attributes
    ----------
    sample_name:
        Identifier for this evaluation run.
    n_total:
        Total number of compounds.
    n_actives:
        Number of known actives.
    ef:
        EF values keyed by percentile (1.0, 2.0, 5.0, 10.0, 20.0).
    ef_max:
        Theoretical maximum EF at each percentile.
    roc_auc:
        Area under the ROC curve.
    bedroc:
        BEDROC score (alpha=20.0).
    fpr:
        False-positive rate array (from sklearn).
    tpr:
        True-positive rate array (from sklearn).
    """

    sample_name: str = ""
    n_total: int = 0
    n_actives: int = 0
    ef: dict[float, float] = field(default_factory=dict)
    ef_max: dict[float, float] = field(default_factory=dict)
    roc_auc: float = 0.0
    bedroc: float = 0.0
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ef_norm(self) -> dict[float, float]:
        """Normalised EF in [0, 1] — ``ef / ef_max`` per percentile."""
        return {
            pct: (
                self.ef[pct] / self.ef_max[pct] if self.ef_max.get(pct, 0) > 0 else 0.0
            )
            for pct in self.ef
        }


def _enrichment_factor(
    scores: np.ndarray,
    labels: np.ndarray,
    pct: float,
) -> tuple[float, float]:
    """Compute EF and EF_max at the top-*pct*% of the ranked list.

    Parameters
    ----------
    scores:
        Docking scores (lower = better; will be negated internally).
    labels:
        Binary activity labels (1 = active).
    pct:
        Percentage cutoff in the range (0, 100].

    Returns
    -------
    tuple[float, float]
        ``(ef, ef_max)``
    """
    n = len(scores)
    n_act = int(labels.sum())
    if n_act == 0 or n == 0:
        return 0.0, 0.0

    n_top = max(1, int(np.ceil(n * pct / 100.0)))
    sorted_idx = np.argsort(scores)  # ascending (lower score = better rank)
    top_actives = int(labels[sorted_idx[:n_top]].sum())

    ef = (top_actives / n_top) / (n_act / n)
    ef_max_val = min(1.0, n_act / n_top) / (n_act / n)
    return float(ef), float(ef_max_val)


def _bedroc(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 20.0,
) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    Parameters
    ----------
    scores:
        Docking scores (lower = better, will be negated internally).
    labels:
        Binary activity labels.
    alpha:
        Exponential decay parameter (default 20.0 — early recognition focus).

    Returns
    -------
    float
        BEDROC in [0, 1].
    """
    n = len(scores)
    n_act = int(labels.sum())
    if n_act == 0 or n == 0:
        return 0.0

    sorted_idx = np.argsort(scores)  # ascending ranks
    active_ranks = np.where(labels[sorted_idx] == 1)[0] + 1  # 1-indexed

    # Weighted sum of active ranks
    rie = np.sum(np.exp(-alpha * active_ranks / n)) / (
        n_act / n * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    )

    # Bounds for normalisation (Truchon & Bayly 2007, Eqs. 16-17)
    ra = n_act / n
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))

    if rie_max - rie_min < 1e-12:
        return 0.0
    return float((rie - rie_min) / (rie_max - rie_min))


class VirtualScreeningEvaluator:
    """Evaluate virtual screening performance against known actives.

    Parameters
    ----------
    ef_percentiles:
        Percentage cutpoints for EF calculation.
    bedroc_alpha:
        Decay parameter for BEDROC (default 20.0).
    higher_is_better:
        Set to ``True`` for probability-like scores where higher = more
        active.  Default ``False`` matches the docking score convention
        (lower = better binding energy).
    """

    def __init__(
        self,
        ef_percentiles: tuple[float, ...] = EF_PERCENTILES,
        bedroc_alpha: float = 20.0,
        higher_is_better: bool = False,
    ) -> None:
        self.ef_percentiles = ef_percentiles
        self.bedroc_alpha = bedroc_alpha
        self.higher_is_better = higher_is_better

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        scores: Any,
        labels: Any,
        sample_name: str = "",
    ) -> VSMetricsResult:
        """Evaluate one scoring vector.

        Parameters
        ----------
        scores:
            Array-like of docking scores (n_compounds,).
        labels:
            Binary activity labels (1=active, 0=inactive).
        sample_name:
            Identifier for this run.

        Returns
        -------
        VSMetricsResult
        """
        from sklearn.metrics import roc_auc_score, roc_curve  # noqa: PLC0415

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)

        # Normalise direction: always ascending = lower rank = better
        if self.higher_is_better:
            scores_for_rank = -scores
        else:
            scores_for_rank = scores

        n_total = len(scores)
        n_actives = int(labels.sum())

        ef_dict: dict[float, float] = {}
        ef_max_dict: dict[float, float] = {}
        for pct in self.ef_percentiles:
            ef, ef_max = _enrichment_factor(scores_for_rank, labels, pct)
            ef_dict[pct] = ef
            ef_max_dict[pct] = ef_max

        # ROC-AUC: higher score = predicted active, so negate if lower-is-better
        predict_scores = -scores_for_rank
        try:
            roc_auc = float(roc_auc_score(labels, predict_scores))
            fpr, tpr, _ = roc_curve(labels, predict_scores)
        except ValueError:
            roc_auc = 0.0
            fpr = tpr = np.array([0.0, 1.0])

        bedroc = _bedroc(scores_for_rank, labels, alpha=self.bedroc_alpha)

        return VSMetricsResult(
            sample_name=sample_name,
            n_total=n_total,
            n_actives=n_actives,
            ef=ef_dict,
            ef_max=ef_max_dict,
            roc_auc=roc_auc,
            bedroc=bedroc,
            fpr=fpr,
            tpr=tpr,
        )

    def run_multi(
        self,
        scores_dict: dict[str, Any],
        labels: Any,
    ) -> dict[str, VSMetricsResult]:
        """Evaluate multiple scoring vectors against the same labels.

        Parameters
        ----------
        scores_dict:
            ``{run_name: scores_array}`` mapping.
        labels:
            Shared binary activity labels.

        Returns
        -------
        dict[str, VSMetricsResult]
        """
        return {
            name: self.run(scores, labels, sample_name=name)
            for name, scores in scores_dict.items()
        }

    def run_from_df(
        self,
        df: pd.DataFrame,
        score_col: str = "docking_score",
        active_col: str = "is_active",
        group_col: str | None = None,
        sample_name: str = "",
    ) -> "VSMetricsResult | dict[str, VSMetricsResult]":
        """Evaluate from a DataFrame, optionally grouping by a column.

        Parameters
        ----------
        df:
            DataFrame with at least *score_col* and *active_col*.
        score_col:
            Name of the docking score column.
        active_col:
            Name of the binary activity column.
        group_col:
            When provided, compute metrics per group and return a dict.
        sample_name:
            Identifier for ungrouped runs.

        Returns
        -------
        VSMetricsResult or dict[str, VSMetricsResult]
        """
        if group_col is not None:
            return {
                name: self.run(
                    grp[score_col].values, grp[active_col].values, sample_name=name
                )
                for name, grp in df.groupby(group_col)
            }
        return self.run(
            df[score_col].values, df[active_col].values, sample_name=sample_name
        )
