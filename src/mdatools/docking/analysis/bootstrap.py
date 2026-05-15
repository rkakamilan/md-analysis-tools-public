"""Bootstrap confidence intervals for ML evaluation metrics.

Provides non-parametric confidence intervals for any scalar metric
via the percentile bootstrap method.  Supports both classification
(``evaluate``) and regression (``evaluate_regression``) targets.

Classification example::

    from sklearn.metrics import roc_auc_score
    from mdatools.docking.analysis.bootstrap import BootstrapEvaluator

    ev = BootstrapEvaluator(n_iterations=1000, confidence_level=0.95)
    result = ev.evaluate(scores, labels, metric_fn=roc_auc_score)
    print(f"AUC-ROC = {result.mean:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

Regression example (pIC50, log solubility, ...)::

    from sklearn.metrics import r2_score
    result = ev.evaluate_regression(y_pred, y_true, metric_fn=r2_score)
    print(f"R² = {result.mean:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from mdatools.docking.analysis.vs_metrics import (
    _bedroc,
    _enrichment_factor,
)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval for a single metric.

    Parameters
    ----------
    metric_name:
        Human-readable name of the evaluated metric.
    mean:
        Mean metric value across bootstrap iterations.
    ci_lower:
        Lower bound of the confidence interval.
    ci_upper:
        Upper bound of the confidence interval.
    std:
        Standard deviation across bootstrap iterations.
    n_iterations:
        Number of bootstrap iterations performed.
    samples:
        Per-iteration metric values (stored for downstream statistical tests).
    """

    metric_name: str
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_iterations: int
    samples: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.mean:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(n={self.n_iterations})"
        )


class BootstrapEvaluator:
    """Compute bootstrap confidence intervals for evaluation metrics.

    Uses the percentile bootstrap: resample with replacement, compute
    metric on each resample, then take the empirical quantiles as CI.

    Parameters
    ----------
    n_iterations:
        Number of bootstrap resamples.
    confidence_level:
        Confidence level for the interval (e.g., 0.95 for 95% CI).
    random_state:
        Random seed for reproducibility.

    Examples
    --------
    Custom metric::

        ev = BootstrapEvaluator(n_iterations=1000)
        result = ev.evaluate(scores, labels, metric_fn=roc_auc_score)

    All VS metrics at once::

        results = ev.evaluate_vs_metrics(scores, labels)
        for name, r in results.items():
            print(r)
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 0,
    ) -> None:
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_state = random_state

    def evaluate(
        self,
        scores: np.ndarray | list[float],
        labels: np.ndarray | list[int],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "metric",
        higher_is_better: bool = True,
    ) -> BootstrapResult:
        """Bootstrap a single scalar metric.

        Parameters
        ----------
        scores:
            Model output scores.  Sign is not flipped internally;
            pass the scores in the convention expected by *metric_fn*.
        labels:
            Binary activity labels (1 = positive, 0 = negative).
        metric_fn:
            Callable ``(labels, scores) -> float`` following the
            sklearn convention (labels first, scores second).
        metric_name:
            Label stored in the result.
        higher_is_better:
            Not used internally; stored for documentation purposes only.

        Returns
        -------
        BootstrapResult

        Raises
        ------
        ValueError
            If arrays are empty or lengths differ.
        """
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)

        if len(scores_arr) != len(labels_arr):
            raise ValueError(
                f"scores and labels length mismatch: "
                f"{len(scores_arr)} vs {len(labels_arr)}"
            )
        if len(scores_arr) == 0:
            raise ValueError("scores and labels must not be empty")

        rng = np.random.default_rng(self.random_state)
        n = len(scores_arr)
        sample_values: list[float] = []

        for _ in range(self.n_iterations):
            idx = rng.integers(0, n, size=n)
            s = scores_arr[idx]
            y = labels_arr[idx]
            if y.sum() == 0 or (y == 0).sum() == 0:
                continue
            try:
                val = float(metric_fn(y, s))
                sample_values.append(val)
            except Exception:
                continue

        if not sample_values:
            raise RuntimeError(
                "No valid bootstrap samples could be computed. "
                "Check that labels contain both positives and negatives."
            )

        samples = np.array(sample_values)
        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(samples, 100 * alpha / 2))
        ci_upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            metric_name=metric_name,
            mean=float(samples.mean()),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std=float(samples.std()),
            n_iterations=len(samples),
            samples=samples,
        )

    def evaluate_regression(
        self,
        y_pred: np.ndarray | list[float],
        y_true: np.ndarray | list[float],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "metric",
        higher_is_better: bool = True,
    ) -> BootstrapResult:
        """Bootstrap a single scalar regression metric (e.g. R², RMSE, MAE).

        Mirrors :meth:`evaluate` but is designed for **continuous-valued
        targets** (pIC50, log solubility, ...).  Two key differences:

        - Labels are stored as ``float`` (no ``int`` cast).
        - No binary-class balance filter is applied to resamples — every
          resample contributes a metric value (subject only to
          ``metric_fn`` raising an exception, which is silently skipped).

        Parameters
        ----------
        y_pred:
            Model predictions (continuous values).
        y_true:
            Ground-truth target values (continuous values, not class
            labels).
        metric_fn:
            Callable ``(y_true, y_pred) -> float`` following the
            sklearn convention (true values first).  Examples:
            ``sklearn.metrics.r2_score``,
            ``sklearn.metrics.mean_absolute_error``.
        metric_name:
            Label stored in the result.
        higher_is_better:
            Stored for documentation purposes only (R² is higher-is-better,
            RMSE / MAE are lower-is-better).

        Returns
        -------
        BootstrapResult

        Raises
        ------
        ValueError
            If arrays are empty or lengths differ.
        RuntimeError
            If every bootstrap iteration's *metric_fn* call raised an
            exception.  Indicates ``metric_fn`` is incompatible with the
            inputs — verify it accepts ``(y_true, y_pred)`` numpy arrays.
        """
        y_pred_arr = np.asarray(y_pred, dtype=float)
        y_true_arr = np.asarray(y_true, dtype=float)

        if len(y_pred_arr) != len(y_true_arr):
            raise ValueError(
                f"y_pred and y_true length mismatch: "
                f"{len(y_pred_arr)} vs {len(y_true_arr)}"
            )
        if len(y_pred_arr) == 0:
            raise ValueError("y_pred and y_true must not be empty")

        rng = np.random.default_rng(self.random_state)
        n = len(y_pred_arr)
        sample_values: list[float] = []

        for _ in range(self.n_iterations):
            idx = rng.integers(0, n, size=n)
            try:
                val = float(metric_fn(y_true_arr[idx], y_pred_arr[idx]))
            except Exception:
                continue
            sample_values.append(val)

        if not sample_values:
            raise RuntimeError(
                "No valid bootstrap samples could be computed. "
                "Verify that metric_fn accepts (y_true, y_pred) numpy "
                "arrays and is appropriate for these inputs."
            )

        samples = np.array(sample_values)
        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(samples, 100 * alpha / 2))
        ci_upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            metric_name=metric_name,
            mean=float(samples.mean()),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std=float(samples.std()),
            n_iterations=len(samples),
            samples=samples,
        )

    def evaluate_vs_metrics(
        self,
        scores: np.ndarray | list[float],
        labels: np.ndarray | list[int],
        higher_is_better: bool = False,
    ) -> dict[str, BootstrapResult]:
        """Bootstrap all standard VS metrics: EF@1/5%, BEDROC, AUC-ROC.

        Parameters
        ----------
        scores:
            Docking scores (lower = better by default; negate internally
            when ``higher_is_better=False``).
        labels:
            Binary activity labels.
        higher_is_better:
            Set ``True`` for probability-like outputs.

        Returns
        -------
        dict[str, BootstrapResult]
            Keys: ``"ef_1pct"``, ``"ef_5pct"``, ``"bedroc"``, ``"roc_auc"``.
        """
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        ranking = scores_arr if higher_is_better else -scores_arr

        def ef_at(pct: float) -> Callable:
            def _fn(y: np.ndarray, s: np.ndarray) -> float:
                return _enrichment_factor(s, y, pct)[0]
            return _fn

        def bedroc_fn(y: np.ndarray, s: np.ndarray) -> float:
            return _bedroc(s, y)

        def auc_fn(y: np.ndarray, s: np.ndarray) -> float:
            return float(roc_auc_score(y, s))

        metrics: dict[str, tuple[Callable, str]] = {
            "ef_1pct":  (ef_at(1.0),  "EF@1%"),
            "ef_5pct":  (ef_at(5.0),  "EF@5%"),
            "bedroc":   (bedroc_fn,    "BEDROC"),
            "roc_auc":  (auc_fn,       "AUC-ROC"),
        }

        return {
            key: self.evaluate(
                ranking, labels_arr,
                metric_fn=fn,
                metric_name=name,
                higher_is_better=True,
            )
            for key, (fn, name) in metrics.items()
        }
