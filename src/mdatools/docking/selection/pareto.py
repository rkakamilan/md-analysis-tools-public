"""Pareto front ranking for multi-objective compound selection.

Implements non-dominated sorting to rank compounds across multiple
scoring objectives (e.g. docking score + LE + QED).

Ported from ``docking_analysis.selection.pareto``.

Example::

    import numpy as np
    from mdatools.docking.selection.pareto import compute_pareto_rank

    # 3 compounds × 2 objectives (docking_score to minimise, QED to maximise)
    scores = np.array([[-8.0, 0.7], [-7.0, 0.9], [-9.0, 0.5]])
    directions = ["min", "max"]
    ranks = compute_pareto_rank(scores, directions)
    # rank 1 = Pareto-optimal, rank 2 = dominated by rank-1 only, …
"""

from __future__ import annotations


import numpy as np


def _normalize_directions(
    matrix: np.ndarray,
    directions: list[str],
) -> np.ndarray:
    """Flip signs for minimisation objectives so all objectives are
    'higher is better' before domination checks.

    Parameters
    ----------
    matrix:
        Array of shape ``(n, k)`` — n compounds × k objectives.
    directions:
        List of ``"min"`` or ``"max"`` for each column.

    Returns
    -------
    np.ndarray
        A copy with minimisation columns negated.
    """
    normed = matrix.astype(float).copy()
    for j, direction in enumerate(directions):
        if direction == "min":
            normed[:, j] = -normed[:, j]
    return normed


def _is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Return ``True`` if solution *b* dominates solution *a*.

    *b* dominates *a* when *b* is at least as good in all objectives
    and strictly better in at least one (higher = better convention).
    """
    return bool(np.all(b >= a) and np.any(b > a))


def compute_pareto_front(
    matrix: np.ndarray,
    directions: list[str] | None = None,
) -> list[int]:
    """Return indices of non-dominated compounds (Pareto rank 1).

    Parameters
    ----------
    matrix:
        Array of shape ``(n, k)`` — n compounds × k objectives.
    directions:
        ``"min"`` or ``"max"`` per column (default: all ``"max"``).

    Returns
    -------
    list[int]
        Row indices of Pareto-optimal compounds.
    """
    n, k = matrix.shape
    if directions is None:
        directions = ["max"] * k
    normed = _normalize_directions(matrix, directions)

    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _is_dominated(normed[i], normed[j]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def compute_pareto_rank(
    matrix: np.ndarray,
    directions: list[str] | None = None,
) -> np.ndarray:
    """Assign 1-based Pareto ranks to all compounds via iterative peeling.

    Rank 1 = Pareto-optimal (non-dominated by any other compound).
    Rank 2 = non-dominated once rank-1 is removed, etc.

    Parameters
    ----------
    matrix:
        Array of shape ``(n, k)``.
    directions:
        ``"min"`` / ``"max"`` per column.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n,)`` with Pareto ranks.
    """
    n, k = matrix.shape
    if directions is None:
        directions = ["max"] * k
    normed = _normalize_directions(matrix, directions)

    ranks = np.zeros(n, dtype=int)
    remaining = list(range(n))
    rank = 1
    while remaining:
        sub = normed[remaining]
        front_local = compute_pareto_front(sub, ["max"] * k)
        global_indices = [remaining[i] for i in front_local]
        for idx in global_indices:
            ranks[idx] = rank
        remaining = [idx for idx in remaining if idx not in global_indices]
        rank += 1

    return ranks
