"""Multi-receptor consensus scoring for ensemble docking.

Aggregates docking scores from multiple receptor conformations (or
multiple docking engines) into a single consensus score per compound.

Ported from ``docking_analysis.analysis.consensus``.

Example::

    df = compute_consensus_score(df, ["score_6x1a", "score_7e14"], method="ecr")
    df = filter_by_consensus(df, ["score_6x1a", "score_7e14"], min_receptors=2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_consensus_score(
    df: pd.DataFrame,
    score_columns: list[str],
    method: str = "rank_mean",
    output_column: str = "consensus_score",
    ecr_sigma_fraction: float = 0.05,
) -> pd.DataFrame:
    """Compute a consensus score from multiple docking score columns.

    Parameters
    ----------
    df:
        DataFrame with one row per compound and multiple score columns.
    score_columns:
        Names of the docking score columns to aggregate.
    method:
        Aggregation method:

        - ``"mean"`` — simple average (assumes comparable scales).
        - ``"min"`` — worst (highest) score across receptors.
        - ``"rank_mean"`` — rank within each column (ascending; rank 1 = best
          score), then average ranks.  Scale-independent.
        - ``"ecr"`` — Exponential Consensus Ranking.  Higher values = better.
          ``score = Σ exp(-rank_i / σ)`` where
          ``σ = ecr_sigma_fraction * n_compounds``.

    output_column:
        Name of the output consensus score column.
    ecr_sigma_fraction:
        Fraction of ``n_compounds`` used as σ for ECR (default 0.05).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the *output_column* appended.

    Raises
    ------
    ValueError
        If *method* is not recognised or *score_columns* is empty.
    """
    if not score_columns:
        raise ValueError("score_columns must not be empty")
    valid = {"mean", "min", "rank_mean", "ecr"}
    if method not in valid:
        raise ValueError(f"Unknown method {method!r}. Valid: {sorted(valid)}")

    result = df.copy()
    scores = result[score_columns]

    if method == "mean":
        result[output_column] = scores.mean(axis=1)

    elif method == "min":
        # For Vina-like scores (lower = better), max → worst across receptors.
        result[output_column] = scores.max(axis=1)

    elif method == "rank_mean":
        ranks = scores.rank(ascending=True, method="average")
        result[output_column] = ranks.mean(axis=1)

    elif method == "ecr":
        n = len(result)
        sigma = max(ecr_sigma_fraction * n, 1.0)
        ranks = scores.rank(ascending=True, method="average")
        result[output_column] = np.exp(-ranks / sigma).sum(axis=1)

    return result


def filter_by_consensus(
    df: pd.DataFrame,
    score_columns: list[str],
    min_receptors: int = 2,
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """Keep compounds that pass a score threshold in at least N receptors.

    Parameters
    ----------
    df:
        DataFrame with docking score columns.
    score_columns:
        Names of the score columns to check.
    min_receptors:
        Minimum number of receptors where the compound must pass the
        threshold.
    score_threshold:
        Maximum score to be considered a "hit" (lower = better).
        If ``None``, uses the median of each column as the threshold.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    scores = df[score_columns]
    if score_threshold is not None:
        passes = scores <= score_threshold
    else:
        passes = scores.le(scores.median())
    n_pass = passes.sum(axis=1)
    return df[n_pass >= min_receptors].copy()


def filter_by_pose_consensus(
    poses_by_engine: dict[str, list],
    rmsd_threshold: float = 2.0,
) -> list[int]:
    """Find compounds where multiple engines predict similar poses.

    For each compound index, computes the minimum RMSD between poses from
    different engines. Returns indices where at least one pair of engines
    agrees (RMSD < *rmsd_threshold*).

    Parameters
    ----------
    poses_by_engine:
        ``{engine_name: [mol_per_compound]}``. All lists must have the
        same length (one Mol per compound per engine).
    rmsd_threshold:
        Maximum RMSD (Å) for two poses to be considered "in agreement".

    Returns
    -------
    list[int]
        Compound indices where at least two engines agree.
    """
    engine_names = list(poses_by_engine.keys())
    if len(engine_names) < 2:
        raise ValueError("Need at least 2 engines for pose consensus")

    n_compounds = len(poses_by_engine[engine_names[0]])
    for name in engine_names[1:]:
        if len(poses_by_engine[name]) != n_compounds:
            raise ValueError(
                f"Engine '{name}' has {len(poses_by_engine[name])} compounds, "
                f"expected {n_compounds}"
            )

    agreed: list[int] = []
    for idx in range(n_compounds):
        found = False
        for i in range(len(engine_names)):
            if found:
                break
            for j in range(i + 1, len(engine_names)):
                mol_a = poses_by_engine[engine_names[i]][idx]
                mol_b = poses_by_engine[engine_names[j]][idx]
                if mol_a is None or mol_b is None:
                    continue
                if mol_a.GetNumConformers() == 0 or mol_b.GetNumConformers() == 0:
                    continue
                pos_a = mol_a.GetConformer().GetPositions()
                pos_b = mol_b.GetConformer().GetPositions()
                if pos_a.shape != pos_b.shape:
                    continue
                rmsd = float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1))))
                if rmsd < rmsd_threshold:
                    found = True
                    break
        if found:
            agreed.append(idx)
    return agreed
