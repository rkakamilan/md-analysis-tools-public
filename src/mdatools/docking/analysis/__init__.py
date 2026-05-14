"""Docking result analysis: consensus scoring, SAR, pose quality, VS metrics."""

from .consensus import compute_consensus_score, filter_by_consensus
from .posebusters import PoseBustersResult, bust_rdkit_poses, filter_rdkit_poses
from .properties import MolecularProperties, calculate_properties
from .sar import MCSResult, add_scaffold_to_df, cluster_by_scaffold, find_mcs
from .strain import (
    StrainThresholds,
    add_strain_to_df,
    compute_strain_energy,
    compute_strain_energy_h_relaxed,
    recommend_strain_thresholds,
)
from .validation import (
    BoxValidationResult,
    ValidationResult,
    rmsd_to_reference,
    validate_pose_in_box,
    validate_redocking,
)
from .vs_metrics import (
    EF_PERCENTILES,
    VSMetricsResult,
    VirtualScreeningEvaluator,
)

__all__ = [
    # consensus
    "compute_consensus_score",
    "filter_by_consensus",
    # posebusters
    "PoseBustersResult",
    "bust_rdkit_poses",
    "filter_rdkit_poses",
    # properties
    "MolecularProperties",
    "calculate_properties",
    # sar
    "MCSResult",
    "add_scaffold_to_df",
    "cluster_by_scaffold",
    "find_mcs",
    # strain
    "StrainThresholds",
    "compute_strain_energy",
    "compute_strain_energy_h_relaxed",
    "recommend_strain_thresholds",
    "add_strain_to_df",
    # validation
    "ValidationResult",
    "BoxValidationResult",
    "rmsd_to_reference",
    "validate_redocking",
    "validate_pose_in_box",
    # vs_metrics
    "EF_PERCENTILES",
    "VSMetricsResult",
    "VirtualScreeningEvaluator",
]
