"""mdatools.docking — Docking analysis sub-namespace (Phase 2/3 merger).

Phase 2 (ported in PR #138):
    fingerprints: ProLIFCalculator
    clustering:   cluster_by_kmeans, cluster_by_butina
    analysis:     consensus, sar, properties (re-export), posebusters (re-export)

Phase 3 (ported in PR #143):
    selection:  PoseFilter classes + apply_filters + Pareto ranking
    analysis:   VirtualScreeningEvaluator (EF/ROC-AUC/BEDROC),
                ValidationResult / validate_redocking,
                StrainThresholds / compute_strain_energy

Phase 3 cont. (PR #145):
    preparation: ADMEProfiler, ADMEProfile, plot_adme_radar, plot_adme_distribution

Phase 3 cont. (PR #147):
    library_design: ScaffoldAnalyzer, RGroupDecomposer
"""

from .analysis.consensus import compute_consensus_score, filter_by_consensus
from .analysis.posebusters import (
    PoseBustersResult,
    bust_rdkit_poses,
    filter_rdkit_poses,
)
from .analysis.properties import MolecularProperties, calculate_properties
from .analysis.sar import MCSResult, add_scaffold_to_df, cluster_by_scaffold, find_mcs
from .analysis.strain import (
    StrainThresholds,
    add_strain_to_df,
    compute_strain_energy,
    compute_strain_energy_h_relaxed,
    recommend_strain_thresholds,
)
from .analysis.validation import (
    BoxValidationResult,
    ValidationResult,
    rmsd_to_reference,
    validate_pose_in_box,
    validate_redocking,
)
from .analysis.vs_metrics import (
    EF_PERCENTILES,
    VSMetricsResult,
    VirtualScreeningEvaluator,
)
from .clustering.chemical import (
    ChemicalClusteringResult,
    cluster_by_butina,
    cluster_by_kmeans,
    compute_fp_matrix,
)
from .fingerprints.base import FingerprintCalculator
from .fingerprints.prolif import ProLIFCalculator
from .selection.filters import (
    AggregatorFilter,
    ClusterRepresentativeFilter,
    ECFilter,
    InteractionFilter,
    ParetoFilter,
    PoseFilter,
    PropertyFilter,
    ScoreFilter,
    StrainEnergyFilter,
    StructuralAlertFilter,
    SubstructureFilter,
    apply_filters,
)
from .selection.pareto import compute_pareto_front, compute_pareto_rank
from .preparation.adme_profiler import (
    ADMEProfile,
    ADMEProfiler,
    plot_adme_distribution,
    plot_adme_radar,
)
from .library_design.scaffold_hopping import RGroupDecomposer, ScaffoldAnalyzer
from .library_design.pharmacophore import (
    PharmacophoreFeature,
    PharmacophoreHypothesis,
    PharmacophoreModeler,
    ensemble_pharmacophore,
    screen_library,
)
from .lbvs.screener import LBVSResult, LBVSScreener, diverse_subset

__all__ = [
    # fingerprints
    "FingerprintCalculator",
    "ProLIFCalculator",
    # clustering
    "ChemicalClusteringResult",
    "cluster_by_kmeans",
    "cluster_by_butina",
    "compute_fp_matrix",
    # analysis — properties / posebusters (Phase 1 shared re-exports)
    "MolecularProperties",
    "calculate_properties",
    "PoseBustersResult",
    "bust_rdkit_poses",
    "filter_rdkit_poses",
    # analysis — consensus
    "compute_consensus_score",
    "filter_by_consensus",
    # analysis — sar
    "MCSResult",
    "find_mcs",
    "cluster_by_scaffold",
    "add_scaffold_to_df",
    # analysis — strain (Phase 3)
    "StrainThresholds",
    "compute_strain_energy",
    "compute_strain_energy_h_relaxed",
    "recommend_strain_thresholds",
    "add_strain_to_df",
    # analysis — validation (Phase 3)
    "ValidationResult",
    "BoxValidationResult",
    "rmsd_to_reference",
    "validate_redocking",
    "validate_pose_in_box",
    # analysis — vs_metrics (Phase 3)
    "EF_PERCENTILES",
    "VSMetricsResult",
    "VirtualScreeningEvaluator",
    # selection — filters (Phase 3)
    "PoseFilter",
    "ScoreFilter",
    "InteractionFilter",
    "ClusterRepresentativeFilter",
    "PropertyFilter",
    "SubstructureFilter",
    "StructuralAlertFilter",
    "AggregatorFilter",
    "StrainEnergyFilter",
    "ParetoFilter",
    "ECFilter",
    "apply_filters",
    # selection — pareto (Phase 3)
    "compute_pareto_front",
    "compute_pareto_rank",
    # preparation — ADME (Phase 3 cont.)
    "ADMEProfile",
    "ADMEProfiler",
    "plot_adme_radar",
    "plot_adme_distribution",
    # library_design — scaffold hopping (Phase 3 cont.)
    "ScaffoldAnalyzer",
    "RGroupDecomposer",
    # library_design — pharmacophore (Phase 3 cont.)
    "PharmacophoreFeature",
    "PharmacophoreHypothesis",
    "PharmacophoreModeler",
    "ensemble_pharmacophore",
    "screen_library",
    # lbvs
    "LBVSResult",
    "LBVSScreener",
    "diverse_subset",
]
