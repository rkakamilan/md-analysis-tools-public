"""Smoke tests for the mdatools.docking namespace exports."""

from __future__ import annotations

import pytest


class TestNamespaceImports:
    def test_docking_namespace_importable(self):
        import mdatools.docking  # noqa: F401

    def test_fingerprint_base(self):
        from mdatools.docking import FingerprintCalculator  # noqa: F401

    def test_prolif_calculator(self):
        from mdatools.docking import ProLIFCalculator  # noqa: F401

    def test_clustering_exports(self):
        from mdatools.docking import (  # noqa: F401
            ChemicalClusteringResult,
            cluster_by_butina,
            cluster_by_kmeans,
            compute_fp_matrix,
        )

    def test_properties_exports(self):
        from mdatools.docking import MolecularProperties, calculate_properties  # noqa: F401

    def test_posebusters_exports(self):
        from mdatools.docking import (  # noqa: F401
            PoseBustersResult,
            bust_rdkit_poses,
            filter_rdkit_poses,
        )

    def test_consensus_exports(self):
        from mdatools.docking import (  # noqa: F401
            compute_consensus_score,
            filter_by_consensus,
        )

    def test_sar_exports(self):
        from mdatools.docking import (  # noqa: F401
            MCSResult,
            add_scaffold_to_df,
            cluster_by_scaffold,
            find_mcs,
        )

    def test_top_level_mdatools_has_docking(self):
        import mdatools
        assert hasattr(mdatools, "docking")


class TestPropertiesReexport:
    """Phase 1 shared.properties is accessible via docking.analysis.properties."""

    def test_same_object_as_shared(self):
        from mdatools.docking.analysis.properties import calculate_properties as docking_cp
        from mdatools.shared.properties import calculate_properties as shared_cp
        assert docking_cp is shared_cp

    def test_molecular_properties_is_shared_class(self):
        from mdatools.docking.analysis.properties import MolecularProperties as dm
        from mdatools.shared.properties import MolecularProperties as sm
        assert dm is sm


class TestPosebustersReexport:
    """Phase 1 shared.posebusters is accessible via docking.analysis.posebusters."""

    def test_bust_rdkit_poses_is_shared(self):
        from mdatools.docking.analysis.posebusters import bust_rdkit_poses as d
        from mdatools.shared.posebusters import bust_rdkit_poses as s
        assert d is s
