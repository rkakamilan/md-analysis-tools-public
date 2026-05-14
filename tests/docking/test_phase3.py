"""Tests for Phase 3 docking modules: selection, vs_metrics, validation, strain."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class TestPareto:
    from mdatools.docking.selection.pareto import (
        compute_pareto_front,
        compute_pareto_rank,
    )

    def test_pareto_front_simple(self):
        from mdatools.docking.selection.pareto import compute_pareto_front

        # A=(1,1), B=(2,0), C=(0,2), D=(2,2) → D dominates A, B, C
        matrix = np.array([[1, 1], [2, 0], [0, 2], [2, 2]])
        front = compute_pareto_front(matrix, ["max", "max"])
        assert 3 in front  # D is on the front

    def test_pareto_rank_all_unique(self):
        from mdatools.docking.selection.pareto import compute_pareto_rank

        # Strictly dominated chain: D>C>B>A in all dimensions
        matrix = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        ranks = compute_pareto_rank(matrix, ["max", "max"])
        assert ranks[3] == 1  # highest row is rank 1
        assert ranks[0] == 4  # lowest is rank 4

    def test_pareto_rank_min_direction(self):
        from mdatools.docking.selection.pareto import compute_pareto_rank

        # Lower score = better; one compound clearly best
        matrix = np.array([[-9.0, 0.9], [-7.0, 0.5], [-8.0, 0.7]])
        ranks = compute_pareto_rank(matrix, ["min", "max"])
        assert 1 in ranks

    def test_pareto_rank_identical_rows(self):
        from mdatools.docking.selection.pareto import compute_pareto_rank

        matrix = np.array([[1, 1], [1, 1], [1, 1]])
        ranks = compute_pareto_rank(matrix, ["max", "max"])
        assert (ranks == 1).all()

    def test_pareto_front_returns_list_of_ints(self):
        from mdatools.docking.selection.pareto import compute_pareto_front

        matrix = np.array([[1, 2], [2, 1]])
        front = compute_pareto_front(matrix)
        assert isinstance(front, list)
        assert all(isinstance(i, int) for i in front)


# ---------------------------------------------------------------------------
# VS Metrics
# ---------------------------------------------------------------------------


class TestVSMetrics:
    @pytest.fixture
    def evaluator(self):
        pytest.importorskip("sklearn", reason="scikit-learn required")
        from mdatools.docking.analysis.vs_metrics import VirtualScreeningEvaluator

        return VirtualScreeningEvaluator()

    @pytest.fixture
    def perfect_scores(self):
        # 5 actives ranked first, 5 decoys last
        scores = np.array([-9.0, -8.5, -8.0, -7.5, -7.0, -6.0, -5.5, -5.0, -4.5, -4.0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        return scores, labels

    @pytest.fixture
    def random_scores(self, rng=None):
        rng = np.random.default_rng(42)
        n = 100
        scores = rng.uniform(-10, -4, size=n)
        labels = (scores < np.percentile(scores, 20)).astype(int)
        return scores, labels

    def test_returns_vs_metrics_result(self, evaluator, perfect_scores):
        from mdatools.docking.analysis.vs_metrics import VSMetricsResult

        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        assert isinstance(result, VSMetricsResult)

    def test_perfect_roc_auc(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        assert result.roc_auc == pytest.approx(1.0, abs=1e-9)

    def test_roc_auc_in_unit_interval(self, evaluator, random_scores):
        scores, labels = random_scores
        result = evaluator.run(scores, labels)
        assert 0.0 <= result.roc_auc <= 1.0

    def test_ef_keys(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        from mdatools.docking.analysis.vs_metrics import EF_PERCENTILES

        assert set(result.ef.keys()) == set(EF_PERCENTILES)

    def test_ef_perfect_enrichment(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        # Perfect enrichment: all actives in top 50% → EF@50% = 2.0
        assert result.ef[20.0] > 1.0

    def test_bedroc_high_for_perfect_ranking(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        # Perfect discrimination should give BEDROC close to 1.0
        assert result.bedroc > 0.8

    def test_run_multi(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        results = evaluator.run_multi({"model_A": scores, "model_B": -scores}, labels)
        assert "model_A" in results
        assert "model_B" in results

    def test_run_from_df(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        df = pd.DataFrame({"docking_score": scores, "is_active": labels})
        result = evaluator.run_from_df(df)
        assert result.roc_auc == pytest.approx(1.0, abs=1e-9)

    def test_higher_is_better_flag(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        ev_hi = type(evaluator)(higher_is_better=True)
        result = ev_hi.run(-scores, labels)  # flip sign
        assert result.roc_auc == pytest.approx(1.0, abs=1e-9)

    def test_sample_name_stored(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels, sample_name="test_run")
        assert result.sample_name == "test_run"

    def test_ef_norm_in_unit_interval(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        for pct, ef_n in result.ef_norm.items():
            assert 0.0 <= ef_n <= 1.0 + 1e-9, f"EF_norm at {pct}% = {ef_n}"

    def test_n_total_and_n_actives(self, evaluator, perfect_scores):
        scores, labels = perfect_scores
        result = evaluator.run(scores, labels)
        assert result.n_total == len(scores)
        assert result.n_actives == int(labels.sum())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validation_result_dataclass(self):
        from mdatools.docking.analysis.validation import ValidationResult

        vr = ValidationResult(best_rmsd=1.5, success=True, best_pose_idx=0)
        assert vr.success is True
        assert vr.threshold == 2.0  # default

    def test_box_validation_result_dataclass(self):
        from mdatools.docking.analysis.validation import BoxValidationResult

        bvr = BoxValidationResult(in_box=True, in_extended=True, distance=-1.0)
        assert bvr.in_box is True

    def test_validate_pose_in_box_inside(self):
        from mdatools.docking.analysis.validation import validate_pose_in_box

        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        center = (0.0, 0.0, 0.0)
        size = (20.0, 20.0, 20.0)
        result = validate_pose_in_box(mol, center, size)
        assert result.in_box is True

    def test_validate_pose_in_box_outside(self):
        from mdatools.docking.analysis.validation import validate_pose_in_box

        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        # Box centred far away
        center = (100.0, 100.0, 100.0)
        size = (2.0, 2.0, 2.0)
        result = validate_pose_in_box(mol, center, size)
        assert result.in_box is False

    def test_validate_redocking_empty_poses(self):
        from mdatools.docking.analysis.validation import validate_redocking

        pytest.importorskip("rdkit")
        from rdkit import Chem

        ref = Chem.MolFromSmiles("C")
        result = validate_redocking([], ref)
        assert result.best_pose_idx == -1


# ---------------------------------------------------------------------------
# Strain (import + basic tests)
# ---------------------------------------------------------------------------


class TestStrain:
    def test_strain_thresholds_dataclass(self):
        from mdatools.docking.analysis.strain import StrainThresholds

        st = StrainThresholds(warn=10.0, reject=20.0)
        assert st.protocol == "standard"

    def test_recommend_strain_thresholds_small_mol(self):
        pytest.importorskip("rdkit")
        from mdatools.docking.analysis.strain import recommend_strain_thresholds
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid — small
        thresholds = recommend_strain_thresholds(mol)
        assert thresholds.warn == 10.0

    def test_recommend_strain_thresholds_relaxed_for_large(self):
        pytest.importorskip("rdkit")
        from mdatools.docking.analysis.strain import recommend_strain_thresholds
        from rdkit import Chem

        # Cyclosporin A — large flexible macrolide
        smi = "CCC1NC(=O)C(CC(C)C)N(C)C(=O)C(C(CC)C)OC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C1C"
        mol = Chem.MolFromSmiles(smi)
        thresholds = recommend_strain_thresholds(mol, protocol="h_relaxed")
        assert thresholds.warn >= 50.0

    def test_add_strain_to_df_adds_column(self):
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from mdatools.docking.analysis.strain import add_strain_to_df

        mol = Chem.MolFromSmiles("CC(=O)O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        df = pd.DataFrame({"name": ["acetic_acid"]})
        result = add_strain_to_df(df, [mol])
        assert "strain_kcal" in result.columns


# ---------------------------------------------------------------------------
# PoseFilter classes
# ---------------------------------------------------------------------------


class TestPoseFilters:
    @pytest.fixture
    def sample_df(self):
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "docking_score": [-9.0, -7.0, -8.5, -6.0],
                "cluster": [0, 0, 1, 1],
                "GLN30_HBAcceptor": [1, 0, 1, 0],
                "ARG38_HBDonor": [1, 0, 0, 0],
                "mw": [350.0, 450.0, 380.0, 520.0],
                "lipinski_pass": [True, True, True, False],
                "ec_score": [0.35, 0.25, 0.30, 0.40],
            }
        )
        return df.set_index("name")

    def test_score_filter(self, sample_df):
        from mdatools.docking.selection.filters import ScoreFilter

        f = ScoreFilter(threshold=-8.0)
        mask = f.apply(sample_df)
        assert mask["A"] is np.bool_(True)
        assert mask["B"] is np.bool_(False)

    def test_interaction_filter_and(self, sample_df):
        from mdatools.docking.selection.filters import InteractionFilter

        f = InteractionFilter(residues=["GLN30", "ARG38"], logic="AND")
        mask = f.apply(sample_df)
        # Only A has both interactions
        assert mask["A"] is np.bool_(True)
        assert mask["B"] is np.bool_(False)

    def test_interaction_filter_or(self, sample_df):
        from mdatools.docking.selection.filters import InteractionFilter

        f = InteractionFilter(residues=["GLN30", "ARG38"], logic="OR")
        mask = f.apply(sample_df)
        # A and C have GLN30
        assert mask["A"] is np.bool_(True)
        assert mask["C"] is np.bool_(True)
        assert mask["D"] is np.bool_(False)

    def test_cluster_representative_filter(self, sample_df):
        from mdatools.docking.selection.filters import ClusterRepresentativeFilter

        f = ClusterRepresentativeFilter()
        mask = f.apply(sample_df)
        assert mask["A"] is np.bool_(True)  # cluster 0 best: -9.0
        assert mask["B"] is np.bool_(False)
        assert mask["C"] is np.bool_(True)  # cluster 1 best: -8.5

    def test_property_filter_mw(self, sample_df):
        from mdatools.docking.selection.filters import PropertyFilter

        f = PropertyFilter(max_mw=400.0)
        mask = f.apply(sample_df)
        assert mask["A"] is np.bool_(True)
        assert mask["D"] is np.bool_(False)

    def test_property_filter_ro5(self, sample_df):
        from mdatools.docking.selection.filters import PropertyFilter

        f = PropertyFilter(require_ro5=True)
        mask = f.apply(sample_df)
        assert mask["D"] is np.bool_(False)

    def test_ec_filter(self, sample_df):
        from mdatools.docking.selection.filters import ECFilter

        f = ECFilter(min_ec=0.30)
        mask = f.apply(sample_df)
        assert mask["B"] is np.bool_(False)
        assert mask["A"] is np.bool_(True)

    def test_pareto_filter(self, sample_df):
        pytest.importorskip("sklearn")
        from mdatools.docking.selection.filters import ParetoFilter

        f = ParetoFilter(
            objective_cols=["docking_score", "mw"],
            directions=["min", "min"],
            max_rank=1,
        )
        mask = f.apply(sample_df)
        assert isinstance(mask, pd.Series)

    def test_strain_energy_filter_nan_pass(self):
        from mdatools.docking.selection.filters import StrainEnergyFilter

        df = pd.DataFrame({"strain_kcal": [5.0, np.nan, 25.0]})
        f = StrainEnergyFilter(max_strain=20.0, pass_nan=True)
        mask = f.apply(df)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(True)  # NaN passes
        assert mask[2] is np.bool_(False)

    def test_strain_energy_filter_nan_reject(self):
        from mdatools.docking.selection.filters import StrainEnergyFilter

        df = pd.DataFrame({"strain_kcal": [np.nan]})
        f = StrainEnergyFilter(pass_nan=False)
        mask = f.apply(df)
        assert mask[0] is np.bool_(False)


class TestApplyFilters:
    @pytest.fixture
    def df_mols(self):
        df = pd.DataFrame(
            {
                "docking_score": [-9.0, -7.0, -8.5, -6.0, -5.0],
                "cluster": [0, 0, 1, 1, 2],
            }
        )
        mols = [MagicMock() for _ in range(5)]
        return df, mols

    def test_apply_filters_and(self, df_mols):
        from mdatools.docking.selection.filters import ScoreFilter, apply_filters

        df, mols = df_mols
        filt_df, filt_mols = apply_filters(
            df, mols, [ScoreFilter(threshold=-8.0)], logic="AND"
        )
        assert len(filt_df) == 2  # -9.0 and -8.5
        assert len(filt_mols) == 2

    def test_apply_filters_no_filters(self, df_mols):
        from mdatools.docking.selection.filters import apply_filters

        df, mols = df_mols
        filt_df, filt_mols = apply_filters(df, mols, [])
        assert len(filt_df) == len(df)

    def test_apply_filters_preserves_alignment(self, df_mols):
        from mdatools.docking.selection.filters import ScoreFilter, apply_filters

        df, mols = df_mols
        filt_df, filt_mols = apply_filters(df, mols, [ScoreFilter(threshold=-8.0)])
        # Both returned objects have same length
        assert len(filt_df) == len(filt_mols)

    def test_apply_filters_or_logic(self, df_mols):
        from mdatools.docking.selection.filters import ScoreFilter, apply_filters

        df, mols = df_mols
        # Scores: -9, -7, -8.5, -6, -5
        # ScoreFilter(-8.5): -9 and -8.5 pass (2 rows)
        # ScoreFilter(-7.0): -9, -7, -8.5 pass (3 rows)
        # OR union: rows with -9, -7, -8.5 pass = 3 rows
        filt_df, _ = apply_filters(
            df,
            mols,
            [ScoreFilter(threshold=-8.5), ScoreFilter(threshold=-7.0)],
            logic="OR",
        )
        assert len(filt_df) == 3  # -9, -8.5, and -7 all pass OR


# ---------------------------------------------------------------------------
# Namespace imports (Phase 3 export check)
# ---------------------------------------------------------------------------


class TestPhase3Exports:
    def test_vs_metrics_from_docking(self):
        from mdatools.docking import (  # noqa: F401
            VirtualScreeningEvaluator,
            VSMetricsResult,
            EF_PERCENTILES,
        )

    def test_validation_from_docking(self):
        from mdatools.docking import (  # noqa: F401
            ValidationResult,
            validate_redocking,
            BoxValidationResult,
        )

    def test_strain_from_docking(self):
        from mdatools.docking import StrainThresholds, compute_strain_energy  # noqa: F401

    def test_filters_from_docking(self):
        from mdatools.docking import (  # noqa: F401
            PoseFilter,
            ScoreFilter,
            InteractionFilter,
            ClusterRepresentativeFilter,
            PropertyFilter,
            SubstructureFilter,
            StructuralAlertFilter,
            AggregatorFilter,
            StrainEnergyFilter,
            ParetoFilter,
            ECFilter,
            apply_filters,
        )

    def test_pareto_from_docking(self):
        from mdatools.docking import compute_pareto_rank, compute_pareto_front  # noqa: F401

    def test_selection_subpackage(self):
        from mdatools.docking.selection import (  # noqa: F401
            ScoreFilter,
            apply_filters,
            compute_pareto_rank,
        )
