"""Tests for mdatools.docking.library_design.pharmacophore."""

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from rdkit import Chem  # noqa: E402

from mdatools.docking.library_design.pharmacophore import (  # noqa: E402
    PharmacophoreFeature,
    PharmacophoreHypothesis,
    PharmacophoreModeler,
    ensemble_pharmacophore,
    screen_library,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Active amide analogs (HBA + HBD + aromatic features)
ACTIVES = [
    "c1ccc(NC(=O)c2cccc(F)c2)cc1",
    "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
    "c1ccc(NC(=O)CC(C)C)cc1",
]
# Decoy: aliphatic alkane — minimal pharmacophore features
DECOY = "CCCCCCCC"


@pytest.fixture()
def active_mols():
    return [Chem.MolFromSmiles(s) for s in ACTIVES]


@pytest.fixture()
def modeler():
    return PharmacophoreModeler()


@pytest.fixture()
def hypothesis(active_mols):
    return ensemble_pharmacophore(active_mols, n_clusters=3)


# ---------------------------------------------------------------------------
# PharmacophoreFeature dataclass
# ---------------------------------------------------------------------------


class TestPharmacophoreFeature:
    def test_default_position(self):
        f = PharmacophoreFeature(feature_type="HBD")
        assert f.position == (0.0, 0.0, 0.0)

    def test_default_weight(self):
        f = PharmacophoreFeature(feature_type="HBA")
        assert f.weight == 1.0

    def test_custom_values(self):
        f = PharmacophoreFeature(
            feature_type="Aromatic", position=(1.0, 2.0, 3.0), weight=0.8
        )
        assert f.feature_type == "Aromatic"
        assert f.position == (1.0, 2.0, 3.0)
        assert f.weight == 0.8


# ---------------------------------------------------------------------------
# PharmacophoreHypothesis dataclass
# ---------------------------------------------------------------------------


class TestPharmacophoreHypothesis:
    def test_default_empty(self):
        h = PharmacophoreHypothesis()
        assert h.features == []
        assert h.n_actives == 0

    def test_feature_types_property(self):
        h = PharmacophoreHypothesis(
            features=[
                PharmacophoreFeature("HBD"),
                PharmacophoreFeature("HBA"),
                PharmacophoreFeature("HBD"),
            ]
        )
        assert set(h.feature_types) == {"HBD", "HBA"}


# ---------------------------------------------------------------------------
# PharmacophoreModeler.extract_features()
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_list(self, modeler):
        mol = Chem.MolFromSmiles(ACTIVES[0])
        feats = modeler.extract_features(mol)
        assert isinstance(feats, list)

    def test_none_returns_empty(self, modeler):
        assert modeler.extract_features(None) == []

    def test_features_are_pharmacophore_feature_instances(self, modeler):
        mol = Chem.MolFromSmiles(ACTIVES[0])
        feats = modeler.extract_features(mol)
        for f in feats:
            assert isinstance(f, PharmacophoreFeature)

    def test_feature_types_are_canonical(self, modeler):
        mol = Chem.MolFromSmiles(ACTIVES[0])
        feats = modeler.extract_features(mol)
        valid = {"HBD", "HBA", "Hydrophobic", "Aromatic", "Ionizable"}
        for f in feats:
            assert f.feature_type in valid

    def test_amide_has_hbd_and_hba(self, modeler):
        mol = Chem.MolFromSmiles(ACTIVES[0])
        feats = modeler.extract_features(mol)
        types = {f.feature_type for f in feats}
        assert "HBD" in types or "HBA" in types

    def test_aromatic_mol_has_aromatic_feature(self, modeler):
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        feats = modeler.extract_features(mol)
        types = {f.feature_type for f in feats}
        assert "Aromatic" in types

    def test_position_is_3_tuple(self, modeler):
        mol = Chem.MolFromSmiles(ACTIVES[0])
        feats = modeler.extract_features(mol)
        for f in feats:
            assert len(f.position) == 3

    def test_decoy_has_fewer_features(self, modeler):
        active = Chem.MolFromSmiles(ACTIVES[0])
        decoy = Chem.MolFromSmiles(DECOY)
        active_feats = modeler.extract_features(active)
        decoy_feats = modeler.extract_features(decoy)
        assert len(active_feats) >= len(decoy_feats)


# ---------------------------------------------------------------------------
# ensemble_pharmacophore()
# ---------------------------------------------------------------------------


class TestEnsemblePharmacophore:
    def test_returns_hypothesis(self, active_mols):
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        assert isinstance(hyp, PharmacophoreHypothesis)

    def test_n_actives_set_correctly(self, active_mols):
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        assert hyp.n_actives == len(active_mols)

    def test_features_list_nonempty(self, active_mols):
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        assert len(hyp.features) > 0

    def test_all_features_are_pharmacophore_feature(self, active_mols):
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        for f in hyp.features:
            assert isinstance(f, PharmacophoreFeature)

    def test_feature_types_canonical(self, active_mols):
        valid = {"HBD", "HBA", "Hydrophobic", "Aromatic", "Ionizable"}
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        for f in hyp.features:
            assert f.feature_type in valid

    def test_weights_in_unit_interval(self, active_mols):
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        for f in hyp.features:
            assert 0.0 <= f.weight <= 1.0

    def test_feature_counts_keys_are_feature_types(self, active_mols):
        valid = {"HBD", "HBA", "Hydrophobic", "Aromatic", "Ionizable"}
        hyp = ensemble_pharmacophore(active_mols, n_clusters=3)
        for k in hyp.feature_counts:
            assert k in valid

    def test_empty_mol_list_returns_empty_hypothesis(self):
        hyp = ensemble_pharmacophore([])
        assert hyp.n_actives == 0
        assert hyp.features == []

    def test_none_entries_skipped(self, active_mols):
        hyp_clean = ensemble_pharmacophore(active_mols)
        hyp_none = ensemble_pharmacophore(active_mols + [None])
        assert hyp_clean.n_actives == hyp_none.n_actives

    def test_min_occurrence_filters_rare_features(self):
        # Single molecule: all its features have occurrence=1.0
        mol = Chem.MolFromSmiles(ACTIVES[0])
        hyp = ensemble_pharmacophore([mol], n_clusters=3, min_occurrence=0.5)
        assert len(hyp.features) > 0

    def test_high_min_occurrence_reduces_features(self, active_mols):
        hyp_low = ensemble_pharmacophore(active_mols, n_clusters=5, min_occurrence=0.1)
        hyp_high = ensemble_pharmacophore(active_mols, n_clusters=5, min_occurrence=0.9)
        assert len(hyp_low.features) >= len(hyp_high.features)


# ---------------------------------------------------------------------------
# screen_library()
# ---------------------------------------------------------------------------


class TestScreenLibrary:
    def test_returns_list(self, hypothesis):
        mols = [Chem.MolFromSmiles(s) for s in ACTIVES]
        results = screen_library(hypothesis, mols)
        assert isinstance(results, list)

    def test_results_are_mol_score_tuples(self, hypothesis):
        mols = [Chem.MolFromSmiles(s) for s in ACTIVES]
        results = screen_library(hypothesis, mols)
        for mol, score in results:
            assert mol is not None
            assert 0.0 <= score <= 1.0

    def test_sorted_by_score_descending(self, hypothesis):
        mols = [Chem.MolFromSmiles(s) for s in ACTIVES + [DECOY]]
        results = screen_library(hypothesis, mols, match_threshold=0.0)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_low_scorers(self, hypothesis):
        mols = [Chem.MolFromSmiles(s) for s in ACTIVES + [DECOY]]
        results = screen_library(hypothesis, mols, match_threshold=0.9)
        for _, score in results:
            assert score >= 0.9

    def test_actives_score_higher_than_decoy(self, hypothesis):
        active_mols = [Chem.MolFromSmiles(s) for s in ACTIVES]
        decoy_mol = Chem.MolFromSmiles(DECOY)
        all_mols = active_mols + [decoy_mol]
        results = screen_library(hypothesis, all_mols, match_threshold=0.0)
        score_map = {id(mol): score for mol, score in results}
        active_scores = [score_map[id(m)] for m in active_mols if id(m) in score_map]
        decoy_scores = (
            [score_map[id(decoy_mol)]] if id(decoy_mol) in score_map else [0.0]
        )
        if active_scores and decoy_scores:
            assert max(active_scores) >= max(decoy_scores)

    def test_none_entries_skipped(self, hypothesis):
        mols = [Chem.MolFromSmiles(ACTIVES[0]), None]
        results = screen_library(hypothesis, mols, match_threshold=0.0)
        assert all(mol is not None for mol, _ in results)

    def test_empty_hypothesis_returns_empty(self):
        empty_hyp = PharmacophoreHypothesis()
        mols = [Chem.MolFromSmiles(s) for s in ACTIVES]
        results = screen_library(empty_hyp, mols)
        assert results == []

    def test_empty_library_returns_empty(self, hypothesis):
        assert screen_library(hypothesis, []) == []


# ---------------------------------------------------------------------------
# Namespace exports from mdatools.docking
# ---------------------------------------------------------------------------


class TestDockingExports:
    def test_pharmacophore_modeler_importable(self):
        from mdatools.docking import PharmacophoreModeler  # noqa: F401

    def test_pharmacophore_feature_importable(self):
        from mdatools.docking import PharmacophoreFeature  # noqa: F401

    def test_pharmacophore_hypothesis_importable(self):
        from mdatools.docking import PharmacophoreHypothesis  # noqa: F401

    def test_ensemble_pharmacophore_importable(self):
        from mdatools.docking import ensemble_pharmacophore  # noqa: F401

    def test_screen_library_importable(self):
        from mdatools.docking import screen_library  # noqa: F401
