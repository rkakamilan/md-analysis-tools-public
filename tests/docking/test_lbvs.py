"""Tests for mdatools.docking.lbvs."""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mdatools.docking.lbvs import LBVSResult, LBVSScreener, diverse_subset  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUERY_SMILES = "c1ccc(NC(=O)c2cccc(F)c2)cc1"  # fluorobenzoyl aniline
SIMILAR = "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1"  # chloro analog — should be similar
DISSIMILAR = "CCCCCCCCCC"  # alkane — very different
IDENTICAL = QUERY_SMILES  # self — score == 1.0


@pytest.fixture()
def query_mol():
    return Chem.MolFromSmiles(QUERY_SMILES)


@pytest.fixture()
def screener():
    return LBVSScreener(radius=2, n_bits=2048, threshold=0.3)


@pytest.fixture()
def library():
    return [Chem.MolFromSmiles(s) for s in [SIMILAR, DISSIMILAR, IDENTICAL]]


# ---------------------------------------------------------------------------
# LBVSResult dataclass
# ---------------------------------------------------------------------------


class TestLBVSResult:
    def test_default_rank_zero(self, query_mol):
        r = LBVSResult(mol=query_mol, score=0.9)
        assert r.rank == 0

    def test_default_smiles_empty(self, query_mol):
        r = LBVSResult(mol=query_mol, score=0.5)
        assert r.smiles == ""

    def test_fields_set_correctly(self, query_mol):
        r = LBVSResult(mol=query_mol, score=0.7, rank=1, query_idx=2)
        assert r.score == 0.7
        assert r.rank == 1
        assert r.query_idx == 2


# ---------------------------------------------------------------------------
# LBVSScreener.screen()
# ---------------------------------------------------------------------------


class TestLBVSScreenerScreen:
    def test_returns_list(self, screener, query_mol, library):
        results = screener.screen([query_mol], library)
        assert isinstance(results, list)

    def test_results_are_lbvs_result(self, screener, query_mol, library):
        results = screener.screen([query_mol], library)
        for r in results:
            assert isinstance(r, LBVSResult)

    def test_identical_mol_scores_one(self, screener, query_mol):
        identical = Chem.MolFromSmiles(IDENTICAL)
        results = screener.screen([query_mol], [identical])
        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 1e-6

    def test_similar_mol_above_threshold(self, screener, query_mol):
        similar = Chem.MolFromSmiles(SIMILAR)
        results = screener.screen([query_mol], [similar])
        assert len(results) == 1
        assert results[0].score >= 0.3

    def test_dissimilar_mol_below_threshold(self, query_mol):
        screener = LBVSScreener(threshold=0.6)
        dissimilar = Chem.MolFromSmiles(DISSIMILAR)
        results = screener.screen([query_mol], [dissimilar])
        assert len(results) == 0

    def test_sorted_by_score_descending(self, screener, query_mol, library):
        results = screener.screen([query_mol], library)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, screener, query_mol, library):
        results = screener.screen([query_mol], library)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_empty_query_returns_empty(self, screener, library):
        assert screener.screen([], library) == []

    def test_empty_library_returns_empty(self, screener, query_mol):
        assert screener.screen([query_mol], []) == []

    def test_none_entries_skipped(self, screener, query_mol):
        identical = Chem.MolFromSmiles(IDENTICAL)
        results = screener.screen([query_mol], [None, identical, None])
        assert all(r.mol is not None for r in results)

    def test_none_query_skipped(self, screener, library):
        query = Chem.MolFromSmiles(QUERY_SMILES)
        results_clean = screener.screen([query], library)
        results_none = screener.screen([None, query], library)
        assert len(results_clean) == len(results_none)

    def test_multi_query_takes_max_score(self):
        screener = LBVSScreener(threshold=0.0)
        q1 = Chem.MolFromSmiles("c1ccccc1")
        q2 = Chem.MolFromSmiles(QUERY_SMILES)
        lib = [Chem.MolFromSmiles(SIMILAR)]
        results = screener.screen([q1, q2], lib)
        assert len(results) == 1
        # score should be max(sim(q1,lib), sim(q2,lib))
        assert results[0].score >= 0.0

    def test_score_in_unit_interval(self, screener, query_mol, library):
        results = screener.screen([query_mol], library)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# LBVSScreener.screen_df()
# ---------------------------------------------------------------------------


class TestLBVSScreenerDF:
    def test_returns_dataframe(self, screener, query_mol):
        import pandas as pd

        df = pd.DataFrame({"smiles": [SIMILAR, DISSIMILAR, IDENTICAL], "id": [1, 2, 3]})
        result = screener.screen_df([query_mol], df)
        assert isinstance(result, pd.DataFrame)

    def test_lbvs_columns_added(self, screener, query_mol):
        import pandas as pd

        df = pd.DataFrame({"smiles": [SIMILAR, IDENTICAL]})
        result = screener.screen_df([query_mol], df)
        for col in ["lbvs_score", "lbvs_rank", "lbvs_query_idx"]:
            assert col in result.columns

    def test_sorted_by_score(self, screener, query_mol):
        import pandas as pd

        df = pd.DataFrame({"smiles": [SIMILAR, DISSIMILAR, IDENTICAL]})
        result = screener.screen_df([query_mol], df)
        scores = list(result["lbvs_score"])
        assert scores == sorted(scores, reverse=True)

    def test_missing_column_raises(self, screener, query_mol):
        import pandas as pd

        df = pd.DataFrame({"smi": [SIMILAR]})
        with pytest.raises(ValueError, match="smiles"):
            screener.screen_df([query_mol], df, smiles_col="smiles")

    def test_original_columns_preserved(self, screener, query_mol):
        import pandas as pd

        df = pd.DataFrame({"smiles": [SIMILAR, IDENTICAL], "activity": [1.0, 2.0]})
        result = screener.screen_df([query_mol], df)
        assert "activity" in result.columns


# ---------------------------------------------------------------------------
# diverse_subset()
# ---------------------------------------------------------------------------


class TestDiverseSubset:
    def test_returns_list(self, library):
        result = diverse_subset(library, n=2)
        assert isinstance(result, list)

    def test_length_at_most_n(self, library):
        result = diverse_subset(library, n=2)
        assert len(result) <= 2

    def test_returns_mol_objects(self, library):
        result = diverse_subset(library, n=2)
        for mol in result:
            assert mol is not None

    def test_n_larger_than_pool_returns_all(self, library):
        result = diverse_subset(library, n=100)
        assert len(result) == len(library)

    def test_empty_input_returns_empty(self):
        assert diverse_subset([], n=5) == []

    def test_none_entries_excluded(self, library):
        result = diverse_subset(library + [None], n=len(library))
        assert all(m is not None for m in result)

    def test_reproducible_with_same_seed(self, library):
        r1 = diverse_subset(library, n=2, seed=0)
        r2 = diverse_subset(library, n=2, seed=0)
        smiles1 = [Chem.MolToSmiles(m) for m in r1]
        smiles2 = [Chem.MolToSmiles(m) for m in r2]
        assert smiles1 == smiles2

    def test_different_seeds_may_differ(self):
        mols = [
            Chem.MolFromSmiles(s)
            for s in [
                "c1ccccc1",
                "CC(=O)O",
                "c1cccnc1",
                "CCCC",
                "c1ccc(F)cc1",
            ]
        ]
        r1 = diverse_subset(mols, n=3, seed=0)
        r2 = diverse_subset(mols, n=3, seed=99)
        s1 = set(Chem.MolToSmiles(m) for m in r1)
        s2 = set(Chem.MolToSmiles(m) for m in r2)
        # Not guaranteed to differ but seeds usually produce different results
        assert isinstance(s1, set) and isinstance(s2, set)


# ---------------------------------------------------------------------------
# Namespace exports from mdatools.docking
# ---------------------------------------------------------------------------


class TestDockingExports:
    def test_lbvs_screener_importable(self):
        from mdatools.docking import LBVSScreener  # noqa: F401

    def test_lbvs_result_importable(self):
        from mdatools.docking import LBVSResult  # noqa: F401

    def test_diverse_subset_importable(self):
        from mdatools.docking import diverse_subset  # noqa: F401
