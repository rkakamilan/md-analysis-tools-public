"""Tests for mdatools.docking.analysis.consensus."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdatools.docking.analysis.consensus import (
    compute_consensus_score,
    filter_by_consensus,
    filter_by_pose_consensus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def score_df():
    return pd.DataFrame({
        "name": ["A", "B", "C", "D", "E"],
        "score_rec1": [-9.0, -7.5, -6.0, -8.5, -5.0],
        "score_rec2": [-8.5, -8.0, -7.0, -7.5, -6.0],
        "score_rec3": [-7.0, -9.0, -6.5, -8.0, -5.5],
    })


# ---------------------------------------------------------------------------
# compute_consensus_score
# ---------------------------------------------------------------------------


class TestComputeConsensusScore:
    def test_returns_dataframe_with_consensus_col(self, score_df):
        result = compute_consensus_score(score_df, ["score_rec1", "score_rec2"])
        assert "consensus_score" in result.columns
        assert len(result) == len(score_df)

    def test_original_columns_preserved(self, score_df):
        result = compute_consensus_score(score_df, ["score_rec1", "score_rec2"])
        assert "name" in result.columns

    def test_method_mean(self, score_df):
        result = compute_consensus_score(score_df, ["score_rec1", "score_rec2"], method="mean")
        expected = (score_df["score_rec1"] + score_df["score_rec2"]) / 2
        np.testing.assert_allclose(result["consensus_score"].values, expected.values)

    def test_method_rank_mean_returns_ranks(self, score_df):
        cols = ["score_rec1", "score_rec2", "score_rec3"]
        result = compute_consensus_score(score_df, cols, method="rank_mean")
        # Rank mean should be between 1 and n (5 compounds)
        assert result["consensus_score"].min() >= 1.0
        assert result["consensus_score"].max() <= 5.0

    def test_method_ecr_higher_is_better(self, score_df):
        cols = ["score_rec1", "score_rec2", "score_rec3"]
        result = compute_consensus_score(score_df, cols, method="ecr")
        # Compound A has consistently good scores → should have high ECR
        scores = result.set_index("name")["consensus_score"]
        assert scores["A"] > scores["E"]  # A is better than E

    def test_method_min(self, score_df):
        result = compute_consensus_score(score_df, ["score_rec1", "score_rec2"], method="min")
        expected = score_df[["score_rec1", "score_rec2"]].max(axis=1)
        np.testing.assert_allclose(result["consensus_score"].values, expected.values)

    def test_invalid_method_raises(self, score_df):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_consensus_score(score_df, ["score_rec1"], method="bogus")

    def test_empty_columns_raises(self, score_df):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_consensus_score(score_df, [])

    def test_custom_output_column(self, score_df):
        result = compute_consensus_score(
            score_df, ["score_rec1", "score_rec2"], output_column="my_score"
        )
        assert "my_score" in result.columns

    def test_does_not_modify_input(self, score_df):
        orig = score_df.copy()
        compute_consensus_score(score_df, ["score_rec1", "score_rec2"])
        pd.testing.assert_frame_equal(score_df, orig)


# ---------------------------------------------------------------------------
# filter_by_consensus
# ---------------------------------------------------------------------------


class TestFilterByConsensus:
    def test_threshold_filters(self, score_df):
        result = filter_by_consensus(
            score_df, ["score_rec1", "score_rec2", "score_rec3"],
            min_receptors=3, score_threshold=-7.0
        )
        # Only rows where all 3 receptors have score <= -7.0
        for _, row in result.iterrows():
            passing = sum([
                row["score_rec1"] <= -7.0,
                row["score_rec2"] <= -7.0,
                row["score_rec3"] <= -7.0,
            ])
            assert passing >= 3

    def test_min_receptors_1_returns_all_or_most(self, score_df):
        result = filter_by_consensus(
            score_df, ["score_rec1", "score_rec2"],
            min_receptors=1, score_threshold=-5.5
        )
        assert len(result) == len(score_df)

    def test_min_receptors_n_returns_strictest(self, score_df):
        n_cols = 3
        result = filter_by_consensus(
            score_df, ["score_rec1", "score_rec2", "score_rec3"],
            min_receptors=n_cols
        )
        assert len(result) <= len(score_df)

    def test_does_not_modify_input(self, score_df):
        orig = score_df.copy()
        filter_by_consensus(score_df, ["score_rec1", "score_rec2"])
        pd.testing.assert_frame_equal(score_df, orig)


# ---------------------------------------------------------------------------
# filter_by_pose_consensus
# ---------------------------------------------------------------------------


class TestFilterByPoseConsensus:
    def _make_mol_with_conformer(self, coords):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for i, (x, y, z) in enumerate(coords):
                if i < mol.GetNumAtoms():
                    conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        return mol

    def test_raises_with_one_engine(self):
        with pytest.raises(ValueError, match="at least 2 engines"):
            filter_by_pose_consensus({"A": []})

    def test_raises_on_length_mismatch(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C")
        with pytest.raises(ValueError, match="expected"):
            filter_by_pose_consensus({
                "A": [mol, mol],
                "B": [mol],
            })
