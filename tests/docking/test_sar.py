"""Tests for mdatools.docking.analysis.sar."""

from __future__ import annotations

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit required")
from rdkit import Chem  # noqa: E402

from mdatools.docking.analysis.sar import (
    MCSResult,
    add_scaffold_to_df,
    cluster_by_scaffold,
    compute_pairwise_mcs_size,
    find_mcs,
    get_murcko_scaffold,
)

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

KINASE_SMILES = [
    "CCc1nn(C)c2ccc(nc12)C1CCN(CC1)C(=O)c1cncs1",  # sildenafil-like
    "Cc1cc2c(ncnc2s1)N",                            # aminothienopyrimidine
    "c1ccc(cc1)Nc2ncnc3[nH]ccc23",                  # anilinopurine
    "CC(C)c1nc2ccc(F)cc2c(=O)n1CC(=O)O",            # pyrimidinone
]

@pytest.fixture(scope="module")
def kinase_mols():
    return [Chem.MolFromSmiles(s) for s in KINASE_SMILES]


@pytest.fixture(scope="module")
def simple_mols():
    return [Chem.MolFromSmiles(s) for s in ["c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1"]]


# ---------------------------------------------------------------------------
# find_mcs
# ---------------------------------------------------------------------------


class TestFindMCS:
    def test_returns_mcs_result(self, simple_mols):
        result = find_mcs(simple_mols)
        assert isinstance(result, MCSResult)

    def test_mcs_smarts_non_empty_for_similar(self, simple_mols):
        # All three are substituted benzenes — should share a benzene core
        result = find_mcs(simple_mols)
        assert result.n_atoms >= 6  # at least benzene ring

    def test_mcs_mol_is_none_or_mol(self, simple_mols):
        result = find_mcs(simple_mols)
        assert result.mol is None or isinstance(result.mol, Chem.Mol)

    def test_fewer_than_2_mols_raises(self, simple_mols):
        with pytest.raises(ValueError, match="At least 2"):
            find_mcs(simple_mols[:1])

    def test_mcs_n_atoms_nonnegative(self, kinase_mols):
        result = find_mcs(kinase_mols)
        assert result.n_atoms >= 0

    def test_mcs_smarts_parseable(self, simple_mols):
        result = find_mcs(simple_mols)
        if result.smarts:
            mol = Chem.MolFromSmarts(result.smarts)
            assert mol is not None


# ---------------------------------------------------------------------------
# get_murcko_scaffold
# ---------------------------------------------------------------------------


class TestGetMurckoScaffold:
    def test_scaffold_is_mol(self, simple_mols):
        scaffold = get_murcko_scaffold(simple_mols[0])
        assert isinstance(scaffold, Chem.Mol)

    def test_phenol_scaffold_is_benzene(self):
        phenol = Chem.MolFromSmiles("c1ccc(O)cc1")
        scaffold = get_murcko_scaffold(phenol)
        smi = Chem.MolToSmiles(scaffold)
        assert "c1ccccc1" in smi

    def test_generic_scaffold_reduces_heteroatoms(self):
        pyridine = Chem.MolFromSmiles("c1ccncc1")
        generic = get_murcko_scaffold(pyridine, generic=True)
        # Generic scaffold should have no N
        assert "n" not in Chem.MolToSmiles(generic)


# ---------------------------------------------------------------------------
# cluster_by_scaffold
# ---------------------------------------------------------------------------


class TestClusterByScaffold:
    def test_returns_dict(self, simple_mols):
        result = cluster_by_scaffold(simple_mols)
        assert isinstance(result, dict)
        # All phenyl derivatives → same scaffold
        assert len(result) == 1

    def test_different_scaffolds_give_different_keys(self):
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),   # benzene
            Chem.MolFromSmiles("C1CCCC1"),    # cyclopentane
        ]
        result = cluster_by_scaffold(mols)
        assert len(result) == 2

    def test_all_indices_covered(self, simple_mols):
        result = cluster_by_scaffold(simple_mols)
        all_indices = [idx for idxs in result.values() for idx in idxs]
        assert sorted(all_indices) == list(range(len(simple_mols)))

    def test_sorted_by_size_descending(self, simple_mols):
        result = cluster_by_scaffold(simple_mols)
        sizes = [len(v) for v in result.values()]
        assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# add_scaffold_to_df
# ---------------------------------------------------------------------------


class TestAddScaffoldToDf:
    def test_column_added(self, simple_mols):
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        result = add_scaffold_to_df(df, simple_mols)
        assert "scaffold" in result.columns

    def test_custom_column_name(self, simple_mols):
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        result = add_scaffold_to_df(df, simple_mols, scaffold_col="murcko")
        assert "murcko" in result.columns

    def test_original_columns_preserved(self, simple_mols):
        df = pd.DataFrame({"name": ["a", "b", "c"], "val": [1, 2, 3]})
        result = add_scaffold_to_df(df, simple_mols)
        assert "name" in result.columns
        assert "val" in result.columns

    def test_scaffold_values_are_smiles_strings(self, simple_mols):
        df = pd.DataFrame({"n": range(len(simple_mols))})
        result = add_scaffold_to_df(df, simple_mols)
        for smi in result["scaffold"]:
            assert isinstance(smi, str)
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
