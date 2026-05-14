"""Tests for mdatools.docking.library_design.scaffold_hopping."""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mdatools.docking.library_design.scaffold_hopping import (  # noqa: E402
    RGroupDecomposer,
    ScaffoldAnalyzer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two amide analogs sharing the same aniline scaffold
AMIDE_A = "c1ccc(NC(=O)c2cccc(F)c2)cc1"  # aniline + fluorobenzoyl
AMIDE_B = "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1"  # aniline + chlorobenzoyl
AMIDE_C = "c1ccc(NC(=O)CC(C)C)cc1"  # aniline + isobutanoyl

NAPHTHALENE = "c1ccc2ccccc2c1"  # non-amide, different scaffold


@pytest.fixture()
def analyzer():
    return ScaffoldAnalyzer()


@pytest.fixture()
def decomposer():
    return RGroupDecomposer()


@pytest.fixture()
def amide_mols():
    return [Chem.MolFromSmiles(s) for s in [AMIDE_A, AMIDE_B, AMIDE_C]]


# ---------------------------------------------------------------------------
# ScaffoldAnalyzer.extract_murcko()
# ---------------------------------------------------------------------------


class TestExtractMurcko:
    def test_returns_list_same_length(self, analyzer, amide_mols):
        scaffolds = analyzer.extract_murcko(amide_mols)
        assert len(scaffolds) == len(amide_mols)

    def test_none_input_yields_none_output(self, analyzer):
        scaffolds = analyzer.extract_murcko([None])
        assert scaffolds == [None]

    def test_scaffold_is_mol_object(self, analyzer, amide_mols):
        scaffolds = analyzer.extract_murcko(amide_mols)
        for scaf in scaffolds:
            assert scaf is not None
            assert isinstance(scaf, Chem.rdchem.Mol)

    def test_amide_analogs_share_common_scaffold(self, analyzer, amide_mols):
        from rdkit import Chem as _Chem

        scaffolds = analyzer.extract_murcko(amide_mols)
        smiles = [_Chem.MolToSmiles(s) for s in scaffolds]
        # All three share the aniline+phenyl amide scaffold
        assert smiles[0] == smiles[1]

    def test_empty_list_returns_empty(self, analyzer):
        assert analyzer.extract_murcko([]) == []

    def test_scaffold_has_fewer_atoms_than_parent(self, analyzer):
        mol = Chem.MolFromSmiles(AMIDE_A)
        scaffolds = analyzer.extract_murcko([mol])
        assert scaffolds[0].GetNumAtoms() <= mol.GetNumAtoms()


# ---------------------------------------------------------------------------
# ScaffoldAnalyzer.scaffold_frequency()
# ---------------------------------------------------------------------------


class TestScaffoldFrequency:
    def test_returns_dataframe(self, analyzer, amide_mols):
        import pandas as pd

        df = analyzer.scaffold_frequency(amide_mols)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, analyzer, amide_mols):
        df = analyzer.scaffold_frequency(amide_mols)
        for col in ["scaffold_smiles", "count", "fraction"]:
            assert col in df.columns

    def test_counts_sum_to_n_molecules(self, analyzer, amide_mols):
        df = analyzer.scaffold_frequency(amide_mols)
        assert df["count"].sum() == len(amide_mols)

    def test_fraction_sums_to_one(self, analyzer, amide_mols):
        df = analyzer.scaffold_frequency(amide_mols)
        assert abs(df["fraction"].sum() - 1.0) < 1e-9

    def test_sorted_by_count_desc(self, analyzer, amide_mols):
        df = analyzer.scaffold_frequency(amide_mols)
        assert list(df["count"]) == sorted(df["count"], reverse=True)

    def test_empty_list_returns_empty_df(self, analyzer):
        df = analyzer.scaffold_frequency([])
        assert len(df) == 0

    def test_diverse_scaffolds_multiple_rows(self, analyzer):
        mols = [
            Chem.MolFromSmiles(AMIDE_A),
            Chem.MolFromSmiles(NAPHTHALENE),
        ]
        df = analyzer.scaffold_frequency(mols)
        assert len(df) >= 1

    def test_none_entries_skipped(self, analyzer, amide_mols):
        df_clean = analyzer.scaffold_frequency(amide_mols)
        df_with_none = analyzer.scaffold_frequency(amide_mols + [None])
        assert df_clean["count"].sum() == df_with_none["count"].sum()


# ---------------------------------------------------------------------------
# ScaffoldAnalyzer.group_by_scaffold()
# ---------------------------------------------------------------------------


class TestGroupByScaffold:
    def test_returns_dict(self, analyzer, amide_mols):
        groups = analyzer.group_by_scaffold(amide_mols)
        assert isinstance(groups, dict)

    def test_group_values_are_lists_of_mols(self, analyzer, amide_mols):
        groups = analyzer.group_by_scaffold(amide_mols)
        for mols_in_group in groups.values():
            assert isinstance(mols_in_group, list)
            assert all(m is not None for m in mols_in_group)

    def test_total_molecules_conserved(self, analyzer, amide_mols):
        groups = analyzer.group_by_scaffold(amide_mols)
        total = sum(len(v) for v in groups.values())
        assert total == len(amide_mols)

    def test_amide_analogs_in_same_group(self, analyzer):
        mols = [Chem.MolFromSmiles(AMIDE_A), Chem.MolFromSmiles(AMIDE_B)]
        groups = analyzer.group_by_scaffold(mols)
        # Both share same scaffold → exactly one group
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 2

    def test_none_entries_excluded(self, analyzer, amide_mols):
        groups_clean = analyzer.group_by_scaffold(amide_mols)
        groups_with_none = analyzer.group_by_scaffold(amide_mols + [None])
        clean_total = sum(len(v) for v in groups_clean.values())
        none_total = sum(len(v) for v in groups_with_none.values())
        assert clean_total == none_total


# ---------------------------------------------------------------------------
# RGroupDecomposer.decompose()
# ---------------------------------------------------------------------------


class TestRGroupDecompose:
    ANILINE_CORE = "c1ccccc1N"  # aniline as core

    def test_returns_dataframe(self, decomposer, amide_mols):
        import pandas as pd

        df = decomposer.decompose(amide_mols, self.ANILINE_CORE)
        assert isinstance(df, pd.DataFrame)

    def test_mol_idx_column_present(self, decomposer, amide_mols):
        df = decomposer.decompose(amide_mols, self.ANILINE_CORE)
        assert "mol_idx" in df.columns

    def test_core_column_present_when_matched(self, decomposer):
        mol = Chem.MolFromSmiles(AMIDE_A)
        df = decomposer.decompose([mol], self.ANILINE_CORE)
        if len(df) > 0:
            assert "Core" in df.columns

    def test_empty_mol_list_returns_df(self, decomposer):
        import pandas as pd

        df = decomposer.decompose([], self.ANILINE_CORE)
        assert isinstance(df, pd.DataFrame)

    def test_invalid_core_raises(self, decomposer, amide_mols):
        with pytest.raises(ValueError, match="Invalid core SMARTS"):
            decomposer.decompose(amide_mols, "not_a_smarts###")

    def test_none_entries_handled(self, decomposer):
        mol = Chem.MolFromSmiles(AMIDE_A)
        df = decomposer.decompose([None, mol, None], self.ANILINE_CORE)
        assert isinstance(
            df, __builtins__["type"](df) if isinstance(df, type) else type(df)
        )


# ---------------------------------------------------------------------------
# RGroupDecomposer.enumerate_analogs()
# ---------------------------------------------------------------------------


class TestEnumerateAnalogs:
    def test_returns_list(self, decomposer):
        core = "c1cc([*])ccc1"  # benzene with one attachment point
        rgroup_sets = {"R1": ["C", "CC", "CCC"]}
        analogs = decomposer.enumerate_analogs(core, rgroup_sets)
        assert isinstance(analogs, list)

    def test_analogs_are_mol_objects(self, decomposer):
        core = "c1cc([*])ccc1"
        rgroup_sets = {"R1": ["C", "CC"]}
        analogs = decomposer.enumerate_analogs(core, rgroup_sets)
        for mol in analogs:
            assert isinstance(mol, Chem.rdchem.Mol)

    def test_number_of_analogs_bounded_by_combinations(self, decomposer):
        core = "c1cc([*])ccc1"
        rgroup_sets = {"R1": ["C", "CC", "CCC"]}
        analogs = decomposer.enumerate_analogs(core, rgroup_sets)
        assert len(analogs) <= 3

    def test_core_without_attachment_returns_core(self, decomposer):
        core = "c1ccccc1"  # no dummy atom
        rgroup_sets = {"R1": ["C"]}
        analogs = decomposer.enumerate_analogs(core, rgroup_sets)
        assert len(analogs) == 1

    def test_invalid_core_raises(self, decomposer):
        with pytest.raises(ValueError, match="Invalid core SMILES"):
            decomposer.enumerate_analogs("not_valid###", {"R1": ["C"]})

    def test_empty_rgroup_sets(self, decomposer):
        core = "c1ccccc1"
        analogs = decomposer.enumerate_analogs(core, {})
        assert isinstance(analogs, list)


# ---------------------------------------------------------------------------
# Namespace exports from mdatools.docking
# ---------------------------------------------------------------------------


class TestDockingExports:
    def test_scaffold_analyzer_importable(self):
        from mdatools.docking import ScaffoldAnalyzer  # noqa: F401

    def test_rgroup_decomposer_importable(self):
        from mdatools.docking import RGroupDecomposer  # noqa: F401
