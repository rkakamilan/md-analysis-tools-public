"""Tests for mdatools.docking.preparation.adme_profiler."""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mdatools.docking.preparation.adme_profiler import (  # noqa: E402
    ADMEProfile,
    ADMEProfiler,
    plot_adme_distribution,
    plot_adme_radar,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"  # MW ~180, drug-like
IBUPROFEN = "CC(C)Cc1ccc(CC(C)C(=O)O)cc1"  # MW ~206, drug-like
PAINS_EXAMPLE = "O=C1c2ccccc2C(=O)c2ccccc21"  # anthraquinone — PAINS hit


@pytest.fixture()
def profiler():
    return ADMEProfiler()


@pytest.fixture()
def aspirin_mol():
    return Chem.MolFromSmiles(ASPIRIN)


@pytest.fixture()
def ibuprofen_mol():
    return Chem.MolFromSmiles(IBUPROFEN)


@pytest.fixture()
def pains_mol():
    return Chem.MolFromSmiles(PAINS_EXAMPLE)


# ---------------------------------------------------------------------------
# ADMEProfile dataclass
# ---------------------------------------------------------------------------


class TestADMEProfile:
    def test_default_values(self):
        p = ADMEProfile()
        assert p.mol_weight == 0.0
        assert p.pains_alerts == []
        assert p.is_drug_like is False

    def test_is_drug_like_when_all_pass(self):
        p = ADMEProfile(
            mol_weight=200.0,
            logp=2.0,
            hbd=2,
            hba=4,
            tpsa=60.0,
            rotatable_bonds=3,
            qed=0.7,
            passes_ro5=True,
            passes_veber=True,
            pains_alerts=[],
        )
        assert p.is_drug_like is True

    def test_not_drug_like_with_pains(self):
        p = ADMEProfile(passes_ro5=True, passes_veber=True, pains_alerts=["alert_a"])
        assert p.is_drug_like is False

    def test_not_drug_like_fails_ro5(self):
        p = ADMEProfile(passes_ro5=False, passes_veber=True, pains_alerts=[])
        assert p.is_drug_like is False


# ---------------------------------------------------------------------------
# ADMEProfiler.profile()
# ---------------------------------------------------------------------------


class TestADMEProfilerProfile:
    def test_aspirin_profile_returns_adme_profile(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert isinstance(result, ADMEProfile)

    def test_aspirin_mol_weight_range(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert 170 < result.mol_weight < 200

    def test_aspirin_logp_positive(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert result.logp > 0

    def test_aspirin_passes_ro5(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert result.passes_ro5 is True

    def test_aspirin_passes_veber(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert result.passes_veber is True

    def test_aspirin_no_pains(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert result.pains_alerts == []

    def test_aspirin_qed_in_unit_interval(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert 0.0 <= result.qed <= 1.0

    def test_large_molecule_fails_ro5(self, profiler):
        # Cyclosporin A-like: MW ~1200 → fails Ro5 even with one-violation rule
        smi = "CCC1NC(=O)C(CC(C)C)N(C)C(=O)C(C(CC)C)OC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C1C"
        mol = Chem.MolFromSmiles(smi)
        result = profiler.profile(mol)
        assert result.passes_ro5 is False

    def test_pains_molecule_has_alerts(self, profiler, pains_mol):
        result = profiler.profile(pains_mol)
        assert len(result.pains_alerts) > 0

    def test_hbd_hba_nonnegative(self, profiler, ibuprofen_mol):
        result = profiler.profile(ibuprofen_mol)
        assert result.hbd >= 0
        assert result.hba >= 0

    def test_tpsa_nonnegative(self, profiler, aspirin_mol):
        result = profiler.profile(aspirin_mol)
        assert result.tpsa >= 0.0

    def test_rotatable_bonds_nonnegative(self, profiler, ibuprofen_mol):
        result = profiler.profile(ibuprofen_mol)
        assert result.rotatable_bonds >= 0


# ---------------------------------------------------------------------------
# ADMEProfiler.profile_batch()
# ---------------------------------------------------------------------------


class TestADMEProfilerBatch:
    def test_returns_dataframe(self, profiler, aspirin_mol, ibuprofen_mol):
        import pandas as pd

        df = profiler.profile_batch([aspirin_mol, ibuprofen_mol])
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self, profiler, aspirin_mol, ibuprofen_mol):
        df = profiler.profile_batch([aspirin_mol, ibuprofen_mol])
        assert len(df) == 2

    def test_required_columns_present(self, profiler, aspirin_mol):
        df = profiler.profile_batch([aspirin_mol])
        for col in [
            "mol_weight",
            "logp",
            "hbd",
            "hba",
            "tpsa",
            "rotatable_bonds",
            "qed",
            "passes_ro5",
            "passes_veber",
            "pains_alerts",
            "is_drug_like",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_none_entry_handled_gracefully(self, profiler, aspirin_mol):
        df = profiler.profile_batch([aspirin_mol, None])
        assert len(df) == 2

    def test_empty_list_returns_empty_df(self, profiler):
        import pandas as pd

        df = profiler.profile_batch([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# ADMEProfiler.filter()
# ---------------------------------------------------------------------------


class TestADMEProfilerFilter:
    def test_filter_ro5_removes_large_molecule(self, profiler, aspirin_mol):
        smi = "CCC1NC(=O)C(CC(C)C)N(C)C(=O)C(C(CC)C)OC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C1C"
        large_mol = Chem.MolFromSmiles(smi)
        kept = profiler.filter([aspirin_mol, large_mol], rules=["ro5"])
        assert aspirin_mol in kept
        assert large_mol not in kept

    def test_filter_no_pains_removes_pains_hits(self, profiler, aspirin_mol, pains_mol):
        kept = profiler.filter([aspirin_mol, pains_mol], rules=["no_pains"])
        assert aspirin_mol in kept
        assert pains_mol not in kept

    def test_filter_empty_rules_keeps_all(self, profiler, aspirin_mol, ibuprofen_mol):
        kept = profiler.filter([aspirin_mol, ibuprofen_mol], rules=[])
        assert len(kept) == 2

    def test_filter_none_entries_excluded(self, profiler, aspirin_mol):
        kept = profiler.filter([aspirin_mol, None], rules=[])
        assert None not in kept
        assert aspirin_mol in kept

    def test_combined_rules_and_logic(self, profiler, aspirin_mol, pains_mol):
        kept = profiler.filter([aspirin_mol, pains_mol], rules=["ro5", "no_pains"])
        assert aspirin_mol in kept
        assert pains_mol not in kept


# ---------------------------------------------------------------------------
# Top-level docking namespace exports
# ---------------------------------------------------------------------------


class TestDockingExports:
    def test_adme_profile_importable_from_docking(self):
        from mdatools.docking import ADMEProfile  # noqa: F401

    def test_adme_profiler_importable_from_docking(self):
        from mdatools.docking import ADMEProfiler  # noqa: F401

    def test_plot_adme_radar_importable(self):
        from mdatools.docking import plot_adme_radar  # noqa: F401

    def test_plot_adme_distribution_importable(self):
        from mdatools.docking import plot_adme_distribution  # noqa: F401


# ---------------------------------------------------------------------------
# Plotting (import-guarded — requires matplotlib)
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_plot_adme_radar_returns_figure(self, profiler, aspirin_mol):
        matplotlib = pytest.importorskip("matplotlib")  # noqa: F841
        import matplotlib.pyplot as plt

        profile = profiler.profile(aspirin_mol)
        fig = plot_adme_radar(profile, title="Aspirin")
        assert fig is not None
        plt.close(fig)

    def test_plot_adme_distribution_returns_figure(
        self, profiler, aspirin_mol, ibuprofen_mol
    ):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = profiler.profile_batch([aspirin_mol, ibuprofen_mol])
        fig = plot_adme_distribution(df, property_col="mol_weight")
        assert fig is not None
        plt.close(fig)

    def test_plot_adme_distribution_logp(self, profiler, aspirin_mol):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = profiler.profile_batch([aspirin_mol])
        fig = plot_adme_distribution(df, property_col="logp")
        assert fig is not None
        plt.close(fig)
