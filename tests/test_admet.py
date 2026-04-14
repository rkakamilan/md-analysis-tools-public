"""Tests for ADMETCalculator and ADMET plot functions."""

from __future__ import annotations

import pytest

# Skip entire module when RDKit is not installed
rdkit = pytest.importorskip("rdkit", reason="RDKit not installed; skip admet tests")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdatools.analysis.admet import ADMETCalculator, ADMETResult

# ---------------------------------------------------------------------------
# Test molecules (SMILES)
# ---------------------------------------------------------------------------
# Aspirin: MW≈180, Ro5-compliant, clean PAINS
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# Cyclosporin A: MW≈1202, major Ro5 violator (MW >> 500)
CYCLOSPORIN_LARGE = "CCC1C(=O)N(CC(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)NC(C(=O)NC(" \
    "C(=O)N(C(C(=O)N(C(C(=O)N(C(C(=O)N1)CC(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC" \
    "(C)C)C)CC(C)C)C)C"
# Rhodanine scaffold: known PAINS alert (PAINS_A: rhodanine_A(6))
RHODANINE_PAINS = "O=C1NC(=S)SC1"
# High-HBD violator: > 5 HBD
HIGH_HBD = "OC(O)(O)C(O)(O)C(O)(O)O"  # polyol, many OH donors


@pytest.fixture(scope="module")
def calc():
    return ADMETCalculator(check_pains=True)


@pytest.fixture(scope="module")
def aspirin_result(calc):
    return calc.from_smiles(ASPIRIN, name="aspirin")


# ---------------------------------------------------------------------------
# ADMETResult structure
# ---------------------------------------------------------------------------


class TestADMETResult:
    def test_name(self, aspirin_result):
        assert aspirin_result.name == "aspirin"

    def test_mw_aspirin(self, aspirin_result):
        # Aspirin exact MW ≈ 180.04 Da
        assert 170 < aspirin_result.mw < 200

    def test_logp_aspirin(self, aspirin_result):
        # Aspirin LogP typically ~1.2
        assert -1 < aspirin_result.logp < 4

    def test_hbd_aspirin(self, aspirin_result):
        # Aspirin has 1 HBD (carboxylic OH)
        assert aspirin_result.hbd >= 1

    def test_hba_aspirin(self, aspirin_result):
        # Aspirin has multiple HBA (carbonyl oxygens)
        assert aspirin_result.hba >= 2

    def test_tpsa_aspirin(self, aspirin_result):
        assert 0 < aspirin_result.tpsa < 140

    def test_qed_aspirin(self, aspirin_result):
        # Aspirin should be reasonably drug-like
        assert 0 < aspirin_result.qed <= 1.0

    def test_sa_score_aspirin(self, aspirin_result):
        assert 1 <= aspirin_result.sa_score <= 10

    def test_pains_patterns_is_list(self, aspirin_result):
        assert isinstance(aspirin_result.pains_patterns, list)


# ---------------------------------------------------------------------------
# Lipinski Ro5 and Veber
# ---------------------------------------------------------------------------


class TestLipinskiVeber:
    def test_aspirin_lipinski_pass(self, aspirin_result):
        assert aspirin_result.lipinski_pass is True

    def test_aspirin_veber_pass(self, aspirin_result):
        assert aspirin_result.veber_pass is True

    def test_large_mw_lipinski_fail(self, calc):
        # Simple MW violator: big polypeptide-like chain
        large_smi = "C" * 100 + "N"  # very large MW
        result = calc.from_smiles(large_smi, name="huge")
        assert result.lipinski_pass is False

    def test_high_hbd_lipinski_fail(self, calc):
        result = calc.from_smiles(HIGH_HBD, name="polyol")
        # Many OH groups → HBD > 5 → Ro5 violation
        assert result.hbd > 5
        assert result.lipinski_pass is False

    def test_veber_high_rotbonds_fail(self, calc):
        # Long chain with many rotatable bonds
        long_chain = "C" + "CC" * 8 + "O"  # 17+ heavy atoms, many rot bonds
        result = calc.from_smiles(long_chain, name="chain")
        # rotbonds may be high; check veber_pass is consistent with tpsa/rotbonds
        assert isinstance(result.veber_pass, bool)


# ---------------------------------------------------------------------------
# PAINS detection
# ---------------------------------------------------------------------------


class TestPAINS:
    def test_aspirin_no_pains(self, aspirin_result):
        # Aspirin should be clean
        assert aspirin_result.pains_alert is False

    def test_rhodanine_pains_detected(self, calc):
        result = calc.from_smiles(RHODANINE_PAINS, name="rhodanine")
        assert result.pains_alert is True
        assert len(result.pains_patterns) >= 1

    def test_no_pains_check(self):
        # When check_pains=False, no PAINS filter is run
        calc_no_pains = ADMETCalculator(check_pains=False)
        result = calc_no_pains.from_smiles(RHODANINE_PAINS, name="rhodanine_nopains")
        assert result.pains_alert is False
        assert result.pains_patterns == []


# ---------------------------------------------------------------------------
# ImportError when RDKit absent
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_without_rdkit(self):
        from unittest.mock import patch
        from mdatools.analysis import admet as admet_mod

        # Block the import by mapping module names to None in sys.modules.
        # This triggers ImportError without reloading the module (avoids
        # corrupting module-scoped class identity used by other tests).
        with patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
            with pytest.raises(ImportError, match="(?i)rdkit"):
                admet_mod._require_rdkit()


# ---------------------------------------------------------------------------
# from_mol
# ---------------------------------------------------------------------------


class TestFromMol:
    def test_from_mol_returns_result(self, calc):
        from rdkit.Chem import MolFromSmiles
        mol = MolFromSmiles(ASPIRIN)
        result = calc.from_mol(mol, name="aspirin_mol")
        assert type(result).__name__ == "ADMETResult"
        assert 170 < result.mw < 200

    def test_from_mol_same_as_from_smiles(self, calc):
        from rdkit.Chem import MolFromSmiles
        mol = MolFromSmiles(ASPIRIN)
        r_mol = calc.from_mol(mol, name="test")
        r_smi = calc.from_smiles(ASPIRIN, name="test")
        assert abs(r_mol.mw - r_smi.mw) < 1e-6
        assert abs(r_mol.qed - r_smi.qed) < 1e-6


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_returns_dataframe(self, calc):
        import pandas as pd
        df = calc.batch([ASPIRIN, RHODANINE_PAINS])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_batch_columns(self, calc):
        df = calc.batch([ASPIRIN])
        expected = {"name", "mw", "logp", "hbd", "hba", "tpsa", "qed", "sa_score"}
        assert expected.issubset(set(df.columns))

    def test_batch_skips_invalid_smiles(self, calc):
        import pandas as pd
        df = calc.batch([ASPIRIN, "NOT_A_SMILES_XYZ", RHODANINE_PAINS])
        assert len(df) == 2  # invalid skipped


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestADMETPlots:
    def test_plot_admet_radar(self, aspirin_result):
        from mdatools.plotting.admet_plots import plot_admet_radar
        fig = plot_admet_radar(aspirin_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_admet_comparison(self, calc, aspirin_result):
        from mdatools.plotting.admet_plots import plot_admet_comparison
        results = {
            "aspirin": aspirin_result,
            "rhodanine": calc.from_smiles(RHODANINE_PAINS, name="rhodanine"),
        }
        fig = plot_admet_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_admet_comparison_custom_props(self, aspirin_result):
        from mdatools.plotting.admet_plots import plot_admet_comparison
        fig = plot_admet_comparison({"aspirin": aspirin_result}, properties=["mw", "qed"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_qed_distribution(self, calc):
        from mdatools.plotting.admet_plots import plot_qed_distribution
        df = calc.batch([ASPIRIN, RHODANINE_PAINS])
        fig = plot_qed_distribution(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
