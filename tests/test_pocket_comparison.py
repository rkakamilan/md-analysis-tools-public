"""Tests for PocketComparator (Feature C).

No rdkit/cairosvg required for these tests.
"""

from __future__ import annotations

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metrics(n_snaps: int = 3, base_dmin: float = 4.0) -> dict[str, list[dict]]:
    """Build synthetic metrics_dict with n_snaps snapshots, 3 atoms each."""
    atoms = [
        ("C1", "C"),
        ("N2", "N"),
        ("O3", "O"),
    ]
    result = {}
    for i in range(n_snaps):
        snap_metrics = []
        for j, (name, elem) in enumerate(atoms):
            snap_metrics.append({
                "atom":      name,
                "element":   elem,
                "d_min":     round(base_dmin + i * 0.5 + j * 0.3, 2),
                "d_margin":  round(base_dmin + i * 0.5 + j * 0.3 - 3.2, 2),
                "nearest_e": "C",
                "cone_dist": round(base_dmin + i * 0.5 + j * 0.3 + 0.2, 2),
                "hydrophob": 0.5 + j * 0.1,
                "n_NO":      j,
                "n_local":   10,
            })
        result[f"snap{i:02d}"] = snap_metrics
    return result


# ---------------------------------------------------------------------------
# PocketComparator.run()
# ---------------------------------------------------------------------------

class TestPocketComparatorRun:
    def test_returns_dataframe(self):
        from mdatools.pocket.comparator import PocketComparator

        md = _make_metrics()
        comp = PocketComparator()
        df = comp.run(md)

        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_atom(self):
        from mdatools.pocket.comparator import PocketComparator

        md = _make_metrics(n_snaps=3)
        df = PocketComparator().run(md)

        assert set(df.index) == {"C1", "N2", "O3"}

    def test_mean_columns_present(self):
        from mdatools.pocket.comparator import PocketComparator

        df = PocketComparator().run(_make_metrics())

        for col in ("d_min_mean", "d_min_std", "d_min_range",
                    "hydrophob_mean", "n_NO_mean", "cone_dist_mean"):
            assert col in df.columns, f"Missing column: {col}"

    def test_d_min_mean_is_correct(self):
        """d_min_mean should equal the arithmetic mean across snapshots."""
        from mdatools.pocket.comparator import PocketComparator

        # 3 snaps, atom C1 (j=0):
        #   snap0: d_min=4.0, snap1: d_min=4.5, snap2: d_min=5.0  → mean=4.5
        md = _make_metrics(n_snaps=3, base_dmin=4.0)
        df = PocketComparator().run(md)

        assert df.loc["C1", "d_min_mean"] == pytest.approx(4.5, abs=0.01)

    def test_d_min_range_is_max_minus_min(self):
        from mdatools.pocket.comparator import PocketComparator

        md = _make_metrics(n_snaps=3, base_dmin=4.0)
        df = PocketComparator().run(md)

        # C1: snap0=4.0, snap1=4.5, snap2=5.0 → range=1.0
        assert df.loc["C1", "d_min_range"] == pytest.approx(1.0, abs=0.01)

    def test_empty_input_returns_empty_dataframe(self):
        from mdatools.pocket.comparator import PocketComparator

        df = PocketComparator().run({})
        assert df.empty

    def test_single_snapshot_std_is_zero(self):
        from mdatools.pocket.comparator import PocketComparator

        md = _make_metrics(n_snaps=1)
        df = PocketComparator().run(md)

        assert df.loc["C1", "d_min_std"] == pytest.approx(0.0, abs=0.001)

    def test_sorted_by_d_min_mean(self):
        from mdatools.pocket.comparator import PocketComparator

        df = PocketComparator().run(_make_metrics(n_snaps=2))
        means = df["d_min_mean"].tolist()

        assert means == sorted(means)


# ---------------------------------------------------------------------------
# PocketComparator.comparison_table()
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def test_one_column_per_snapshot(self):
        from mdatools.pocket.comparator import PocketComparator

        md = _make_metrics(n_snaps=3)
        tbl = PocketComparator().comparison_table(md)

        dmin_cols = [c for c in tbl.columns if c.startswith("d_min_snap")]
        assert len(dmin_cols) == 3

    def test_range_column_present(self):
        from mdatools.pocket.comparator import PocketComparator

        tbl = PocketComparator().comparison_table(_make_metrics())
        assert "d_min_range" in tbl.columns

    def test_sorted_by_range_descending(self):
        from mdatools.pocket.comparator import PocketComparator

        tbl = PocketComparator().comparison_table(_make_metrics(n_snaps=3))
        ranges = tbl["d_min_range"].tolist()
        assert ranges == sorted(ranges, reverse=True)


# ---------------------------------------------------------------------------
# PocketComparator.auto_report()
# ---------------------------------------------------------------------------

class TestAutoReport:
    def test_returns_string(self):
        from mdatools.pocket.comparator import PocketComparator

        comp = PocketComparator()
        df = comp.run(_make_metrics())
        report = comp.auto_report(df)

        assert isinstance(report, str)

    def test_tight_contact_detected(self):
        """Atom with d_min_mean below threshold should appear in report."""
        from mdatools.pocket.comparator import PocketComparator

        comp = PocketComparator(tight_threshold=5.0)  # raise threshold to catch C1
        df = comp.run(_make_metrics(base_dmin=3.0))
        report = comp.auto_report(df)

        assert "Tight contacts" in report
        assert "C1" in report

    def test_hbond_candidate_detected(self):
        """Atom with n_NO_mean >= threshold should appear in H-bond section."""
        from mdatools.pocket.comparator import PocketComparator

        # O3 has n_NO=2 in our fixture
        comp = PocketComparator(hbond_n_NO_min=2)
        df = comp.run(_make_metrics())
        report = comp.auto_report(df)

        assert "H-bond" in report
        assert "O3" in report

    def test_empty_dataframe_report(self):
        from mdatools.pocket.comparator import PocketComparator

        report = PocketComparator().auto_report(pd.DataFrame())
        assert "No data" in report

    def test_total_atoms_in_report(self):
        from mdatools.pocket.comparator import PocketComparator

        comp = PocketComparator()
        df = comp.run(_make_metrics(n_snaps=2))
        report = comp.auto_report(df)

        assert "3" in report   # 3 atoms total
