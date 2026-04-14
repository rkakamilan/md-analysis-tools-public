"""Pocket environment comparison across multiple snapshots.

Aggregates per-atom metrics from :class:`~mdatools.pocket.profiler.PocketProfiler`
over several snapshots and identifies dynamically changing regions.

"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PocketComparator:
    """Aggregate and compare pocket metrics across multiple snapshots.

    Parameters
    ----------
    tight_threshold:
        d_min threshold (Å) below which an atom is considered in tight
        contact (default ``3.5``).
    open_threshold:
        d_min threshold (Å) above which an atom is considered solvent
        exposed (default ``5.0``).
    hbond_n_NO_min:
        Minimum ``n_NO`` count to flag an atom as an H-bond candidate
        (default ``2``).
    """

    def __init__(
        self,
        tight_threshold: float = 3.5,
        open_threshold: float = 5.0,
        hbond_n_NO_min: int = 2,
    ) -> None:
        self.tight_threshold = tight_threshold
        self.open_threshold = open_threshold
        self.hbond_n_NO_min = hbond_n_NO_min

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def run(
        self,
        metrics_dict: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Aggregate per-snapshot metrics into a comparison DataFrame.

        Parameters
        ----------
        metrics_dict:
            Mapping of snapshot name → list of per-atom metric dicts as
            returned by :func:`~mdatools.pocket.profiler.compute_metrics`.

        Returns
        -------
        DataFrame indexed by atom name with columns:
        ``element``, ``nearest_e``,
        ``d_min_mean``, ``d_min_std``, ``d_min_range``,
        ``d_margin_mean``, ``d_margin_std``,
        ``hydrophob_mean``, ``hydrophob_std``,
        ``n_NO_mean``, ``n_local_mean``,
        ``cone_dist_mean``, ``cone_dist_std``.
        """
        if not metrics_dict:
            return pd.DataFrame()

        by_atom: dict[str, list[dict]] = defaultdict(list)
        for snap_metrics in metrics_dict.values():
            for m in snap_metrics:
                by_atom[m["atom"]].append(m)

        rows: list[dict] = []
        for atom, recs in by_atom.items():
            row: dict = {
                "atom":      atom,
                "element":   recs[0]["element"],
                "nearest_e": recs[0]["nearest_e"],
            }
            for key in ("d_min", "d_margin", "hydrophob", "n_NO", "n_local", "cone_dist"):
                vals = [r[key] for r in recs]
                row[f"{key}_mean"] = round(float(np.mean(vals)), 3)
                row[f"{key}_std"]  = round(float(np.std(vals)), 3)
            dmin_vals = [r["d_min"] for r in recs]
            row["d_min_range"] = round(float(max(dmin_vals) - min(dmin_vals)), 3)
            rows.append(row)

        df = (
            pd.DataFrame(rows)
            .set_index("atom")
            .sort_values("d_min_mean")
        )
        return df

    # ------------------------------------------------------------------
    # Per-snapshot d_min comparison table
    # ------------------------------------------------------------------

    def comparison_table(
        self,
        metrics_dict: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Build a wide d_min table: one column per snapshot.

        Parameters
        ----------
        metrics_dict:
            Same format as :meth:`run`.

        Returns
        -------
        DataFrame with atom as index, columns ``d_min_<snapshot_name>``
        for each snapshot plus ``d_min_mean``, ``d_min_std``,
        ``d_min_range``. Sorted by ``d_min_range`` descending.
        """
        if not metrics_dict:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for snap_name, snap_metrics in metrics_dict.items():
            df = pd.DataFrame(snap_metrics).set_index("atom")[["d_min"]]
            df = df.rename(columns={"d_min": f"d_min_{snap_name}"})
            frames.append(df)

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how="outer")

        dmin_cols = [c for c in merged.columns if c.startswith("d_min_")]
        merged["d_min_mean"]  = merged[dmin_cols].mean(axis=1).round(3)
        merged["d_min_std"]   = merged[dmin_cols].std(axis=1).round(3)
        merged["d_min_range"] = (merged[dmin_cols].max(axis=1) - merged[dmin_cols].min(axis=1)).round(3)

        return merged.sort_values("d_min_range", ascending=False)

    # ------------------------------------------------------------------
    # Auto report
    # ------------------------------------------------------------------

    def auto_report(self, comparison: pd.DataFrame) -> str:
        """Generate a plain-text summary report from comparison results.

        Parameters
        ----------
        comparison:
            Output of :meth:`run`.

        Returns
        -------
        Multi-line string describing tight contacts, open regions,
        dynamically variable atoms, and H-bond candidates.
        """
        if comparison.empty:
            return "No data available for report generation."

        lines: list[str] = ["=== Pocket Environment Comparison Report ===", ""]

        # Tight contacts
        tight = comparison[comparison["d_min_mean"] <= self.tight_threshold]
        if len(tight) > 0:
            names = ", ".join(tight.index.tolist())
            lines.append(
                f"[Tight contacts (d_min_mean <= {self.tight_threshold} Å)]"
                f" {len(tight)} atom(s): {names}"
            )
        else:
            lines.append(f"[Tight contacts] None (threshold: {self.tight_threshold} Å)")

        # Open / exposed
        exposed = comparison[comparison["d_min_mean"] > self.open_threshold]
        if len(exposed) > 0:
            names = ", ".join(exposed.index.tolist())
            lines.append(
                f"[Solvent-exposed (d_min_mean > {self.open_threshold} Å)]"
                f" {len(exposed)} atom(s): {names}"
            )

        # High variability
        if "d_min_std" in comparison.columns:
            std_thresh = comparison["d_min_std"].quantile(0.75)
            variable = comparison[comparison["d_min_std"] > std_thresh]
            if len(variable) > 0:
                names = ", ".join(variable.index.tolist())
                lines.append(
                    f"[Dynamic atoms (d_min_std > {std_thresh:.2f} Å, top quartile)]"
                    f" {len(variable)} atom(s): {names}"
                )

        # H-bond candidates
        if "n_NO_mean" in comparison.columns:
            hbond = comparison[comparison["n_NO_mean"] >= self.hbond_n_NO_min]
            hbond = hbond.sort_values("n_NO_mean", ascending=False)
            if len(hbond) > 0:
                candidates = ", ".join(
                    f"{atom}(n_NO={row['n_NO_mean']:.1f})"
                    for atom, row in hbond.iterrows()
                )
                lines.append(
                    f"[H-bond candidates (n_NO_mean >= {self.hbond_n_NO_min})]"
                    f" {len(hbond)} atom(s): {candidates}"
                )

        # Hydrophobic environment
        if "hydrophob_mean" in comparison.columns:
            hydro = comparison[comparison["hydrophob_mean"] >= 0.65]
            polar = comparison[comparison["hydrophob_mean"] <= 0.40]
            if len(hydro) > 0:
                lines.append(
                    f"[Hydrophobic environment (hydrophob >= 0.65)]"
                    f" {len(hydro)} atom(s): {', '.join(hydro.index.tolist())}"
                )
            if len(polar) > 0:
                lines.append(
                    f"[Polar environment (hydrophob <= 0.40)]"
                    f" {len(polar)} atom(s): {', '.join(polar.index.tolist())}"
                )

        lines.append("")
        lines.append(f"Total atoms analysed: {len(comparison)}")

        return "\n".join(lines)
