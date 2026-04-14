"""Simulation convergence assessment — RMSD half-split, block error, autocorrelation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Result of an RMSD half-split convergence test.

    Attributes
    ----------
    is_converged:
        Heuristic judgement: ``True`` when ``ks_pvalue >= 0.05``.
    first_half_mean:
        Mean RMSD of the first half of the trajectory.
    second_half_mean:
        Mean RMSD of the second half of the trajectory.
    ks_statistic:
        Kolmogorov-Smirnov test statistic between the two halves.
    ks_pvalue:
        KS test p-value.  Small values indicate the distributions differ
        significantly (non-converged).
    recommendation:
        Human-readable assessment string.
    """

    is_converged: bool
    first_half_mean: float
    second_half_mean: float
    ks_statistic: float
    ks_pvalue: float
    recommendation: str


class ConvergenceAnalyzer:
    """Assess simulation convergence via multiple statistical diagnostics.

    Diagnostics provided:

    1. **RMSD half-split comparison** — KS test between first and second
       half of the trajectory.
    2. **Block error analysis** — SEM as a function of block size; a plateau
       indicates the sampling is uncorrelated.
    3. **Autocorrelation time** — integrated autocorrelation time τ_int via
       the FFT-based normalised autocorrelation function.
    4. **Replica consistency** — pairwise KS statistic + Wasserstein distance
       across replicas.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # 1. RMSD convergence
    # ------------------------------------------------------------------

    def rmsd_convergence(
        self,
        rmsd_df: pd.DataFrame,
        split_frac: float = 0.5,
        column: str = "Backbone",
    ) -> ConvergenceResult:
        """Compare RMSD distribution in first vs second half of simulation.

        Parameters
        ----------
        rmsd_df:
            DataFrame from :class:`~mdatools.analysis.rmsd.RMSDResult`.
            Must contain a column named *column*.
        split_frac:
            Fraction of frames used as the first half (default 0.5).
        column:
            Column to analyse (default ``'Backbone'``).

        Returns
        -------
        ConvergenceResult
        """
        from scipy.stats import ks_2samp

        if column not in rmsd_df.columns:
            raise ValueError(f"Column '{column}' not found in rmsd_df.")

        values = rmsd_df[column].dropna().to_numpy()
        n = len(values)
        if n < 4:
            raise ValueError(
                f"Too few frames ({n}) for convergence analysis. Need at least 4."
            )

        split = max(1, int(n * split_frac))
        first = values[:split]
        second = values[split:]

        stat, pvalue = ks_2samp(first, second)
        is_converged = bool(pvalue >= 0.05)
        delta = abs(float(second.mean()) - float(first.mean()))

        if is_converged:
            recommendation = (
                f"Converged (KS p={pvalue:.3f} ≥ 0.05). "
                f"Mean RMSD shift: {delta:.2f} Å."
            )
        else:
            recommendation = (
                f"NOT converged (KS p={pvalue:.3f} < 0.05). "
                f"Mean RMSD shift: {delta:.2f} Å. "
                "Consider extending the simulation or discarding more equilibration."
            )

        return ConvergenceResult(
            is_converged=is_converged,
            first_half_mean=round(float(first.mean()), 4),
            second_half_mean=round(float(second.mean()), 4),
            ks_statistic=round(float(stat), 6),
            ks_pvalue=round(float(pvalue), 6),
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # 2. Block error analysis
    # ------------------------------------------------------------------

    def block_error(
        self,
        series: np.ndarray | pd.Series,
        max_block_size: int | None = None,
    ) -> pd.DataFrame:
        """Block averaging error analysis.

        Splits *series* into non-overlapping blocks of increasing size and
        computes the standard error of the mean (SEM) for each block size.
        When the SEM plateaus, sampling is considered uncorrelated.

        Parameters
        ----------
        series:
            1-D array of scalar observations (e.g. RMSD, RMSF, occupancy).
        max_block_size:
            Maximum block size to test. Defaults to ``len(series) // 4``.

        Returns
        -------
        DataFrame with columns ``block_size``, ``mean``, ``sem``, ``n_blocks``.
        """
        arr = np.asarray(series, dtype=np.float64).ravel()
        n = len(arr)
        if n < 4:
            raise ValueError(f"Series too short ({n}) for block error analysis.")

        max_bs = max_block_size or max(1, n // 4)
        max_bs = min(max_bs, n // 2)

        rows = []
        for bs in range(1, max_bs + 1):
            n_blocks = n // bs
            if n_blocks < 2:
                break
            trimmed = arr[: n_blocks * bs].reshape(n_blocks, bs)
            block_means = trimmed.mean(axis=1)
            rows.append(
                {
                    "block_size": bs,
                    "mean": float(block_means.mean()),
                    "sem": float(block_means.std(ddof=1) / np.sqrt(n_blocks)),
                    "n_blocks": n_blocks,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Autocorrelation time
    # ------------------------------------------------------------------

    def autocorrelation_time(
        self,
        series: np.ndarray | pd.Series,
        max_lag: int = 500,
    ) -> float:
        """Estimate integrated autocorrelation time τ_int.

        Uses the FFT-based normalised autocorrelation function.  Integration
        is stopped at the first zero-crossing (or at *max_lag*) to avoid
        noise accumulation.

        Parameters
        ----------
        series:
            1-D array of scalar observations.
        max_lag:
            Maximum lag to consider.

        Returns
        -------
        Estimated τ_int (float).  Returns 0.5 (minimum) when the series is
        white noise.
        """
        arr = np.asarray(series, dtype=np.float64).ravel()
        arr = arr - arr.mean()
        n = len(arr)
        max_lag = min(max_lag, n - 1)

        # FFT-based full autocorrelation
        fft = np.fft.rfft(arr, n=2 * n)
        power = fft * np.conj(fft)
        acf_full = np.fft.irfft(power)[:n].real
        if acf_full[0] == 0:
            return 0.5
        acf = acf_full / acf_full[0]

        # Integrate up to first zero-crossing or max_lag
        tau = 0.5
        for lag in range(1, max_lag + 1):
            if lag >= n or acf[lag] <= 0:
                break
            tau += acf[lag]

        return float(max(0.5, tau))

    # ------------------------------------------------------------------
    # 4. Replica consistency
    # ------------------------------------------------------------------

    def replica_consistency(
        self,
        results: list,
        metric: str = "rmsd",
    ) -> pd.DataFrame:
        """Compute pairwise distribution overlap across replicas.

        Parameters
        ----------
        results:
            List of result objects.  Supported types:

            - ``RMSDResult``: uses ``df['Backbone']``
            - ``RMSFResult``: uses ``protein_df['rmsf']``
            - Any object with a ``df`` attribute containing a numeric column
              matching *metric* (case-insensitive), or a plain ``pd.Series``
              / ``np.ndarray``.
        metric:
            Column name hint for ``RMSDResult`` / generic DataFrames.

        Returns
        -------
        DataFrame with columns ``replica_a``, ``replica_b``,
        ``ks_statistic``, ``wasserstein_distance``.
        """
        from scipy.stats import ks_2samp
        from scipy.stats import wasserstein_distance as wdist

        arrays = _extract_arrays(results, metric)
        names = [getattr(r, "sample_name", f"replica_{i}") for i, r in enumerate(results)]

        rows = []
        n = len(arrays)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = arrays[i], arrays[j]
                ks_stat, _ = ks_2samp(a, b)
                wd = wdist(a, b)
                rows.append(
                    {
                        "replica_a": names[i],
                        "replica_b": names[j],
                        "ks_statistic": round(float(ks_stat), 6),
                        "wasserstein_distance": round(float(wd), 4),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["replica_a", "replica_b", "ks_statistic", "wasserstein_distance"]
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_arrays(results: list, metric: str) -> list[np.ndarray]:
    """Extract 1-D numeric arrays from heterogeneous result objects."""
    arrays = []
    for r in results:
        if isinstance(r, (np.ndarray, pd.Series)):
            arrays.append(np.asarray(r, dtype=np.float64).ravel())
            continue
        # Try common mdatools result types
        try:
            # RMSDResult → df has columns Frame/Time/Backbone/Ligand
            col = metric.capitalize()
            if hasattr(r, "df") and col in r.df.columns:
                arrays.append(r.df[col].dropna().to_numpy(dtype=np.float64))
                continue
        except Exception:
            pass
        try:
            # RMSFResult → protein_df['rmsf']
            if hasattr(r, "protein_df") and "rmsf" in r.protein_df.columns:
                arrays.append(r.protein_df["rmsf"].dropna().to_numpy(dtype=np.float64))
                continue
        except Exception:
            pass
        try:
            # Generic: find first numeric column
            if hasattr(r, "df"):
                num_cols = r.df.select_dtypes(include=np.number).columns
                if len(num_cols):
                    arrays.append(r.df[num_cols[0]].dropna().to_numpy(dtype=np.float64))
                    continue
        except Exception:
            pass
        raise TypeError(
            f"Cannot extract numeric array from result object: {type(r)}. "
            "Pass RMSDResult, RMSFResult, np.ndarray, or pd.Series."
        )
    return arrays
