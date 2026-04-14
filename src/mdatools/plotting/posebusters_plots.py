"""PoseBusters result visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_pb_pass_rate(
    result,  # PoseBustersResult
    *,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Bar chart: per-check pass rates across all frames.

    Parameters
    ----------
    result:
        :class:`~mdatools.posebusters.PoseBustersResult`.
    """
    df = result.results
    bool_cols = [
        c for c in df.columns
        if df[c].dtype == bool and c != "all_pass"
    ]
    if not bool_cols:
        raise ValueError("No boolean check columns found in results.")

    pass_rates = df[bool_cols].mean() * 100
    pass_rates = pass_rates.sort_values(ascending=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(pass_rates) * 0.4)))

    colors = ["#E05C5C" if v < 50 else "#4878CF" if v < 90 else "#5CB85C"
              for v in pass_rates]
    ax.barh(pass_rates.index, pass_rates.values, color=colors)
    ax.axvline(100, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Pass rate (%)")
    ax.set_title(
        f"PoseBusters Check Pass Rates\n{result.sample_name}", fontsize=12
    )
    ax.set_xlim(0, 110)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax


def plot_pb_check_heatmap(
    result,  # PoseBustersResult
    *,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Heatmap: pass/fail per frame per check (frames × checks)."""
    df = result.results
    bool_cols = [
        c for c in df.columns
        if df[c].dtype == bool and c != "all_pass"
    ]
    if not bool_cols:
        raise ValueError("No boolean check columns found in results.")

    matrix = df.set_index("frame")[bool_cols].astype(int)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, len(bool_cols) * 0.6), 6))

    sns.heatmap(
        matrix.T,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.6},
        linewidths=0.3,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Check")
    ax.set_title(f"PoseBusters Check Heatmap\n{result.sample_name}", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax


def plot_pb_score_timeseries(
    result,  # PoseBustersResult
    hbond_events: pd.DataFrame | None = None,
    *,
    dt_ns: float = 2.0,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Time series of pass/fail and energy_ratio per frame.

    Optionally overlays H-bond forming frames as shaded regions.

    Parameters
    ----------
    hbond_events:
        HBond events DataFrame (from ``HBondResult.events``); used for
        shading H-bond forming frames.
    dt_ns:
        Time step in nanoseconds.
    """
    from .validation_plots import shade_hbond

    df = result.results.copy()
    df["time_ns"] = df["frame"] * dt_ns

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    hb_mask = np.zeros(len(df), dtype=bool)
    if hbond_events is not None and not hbond_events.empty:
        hb_frames = set(hbond_events["frame"].tolist())
        hb_mask = np.array([f in hb_frames for f in df["frame"]])
        shade_hbond(ax, df["time_ns"], hb_mask)

    # Plot energy_ratio if available, else pass/fail as 0/1
    if "energy_ratio" in df.columns:
        ax.plot(df["time_ns"], df["energy_ratio"], lw=1.2, color="#4878CF",
                label="energy_ratio")
        ax.set_ylabel("energy_ratio")
    else:
        ax.plot(df["time_ns"], df["all_pass"].astype(int), lw=1.0,
                color="#4878CF", label="pass (1) / fail (0)")
        ax.set_ylabel("Pass (1) / Fail (0)")

    # Mark passing frames
    passing = df[df["all_pass"]]
    ax.scatter(
        passing["time_ns"],
        passing.get("energy_ratio", passing["all_pass"].astype(int)),
        color="#5CB85C", s=25, zorder=3, label="pass",
    )

    ax.set_xlabel("Time (ns)")
    ax.set_title(
        f"PoseBusters Validation Timeseries\n{result.sample_name}", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax
