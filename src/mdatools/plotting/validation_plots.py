"""Validation timeline and contact fingerprint plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_C_SHADE = "#FFFACD"


def shade_hbond(
    ax: plt.Axes,
    time: pd.Series,
    mask: np.ndarray,
    color: str = _C_SHADE,
    alpha: float = 0.45,
) -> None:
    """Shade time-axis spans where mask is True (H-bond forming frames)."""
    in_run = False
    start = 0.0
    for ti, m in zip(time, mask):
        if m and not in_run:
            start = ti
            in_run = True
        elif not m and in_run:
            ax.axvspan(start, ti, color=color, alpha=alpha, zorder=0)
            in_run = False
    if in_run:
        ax.axvspan(start, time.iloc[-1], color=color, alpha=alpha, zorder=0)


def plot_hbond_timeline(
    df_conf: pd.DataFrame,
    df_hb: pd.DataFrame,
    *,
    hbond_label: str = "H-bond",
    dist_col: str | None = None,
    angle_col: str | None = None,
    rmsd_col: str = "rmsd_global",
    dt_ns: float = 2.0,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Three-panel timeline: RMSD | H-bond distance | H-bond angle.

    Parameters
    ----------
    df_conf:
        DataFrame with frame, time_ns, has_hbond columns (from PyMOLExtractor or
        constructed from HBondResult + RMSD data).
    df_hb:
        HBond events DataFrame (from HBondResult.events).
    dist_col:
        Column in *df_hb* for distance values.  Defaults to ``"distance"``.
    angle_col:
        Column in *df_hb* for angle values.  Defaults to ``"angle"``.
    """
    dist_col = dist_col or "distance"
    angle_col = angle_col or "angle"

    time_ax = df_conf["time_ns"]
    hb_mask = df_conf["has_hbond"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"H-bond Timeline: {hbond_label}", fontsize=13, fontweight="bold")

    # Panel 1: global RMSD (if available)
    ax = axes[0]
    shade_hbond(ax, time_ax, hb_mask)
    if rmsd_col in df_conf.columns:
        ax.plot(time_ax, df_conf[rmsd_col], color="#E05C5C", lw=1.5, label="Global RMSD")
        ax.axhline(2.0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("(1) Global Backbone RMSD")
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Panel 2: H-bond distance
    hb_frames = df_conf[df_conf["has_hbond"]]["time_ns"]
    ax = axes[1]
    shade_hbond(ax, time_ax, hb_mask)
    if not df_hb.empty:
        ax.scatter(hb_frames, df_hb[dist_col], color="#2DA84F", s=20, zorder=3,
                   label="H-bond distance")
        ax.plot(hb_frames, df_hb[dist_col], color="#2DA84F", lw=1.0, alpha=0.6)
    ax.axhline(3.5, color="gray", ls="--", lw=0.8, label="Cutoff 3.5 Å")
    ax.set_ylabel("D–A Distance (Å)")
    ax.set_title("(2) H-bond Distance")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Panel 3: H-bond angle
    ax = axes[2]
    shade_hbond(ax, time_ax, hb_mask)
    if not df_hb.empty:
        ax.scatter(hb_frames, df_hb[angle_col], color="#5B8DD9", s=20, zorder=3,
                   label="D-H…A angle")
        ax.plot(hb_frames, df_hb[angle_col], color="#5B8DD9", lw=1.0, alpha=0.6)
    ax.axhline(150, color="gray", ls="--", lw=0.8, label="Cutoff 150°")
    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("Angle (°)")
    ax.set_title("(3) H-bond Angle")
    ax.set_ylim(100, 185)
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    for ax in axes:
        ax.set_xlim(time_ax.min(), time_ax.max())

    hb_patch = mpatches.Patch(color=_C_SHADE, alpha=0.7, label="H-bond forming frames")
    fig.legend(handles=[hb_patch], loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_contact_fingerprint(
    fp_df: pd.DataFrame,
    sample_name: str,
    *,
    ecd_col: str = "is_ecd",
    top_n: int | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Axes:
    """Horizontal bar chart for contact fingerprint."""
    color_map = {
        "hydrophobic": "#5CA8E0",
        "polar": "#5CB85C",
        "charged": "#E05C5C",
        "other": "#AAAAAA",
    }
    df = fp_df.copy()
    if top_n is not None:
        df = df.head(top_n)

    colors = [color_map.get(ct, "#888888") for ct in df["contact_type"]]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.35)))

    ax.barh(df["residue"], df["occupancy_%"], color=colors, edgecolor="none", height=0.7)

    ecd_residues = df[df[ecd_col] == True]["residue"].tolist() if ecd_col in df.columns else []
    for label in ax.get_yticklabels():
        if label.get_text() in ecd_residues:
            label.set_color("#B8360A")
            label.set_fontweight("bold")

    ax.axvline(80, color="gray", ls="--", lw=0.8, alpha=0.8, label="80% threshold")
    ax.set_xlabel("Contact Occupancy (%)")
    ax.set_title(f"Protein–Ligand Contact Fingerprint\n{sample_name}", fontsize=12)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    handles = [mpatches.Patch(color=c, label=t) for t, c in color_map.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return ax
