"""RMSF visualisation — protein per-residue and ligand per-atom."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Element → colour mapping for ligand atoms
_ELEMENT_COLORS: dict[str, str] = {
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "F": "#90E050",
    "H": "#FFFFFF",
    "P": "#FF8000",
}
_DEFAULT_COLOR = "#B0B0B0"


def plot_protein_rmsf(
    result,  # RMSFResult
    pocket_resids: list[int] | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of per-residue Cα RMSF.

    Binding site residues (``pocket_resids``) are highlighted in orange.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.rmsf.RMSFResult` from
        :meth:`RMSFAnalyzer.run`.
    pocket_resids:
        Residue IDs to highlight.  Pass ``None`` to skip highlighting.
    save_path:
        If given, save figure to this path.
    dpi:
        Figure resolution for saved image.
    """
    df: pd.DataFrame = result.protein_df
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Protein RMSF — {result.sample_name} (no data)")
        return fig

    pocket_set = set(pocket_resids) if pocket_resids else set()
    colors = [
        "darkorange" if int(row.resid) in pocket_set else "steelblue"
        for row in df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.25), 4), constrained_layout=True)
    ax.bar(range(len(df)), df["rmsf"], color=colors, width=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f"{row.resname}\n{row.resid}" for row in df.itertuples()],
        fontsize=6,
        rotation=90,
    )
    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(f"Protein Backbone RMSF — {result.sample_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if pocket_set:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(color="steelblue", label="Other residues"),
            Patch(color="darkorange", label="Binding site"),
        ]
        ax.legend(handles=legend_elements, fontsize="small")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_ligand_rmsf(
    result,  # RMSFResult
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of per-atom ligand RMSF, colour-coded by element.

    Parameters
    ----------
    result:
        :class:`~mdatools.analysis.rmsf.RMSFResult`.
    save_path:
        If given, save figure to this path.
    """
    df: pd.DataFrame = result.ligand_df
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Ligand RMSF — {result.sample_name} (no data)")
        return fig

    colors = [_ELEMENT_COLORS.get(str(e).upper(), _DEFAULT_COLOR) for e in df["element"]]

    fig, ax = plt.subplots(
        figsize=(max(6, len(df) * 0.4), 4), constrained_layout=True
    )
    ax.bar(range(len(df)), df["rmsf"], color=colors, width=0.8, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["atom"], fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Atom")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(f"Ligand RMSF — {result.sample_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Legend for elements present
    present_elements = df["element"].str.upper().unique()
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=_ELEMENT_COLORS.get(e, _DEFAULT_COLOR), label=e, edgecolor="grey")
        for e in sorted(present_elements)
    ]
    ax.legend(handles=legend_elements, fontsize="small", title="Element")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_rmsf_comparison(
    results: list,  # list[RMSFResult]
    pocket_resids: list[int] | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Overlay protein Cα RMSF across multiple replicas.

    Parameters
    ----------
    results:
        List of :class:`~mdatools.analysis.rmsf.RMSFResult` objects.
    pocket_resids:
        Residue IDs to shade in the background.
    save_path:
        If given, save figure to this path.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("RMSF Comparison (no data)")
        return fig

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    cmap = matplotlib.colormaps.get_cmap("tab10")

    for i, result in enumerate(results):
        df = result.protein_df
        if df.empty:
            continue
        color = cmap(i % 10)
        ax.plot(
            range(len(df)),
            df["rmsf"],
            label=result.sample_name,
            color=color,
            linewidth=1.5,
            alpha=0.85,
        )

    # Shade binding site residues using first result's DataFrame as reference
    if pocket_resids:
        ref_df = results[0].protein_df
        pocket_set = set(pocket_resids)
        for j, row in enumerate(ref_df.itertuples()):
            if int(row.resid) in pocket_set:
                ax.axvspan(j - 0.5, j + 0.5, color="orange", alpha=0.25, zorder=0)

    # Build x-tick labels from first non-empty result
    ref_df = next((r.protein_df for r in results if not r.protein_df.empty), pd.DataFrame())
    if not ref_df.empty:
        ax.set_xticks(range(len(ref_df)))
        ax.set_xticklabels(
            [f"{row.resname}{row.resid}" for row in ref_df.itertuples()],
            fontsize=6,
            rotation=90,
        )

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Protein RMSF Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize="small")

    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
