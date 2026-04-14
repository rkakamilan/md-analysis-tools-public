"""Plotting functions for PCA/TICA dimensionality reduction results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.dimensionality import DimRedResult


def plot_pca_landscape(
    result: DimRedResult,
    color_by: np.ndarray | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """2D scatter of PC1 vs PC2 (or IC1 vs IC2 for TICA).

    Parameters
    ----------
    result:
        :class:`DimRedResult` from :class:`TrajectoryPCA` or :class:`TrajectoryTICA`.
    color_by:
        Array of length ``n_frames`` used to colour points (e.g. cluster labels,
        time indices). If ``None``, points are coloured by frame index.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    proj = result.projection
    if proj.shape[1] < 2:
        raise ValueError("projection must have at least 2 components for a landscape plot.")

    if color_by is None:
        color_by = np.arange(len(proj))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_by, cmap="viridis", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Frame" if color_by is None else "Value")

    prefix = "PC" if result.method == "pca" else "IC"
    ax.set_xlabel(f"{prefix}1")
    ax.set_ylabel(f"{prefix}2")
    ax.set_title(f"{result.method.upper()} landscape — {result.sample_name}")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_explained_variance(
    result: DimRedResult,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Scree plot showing explained variance ratio per principal component.

    Only meaningful for PCA results (``result.method == 'pca'``).

    Parameters
    ----------
    result:
        :class:`DimRedResult` from :class:`TrajectoryPCA`.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    evr = result.explained_variance_ratio
    if len(evr) == 0:
        raise ValueError("explained_variance_ratio is empty — only available for PCA.")

    n = len(evr)
    cumulative = np.cumsum(evr)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, evr * 100, color="steelblue", alpha=0.8, label="Individual")
    ax.plot(x, cumulative * 100, "o-", color="darkorange", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"PCA scree plot — {result.sample_name}")
    ax.legend()
    ax.set_xticks(x)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_pca_projection_time(
    result: DimRedResult,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """PC1 and PC2 (or IC1/IC2) as time series.

    Parameters
    ----------
    result:
        :class:`DimRedResult` from PCA or TICA.
    save_path:
        If provided, save the figure to this path.
    dpi:
        Resolution for saved figure.
    """
    proj = result.projection
    if proj.shape[1] < 2:
        raise ValueError("projection must have at least 2 components.")

    frames = np.arange(len(proj))
    prefix = "PC" if result.method == "pca" else "IC"

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(frames, proj[:, i], lw=0.8, color=f"C{i}")
        ax.set_ylabel(f"{prefix}{i + 1}")
        ax.axhline(0, color="gray", lw=0.5, ls="--")

    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"{result.method.upper()} time series — {result.sample_name}")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
