"""PCA and TICA trajectory dimensionality reduction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import MDAnalysis as mda
import numpy as np

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class DimRedResult:
    """Results from trajectory dimensionality reduction.

    Attributes
    ----------
    sample_name:
        Identifier for the replica/sample.
    method:
        Dimensionality reduction method used: ``'pca'`` or ``'tica'``.
    projection:
        Low-dimensional embedding, shape ``(n_frames, n_components)``.
        For TICA the first axis length is ``n_frames - lag`` (lagged pairs).
    explained_variance_ratio:
        Fraction of variance explained by each component (PCA only).
        Empty array for TICA.
    components:
        Principal/independent components in feature space,
        shape ``(n_components, n_features)``.
    feature_mean:
        Mean of the feature vectors used for centering, shape ``(n_features,)``.
    """

    sample_name: str
    method: str
    projection: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    feature_mean: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


class TrajectoryPCA:
    """PCA on Cα backbone or ligand heavy-atom coordinates.

    Implemented via numpy SVD — no scikit-learn required.

    Parameters
    ----------
    cfg:
        Analysis configuration.
    n_components:
        Number of principal components to retain.
    selection:
        MDAnalysis atom selection string for the atoms to use as features.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        n_components: int = 3,
        selection: str = "name CA",
    ) -> None:
        self.cfg = cfg
        self.n_components = n_components
        self.selection = selection

    def fit_transform(
        self, u: mda.Universe, sample_name: str = ""
    ) -> DimRedResult:
        """Run PCA on trajectory frames.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`DimRedResult`.

        Returns
        -------
        :class:`DimRedResult` with ``method='pca'``.
        """
        ag = u.select_atoms(self.selection)
        if len(ag) == 0:
            raise ValueError(
                f"Selection '{self.selection}' returned no atoms."
            )

        # Collect positions: (n_frames, n_atoms * 3)
        X = np.array(
            [ag.positions.copy().ravel() for _ in u.trajectory],
            dtype=np.float64,
        )

        n_components = min(self.n_components, X.shape[0], X.shape[1])
        projection, explained_variance_ratio, components, mean = _pca_numpy(
            X, n_components
        )

        logger.info(
            "PCA done for %s: %d frames, %d components (%.1f%% variance)",
            sample_name,
            len(X),
            n_components,
            float(explained_variance_ratio.sum() * 100),
        )
        return DimRedResult(
            sample_name=sample_name,
            method="pca",
            projection=projection,
            explained_variance_ratio=explained_variance_ratio,
            components=components,
            feature_mean=mean,
        )


# ---------------------------------------------------------------------------
# TICA
# ---------------------------------------------------------------------------


class TrajectoryTICA:
    """Time-lagged Independent Component Analysis on trajectory coordinates.

    Implemented with numpy/scipy eigendecomposition — no deeptime required.

    Parameters
    ----------
    cfg:
        Analysis configuration.
    lag:
        Lag time in frames for the time-lagged covariance matrix.
    n_components:
        Number of slow components to retain.
    selection:
        MDAnalysis atom selection string for the atoms to use as features.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        lag: int = 10,
        n_components: int = 3,
        selection: str = "name CA",
    ) -> None:
        if lag < 1:
            raise ValueError("lag must be >= 1.")
        self.cfg = cfg
        self.lag = lag
        self.n_components = n_components
        self.selection = selection

    def fit_transform(
        self, u: mda.Universe, sample_name: str = ""
    ) -> DimRedResult:
        """Run TICA on trajectory frames.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`DimRedResult`.

        Returns
        -------
        :class:`DimRedResult` with ``method='tica'``.
        Projection shape is ``(n_frames - lag, n_components)``.
        """
        ag = u.select_atoms(self.selection)
        if len(ag) == 0:
            raise ValueError(
                f"Selection '{self.selection}' returned no atoms."
            )

        X = np.array(
            [ag.positions.copy().ravel() for _ in u.trajectory],
            dtype=np.float64,
        )

        if self.lag >= len(X):
            logger.warning(
                "TICA skipped for %s: lag (%d) must be less than n_frames (%d). "
                "Reduce TICA_LAG or use a longer trajectory.",
                sample_name,
                self.lag,
                len(X),
            )
            return DimRedResult(
                sample_name=sample_name,
                method="tica",
                projection=np.empty((0, self.n_components)),
                components=np.empty((self.n_components, X.shape[1])),
                explained_variance_ratio=np.zeros(self.n_components),
                feature_mean=X.mean(axis=0),
            )

        n_components = min(self.n_components, X.shape[1], len(X) - self.lag)
        projection, components, mean = _tica_numpy(X, self.lag, n_components)

        logger.info(
            "TICA done for %s: %d frame pairs (lag=%d), %d components",
            sample_name,
            len(projection),
            self.lag,
            n_components,
        )
        return DimRedResult(
            sample_name=sample_name,
            method="tica",
            projection=projection,
            explained_variance_ratio=np.array([]),
            components=components,
            feature_mean=mean,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pca_numpy(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center X, compute SVD, return (projection, evr, components, mean)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # Economy SVD: U (n,k), s (k,), Vt (k,d)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = float(np.sum(s ** 2))
    evr = (s[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    projection = U[:, :n_components] * s[:n_components]
    components = Vt[:n_components]
    return projection, evr, components, mean


def _tica_numpy(
    X: np.ndarray, lag: int, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TICA via generalized eigenvalue problem.

    Solves  C_lag v = λ C_0 v  where:
    - C_0 is the instantaneous covariance matrix
    - C_lag is the time-lagged covariance matrix

    Returns (projection, components, mean).
    """
    mean = X.mean(axis=0)
    Xc = X - mean

    X0 = Xc[:-lag]   # (n-lag, d)
    Xt = Xc[lag:]    # (n-lag, d)

    n = len(X0)
    # Instantaneous covariance (symmetric)
    C0 = (X0.T @ X0) / n
    # Time-lagged covariance (symmetrised)
    Ct = (X0.T @ Xt) / n
    Ct = 0.5 * (Ct + Ct.T)

    # Regularise C0 for numerical stability
    eps = 1e-10 * np.trace(C0) / C0.shape[0]
    C0 += eps * np.eye(C0.shape[0])

    from scipy.linalg import eigh

    # eigh solves C_lag v = λ C_0 v and returns eigenvalues in ascending order
    eigenvalues, eigenvectors = eigh(Ct, C0)

    # Sort by descending eigenvalue (slowest modes first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components].T  # (n_components, d)
    projection = X0 @ eigenvectors[:, :n_components]  # (n-lag, n_components)

    return projection, components, mean
