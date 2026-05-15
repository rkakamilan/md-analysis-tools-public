"""Abstract I/O interface and backend factory.

All docking backends return a :class:`~mdatools.docking._typing.DockingResult`
so that downstream analysis code is independent of file format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mdatools.docking._typing import DockingResult


class PoseReader(ABC):
    """Abstract reader for docking output files."""

    @abstractmethod
    def read(self, path: str | Path) -> DockingResult:
        """Read a docking output file and return a standardised result.

        Parameters
        ----------
        path:
            Path to the docking output file.

        Returns
        -------
        DockingResult
            Poses as individual RDKit Mols with parallel score list.
        """


def get_reader(backend: str) -> PoseReader:
    """Return the appropriate :class:`PoseReader` for the given backend.

    Parameters
    ----------
    backend:
        One of ``"vina"``, ``"sdf"``, ``"glide"``, ``"unidock"``,
        ``"unidock2"``.

    Returns
    -------
    PoseReader

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    """
    from mdatools.docking.io.vina import VinaPoseReader
    from mdatools.docking.io.sdf import SDFPoseReader
    from mdatools.docking.io.glide import GlidePoseReader
    from mdatools.docking.io.unidock2 import UniDock2PoseReader

    readers: dict[str, PoseReader] = {
        "vina": VinaPoseReader(),
        "sdf": SDFPoseReader(),
        "glide": GlidePoseReader(),
        # UniDock outputs PDBQT (same format as Vina)
        "unidock": VinaPoseReader(),
        "unidock2": UniDock2PoseReader(),
    }

    if backend not in readers:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(readers)}"
        )
    return readers[backend]
