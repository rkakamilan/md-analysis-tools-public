"""Generic SDF pose reader.

Reads pre-converted ``.sdf`` files where each molecule record is a single
docked pose.  Docking scores are expected in the ``docking_score`` SDF
property; if absent, ``0.0`` is used.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem, RDLogger

from mdatools.docking._typing import DockingResult
from mdatools.docking.io.base import PoseReader

RDLogger.DisableLog("rdApp.warning")


class SDFPoseReader(PoseReader):
    """Read docked poses from a ``.sdf`` file."""

    def read(self, path: str | Path) -> DockingResult:
        """Parse an SDF file where each record is one docked pose.

        Parameters
        ----------
        path:
            Path to the ``.sdf`` file.

        Returns
        -------
        DockingResult

        Raises
        ------
        ValueError
            If the file contains no valid molecules.
        """
        path = Path(path)

        try:
            supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        except OSError as e:
            raise ValueError(f"No valid molecules found in {path}") from e
        poses = [m for m in supplier if m is not None]

        if not poses:
            raise ValueError(f"No valid molecules found in {path}")

        scores: list[float] = []
        for mol in poses:
            try:
                scores.append(float(mol.GetProp("docking_score")))
            except KeyError:
                scores.append(0.0)

        return DockingResult(
            poses=poses,
            scores=scores,
            source_file=path,
            backend="sdf",
        )
