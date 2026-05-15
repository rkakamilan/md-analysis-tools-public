"""Shared data types used across the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem


@dataclass
class DockingResult:
    """Standardized output of any PoseReader.

    All backend-specific readers return this type, allowing downstream
    analysis code to remain independent of the docking backend.

    Attributes
    ----------
    poses:
        One RDKit Mol per pose, each with exactly one conformer.
    scores:
        Docking scores parallel to *poses* (lower = better for Vina kcal/mol).
    metadata:
        Backend-specific extras (e.g. n_modes, pose_data dict from meeko).
    source_file:
        Path to the file that was read.
    backend:
        Identifier string: ``"vina"``, ``"glide"``, or ``"sdf"``.
    """

    poses: list[Chem.Mol]
    scores: list[float]
    source_file: Path
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)
