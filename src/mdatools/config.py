"""Pydantic v2 configuration models for mdatools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ResidueGroup(BaseModel):
    """A named group of residues used for bonus scoring."""

    name: str
    resids: list[int] = Field(default_factory=list)
    resnames: list[str] = Field(default_factory=list)
    bonus: float = 20.0


class HBondConfig(BaseModel):
    """Parameters for HydrogenBondAnalysis."""

    d_a_cutoff: float = 3.5
    angle_cutoff: float = 150.0
    start: int | None = None
    stop: int | None = None
    step: int = 1


class RMSDConfig(BaseModel):
    """Parameters for RMSD calculation."""

    align_select: str = "protein and name CA"
    backbone_select: str = "backbone"
    start: int | None = None
    stop: int | None = None
    step: int = 1
    ref_frame: int = 0


class ScoringWeights(BaseModel):
    """Weights and thresholds for template scoring."""

    occ_weight: float = 50.0
    dist_weight: float = 30.0
    angle_weight: float = 20.0
    stability_weight: float = 15.0
    dist_ref: float = 3.5
    dist_range: float = 0.7
    angle_ref: float = 120.0
    angle_range: float = 60.0
    min_frame_percentile: float = 0.3


class AnalysisConfig(BaseModel):
    """Top-level configuration for an MD analysis run.

    Edit only this in the config cell of each notebook.
    """

    topology_glob: str = "equilibrating_topology.pdb"
    trajectory_glob: str = "trajectory.xtc"
    ligand_resname: str = "UNK"
    output_dir: Path = Path("./results")
    figures_dir: Path = Path("./figures")

    # Named groups of residues (e.g. ECD pocket) for bonus scoring
    residue_groups: dict[str, ResidueGroup] = Field(default_factory=dict)

    hbond: HBondConfig = Field(default_factory=HBondConfig)
    rmsd: RMSDConfig = Field(default_factory=RMSDConfig)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    dt_ns: float = 2.0
    n_frames: int | None = None

    pymol_executable: str = "pymol"
    ffmpeg_executable: str = "ffmpeg"

    # Contact fingerprint cutoff (Å)
    contact_cutoff: float = 4.5

    model_config = {"arbitrary_types_allowed": True}

    def make_dirs(self) -> None:
        """Create output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
