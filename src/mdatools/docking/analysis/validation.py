"""Docking pose validation — re-docking RMSD and grid box checks.

Provides utilities for benchmarking docking protocols via RMSD to
crystal-structure reference poses, and for verifying that docked poses
lie within the sampling grid box.

Ported from ``docking_analysis.analysis.validation``.

Example::

    from rdkit import Chem
    from mdatools.docking.analysis.validation import validate_redocking

    docked_mols = list(Chem.SDMolSupplier("poses.sdf"))
    reference = next(Chem.SDMolSupplier("crystal.sdf"))
    result = validate_redocking(docked_mols, reference)
    print(result.success, result.best_rmsd)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of re-docking RMSD validation.

    Attributes
    ----------
    best_rmsd:
        RMSD of the best-ranked pose to the reference (Å).
    success:
        ``True`` when *best_rmsd* ≤ *threshold*.
    best_pose_idx:
        Index in the input list of the best pose (lowest RMSD).
    all_rmsds:
        RMSD for every pose in the order provided.
    threshold:
        Threshold used to define success (Å).
    """

    best_rmsd: float
    success: bool
    best_pose_idx: int
    all_rmsds: list[float] = field(default_factory=list)
    threshold: float = 2.0


@dataclass
class BoxValidationResult:
    """Result of grid box containment check.

    Attributes
    ----------
    in_box:
        ``True`` when the ligand centroid is inside the strict grid box.
    in_extended:
        ``True`` when the centroid is within *extension* Å of the box
        boundary.
    distance:
        Distance from the centroid to the nearest box face (Å).
        Negative = inside the box.
    """

    in_box: bool
    in_extended: bool
    distance: float


def rmsd_to_reference(
    pose: Any,
    reference: Any,
    align: bool = False,
) -> float | None:
    """Compute RMSD between a docked pose and a reference molecule.

    Uses RDKit's ``GetBestRMS`` (with automorphism correction) when
    *align* is ``True``, or the symmetric RMSD without alignment otherwise.

    Parameters
    ----------
    pose:
        Docked pose RDKit Mol with 3-D coordinates.
    reference:
        Crystal structure reference RDKit Mol with 3-D coordinates.
    align:
        When ``True``, find the minimum RMSD over all automorphisms
        and with re-alignment.  Default ``False`` for faster evaluation
        of already-aligned poses.

    Returns
    -------
    float or None
        RMSD in Å, or ``None`` when the atoms cannot be matched.
    """
    try:
        from rdkit.Chem import AllChem, rdMolAlign  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("RDKit is required for RMSD calculation.") from exc

    try:
        if align:
            rmsd = float(AllChem.GetBestRMS(reference, pose))
        else:
            rmsd = float(rdMolAlign.CalcRMS(pose, reference))
        return rmsd
    except Exception as exc:  # noqa: BLE001
        logger.debug("RMSD calculation failed: %s", exc)
        return None


def validate_redocking(
    poses: list,
    reference: Any,
    threshold: float = 2.0,
    align: bool = False,
) -> ValidationResult:
    """Validate re-docking success rate across a list of poses.

    Parameters
    ----------
    poses:
        List of docked RDKit Mols.
    reference:
        Crystal structure Mol.
    threshold:
        RMSD cutoff for calling a pose successful (Å, default 2.0).
    align:
        Pass to :func:`rmsd_to_reference`.

    Returns
    -------
    ValidationResult
    """
    rmsds: list[float] = []
    for pose in poses:
        r = rmsd_to_reference(pose, reference, align=align)
        rmsds.append(r if r is not None else float("inf"))

    if not rmsds:
        return ValidationResult(
            best_rmsd=float("inf"),
            success=False,
            best_pose_idx=-1,
            all_rmsds=[],
            threshold=threshold,
        )

    best_idx = int(min(range(len(rmsds)), key=lambda i: rmsds[i]))
    best_rmsd = rmsds[best_idx]

    return ValidationResult(
        best_rmsd=best_rmsd,
        success=best_rmsd <= threshold,
        best_pose_idx=best_idx,
        all_rmsds=rmsds,
        threshold=threshold,
    )


def validate_pose_in_box(
    pose: Any,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    extension: float = 2.0,
) -> BoxValidationResult:
    """Check whether the ligand centroid lies within a docking grid box.

    Parameters
    ----------
    pose:
        RDKit Mol with a 3-D conformer.
    center:
        Grid box center ``(x, y, z)`` in Å.
    size:
        Grid box dimensions ``(sx, sy, sz)`` in Å.
    extension:
        Tolerance buffer around the box boundary (Å).

    Returns
    -------
    BoxValidationResult
    """
    import numpy as np  # noqa: PLC0415

    conf = pose.GetConformer()
    positions = conf.GetPositions()
    centroid = positions.mean(axis=0)

    cx, cy, cz = center
    sx, sy, sz = size
    half = np.array([sx, sy, sz]) / 2.0
    lo = np.array([cx, cy, cz]) - half
    hi = np.array([cx, cy, cz]) + half

    # Distance to nearest face (negative = inside)
    dx = max(lo[0] - centroid[0], 0.0, centroid[0] - hi[0])
    dy = max(lo[1] - centroid[1], 0.0, centroid[1] - hi[1])
    dz = max(lo[2] - centroid[2], 0.0, centroid[2] - hi[2])
    dist = float(np.sqrt(dx**2 + dy**2 + dz**2))

    in_box = dist <= 0.0
    in_extended = dist <= extension

    return BoxValidationResult(
        in_box=in_box,
        in_extended=in_extended,
        distance=dist,
    )
