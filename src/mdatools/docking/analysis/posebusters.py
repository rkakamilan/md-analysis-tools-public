"""PoseBusters pose quality gate — docking analysis namespace.

Re-exports from :mod:`mdatools.shared.posebusters`.  All new code should
import directly from that module.  This wrapper exists to satisfy the
``mdatools.docking.analysis.posebusters`` path used in the merged
library's ``docking`` namespace.
"""

from __future__ import annotations

from mdatools.shared.posebusters import (  # noqa: F401
    PoseBustersResult,
    bust_rdkit_poses,
    bust_trajectory_poses,
    filter_rdkit_poses,
)

__all__ = [
    "PoseBustersResult",
    "bust_rdkit_poses",
    "bust_trajectory_poses",
    "filter_rdkit_poses",
]
