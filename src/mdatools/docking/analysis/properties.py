"""Molecular property calculation — docking analysis namespace.

Re-exports from :mod:`mdatools.shared.properties`.  All new code should
import directly from that module.  This wrapper exists to satisfy the
``mdatools.docking.analysis.properties`` path used in the merged
library's ``docking`` namespace.
"""

from __future__ import annotations

# Re-export everything from the shared canonical implementation.
from mdatools.shared.properties import (  # noqa: F401
    LIPINSKI_HBA_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_MW_MAX,
    VEBER_ROTBONDS_MAX,
    VEBER_TPSA_MAX,
    MolecularProperties,
    add_properties_to_df,
    batch_from_smiles,
    calculate_properties,
)

__all__ = [
    "MolecularProperties",
    "calculate_properties",
    "add_properties_to_df",
    "batch_from_smiles",
    "LIPINSKI_MW_MAX",
    "LIPINSKI_LOGP_MAX",
    "LIPINSKI_HBD_MAX",
    "LIPINSKI_HBA_MAX",
    "VEBER_TPSA_MAX",
    "VEBER_ROTBONDS_MAX",
]
