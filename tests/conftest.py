"""Shared pytest fixtures using synthetic MDAnalysis Universes.

No real topology/trajectory files are required.
"""

from __future__ import annotations

import numpy as np
import pytest

import MDAnalysis as mda
from MDAnalysis.core.universe import Universe

_MASS_MAP = {
    "C": 12.011, "N": 14.007, "O": 15.999,
    "H": 1.008,  "F": 18.998, "S": 32.06,
}


def _element_mass(name: str) -> float:
    return _MASS_MAP.get(name[0], 12.0)


def _make_universe(n_frames: int = 20) -> Universe:
    """Build a minimal synthetic Universe with protein + ligand atoms.

    Includes masses and explicit H–heavy bonds so that MDAnalysis
    ``bonded`` selections and ``rms.RMSD`` work correctly.
    """
    # Atom list: protein residues 1-5 + ligand residue 0 (resname UNK)
    atom_names = [
        # res1 ALA
        "N", "CA", "C", "O", "H",
        # res2 GLY
        "N", "CA", "C", "O", "H",
        # res3 ASN
        "N", "CA", "C", "O", "ND2", "OD1", "H", "HD21",
        # res4 SER
        "N", "CA", "C", "O", "OG", "H",
        # res5 TRP
        "N", "CA", "C", "O", "H",
        # ligand UNK (resid 0)
        "N4", "H4", "C1", "C2", "O1", "F1",
    ]
    resnames_per_atom = (
        ["ALA"] * 5 + ["GLY"] * 5 + ["ASN"] * 8 + ["SER"] * 6 + ["TRP"] * 5
        + ["UNK"] * 6
    )
    resids_per_atom = (
        [1] * 5 + [2] * 5 + [3] * 8 + [4] * 6 + [5] * 5 + [0] * 6
    )
    n_total = len(atom_names)
    n_residues = 6  # 5 protein + 1 ligand

    u = mda.Universe.empty(
        n_total,
        n_residues=n_residues,
        n_segments=1,
        atom_resindex=resids_per_atom,
        residue_segindex=[0] * n_residues,
        trajectory=True,
    )

    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("resnames", ["ALA", "GLY", "ASN", "SER", "TRP", "UNK"])
    u.add_TopologyAttr("resids", [1, 2, 3, 4, 5, 0])
    u.add_TopologyAttr("segids", ["A"])
    u.add_TopologyAttr("masses", [_element_mass(n) for n in atom_names])

    # Build bonds: H → heavy atom within same residue
    # (H is always the atom immediately after the heavy atom it bonds to)
    bonds = []
    # res1 ALA: H(4) – N(0)
    bonds.append((0, 4))   # N-H
    # res2 GLY: H(9) – N(5)
    bonds.append((5, 9))
    # res3 ASN: H(16) – N(10), HD21(17) – ND2(14)
    bonds.append((10, 16))
    bonds.append((14, 17))
    # res4 SER: H(21) – N(18)
    bonds.append((18, 21))  # wait — let me count properly
    # res5 TRP: H(27) – N(23)

    # Recalculate atom offsets
    # res1 ALA  idx 0-4:  N(0) CA(1) C(2) O(3) H(4)   → H(4)-N(0)
    # res2 GLY  idx 5-9:  N(5) CA(6) C(7) O(8) H(9)   → H(9)-N(5)
    # res3 ASN  idx10-17: N(10) CA(11) C(12) O(13) ND2(14) OD1(15) H(16) HD21(17)
    #           → H(16)-N(10), HD21(17)-ND2(14)
    # res4 SER  idx18-23: N(18) CA(19) C(20) O(21) OG(22) H(23)
    #           → H(23)-N(18)
    # res5 TRP  idx24-28: N(24) CA(25) C(26) O(27) H(28)
    #           → H(28)-N(24)
    # UNK       idx29-34: N4(29) H4(30) C1(31) C2(32) O1(33) F1(34)
    #           → H4(30)-N4(29)
    bonds = [
        (0, 4),   # ALA N-H
        (5, 9),   # GLY N-H
        (10, 16), # ASN N-H
        (14, 17), # ASN ND2-HD21
        (18, 23), # SER N-H
        (24, 28), # TRP N-H
        (29, 30), # UNK N4-H4
    ]
    u.add_TopologyAttr("bonds", bonds)

    # Multi-frame in-memory trajectory
    rng = np.random.default_rng(42)
    base_coords = rng.uniform(-10, 10, (n_total, 3)).astype(np.float32)
    # Place ligand N4 near ASN OD1 so H-bond detection can in principle work
    asn_od1_idx = 15  # OD1
    unk_n4_idx  = 29  # N4
    base_coords[unk_n4_idx] = base_coords[asn_od1_idx] + np.array([3.2, 0.1, 0.1])
    base_coords[30] = base_coords[unk_n4_idx] + np.array([1.0, 0.0, 0.0])  # H4

    traj = np.stack(
        [base_coords + rng.normal(0, 0.05, base_coords.shape) for _ in range(n_frames)],
    ).astype(np.float32)

    from MDAnalysis.coordinates.memory import MemoryReader
    u.load_new(traj, format=MemoryReader)
    return u


@pytest.fixture(scope="session")
def synthetic_universe() -> Universe:
    return _make_universe()


@pytest.fixture(scope="session")
def simple_cfg():
    """A minimal AnalysisConfig with temporary directories."""
    import tempfile
    from pathlib import Path
    from mdatools.config import AnalysisConfig

    tmp = tempfile.mkdtemp()
    return AnalysisConfig(
        ligand_resname="UNK",
        output_dir=Path(tmp) / "results",
        figures_dir=Path(tmp) / "figures",
    )
