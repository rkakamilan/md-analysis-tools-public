"""Pocket environment profiler for protein-ligand snapshots.

Computes per-heavy-atom metrics for each ligand atom:
  - d_min     : distance to nearest protein heavy atom (Å)
  - d_margin  : d_min minus sum of VdW radii (contact slack)
  - nearest_e : element of nearest protein atom
  - cone_dist : min distance to protein within an outward-pointing cone
  - hydrophob : fraction of C atoms in local environment sphere
  - n_NO      : count of N/O atoms within ENV_RADIUS
  - n_local   : total protein heavy atoms within ENV_RADIUS

Reusable, configurable class.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_RADIUS: float = 5.0      # Å – local environment sphere radius
CONE_ANGLE_DEG: float = 30.0  # ° – half-angle of outward cone

VDW_RADII: dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "F": 1.47, "P": 1.80, "CL": 1.75, "BR": 1.85,
}
VDW_DEFAULT: float = 1.70


# ---------------------------------------------------------------------------
# PDB parsing helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip trailing 'x' from atom names that use digit+x suffix style naming."""
    n = name.strip()
    if n.endswith("x") and len(n) >= 2 and n[-2].isdigit():
        return n[:-1]
    return n


def _is_hydrogen(name: str, element: str) -> bool:
    if element.strip().upper().startswith("H"):
        return True
    n = name.strip()
    if n.startswith("H"):
        return True
    if len(n) >= 2 and n[0].isdigit() and n[1].upper() == "H":
        return True
    return False


def _parse_element(line: str) -> str:
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if not elem:
        a = line[12:16].strip().lstrip("0123456789")
        elem = a[0].upper() if a else "X"
    return elem.upper()


def parse_clean_pdb(
    pdb_path: Path,
    ligand_resname: str = "UNK",
) -> tuple[list[tuple], list[tuple]]:
    """Parse a clean (solvent-stripped) PDB and return heavy-atom arrays.

    Parameters
    ----------
    pdb_path:
        Path to the PDB file.
    ligand_resname:
        Residue name of the ligand (default ``"UNK"``).

    Returns
    -------
    prot_atoms:
        List of ``(x, y, z, element)`` tuples for protein heavy atoms.
    lig_atoms:
        List of ``(serial, canon_name, x, y, z, element)`` tuples for
        ligand heavy atoms.
    """
    prot_atoms: list[tuple] = []
    lig_atoms: list[tuple] = []

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ATOM":
                elem = _parse_element(line)
                aname = line[12:16].strip()
                if _is_hydrogen(aname, elem):
                    continue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                prot_atoms.append((x, y, z, elem))
            elif rec == "HETATM":
                if line[17:20].strip() != ligand_resname:
                    continue
                elem = _parse_element(line)
                aname = line[12:16].strip()
                serial = int(line[6:11])
                if _is_hydrogen(aname, elem):
                    continue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                lig_atoms.append((serial, _normalize_name(aname), x, y, z, elem))

    return prot_atoms, lig_atoms


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    prot_atoms: list[tuple],
    lig_atoms: list[tuple],
    env_radius: float = ENV_RADIUS,
    cone_angle_deg: float = CONE_ANGLE_DEG,
) -> list[dict]:
    """Compute pocket environment metrics for each ligand heavy atom.

    Parameters
    ----------
    prot_atoms:
        Output of :func:`parse_clean_pdb` – ``[(x, y, z, element), ...]``.
    lig_atoms:
        Output of :func:`parse_clean_pdb` –
        ``[(serial, name, x, y, z, element), ...]``.
    env_radius:
        Radius of local environment sphere (Å).
    cone_angle_deg:
        Half-angle of the outward-pointing cone for ``cone_dist`` (°).

    Returns
    -------
    List of metric dicts with keys:
    ``atom``, ``element``, ``d_min``, ``d_margin``, ``nearest_e``,
    ``cone_dist``, ``hydrophob``, ``n_NO``, ``n_local``.
    """
    if not prot_atoms or not lig_atoms:
        return []

    prot_xyz = np.array([(a[0], a[1], a[2]) for a in prot_atoms])
    prot_elem = [a[3] for a in prot_atoms]
    lig_xyz = np.array([(a[2], a[3], a[4]) for a in lig_atoms])
    lig_names = [a[1] for a in lig_atoms]
    lig_elems = [a[5] for a in lig_atoms]

    lig_centroid = lig_xyz.mean(axis=0)
    cone_cos = np.cos(np.radians(cone_angle_deg))

    def _vdw(e: str) -> float:
        return VDW_RADII.get(e.upper(), VDW_DEFAULT)

    results: list[dict] = []
    for i, (name, elem) in enumerate(zip(lig_names, lig_elems)):
        pos = lig_xyz[i]
        diff = prot_xyz - pos
        dists = np.linalg.norm(diff, axis=1)

        j_min = int(np.argmin(dists))
        d_min = float(dists[j_min])
        near_e = prot_elem[j_min]
        d_margin = d_min - (_vdw(elem) + _vdw(near_e))

        mask = dists <= env_radius
        loc_elems = [prot_elem[j] for j in range(len(prot_elem)) if mask[j]]
        n_local = len(loc_elems)
        n_C = sum(1 for e in loc_elems if e == "C")
        n_NO = sum(1 for e in loc_elems if e in ("N", "O"))
        hydrophob = n_C / n_local if n_local > 0 else 0.0

        ext_dir = pos - lig_centroid
        ext_norm = np.linalg.norm(ext_dir)
        if ext_norm < 1e-6:
            cone_dist = d_min
        else:
            ext_dir = ext_dir / ext_norm
            dn = np.linalg.norm(diff, axis=1, keepdims=True)
            dn = np.where(dn < 1e-8, 1e-8, dn)
            cos = (diff / dn) @ ext_dir
            cm = cos >= cone_cos
            cone_dist = float(dists[cm].min()) if cm.any() else float(dists.min())

        results.append({
            "atom":      name,
            "element":   elem,
            "d_min":     round(d_min, 2),
            "d_margin":  round(d_margin, 2),
            "nearest_e": near_e,
            "cone_dist": round(cone_dist, 2),
            "hydrophob": round(hydrophob, 3),
            "n_NO":      int(n_NO),
            "n_local":   int(n_local),
        })

    return results


# ---------------------------------------------------------------------------
# 2D mol builder (requires RDKit)
# ---------------------------------------------------------------------------

def build_2d_mol(pdb_path: Path, ligand_resname: str = "UNK"):
    """Extract ligand from PDB, determine bonds, and generate 2D coordinates.

    Requires RDKit (``pip install 'mdatools[pocket]'``).

    Parameters
    ----------
    pdb_path:
        Path to the snapshot PDB file.
    ligand_resname:
        Residue name of the ligand.

    Returns
    -------
    mol:
        RDKit Mol with 2D coordinates (H removed).
    name2idx:
        Mapping from canonical atom name → RDKit atom index.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds, AllChem

    lines: list[str] = []
    serials: set[int] = set()

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "HETATM" and line[17:20].strip() == ligand_resname:
                serials.add(int(line[6:11]))
                lines.append(line)
            elif rec == "CONECT":
                nums_raw = [line[i:i + 5].strip() for i in range(6, min(len(line.rstrip()), 36), 5)]
                nums = [int(n) for n in nums_raw if n.isdigit()]
                if nums and nums[0] in serials:
                    lines.append(line)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name

    try:
        raw = Chem.MolFromPDBFile(tmp_path, removeHs=False, sanitize=False)
        if raw is None:
            raise ValueError(f"RDKit could not parse ligand from {pdb_path}")
        rdDetermineBonds.DetermineBonds(raw, charge=0)
        Chem.SanitizeMol(raw)
        mol = Chem.RemoveAllHs(raw)
        AllChem.Compute2DCoords(mol)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    name2idx: dict[str, int] = {}
    for i in range(mol.GetNumAtoms()):
        mi = mol.GetAtomWithIdx(i).GetMonomerInfo()
        if mi:
            name2idx[_normalize_name(mi.GetName().strip())] = i

    return mol, name2idx


# ---------------------------------------------------------------------------
# High-level PocketProfiler class
# ---------------------------------------------------------------------------

class PocketProfiler:
    """Profile pocket environment for one or more snapshot PDB files.

    Parameters
    ----------
    ligand_resname:
        Residue name of the ligand in the PDB files (default ``"UNK"``).
    env_radius:
        Local environment sphere radius in Å (default ``5.0``).
    cone_angle_deg:
        Half-angle of outward-pointing cone for ``cone_dist`` (default ``30.0``).
    """

    def __init__(
        self,
        ligand_resname: str = "UNK",
        env_radius: float = ENV_RADIUS,
        cone_angle_deg: float = CONE_ANGLE_DEG,
    ) -> None:
        self.ligand_resname = ligand_resname
        self.env_radius = env_radius
        self.cone_angle_deg = cone_angle_deg

    def profile(self, pdb_path: Path) -> list[dict]:
        """Compute pocket metrics for a single snapshot PDB.

        Parameters
        ----------
        pdb_path:
            Path to a clean (solvent-stripped) snapshot PDB.

        Returns
        -------
        List of per-atom metric dicts.
        """
        prot, lig = parse_clean_pdb(pdb_path, ligand_resname=self.ligand_resname)
        if not lig:
            raise ValueError(
                f"No ligand atoms found with resname '{self.ligand_resname}' in {pdb_path}"
            )
        return compute_metrics(prot, lig, self.env_radius, self.cone_angle_deg)

    def profile_batch(
        self,
        pdb_paths: list[Path],
        output_dir: Path | None = None,
    ) -> dict[str, list[dict]]:
        """Profile multiple snapshot PDBs.

        Parameters
        ----------
        pdb_paths:
            List of snapshot PDB paths.
        output_dir:
            If given, save per-snapshot CSV files here.

        Returns
        -------
        Dict mapping PDB stem → list of metric dicts.
        """
        results: dict[str, list[dict]] = {}
        for pdb_path in pdb_paths:
            pdb_path = Path(pdb_path)
            logger.info("Profiling %s", pdb_path.name)
            metrics = self.profile(pdb_path)
            results[pdb_path.stem] = metrics

            if output_dir is not None:
                import pandas as pd
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"pocket_{pdb_path.stem}.csv"
                pd.DataFrame(metrics).sort_values("d_min").reset_index(drop=True).to_csv(
                    csv_path, index=False
                )
                logger.info("  CSV → %s", csv_path.name)

        return results
