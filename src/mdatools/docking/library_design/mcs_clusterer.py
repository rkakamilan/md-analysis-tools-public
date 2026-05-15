"""Murcko-based MCS clustering for fragment library curation.

Groups molecules sharing a Murcko scaffold into clusters and computes
the maximum common substructure (MCS) per cluster as the canonical
"core scaffold".  This is the substructure-analysis stage of the
EU-OPENSCREEN EFSL design methodology
(Jalencas, Mestres et al., RSC Med. Chem. 2024, doi:10.1039/d3md00724c).

Reuses the project's existing :func:`mdatools.docking.analysis.sar.find_mcs`
wrapper around RDKit's :mod:`rdFMCS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class MCSClustererConfig:
    """Configuration for :class:`MCSClusterer`."""

    min_cluster_size: int = 1
    """Drop Murcko buckets with fewer than this many members.  ``1``
    keeps singletons (each becomes its own cluster); ``2`` requires at
    least two members; etc.  EFSL-style curation typically uses ``2``
    so singletons fall through to the diversity selection stage."""

    timeout_per_cluster_s: int = 5
    """Per-cluster ``rdFMCS`` timeout (s).  Large or chemically diverse
    Murcko buckets can hang the FMCS algorithm; the timeout caps each
    cluster's MCS computation."""

    ring_matches_ring_only: bool = True
    """Forwarded to ``rdFMCS.FindMCS``.  Constrains MCS atoms in rings
    to match only ring atoms — the canonical EFSL choice for fragment
    scaffolds."""

    complete_rings_only: bool = True
    """Forwarded to ``rdFMCS.FindMCS``.  Requires the MCS to contain
    complete rings rather than partial rings."""


@dataclass
class CoreScaffold:
    """A cluster of molecules sharing a Murcko scaffold + their MCS."""

    smarts: str
    """SMARTS string of the cluster's MCS, or empty when the cluster
    is a singleton (FMCS not run)."""

    member_indices: list[int]
    """0-indexed positions in the input mol list."""

    representative_smiles: str
    """Canonical SMILES of the first cluster member, used as the
    human-readable label for this scaffold."""

    murcko_smiles: str
    """Canonical Murcko scaffold SMILES that drove the bucketing.
    Empty when a member has no ring system."""

    n_atoms_mcs: int = 0
    """Number of atoms in the MCS pattern (0 for singletons)."""


class MCSClusterer:
    """Cluster molecules by shared Murcko scaffold and compute per-cluster MCS.

    Algorithm:
    1. For each input mol, compute its Murcko scaffold canonical SMILES.
    2. Group mols by Murcko SMILES (members sharing the same scaffold).
    3. For each group of size ≥ ``min_cluster_size``:
       a. If size > 1, run ``rdFMCS.FindMCS`` on the cluster members
          to get the canonical MCS SMARTS.
       b. If size == 1, skip FMCS (the Murcko scaffold itself is the
          canonical pattern).
    4. Return one :class:`CoreScaffold` per surviving cluster.

    Example
    -------
    ::

        clusterer = MCSClusterer()
        scaffolds = clusterer.cluster(mols)
        for s in scaffolds:
            print(f"{s.murcko_smiles}: {len(s.member_indices)} members")
    """

    def __init__(self, config: MCSClustererConfig | None = None) -> None:
        self.config = config or MCSClustererConfig()

    def cluster(self, mols: list[Chem.Mol]) -> list[CoreScaffold]:
        """Run the Murcko-bucket → per-cluster-MCS pipeline.

        Mols failing Murcko computation (e.g. fully acyclic molecules
        with no scaffold) are bucketed under the empty-scaffold key
        ``""`` and treated as a single cluster of "scaffoldless" mols.
        """
        if not mols:
            return []

        buckets: dict[str, list[int]] = {}
        for idx, mol in enumerate(mols):
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            murcko_smi = self._murcko_smiles(mol)
            buckets.setdefault(murcko_smi, []).append(idx)

        scaffolds: list[CoreScaffold] = []
        for murcko_smi, indices in buckets.items():
            if len(indices) < self.config.min_cluster_size:
                continue

            members = [mols[i] for i in indices]
            mcs_smarts, mcs_n_atoms = self._cluster_mcs(members)
            rep_smiles = Chem.MolToSmiles(members[0])

            scaffolds.append(
                CoreScaffold(
                    smarts=mcs_smarts,
                    member_indices=indices,
                    representative_smiles=rep_smiles,
                    murcko_smiles=murcko_smi,
                    n_atoms_mcs=mcs_n_atoms,
                )
            )

        # Stable order: largest cluster first (so callers can pick the
        # most populated scaffolds to keep in a fragment library).
        scaffolds.sort(key=lambda s: len(s.member_indices), reverse=True)
        return scaffolds

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _murcko_smiles(mol: Chem.Mol) -> str:
        """Canonical Murcko scaffold SMILES, or empty string if the mol
        has no ring system."""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            return ""
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold)

    def _cluster_mcs(self, members: list[Chem.Mol]) -> tuple[str, int]:
        """Compute MCS SMARTS for a cluster.  Singletons get the
        Murcko scaffold itself; otherwise run rdFMCS."""
        if len(members) == 1:
            scaffold = MurckoScaffold.GetScaffoldForMol(members[0])
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                return Chem.MolToSmarts(scaffold), scaffold.GetNumAtoms()
            return "", 0

        try:
            result = rdFMCS.FindMCS(
                members,
                ringMatchesRingOnly=self.config.ring_matches_ring_only,
                completeRingsOnly=self.config.complete_rings_only,
                timeout=self.config.timeout_per_cluster_s,
            )
        except Exception:
            return "", 0

        if result.canceled or result.numAtoms == 0:
            return "", 0
        return result.smartsString, result.numAtoms
