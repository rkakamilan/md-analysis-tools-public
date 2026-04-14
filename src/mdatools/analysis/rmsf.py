"""RMSF calculation — per-residue protein and per-atom ligand fluctuation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class RMSFResult:
    """Results from RMSF analysis.

    Attributes
    ----------
    sample_name:
        Identifier for the replica/sample.
    protein_df:
        Per-residue Cα RMSF. Columns: resid, resname, rmsf.
    ligand_df:
        Per-atom ligand RMSF. Columns: atom, element, rmsf.
    """

    sample_name: str
    protein_df: pd.DataFrame  # columns: resid, resname, rmsf
    ligand_df: pd.DataFrame  # columns: atom, element, rmsf


class RMSFAnalyzer:
    """Compute per-residue protein RMSF and per-atom ligand RMSF.

    Aligns trajectory on backbone Cα atoms before computing fluctuations
    so that global translational/rotational motion is removed.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def run(self, u: mda.Universe, sample_name: str = "") -> RMSFResult:
        """Compute full-protein and ligand RMSF.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`RMSFResult`.

        Returns
        -------
        RMSFResult
        """
        ca_atoms = u.select_atoms("name CA")
        ligand_atoms = u.select_atoms(f"resname {self.cfg.ligand_resname}")

        protein_df = self._compute_rmsf(u, ca_atoms, resid_level=True)
        ligand_df = self._compute_rmsf(u, ligand_atoms, resid_level=False)

        logger.info(
            "RMSF done for %s: %d residues, %d ligand atoms",
            sample_name,
            len(protein_df),
            len(ligand_df),
        )
        return RMSFResult(
            sample_name=sample_name,
            protein_df=protein_df,
            ligand_df=ligand_df,
        )

    def run_binding_site(
        self,
        u: mda.Universe,
        pocket_resids: list[int],
        sample_name: str = "",
    ) -> RMSFResult:
        """Compute RMSF restricted to binding site residues.

        Parameters
        ----------
        u:
            MDAnalysis Universe.
        pocket_resids:
            Residue IDs that define the binding site.
        sample_name:
            Label stored in the returned :class:`RMSFResult`.
        """
        if not pocket_resids:
            return RMSFResult(
                sample_name=sample_name,
                protein_df=pd.DataFrame(columns=["resid", "resname", "rmsf"]),
                ligand_df=self._compute_rmsf(
                    u,
                    u.select_atoms(f"resname {self.cfg.ligand_resname}"),
                    resid_level=False,
                ),
            )
        resid_sel = " or ".join(f"resid {r}" for r in pocket_resids)
        ca_atoms = u.select_atoms(f"name CA and ({resid_sel})")
        ligand_atoms = u.select_atoms(f"resname {self.cfg.ligand_resname}")

        protein_df = self._compute_rmsf(u, ca_atoms, resid_level=True)
        ligand_df = self._compute_rmsf(u, ligand_atoms, resid_level=False)

        logger.info(
            "Binding-site RMSF done for %s: %d residues, %d ligand atoms",
            sample_name,
            len(protein_df),
            len(ligand_df),
        )
        return RMSFResult(
            sample_name=sample_name,
            protein_df=protein_df,
            ligand_df=ligand_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rmsf(
        u: mda.Universe,
        atom_group: mda.AtomGroup,
        resid_level: bool,
    ) -> pd.DataFrame:
        """Run ``rms.RMSF`` on *atom_group* and return a tidy DataFrame.

        Parameters
        ----------
        resid_level:
            If ``True`` return one row per residue (Cα usage); if ``False``
            return one row per atom (ligand usage).
        """
        if len(atom_group) == 0:
            if resid_level:
                return pd.DataFrame(columns=["resid", "resname", "rmsf"])
            return pd.DataFrame(columns=["atom", "element", "rmsf"])

        rmsf_calc = rms.RMSF(atom_group)
        rmsf_calc.run()
        values: np.ndarray = rmsf_calc.results.rmsf

        if resid_level:
            rows = [
                {
                    "resid": int(atom.resid),
                    "resname": atom.resname,
                    "rmsf": float(v),
                }
                for atom, v in zip(atom_group, values)
            ]
            return pd.DataFrame(rows)
        else:
            # Determine element: first character of atom name unless explicitly set
            def _elem(atom: mda.core.groups.Atom) -> str:
                try:
                    e = atom.element
                    if e:
                        return e
                except Exception:
                    pass
                return atom.name[0].upper()

            rows = [
                {
                    "atom": atom.name,
                    "element": _elem(atom),
                    "rmsf": float(v),
                }
                for atom, v in zip(atom_group, values)
            ]
            return pd.DataFrame(rows)


def run_rmsf_batch(
    replica_dirs: list,
    cfg: AnalysisConfig,
) -> dict[str, RMSFResult]:
    """Run RMSF analysis for all replicas.

    Parameters
    ----------
    replica_dirs:
        List of directories containing topology + trajectory files
        matched by ``cfg.topology_glob`` / ``cfg.trajectory_glob``.
    cfg:
        Analysis configuration.

    Returns
    -------
    dict mapping *sample_name* → :class:`RMSFResult`.
    """
    from ..io.loaders import discover_replicas
    from ..universe import load_and_align

    replicas = discover_replicas(replica_dirs, cfg)
    analyzer = RMSFAnalyzer(cfg)
    results: dict[str, RMSFResult] = {}
    for rep in replicas:
        u = load_and_align(rep["topology"], rep["trajectory"], cfg)
        results[rep["name"]] = analyzer.run(u, rep["name"])
    return results
