"""Protein-ligand interaction fingerprint analysis via ProLIF."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import pandas as pd

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)

# Default interaction types to detect
_DEFAULT_INTERACTIONS = [
    "HBDonor",
    "HBAcceptor",
    "Hydrophobic",
    "PiStacking",
    "PiCation",
    "CationPi",
    "Anionic",
    "Cationic",
    "VdWContact",
]


@dataclass
class InteractionResult:
    """Results from interaction fingerprint analysis.

    Attributes
    ----------
    sample_name:
        Identifier for the replica/sample.
    df:
        Per-frame fingerprint. Rows = frames, columns = interaction labels
        formatted as ``ResName-ChainID-ResID_InteractionType`` (e.g.
        ``ASN-A-58_HBAcceptor``). Values are 0/1 (or counts when
        ``count=True``).
    summary:
        Per-interaction occupancy. Columns: ``interaction``,
        ``occupancy_pct``.
    """

    sample_name: str
    df: pd.DataFrame
    summary: pd.DataFrame


class InteractionAnalyzer:
    """Protein-Ligand Interaction Fingerprint using ProLIF.

    Wraps ``prolif.Fingerprint`` to compute per-frame interaction vectors.
    ProLIF is an optional dependency; install it with::

        pip install mdatools[interactions]

    Interaction types detected by default:
    HBDonor, HBAcceptor, Hydrophobic, PiStacking, PiCation, CationPi,
    Anionic, Cationic, VdWContact.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        interactions: list[str] | None = None,
        count: bool = False,
    ) -> None:
        self.cfg = cfg
        self.interactions = interactions or _DEFAULT_INTERACTIONS
        self.count = count

    def run(self, u: mda.Universe, sample_name: str = "") -> InteractionResult:
        """Compute per-frame interaction fingerprint.

        Parameters
        ----------
        u:
            MDAnalysis Universe with a loaded trajectory.
        sample_name:
            Label stored in the returned :class:`InteractionResult`.

        Returns
        -------
        InteractionResult
        """
        try:
            import prolif as plf
        except ImportError as exc:
            raise ImportError(
                "ProLIF is required for InteractionAnalyzer. "
                "Install it with: pip install mdatools[interactions]"
            ) from exc

        ligand_ag = u.select_atoms(f"resname {self.cfg.ligand_resname}")
        protein_ag = u.select_atoms("protein")

        if len(ligand_ag) == 0:
            raise ValueError(
                f"Ligand '{self.cfg.ligand_resname}' not found in universe."
            )
        if len(protein_ag) == 0:
            raise ValueError("No protein atoms found in universe.")

        fp = plf.Fingerprint(interactions=self.interactions, count=self.count)
        fp.run(u.trajectory, ligand_ag, protein_ag)

        raw_df: pd.DataFrame = fp.to_dataframe()
        flat_df = self._flatten_columns(raw_df)
        summary = self._compute_summary(flat_df)

        logger.info(
            "Interaction fingerprint done for %s: %d frames, %d interaction columns",
            sample_name,
            len(flat_df),
            len(flat_df.columns),
        )
        return InteractionResult(
            sample_name=sample_name,
            df=flat_df,
            summary=summary,
        )

    def run_batch(
        self,
        replica_dirs: list[Path | str],
        cfg: AnalysisConfig,
    ) -> list[InteractionResult]:
        """Run interaction fingerprint for all replicas.

        Parameters
        ----------
        replica_dirs:
            List of directories containing topology + trajectory files
            matched by ``cfg.topology_glob`` / ``cfg.trajectory_glob``.
        cfg:
            Analysis configuration.

        Returns
        -------
        List of :class:`InteractionResult` objects (one per replica).
        """
        from ..io.loaders import discover_replicas
        from ..universe import load_and_align

        replicas = discover_replicas(replica_dirs, cfg)
        results: list[InteractionResult] = []
        for rep in replicas:
            u = load_and_align(rep["topology"], rep["trajectory"], cfg)
            results.append(self.run(u, rep["name"]))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert ProLIF MultiIndex columns to flat string labels.

        ProLIF columns are tuples like
        ``(('LIG', 'A', 1), ('ASN', 'A', 58), 'HBAcceptor')``.
        We flatten them to ``ASN-A-58_HBAcceptor`` (residue side only,
        since the ligand column is constant).
        """
        new_cols: list[str] = []
        for col in df.columns:
            if isinstance(col, tuple):
                # col may be (ligand_residue, protein_residue, interaction)
                # or (residue, interaction) depending on ProLIF version.
                if len(col) == 3:
                    res_tuple, itype = col[1], col[2]
                elif len(col) == 2:
                    res_tuple, itype = col[0], col[1]
                else:
                    new_cols.append("_".join(str(c) for c in col))
                    continue
                # res_tuple is a prolif.Residue or tuple (resname, chain, resid)
                res_str = _residue_label(res_tuple)
                new_cols.append(f"{res_str}_{itype}")
            else:
                new_cols.append(str(col))
        result = df.copy()
        result.columns = new_cols
        return result

    @staticmethod
    def _compute_summary(flat_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-interaction occupancy percentages."""
        if flat_df.empty:
            return pd.DataFrame(columns=["interaction", "occupancy_pct"])
        occ = (flat_df > 0).mean() * 100
        summary = occ.reset_index()
        summary.columns = ["interaction", "occupancy_pct"]
        summary = summary.sort_values("occupancy_pct", ascending=False).reset_index(
            drop=True
        )
        return summary


def _residue_label(res) -> str:
    """Convert a ProLIF Residue object (or tuple) to a string label."""
    try:
        # ProLIF Residue has .resname, .chain, .resid attributes
        return f"{res.resname}-{res.chain}-{res.resid}"
    except AttributeError:
        pass
    try:
        # Fallback: treat as tuple (resname, chain, resid)
        return f"{res[0]}-{res[1]}-{res[2]}"
    except (TypeError, IndexError):
        return str(res)
