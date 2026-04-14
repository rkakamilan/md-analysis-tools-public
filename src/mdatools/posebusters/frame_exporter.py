"""Export MD trajectory frames as PDB files for downstream validation.

Supports two backends:
- MDAnalysis (preferred, no PyMOL required)
- PyMOL subprocess (PSE-only trajectories)

Solvent / ion removal is applied before writing so that PoseBusters
receives clean protein + ligand structures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import MDAnalysis as mda

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)

# Residue names that are unconditionally removed (solvent + common ions)
_DEFAULT_REMOVE_RESNAMES: tuple[str, ...] = (
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4",
    "NA", "CL", "K", "CA", "MG", "ZN", "FE", "MN", "CU", "CO", "NI",
    "SO4", "PO4", "GOL", "EDO", "PEG", "DMS", "ACE", "ACT",
)


class FrameExporter:
    """Export trajectory frames as individual PDB files.

    Parameters
    ----------
    cfg:
        Analysis configuration.  ``cfg.ligand_resname`` is used to
        ensure the ligand is retained when solvent is stripped.
    remove_resnames:
        Residue names to strip (solvent, ions).  Defaults to a standard
        list of water models and common ions.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        remove_resnames: tuple[str, ...] | None = None,
    ) -> None:
        self.cfg = cfg
        self.remove_resnames = remove_resnames or _DEFAULT_REMOVE_RESNAMES

    # ------------------------------------------------------------------
    # MDAnalysis backend (preferred)
    # ------------------------------------------------------------------

    def export_from_universe(
        self,
        u: mda.Universe,
        output_dir: Path,
        prefix: str,
        *,
        step: int = 1,
        max_frames: int | None = None,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Export frames from an MDAnalysis Universe.

        Parameters
        ----------
        u:
            Universe (already aligned if needed).
        output_dir:
            Directory to write PDB files.
        prefix:
            Filename prefix, e.g. ``"snapshot_run01"``.
        step:
            Export every *step*-th frame.
        max_frames:
            Cap the total number of frames exported.
        skip_existing:
            Skip frames whose output PDB already exists.

        Returns
        -------
        List of written PDB paths (existing skipped files are included).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build selection string: protein + ligand, no solvent/ions
        remove_sel = " or ".join(
            f"resname {r}" for r in self.remove_resnames
        )
        keep_sel = f"(protein or resname {self.cfg.ligand_resname}) and not ({remove_sel})"
        ligand_atoms = u.select_atoms(f"resname {self.cfg.ligand_resname}")
        if len(ligand_atoms) == 0:
            raise ValueError(
                f"Selection returned 0 atoms for ligand resname "
                f"'{self.cfg.ligand_resname}'. Check cfg.ligand_resname."
            )
        atoms = u.select_atoms(keep_sel)

        written: list[Path] = []
        frame_indices = list(range(0, len(u.trajectory), step))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        logger.info(
            "Exporting %d frames (step=%d) → %s", len(frame_indices), step, output_dir
        )
        for fi in frame_indices:
            out_path = output_dir / f"{prefix}_frame{fi:04d}.pdb"
            written.append(out_path)
            if skip_existing and out_path.exists():
                continue
            u.trajectory[fi]
            atoms.write(str(out_path))
            if fi % 10 == 0:
                logger.debug("Exported frame %d → %s", fi, out_path.name)

        logger.info("Done: %d PDB files in %s", len(written), output_dir)
        return written

    # ------------------------------------------------------------------
    # PyMOL subprocess backend (PSE-only trajectories)
    # ------------------------------------------------------------------

    def export_from_pse(
        self,
        pse_path: Path,
        output_dir: Path,
        prefix: str,
        *,
        max_frames: int = 100,
        step: int = 1,
        object_name: str = "traj",
        skip_existing: bool = True,
    ) -> list[Path]:
        """Export frames from a PyMOL PSE session file.

        Uses :func:`~mdatools.pymol_bridge.pml_builder.build_frame_export_pml`
        to generate a PML script that is executed via ``pymol -c``.

        Returns
        -------
        List of expected output PDB paths (may not all exist if PyMOL fails).
        """
        import subprocess

        from ..pymol_bridge.pml_builder import build_frame_export_pml

        pse_path = Path(pse_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pml = build_frame_export_pml(
            pse_path=pse_path,
            output_dir=output_dir,
            prefix=prefix,
            max_frames=max_frames,
            step=step,
            object_name=object_name,
            remove_resnames=list(self.remove_resnames),
            skip_existing=skip_existing,
        )
        pml_path = output_dir / "_export.pml"
        pml_path.write_text(pml, encoding="utf-8")

        logger.info("Running PyMOL frame export: %s", pml_path)
        result = subprocess.run(
            [self.cfg.pymol_executable, "-c", str(pml_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("PyMOL stderr:\n%s", result.stderr[-2000:])
            raise RuntimeError("PyMOL frame export failed.")

        frame_indices = list(range(0, max_frames * step, step))[:max_frames]
        return [output_dir / f"{prefix}_frame{i:04d}.pdb" for i in frame_indices]
