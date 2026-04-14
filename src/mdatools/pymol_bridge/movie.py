"""Trajectory movie generation via PyMOL + ffmpeg."""

from __future__ import annotations

import logging
import subprocess
import textwrap
from pathlib import Path

from ..config import AnalysisConfig

logger = logging.getLogger(__name__)


class MovieMaker:
    """Generate MP4 movies from a PyMOL PSE trajectory.

    Parameters
    ----------
    cfg:
        Analysis configuration.  Uses ``cfg.pymol_executable`` and
        ``cfg.ffmpeg_executable``.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def make_movie(
        self,
        pse_path: Path | str,
        output_mp4: Path | str,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        ray: bool = False,
        frame_step: int = 1,
        pocket_resids: list[int] | None = None,
        highlight_resids: list[int] | None = None,
        hbond_pairs: list[tuple[str, str]] | None = None,
        hbond_events_csv: Path | str | None = None,
        object_name: str = "traj",
    ) -> Path:
        """Render a trajectory to MP4.

        Parameters
        ----------
        pse_path:
            PyMOL session file.
        output_mp4:
            Destination MP4 file.
        width, height:
            Frame dimensions in pixels.
        fps:
            Frames per second in the output video.
        ray:
            Whether to use PyMOL ray tracing (slow but high quality).
        frame_step:
            Only render every *frame_step*-th trajectory frame.
        pocket_resids:
            Residue IDs to show as sticks in the binding pocket.
        highlight_resids:
            Residue IDs to colour red (e.g. ECD target residues).
        hbond_pairs:
            Pairs of atom selections to draw distance dashes for H-bonds.
        hbond_events_csv:
            Optional CSV from HBondAnalyzer; used to highlight H-bond frames.
        object_name:
            PyMOL trajectory object name.
        """
        pse_path = Path(pse_path)
        output_mp4 = Path(output_mp4)
        frame_dir = output_mp4.parent / (output_mp4.stem + "_frames")
        frame_dir.mkdir(parents=True, exist_ok=True)

        pml = self._build_movie_pml(
            pse_path=pse_path,
            frame_dir=frame_dir,
            width=width,
            height=height,
            ray=ray,
            frame_step=frame_step,
            pocket_resids=pocket_resids or [],
            highlight_resids=highlight_resids or [],
            hbond_pairs=hbond_pairs or [],
            hbond_events_csv=hbond_events_csv,
            object_name=object_name,
            ligand_resname=self.cfg.ligand_resname,
        )
        pml_path = frame_dir / "render.pml"
        pml_path.write_text(pml, encoding="utf-8")

        logger.info("Running PyMOL render: %s", pml_path)
        res = subprocess.run(
            [self.cfg.pymol_executable, "-c", str(pml_path)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            logger.error("PyMOL stderr:\n%s", res.stderr[-3000:])
            raise RuntimeError("PyMOL rendering failed.")

        logger.info("Encoding video with ffmpeg: %s", output_mp4)
        ffmpeg_cmd = [
            self.cfg.ffmpeg_executable,
            "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_mp4),
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", res.stderr[-3000:])
            raise RuntimeError("ffmpeg encoding failed.")

        logger.info("Movie saved: %s", output_mp4)
        return output_mp4

    def _build_movie_pml(
        self,
        pse_path: Path,
        frame_dir: Path,
        width: int,
        height: int,
        ray: bool,
        frame_step: int,
        pocket_resids: list[int],
        highlight_resids: list[int],
        hbond_pairs: list[tuple[str, str]],
        hbond_events_csv: Path | str | None,
        object_name: str,
        ligand_resname: str,
    ) -> str:
        pocket_sel = (
            f"{object_name} and polymer and resi "
            + "+".join(str(r) for r in pocket_resids)
            if pocket_resids
            else ""
        )
        highlight_sel = (
            f"{object_name} and polymer and resi "
            + "+".join(str(r) for r in highlight_resids)
            if highlight_resids
            else ""
        )
        hbond_csv_line = (
            f"HBOND_CSV = r'{hbond_events_csv}'"
            if hbond_events_csv
            else "HBOND_CSV = None"
        )
        ray_cmd = "cmd.ray()" if ray else "cmd.draw()"
        pocket_lines = (
            f"""
show sticks, {pocket_sel}
"""
            if pocket_sel
            else ""
        )
        highlight_lines = (
            f"""
color red, {highlight_sel}
"""
            if highlight_sel
            else ""
        )

        return textwrap.dedent(
            f"""\
# Auto-generated movie PML
load {pse_path}
disable all
enable {object_name}

bg_color white
set cartoon_fancy_helices, 1
show cartoon, {object_name} and polymer
show sticks, {object_name} and resname {ligand_resname}
{pocket_lines}
{highlight_lines}
viewport {width}, {height}

python
import csv, os
from pathlib import Path

{hbond_csv_line}
FRAME_DIR = r"{frame_dir}"
STEP = {frame_step}
OBJ = "{object_name}"

hbond_frames = set()
if HBOND_CSV and os.path.exists(HBOND_CSV):
    with open(HBOND_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hbond_frames.add(int(row["frame"]))

n_states = cmd.count_states(OBJ)
out_idx = 0
for i in range(1, n_states + 1, STEP):
    cmd.set_state(i, OBJ)
    # colour ligand by H-bond status
    if (i - 1) in hbond_frames:
        cmd.color("green", f"{{OBJ}} and resname {ligand_resname}")
    else:
        cmd.color("yellow", f"{{OBJ}} and resname {ligand_resname}")
    {ray_cmd}
    png_path = os.path.join(FRAME_DIR, f"frame_{{out_idx:04d}}.png")
    cmd.png(png_path, {width}, {height}, dpi=150, ray=0)
    out_idx += 1

python end
quit
"""
        )
