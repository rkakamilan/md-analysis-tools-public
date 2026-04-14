"""CLI: generate a trajectory movie from a PyMOL PSE file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an MP4 trajectory movie from a PyMOL PSE file."
    )
    parser.add_argument("--pse", type=Path, required=True, help="Input .pse file")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 file")
    parser.add_argument("--ligand", default="UNK")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ray", action="store_true", help="Enable PyMOL ray tracing")
    parser.add_argument(
        "--pocket-resids",
        nargs="*",
        type=int,
        default=[],
        help="Residue IDs to show as sticks in pocket",
    )
    parser.add_argument(
        "--highlight-resids",
        nargs="*",
        type=int,
        default=[],
        help="Residue IDs to colour red",
    )
    parser.add_argument(
        "--hbond-csv",
        type=Path,
        default=None,
        help="H-bond events CSV for frame colouring",
    )
    parser.add_argument("--pymol", default="pymol")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    from mdatools.config import AnalysisConfig
    from mdatools.pymol_bridge.movie import MovieMaker

    cfg = AnalysisConfig(
        ligand_resname=args.ligand,
        pymol_executable=args.pymol,
        ffmpeg_executable=args.ffmpeg,
    )
    maker = MovieMaker(cfg)
    maker.make_movie(
        pse_path=args.pse,
        output_mp4=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        ray=args.ray,
        frame_step=args.step,
        pocket_resids=args.pocket_resids,
        highlight_resids=args.highlight_resids,
        hbond_events_csv=args.hbond_csv,
    )


if __name__ == "__main__":
    main()
