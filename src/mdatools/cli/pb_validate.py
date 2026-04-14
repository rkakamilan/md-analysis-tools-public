"""CLI: run PoseBusters batch validation on a trajectory replica."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PoseBusters batch validation for MD trajectory frames.\n"
            "Requires: pip install 'mdatools[posebusters]'\n"
            "See docs/POSEBUSTERS_PREREQUISITES.md for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topology", type=Path, help="Topology PDB")
    parser.add_argument("--trajectory", type=Path, help="Trajectory XTC/NC")
    parser.add_argument("--pse", type=Path, default=None, help="PyMOL PSE file (alternative to --topology/--trajectory)")
    parser.add_argument("--protein", type=Path, required=True, help="Clean protein PDB (no solvent/ions)")
    parser.add_argument(
        "--smiles", default=None,
        help="Ligand SMILES string (required for bond-order assignment)"
    )
    parser.add_argument(
        "--sdf", type=Path, default=None,
        help="Ligand SDF/MOL file (alternative to --smiles)"
    )
    parser.add_argument("--ligand", default="UNK", help="Ligand residue name")
    parser.add_argument("--output-dir", type=Path, default=Path("./hbond_results"))
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="snapshot")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--full-report", action="store_true")
    args = parser.parse_args()

    if args.smiles is None and args.sdf is None:
        parser.error(
            "--smiles or --sdf is required.\n"
            "PDB format lacks bond-order information. See docs/POSEBUSTERS_PREREQUISITES.md"
        )

    from mdatools.config import AnalysisConfig
    from mdatools.posebusters import FrameExporter, PoseBustersValidator

    cfg = AnalysisConfig(ligand_resname=args.ligand, output_dir=args.output_dir)
    cfg.make_dirs()

    frames_dir = args.frames_dir or (args.output_dir / "pb_frames" / args.prefix)

    exporter = FrameExporter(cfg)
    if args.pse:
        pdb_files = exporter.export_from_pse(
            pse_path=args.pse,
            output_dir=frames_dir,
            prefix=args.prefix,
            max_frames=args.max_frames or 100,
            step=args.step,
        )
    else:
        if not args.topology or not args.trajectory:
            parser.error("Either --pse or both --topology and --trajectory are required.")
        from mdatools.universe import load_and_align
        u = load_and_align(args.topology, args.trajectory, cfg)
        pdb_files = exporter.export_from_universe(
            u,
            output_dir=frames_dir,
            prefix=args.prefix,
            step=args.step,
            max_frames=args.max_frames,
        )

    validator = PoseBustersValidator(
        cfg,
        ligand_smiles=args.smiles,
        ligand_sdf=args.sdf,
        full_report=args.full_report,
    )
    result = validator.run(
        pdb_dir=frames_dir,
        protein_path=args.protein,
        prefix=args.prefix,
    )

    out_csv = cfg.output_dir / f"posebusters_{args.prefix}.csv"
    result.results.to_csv(out_csv, index=False)
    print(f"Pass rate: {result.pass_rate:.1%}  ({result.results['all_pass'].sum()}/{len(result.results)} frames)")
    print(f"Results saved: {out_csv}")
    print("\nTop 5 frames:")
    print(result.top_frames(5).to_string(index=False))


if __name__ == "__main__":
    main()
