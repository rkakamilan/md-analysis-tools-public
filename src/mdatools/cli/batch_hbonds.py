"""CLI: run H-bond analysis for a list of replica directories."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hydrogen bond analysis batch for MD replicas."
    )
    parser.add_argument(
        "replica_dirs",
        nargs="+",
        type=Path,
        help="Replica directories containing topology and trajectory files.",
    )
    parser.add_argument(
        "--ligand", default="UNK", help="Ligand residue name (default: UNK)"
    )
    parser.add_argument(
        "--topology-glob",
        default="equilibrating_topology.pdb",
        help="Glob pattern for topology file",
    )
    parser.add_argument(
        "--traj-glob",
        default="trajectory.xtc",
        help="Glob pattern for trajectory file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./hbond_results"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--d-a-cutoff", type=float, default=3.5, help="Donor-acceptor cutoff (Å)"
    )
    parser.add_argument(
        "--angle-cutoff", type=float, default=150.0, help="D-H-A angle cutoff (°)"
    )
    args = parser.parse_args()

    from mdatools.config import AnalysisConfig, HBondConfig
    from mdatools.analysis.hbonds import run_hbond_batch

    cfg = AnalysisConfig(
        ligand_resname=args.ligand,
        topology_glob=args.topology_glob,
        trajectory_glob=args.traj_glob,
        output_dir=args.output_dir,
        hbond=HBondConfig(
            d_a_cutoff=args.d_a_cutoff,
            angle_cutoff=args.angle_cutoff,
        ),
    )
    cfg.make_dirs()
    results = run_hbond_batch(args.replica_dirs, cfg)
    print(f"Done. Analysed {len(results)} replicas → {cfg.output_dir}")


if __name__ == "__main__":
    main()
