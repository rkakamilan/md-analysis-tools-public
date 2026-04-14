"""CLI: run RMSD analysis for a list of replica directories."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RMSD analysis batch for MD replicas."
    )
    parser.add_argument("replica_dirs", nargs="+", type=Path)
    parser.add_argument("--ligand", default="UNK")
    parser.add_argument("--topology-glob", default="equilibrating_topology.pdb")
    parser.add_argument("--traj-glob", default="trajectory.xtc")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("./figures"))
    parser.add_argument("--n-cols", type=int, default=3)
    args = parser.parse_args()

    from mdatools.config import AnalysisConfig
    from mdatools.analysis.rmsd import run_rmsd_batch
    from mdatools.plotting.rmsd_plots import plot_rmsd_grid
    import pickle

    cfg = AnalysisConfig(
        ligand_resname=args.ligand,
        topology_glob=args.topology_glob,
        trajectory_glob=args.traj_glob,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
    cfg.make_dirs()
    results = run_rmsd_batch(args.replica_dirs, cfg)

    with open(cfg.output_dir / "result_dfs.pkl", "wb") as f:
        pickle.dump(results, f)

    fig = plot_rmsd_grid(
        results,
        n_cols=args.n_cols,
        save_path=cfg.figures_dir / "combined_rmsd_plots.png",
    )
    print(f"Done. {len(results)} replicas → {cfg.figures_dir}/combined_rmsd_plots.png")


if __name__ == "__main__":
    main()
