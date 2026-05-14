"""Py3Dmol-based 3D protein-ligand complex viewer for HITL review.

Provides helpers to display docked poses in a 3D viewer within
JupyterLab, with optional interaction highlights.

Requires ``py3Dmol`` (install via ``uv sync --group notebook``).

Example::

    view = show_complex("receptor.pdb", ligand_mol)
    # In JupyterLab: view is displayed as interactive 3D widget
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from rdkit import Chem


def _mol_to_sdf_string(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol to an SDF string."""
    writer = Chem.SDWriter(StringIO())
    sio = StringIO()
    writer = Chem.SDWriter(sio)
    writer.write(mol)
    writer.close()
    return sio.getvalue()


def show_complex(
    receptor_pdb: str | Path,
    ligand_mol: Chem.Mol,
    interactions: pd.Series | None = None,
    width: int = 800,
    height: int = 500,
    receptor_style: str = "cartoon",
    ligand_color: str = "cyan",
):
    """Display a protein-ligand complex in a Py3Dmol viewer.

    Parameters
    ----------
    receptor_pdb:
        Path to the receptor PDB file.
    ligand_mol:
        Docked ligand pose (RDKit Mol with 3D conformer).
    interactions:
        Optional ProLIF fingerprint row for annotation display.
    width, height:
        Viewer size in pixels.
    receptor_style:
        Protein display style: ``"cartoon"``, ``"stick"``, ``"surface"``.
    ligand_color:
        Ligand carbon colour.

    Returns
    -------
    py3Dmol.view
        Interactive 3D viewer (displays in JupyterLab).
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install it with: uv sync --group notebook"
        )

    pdb_text = Path(receptor_pdb).read_text()
    ligand_sdf = _mol_to_sdf_string(ligand_mol)

    view = py3Dmol.view(width=width, height=height)

    # Receptor
    view.addModel(pdb_text, "pdb")
    if receptor_style == "cartoon":
        view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
    elif receptor_style == "stick":
        view.setStyle({"model": 0}, {"stick": {}})
    elif receptor_style == "surface":
        view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
        view.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "white"}, {"model": 0})

    # Ligand
    view.addModel(ligand_sdf, "sdf")
    view.setStyle(
        {"model": 1},
        {"stick": {"colorscheme": f"{ligand_color}Carbon"}},
    )

    view.zoomTo()
    return view


def show_pose_comparison(
    receptor_pdb: str | Path,
    mols: list[Chem.Mol],
    labels: list[str] | None = None,
    width: int = 800,
    height: int = 500,
):
    """Overlay multiple ligand poses on the same receptor.

    Parameters
    ----------
    receptor_pdb:
        Path to the receptor PDB file.
    mols:
        List of docked poses to overlay.
    labels:
        Optional labels for each pose.
    width, height:
        Viewer size in pixels.

    Returns
    -------
    py3Dmol.view
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install it with: uv sync --group notebook"
        )

    colors = ["cyan", "magenta", "yellow", "lime", "orange", "pink", "white", "red"]
    pdb_text = Path(receptor_pdb).read_text()

    view = py3Dmol.view(width=width, height=height)

    # Receptor
    view.addModel(pdb_text, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum", "opacity": 0.7}})

    # Ligands
    for i, mol in enumerate(mols):
        sdf = _mol_to_sdf_string(mol)
        view.addModel(sdf, "sdf")
        color = colors[i % len(colors)]
        view.setStyle(
            {"model": i + 1},
            {"stick": {"colorscheme": f"{color}Carbon"}},
        )

    view.zoomTo()
    return view


def export_review_results(
    results: list[dict],
    output_path: str | Path,
) -> Path:
    """Export HITL review results to CSV.

    Parameters
    ----------
    results:
        List of dicts with keys: compound_id, decision, comment, reviewer, timestamp.
    output_path:
        Path for the CSV file.

    Returns
    -------
    Path
        Absolute path to the written CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(str(output_path), index=False)
    return output_path.resolve()
