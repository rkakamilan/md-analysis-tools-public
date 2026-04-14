"""Interactive 3D trajectory viewer using NGLView (optional extra: viewer).

Install the optional dependency with::

    pip install "mdatools[viewer]"

All functions in this module raise :class:`ImportError` with an instructive
message when ``nglview`` is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import MDAnalysis as mda
    import nglview


def _require_nglview():
    """Return the nglview module or raise an informative ImportError."""
    try:
        import nglview as _nv
        return _nv
    except ImportError as exc:
        raise ImportError(
            "nglview is required for interactive visualization. "
            "Install it with:  pip install 'mdatools[viewer]'"
        ) from exc


def show_universe(
    u: "mda.Universe",
    selection: str = "protein or resname UNK",
    trajectory: bool = True,
) -> "nglview.NGLWidget":
    """Display an MDAnalysis Universe interactively in Jupyter.

    Parameters
    ----------
    u:
        MDAnalysis Universe with an optional trajectory.
    selection:
        Atom selection string passed to ``u.select_atoms()``.
        Use ``"all"`` to include solvent.
    trajectory:
        When ``True`` (default) the widget plays the trajectory
        frame-by-frame.  Pass ``False`` to show only the first frame.

    Returns
    -------
    nglview.NGLWidget
        Ready to display — just return it from a Jupyter cell.

    Examples
    --------
    >>> view = show_universe(u, selection="protein or resname LIG")
    >>> view  # displays in Jupyter
    """
    nv = _require_nglview()
    atoms = u.select_atoms(selection)
    view = nv.show_mdanalysis(atoms)
    return view


def show_snapshot(
    pdb_path: "Path | str",
    ligand_resname: str = "UNK",
    highlight_residues: "list[int] | None" = None,
) -> "nglview.NGLWidget":
    """Display a single PDB snapshot interactively in Jupyter.

    The ligand is shown in ball-and-stick.  Optional protein residues
    can be highlighted in yellow.

    Parameters
    ----------
    pdb_path:
        Path to the PDB file (e.g. a snapshot from
        :class:`~mdatools.scoring.snapshot.SnapshotSelector`).
    ligand_resname:
        Residue name of the ligand.
    highlight_residues:
        Optional list of protein residue IDs to render in yellow
        ball-and-stick (useful for key binding-pocket residues).

    Returns
    -------
    nglview.NGLWidget

    Raises
    ------
    FileNotFoundError
        If *pdb_path* does not exist.
    """
    nv = _require_nglview()
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    view = nv.show_file(str(pdb_path))
    view.add_ball_and_stick(f"[{ligand_resname}]")
    if highlight_residues:
        for resid in highlight_residues:
            view.add_ball_and_stick(f":{resid}", color="yellow")
    return view


def show_cluster_representatives(
    pdb_paths: "list[Path | str]",
    labels: "list[str] | None" = None,
) -> "nglview.NGLWidget":
    """Overlay multiple representative PDB structures in one viewer.

    Convenient for comparing cluster centroids from
    :class:`~mdatools.analysis.clustering.PoseClusterer` or
    :class:`~mdatools.analysis.clustering.DIVINEClusterer`.

    Parameters
    ----------
    pdb_paths:
        Paths to PDB files — one per cluster representative.
    labels:
        Display names for each structure in the NGLView component list.
        Defaults to ``cluster00``, ``cluster01``, …

    Returns
    -------
    nglview.NGLWidget

    Raises
    ------
    FileNotFoundError
        If any PDB file does not exist.
    """
    nv = _require_nglview()
    view = nv.NGLWidget()
    for i, path in enumerate(pdb_paths):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDB file not found: {path}")
        label = (labels[i] if labels else None) or f"cluster{i:02d}"
        view.add_component(str(path), name=label)
    return view


def add_contact_map_overlay(
    widget: "nglview.NGLWidget",
    contacts: pd.DataFrame,
    threshold: float = 0.5,
) -> "nglview.NGLWidget":
    """Highlight high-occupancy contact residues in an existing NGLWidget.

    Residues whose ``occupancy_%`` value exceeds *threshold* × 100 are
    shown in red ball-and-stick.  This is designed to work with the
    :class:`~mdatools.analysis.contacts.ContactFingerprint` output DataFrame.

    Parameters
    ----------
    widget:
        An existing NGLWidget (e.g. from :func:`show_snapshot`).
    contacts:
        DataFrame with at least ``resid`` (int) and ``occupancy_%`` (float)
        columns, as returned by
        :class:`~mdatools.analysis.contacts.ContactFingerprint`.
    threshold:
        Occupancy fraction cutoff in the range 0–1.  Only residues with
        ``occupancy_% / 100 > threshold`` are highlighted.

    Returns
    -------
    The same *widget* with additional representations added in-place.
    """
    _require_nglview()
    cutoff_pct = threshold * 100.0
    high_occ = contacts[contacts["occupancy_%"] > cutoff_pct]
    for resid in high_occ["resid"].unique():
        widget.add_ball_and_stick(f":{resid}", color="red")
    return widget
