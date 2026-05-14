"""2D per-pose interaction diagram generation.

Produces annotated 2D ligand images showing which residues form
interactions, colour-coded by interaction type.  Uses only RDKit's
built-in drawing facilities (no external dependencies).

Example::

    # New API (matplotlib Axes)
    fig, ax = plt.subplots()
    draw_interaction_map(mol=pose_mol, interactions=["LYS745_HBDonor"], ax=ax)

    # Legacy API (PIL Image)
    img = draw_interaction_map(pose_mol, fp_row)
    img.save("interaction_map.png")
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Default colour palette for interaction types
DEFAULT_INTERACTION_COLORS: dict[str, str] = {
    "HBAcceptor": "#4472C4",
    "HBDonor": "#ED7D31",
    "Hydrophobic": "#70AD47",
    "PiStacking": "#9E480E",
    "PiCation": "#FFC000",
    "Cationic": "#FF0000",
    "Anionic": "#7030A0",
    "VdWContact": "#A5A5A5",
}


def _parse_interactions(
    fp_row: pd.Series,
) -> list[tuple[str, str]]:
    """Extract active (residue, interaction_type) pairs from a fingerprint row.

    Column names are expected in ``<RESIDUE>_<InteractionType>`` format.
    Returns only columns with value > 0.
    """
    active = []
    for col_name, value in fp_row.items():
        if value > 0 and "_" in str(col_name):
            parts = str(col_name).rsplit("_", 1)
            if len(parts) == 2:
                residue, interaction_type = parts
                active.append((residue, interaction_type))
    return active


def _interactions_from_col_names(
    col_names: list[str],
) -> list[tuple[str, str]]:
    """Convert a list of column names to (residue, interaction_type) pairs."""
    result = []
    for col in col_names:
        if "_" in str(col):
            parts = str(col).rsplit("_", 1)
            if len(parts) == 2:
                result.append((parts[0], parts[1]))
    return result


def _build_legend_text(interactions: list[tuple[str, str]]) -> str:
    """Build a compact text legend from interaction pairs."""
    if not interactions:
        return "No interactions"
    lines = []
    for residue, itype in interactions:
        lines.append(f"{residue}:{itype}")
    return " | ".join(lines)


def _mol_to_pil(mol_2d: Chem.Mol, size: tuple[int, int], legend: str) -> Image.Image:
    """Render a 2D mol to a PIL Image, with optional legend."""
    try:
        drawer = Draw.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        try:
            import cairosvg
            png_data = cairosvg.svg2png(
                bytestring=svg_text.encode(),
                output_width=size[0],
                output_height=size[1],
            )
            return Image.open(BytesIO(png_data))
        except ImportError:
            return Draw.MolToImage(mol_2d, size=size, legend=legend)
    except Exception:
        return Draw.MolToImage(mol_2d, size=size, legend=legend)


def draw_interaction_map(
    pose_mol: Chem.Mol | None = None,
    fp_row: pd.Series | None = None,
    output_path: str | Path | None = None,
    size: tuple[int, int] = (400, 300),
    interaction_colors: dict[str, str] | None = None,
    # --- new-style keyword arguments ---
    mol: Chem.Mol | None = None,
    interactions: list[str] | None = None,
    ax=None,
    title: str | None = None,
) -> Image.Image | None:
    """Draw a 2D interaction diagram for a single pose.

    Supports two calling conventions:

    **New-style** (recommended, works with matplotlib Axes)::

        fig, ax = plt.subplots()
        draw_interaction_map(
            mol=pose_mol,
            interactions=active_cols,   # list[str] from fp_df.columns
            ax=ax,
            title="My Compound",
        )

    **Legacy-style** (PIL Image)::

        img = draw_interaction_map(pose_mol, fp_row)
        img.save("interaction_map.png")

    Parameters
    ----------
    pose_mol:
        Docked pose (positional, legacy API).
    fp_row:
        One row from a ProLIF fingerprint DataFrame (legacy API).
    output_path:
        If provided, save the image to this path.
    size:
        Image size ``(width, height)`` in pixels.
    interaction_colors:
        Override the default colour mapping.
    mol:
        Docked pose RDKit Mol (keyword alias for *pose_mol*).
    interactions:
        List of active interaction column names (e.g.
        ``fp_df.columns[fp_df.any()].tolist()``).  Parsed as
        ``"<RESIDUE>_<InteractionType>"`` strings.
    ax:
        Matplotlib ``Axes`` to draw into.  When provided, the image is
        displayed on *ax* and the function returns ``None``.
    title:
        Axes title string (only used when *ax* is given).

    Returns
    -------
    PIL.Image.Image or None
        PIL Image when *ax* is ``None``; ``None`` when drawn onto *ax*.
    """
    # Resolve mol
    effective_mol = mol if mol is not None else pose_mol
    if effective_mol is None:
        raise ValueError("Provide either 'mol' or 'pose_mol'.")

    # Resolve interaction pairs
    if interactions is not None:
        interaction_pairs = _interactions_from_col_names(interactions)
    elif fp_row is not None:
        interaction_pairs = _parse_interactions(fp_row)
    else:
        interaction_pairs = []

    legend = _build_legend_text(interaction_pairs)

    # Prepare 2D depiction
    mol_2d = Chem.RWMol(effective_mol)
    try:
        AllChem.Compute2DCoords(mol_2d)
    except Exception:
        pass
    mol_2d = mol_2d.GetMol()

    img = _mol_to_pil(mol_2d, size, legend)

    # --- matplotlib Axes mode ---
    if ax is not None:
        ax.imshow(np.array(img))
        ax.axis("off")
        if title is not None:
            ax.set_title(title, fontsize=9)
        # Annotate interactions as a text block
        if interaction_pairs:
            colors = interaction_colors or DEFAULT_INTERACTION_COLORS
            y_offset = 0.01
            for residue, itype in interaction_pairs[:8]:  # max 8 labels to avoid clutter
                color = colors.get(itype, "#333333")
                ax.text(
                    0.02, y_offset,
                    f"{residue}:{itype}",
                    transform=ax.transAxes,
                    fontsize=6,
                    color=color,
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"),
                )
                y_offset += 0.08
        return None

    # --- PIL Image mode (legacy / output_path) ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return img


def draw_interaction_grid(
    poses: list[Chem.Mol],
    fp_df: pd.DataFrame,
    indices: list[int] | None = None,
    n_cols: int = 4,
    mol_size: tuple[int, int] = (300, 200),
    output_path: str | Path | None = None,
) -> Image.Image:
    """Draw a grid of interaction diagrams for multiple poses.

    Parameters
    ----------
    poses:
        List of docked poses.
    fp_df:
        ProLIF fingerprint DataFrame (rows parallel to *poses*).
    indices:
        Subset of pose indices to include.  ``None`` = all poses.
    n_cols:
        Number of columns in the grid.
    mol_size:
        Size of each molecule image ``(width, height)`` in pixels.
    output_path:
        If provided, save the grid image to this path.

    Returns
    -------
    PIL.Image.Image
    """
    if indices is None:
        indices = list(range(len(poses)))

    mols_2d = []
    legends = []

    for idx in indices:
        mol = Chem.RWMol(poses[idx])
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        mols_2d.append(mol.GetMol())

        fp_row = fp_df.iloc[idx]
        interactions = _parse_interactions(fp_row)
        legends.append(_build_legend_text(interactions))

    img = Draw.MolsToGridImage(
        mols_2d,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=legends,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return img
