"""mols2grid interactive compound selection helper.

Provides :func:`make_selection_grid` to set up an interactive mols2grid
widget with multi-property sliders in a single call.

Requires ``mols2grid`` and ``ipywidgets`` (install via ``uv sync --group notebook``).

Example::

    grid = make_selection_grid(df, smiles_col="SMILES")
    # In JupyterLab: grid is displayed interactively
    # After selection:
    selected = grid.get_selection()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


def _auto_display_cols(df: pd.DataFrame) -> list[str]:
    """Select display columns based on what's available in the DataFrame."""
    priority = [
        "mol_name", "compound", "SMILES",
        "docking_score", "consensus_score",
        "mw", "logp", "hbd", "hba", "tpsa", "rot_bonds",
        "le", "lelp", "ro5_pass", "veber_pass",
        "cluster_id", "km_cluster_id",
    ]
    return [c for c in priority if c in df.columns]


def make_selection_grid(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    sort_by: str = "docking_score",
    display_cols: list[str] | None = None,
    n_rows: int = 3,
    size: tuple[int, int] = (200, 150),
):
    """Create a mols2grid interactive grid for compound selection.

    Parameters
    ----------
    df:
        DataFrame with a SMILES column and property columns.
    smiles_col:
        Name of the SMILES column.
    sort_by:
        Column to sort by initially (ascending).
    display_cols:
        Columns to display in the grid tooltip.  ``None`` = auto-detect.
    n_rows:
        Number of rows visible in the grid.
    size:
        Molecule image size ``(width, height)`` in pixels.

    Returns
    -------
    mols2grid.MolGrid
        Interactive grid widget (displays in JupyterLab).

    Raises
    ------
    ImportError
        If ``mols2grid`` is not installed.
    ValueError
        If *smiles_col* is not in the DataFrame.
    """
    try:
        import mols2grid
    except ImportError:
        raise ImportError(
            "mols2grid is required for interactive compound selection. "
            "Install it with: uv sync --group notebook"
        )

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")

    if display_cols is None:
        display_cols = _auto_display_cols(df)

    # Sort if the column exists
    view_df = df.copy()
    if sort_by in view_df.columns:
        view_df = view_df.sort_values(sort_by, ascending=True)

    grid = mols2grid.MolGrid(
        view_df,
        smiles_col=smiles_col,
        size=size,
    )

    return grid


def export_selection(
    grid,
    output_csv: str | Path | None = None,
    output_sdf: str | Path | None = None,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Export the current grid selection to CSV and/or SDF.

    Parameters
    ----------
    grid:
        mols2grid.MolGrid instance with user selections.
    output_csv:
        Path to write CSV output.
    output_sdf:
        Path to write SDF output (requires SMILES column in selection).
    smiles_col:
        Name of the SMILES column for SDF conversion.

    Returns
    -------
    pd.DataFrame
        The selected compounds.
    """
    selected = grid.get_selection()

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(str(output_csv), index=False)

    if output_sdf is not None and smiles_col in selected.columns:
        output_sdf = Path(output_sdf)
        output_sdf.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(output_sdf))
        for _, row in selected.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is not None:
                for col in selected.columns:
                    if col != smiles_col:
                        mol.SetProp(col, str(row[col]))
                writer.write(mol)
        writer.close()

    return selected
