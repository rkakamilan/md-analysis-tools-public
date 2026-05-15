"""Hierarchical Scaffold Network analysis using RDKit rdScaffoldNetwork.

Builds a scaffold network from a set of molecules and provides frequency
counts and visualization for scaffold-based SAR analysis.

Example::

    net = build_scaffold_network(mols)
    counts_df = scaffold_counts(net, mols)
    fig = plot_scaffold_frequency(counts_df, top_n=15)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork


def build_scaffold_network(
    mols: list[Chem.Mol],
    include_generic_scaffolds: bool = False,
) -> rdScaffoldNetwork.ScaffoldNetwork:
    """Build a hierarchical scaffold network from molecules.

    Parameters
    ----------
    mols:
        List of RDKit Mols.
    include_generic_scaffolds:
        If ``True``, include generic scaffolds (all atoms → carbon,
        all bonds → single) as additional nodes.

    Returns
    -------
    rdScaffoldNetwork.ScaffoldNetwork
    """
    params = rdScaffoldNetwork.ScaffoldNetworkParams()
    params.includeGenericScaffolds = include_generic_scaffolds
    return rdScaffoldNetwork.CreateScaffoldNetwork(mols, params)


def scaffold_counts(
    network: rdScaffoldNetwork.ScaffoldNetwork,
    mols: list[Chem.Mol],
) -> pd.DataFrame:
    """Count how many input molecules contain each scaffold node.

    Parameters
    ----------
    network:
        Scaffold network from :func:`build_scaffold_network`.
    mols:
        The same molecules used to build the network (for substructure matching).

    Returns
    -------
    pd.DataFrame
        Columns: ``scaffold_smiles``, ``count``, ``fraction``.
        Sorted by count descending.
    """
    nodes = list(network.nodes)
    counts = []

    for node_smiles in nodes:
        query = Chem.MolFromSmarts(node_smiles)
        if query is None:
            counts.append(0)
            continue
        n = sum(1 for mol in mols if mol.HasSubstructMatch(query))
        counts.append(n)

    df = pd.DataFrame({
        "scaffold_smiles": nodes,
        "count": counts,
        "fraction": [c / len(mols) if len(mols) > 0 else 0.0 for c in counts],
    })
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def plot_scaffold_frequency(
    counts_df: pd.DataFrame,
    top_n: int = 20,
    output_path: str | Path | None = None,
) -> "Any":
    """Plot a bar chart of the most frequent scaffolds.

    Parameters
    ----------
    counts_df:
        Output from :func:`scaffold_counts`.
    top_n:
        Number of top scaffolds to display.
    output_path:
        If provided, save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = counts_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    ax.barh(range(len(top)), top["count"].values, color="#4472C4", edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["scaffold_smiles"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Compound count")
    ax.set_title(f"Top {min(top_n, len(top))} Scaffold Frequencies")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


def add_scaffold_network_level_to_df(
    df: pd.DataFrame,
    mols: list[Chem.Mol],
    network: rdScaffoldNetwork.ScaffoldNetwork,
    col_name: str = "scaffold_network",
) -> pd.DataFrame:
    """Add the most specific scaffold network node as a column.

    For each molecule, finds the most specific (largest) scaffold node
    in the network that the molecule contains.

    Parameters
    ----------
    df:
        DataFrame with rows parallel to *mols*.
    mols:
        List of RDKit Mols.
    network:
        Scaffold network.
    col_name:
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the scaffold column appended.
    """
    nodes = list(network.nodes)
    queries = []
    for smi in nodes:
        q = Chem.MolFromSmarts(smi)
        queries.append((smi, q))

    # Sort by specificity (more atoms = more specific)
    queries.sort(key=lambda x: x[1].GetNumAtoms() if x[1] is not None else 0, reverse=True)

    scaffold_labels = []
    for mol in mols:
        label = ""
        for smi, q in queries:
            if q is not None and mol.HasSubstructMatch(q):
                label = smi
                break
        scaffold_labels.append(label)

    result = df.copy()
    result[col_name] = scaffold_labels
    return result
