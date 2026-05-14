"""ADME profiling for virtual screening hit triage.

Computes Lipinski Ro5, Veber, QED, and PAINS flags for compound libraries.
Includes radar chart and distribution plot utilities.

(TeachOpenCADD T002 inspired).

Example::

    from rdkit import Chem
    from mdatools.docking.preparation.adme_profiler import ADMEProfiler

    mols = [Chem.MolFromSmiles(s) for s in ["CC(=O)O", "c1ccccc1"]]
    profiler = ADMEProfiler()
    df = profiler.profile_batch(mols)
    hits = profiler.filter(mols, rules=["ro5", "veber", "no_pains"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADMEProfile dataclass
# ---------------------------------------------------------------------------


@dataclass
class ADMEProfile:
    """Physicochemical and ADME properties for one molecule.

    Attributes
    ----------
    mol_weight:
        Molecular weight (Da).
    logp:
        Wildman-Crippen LogP.
    hbd:
        H-bond donor count.
    hba:
        H-bond acceptor count.
    tpsa:
        Topological polar surface area (Å²).
    rotatable_bonds:
        Number of rotatable bonds.
    qed:
        Quantitative Estimate of Drug-likeness (0–1).
    passes_ro5:
        ``True`` when the molecule satisfies Lipinski's Ro5
        (MW≤500, LogP≤5, HBD≤5, HBA≤10; max one violation allowed).
    passes_veber:
        ``True`` when RotBonds≤10 and TPSA≤140.
    pains_alerts:
        PAINS substructure alert names (empty list = clean).
    """

    mol_weight: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    rotatable_bonds: int = 0
    qed: float = 0.0
    passes_ro5: bool = False
    passes_veber: bool = False
    pains_alerts: list[str] = field(default_factory=list)

    @property
    def is_drug_like(self) -> bool:
        """``True`` when both Ro5 and Veber pass and no PAINS alerts."""
        return self.passes_ro5 and self.passes_veber and not self.pains_alerts


# ---------------------------------------------------------------------------
# ADMEProfiler
# ---------------------------------------------------------------------------


class ADMEProfiler:
    """Compute ADME/physicochemical properties for compound libraries.

    Parameters
    ----------
    ro5_max_violations:
        Maximum Ro5 violations before flagging a compound as non-compliant.
        Default ``1`` (classic Lipinski one-violation rule).
    """

    def __init__(self, ro5_max_violations: int = 1) -> None:
        self.ro5_max_violations = ro5_max_violations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, mol: Any) -> ADMEProfile:
        """Compute ADME properties for a single molecule.

        Parameters
        ----------
        mol:
            RDKit Mol object.

        Returns
        -------
        ADMEProfile
        """
        try:
            from rdkit.Chem import Descriptors, QED, rdMolDescriptors  # noqa: PLC0415
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for ADME profiling. "
                "Install it with: pip install 'mdatools[docking]'"
            ) from exc

        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        tpsa = float(Descriptors.TPSA(mol))
        rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        qed_val = float(QED.qed(mol))

        # Lipinski Ro5 (one violation allowed by default)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        passes_ro5 = violations <= self.ro5_max_violations

        # Veber oral bioavailability rules
        passes_veber = rot <= 10 and tpsa <= 140

        # PAINS
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(pains_params)
        pains_alerts = [e.GetDescription() for e in pains_catalog.GetMatches(mol)]

        return ADMEProfile(
            mol_weight=mw,
            logp=logp,
            hbd=hbd,
            hba=hba,
            tpsa=tpsa,
            rotatable_bonds=rot,
            qed=qed_val,
            passes_ro5=passes_ro5,
            passes_veber=passes_veber,
            pains_alerts=pains_alerts,
        )

    def profile_batch(self, mols: list) -> "Any":
        """Compute ADME profiles for a list of molecules.

        Parameters
        ----------
        mols:
            List of RDKit Mols (``None`` entries are skipped with NaN row).

        Returns
        -------
        pandas.DataFrame
            One row per molecule with all ``ADMEProfile`` fields as columns,
            plus ``is_drug_like``.
        """
        import pandas as pd  # noqa: PLC0415

        rows = []
        for mol in mols:
            if mol is None:
                rows.append(None)
                continue
            try:
                p = self.profile(mol)
                rows.append(
                    {
                        "mol_weight": p.mol_weight,
                        "logp": p.logp,
                        "hbd": p.hbd,
                        "hba": p.hba,
                        "tpsa": p.tpsa,
                        "rotatable_bonds": p.rotatable_bonds,
                        "qed": p.qed,
                        "passes_ro5": p.passes_ro5,
                        "passes_veber": p.passes_veber,
                        "pains_alerts": p.pains_alerts,
                        "is_drug_like": p.is_drug_like,
                    }
                )
            except Exception:  # noqa: BLE001
                rows.append(None)

        valid_rows = [r if r is not None else {} for r in rows]
        return pd.DataFrame(valid_rows)

    def filter(
        self,
        mols: list,
        rules: list[str],
    ) -> list:
        """Return only molecules that pass all specified rules.

        Parameters
        ----------
        mols:
            List of RDKit Mols.
        rules:
            Subset of ``["ro5", "veber", "no_pains"]``.

        Returns
        -------
        list[Mol]
            Filtered molecule list (``None`` entries always excluded).
        """
        kept = []
        for mol in mols:
            if mol is None:
                continue
            try:
                p = self.profile(mol)
            except Exception:  # noqa: BLE001
                continue
            ok = True
            if "ro5" in rules and not p.passes_ro5:
                ok = False
            if "veber" in rules and not p.passes_veber:
                ok = False
            if "no_pains" in rules and p.pains_alerts:
                ok = False
            if ok:
                kept.append(mol)
        return kept


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_adme_radar(
    profile: ADMEProfile,
    title: str = "",
    figsize: tuple[float, float] = (5.0, 5.0),
) -> "Any":
    """Radar chart of six ADME axes with Lipinski reference zone.

    Axes: MW (÷500), LogP (÷5), HBD (÷5), HBA (÷10), TPSA (÷140), RotBonds (÷10).
    All values normalised to [0, 1] relative to Lipinski / Veber limits.
    The green shaded zone represents the drug-like region (normalised value ≤ 1).

    Parameters
    ----------
    profile:
        ``ADMEProfile`` to visualise.
    title:
        Optional plot title (e.g. compound name).
    figsize:
        Matplotlib figure size tuple.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np  # noqa: PLC0415

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("matplotlib is required for radar charts.") from exc

    labels = [
        "MW\n(/500)",
        "LogP\n(/5)",
        "HBD\n(/5)",
        "HBA\n(/10)",
        "TPSA\n(/140)",
        "RotBonds\n(/10)",
    ]
    limits = [500.0, 5.0, 5.0, 10.0, 140.0, 10.0]
    values_raw = [
        profile.mol_weight,
        profile.logp,
        float(profile.hbd),
        float(profile.hba),
        profile.tpsa,
        float(profile.rotatable_bonds),
    ]
    values = [v / lim for v, lim in zip(values_raw, limits)]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Close the polygon
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Reference circle at limit (normalised = 1)
    ref = [1.0] * n + [1.0]
    ax.fill(angles_plot, ref, color="lightgreen", alpha=0.2, label="Ro5/Veber zone")
    ax.plot(angles_plot, ref, color="green", linewidth=1, linestyle="--")

    # Compound profile
    ax.fill(angles_plot, values_plot, color="steelblue", alpha=0.4)
    ax.plot(angles_plot, values_plot, color="steelblue", linewidth=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, max(1.3, max(values_plot) * 1.1))
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5×", "limit"], size=7)

    title_str = title or ("Drug-like" if profile.is_drug_like else "Not drug-like")
    ax.set_title(title_str, pad=15, size=10)
    fig.tight_layout()
    return fig


def plot_adme_distribution(
    profiles_df: "Any",
    property_col: str = "mol_weight",
    bins: int = 30,
    figsize: tuple[float, float] = (6.0, 3.5),
) -> "Any":
    """Histogram of an ADME property across a compound library.

    Parameters
    ----------
    profiles_df:
        DataFrame returned by :meth:`ADMEProfiler.profile_batch`.
    property_col:
        Column to plot (e.g. ``"mol_weight"``, ``"logp"``).
    bins:
        Number of histogram bins.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("matplotlib is required for distribution plots.") from exc

    _LIMITS = {
        "mol_weight": 500.0,
        "logp": 5.0,
        "hbd": 5.0,
        "hba": 10.0,
        "tpsa": 140.0,
        "rotatable_bonds": 10.0,
    }

    fig, ax = plt.subplots(figsize=figsize)
    data = profiles_df[property_col].dropna()
    ax.hist(data, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)

    if property_col in _LIMITS:
        limit = _LIMITS[property_col]
        ax.axvline(
            limit, color="red", linewidth=1.5, linestyle="--", label=f"Limit ({limit})"
        )
        ax.legend(fontsize=8)

    ax.set_xlabel(property_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{property_col.replace('_', ' ').title()} distribution", fontsize=11)
    fig.tight_layout()
    return fig
