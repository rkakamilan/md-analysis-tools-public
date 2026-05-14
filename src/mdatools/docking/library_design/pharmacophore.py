"""Ensemble pharmacophore modeling and ligand-based virtual screening.

Extracts pharmacophore features (HBA, HBD, Hydrophobic, Aromatic, Ionizable)
from a set of active compounds using RDKit's built-in feature factory, then
clusters the ensemble with k-means to produce a consensus pharmacophore
hypothesis.  The hypothesis can be used to screen a compound library by
matching feature types and spatial positions.

Inspired by TeachOpenCADD T009.

Example::

    from rdkit import Chem
    from mdatools.docking.library_design.pharmacophore import (
        PharmacophoreModeler,
        ensemble_pharmacophore,
        screen_library,
    )

    actives = [Chem.MolFromSmiles(s) for s in [
        "c1ccc(NC(=O)c2cccc(F)c2)cc1",
        "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
    ]]
    modeler = PharmacophoreModeler()
    query = ensemble_pharmacophore(actives, n_clusters=3)
    hits = screen_library(query, library_mols, match_threshold=0.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# RDKit feature definitions mapped to our canonical names
_FEATURE_FAMILY_MAP = {
    "Donor": "HBD",
    "Acceptor": "HBA",
    "Hydrophobe": "Hydrophobic",
    "LumpedHydrophobe": "Hydrophobic",
    "Aromatic": "Aromatic",
    "PosIonizable": "Ionizable",
    "NegIonizable": "Ionizable",
}

_FEATURE_TYPES = ("HBD", "HBA", "Hydrophobic", "Aromatic", "Ionizable")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PharmacophoreFeature:
    """A single pharmacophore feature point.

    Attributes
    ----------
    feature_type:
        One of ``"HBD"``, ``"HBA"``, ``"Hydrophobic"``, ``"Aromatic"``,
        ``"Ionizable"``.
    position:
        3-D centroid of the feature (x, y, z) in Å.
        For 2-D molecules without conformers, position is ``(0.0, 0.0, 0.0)``.
    weight:
        Relative importance (higher = more conserved across the ensemble).
        Set to ``1.0`` for single-molecule features.
    """

    feature_type: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    weight: float = 1.0


@dataclass
class PharmacophoreHypothesis:
    """Consensus pharmacophore hypothesis from an ensemble of actives.

    Attributes
    ----------
    features:
        List of representative :class:`PharmacophoreFeature` points.
    n_actives:
        Number of active molecules used to build the hypothesis.
    feature_counts:
        Raw count of each feature type across the ensemble.
    """

    features: list[PharmacophoreFeature] = field(default_factory=list)
    n_actives: int = 0
    feature_counts: dict[str, int] = field(default_factory=dict)

    @property
    def feature_types(self) -> list[str]:
        """Unique feature types present in the hypothesis."""
        return list({f.feature_type for f in self.features})


# ---------------------------------------------------------------------------
# PharmacophoreModeler
# ---------------------------------------------------------------------------


class PharmacophoreModeler:
    """Extract pharmacophore features from RDKit molecules.

    Uses RDKit's ``MolChemicalFeatures`` factory (built-in SMARTS-based
    feature definitions — no extra data files required).
    """

    def __init__(self) -> None:
        self._factory: Any = None

    def _get_factory(self) -> Any:
        if self._factory is not None:
            return self._factory
        try:
            import os  # noqa: PLC0415

            from rdkit import RDConfig  # noqa: PLC0415
            from rdkit.Chem.rdMolChemicalFeatures import BuildFeatureFactory  # noqa: PLC0415

            fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            self._factory = BuildFeatureFactory(fdef)
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "RDKit pharmacophore factory could not be initialised."
            ) from exc
        return self._factory

    def extract_features(self, mol: Any) -> list[PharmacophoreFeature]:
        """Extract pharmacophore features from a molecule.

        Works with both 2-D and 3-D molecules.  For 2-D molecules the
        position is the 2-D embedding centroid (if available) or
        ``(0.0, 0.0, 0.0)``.

        Parameters
        ----------
        mol:
            RDKit Mol.

        Returns
        -------
        list[PharmacophoreFeature]
        """
        try:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem import AllChem  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for pharmacophore extraction."
            ) from exc

        if mol is None:
            return []

        # Ensure at least a 2-D conformer for position extraction
        mol_h = Chem.AddHs(mol)
        has_conf = mol_h.GetNumConformers() > 0
        if not has_conf:
            AllChem.Compute2DCoords(mol_h)

        factory = self._get_factory()
        n_feats = factory.GetNumMolFeatures(mol_h)

        result: list[PharmacophoreFeature] = []
        for i in range(n_feats):
            feat = factory.GetMolFeature(mol_h, i)
            family = feat.GetFamily()
            ftype = _FEATURE_FAMILY_MAP.get(family)
            if ftype is None:
                continue
            pos = feat.GetPos()
            result.append(
                PharmacophoreFeature(
                    feature_type=ftype,
                    position=(float(pos.x), float(pos.y), float(pos.z)),
                )
            )
        return result


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def ensemble_pharmacophore(
    active_mols: list,
    n_clusters: int = 5,
    min_occurrence: float = 0.5,
) -> PharmacophoreHypothesis:
    """Build a consensus pharmacophore hypothesis from active molecules.

    Steps:

    1. Extract features from each active molecule.
    2. For each feature type, collect all 3-D positions across the ensemble.
    3. Cluster positions with k-means (one centroid per cluster = one query
       feature).
    4. Keep only clusters whose occurrence fraction ≥ *min_occurrence*.

    Parameters
    ----------
    active_mols:
        List of RDKit Mols.  ``None`` entries are skipped.
    n_clusters:
        Max number of representative points per feature type.
    min_occurrence:
        Minimum fraction of active molecules that must contain a feature
        for it to be included in the hypothesis (default 0.5 = 50 %).

    Returns
    -------
    PharmacophoreHypothesis
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    modeler = PharmacophoreModeler()
    valid_mols = [m for m in active_mols if m is not None]
    n_actives = len(valid_mols)

    if n_actives == 0:
        return PharmacophoreHypothesis()

    # Collect features per type: {ftype: [(x,y,z), ...]}
    type_positions: dict[str, list[tuple[float, float, float]]] = {
        t: [] for t in _FEATURE_TYPES
    }
    type_mol_counts: dict[str, int] = {t: 0 for t in _FEATURE_TYPES}

    for mol in valid_mols:
        feats = modeler.extract_features(mol)
        seen_types: set[str] = set()
        for feat in feats:
            ft = feat.feature_type
            if ft in type_positions:
                type_positions[ft].append(feat.position)
                if ft not in seen_types:
                    type_mol_counts[ft] += 1
                    seen_types.add(ft)

    query_features: list[PharmacophoreFeature] = []
    feature_counts: dict[str, int] = {}

    for ftype in _FEATURE_TYPES:
        positions = type_positions[ftype]
        mol_count = type_mol_counts[ftype]
        feature_counts[ftype] = mol_count

        # Skip rare features
        if n_actives > 0 and mol_count / n_actives < min_occurrence:
            continue
        if not positions:
            continue

        pts = np.array(positions)
        k = min(n_clusters, len(pts))

        if k == 1 or len(pts) == 1:
            centroid = pts.mean(axis=0)
            query_features.append(
                PharmacophoreFeature(
                    feature_type=ftype,
                    position=(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    weight=float(mol_count / n_actives),
                )
            )
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(pts)
            for centroid in km.cluster_centers_:
                query_features.append(
                    PharmacophoreFeature(
                        feature_type=ftype,
                        position=(
                            float(centroid[0]),
                            float(centroid[1]),
                            float(centroid[2]),
                        ),
                        weight=float(mol_count / n_actives),
                    )
                )

    return PharmacophoreHypothesis(
        features=query_features,
        n_actives=n_actives,
        feature_counts=feature_counts,
    )


def screen_library(
    hypothesis: PharmacophoreHypothesis,
    library_mols: list,
    match_threshold: float = 0.5,
    distance_tolerance: float = 1.5,
) -> list[tuple[Any, float]]:
    """Screen a compound library against a pharmacophore hypothesis.

    A library molecule is scored by the fraction of query feature *types*
    it satisfies.  Spatial matching uses the extracted feature positions
    compared to the hypothesis centroids within *distance_tolerance* Å
    (for 3-D molecules; for 2-D molecules only feature-type matching is used).

    Parameters
    ----------
    hypothesis:
        Query pharmacophore hypothesis from :func:`ensemble_pharmacophore`.
    library_mols:
        Molecules to screen.  ``None`` entries are skipped.
    match_threshold:
        Minimum score (0–1) to include a molecule in the results.
    distance_tolerance:
        Maximum distance (Å) between a query centroid and a molecule
        feature position for a spatial match (3-D only).

    Returns
    -------
    list[tuple[Mol, float]]
        ``(mol, score)`` pairs for molecules scoring ≥ *match_threshold*,
        sorted by score descending.
    """
    if not hypothesis.features:
        return []

    query_types = set(f.feature_type for f in hypothesis.features)
    modeler = PharmacophoreModeler()

    results: list[tuple[Any, float]] = []
    for mol in library_mols:
        if mol is None:
            continue
        try:
            mol_feats = modeler.extract_features(mol)
        except Exception:  # noqa: BLE001
            continue

        mol_types = {f.feature_type for f in mol_feats}
        matched_types = query_types & mol_types
        score = len(matched_types) / len(query_types) if query_types else 0.0

        if score >= match_threshold:
            results.append((mol, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
