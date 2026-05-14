"""KLIFS REST API client — shared between md-analysis-tools and docking-analysis-tools.

KLIFS (Kinase–Ligand Interaction Fingerprints and Structures) is an
online resource for kinase–ligand structures with a consistent 85-residue
pocket numbering scheme across all kinase structures.

This client implements the KLIFS REST API v2 (https://klifs.net/api/v2/).
Network access is required for live queries; results are cached on disk to
avoid repeated requests during offline analysis.

Typical usage
-------------
>>> from mdatools.shared.data.klifs import KLIFSClient
>>> klifs = KLIFSClient()
>>> info = klifs.get_kinase_info("EGFR")
>>> info.family
'EGFR'
>>> structs = klifs.get_structures("EGFR", species="Human")
>>> len(structs)
326
>>> ifp = klifs.get_ifp(structure_id=structs[0].structure_ID)
>>> len(ifp)
85

Cache
-----
Results are cached as JSON files under *cache_dir* (default:
``~/.mdatools/cache/klifs/``).  Pass ``use_cache=False`` to disable
caching.  Delete the cache directory to force a fresh fetch.

Rate limiting
-------------
KLIFS requests are rate-limited to ``requests_per_second`` (default: 5)
to comply with KLIFS' fair-use policy.  Increase at your own risk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependency — imported at module level so tests can patch it.
try:
    import requests  # type: ignore[import-untyped]
except ImportError:
    requests = None  # type: ignore[assignment]

_KLIFS_BASE = "https://klifs.net/api/v2/"
_DEFAULT_CACHE = Path.home() / ".mdatools" / "cache" / "klifs"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KinaseInfo:
    """Basic kinase metadata from KLIFS.

    Attributes
    ----------
    kinase_ID:
        KLIFS internal kinase ID.
    kinase_name:
        KLIFS canonical kinase name (e.g. ``"EGFR"``).
    full_name:
        Full kinase name (e.g. ``"Epidermal growth factor receptor"``).
    group:
        Kinome group (e.g. ``"TK"`` for tyrosine kinase).
    family:
        Kinase family (e.g. ``"EGFR"``).
    subfamily:
        Kinase subfamily or ``None``.
    species:
        Species (``"Human"``, ``"Mouse"``, etc.).
    uniprot:
        UniProt accession or ``None``.
    iuphar:
        IUPHAR ID or ``None``.
    """

    kinase_ID: int
    kinase_name: str
    full_name: str
    group: str
    family: str
    subfamily: str | None
    species: str
    uniprot: str | None = None
    iuphar: int | None = None


@dataclass
class KLIFSStructure:
    """One kinase–ligand complex entry in KLIFS.

    Attributes
    ----------
    structure_ID:
        KLIFS structure ID (primary key in KLIFS).
    kinase_ID:
        KLIFS kinase ID.
    kinase_name:
        Canonical kinase name.
    pdb:
        4-character PDB ID.
    chain:
        PDB chain identifier.
    alt:
        Alternate location indicator (``"A"``, ``"B"``, or ``""``)
    resolution:
        X-ray resolution in Å (or ``None``).
    quality_score:
        KLIFS quality score [0, 10].
    missing_residues:
        Number of missing KLIFS pocket residues.
    missing_atoms:
        Number of missing pocket heavy atoms.
    ligand:
        Ligand PDB code or ``None`` (apo structures).
    dfg:
        DFG-loop conformation (``"in"``, ``"out"``, ``"out-like"``, ``"na"``).
    aC_helix:
        αC-helix conformation (``"in"``, ``"out"``, ``"na"``).
    allosteric_ligand:
        PDB code of any allosteric ligand or ``None``.
    """

    structure_ID: int
    kinase_ID: int
    kinase_name: str
    pdb: str
    chain: str
    alt: str
    resolution: float | None
    quality_score: float
    missing_residues: int
    missing_atoms: int
    ligand: str | None
    dfg: str
    aC_helix: str
    allosteric_ligand: str | None = None


@dataclass
class KLIFSPocket:
    """KLIFS 85-position pocket residue assignment for one structure.

    Attributes
    ----------
    structure_ID:
        KLIFS structure ID.
    klifs_numbers:
        List of KLIFS pocket positions 1–85.
    residue_ids:
        PDB residue IDs at each KLIFS position (``None`` for missing).
    residue_names:
        Amino acid 1-letter codes (``"-"`` for missing positions).
    """

    structure_ID: int
    klifs_numbers: list[int] = field(default_factory=list)
    residue_ids: list[int | None] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[int, int | None]:
        """Return ``{klifs_pos: pdb_resid}`` mapping."""
        return dict(zip(self.klifs_numbers, self.residue_ids))

    def missing_positions(self) -> list[int]:
        """Return KLIFS positions where the residue is absent."""
        return [k for k, r in zip(self.klifs_numbers, self.residue_ids) if r is None]


@dataclass
class KLIFS_IFP:
    """85-bit interaction fingerprint from KLIFS.

    Attributes
    ----------
    structure_ID:
        KLIFS structure ID.
    IFP:
        85-character binary string (``"0"`` / ``"1"``).  Each position
        corresponds to one of the 85 KLIFS pocket residues; ``"1"``
        indicates at least one interaction with the bound ligand.
    IFP_array:
        NumPy uint8 array of shape ``(85,)`` derived from :attr:`IFP`.
    """

    structure_ID: int
    IFP: str  # 85-char "01..." string

    @property
    def IFP_array(self) -> np.ndarray:
        """85-element uint8 array derived from the IFP string."""
        return np.frombuffer(self.IFP.encode(), dtype=np.uint8) - 48  # '0'=48

    def tanimoto(self, other: KLIFS_IFP) -> float:
        """Binary Tanimoto similarity to *other* IFP."""
        a = self.IFP_array.astype(bool)
        b = other.IFP_array.astype(bool)
        intersection = int((a & b).sum())
        union = int((a | b).sum())
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KLIFSClient:
    """KLIFS REST API v2 client with disk caching.

    Parameters
    ----------
    cache_dir:
        Root directory for cached JSON responses.  Created on first use.
        Pass ``None`` to disable caching.
    use_cache:
        If ``False``, every request hits the live KLIFS API.
    requests_per_second:
        Throttle parameter.  KLIFS fair-use policy suggests ≤ 5 req/s.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cache_dir: Path | None = _DEFAULT_CACHE,
        *,
        use_cache: bool = True,
        requests_per_second: float = 5.0,
        timeout: int = 30,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._use_cache = use_cache and cache_dir is not None
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._timeout = timeout
        self._last_request: float = 0.0
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_kinase_info(self, kinase_name: str, species: str = "Human") -> KinaseInfo:
        """Retrieve metadata for a single kinase by canonical name.

        Parameters
        ----------
        kinase_name:
            KLIFS canonical kinase name (case-insensitive, e.g. ``"EGFR"``).
        species:
            Target species (``"Human"``, ``"Mouse"`` etc.).

        Returns
        -------
        KinaseInfo

        Raises
        ------
        KeyError
            When the kinase is not found in KLIFS.
        """
        params = {"kinase_name": kinase_name, "species": species}
        data = self._get(
            "kinases", params=params, cache_key=f"kinase_{kinase_name}_{species}"
        )
        # API returns a list; pick the first exact-name match
        for item in data if isinstance(data, list) else [data]:
            if item.get("kinase_name", "").upper() == kinase_name.upper():
                return _parse_kinase_info(item)
        raise KeyError(
            f"Kinase '{kinase_name}' not found in KLIFS for species '{species}'. "
            "Check https://klifs.net for canonical names."
        )

    def search_kinases(
        self,
        group: str | None = None,
        family: str | None = None,
        species: str = "Human",
    ) -> list[KinaseInfo]:
        """Return all kinases matching optional group / family filters.

        Parameters
        ----------
        group:
            Kinome group (e.g. ``"TK"``).  ``None`` = all groups.
        family:
            Kinase family (e.g. ``"EGFR"``).  ``None`` = all families.
        species:
            Target species.

        Returns
        -------
        list[KinaseInfo]
        """
        params: dict[str, str] = {"species": species}
        if group:
            params["kinase_group"] = group
        if family:
            params["kinase_family"] = family
        cache_key = f"kinases_{group or 'all'}_{family or 'all'}_{species}"
        data = self._get("kinases", params=params, cache_key=cache_key)
        if isinstance(data, list):
            return [_parse_kinase_info(d) for d in data]
        return [_parse_kinase_info(data)]

    def get_kinase_id(self, kinase_name: str, species: str = "Human") -> int:
        """Return the integer KLIFS kinase ID for *kinase_name*."""
        return self.get_kinase_info(kinase_name, species=species).kinase_ID

    def get_structures(
        self,
        kinase_name: str,
        *,
        species: str = "Human",
        only_with_ligand: bool = False,
        min_quality: float = 0.0,
    ) -> list[KLIFSStructure]:
        """Return all structures for a given kinase from KLIFS.

        Parameters
        ----------
        kinase_name:
            KLIFS canonical kinase name.
        species:
            Target species.
        only_with_ligand:
            When ``True``, exclude apo structures.
        min_quality:
            Minimum KLIFS quality score [0, 10].

        Returns
        -------
        list[KLIFSStructure]
        """
        kinase_id = self.get_kinase_id(kinase_name, species=species)
        params = {"kinase_ID": str(kinase_id)}
        cache_key = f"structures_{kinase_id}"
        data = self._get("structures", params=params, cache_key=cache_key)
        structs = [
            _parse_structure(d) for d in (data if isinstance(data, list) else [data])
        ]
        if only_with_ligand:
            structs = [s for s in structs if s.ligand is not None]
        if min_quality > 0:
            structs = [s for s in structs if s.quality_score >= min_quality]
        return structs

    def get_ifp(self, structure_id: int) -> KLIFS_IFP:
        """Retrieve the 85-bit interaction fingerprint for one structure.

        Parameters
        ----------
        structure_id:
            KLIFS structure ID.

        Returns
        -------
        KLIFS_IFP
        """
        cache_key = f"ifp_{structure_id}"
        data = self._get(f"interactions/IFP/{structure_id}", cache_key=cache_key)
        ifp_str = data.get("IFP", data) if isinstance(data, dict) else str(data)
        if len(ifp_str) != 85 or not all(c in "01" for c in ifp_str):
            raise ValueError(
                f"Unexpected IFP format for structure {structure_id}: {ifp_str!r}"
            )
        return KLIFS_IFP(structure_ID=structure_id, IFP=ifp_str)

    def get_pocket(self, structure_id: int) -> KLIFSPocket:
        """Retrieve the 85-residue pocket definition for one structure.

        Parameters
        ----------
        structure_id:
            KLIFS structure ID.

        Returns
        -------
        KLIFSPocket
        """
        cache_key = f"pocket_{structure_id}"
        data = self._get(f"pockets/{structure_id}", cache_key=cache_key)
        residues = data if isinstance(data, list) else data.get("residues", [])
        numbers: list[int] = []
        res_ids: list[int | None] = []
        res_names: list[str] = []
        for r in residues:
            numbers.append(int(r.get("KLIFS_position", 0)))
            rid = r.get("residue_id")
            res_ids.append(int(rid) if rid not in (None, "", "x") else None)
            res_names.append(str(r.get("residue_name", "-")) or "-")
        return KLIFSPocket(
            structure_ID=structure_id,
            klifs_numbers=numbers,
            residue_ids=res_ids,
            residue_names=res_names,
        )

    def tanimoto_matrix(self, ifps: list[KLIFS_IFP]) -> np.ndarray:
        """Compute an N×N binary Tanimoto similarity matrix from a list of IFPs.

        Parameters
        ----------
        ifps:
            List of :class:`KLIFS_IFP` instances.

        Returns
        -------
        np.ndarray
            Shape ``(N, N)`` float64 array; diagonal is 1.0.
        """
        n = len(ifps)
        mat = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                t = ifps[i].tanimoto(ifps[j])
                mat[i, j] = t
                mat[j, i] = t
        return mat

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        *,
        params: dict[str, str] | None = None,
        cache_key: str | None = None,
    ) -> Any:
        """Perform a GET request, using the disk cache when available."""
        if self._use_cache and cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        self._throttle()
        url = urljoin(_KLIFS_BASE, endpoint)
        if requests is None:
            raise ImportError(
                "The 'requests' library is required for KLIFSClient. "
                "Install it with:  pip install requests"
            )

        logger.debug("KLIFS GET %s params=%s", url, params)
        resp = requests.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        if self._use_cache and cache_key:
            self._save_cache(cache_key, data)
        return data

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self._cache_dir / f"{safe}.json"  # type: ignore[operator]

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.debug("Cache read failed for %s; will re-fetch.", key)
        return None

    def _save_cache(self, key: str, data: Any) -> None:
        try:
            path = self._cache_path(key)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            logger.debug("Cache write failed for key %s", key)

    def clear_cache(self) -> None:
        """Delete all cached KLIFS JSON files."""
        if self._cache_dir and self._cache_dir.exists():
            for f in self._cache_dir.glob("*.json"):
                f.unlink()
            logger.info("KLIFS cache cleared: %s", self._cache_dir)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_kinase_info(d: dict[str, Any]) -> KinaseInfo:
    return KinaseInfo(
        kinase_ID=int(d.get("kinase_ID", 0)),
        kinase_name=str(d.get("kinase_name", "")),
        full_name=str(d.get("full_name", d.get("kinase_name", ""))),
        group=str(d.get("kinase_group", d.get("group", ""))),
        family=str(d.get("kinase_family", d.get("family", ""))),
        subfamily=d.get("kinase_subfamily") or d.get("subfamily") or None,
        species=str(d.get("species", "")),
        uniprot=d.get("uniprot") or None,
        iuphar=int(d["iuphar"]) if d.get("iuphar") else None,
    )


def _parse_structure(d: dict[str, Any]) -> KLIFSStructure:
    return KLIFSStructure(
        structure_ID=int(d.get("structure_ID", 0)),
        kinase_ID=int(d.get("kinase_ID", 0)),
        kinase_name=str(d.get("kinase_name", "")),
        pdb=str(d.get("pdb", "")),
        chain=str(d.get("chain", "")),
        alt=str(d.get("alt", "")),
        resolution=float(d["resolution"])
        if d.get("resolution") not in (None, "")
        else None,
        quality_score=float(d.get("quality_score", 0.0)),
        missing_residues=int(d.get("missing_residues", 0)),
        missing_atoms=int(d.get("missing_atoms", 0)),
        ligand=str(d["ligand"]) if d.get("ligand") not in (None, "", "0", 0) else None,
        dfg=str(d.get("DFG", d.get("dfg", "na"))),
        aC_helix=str(d.get("aC_helix", "na")),
        allosteric_ligand=d.get("allosteric_ligand") or None,
    )
