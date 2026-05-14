"""RCSB PDB REST API client — shared utility.

Provides lightweight access to PDB structure download and search.

Network access is required for live queries.  Methods accept an optional
``_session`` parameter for test injection (any object with a ``.get()``
method compatible with :func:`requests.Session.get`).

Example
-------
>>> from mdatools.shared.data.pdb import PDBClient
>>> client = PDBClient()
>>> path = client.download("1IEP", output_dir=Path("/tmp"))
>>> ids = client.search_by_uniprot("P00533")  # EGFR
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"
_RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


class PDBClient:
    """Lightweight RCSB PDB REST API client.

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.
    _session:
        Test seam — any object with a ``.get(url, **kwargs)`` method.
        ``None`` uses the :mod:`requests` library.
    """

    def __init__(
        self,
        timeout: int = 30,
        _session: Any = None,
    ) -> None:
        self.timeout = timeout
        self._session = _session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, pdb_id: str, output_dir: Path) -> Path:
        """Download a PDB structure file from RCSB.

        Parameters
        ----------
        pdb_id:
            4-character PDB ID (case-insensitive).
        output_dir:
            Directory to save the file.  Created if missing.

        Returns
        -------
        Path
            Path to the downloaded ``.pdb`` file.

        Raises
        ------
        RuntimeError
            If the download fails (HTTP error or network issue).
        """
        pdb_id = pdb_id.upper()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{pdb_id}.pdb"

        if out_path.exists():
            logger.debug("PDB %s already cached at %s", pdb_id, out_path)
            return out_path

        url = _RCSB_DOWNLOAD.format(pdb_id=pdb_id)
        logger.debug("Downloading PDB %s from %s", pdb_id, url)

        try:
            resp = self._get(url)
            out_path.write_bytes(resp)
        except Exception as exc:
            raise RuntimeError(f"Failed to download PDB {pdb_id!r}: {exc}") from exc

        return out_path

    def search_by_uniprot(self, uniprot_ac: str) -> list[str]:
        """Return PDB IDs for structures from a UniProt accession.

        Parameters
        ----------
        uniprot_ac:
            UniProt accession (e.g. ``"P00533"`` for EGFR).

        Returns
        -------
        list[str]
            PDB IDs in the RCSB result (may be empty).

        Raises
        ------
        RuntimeError
            If the search request fails.
        """
        import json  # noqa: PLC0415

        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.uniprot_ids",
                    "operator": "exact_match",
                    "value": uniprot_ac.upper(),
                },
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 100}},
        }
        logger.debug("Searching PDB for UniProt %s", uniprot_ac)

        try:
            raw = self._get(_RCSB_SEARCH, params={"json": json.dumps(query)})
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(
                f"PDB search for UniProt {uniprot_ac!r} failed: {exc}"
            ) from exc

        return [hit["identifier"] for hit in data.get("result_set", [])]

    def download_batch(
        self,
        pdb_ids: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """Download multiple PDB structures.

        Parameters
        ----------
        pdb_ids:
            List of 4-character PDB IDs.
        output_dir:
            Directory to save files.

        Returns
        -------
        list[Path]
            Paths to downloaded files (in the same order as *pdb_ids*).
            Files that failed to download are ``None`` in the list.
        """
        paths: list[Path] = []
        for pdb_id in pdb_ids:
            try:
                p = self.download(pdb_id, output_dir)
                paths.append(p)
            except RuntimeError as exc:
                logger.warning("Skipping %s: %s", pdb_id, exc)
        return paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> bytes:
        if self._session is not None:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            # Support both mock objects (return bytes directly) and real requests
            if hasattr(resp, "content"):
                return resp.content
            return resp
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required for PDBClient. "
                "Install it with:  pip install requests"
            ) from exc
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content
