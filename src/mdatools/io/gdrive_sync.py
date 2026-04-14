"""Google Drive differential upload for MD result files.

Uploads files matching glob patterns to a shared Google Drive folder,
skipping files whose MD5 checksum matches the already-uploaded version.

Requires: ``pip install 'mdatools[gdrive]'``
  (google-auth, google-auth-oauthlib, google-api-python-client, tqdm)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATTERNS: list[str] = ["*.pse", "*.pdb", "analysis_*.csv"]

# Scopes required for file upload/read to Drive
_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class GDriveSync:
    """Differential upload of MD result files to a Google Drive folder.

    Authentication priority:
    1. Service account JSON (``credentials`` path points to a service account key)
    2. OAuth2 user credentials (``credentials`` path points to an OAuth2 client
       secrets JSON; a browser flow is triggered on first use)

    Parameters
    ----------
    folder_id:
        Google Drive folder ID (from the folder URL).
    credentials:
        Path to either a service account key JSON or OAuth2 client secrets JSON.
    """

    def __init__(self, folder_id: str, credentials: Path) -> None:
        self.folder_id = folder_id
        self.credentials = Path(credentials)
        self._service = self._build_service()

    # ------------------------------------------------------------------ #
    # Authentication                                                        #
    # ------------------------------------------------------------------ #

    def _build_service(self):
        """Build and return a Drive API v3 service object."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                str(self.credentials), scopes=_SCOPES
            )
            logger.info("Authenticated via service account: %s", self.credentials.name)
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as sa_err:
            logger.debug("Service account auth failed (%s), trying OAuth2.", sa_err)

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            import pickle

            token_path = self.credentials.parent / "token.pkl"
            creds = None
            if token_path.exists():
                with open(token_path, "rb") as f:
                    creds = pickle.load(f)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials), _SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                with open(token_path, "wb") as f:
                    pickle.dump(creds, f)

            logger.info("Authenticated via OAuth2.")
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as oauth_err:
            raise RuntimeError(
                f"Authentication failed.\n"
                f"  service account error: {sa_err}\n"
                f"  OAuth2 error: {oauth_err}"
            ) from oauth_err

    # ------------------------------------------------------------------ #
    # Drive helpers                                                         #
    # ------------------------------------------------------------------ #

    def _list_remote(self) -> dict[str, dict]:
        """Return {filename: {id, md5Checksum}} for files in *folder_id*."""
        remote: dict[str, dict] = {}
        page_token = None
        while True:
            resp = (
                self._service.files()
                .list(
                    q=f"'{self.folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, md5Checksum)",
                    pageToken=page_token,
                )
                .execute()
            )
            for f in resp.get("files", []):
                remote[f["name"]] = {"id": f["id"], "md5": f.get("md5Checksum", "")}
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return remote

    def _upload(self, local_path: Path, existing_id: str | None) -> None:
        """Upload (create or update) a single file."""
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(local_path), resumable=True)
        if existing_id:
            self._service.files().update(
                fileId=existing_id, media_body=media
            ).execute()
        else:
            self._service.files().create(
                body={"name": local_path.name, "parents": [self.folder_id]},
                media_body=media,
            ).execute()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def sync(
        self,
        local_root: Path,
        patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, list[Path]]:
        """Sync files from *local_root* to the Drive folder.

        Parameters
        ----------
        local_root:
            Local directory to scan.
        patterns:
            Glob patterns relative to *local_root*.
            Defaults to ``["*.pse", "*.pdb", "analysis_*.csv"]``.
        dry_run:
            If *True*, collect the list of files that *would* be uploaded
            without actually uploading anything.

        Returns
        -------
        dict with keys ``"uploaded"``, ``"skipped"``, ``"failed"`` — each
        mapping to a list of :class:`~pathlib.Path` objects.
        """
        if patterns is None:
            patterns = _DEFAULT_PATTERNS

        local_root = Path(local_root)
        result: dict[str, list[Path]] = {"uploaded": [], "skipped": [], "failed": []}

        # Collect candidate files
        candidates: list[Path] = []
        for pat in patterns:
            candidates.extend(local_root.glob(pat))
        candidates = sorted(set(candidates))

        if not candidates:
            logger.info("No files matched patterns %s under %s", patterns, local_root)
            return result

        # Fetch remote file list (skip in dry_run to avoid network call)
        remote: dict[str, dict] = {} if dry_run else self._list_remote()

        try:
            from tqdm import tqdm
            iterator = tqdm(candidates, desc="Syncing to Drive", unit="file")
        except ImportError:
            iterator = candidates  # type: ignore[assignment]

        for path in iterator:
            try:
                fname = path.name
                local_md5 = _md5(path)
                remote_info = remote.get(fname)

                if remote_info and remote_info["md5"] == local_md5:
                    logger.debug("SKIP %s (MD5 match)", fname)
                    result["skipped"].append(path)
                    continue

                if dry_run:
                    logger.info("DRY-RUN upload: %s", fname)
                    result["uploaded"].append(path)
                else:
                    self._upload(path, remote_info["id"] if remote_info else None)
                    logger.info("UPLOADED %s", fname)
                    result["uploaded"].append(path)

            except Exception as exc:
                logger.warning("FAILED %s: %s", path.name, exc)
                result["failed"].append(path)

        return result
