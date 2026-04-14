"""Tests for GDriveSync (Feature F).

All Google API calls are mocked — gdrive extra is not required.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _make_sync(tmp_path: Path, folder_id: str = "FOLDER_ID") -> "GDriveSync":
    """Return a GDriveSync with a mocked service (no real auth)."""
    from mdatools.io.gdrive_sync import GDriveSync

    creds_path = tmp_path / "sa_key.json"
    creds_path.write_text("{}")  # dummy

    mock_service = MagicMock()
    with patch.object(GDriveSync, "_build_service", return_value=mock_service):
        syncer = GDriveSync(folder_id=folder_id, credentials=creds_path)
    syncer._service = mock_service
    return syncer


# ---------------------------------------------------------------------------
# _md5 helper
# ---------------------------------------------------------------------------

class TestMd5Helper:

    def test_consistent_hash(self, tmp_path):
        from mdatools.io.gdrive_sync import _md5
        p = _write(tmp_path / "f.dat", b"hello world")
        assert _md5(p) == _md5(p)

    def test_different_content_different_hash(self, tmp_path):
        from mdatools.io.gdrive_sync import _md5
        p1 = _write(tmp_path / "a.dat", b"aaa")
        p2 = _write(tmp_path / "b.dat", b"bbb")
        assert _md5(p1) != _md5(p2)

    def test_matches_hashlib(self, tmp_path):
        from mdatools.io.gdrive_sync import _md5
        data = b"test data"
        p = _write(tmp_path / "f.dat", data)
        assert _md5(p) == hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# GDriveSync.sync — dry_run
# ---------------------------------------------------------------------------

class TestSyncDryRun:

    def test_dry_run_no_upload_called(self, tmp_path):
        syncer = _make_sync(tmp_path)
        _write(tmp_path / "snap.pdb", b"ATOM  1  CA  ALA\n")
        result = syncer.sync(tmp_path, patterns=["*.pdb"], dry_run=True)
        syncer._service.files().create.assert_not_called()
        syncer._service.files().update.assert_not_called()

    def test_dry_run_returns_uploaded_list(self, tmp_path):
        syncer = _make_sync(tmp_path)
        _write(tmp_path / "snap.pdb", b"ATOM 1\n")
        result = syncer.sync(tmp_path, patterns=["*.pdb"], dry_run=True)
        assert len(result["uploaded"]) == 1
        assert result["uploaded"][0].name == "snap.pdb"

    def test_dry_run_skipped_and_failed_empty(self, tmp_path):
        syncer = _make_sync(tmp_path)
        _write(tmp_path / "snap.pdb", b"X\n")
        result = syncer.sync(tmp_path, patterns=["*.pdb"], dry_run=True)
        assert result["skipped"] == []
        assert result["failed"] == []


# ---------------------------------------------------------------------------
# GDriveSync.sync — live (mocked API)
# ---------------------------------------------------------------------------

class TestSyncLive:

    def _setup_remote(self, syncer, remote_files: list[dict]) -> None:
        """Patch _list_remote to return *remote_files* without network calls."""
        syncer._list_remote = MagicMock(return_value={
            f["name"]: {"id": f["id"], "md5": f.get("md5Checksum", "")}
            for f in remote_files
        })
        syncer._upload = MagicMock()  # avoid googleapiclient import

    def test_new_file_uploaded(self, tmp_path):
        syncer = _make_sync(tmp_path)
        self._setup_remote(syncer, [])
        _write(tmp_path / "snap.pdb", b"NEW\n")
        result = syncer.sync(tmp_path, patterns=["*.pdb"])
        assert len(result["uploaded"]) == 1
        assert result["skipped"] == []
        syncer._upload.assert_called_once()

    def test_unchanged_file_skipped(self, tmp_path):
        syncer = _make_sync(tmp_path)
        data = b"UNCHANGED\n"
        _write(tmp_path / "snap.pdb", data)
        self._setup_remote(syncer, [
            {"id": "FILE_ID", "name": "snap.pdb", "md5Checksum": _md5(data)}
        ])
        result = syncer.sync(tmp_path, patterns=["*.pdb"])
        assert result["skipped"] == [tmp_path / "snap.pdb"]
        assert result["uploaded"] == []
        syncer._upload.assert_not_called()

    def test_changed_file_updated(self, tmp_path):
        syncer = _make_sync(tmp_path)
        new_data = b"CHANGED\n"
        _write(tmp_path / "snap.pdb", new_data)
        self._setup_remote(syncer, [
            {"id": "OLD_ID", "name": "snap.pdb", "md5Checksum": "00000000"}
        ])
        result = syncer.sync(tmp_path, patterns=["*.pdb"])
        assert len(result["uploaded"]) == 1
        syncer._upload.assert_called_once()

    def test_no_matching_files_returns_empty(self, tmp_path):
        syncer = _make_sync(tmp_path)
        self._setup_remote(syncer, [])
        result = syncer.sync(tmp_path, patterns=["*.xyz"])
        assert result == {"uploaded": [], "skipped": [], "failed": []}

    def test_failed_file_in_failed_list(self, tmp_path):
        syncer = _make_sync(tmp_path)
        _write(tmp_path / "snap.pdb", b"DATA\n")
        self._setup_remote(syncer, [])
        syncer._upload.side_effect = RuntimeError("upload error")
        result = syncer.sync(tmp_path, patterns=["*.pdb"])
        assert len(result["failed"]) == 1
        assert result["uploaded"] == []

    def test_multiple_patterns(self, tmp_path):
        syncer = _make_sync(tmp_path)
        self._setup_remote(syncer, [])
        _write(tmp_path / "run.pse", b"PSE\n")
        _write(tmp_path / "snap.pdb", b"PDB\n")
        _write(tmp_path / "analysis_rmsd.csv", b"frame,rmsd\n1,1.2\n")
        result = syncer.sync(tmp_path, patterns=["*.pse", "*.pdb", "analysis_*.csv"])
        assert len(result["uploaded"]) == 3


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCli:

    def test_dry_run_flag_passed_to_sync(self, tmp_path, capsys, monkeypatch):
        import sys
        from mdatools.cli import gdrive_sync as cli_mod

        creds = tmp_path / "sa.json"
        creds.write_text("{}")
        local_dir = tmp_path / "data"
        local_dir.mkdir()
        _write(local_dir / "snap.pdb", b"X\n")

        mock_syncer = MagicMock()
        mock_syncer.sync.return_value = {
            "uploaded": [local_dir / "snap.pdb"],
            "skipped": [],
            "failed": [],
        }
        MockClass = MagicMock(return_value=mock_syncer)

        monkeypatch.setattr(sys, "argv", [
            "mdatools-gdrive-sync",
            "--folder-id", "FOLDER",
            "--credentials", str(creds),
            "--local-dir", str(local_dir),
            "--dry-run",
        ])

        # Patch the import inside main() by inserting a fake module
        import types
        fake_mod = types.ModuleType("mdatools.io.gdrive_sync")
        fake_mod.GDriveSync = MockClass
        monkeypatch.setitem(sys.modules, "mdatools.io.gdrive_sync", fake_mod)

        cli_mod.main()

        mock_syncer.sync.assert_called_once()
        call_kwargs = mock_syncer.sync.call_args[1]
        assert call_kwargs.get("dry_run") is True
        out = capsys.readouterr().out
        assert "Uploaded" in out

    def test_output_shows_counts(self, tmp_path, capsys, monkeypatch):
        import sys
        import types
        from mdatools.cli import gdrive_sync as cli_mod

        creds = tmp_path / "sa.json"
        creds.write_text("{}")
        local_dir = tmp_path / "data"
        local_dir.mkdir()

        mock_syncer = MagicMock()
        mock_syncer.sync.return_value = {
            "uploaded": [local_dir / "a.pdb"],
            "skipped":  [local_dir / "b.pdb"],
            "failed":   [],
        }
        MockClass = MagicMock(return_value=mock_syncer)

        monkeypatch.setattr(sys, "argv", [
            "mdatools-gdrive-sync",
            "--folder-id", "FOLDER",
            "--credentials", str(creds),
            "--local-dir", str(local_dir),
        ])
        fake_mod = types.ModuleType("mdatools.io.gdrive_sync")
        fake_mod.GDriveSync = MockClass
        monkeypatch.setitem(sys.modules, "mdatools.io.gdrive_sync", fake_mod)

        cli_mod.main()
        out = capsys.readouterr().out
        assert "1" in out  # uploaded count
        assert "Skipped" in out
