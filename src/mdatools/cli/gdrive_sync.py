"""CLI: sync MD result files to Google Drive."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Differential upload of MD result files to Google Drive.\n"
            "Requires: pip install 'mdatools[gdrive]'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder-id", required=True,
        help="Google Drive folder ID (from the folder URL)",
    )
    parser.add_argument(
        "--credentials", required=True, type=Path,
        help="Path to service account key JSON or OAuth2 client secrets JSON",
    )
    parser.add_argument(
        "--local-dir", required=True, type=Path,
        help="Local directory to scan for files",
    )
    parser.add_argument(
        "--patterns", nargs="+",
        default=["*.pse", "*.pdb", "analysis_*.csv"],
        help="Glob patterns to match (default: *.pse *.pdb analysis_*.csv)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without uploading",
    )
    args = parser.parse_args()

    try:
        from mdatools.io.gdrive_sync import GDriveSync
    except ImportError:
        parser.error(
            "gdrive extra is not installed.\n"
            "Run: pip install 'mdatools[gdrive]'"
        )

    syncer = GDriveSync(folder_id=args.folder_id, credentials=args.credentials)
    result = syncer.sync(
        local_root=args.local_dir,
        patterns=args.patterns,
        dry_run=args.dry_run,
    )

    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"{prefix}Uploaded : {len(result['uploaded'])}")
    print(f"{prefix}Skipped  : {len(result['skipped'])}")
    print(f"{prefix}Failed   : {len(result['failed'])}")
    if result["failed"]:
        for p in result["failed"]:
            print(f"  FAILED: {p}")


if __name__ == "__main__":
    main()
