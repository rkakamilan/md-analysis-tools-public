from .loaders import discover_replicas, load_hbond_summary, load_hbond_events
from .writers import write_snapshot_pdb, write_summary_csv, write_clean_pdb

__all__ = [
    "discover_replicas",
    "load_hbond_summary",
    "load_hbond_events",
    "write_snapshot_pdb",
    "write_summary_csv",
    "write_clean_pdb",
    "GDriveSync",
]


def __getattr__(name: str):
    if name == "GDriveSync":
        from .gdrive_sync import GDriveSync
        return GDriveSync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
