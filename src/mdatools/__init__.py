from ._version import __version__

# Phase 2: docking analysis sub-namespace (md-analysis-tools + docking-analysis-tools merger)
# Guard: mdatools.docking requires RDKit; skip gracefully when not installed.
try:
    from . import docking  # noqa: F401  — exposes mdatools.docking.*
    __all__ = ["__version__", "docking"]
except ImportError:
    __all__ = ["__version__"]
