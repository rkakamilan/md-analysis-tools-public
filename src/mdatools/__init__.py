from ._version import __version__

# Guard: mdatools.docking requires RDKit; skip gracefully when not installed.
try:
    from . import docking  # noqa: F401  — exposes mdatools.docking.*
    __all__ = ["__version__", "docking"]
except ImportError:
    __all__ = ["__version__"]
