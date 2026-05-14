"""Abstract base class for fingerprint calculators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FingerprintCalculator(ABC):
    """Protocol-style ABC for protein-ligand fingerprint calculators."""

    @abstractmethod
    def calculate(
        self,
        poses: list,
        protein: object,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute interaction fingerprints for a list of docked poses."""
