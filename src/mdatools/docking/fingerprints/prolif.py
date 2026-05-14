"""ProLIF-backed protein-ligand interaction fingerprint calculator.

Wraps the ProLIF library with a stable, backend-agnostic interface.
The resulting DataFrame uses flat column names of the form
``<RESIDUE>_<InteractionType>`` (e.g. ``GLN30_HBAcceptor``).

"""

from __future__ import annotations

import pandas as pd

from mdatools.docking.fingerprints.base import FingerprintCalculator


def _require_prolif() -> object:
    try:
        import prolif  # noqa: PLC0415

        return prolif
    except ImportError as exc:
        raise ImportError(
            "prolif is required for ProLIFCalculator. "
            "Install it with:  pip install 'mdatools[docking]'"
        ) from exc


def _require_rdkit() -> object:
    try:
        from rdkit import Chem  # noqa: PLC0415

        return Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for ProLIFCalculator. "
            "Install it with:  pip install 'mdatools[docking]'"
        ) from exc


class ProLIFCalculator(FingerprintCalculator):
    """Calculate protein-ligand interaction fingerprints using ProLIF.

    Parameters
    ----------
    interaction_types:
        List of ProLIF interaction type names to compute.  Pass ``None``
        (default) to use ProLIF's built-in defaults.
    count_fingerprint:
        If ``True``, count interactions instead of binary 0/1.
    """

    def __init__(
        self,
        interaction_types: list[str] | None = None,
        count_fingerprint: bool = False,
    ) -> None:
        self._interaction_types = interaction_types
        self._count = count_fingerprint

    def calculate(
        self,
        poses: list,
        protein: object,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute interaction fingerprints for a list of docked poses.

        Parameters
        ----------
        poses:
            List of RDKit Mols, one conformer each.
        protein:
            Receptor structure as an RDKit Mol (with hydrogens).
        show_progress:
            Show a ProLIF progress bar.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_poses, n_interactions)``.
            Columns: ``<RESIDUE>_<InteractionType>`` (int8 0/1).
            Index aligns with the input *poses* list.

        Raises
        ------
        ValueError
            If *poses* is empty or *protein* is ``None``.
        ImportError
            If ``prolif`` or ``rdkit`` are not installed.
        """
        if not poses:
            raise ValueError("poses list is empty")
        if protein is None:
            raise ValueError("protein is None")

        plf = _require_prolif()

        plf_protein = plf.Molecule(protein)
        plf_ligands = [plf.Molecule.from_rdkit(m) for m in poses]

        fp_kwargs: dict = {}
        if self._interaction_types is not None:
            fp_kwargs["interactions"] = self._interaction_types
        if self._count:
            fp_kwargs["count"] = True

        fp = plf.Fingerprint(**fp_kwargs)
        fp.run_from_iterable(plf_ligands, plf_protein, progress=show_progress)

        df = fp.to_dataframe()

        if df.empty or df.shape[1] == 0:
            return pd.DataFrame(index=range(len(poses)))

        # Flatten MultiIndex: (residue, ligand, interaction) → RESIDUE_Interaction
        df = df.droplevel("ligand", axis=1).astype("int8")
        df.columns = [f"{col[0].split('.')[0]}_{col[1]}" for col in df.columns.values]
        df = df.reset_index(drop=True)
        return df

    def summarize_by_residue(
        self,
        fp_df: pd.DataFrame,
        residues: list[str],
    ) -> pd.DataFrame:
        """Add per-residue interaction summary columns.

        For each residue, adds a binary column ``int_w_<residue_lower>``
        that is ``1`` when the pose interacts with any interaction type
        involving that residue.

        Parameters
        ----------
        fp_df:
            DataFrame returned by :meth:`calculate`.
        residues:
            Residue identifiers (e.g. ``["GLN30", "ARG38"]``).

        Returns
        -------
        pd.DataFrame
        """
        df = fp_df.copy()
        for res in residues:
            col_name = f"int_w_{res.lower()}"
            matching = df.columns[df.columns.str.startswith(res + "_")]
            df[col_name] = (
                df[matching].any(axis=1).astype("int8") if len(matching) > 0 else 0
            )
        return df

    @staticmethod
    def add_hbond_summary_flags(fp_df: pd.DataFrame) -> pd.DataFrame:
        """Add ``has_hbdonor`` and ``has_hbacceptor`` global summary columns.

        Parameters
        ----------
        fp_df:
            DataFrame returned by :meth:`calculate`.

        Returns
        -------
        pd.DataFrame
        """
        df = fp_df.copy()
        hbdonor_cols = df.columns[df.columns.str.endswith("HBDonor")]
        hbacceptor_cols = df.columns[df.columns.str.endswith("HBAcceptor")]
        df["has_hbdonor"] = (
            df[hbdonor_cols].any(axis=1).astype("int8") if len(hbdonor_cols) > 0 else 0
        )
        df["has_hbacceptor"] = (
            df[hbacceptor_cols].any(axis=1).astype("int8")
            if len(hbacceptor_cols) > 0
            else 0
        )
        return df
