"""Tests for PML builder (pure functions, no PyMOL required)."""

from pathlib import Path

from mdatools.pymol_bridge.pml_builder import (
    build_clean_snapshot_pml,
    build_dihedral_pml,
    build_validation_pml,
)


def test_dihedral_pml_contains_key_tokens():
    defs = [("C6-C7", "N3", "C6", "C7", "F1")]
    pml = build_dihedral_pml(
        pse_path=Path("/tmp/traj.pse"),
        output_csv=Path("/tmp/out.csv"),
        ligand_resname="UNK",
        dihedral_defs=defs,
    )
    assert "load /tmp/traj.pse" in pml
    assert "C6-C7" in pml
    assert "/tmp/out.csv" in pml
    assert "python end" in pml
    assert "quit" in pml


def test_validation_pml_contains_pocket_resids():
    pml = build_validation_pml(
        pse_path=Path("/tmp/traj.pse"),
        output_csv=Path("/tmp/conf.csv"),
        pocket_resids=[130, 290, 291],
        hbond_pairs=[("resi 290 and name OD1", "resname UNK and name N4")],
    )
    assert "130+290+291" in pml
    assert "/tmp/conf.csv" in pml
    assert "python end" in pml


def test_dihedral_pml_multiple_bonds():
    defs = [
        ("C6-C7", "N3", "C6", "C7", "F1"),
        ("C8-C9", "N3", "C8", "C9", "C10"),
    ]
    pml = build_dihedral_pml(
        pse_path=Path("/x/traj.pse"),
        output_csv=Path("/x/out.csv"),
        ligand_resname="LIG",
        dihedral_defs=defs,
    )
    assert "C6-C7" in pml
    assert "C8-C9" in pml
    assert "LIG" in pml


# ---------------------------------------------------------------------------
# build_clean_snapshot_pml
# ---------------------------------------------------------------------------

class TestBuildCleanSnapshotPml:

    def test_contains_output_dir(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap_0001.pdb")],
            output_dir=Path("/out/clean"),
        )
        assert "/out/clean" in pml

    def test_contains_input_pdb(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap_0001.pdb"), Path("/data/snap_0002.pdb")],
            output_dir=Path("/out"),
        )
        assert "snap_0001.pdb" in pml
        assert "snap_0002.pdb" in pml

    def test_default_remove_includes_water_and_ions(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap.pdb")],
            output_dir=Path("/out"),
        )
        assert "HOH" in pml
        assert "NA" in pml
        assert "CL" in pml

    def test_custom_remove_resnames(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap.pdb")],
            output_dir=Path("/out"),
            remove_resnames=["MYSOLVENT"],
        )
        assert "MYSOLVENT" in pml
        # default list should NOT appear when overridden
        assert "HOH" not in pml

    def test_script_ends_with_quit(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap.pdb")],
            output_dir=Path("/out"),
        )
        assert "quit" in pml

    def test_script_has_python_block(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[Path("/data/snap.pdb")],
            output_dir=Path("/out"),
        )
        assert "python" in pml
        assert "python end" in pml

    def test_empty_input_list(self):
        pml = build_clean_snapshot_pml(
            input_pdbs=[],
            output_dir=Path("/out"),
        )
        assert isinstance(pml, str)
        assert "python end" in pml
