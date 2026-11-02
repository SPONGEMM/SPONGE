import numpy as np
import pytest

from benchmarks.utils import Extractor, Runner


@pytest.mark.parametrize(
    "kind,periodic,box,position",
    [
        ("listed", True, 10.0, 8.5),
        ("listed", True, 20.0, 18.5),
        ("listed", False, 1000.0, 600.0),
        ("pairwise", True, 10.0, 8.5),
    ],
)
def test_jit_consumes_boundary(tmp_path, kind, periodic, box, position):
    """Compile and execute the actual listed/pairwise JIT argument ABI."""
    (tmp_path / "mass.txt").write_text("2\n12\n12\n")
    (tmp_path / "charge.txt").write_text("2\n0\n0\n")
    (tmp_path / "coordinate.txt").write_text(
        f"2 0\n0 0 0\n{position} 0 0\n{box} {box} {box}\n90 90 90\n"
    )
    if kind == "listed":
        descriptor = (
            "[[[ boundary_bond ]]]\n[[ parameters ]]\n"
            "int atom_i, int atom_j, float k_ij\n"
            "[[ potential ]]\nE = k_ij * r_ij * r_ij;\n[[ end ]]\n"
        )
        data = "1\n0 1 0.001\n"
        command = 'listed_forces_in_file = "force.txt"\n'
        command += 'boundary_bond_in_file = "parameters.txt"\n'
    else:
        descriptor = (
            "[[[ boundary_pair ]]]\n[[ potential ]]\n"
            "E = epsilon_ij * powf(sigma_ij / r_ij, 12.0f);\n"
            "[[ parameters ]]\nfloat epsilon_ij, float sigma_ij\n"
            "[[ with_ele ]]\nfalse\n[[ end ]]\n"
        )
        data = "2 1\n0.1\n1.0\n0\n0\n"
        command = 'pairwise_force_in_file = "force.txt"\n'
        command += 'boundary_pair_in_file = "parameters.txt"\n'
    (tmp_path / "force.txt").write_text(descriptor)
    (tmp_path / "parameters.txt").write_text(data)
    (tmp_path / "mdin.spg.toml").write_text(
        'md_name = "boundary JIT regression"\nmode = "nve"\n'
        "step_limit = 0\ndt = 0\n"
        f"pbc = {str(periodic).lower()}\n"
        f"cutoff = {4.0 if periodic else 999.0}\nskin = 0.4\n"
        'mass_in_file = "mass.txt"\ncharge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        "print_zeroth_frame = 1\nwrite_mdout_interval = 1\n"
        'write_trajectory_interval = 1\nfrc = "frc.dat"\n' + command
    )
    Runner.run_sponge(tmp_path, timeout=120, mpi_np=None)
    dr = position - (box if periodic else 0.0)
    force = (
        0.002 * dr if kind == "listed" else -12.0 * (0.1 / abs(dr) ** 12) / dr
    )
    np.testing.assert_allclose(
        Extractor.extract_sponge_forces(tmp_path, 2),
        [[force, 0.0, 0.0], [-force, 0.0, 0.0]],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
