import shutil

import numpy as np
import pytest
from ase.build import bulk

from benchmarks.utils import Outputer, Runner

from benchmarks.comparison.tests.lammps.tests.utils import (
    EV_TO_KCAL_MOL,
    extract_sponge_forces,
    extract_sponge_potential,
    extract_sponge_pressure,
    extract_sponge_stress,
    load_lammps_reference_entry,
    load_lammps_reference_forces,
    load_lammps_reference_stress,
    prepare_case_dir,
    write_lammps_data,
    write_sponge_coords,
    write_sponge_mass,
)


@pytest.mark.parametrize("iteration", range(3))
def test_cu_eam(
    iteration,
    statics_path,
    outputs_path,
    mpi_np,
):
    curr_perturbation = 0.1 * iteration
    case_dir = prepare_case_dir(outputs_path, "eam", iteration, mpi_np)
    lammps_dir = case_dir / "lammps"
    sponge_dir = case_dir / "sponge"

    shutil.copytree(
        statics_path / "eam" / "lammps", lammps_dir, dirs_exist_ok=True
    )
    shutil.copytree(
        statics_path / "eam" / "sponge", sponge_dir, dirs_exist_ok=True
    )

    # Use ASE to generate Cu FCC structure
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    atoms *= (14, 14, 14)  # 4 * 14^3 = 10976 atoms

    # Introduce randomness
    np.random.seed(42 + iteration * 100)
    if iteration > 0:
        atoms.positions += (
            np.random.rand(*atoms.positions.shape) - 0.5
        ) * curr_perturbation

    num_atoms = len(atoms)
    box = atoms.get_cell().diagonal()
    coords = atoms.positions

    # SPONGE coordinate file
    coord_file = sponge_dir / "system" / "test_coordinate.txt"
    write_sponge_coords(coord_file, coords, box)

    # SPONGE mass file
    mass_file = sponge_dir / "system" / "test_mass.txt"
    write_sponge_mass(mass_file, [63.546] * num_atoms)

    # LAMMPS data file
    data_file = lammps_dir / "data.lammps"
    write_lammps_data(data_file, coords, box, masses=[63.546])

    # Run SPONGE
    Runner.run_sponge(sponge_dir, mpi_np=mpi_np)

    ref_entry = load_lammps_reference_entry(statics_path, "cu_eam", iteration)
    assert abs(float(ref_entry["perturbation"]) - curr_perturbation) <= 1.0e-12
    assert int(ref_entry["natom"]) == num_atoms
    lammps_energy = float(ref_entry["energy"])
    sponge_energy = extract_sponge_potential(sponge_dir)

    lammps_pressure = float(ref_entry["pressure"])
    sponge_pressure = extract_sponge_pressure(sponge_dir)

    lammps_stress = load_lammps_reference_stress(
        statics_path, "cu_eam", iteration
    )
    sponge_stress = extract_sponge_stress(sponge_dir)

    lammps_forces = load_lammps_reference_forces(
        statics_path, "cu_eam", iteration
    )
    sponge_forces = extract_sponge_forces(sponge_dir, num_atoms)

    lammps_mag = np.linalg.norm(lammps_forces, axis=1)
    sponge_mag = np.linalg.norm(sponge_forces, axis=1)

    diff = np.abs(lammps_forces - sponge_forces)
    max_force_diff = np.max(diff)

    dot = np.sum(lammps_forces * sponge_forces, axis=1)
    mags = lammps_mag * sponge_mag
    valid = mags > 1e-6

    cos_sim_val = 1.0
    if iteration > 0:
        cos_sim = dot[valid] / mags[valid]
        cos_sim_val = np.mean(cos_sim)

    e_diff = abs(lammps_energy - sponge_energy)
    p_diff = abs(lammps_pressure - sponge_pressure)
    cos_sim_str = f"{cos_sim_val:.5e}" if iteration > 0 else "N/A"

    print(f"\n\nIteration: {iteration}, Perturbation: {curr_perturbation:.2e}")
    headers = ["Item", "LAMMPS", "SPONGE", "Diff", "Other"]
    rows = []

    # Energy
    rows.append(
        [
            "Energy",
            f"{lammps_energy:.6e}",
            f"{sponge_energy:.6e}",
            f"{e_diff:.4e}",
            "",
        ]
    )

    # Pressure
    rows.append(
        [
            "Pressure",
            f"{lammps_pressure:.4e}",
            f"{sponge_pressure:.4e}",
            f"{p_diff:.4e}",
            "",
        ]
    )

    # Stress components
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        l_val = lammps_stress.get(key, float("nan"))
        s_val = sponge_stress.get(key, float("nan"))
        d_val = abs(l_val - s_val)
        rows.append([key, f"{l_val:.4e}", f"{s_val:.4e}", f"{d_val:.4e}", ""])

    # Force Stats
    rows.append(["Force Max Diff", "", "", "", f"{max_force_diff:.4e}"])
    rows.append(["Cos Sim", "", "", "", cos_sim_str])

    Outputer.print_table(headers, rows)

    assert abs(sponge_energy - lammps_energy) / abs(lammps_energy) < 1e-4
    assert max_force_diff < 0.1
    assert abs(sponge_pressure - lammps_pressure) < 110.0
