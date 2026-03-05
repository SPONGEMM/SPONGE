import pytest
import shutil
import numpy as np
from utils import (
    load_lammps_reference_entry,
    load_lammps_reference_forces,
    load_lammps_reference_stress,
    extract_sponge_forces,
    extract_sponge_potential,
    extract_sponge_pressure,
    extract_sponge_stress,
    write_lammps_charge_data,
    write_sponge_coords,
    write_sponge_mass,
    write_sponge_types,
    print_validation_table,
    run_sponge_command,
)


def _within_hard_threshold(diff, ref, atol=2.0, rtol=1e-3):
    return diff <= (atol + rtol * max(abs(ref), 1.0))


@pytest.mark.parametrize("iteration", range(5))
def test_reaxff_h2(
    iteration,
    statics_path,
    outputs_path,
):
    curr_perturbation = 0.1 * iteration
    print(f"\n\nIteration: {iteration}, Perturbation: {curr_perturbation:.2e}")

    static_dir = statics_path / "reaxff_h2"
    reaxff_static_dir = statics_path / "reaxff"
    case_dir = outputs_path / "reaxff_h2" / str(iteration)
    lammps_dir = case_dir / "lammps"
    sponge_dir = case_dir / "sponge"

    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(static_dir / "lammps", lammps_dir, dirs_exist_ok=True)
    shutil.copytree(static_dir / "sponge", sponge_dir, dirs_exist_ok=True)

    # 复制势函数文件
    shutil.copy(
        reaxff_static_dir / "lammps" / "ffield.reax.cho",
        lammps_dir / "ffield.reax.cho",
    )
    shutil.copy(
        reaxff_static_dir / "lammps" / "ffield.reax.cho",
        sponge_dir / "ffield.reax.cho",
    )
    # 复制lmp_control
    shutil.copy(
        reaxff_static_dir / "lammps" / "lmp_control",
        lammps_dir / "lmp_control",
    )

    # 生成H2分子坐标
    np.random.seed(12345 + iteration)
    r = 1.0 + np.random.rand() * curr_perturbation
    box_size = 25.0
    half_box = box_size / 2.0

    coords = np.array(
        [[half_box, half_box, half_box], [half_box + r, half_box, half_box]]
    )

    # 生成LAMMPS data文件
    write_lammps_charge_data(
        lammps_dir / "data.lammps",
        coords,
        [box_size, box_size, box_size],
        masses={"H": 1.008},
        atom_types=["H", "H"],
        type_id_map={"H": 1},
        title="H2 ReaxFF",
    )

    # 生成SPONGE coordinate文件
    write_sponge_coords(
        sponge_dir / "coordinate.txt",
        coords,
        [box_size, box_size, box_size],
    )

    # 生成SPONGE mass文件
    write_sponge_mass(sponge_dir / "mass.txt", [1.008, 1.008])

    # 生成SPONGE type文件
    write_sponge_types(sponge_dir / "type.txt", ["H", "H"])

    # 运行SPONGE
    run_sponge_command(sponge_dir)

    ref_entry = load_lammps_reference_entry(
        statics_path, "reaxff_h2", iteration
    )
    assert abs(float(ref_entry["perturbation"]) - curr_perturbation) <= 1.0e-12
    assert int(ref_entry["natom"]) == 2
    lammps_energy = float(ref_entry["energy"])
    sponge_energy = extract_sponge_potential(sponge_dir)

    lammps_forces = load_lammps_reference_forces(
        statics_path, "reaxff_h2", iteration
    )
    sponge_forces = extract_sponge_forces(sponge_dir, 2)

    energy_diff = abs(lammps_energy - sponge_energy)
    force_diff = np.max(np.abs(lammps_forces - sponge_forces))
    lammps_pressure = float(ref_entry["pressure"])
    lammps_stress = load_lammps_reference_stress(
        statics_path, "reaxff_h2", iteration
    )
    sponge_pressure = extract_sponge_pressure(sponge_dir)
    sponge_stress = extract_sponge_stress(sponge_dir)
    pressure_diff = abs(lammps_pressure - sponge_pressure)
    max_stress_diff = max(
        abs(lammps_stress[k] - sponge_stress[k])
        for k in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    )

    headers = ["Term", "LAMMPS", "SPONGE", "Diff", "Status"]
    e_status = "PASS" if energy_diff < 1e-2 else "FAIL"
    f_status = "PASS" if force_diff < 1e-2 else "FAIL"
    p_pass = _within_hard_threshold(pressure_diff, lammps_pressure)
    p_status = "PASS" if p_pass else "FAIL"
    s_pass = _within_hard_threshold(
        max_stress_diff,
        max(
            abs(lammps_stress[k])
            for k in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
        ),
    )
    s_status = "PASS" if s_pass else "FAIL"

    rows = [
        [
            "Total Energy",
            f"{lammps_energy:.6f}",
            f"{sponge_energy:.6f}",
            f"{energy_diff:.4e}",
            e_status,
        ],
        ["Max Force", "N/A", "N/A", f"{force_diff:.4e}", f_status],
        [
            "Pressure",
            f"{lammps_pressure:.6f}",
            f"{sponge_pressure:.6f}",
            f"{pressure_diff:.4e}",
            p_status,
        ],
        ["Max Stress Diff", "N/A", "N/A", f"{max_stress_diff:.4e}", s_status],
    ]
    stress_components_pass = True
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        diff = abs(lammps_stress[key] - sponge_stress[key])
        comp_pass = _within_hard_threshold(diff, lammps_stress[key])
        if not comp_pass:
            stress_components_pass = False
        rows.append(
            [
                key,
                f"{lammps_stress[key]:.6f}",
                f"{sponge_stress[key]:.6f}",
                f"{diff:.4e}",
                "PASS" if comp_pass else "FAIL",
            ]
        )
    print_validation_table(headers, rows, title=f"H2 Distance: {r:.4f} A")

    assert energy_diff < 1e-2
    assert force_diff < 1e-2
    assert np.isfinite(lammps_pressure) and np.isfinite(sponge_pressure)
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        assert np.isfinite(lammps_stress[key]) and np.isfinite(
            sponge_stress[key]
        )
    assert p_pass
    assert s_pass
    assert stress_components_pass
