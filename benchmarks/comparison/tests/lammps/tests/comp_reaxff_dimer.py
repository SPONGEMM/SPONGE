import pytest
import shutil
import numpy as np
from utils import (
    load_lammps_reference_entry,
    load_lammps_reference_forces,
    load_lammps_reference_stress,
    load_lammps_reference_thermo,
    extract_sponge_forces,
    extract_sponge_pressure,
    extract_sponge_stress,
    write_lammps_charge_data,
    write_sponge_coords,
    write_sponge_mass,
    write_sponge_types,
    print_validation_table,
    run_sponge_command,
)
from Xponge.analysis import MdoutReader


def _within_hard_threshold(diff, ref, atol=2.0, rtol=1e-3):
    return diff <= (atol + rtol * max(abs(ref), 1.0))


EEQ_TOL = 0.2
POTENTIAL_TOL = 0.2


@pytest.mark.parametrize("iteration", range(3))
def test_reaxff_dimer(
    iteration,
    statics_path,
    outputs_path,
):
    curr_perturbation = 0.1 * iteration
    print(f"\n\nIteration: {iteration}, Perturbation: {curr_perturbation:.2e}")
    static_dir = statics_path / "reaxff_dimer"
    case_dir = outputs_path / "reaxff_dimer" / str(iteration)
    lammps_dir = case_dir / "lammps"
    sponge_dir = case_dir / "sponge"

    shutil.copytree(static_dir / "lammps", lammps_dir, dirs_exist_ok=True)
    shutil.copytree(static_dir / "sponge", sponge_dir, dirs_exist_ok=True)

    shutil.copy(
        statics_path / "reaxff" / "lammps" / "ffield.reax.cho",
        lammps_dir / "ffield.reax.cho",
    )
    shutil.copy(
        statics_path / "reaxff" / "lammps" / "ffield.reax.cho",
        sponge_dir / "ffield.reax.cho",
    )
    shutil.copy(
        statics_path / "reaxff" / "lammps" / "lmp_control",
        lammps_dir / "lmp_control",
    )

    # 生成甲酸二聚体坐标 (HCOOH)2
    np.random.seed(12345 + iteration)
    box_size = 30.0
    offset = box_size / 2.0

    # 基本单位 (HCOOH)
    # C, O(carbonyl), O(hydroxyl), H(hydroxyl), H(on C)
    mol1 = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [1.2, 0.0, 0.0],  # O1
            [-0.6, 1.1, 0.0],  # O2
            [-1.5, 0.8, 0.0],  # H1
            [-0.5, -0.9, 0.0],  # H2
        ]
    )

    # 扰动
    mol1 += np.random.rand(5, 3) * curr_perturbation

    # 第二个分子 (镜像并平移)
    mol2 = mol1.copy()
    mol2[:, 0] = 4.0 - mol2[:, 0]
    mol2[:, 1] = 1.1 - mol2[:, 1]

    coords = np.vstack([mol1, mol2]) + offset

    # 元素类型 (C, O, O, H, H, C, O, O, H, H)
    types = ["C", "O", "O", "H", "H", "C", "O", "O", "H", "H"]
    masses = {"C": 12.011, "O": 15.999, "H": 1.008}
    type_id_map = {"C": 1, "H": 2, "O": 3}

    write_lammps_charge_data(
        lammps_dir / "data.lammps",
        coords,
        [box_size, box_size, box_size],
        masses=masses,
        atom_types=types,
        type_id_map=type_id_map,
        title="Formic Acid Dimer ReaxFF",
    )

    # SPONGE coordinate文件
    write_sponge_coords(
        sponge_dir / "coordinate.txt",
        coords,
        [box_size, box_size, box_size],
    )

    # SPONGE mass文件
    write_sponge_mass(sponge_dir / "mass.txt", [masses[t] for t in types])

    # SPONGE type文件
    write_sponge_types(sponge_dir / "type.txt", types)

    run_sponge_command(sponge_dir)

    ref_entry = load_lammps_reference_entry(
        statics_path, "reaxff_dimer", iteration
    )
    assert abs(float(ref_entry["perturbation"]) - curr_perturbation) <= 1.0e-12
    assert int(ref_entry["natom"]) == 10
    lammps_thermo = load_lammps_reference_thermo(
        statics_path, "reaxff_dimer", iteration
    )

    # 提取SPONGE能量
    mdout = MdoutReader(str(sponge_dir / "mdout.txt"))

    # 映射表: (SPONGE_Header, LAMMPS_Variable, Tolerance)
    energy_maps = [
        ("REAXFF_BOND", "v_eb", 0.01),
        ("REAXFF_OVUN", "v_ea", 0.01),
        ("REAXFF_ELP", "v_elp", 0.01),
        ("REAXFF_ANG", "v_ev", 0.01),
        ("REAXFF_PEN", "v_epen", 0.01),
        ("REAXFF_COA", "v_ecoa", 0.01),
        ("REAXFF_HB", "v_ehb", 0.01),
        ("REAXFF_TOR", "v_et", 0.01),
        ("REAXFF_CONJ", "v_eco", 0.01),
        ("REAXFF_VDW", "v_ew", 0.01),
    ]

    headers = ["Term", "LAMMPS", "SPONGE", "Diff", "Status"]
    rows = []

    all_pass = True
    for s_key, l_key, tol in energy_maps:
        l_val = lammps_thermo[l_key]
        s_val = getattr(mdout, s_key)[0]
        diff = abs(l_val - s_val)
        status = "PASS" if diff < tol else "FAIL"
        rows.append(
            [s_key, f"{l_val:.4f}", f"{s_val:.4f}", f"{diff:.4e}", status]
        )
        if status == "FAIL":
            all_pass = False

    # Coulomb/EEQ 特殊处理
    l_coul = lammps_thermo["v_ep"] + lammps_thermo["v_eqeq"]
    s_coul = mdout.REAXFF_EEQ[0]
    coul_diff = abs(l_coul - s_coul)
    c_status = "PASS" if coul_diff < EEQ_TOL else "FAIL"
    rows.append(
        [
            "REAXFF_EEQ",
            f"{l_coul:.4f}",
            f"{s_coul:.4f}",
            f"{coul_diff:.4e}",
            c_status,
        ]
    )
    if c_status == "FAIL":
        all_pass = False

    # 总势能
    l_pot = lammps_thermo["PotEng"]
    s_pot = mdout.potential[0]
    pot_diff = abs(l_pot - s_pot)
    p_status = "PASS" if pot_diff < POTENTIAL_TOL else "FAIL"
    rows.append(
        [
            "Potential",
            f"{l_pot:.4f}",
            f"{s_pot:.4f}",
            f"{pot_diff:.4e}",
            p_status,
        ]
    )
    if p_status == "FAIL":
        all_pass = False

    # 提取力
    lammps_forces = load_lammps_reference_forces(
        statics_path, "reaxff_dimer", iteration
    )
    sponge_forces = extract_sponge_forces(sponge_dir, 10)

    force_diff = np.max(np.abs(lammps_forces - sponge_forces))
    f_status = "PASS" if force_diff < 0.05 else "FAIL"
    rows.append(["Max Force Diff", "N/A", "N/A", f"{force_diff:.4e}", f_status])

    if f_status == "FAIL":
        all_pass = False

    # 压强和应力
    lammps_pressure = float(ref_entry["pressure"])
    lammps_stress = load_lammps_reference_stress(
        statics_path, "reaxff_dimer", iteration
    )
    sponge_pressure = extract_sponge_pressure(sponge_dir)
    sponge_stress = extract_sponge_stress(sponge_dir)

    pressure_diff = abs(lammps_pressure - sponge_pressure)
    p_pass = _within_hard_threshold(pressure_diff, lammps_pressure)
    p_status = "PASS" if p_pass else "FAIL"
    rows.append(
        [
            "Pressure",
            f"{lammps_pressure:.4f}",
            f"{sponge_pressure:.4f}",
            f"{pressure_diff:.4e}",
            p_status,
        ]
    )
    if not p_pass:
        all_pass = False
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        diff = abs(lammps_stress[key] - sponge_stress[key])
        s_pass = _within_hard_threshold(diff, lammps_stress[key])
        s_status = "PASS" if s_pass else "FAIL"
        rows.append(
            [
                key,
                f"{lammps_stress[key]:.4f}",
                f"{sponge_stress[key]:.4f}",
                f"{diff:.4e}",
                s_status,
            ]
        )
        if not s_pass:
            all_pass = False

    print_validation_table(headers, rows)

    assert np.isfinite(lammps_pressure) and np.isfinite(sponge_pressure)
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        assert np.isfinite(lammps_stress[key]) and np.isfinite(
            sponge_stress[key]
        )
    assert all_pass, "Some ReaxFF energy/force/pressure/stress terms mismatch"
