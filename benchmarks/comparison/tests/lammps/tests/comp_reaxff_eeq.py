import pytest
import shutil
from utils import (
    generate_perturbed_water_system,
    load_lammps_reference_charges,
    load_lammps_reference_entry,
    print_validation_table,
    run_sponge_command,
    write_lammps_charge_data,
    write_sponge_coords,
)


def extract_lammps_charges(dump_path):
    charges = {}
    with open(dump_path, "r") as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if "ITEM: ATOMS" in line:
                start = True
                continue
            if start:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        charges[int(parts[0])] = float(parts[1])
                    except ValueError:
                        continue
    return charges


def extract_sponge_charges(charge_file):
    charges = {}
    if not charge_file.exists():
        return charges
    with open(charge_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    charges[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    return charges


@pytest.mark.parametrize("iteration", range(3))
def test_reaxff_eeq(
    iteration,
    statics_path,
    outputs_path,
):
    curr_perturbation = 0.1 * iteration
    print(f"\n\nIteration: {iteration}, Perturbation: {curr_perturbation:.2e}")

    static_dir = statics_path / "reaxff"
    case_dir = outputs_path / "reaxff_eeq" / str(iteration)
    lammps_dir = case_dir / "lammps"
    sponge_dir = case_dir / "sponge"

    shutil.copytree(static_dir / "lammps", lammps_dir, dirs_exist_ok=True)
    shutil.copytree(static_dir / "sponge", sponge_dir, dirs_exist_ok=True)

    nx, ny, nz = 15, 15, 15
    spacing = 4.0

    coords, box_size, atom_types = generate_perturbed_water_system(
        nx=nx,
        ny=ny,
        nz=nz,
        spacing=spacing,
        perturbation=curr_perturbation,
        seed=1919 + iteration * 114514,
    )

    masses = {"O": 15.999, "H": 1.008}
    lammps_type_map = {"O": 1, "H": 2}

    num_atoms = len(coords)

    write_lammps_charge_data(
        lammps_dir / "data.lammps",
        coords,
        box_size,
        masses=masses,
        atom_types=atom_types,
        type_id_map=lammps_type_map,
        title="Water ReaxFF 10k perturbed",
    )

    write_sponge_coords(sponge_dir / "coordinate.txt", coords, box_size)

    run_sponge_command(sponge_dir)

    ref_entry = load_lammps_reference_entry(
        statics_path, "reaxff_eeq", iteration
    )
    assert abs(float(ref_entry["perturbation"]) - curr_perturbation) <= 1.0e-12
    assert int(ref_entry["natom"]) == num_atoms
    lammps_charges = load_lammps_reference_charges(
        statics_path, "reaxff_eeq", iteration
    )
    sponge_charges = extract_sponge_charges(sponge_dir / "eeq_charges.txt")

    assert len(lammps_charges) == num_atoms
    assert len(sponge_charges) == num_atoms

    max_diff = 0.0
    for i in range(1, num_atoms + 1):
        diff = abs(lammps_charges[i - 1] - sponge_charges[i])
        max_diff = max(max_diff, diff)

    headers = ["Term", "Max Diff", "Status"]
    status = "PASS" if max_diff < 0.01 else "FAIL"
    rows = [["Charge", f"{max_diff:.6e}", status]]
    print_validation_table(headers, rows)

    assert max_diff < 0.01
