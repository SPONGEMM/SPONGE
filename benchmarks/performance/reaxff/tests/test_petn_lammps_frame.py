import math
from pathlib import Path

import numpy as np

from benchmarks.utils import Extractor, Outputer, Runner

from benchmarks.comparison.tests.lammps.tests.utils import (
    extract_lammps_potential,
    extract_lammps_pressure,
    extract_lammps_stress,
)
from benchmarks.performance.reaxff.tests.utils import (
    read_atom_count_from_coordinate,
    write_reaxff_perf_mdin,
)


def _extract_lammps_charges(dump_path):
    charges = {}
    with open(dump_path, "r") as f:
        lines = f.readlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start_idx = i + 1
            break
    if start_idx == -1:
        raise ValueError(
            f"Invalid LAMMPS charge dump format: missing atomic section in {dump_path}"
        )

    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 2:
            charges[int(parts[0])] = float(parts[1])
    return charges


def _extract_sponge_charges(charge_path):
    charges = {}
    with open(charge_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                charges[int(parts[0])] = float(parts[1])
    return charges


def _extract_lammps_forces_from_dump(dump_path):
    forces = {}
    with open(dump_path, "r") as f:
        lines = f.readlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start_idx = i + 1
            break
    if start_idx == -1:
        raise ValueError(
            f"Invalid LAMMPS force dump format: missing atomic section in {dump_path}"
        )

    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            forces[int(parts[0])] = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
    return np.array([forces[i] for i in sorted(forces)], dtype=float)


def test_reaxff_petn_single_frame_matches_lammps(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="petn_16240",
        mpi_np=mpi_np,
        run_name="petn_lammps_frame",
    )

    write_reaxff_perf_mdin(
        case_dir,
        source_mdin="mdin.spg.toml",
        output_mdin="petn.frame.spg.toml",
        step_limit=0,
        write_information_interval=1,
        rst="petn_frame",
    )

    Runner.run_sponge(
        case_dir,
        mdin_name="petn.frame.spg.toml",
        timeout=2400,
        mpi_np=mpi_np,
    )

    reference_dir = case_dir / "reference"
    if not reference_dir.exists():
        raise FileNotFoundError(
            f"Missing PETN LAMMPS reference directory: {reference_dir}"
        )

    reference_in = reference_dir / "in.lammps"
    reference_log = reference_dir / "log.lammps"
    reference_forces_dump = reference_dir / "forces.dump"
    reference_charges_dump = reference_dir / "charges.dump"
    for path in (
        reference_in,
        reference_log,
        reference_forces_dump,
        reference_charges_dump,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing PETN reference file: {path}")

    atom_count = read_atom_count_from_coordinate(case_dir / "coordinate.txt")
    lammps_potential = extract_lammps_potential(reference_log)
    sponge_potential = Extractor.extract_sponge_potential(case_dir)
    lammps_pressure = extract_lammps_pressure(reference_log)
    sponge_pressure = Extractor.extract_sponge_pressure(case_dir)
    lammps_stress = extract_lammps_stress(reference_log)
    sponge_stress = Extractor.extract_sponge_stress(case_dir)

    lammps_forces = _extract_lammps_forces_from_dump(reference_forces_dump)
    sponge_forces = Extractor.extract_sponge_forces(case_dir, atom_count)
    lammps_charges = _extract_lammps_charges(reference_charges_dump)
    sponge_charges = _extract_sponge_charges(case_dir / "eeq_charges.txt")

    force_abs_diff = abs(lammps_forces - sponge_forces)
    max_force_diff = float(force_abs_diff.max())
    rms_force_diff = float((force_abs_diff**2).mean() ** 0.5)
    charge_abs_diff = np.array(
        [
            abs(lammps_charges[atom_id] - sponge_charges[atom_id])
            for atom_id in sorted(lammps_charges)
        ],
        dtype=float,
    )
    max_charge_diff = float(charge_abs_diff.max())
    rms_charge_diff = float((charge_abs_diff**2).mean() ** 0.5)

    pressure_diff = abs(lammps_pressure - sponge_pressure)
    stress_diffs = {
        key: abs(lammps_stress[key] - sponge_stress[key])
        for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    }
    potential_diff = abs(lammps_potential - sponge_potential)
    rel_potential_diff = potential_diff / max(abs(lammps_potential), 1.0)

    rows = [
        ["Case", "petn_16240"],
        ["Atoms", str(atom_count)],
        ["LAMMPS PE", f"{lammps_potential:.6f}"],
        ["SPONGE PE", f"{sponge_potential:.6f}"],
        ["Abs PE Diff", f"{potential_diff:.6f}"],
        ["Rel PE Diff", f"{rel_potential_diff:.6e}"],
        ["LAMMPS Press", f"{lammps_pressure:.6f}"],
        ["SPONGE Press", f"{sponge_pressure:.6f}"],
        ["Abs Press Diff", f"{pressure_diff:.6f}"],
        ["Max Charge Diff", f"{max_charge_diff:.6f}"],
        ["RMS Charge Diff", f"{rms_charge_diff:.6f}"],
        ["Max Force Diff", f"{max_force_diff:.6f}"],
        ["RMS Force Diff", f"{rms_force_diff:.6f}"],
    ]
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        rows.append([f"Abs {key} Diff", f"{stress_diffs[key]:.6f}"])

    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Performance Validation: ReaxFF PETN Single-Frame vs LAMMPS",
    )

    assert math.isfinite(lammps_potential)
    assert math.isfinite(sponge_potential)
    assert math.isfinite(lammps_pressure)
    assert math.isfinite(sponge_pressure)
    assert rel_potential_diff <= 5.0e-3
    assert max_charge_diff <= 5.0e-2
    assert rms_charge_diff <= 2.5e-2
    assert max_force_diff <= 15.0
    assert rms_force_diff <= 5.0
