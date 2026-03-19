import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Extractor
from benchmarks.utils import Outputer

from benchmarks.performance.amber.tests.utils import (
    compute_oo_rdf,
    read_box_trajectory,
    read_coordinate_trajectory,
    run_sponge_amber,
    save_rdf_plot,
    summarize_rdf,
    write_amber_long_run_mdin,
)


def parse_mdout_column(mdout_path, column_name):
    rows = Extractor.parse_mdout_rows(mdout_path, [column_name], int_columns=())
    return [row[column_name] for row in rows]


PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYX",
    "CYM",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASH",
    "GLH",
    "LYN",
    "ACE",
    "NME",
    "NHE",
    "NH2",
}
RNA_RESNAMES = {"A", "U", "G", "C", "RA", "RU", "RG", "RC"}
PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
RNA_BACKBONE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O1P",
    "O2P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
}

MINIMIZATION_STEP_LIMIT = 1000
NPT_STEP_LIMIT = 2000
TAIL_SAMPLES = 50
WATER_RESNAMES = {
    "WAT",
    "HOH",
    "SOL",
    "TIP3",
    "TIP3P",
    "TIP4",
    "TIP4P",
    "SPC",
    "SPCE",
}
WATER_OXYGEN_NAMES = {"O", "OW", "OH2"}


def _read_prmtop_flag_lines(prmtop_path: Path, flag: str):
    lines = prmtop_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("%FLAG ") and lines[i][6:].strip() == flag:
            i += 1
            if i < len(lines) and lines[i].startswith("%FORMAT"):
                i += 1
            out = []
            while i < len(lines) and not lines[i].startswith("%FLAG "):
                out.append(lines[i])
                i += 1
            return out
        i += 1
    return []


def _read_prmtop_a4(prmtop_path: Path, flag: str):
    chunks = "".join(_read_prmtop_flag_lines(prmtop_path, flag))
    values = [chunks[i : i + 4].strip() for i in range(0, len(chunks), 4)]
    return [v for v in values if v]


def _read_prmtop_int(prmtop_path: Path, flag: str):
    return [
        int(x)
        for x in " ".join(_read_prmtop_flag_lines(prmtop_path, flag)).split()
    ]


def _build_atom_residue_names(prmtop_path: Path):
    atom_names = _read_prmtop_a4(prmtop_path, "ATOM_NAME")
    residue_names = _read_prmtop_a4(prmtop_path, "RESIDUE_LABEL")
    residue_pointers = _read_prmtop_int(prmtop_path, "RESIDUE_POINTER")

    atom_count = len(atom_names)
    atom_residue_name = [""] * atom_count
    for i, start_1based in enumerate(residue_pointers):
        start = start_1based - 1
        end = (
            residue_pointers[i + 1] - 1
            if i + 1 < len(residue_pointers)
            else atom_count
        )
        residue_name = residue_names[i]
        for atom_idx in range(start, end):
            atom_residue_name[atom_idx] = residue_name

    return atom_names, atom_residue_name


def _build_backbone_restrain_list(case_dir: Path):
    prmtop_path = case_dir / "model-protein-RNA-complex.prmtop"
    atom_names, atom_residue_name = _build_atom_residue_names(prmtop_path)

    restrained_ids = []
    classes = Counter()
    for atom_idx, atom_name in enumerate(atom_names):
        residue_name = atom_residue_name[atom_idx]
        if (
            residue_name in PROTEIN_RESNAMES
            and atom_name in PROTEIN_BACKBONE_ATOMS
        ):
            restrained_ids.append(atom_idx)
            classes["protein_backbone"] += 1
        elif residue_name in RNA_RESNAMES and atom_name in RNA_BACKBONE_ATOMS:
            restrained_ids.append(atom_idx)
            classes["rna_backbone"] += 1

    if not restrained_ids:
        raise ValueError("No protein/RNA backbone atoms selected for restrain")

    restrain_file = case_dir / "backbone_restrain_atom_id.txt"
    restrain_file.write_text("\n".join(str(i) for i in restrained_ids) + "\n")
    protein_atom_ids = [
        i
        for i, residue_name in enumerate(atom_residue_name)
        if residue_name in PROTEIN_RESNAMES
    ]
    rna_atom_ids = [
        i
        for i, residue_name in enumerate(atom_residue_name)
        if residue_name in RNA_RESNAMES
    ]
    return (
        restrain_file.name,
        restrained_ids,
        classes,
        protein_atom_ids,
        rna_atom_ids,
    )


def _build_water_oxygen_indices(prmtop_path: Path):
    atom_names, atom_residue_name = _build_atom_residue_names(prmtop_path)
    oxygen_ids = [
        atom_idx
        for atom_idx, atom_name in enumerate(atom_names)
        if atom_residue_name[atom_idx] in WATER_RESNAMES
        and atom_name in WATER_OXYGEN_NAMES
    ]
    if not oxygen_ids:
        raise ValueError("No water oxygen atoms found in AMBER system")
    return oxygen_ids


def _read_rst7_coords(rst7_path: Path):
    lines = rst7_path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid rst7 (too short): {rst7_path}")

    head = lines[1].split()
    if not head:
        raise ValueError(f"Invalid rst7 count line: {rst7_path}")
    atom_count = int(head[0])

    target_values = atom_count * 3
    values = []
    line_idx = 2
    while line_idx < len(lines) and len(values) < target_values:
        line = lines[line_idx]
        for i in range(0, len(line), 12):
            chunk = line[i : i + 12]
            if chunk.strip():
                values.append(float(chunk))
                if len(values) == target_values:
                    break
        line_idx += 1

    if len(values) != target_values:
        raise ValueError(
            f"Failed to parse rst7 coordinates: {rst7_path}, "
            f"expected {target_values}, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64).reshape(atom_count, 3)


def _kabsch_rmsd(reference_xyz, mobile_xyz):
    ref = np.asarray(reference_xyz, dtype=np.float64)
    mob = np.asarray(mobile_xyz, dtype=np.float64)
    if ref.shape != mob.shape:
        raise ValueError(
            f"Coordinate shape mismatch for RMSD: {ref.shape} vs {mob.shape}"
        )
    if ref.shape[0] == 0:
        raise ValueError("Cannot compute RMSD for empty atom set")

    ref_centered = ref - ref.mean(axis=0)
    mob_centered = mob - mob.mean(axis=0)
    cov = mob_centered.T @ ref_centered
    v, _, wt = np.linalg.svd(cov)
    det_sign = 1.0 if np.linalg.det(v @ wt) >= 0.0 else -1.0
    rotation = v @ np.diag([1.0, 1.0, det_sign]) @ wt
    mob_fitted = mob_centered @ rotation
    diff = mob_fitted - ref_centered
    return math.sqrt(np.mean(np.sum(diff * diff, axis=1)))


def _write_minimization_mdin(case_dir: Path, restrain_file: str):
    mdin = (
        'md_name = "validation restrain_npt minimization"\n'
        'mode = "minimization"\n'
        'amber_parm7 = "model-protein-RNA-complex.prmtop"\n'
        'amber_rst7 = "model-protein-RNA-complex.inpcrd"\n'
        "cutoff = 8.0\n"
        f"step_limit = {MINIMIZATION_STEP_LIMIT}\n"
        "write_information_interval = 1000\n"
        "write_mdout_interval = 1000\n"
        "print_zeroth_frame = 1\n"
        "minimization_dynamic_dt = 1\n"
        "minimization_max_move = 0.05\n"
        'constrain_mode = "SHAKE"\n'
        f'restrain_atom_id = "{restrain_file}"\n'
        'restrain_refcoord_scaling = "all"\n'
        "restrain_single_weight = 5.0\n"
    )
    (case_dir / "mdin.spg.toml").write_text(mdin)


def _write_restrained_npt_mdin(case_dir: Path, restrain_file: str):
    mdin = (
        'md_name = "validation restrain_npt restrained npt"\n'
        'mode = "npt"\n'
        'amber_parm7 = "model-protein-RNA-complex.prmtop"\n'
        'amber_rst7 = "restart_min.rst7"\n'
        "amber_irest = 0\n"
        "cutoff = 8.0\n"
        "dt = 0.002\n"
        f"step_limit = {NPT_STEP_LIMIT}\n"
        "write_information_interval = 1000\n"
        "write_mdout_interval = 1000\n"
        "print_zeroth_frame = 1\n"
        'thermostat = "middle_langevin"\n'
        "langevin_gamma = 10.0\n"
        "target_temperature = 300.0\n"
        'barostat = "andersen_barostat"\n'
        "barostat_tau = 0.1\n"
        "barostat_update_interval = 10\n"
        "target_pressure = 1.0\n"
        'constrain_mode = "SHAKE"\n'
        f'restrain_atom_id = "{restrain_file}"\n'
        'restrain_refcoord_scaling = "all"\n'
        "restrain_single_weight = 5.0\n"
    )
    (case_dir / "mdin.spg.toml").write_text(mdin)


def _write_unrestrained_npt_mdin(case_dir: Path):
    mdin = (
        'md_name = "validation restrain_npt unrestrained npt"\n'
        'mode = "npt"\n'
        'amber_parm7 = "model-protein-RNA-complex.prmtop"\n'
        'amber_rst7 = "restart_restrained.rst7"\n'
        "amber_irest = 1\n"
        "cutoff = 8.0\n"
        "dt = 0.002\n"
        f"step_limit = {NPT_STEP_LIMIT}\n"
        "write_information_interval = 1000\n"
        "write_mdout_interval = 1000\n"
        "print_zeroth_frame = 1\n"
        'thermostat = "middle_langevin"\n'
        "langevin_gamma = 10.0\n"
        "target_temperature = 300.0\n"
        'barostat = "andersen_barostat"\n'
        "barostat_tau = 0.1\n"
        "barostat_update_interval = 10\n"
        "target_pressure = 1.0\n"
        'constrain_mode = "SHAKE"\n'
    )
    (case_dir / "mdin.spg.toml").write_text(mdin)


def _first_nonfinite_index(series):
    for i, value in enumerate(series):
        if not math.isfinite(value):
            return i
    return None


@pytest.fixture(scope="module")
def amber_equilibrated_case(statics_path, outputs_path, mpi_np):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="restrain_npt",
        mpi_np=mpi_np,
        run_name="restrain_npt_equilibrated",
    )

    (
        restrain_file,
        restrained_ids,
        restrained_classes,
        protein_atom_ids,
        rna_atom_ids,
    ) = _build_backbone_restrain_list(case_dir)
    if not protein_atom_ids:
        raise ValueError("No protein atoms found in the system.")
    if not rna_atom_ids:
        raise ValueError("No RNA atoms found in the system.")

    _write_minimization_mdin(case_dir, restrain_file=restrain_file)
    run_sponge_amber(case_dir, timeout=1200, mpi_np=mpi_np)
    shutil.copyfile(case_dir / "mdout.txt", case_dir / "mdout_min.txt")
    shutil.copyfile(case_dir / "mdbox.txt", case_dir / "mdbox_min.txt")
    shutil.copyfile(case_dir / "restart.rst7", case_dir / "restart_min.rst7")

    _write_restrained_npt_mdin(case_dir, restrain_file=restrain_file)
    run_sponge_amber(case_dir, timeout=1200, mpi_np=mpi_np)
    shutil.copyfile(case_dir / "mdout.txt", case_dir / "mdout_restrained.txt")
    shutil.copyfile(case_dir / "mdbox.txt", case_dir / "mdbox_restrained.txt")
    shutil.copyfile(
        case_dir / "restart.rst7", case_dir / "restart_restrained.rst7"
    )

    _write_unrestrained_npt_mdin(case_dir)
    run_sponge_amber(case_dir, timeout=1200, mpi_np=mpi_np)
    shutil.copyfile(case_dir / "mdout.txt", case_dir / "mdout_unrestrained.txt")
    shutil.copyfile(case_dir / "mdbox.txt", case_dir / "mdbox_unrestrained.txt")
    shutil.copyfile(
        case_dir / "restart.rst7", case_dir / "restart_equilibrated.rst7"
    )

    return {
        "case_dir": case_dir,
        "restrain_file": restrain_file,
        "restrained_ids": restrained_ids,
        "restrained_classes": restrained_classes,
        "protein_atom_ids": protein_atom_ids,
        "rna_atom_ids": rna_atom_ids,
        "water_oxygen_ids": _build_water_oxygen_indices(
            case_dir / "model-protein-RNA-complex.prmtop"
        ),
    }


def test_restrain_npt_after_minimization(amber_equilibrated_case):
    case_dir = amber_equilibrated_case["case_dir"]
    restrained_ids = amber_equilibrated_case["restrained_ids"]
    restrained_classes = amber_equilibrated_case["restrained_classes"]
    protein_atom_ids = amber_equilibrated_case["protein_atom_ids"]
    rna_atom_ids = amber_equilibrated_case["rna_atom_ids"]

    start_coords = _read_rst7_coords(
        case_dir / "model-protein-RNA-complex.inpcrd"
    )
    min_coords = _read_rst7_coords(case_dir / "restart_min.rst7")
    restrained_coords = _read_rst7_coords(case_dir / "restart_restrained.rst7")
    unrestrained_coords = _read_rst7_coords(case_dir / "restart.rst7")

    min_protein_rmsd = _kabsch_rmsd(
        start_coords[protein_atom_ids], min_coords[protein_atom_ids]
    )
    restrained_protein_rmsd = _kabsch_rmsd(
        start_coords[protein_atom_ids], restrained_coords[protein_atom_ids]
    )
    unrestrained_protein_rmsd = _kabsch_rmsd(
        start_coords[protein_atom_ids], unrestrained_coords[protein_atom_ids]
    )
    min_rna_rmsd = _kabsch_rmsd(
        start_coords[rna_atom_ids], min_coords[rna_atom_ids]
    )
    restrained_rna_rmsd = _kabsch_rmsd(
        start_coords[rna_atom_ids], restrained_coords[rna_atom_ids]
    )
    unrestrained_rna_rmsd = _kabsch_rmsd(
        start_coords[rna_atom_ids], unrestrained_coords[rna_atom_ids]
    )

    restrained_mdout = case_dir / "mdout_restrained.txt"
    unrestrained_mdout = case_dir / "mdout_unrestrained.txt"

    restrained_density = parse_mdout_column(restrained_mdout, "density")
    restrained_temperature = parse_mdout_column(restrained_mdout, "temperature")
    restrained_pressure = parse_mdout_column(restrained_mdout, "pressure")

    unrestrained_density = parse_mdout_column(unrestrained_mdout, "density")
    unrestrained_temperature = parse_mdout_column(
        unrestrained_mdout, "temperature"
    )
    unrestrained_pressure = parse_mdout_column(unrestrained_mdout, "pressure")

    restrained_nonfinite = any(
        x is not None
        for x in (
            _first_nonfinite_index(restrained_density),
            _first_nonfinite_index(restrained_temperature),
            _first_nonfinite_index(restrained_pressure),
        )
    )
    unrestrained_nonfinite = any(
        x is not None
        for x in (
            _first_nonfinite_index(unrestrained_density),
            _first_nonfinite_index(unrestrained_temperature),
            _first_nonfinite_index(unrestrained_pressure),
        )
    )

    tail_n = TAIL_SAMPLES
    restrained_tail = restrained_density[-tail_n:]
    unrestrained_tail = unrestrained_density[-tail_n:]
    restrained_tail_mean = statistics.fmean(restrained_tail)
    restrained_tail_std = (
        statistics.stdev(restrained_tail) if len(restrained_tail) > 1 else 0.0
    )
    unrestrained_tail_mean = statistics.fmean(unrestrained_tail)
    unrestrained_tail_std = (
        statistics.stdev(unrestrained_tail)
        if len(unrestrained_tail) > 1
        else 0.0
    )

    final_density = unrestrained_density[-1]
    final_temperature = unrestrained_temperature[-1]
    final_pressure = unrestrained_pressure[-1]
    density_target = 1.0
    density_abs_tol = 0.3
    restrained_density_ok = (
        abs(restrained_tail_mean - density_target) <= density_abs_tol
    )
    unrestrained_density_ok = (
        abs(unrestrained_tail_mean - density_target) <= density_abs_tol
    )
    rmsd_limit = 3.0
    rmsd_ok = all(
        v <= rmsd_limit
        for v in (
            min_protein_rmsd,
            min_rna_rmsd,
            restrained_protein_rmsd,
            restrained_rna_rmsd,
            unrestrained_protein_rmsd,
            unrestrained_rna_rmsd,
        )
    )
    status = (
        "PASS"
        if (
            (not restrained_nonfinite)
            and (not unrestrained_nonfinite)
            and restrained_density_ok
            and unrestrained_density_ok
            and rmsd_ok
        )
        else "FAIL"
    )
    rows = [
        ["Case", "restrain_npt"],
        [
            "Protocol",
            f"minimization({MINIMIZATION_STEP_LIMIT}) -> "
            f"restrained NPT({NPT_STEP_LIMIT}) -> "
            f"unrestrained NPT({NPT_STEP_LIMIT})",
        ],
        ["Barostat", "andersen_barostat"],
        ["ConstrainMode", "SHAKE"],
        ["RestrainedWeight", "5.0"],
        ["RestrainedAtomCount", str(len(restrained_ids))],
        [
            "RestrainedClassCount",
            (
                "protein_backbone="
                f"{restrained_classes.get('protein_backbone', 0)}, "
                "rna_backbone="
                f"{restrained_classes.get('rna_backbone', 0)}"
            ),
        ],
        ["RMSDReference", "model-protein-RNA-complex.inpcrd"],
        ["MinProteinRMSD(A)", f"{min_protein_rmsd:.4f}"],
        ["MinRNARMSD(A)", f"{min_rna_rmsd:.4f}"],
        ["RestrainedProteinRMSD(A)", f"{restrained_protein_rmsd:.4f}"],
        ["RestrainedRNARMSD(A)", f"{restrained_rna_rmsd:.4f}"],
        ["UnrestrainedProteinRMSD(A)", f"{unrestrained_protein_rmsd:.4f}"],
        ["UnrestrainedRNARMSD(A)", f"{unrestrained_rna_rmsd:.4f}"],
        ["TailSamples", str(tail_n)],
        ["RestrainedDensityTailMean(g/cm3)", f"{restrained_tail_mean:.4f}"],
        ["RestrainedDensityTailStd", f"{restrained_tail_std:.4f}"],
        ["UnrestrainedDensityTailMean(g/cm3)", f"{unrestrained_tail_mean:.4f}"],
        ["UnrestrainedDensityTailStd", f"{unrestrained_tail_std:.4f}"],
        ["DensityTarget(g/cm3)", f"{density_target:.1f}"],
        ["DensityAbsTol", f"{density_abs_tol:.1f}"],
        [
            "RestrainedDensityWithinTol",
            "YES" if restrained_density_ok else "NO",
        ],
        [
            "UnrestrainedDensityWithinTol",
            "YES" if unrestrained_density_ok else "NO",
        ],
        ["RMSDMaxAllowed(A)", f"{rmsd_limit:.1f}"],
        ["RMSDWithinLimit", "YES" if rmsd_ok else "NO"],
        ["FinalDensity(g/cm3)", f"{final_density:.4f}"],
        ["FinalTemperature(K)", f"{final_temperature:.2f}"],
        ["FinalPressure(bar)", f"{final_pressure:.2f}"],
        ["RestrainedHasNonFinite", "YES" if restrained_nonfinite else "NO"],
        ["UnrestrainedHasNonFinite", "YES" if unrestrained_nonfinite else "NO"],
        ["Status", status],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Amber Performance: Restrain NPT Stability After Minimization",
    )

    assert all(
        math.isfinite(v)
        for v in (
            min_protein_rmsd,
            min_rna_rmsd,
            restrained_protein_rmsd,
            restrained_rna_rmsd,
            unrestrained_protein_rmsd,
            unrestrained_rna_rmsd,
        )
    )
    assert not restrained_nonfinite
    assert not unrestrained_nonfinite


def test_restrain_npt_long_run_oo_rdf(
    statics_path,
    outputs_path,
    amber_equilibrated_case,
    amber_mode,
    amber_steps,
    mpi_np,
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="restrain_npt",
        mpi_np=mpi_np,
        run_name=f"{amber_mode.lower()}_long_run_rdf",
    )
    equil_case_dir = amber_equilibrated_case["case_dir"]
    shutil.copyfile(
        equil_case_dir / "restart_equilibrated.rst7",
        case_dir / "restart_equilibrated.rst7",
    )

    trajectory_interval = max(100, amber_steps // 50)
    write_amber_long_run_mdin(
        case_dir,
        mode=amber_mode,
        step_limit=amber_steps,
        trajectory_interval=trajectory_interval,
        amber_rst7="restart_equilibrated.rst7",
    )
    timeout = max(2400, amber_steps // 20)
    run_sponge_amber(case_dir, timeout=timeout, mpi_np=mpi_np)

    atom_count = _read_rst7_coords(
        case_dir / "restart_equilibrated.rst7"
    ).shape[0]
    trajectory = read_coordinate_trajectory(case_dir / "mdcrd.dat", atom_count)
    box_trajectory = read_box_trajectory(case_dir / "mdbox.txt")
    radii, rdf, sampled_oxygen_ids = compute_oo_rdf(
        trajectory,
        box_trajectory,
        amber_equilibrated_case["water_oxygen_ids"],
        bin_width=0.05,
    )
    rdf_summary = summarize_rdf(radii, rdf)
    rdf_plot = save_rdf_plot(
        radii,
        rdf,
        case_dir / "amber_oo_rdf.png",
        title=f"Amber O-O RDF ({amber_mode}, {amber_steps} steps)",
    )

    frame_count = min(trajectory.shape[0], box_trajectory.shape[0])
    rows = [
        ["RunName", f"{amber_mode.lower()}_long_run_rdf"],
        ["Mode", amber_mode],
        ["StepLimit", str(amber_steps)],
        ["TrajectoryInterval", str(trajectory_interval)],
        ["FramesUsed", str(frame_count)],
        [
            "WaterOxygenCount",
            str(len(amber_equilibrated_case["water_oxygen_ids"])),
        ],
        ["WaterOxygenSampled", str(len(sampled_oxygen_ids))],
        ["RDFPeakR(A)", f"{rdf_summary['rdf_peak_r']:.3f}"],
        ["RDFPeakG", f"{rdf_summary['rdf_peak_g']:.3f}"],
        ["RDFMax", f"{rdf_summary['rdf_max']:.3f}"],
        ["RDFTailMean", f"{rdf_summary['rdf_tail_mean']:.3f}"],
        ["RDFPlot", rdf_plot.name if rdf_plot is not None else "not-generated"],
        ["Result", "PLOTTED"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title=f"Amber Performance: Long {amber_mode} O-O RDF",
    )

    assert frame_count >= 2
    assert np.all(np.isfinite(rdf))
