import pytest

from benchmarks.utils import Outputer, Runner

from benchmarks.comparison.tests.amber.tests.utils import (
    copy_amber_reference_system_files,
    extract_sponge_forces_frc_dat,
    extract_sponge_potential,
    force_stats,
    force_stats_with_rigid_water_entities,
    load_amber_reference_energy,
    load_amber_reference_forces,
    perturb_rst7_inplace,
    perturb_rst7_with_rigid_water_inplace,
    write_gb_in_file_from_parm7,
    write_tip4p_virtual_atom_from_parm7,
)

TIP4P_CASES = [
    (0, 0.00),
    (1, 0.02),
    (2, 0.05),
]


@pytest.mark.parametrize(
    "iteration,perturbation",
    TIP4P_CASES,
    ids=[f"iter{it}_pert{pert:.2f}" for it, pert in TIP4P_CASES],
)
def test_amber_alanine_dipeptide_tip4pew_run0(
    iteration,
    perturbation,
    statics_path,
    outputs_path,
    mpi_np,
):
    case_name = "alanine_dipeptide_tip4pew"
    case_dir = Outputer.prepare_output_case(
        statics_path,
        outputs_path,
        case_name,
        mpi_np=mpi_np,
        run_name=iteration,
    )

    copy_amber_reference_system_files(statics_path, case_dir, case_name)
    write_tip4p_virtual_atom_from_parm7(
        case_dir / "system.parm7",
        case_dir / "tip4p_virtual_atom.txt",
    )
    (case_dir / "system.rst7").write_text(
        (case_dir / "system_minimized.rst7").read_text()
    )
    perturb_rst7_with_rigid_water_inplace(
        case_dir / "system.rst7",
        case_dir / "system.parm7",
        perturbation=perturbation,
        seed=20260217 + 2000 * iteration,
    )
    Runner.run_sponge(case_dir, mpi_np=mpi_np, mdin_name="sponge.mdin")

    amber_epot = load_amber_reference_energy(statics_path, case_name, iteration)
    sponge_potential = extract_sponge_potential(case_dir)
    energy_abs_diff = abs(amber_epot - sponge_potential)

    amber_forces = load_amber_reference_forces(
        statics_path, case_name, iteration
    )
    sponge_forces = extract_sponge_forces_frc_dat(
        case_dir / "frc.dat", natom=amber_forces.shape[0]
    )
    stats = force_stats_with_rigid_water_entities(
        case_dir / "system.parm7",
        amber_forces,
        sponge_forces,
    )

    headers = [
        "Case",
        "Iteration",
        "Perturb(A)",
        "|dE|",
        "Max |dF|",
        "RMS dF",
        "Cos(F)",
        "Status",
    ]
    energy_tol = 0.50
    force_max_tol = 0.10
    force_rms_tol = 0.03
    force_cos_tol = 0.999
    passed = (
        energy_abs_diff <= energy_tol
        and stats["max_abs_diff"] <= force_max_tol
        and stats["rms_diff"] <= force_rms_tol
        and stats["cosine_similarity"] >= force_cos_tol
    )
    rows = [
        [
            case_name,
            str(iteration),
            f"{perturbation:.2f}",
            f"{energy_abs_diff:.6f}",
            f"{stats['max_abs_diff']:.6f}",
            f"{stats['rms_diff']:.6f}",
            f"{stats['cosine_similarity']:.6f}",
            "PASS" if passed else "FAIL",
        ]
    ]
    Outputer.print_table(
        headers, rows, title="AMBER TIP4P-Ew Perturbed Validation"
    )

    assert energy_abs_diff <= energy_tol
    assert stats["max_abs_diff"] <= force_max_tol
    assert stats["rms_diff"] <= force_rms_tol
    assert stats["cosine_similarity"] >= force_cos_tol


GB_FORCE_CASES = [
    (0, 0.00),
    (1, 0.05),
    (2, 0.10),
]


@pytest.mark.parametrize(
    "iteration,perturbation",
    GB_FORCE_CASES,
    ids=[f"iter{it}_pert{pert:.2f}" for it, pert in GB_FORCE_CASES],
)
def test_amber_alanine_dipeptide_gb1_perturbed_force(
    iteration,
    perturbation,
    statics_path,
    outputs_path,
    mpi_np,
):
    if mpi_np is not None:
        pytest.skip(
            "Skip GB1 AMBER case under MPI due to known NOPBC conflict."
        )

    case_name = "alanine_dipeptide_gb1"
    case_dir = Outputer.prepare_output_case(
        statics_path,
        outputs_path,
        case_name,
        mpi_np=mpi_np,
        run_name=iteration,
    )

    copy_amber_reference_system_files(statics_path, case_dir, case_name)
    write_gb_in_file_from_parm7(
        case_dir / "system.parm7", case_dir / "gb_gb.txt"
    )

    _coords = perturb_rst7_inplace(
        case_dir / "system.rst7",
        perturbation=perturbation,
        seed=20260217 + 1000 * iteration,
    )

    Runner.run_sponge(case_dir, mpi_np=mpi_np, mdin_name="sponge.mdin")

    amber_epot = load_amber_reference_energy(statics_path, case_name, iteration)
    sponge_potential = extract_sponge_potential(case_dir)
    energy_abs_diff = abs(amber_epot - sponge_potential)

    amber_forces = load_amber_reference_forces(
        statics_path, case_name, iteration
    )
    sponge_forces = extract_sponge_forces_frc_dat(
        case_dir / "frc.dat", natom=amber_forces.shape[0]
    )
    stats = force_stats(amber_forces, sponge_forces)

    headers = [
        "Case",
        "Iteration",
        "Perturb(A)",
        "|dE|",
        "Max |dF|",
        "RMS dF",
        "Cos(F)",
        "Status",
    ]

    energy_tol = 0.02
    force_max_tol = 0.06
    force_rms_tol = 0.02
    force_cos_tol = 0.999
    passed = (
        energy_abs_diff <= energy_tol
        and stats["max_abs_diff"] <= force_max_tol
        and stats["rms_diff"] <= force_rms_tol
        and stats["cosine_similarity"] >= force_cos_tol
    )
    rows = [
        [
            case_name,
            str(iteration),
            f"{perturbation:.2f}",
            f"{energy_abs_diff:.6f}",
            f"{stats['max_abs_diff']:.6f}",
            f"{stats['rms_diff']:.6f}",
            f"{stats['cosine_similarity']:.6f}",
            "PASS" if passed else "FAIL",
        ]
    ]
    Outputer.print_table(
        headers, rows, title="AMBER GB1 Perturbed Force Validation"
    )

    assert energy_abs_diff <= energy_tol
    assert stats["max_abs_diff"] <= force_max_tol
    assert stats["rms_diff"] <= force_rms_tol
    assert stats["cosine_similarity"] >= force_cos_tol
