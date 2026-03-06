import math

import pytest

from benchmarks.utils import Outputer, Runner

from benchmarks.comparison.tests.gromacs.tests.utils import (
    copy_gromacs_reference_sponge_inputs,
    extract_sponge_forces,
    extract_sponge_terms,
    force_stats,
    load_gromacs_reference_entry,
    load_gromacs_reference_forces,
    load_gromacs_reference_terms,
)

PERTURB_CASES = [
    (0, 0.00),
    (1, 0.02),
    (2, 0.05),
]


@pytest.mark.parametrize(
    "iteration,perturbation",
    PERTURB_CASES,
    ids=[f"iter{it}_pert{pert:.2f}" for it, pert in PERTURB_CASES],
)
def test_gromacs_alanine_dipeptide_charmm_tip3p_flexible_run0(
    iteration,
    perturbation,
    statics_path,
    outputs_path,
    mpi_np,
):
    case_name = "alanine_dipeptide_charmm_tip3p"
    case_dir = Outputer.prepare_output_case(
        statics_path,
        outputs_path,
        case_name,
        mpi_np=mpi_np,
        run_name=iteration,
    )

    ref_entry = load_gromacs_reference_entry(statics_path, case_name, iteration)
    assert abs(float(ref_entry["perturbation"]) - perturbation) <= 1.0e-12
    copy_gromacs_reference_sponge_inputs(
        statics_path,
        case_dir,
        case_name,
        iteration,
    )

    gmx_terms = load_gromacs_reference_terms(statics_path, case_name, iteration)
    gmx_forces = load_gromacs_reference_forces(
        statics_path, case_name, iteration
    )

    Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        mdin_name="sponge.mdin",
    )

    sponge_terms = extract_sponge_terms(case_dir)
    sponge_forces = extract_sponge_forces(case_dir, natom=gmx_forces.shape[0])

    energy_abs_diff = abs(gmx_terms["potential"] - sponge_terms["potential"])
    bond_abs_diff = abs(gmx_terms["bond"] - sponge_terms["bond"])
    angle_ub_abs_diff = abs(
        (gmx_terms["angle"] + gmx_terms["urey_bradley"])
        - sponge_terms["urey_bradley"]
    )
    proper_abs_diff = abs(
        gmx_terms["proper_dihedral"] - sponge_terms["proper_dihedral"]
    )
    nb14_ee_abs_diff = abs(gmx_terms["coulomb14"] - sponge_terms["nb14_ee"])
    lj_short_abs_diff = abs(gmx_terms["lj_sr"] - sponge_terms["lj_short"])
    pm_abs_diff = abs(
        (gmx_terms["coulomb_sr"] + gmx_terms["coulomb_recip"])
        - sponge_terms["pm"]
    )
    pressure_abs_diff = abs(gmx_terms["pressure"] - sponge_terms["pressure"])

    stats = force_stats(gmx_forces, sponge_forces)

    headers = [
        "Case",
        "Iter",
        "Perturb(A)",
        "|dE|",
        "|dBond|",
        "|d(Angle+UB)|",
        "|dProper|",
        "|dLJ(SR)|",
        "|dPM|",
        "Max |dF|",
        "RMS dF",
        "Cos(F)",
        "P_gmx",
        "P_sponge",
        "|dP|",
        "Status",
    ]

    energy_tol = 40.0
    bond_tol = 0.5
    angle_ub_tol = 0.5
    proper_tol = 0.1
    nb14_ee_tol = 0.2
    lj_short_tol = 25.0
    pm_tol = 2.0
    force_max_tol = 0.07
    force_rms_tol = 0.01
    force_cos_tol = 0.999

    passed = (
        energy_abs_diff <= energy_tol
        and bond_abs_diff <= bond_tol
        and angle_ub_abs_diff <= angle_ub_tol
        and proper_abs_diff <= proper_tol
        and nb14_ee_abs_diff <= nb14_ee_tol
        and lj_short_abs_diff <= lj_short_tol
        and pm_abs_diff <= pm_tol
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
            f"{bond_abs_diff:.6f}",
            f"{angle_ub_abs_diff:.6f}",
            f"{proper_abs_diff:.6f}",
            f"{lj_short_abs_diff:.6f}",
            f"{pm_abs_diff:.6f}",
            f"{stats['max_abs_diff']:.6f}",
            f"{stats['rms_diff']:.6f}",
            f"{stats['cosine_similarity']:.6f}",
            f"{gmx_terms['pressure']:.6f}",
            f"{sponge_terms['pressure']:.6f}",
            f"{pressure_abs_diff:.6f}",
            "PASS" if passed else "FAIL",
        ]
    ]
    Outputer.print_table(
        headers,
        rows,
        title="GROMACS CHARMM27 TIP3P (FLEXIBLE) vs SPONGE",
    )

    assert energy_abs_diff <= energy_tol
    assert bond_abs_diff <= bond_tol
    assert angle_ub_abs_diff <= angle_ub_tol
    assert proper_abs_diff <= proper_tol
    assert nb14_ee_abs_diff <= nb14_ee_tol
    assert lj_short_abs_diff <= lj_short_tol
    assert pm_abs_diff <= pm_tol
    assert stats["max_abs_diff"] <= force_max_tol
    assert stats["rms_diff"] <= force_rms_tol
    assert stats["cosine_similarity"] >= force_cos_tol
    assert math.isfinite(gmx_terms["pressure"])
    assert math.isfinite(sponge_terms["pressure"])
