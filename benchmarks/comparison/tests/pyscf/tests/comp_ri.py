import pytest

from benchmarks.comparison.tests.pyscf.tests.utils import (
    HARTREE_TO_KCAL_MOL,
    run_sponge_vs_pyscf,
)
from benchmarks.utils import Outputer

# RI introduces ~1e-4 Ha approximation error on top of SCF tolerance
RI_RHF_TOL_HA = 2.0e-3
RI_RKS_TOL_HA = 5.0e-3
RI_UHF_TOL_HA = 1.0e-2
RI_UKS_TOL_HA = 1.0e-2

DF_ARGS = ["-qc_density_fit", "1"]

# ---------- RI-RHF ----------

RI_RHF_CASES = [
    ("h2", "def2-svp"),
    ("he", "def2-tzvp"),
    ("oh2", "def2-svp"),
    ("ch4", "def2-svp"),
    ("co2", "def2-svp"),
    ("benzene", "def2-svp"),
]


@pytest.mark.parametrize(
    "case_name,basis_name",
    RI_RHF_CASES,
    ids=[f"{case}_{basis}" for case, basis in RI_RHF_CASES],
)
def test_ri_rhf(case_name, basis_name, statics_path, outputs_path, mpi_np):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name="HF",
        basis_name=basis_name,
        restricted=True,
        run_prefix="ri_rhf",
        mpi_np=mpi_np,
        extra_sponge_args=DF_ARGS,
    )

    tol_kcal = RI_RHF_TOL_HA * HARTREE_TO_KCAL_MOL
    headers = [
        "Case",
        "Method/Basis",
        "PySCF (kcal/mol)",
        "SPONGE RI (kcal/mol)",
        "|Delta| (kcal/mol)",
        "Tol (kcal/mol)",
        "Status",
    ]
    rows = [
        [
            case_name,
            f"RI-HF/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= RI_RHF_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="RI-RHF vs PySCF")

    assert result["abs_diff_ha"] <= RI_RHF_TOL_HA


# ---------- RI-RKS ----------

RI_RKS_CASES = [
    ("oh2", "def2-svp", "PBE"),
    ("oh2", "def2-svp", "B3LYP"),
    ("ch4", "def2-svp", "PBE0"),
]


@pytest.mark.parametrize(
    "case_name,basis_name,method_name",
    RI_RKS_CASES,
    ids=[f"{case}_{method}_{basis}" for case, basis, method in RI_RKS_CASES],
)
def test_ri_rks(
    case_name,
    basis_name,
    method_name,
    statics_path,
    outputs_path,
    mpi_np,
):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=True,
        run_prefix="ri_rks",
        mpi_np=mpi_np,
        extra_sponge_args=DF_ARGS,
    )

    tol_kcal = RI_RKS_TOL_HA * HARTREE_TO_KCAL_MOL
    headers = [
        "Case",
        "Method/Basis",
        "PySCF (kcal/mol)",
        "SPONGE RI (kcal/mol)",
        "|Delta| (kcal/mol)",
        "Tol (kcal/mol)",
        "Status",
    ]
    rows = [
        [
            case_name,
            f"RI-{method_name}/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= RI_RKS_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="RI-RKS vs PySCF")

    assert result["abs_diff_ha"] <= RI_RKS_TOL_HA


# ---------- RI-UHF ----------

RI_UHF_CASES = [
    ("no_doublet", "def2-svp"),
    ("o_triplet", "def2-svp"),
]


@pytest.mark.parametrize(
    "case_name,basis_name",
    RI_UHF_CASES,
    ids=[f"{case}_{basis}" for case, basis in RI_UHF_CASES],
)
def test_ri_uhf(case_name, basis_name, statics_path, outputs_path, mpi_np):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name="HF",
        basis_name=basis_name,
        restricted=False,
        run_prefix="ri_uhf",
        mpi_np=mpi_np,
        extra_sponge_args=DF_ARGS,
    )

    tol_kcal = RI_UHF_TOL_HA * HARTREE_TO_KCAL_MOL
    headers = [
        "Case",
        "Method/Basis",
        "PySCF (kcal/mol)",
        "SPONGE RI (kcal/mol)",
        "|Delta| (kcal/mol)",
        "Tol (kcal/mol)",
        "Status",
    ]
    rows = [
        [
            case_name,
            f"RI-UHF/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= RI_UHF_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="RI-UHF vs PySCF")

    assert result["abs_diff_ha"] <= RI_UHF_TOL_HA


# ---------- RI-UKS ----------

RI_UKS_CASES = [
    ("no_doublet", "def2-svp", "B3LYP"),
]


@pytest.mark.parametrize(
    "case_name,basis_name,method_name",
    RI_UKS_CASES,
    ids=[f"{case}_{method}_{basis}" for case, basis, method in RI_UKS_CASES],
)
def test_ri_uks(
    case_name,
    basis_name,
    method_name,
    statics_path,
    outputs_path,
    mpi_np,
):
    result = run_sponge_vs_pyscf(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=False,
        run_prefix="ri_uks",
        mpi_np=mpi_np,
        extra_sponge_args=DF_ARGS,
    )

    tol_kcal = RI_UKS_TOL_HA * HARTREE_TO_KCAL_MOL
    headers = [
        "Case",
        "Method/Basis",
        "PySCF (kcal/mol)",
        "SPONGE RI (kcal/mol)",
        "|Delta| (kcal/mol)",
        "Tol (kcal/mol)",
        "Status",
    ]
    rows = [
        [
            case_name,
            f"RI-U{method_name}/{basis_name}",
            f"{result['pyscf_energy_kcal_mol']:.6f}",
            f"{result['sponge_energy_kcal_mol']:.6f}",
            f"{result['abs_diff_kcal_mol']:.6f}",
            f"{tol_kcal:.6f}",
            "PASS" if result["abs_diff_ha"] <= RI_UKS_TOL_HA else "FAIL",
        ]
    ]
    Outputer.print_table(headers, rows, title="RI-UKS vs PySCF")

    assert result["abs_diff_ha"] <= RI_UKS_TOL_HA
