#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

REFERENCE_CASES = [
    # RHF (comp_rhf.py)
    ("h2", "HF", "sto-3g", True),
    ("he", "HF", "3-21g", True),
    ("h2", "HF", "6-31g", True),
    ("he", "HF", "6-31g*", True),
    ("h2", "HF", "6-31g**", True),
    ("he", "HF", "6-311g", True),
    ("h2", "HF", "6-311g*", True),
    ("he", "HF", "6-311g**", True),
    ("h2", "HF", "def2-svp", True),
    ("he", "HF", "def2-tzvp", True),
    ("h2", "HF", "def2-tzvpp", True),
    ("he", "HF", "def2-qzvp", True),
    ("h2", "HF", "cc-pvdz", True),
    ("he", "HF", "cc-pvtz", True),
    # RHF 大体系 (perf_rhf.py)
    ("ace_ala4_nme", "HF", "def2-svp", True),
    ("benzene", "HF", "def2-qzvp", True),
    # UHF (comp_uhf.py)
    ("no_doublet", "HF", "sto-3g", False),
    ("no_doublet", "HF", "3-21g", False),
    ("o_triplet", "HF", "6-31g", False),
    ("o_triplet", "HF", "cc-pvdz", False),
    # RKS (comp_rks.py)
    ("h2", "LDA", "6-31g", True),
    ("he", "PBE", "6-31g", True),
    ("oh2", "BLYP", "6-31g", True),
    ("ch4", "PBE0", "6-31g", True),
    ("co2", "B3LYP", "6-31g", True),
    # UKS (comp_uks.py)
    ("o_triplet", "LDA", "6-31g", False),
    ("o_triplet", "PBE", "6-31g", False),
    ("o_triplet", "BLYP", "6-31g", False),
    ("o_triplet", "PBE0", "6-31g", False),
    ("o_triplet", "B3LYP", "6-31g", False),
    # 第四周期 (comp_4th_period.py)
    ("br_anion", "HF", "ma-def2-svp", True),
    ("fe_quintet", "HF", "6-31++g", False),
]

# 梯度参考案例: (case_name, method_name, basis_name, restricted, coords_angstrom)
GRADIENT_CASES = [
    ("h2", "HF", "sto-3g", True, [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]),
    ("h2", "HF", "6-31g", True, [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]),
]

# H2 平衡键长参考案例: (case_name, basis_name)
MINIMIZE_CASES = [
    ("h2_min_sto3g", "HF", "sto-3g"),
]

# 需要做 UHF/UKS 稳定性分析的案例（过渡金属等容易收敛到鞍点）
STABILITY_CASES = {"fe_quintet"}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_with_stability(mf, max_cycles=10):
    """对 UHF/UKS 做稳定性分析，确保收敛到真正的极小值."""
    mf.kernel()
    for _ in range(max_cycles):
        mo_new = mf.stability()[0]
        if mo_new is mf.mo_coeff:
            break
        mf.mo_coeff = mo_new
        mf.kernel()
    return float(mf.e_tot)


def build_reference_entries(statics_path: Path):
    repo_root = get_repo_root()
    tests_dir = (
        repo_root / "benchmarks" / "comparison" / "tests" / "pyscf" / "tests"
    )
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(tests_dir))

    from utils import (
        _build_pyscf_method,
        load_case_definition,
        run_pyscf_energy_ha,
    )

    entries = []
    for case_name, method_name, basis_name, restricted in REFERENCE_CASES:
        case = load_case_definition(statics_path, case_name)

        if case_name in STABILITY_CASES and not restricted:
            _mol, mf = _build_pyscf_method(
                atoms=case["atoms"],
                basis_name=basis_name,
                charge=case["charge"],
                multiplicity=case["multiplicity"],
                method_name=method_name,
                restricted=restricted,
            )
            energy_ha = _run_with_stability(mf)
        else:
            energy_ha = run_pyscf_energy_ha(
                atoms=case["atoms"],
                basis_name=basis_name,
                charge=case["charge"],
                multiplicity=case["multiplicity"],
                method_name=method_name,
                restricted=restricted,
            )

        entries.append(
            {
                "case_name": case_name,
                "method_name": method_name,
                "basis_name": basis_name,
                "restricted": restricted,
                "energy_ha": energy_ha,
            }
        )

    # 梯度参考数据
    from pyscf import gto, scf, grad as pyscf_grad

    for (
        case_name,
        method_name,
        basis_name,
        restricted,
        coords,
    ) in GRADIENT_CASES:
        atom_str = "; ".join(f"H {c[0]} {c[1]} {c[2]}" for c in coords)
        mol = gto.M(atom=atom_str, basis=basis_name, unit="Angstrom", verbose=0)
        mf = scf.RHF(mol) if restricted else scf.UHF(mol)
        mf.kernel()
        g = mf.nuc_grad_method().kernel()  # (natm, 3), Ha/Bohr

        entries.append(
            {
                "case_name": case_name,
                "method_name": method_name,
                "basis_name": basis_name,
                "restricted": restricted,
                "type": "gradient",
                "gradient_ha_bohr": g.tolist(),
                "coords_angstrom": coords,
            }
        )

    # H2 平衡键长参考数据
    import numpy as _np

    for case_name, method_name, basis_name in MINIMIZE_CASES:
        best_r, best_e = None, 1e10
        for r in _np.arange(0.5, 1.2, 0.001):
            mol = gto.M(
                atom=f"H 0 0 {-r / 2}; H 0 0 {r / 2}",
                basis=basis_name,
                unit="Angstrom",
                verbose=0,
            )
            mf = scf.RHF(mol)
            mf.kernel()
            if mf.e_tot < best_e:
                best_e = float(mf.e_tot)
                best_r = float(r)

        entries.append(
            {
                "case_name": case_name,
                "method_name": method_name,
                "basis_name": basis_name,
                "restricted": True,
                "type": "minimize",
                "equilibrium_bond_length_angstrom": best_r,
                "equilibrium_energy_ha": best_e,
            }
        )

    entries.sort(
        key=lambda v: (
            v["case_name"],
            v["method_name"],
            v["basis_name"],
            int(v["restricted"]),
        )
    )
    return entries


def build_payload(statics_path: Path):
    try:
        from pyscf import __version__ as pyscf_version
    except Exception as exc:
        raise RuntimeError(
            "PySCF import failed. Please run in an environment with pyscf installed."
        ) from exc

    entries = build_reference_entries(statics_path)
    return {
        "format_version": 1,
        "unit": "Hartree",
        "pyscf_version": pyscf_version,
        "entries": entries,
    }


def _entries_to_map(payload):
    result = {}
    for entry in payload["entries"]:
        entry_type = entry.get("type", "energy")
        key = (
            entry["case_name"],
            entry["method_name"],
            entry["basis_name"],
            bool(entry["restricted"]),
            entry_type,
        )
        if entry_type == "energy":
            result[key] = float(entry["energy_ha"])
        elif entry_type == "gradient":
            result[key] = entry["gradient_ha_bohr"]
        elif entry_type == "minimize":
            result[key] = (
                entry["equilibrium_bond_length_angstrom"],
                entry["equilibrium_energy_ha"],
            )
    return result


def compare_payloads(current, generated, abs_tol: float):
    for key in ["format_version", "unit", "pyscf_version"]:
        if current.get(key) != generated.get(key):
            return (
                False,
                f"Metadata differs at '{key}': "
                f"current={current.get(key)!r}, generated={generated.get(key)!r}",
            )

    current_map = _entries_to_map(current)
    generated_map = _entries_to_map(generated)

    if set(current_map) != set(generated_map):
        current_only = sorted(set(current_map) - set(generated_map))
        generated_only = sorted(set(generated_map) - set(current_map))
        return (
            False,
            "Entry keys differ. "
            f"current-only={len(current_only)}, generated-only={len(generated_only)}",
        )

    max_diff = 0.0
    max_key = None
    for key in current_map:
        entry_type = key[-1]
        if entry_type == "energy":
            diff = abs(current_map[key] - generated_map[key])
        elif entry_type == "minimize":
            diff = abs(current_map[key][1] - generated_map[key][1])
        elif entry_type == "gradient":
            import numpy as _np

            diff = float(
                _np.max(
                    _np.abs(
                        _np.array(current_map[key])
                        - _np.array(generated_map[key])
                    )
                )
            )
        else:
            continue
        if diff > max_diff:
            max_diff = diff
            max_key = key
    if max_diff > abs_tol:
        return (
            False,
            "Reference data differs above tolerance: "
            f"max_diff={max_diff:.3e} at {max_key}",
        )
    return True, f"max_diff={max_diff:.3e}"


def main():
    repo_root = get_repo_root()
    default_output = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "reference"
        / "pyscf"
        / "reference.json"
    )
    default_statics = (
        repo_root / "benchmarks" / "comparison" / "tests" / "pyscf" / "statics"
    )

    parser = argparse.ArgumentParser(
        description="Generate static PySCF reference energies for comp-pyscf tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output json path.",
    )
    parser.add_argument(
        "--statics-path",
        type=Path,
        default=default_statics,
        help="Path to comparison statics directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated payload with the existing output file.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance (Hartree) used by --check.",
    )
    args = parser.parse_args()

    payload = build_payload(args.statics_path)

    if args.check:
        if not args.output.exists():
            print(f"[FAIL] Missing reference file: {args.output}")
            raise SystemExit(1)
        with open(args.output, "r") as f:
            current = json.load(f)
        same, detail = compare_payloads(current, payload, abs_tol=args.abs_tol)
        if same:
            print(f"[PASS] Reference file is up to date ({detail}).")
            return
        print(
            f"[FAIL] Reference file differs from freshly generated data. {detail}"
        )
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[OK] Wrote {len(payload['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
