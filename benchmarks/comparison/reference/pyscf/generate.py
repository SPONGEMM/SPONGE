#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

REFERENCE_CASES = [
    # RHF
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
    ("ace_ala4_nme", "HF", "def2-svp", True),
    ("benzene", "HF", "def2-qzvp", True),
    ("h2", "HF", "cc-pvdz", True),
    ("he", "HF", "cc-pvtz", True),
    # UHF
    ("no_doublet", "HF", "sto-3g", False),
    ("no_doublet", "HF", "3-21g", False),
    ("o_triplet", "HF", "6-31g", False),
    ("o_triplet", "HF", "cc-pvdz", False),
    # RKS
    ("h2", "LDA", "6-31g", True),
    ("he", "PBE", "6-31g", True),
    ("oh2", "BLYP", "6-31g", True),
    ("ch4", "PBE0", "6-31g", True),
    ("co2", "B3LYP", "6-31g", True),
    # UKS
    ("o_triplet", "LDA", "6-31g", False),
    ("o_triplet", "PBE", "6-31g", False),
    ("o_triplet", "BLYP", "6-31g", False),
    ("o_triplet", "PBE0", "6-31g", False),
    ("o_triplet", "B3LYP", "6-31g", False),
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def build_reference_entries(statics_path: Path):
    repo_root = get_repo_root()
    tests_dir = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "tests"
        / "pyscf"
        / "tests"
    )
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(tests_dir))

    from utils import load_case_definition, run_pyscf_energy_ha

    entries = []
    for case_name, method_name, basis_name, restricted in REFERENCE_CASES:
        case = load_case_definition(statics_path, case_name)
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
        key = (
            entry["case_name"],
            entry["method_name"],
            entry["basis_name"],
            bool(entry["restricted"]),
        )
        result[key] = float(entry["energy_ha"])
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
        diff = abs(current_map[key] - generated_map[key])
        if diff > max_diff:
            max_diff = diff
            max_key = key
    if max_diff > abs_tol:
        return (
            False,
            "Energy differs above tolerance: "
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
