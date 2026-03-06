#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.utils import Outputer, Runner
from benchmarks.comparison.utils import (
    extract_sander_epot,
    extract_sander_forces_mdfrc,
)

TIP4P_CASE_NAME = "alanine_dipeptide_tip4pew"
GB1_CASE_NAME = "alanine_dipeptide_gb1"

TIP4P_CASES = [
    (0, 0.00),
    (1, 0.02),
    (2, 0.05),
]

GB1_CASES = [
    (0, 0.00),
    (1, 0.05),
    (2, 0.10),
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _iter_case_specs():
    for iteration, perturbation in TIP4P_CASES:
        yield {
            "case_name": TIP4P_CASE_NAME,
            "iteration": iteration,
            "perturbation": float(perturbation),
            "seed": 20260217 + 2000 * iteration,
            "kind": "tip4p",
        }
    for iteration, perturbation in GB1_CASES:
        yield {
            "case_name": GB1_CASE_NAME,
            "iteration": iteration,
            "perturbation": float(perturbation),
            "seed": 20260217 + 1000 * iteration,
            "kind": "gb1",
        }


def _extract_sander_banner(sander_out_path: Path):
    text = sander_out_path.read_text()
    match = re.search(r"Amber\s+\d+\s+SANDER\s+\d+", text)
    if match:
        return match.group(0)
    return "unknown"


def require_ambertools():
    for exe in ("tleap", "sander"):
        if shutil.which(exe) is None:
            raise RuntimeError(
                f"Required executable '{exe}' is not available in PATH."
            )


def run_tleap(case_dir):
    return Runner.run_command(["tleap", "-f", "tleap.in"], cwd=case_dir)


def run_sander_run0(case_dir):
    cmd = [
        "sander",
        "-O",
        "-i",
        "sander.in",
        "-o",
        "sander.out",
        "-p",
        "system.parm7",
        "-c",
        "system.rst7",
        "-r",
        "system_out.rst7",
        "-x",
        "system.nc",
        "-frc",
        "mdfrc",
        "-inf",
        "system.mdinfo",
    ]
    return Runner.run_command(cmd, cwd=case_dir)


def _read_parm7_flag_items(parm7_path, flag_name):
    target = f"%FLAG {flag_name}"
    items = []
    in_target_flag = False
    in_data = False

    for line in Path(parm7_path).read_text().splitlines():
        if line.startswith("%FLAG"):
            if in_target_flag:
                break
            in_target_flag = line.strip() == target
            in_data = False
            continue

        if not in_target_flag:
            continue

        if line.startswith("%FORMAT"):
            in_data = True
            continue

        if in_data:
            items.extend(line.split())

    if not items:
        raise ValueError(f"Failed to read %FLAG {flag_name} from {parm7_path}")
    return items


def _read_parm7_flag_values(parm7_path, flag_name):
    return [
        float(v.replace("D", "E"))
        for v in _read_parm7_flag_items(parm7_path, flag_name)
    ]


def _read_parm7_flag_strings(parm7_path, flag_name):
    return _read_parm7_flag_items(parm7_path, flag_name)


def write_gb_in_file_from_parm7(parm7_path, gb_out_path):
    radii = _read_parm7_flag_values(parm7_path, "RADII")
    screen = _read_parm7_flag_values(parm7_path, "SCREEN")
    if len(radii) != len(screen):
        raise ValueError(
            f"RADII/SCREEN length mismatch in {parm7_path}: "
            f"{len(radii)} vs {len(screen)}"
        )

    with open(gb_out_path, "w") as f:
        f.write(f"{len(radii)}\n")
        for r, s in zip(radii, screen):
            f.write(f"{r:.6f} {s:.6f}\n")


def write_tip4p_virtual_atom_from_parm7(
    parm7_path, virtual_atom_out_path, a=0.12797, b=0.12797
):
    atom_names = _read_parm7_flag_strings(parm7_path, "ATOM_NAME")
    atom_types = _read_parm7_flag_strings(parm7_path, "AMBER_ATOM_TYPE")
    residue_labels = _read_parm7_flag_strings(parm7_path, "RESIDUE_LABEL")
    residue_pointer = [
        int(v) for v in _read_parm7_flag_values(parm7_path, "RESIDUE_POINTER")
    ]

    if len(atom_names) != len(atom_types):
        raise ValueError(
            f"ATOM_NAME/AMBER_ATOM_TYPE length mismatch in {parm7_path}: "
            f"{len(atom_names)} vs {len(atom_types)}"
        )
    if len(residue_labels) != len(residue_pointer):
        raise ValueError(
            f"RESIDUE_LABEL/RESIDUE_POINTER length mismatch in {parm7_path}: "
            f"{len(residue_labels)} vs {len(residue_pointer)}"
        )

    residue_pointer.append(len(atom_names) + 1)
    lines = []
    for i, resname in enumerate(residue_labels):
        if resname not in {"WAT", "HOH"}:
            continue

        start = residue_pointer[i] - 1
        end = residue_pointer[i + 1] - 1  # exclusive
        if end <= start:
            continue

        local_names = atom_names[start:end]
        local_types = atom_types[start:end]

        o_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "OW" or name == "O"
        ]
        h_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "HW" or name in {"H1", "H2"} or name.startswith("H")
        ]
        ep_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "EP" or name.startswith("EP") or name == "M"
        ]

        if (
            len(o_candidates) != 1
            or len(ep_candidates) != 1
            or len(h_candidates) < 2
        ):
            raise ValueError(
                f"Unexpected TIP4P water residue layout at residue {i + 1} "
                f"({resname}) in {parm7_path}: names={local_names}, types={local_types}"
            )

        h1, h2 = sorted(h_candidates)[:2]
        o = o_candidates[0]
        ep = ep_candidates[0]
        lines.append(f"2 {ep} {o} {h1} {h2} {a:.6f} {b:.6f}")

    if not lines:
        raise ValueError(
            f"No TIP4P water residues found in {parm7_path}. "
            "Cannot generate virtual_atom_in_file."
        )

    Path(virtual_atom_out_path).write_text("\n".join(lines) + "\n")


def read_rst7_coords(rst7_path):
    lines = Path(rst7_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid rst7 (too short): {rst7_path}")

    title = lines[0]
    count_line = lines[1]
    head = count_line.split()
    if not head:
        raise ValueError(f"Invalid rst7 count line: {rst7_path}")
    natom = int(head[0])

    target = natom * 3
    coords_vals = []
    idx = 2
    while idx < len(lines) and len(coords_vals) < target:
        line = lines[idx]
        for j in range(0, len(line), 12):
            chunk = line[j : j + 12]
            if chunk.strip():
                coords_vals.append(float(chunk))
                if len(coords_vals) == target:
                    break
        idx += 1

    if len(coords_vals) != target:
        raise ValueError(
            f"Failed to parse rst7 coordinates: {rst7_path}, "
            f"expected {target}, got {len(coords_vals)}"
        )

    coords = np.array(coords_vals, dtype=np.float64).reshape(natom, 3)
    tail_lines = lines[idx:]
    return title, count_line, coords, tail_lines


def write_rst7_coords(rst7_path, title, count_line, coords, tail_lines):
    coords = np.asarray(coords, dtype=np.float64)
    natom = coords.shape[0]
    flat = coords.reshape(-1)

    out_lines = [title, count_line]
    for i in range(0, len(flat), 6):
        block = flat[i : i + 6]
        out_lines.append("".join(f"{v:12.7f}" for v in block))
    out_lines.extend(tail_lines)
    Path(rst7_path).write_text("\n".join(out_lines) + "\n")

    if int(count_line.split()[0]) != natom:
        raise ValueError("rst7 atom count mismatch after writing.")


def perturb_rst7_inplace(rst7_path, perturbation, seed):
    title, count_line, coords, tail_lines = read_rst7_coords(rst7_path)
    if perturbation > 0:
        rng = np.random.RandomState(seed)
        noise = (rng.rand(*coords.shape) - 0.5) * 2.0 * perturbation
        coords = coords + noise
    write_rst7_coords(rst7_path, title, count_line, coords, tail_lines)
    return coords


def perturb_rst7_with_rigid_water_inplace(
    rst7_path, parm7_path, perturbation, seed, water_resnames=("WAT", "HOH")
):
    title, count_line, coords, tail_lines = read_rst7_coords(rst7_path)
    if perturbation <= 0:
        return coords

    residue_labels = _read_parm7_flag_strings(parm7_path, "RESIDUE_LABEL")
    residue_pointer = [
        int(v) for v in _read_parm7_flag_values(parm7_path, "RESIDUE_POINTER")
    ]
    residue_pointer.append(coords.shape[0] + 1)

    if len(residue_labels) + 1 != len(residue_pointer):
        raise ValueError(
            f"RESIDUE_LABEL/RESIDUE_POINTER mismatch in {parm7_path}: "
            f"{len(residue_labels)} labels vs {len(residue_pointer) - 1} ranges"
        )

    rng = np.random.RandomState(seed)
    water_resname_set = set(water_resnames)
    for i, resname in enumerate(residue_labels):
        start = residue_pointer[i] - 1
        end = residue_pointer[i + 1] - 1
        if end <= start:
            continue

        if resname in water_resname_set:
            # Keep water internal geometry unchanged: one random translation per water residue.
            delta = (rng.rand(3) - 0.5) * 2.0 * perturbation
            coords[start:end] = coords[start:end] + delta
        else:
            delta = (rng.rand(end - start, 3) - 0.5) * 2.0 * perturbation
            coords[start:end] = coords[start:end] + delta

    write_rst7_coords(rst7_path, title, count_line, coords, tail_lines)
    return coords


def _require_case_templates(case_template_dir: Path, case_name: str):
    required = ["sander.in", "tleap.in"]
    if case_name == TIP4P_CASE_NAME:
        required.append("system_minimized.rst7")
    for file_name in required:
        if not (case_template_dir / file_name).exists():
            raise FileNotFoundError(
                "Missing AMBER reference template file: "
                f"{case_template_dir / file_name}"
            )


def _prepare_case_system_files(
    statics_path: Path,
    outputs_path: Path,
    case_statics_root: Path,
    case_name: str,
):
    case_system_dir = case_statics_root / case_name
    case_system_dir.mkdir(parents=True, exist_ok=True)
    _require_case_templates(case_system_dir, case_name)

    base_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        run_tag=f"_refgen_base/{case_name}",
    )
    # Static templates now live under reference/amber/statics.
    # Copy them into runtime workdir before running tleap/sander.
    for src in sorted(case_system_dir.glob("*")):
        if src.is_file():
            shutil.copy2(src, base_dir / src.name)
    run_tleap(base_dir)

    for file_name in ["system.parm7", "system.rst7"]:
        src = base_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing generated file {src}")
        shutil.copy2(src, case_system_dir / file_name)


def _copy_system_files_for_case(
    case_statics_root: Path, case_name: str, case_dir: Path
):
    case_system_dir = case_statics_root / case_name
    if not case_system_dir.exists():
        raise FileNotFoundError(f"Missing case system dir: {case_system_dir}")

    copied = 0
    for src in sorted(case_system_dir.glob("*")):
        if src.is_file():
            shutil.copy2(src, case_dir / src.name)
            copied += 1
    if copied == 0:
        raise ValueError(f"No system files copied for {case_name}")


def _copy_reference_statics(src_root: Path, dst_root: Path):
    src = src_root / "statics"
    dst = dst_root / "statics"
    if dst.exists():
        shutil.rmtree(dst)
    if not src.exists():
        raise FileNotFoundError(f"Missing AMBER statics templates at {src}")
    shutil.copytree(src, dst)


def generate_payload(statics_path: Path, reference_root: Path):
    require_ambertools()

    systems_root = reference_root / "statics"
    forces_root = reference_root / "forces"
    if not systems_root.exists():
        raise FileNotFoundError(
            f"Missing AMBER reference statics directory: {systems_root}"
        )
    if forces_root.exists():
        shutil.rmtree(forces_root)
    forces_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="amber_refgen_outputs_") as tmpdir:
        outputs_path = Path(tmpdir)

        for case_name in [TIP4P_CASE_NAME, GB1_CASE_NAME]:
            _prepare_case_system_files(
                statics_path=statics_path,
                outputs_path=outputs_path,
                case_statics_root=systems_root,
                case_name=case_name,
            )

        sander_banner = None
        entries = []
        for spec in _iter_case_specs():
            case_name = spec["case_name"]
            iteration = spec["iteration"]
            perturbation = spec["perturbation"]
            seed = spec["seed"]
            kind = spec["kind"]

            run_dir = Outputer.prepare_output_case(
                statics_path=statics_path,
                outputs_path=outputs_path,
                case_name=case_name,
                run_tag=f"_refgen/{case_name}/{iteration}",
            )
            _copy_system_files_for_case(systems_root, case_name, run_dir)

            if kind == "tip4p":
                write_tip4p_virtual_atom_from_parm7(
                    run_dir / "system.parm7",
                    run_dir / "tip4p_virtual_atom.txt",
                )
                (run_dir / "system.rst7").write_text(
                    (run_dir / "system_minimized.rst7").read_text()
                )
                perturb_rst7_with_rigid_water_inplace(
                    run_dir / "system.rst7",
                    run_dir / "system.parm7",
                    perturbation=perturbation,
                    seed=seed,
                )
            elif kind == "gb1":
                write_gb_in_file_from_parm7(
                    run_dir / "system.parm7", run_dir / "gb_gb.txt"
                )
                perturb_rst7_inplace(
                    run_dir / "system.rst7",
                    perturbation=perturbation,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported case kind: {kind}")

            run_sander_run0(run_dir)

            sander_out_path = run_dir / "sander.out"
            if sander_banner is None:
                sander_banner = _extract_sander_banner(sander_out_path)

            energy_epot = extract_sander_epot(sander_out_path)
            amber_forces = extract_sander_forces_mdfrc(run_dir / "mdfrc")

            force_rel_path = Path("forces") / case_name / f"iter{iteration}.npy"
            force_path = reference_root / force_rel_path
            force_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(force_path, amber_forces.astype(np.float64))

            entries.append(
                {
                    "case_name": case_name,
                    "iteration": iteration,
                    "perturbation": perturbation,
                    "seed": seed,
                    "energy_epot": float(energy_epot),
                    "natom": int(amber_forces.shape[0]),
                    "forces_file": str(force_rel_path).replace("\\", "/"),
                }
            )

    entries.sort(key=lambda v: (v["case_name"], v["iteration"]))
    payload = {
        "format_version": 1,
        "energy_unit": "kcal/mol",
        "force_unit": "kcal/mol/Angstrom",
        "sander_banner": sander_banner or "unknown",
        "entries": entries,
    }
    return payload


def _entries_to_map(payload):
    result = {}
    entries = payload.get("entries", [])
    for entry in entries:
        key = (entry["case_name"], int(entry["iteration"]))
        result[key] = entry
    return result


def compare_payloads(
    current_payload,
    generated_payload,
    current_root: Path,
    generated_root: Path,
    abs_tol_energy: float,
    abs_tol_force: float,
):
    for key in ["format_version", "energy_unit", "force_unit", "sander_banner"]:
        if current_payload.get(key) != generated_payload.get(key):
            return (
                False,
                f"Metadata differs at '{key}': current={current_payload.get(key)!r}, "
                f"generated={generated_payload.get(key)!r}",
            )

    current_map = _entries_to_map(current_payload)
    generated_map = _entries_to_map(generated_payload)
    if set(current_map) != set(generated_map):
        current_only = sorted(set(current_map) - set(generated_map))
        generated_only = sorted(set(generated_map) - set(current_map))
        return (
            False,
            "Entry keys differ. "
            f"current-only={len(current_only)}, generated-only={len(generated_only)}",
        )

    max_energy_diff = 0.0
    max_energy_key = None
    max_force_diff = 0.0
    max_force_key = None

    for key in sorted(current_map):
        current_entry = current_map[key]
        generated_entry = generated_map[key]

        if (
            abs(
                float(current_entry["perturbation"])
                - float(generated_entry["perturbation"])
            )
            > 1.0e-12
        ):
            return (
                False,
                f"Perturbation differs for {key}: current={current_entry['perturbation']}, "
                f"generated={generated_entry['perturbation']}",
            )
        if int(current_entry["seed"]) != int(generated_entry["seed"]):
            return (
                False,
                f"Seed differs for {key}: current={current_entry['seed']}, "
                f"generated={generated_entry['seed']}",
            )
        if int(current_entry["natom"]) != int(generated_entry["natom"]):
            return (
                False,
                f"natom differs for {key}: current={current_entry['natom']}, "
                f"generated={generated_entry['natom']}",
            )

        energy_diff = abs(
            float(current_entry["energy_epot"])
            - float(generated_entry["energy_epot"])
        )
        if energy_diff > max_energy_diff:
            max_energy_diff = energy_diff
            max_energy_key = key

        current_force = np.asarray(
            np.load(current_root / current_entry["forces_file"]),
            dtype=np.float64,
        )
        generated_force = np.asarray(
            np.load(generated_root / generated_entry["forces_file"]),
            dtype=np.float64,
        )
        if current_force.shape != generated_force.shape:
            return (
                False,
                f"Force shape differs for {key}: current={current_force.shape}, "
                f"generated={generated_force.shape}",
            )
        force_diff = float(np.max(np.abs(current_force - generated_force)))
        if force_diff > max_force_diff:
            max_force_diff = force_diff
            max_force_key = key

    if max_energy_diff > abs_tol_energy:
        return (
            False,
            f"Energy differs above tolerance: max_diff={max_energy_diff:.3e} at {max_energy_key}",
        )
    if max_force_diff > abs_tol_force:
        return (
            False,
            f"Force differs above tolerance: max_diff={max_force_diff:.3e} at {max_force_key}",
        )

    return (
        True,
        f"max_energy_diff={max_energy_diff:.3e}, max_force_diff={max_force_diff:.3e}",
    )


def main():
    repo_root = get_repo_root()
    default_output = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "reference"
        / "amber"
        / "reference.json"
    )
    default_statics = (
        repo_root / "benchmarks" / "comparison" / "tests" / "amber" / "statics"
    )

    parser = argparse.ArgumentParser(
        description="Generate static AMBER reference energies/forces for comp-amber tests."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--statics-path", type=Path, default=default_statics)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--abs-tol-energy", type=float, default=1.0e-6)
    parser.add_argument("--abs-tol-force", type=float, default=1.0e-6)
    args = parser.parse_args()

    if args.check:
        if not args.output.exists():
            print(f"[FAIL] Missing reference file: {args.output}")
            raise SystemExit(1)
        with tempfile.TemporaryDirectory(
            prefix="amber_refgen_check_"
        ) as tmpdir:
            tmp_output = Path(tmpdir) / "reference.json"
            tmp_root = tmp_output.parent
            _copy_reference_statics(args.output.parent.resolve(), tmp_root)
            generated_payload = generate_payload(args.statics_path, tmp_root)
            tmp_output.write_text(
                json.dumps(generated_payload, indent=2) + "\n"
            )

            current_payload = json.loads(args.output.read_text())
            ok, detail = compare_payloads(
                current_payload=current_payload,
                generated_payload=generated_payload,
                current_root=args.output.parent,
                generated_root=tmp_root,
                abs_tol_energy=args.abs_tol_energy,
                abs_tol_force=args.abs_tol_force,
            )
            if ok:
                print(f"[PASS] AMBER reference is up to date ({detail}).")
                return
            print(f"[FAIL] AMBER reference differs. {detail}")
            raise SystemExit(1)

    payload = generate_payload(args.statics_path, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] Wrote {len(payload['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
