import json
import os
import shlex
import shutil
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np


KJ_PER_MOL_TO_KCAL_PER_MOL = 0.2390057361376673
KJ_PER_MOL_PER_NM_TO_KCAL_PER_MOL_PER_A = 0.02390057361376673
GROMACS_REFERENCE_JSON_REL_PATH = "reference/gromacs/reference.json"
GROMACS_REFERENCE_ROOT_REL_DIR = "reference/gromacs"
GROMACS_REFERENCE_STATICS_REL_DIR = "reference/gromacs/statics"


def print_validation_table(headers, rows, title=None):
    if not headers:
        return

    if title:
        print(f"\n{title}")
    else:
        print()

    if len(rows) == 1:
        row = [str(v) for v in rows[0]]
        if len(row) != len(headers):
            raise ValueError(
                "Header/row length mismatch in validation table: "
                f"headers={len(headers)}, row={len(row)}"
            )
        key_width = max(len(h) for h in headers)
        value_width = max(len(v) for v in row) if row else 0
        divider = "-" * (key_width + 3 + value_width)
        print(divider)
        for key, value in zip(headers, row):
            print(f"{key:<{key_width}} : {value}")
        print(divider)
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            text = str(val)
            col_widths[i] = max(col_widths[i], len(text))
    col_widths = [w + 2 for w in col_widths]
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

    print(divider)
    print(row_fmt.format(*headers))
    print(divider)
    for row in rows:
        print(row_fmt.format(*[str(v) for v in row]))
    print(divider)


def _get_comparison_root_from_statics(statics_path):
    return Path(statics_path).resolve().parents[2]


def _get_gromacs_reference_root(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path)
        / GROMACS_REFERENCE_ROOT_REL_DIR
    )


def _get_gromacs_reference_json_path(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path)
        / GROMACS_REFERENCE_JSON_REL_PATH
    )


def _get_gromacs_reference_case_dir(statics_path, case_name):
    return (
        _get_comparison_root_from_statics(statics_path)
        / GROMACS_REFERENCE_STATICS_REL_DIR
        / case_name
    )


@lru_cache(maxsize=4)
def _load_gromacs_reference_entries(reference_json_path_str):
    reference_json_path = Path(reference_json_path_str)
    if not reference_json_path.exists():
        raise FileNotFoundError(
            f"Missing GROMACS reference file: {reference_json_path}"
        )
    payload = json.loads(reference_json_path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Invalid GROMACS reference format in {reference_json_path}: "
            "'entries' must be a list."
        )

    index = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid GROMACS reference entry #{idx}: expected object"
            )
        try:
            case_name = str(entry["case_name"])
            iteration = int(entry["iteration"])
        except KeyError as exc:
            raise ValueError(
                f"Invalid GROMACS reference entry #{idx}: missing {exc}"
            ) from exc
        key = (case_name, iteration)
        if key in index:
            raise ValueError(
                f"Duplicate GROMACS reference entry for case={case_name}, "
                f"iteration={iteration}"
            )
        index[key] = entry
    return payload, index


def load_gromacs_reference_entry(statics_path, case_name, iteration):
    json_path = _get_gromacs_reference_json_path(statics_path).resolve()
    _payload, index = _load_gromacs_reference_entries(str(json_path))
    key = (str(case_name), int(iteration))
    if key not in index:
        raise KeyError(
            "GROMACS reference entry not found for "
            f"case={case_name}, iteration={iteration} in {json_path}"
        )
    return index[key]


def load_gromacs_reference_terms(statics_path, case_name, iteration):
    entry = load_gromacs_reference_entry(statics_path, case_name, iteration)
    terms = {}
    for term_name in [
        "bond",
        "angle",
        "urey_bradley",
        "proper_dihedral",
        "improper_dihedral",
        "lj14",
        "coulomb14",
        "lj_sr",
        "coulomb_sr",
        "coulomb_recip",
        "potential",
        "pressure",
    ]:
        if term_name not in entry:
            raise ValueError(
                f"Missing term '{term_name}' in GROMACS reference entry: "
                f"case={case_name}, iteration={iteration}"
            )
        terms[term_name] = float(entry[term_name])
    return terms


def load_gromacs_reference_forces(statics_path, case_name, iteration):
    entry = load_gromacs_reference_entry(statics_path, case_name, iteration)
    forces_rel_path = entry.get("forces_file")
    if not isinstance(forces_rel_path, str) or not forces_rel_path:
        raise ValueError(
            "GROMACS reference entry missing 'forces_file': "
            f"case={case_name}, iteration={iteration}"
        )
    force_path = (
        _get_gromacs_reference_root(statics_path) / forces_rel_path
    ).resolve()
    if not force_path.exists():
        raise FileNotFoundError(
            f"Missing GROMACS reference forces file: {force_path}"
        )
    arr = np.load(force_path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Invalid GROMACS reference forces shape {arr.shape} in {force_path}"
        )
    return arr


def copy_gromacs_reference_case_files(statics_path, case_dir, case_name):
    src_dir = _get_gromacs_reference_case_dir(statics_path, case_name).resolve()
    dst_dir = Path(case_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Missing GROMACS reference statics directory: {src_dir}"
        )

    copied = []
    for src_file in sorted(src_dir.glob("*")):
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file.name)
    if not copied:
        raise ValueError(
            f"No reference statics files found in {src_dir} for case {case_name}"
        )
    return copied


def copy_gromacs_reference_sponge_inputs(
    statics_path,
    case_dir,
    case_name,
    iteration,
):
    entry = load_gromacs_reference_entry(statics_path, case_name, iteration)
    rel_dir = entry.get("sponge_inputs_dir")
    if not isinstance(rel_dir, str) or not rel_dir:
        raise ValueError(
            "GROMACS reference entry missing 'sponge_inputs_dir': "
            f"case={case_name}, iteration={iteration}"
        )
    src_dir = (_get_gromacs_reference_root(statics_path) / rel_dir).resolve()
    dst_dir = Path(case_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Missing GROMACS reference sponge inputs directory: {src_dir}"
        )

    copied = []
    for src_file in sorted(src_dir.glob("*")):
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file.name)
    if not copied:
        raise ValueError(
            f"No sponge input files found in {src_dir} for "
            f"case={case_name}, iteration={iteration}"
        )
    return copied


def require_gromacs():
    if shutil.which("gmx") is None:
        raise RuntimeError(
            "Required executable 'gmx' is not available in PATH."
        )


def require_xponge():
    env = os.environ.copy()
    # Avoid polluted site-packages from shell-level AMBER PYTHONPATH.
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        ["python", "-c", "import Xponge"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-1200:]
        raise RuntimeError(
            "Python package 'Xponge' is required for GROMACS->SPONGE extraction.\n"
            f"Output tail:\n{tail}"
        )


def prepare_output_case(statics_path, outputs_path, case_name, run_tag=None):
    static_case = statics_path / case_name
    if not static_case.exists():
        raise FileNotFoundError(f"Static case not found: {static_case}")

    case_dir = outputs_path / (run_tag or case_name)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(static_case, case_dir)
    return case_dir


def _run_command(cmd, cwd, env=None, input_text=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input=input_text,
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        cmd0 = Path(str(cmd[0])).name.lower() if cmd else ""
        if cmd0 in {"sponge", "sponge.exe"}:
            print("\n[SPONGE stdout]\n")
            print(result.stdout)
            print("\n[SPONGE stderr]\n")
            print(result.stderr)
        raise RuntimeError(
            f"Command failed in {cwd} with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{output[-3000:]}"
        )
    return output


def _clean_python_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    return env


def link_charmm27_forcefield(case_dir):
    link_path = Path(case_dir) / "charmm27.ff"
    if link_path.exists():
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(
            Path(conda_prefix) / "share" / "gromacs" / "top" / "charmm27.ff"
        )

    gmx_bin = shutil.which("gmx")
    if gmx_bin:
        candidates.append(
            Path(gmx_bin).resolve().parent.parent
            / "share"
            / "gromacs"
            / "top"
            / "charmm27.ff"
        )

    source = None
    for cand in candidates:
        if cand.exists():
            source = cand
            break

    if source is None:
        raise FileNotFoundError(
            "Failed to locate GROMACS charmm27 forcefield directory (charmm27.ff)."
        )

    link_path.symlink_to(source)


def run_gromacs_flexible_run0(case_dir):
    _run_command(
        [
            "gmx",
            "grompp",
            "-f",
            "run0_flex.mdp",
            "-c",
            "solv.gro",
            "-p",
            "topol.top",
            "-o",
            "topol_flex.tpr",
            "-maxwarn",
            "10",
        ],
        cwd=case_dir,
    )
    _run_command(
        [
            "gmx",
            "mdrun",
            "-s",
            "topol_flex.tpr",
            "-deffnm",
            "run0_flex",
            "-nt",
            "1",
        ],
        cwd=case_dir,
    )


def extract_gromacs_terms(case_dir):
    selection = "\n".join(
        [
            "Bond",
            "Angle",
            "U-B",
            "Proper-Dih.",
            "Improper-Dih.",
            "LJ-14",
            "Coulomb-14",
            "LJ-(SR)",
            "Coulomb-(SR)",
            "Coul.-recip.",
            "Potential",
            "Pressure",
            "",
        ]
    )
    _run_command(
        ["gmx", "energy", "-f", "run0_flex.edr", "-o", "terms_flex_full.xvg"],
        cwd=case_dir,
        input_text=selection,
    )

    values = None
    for line in (
        (Path(case_dir) / "terms_flex_full.xvg").read_text().splitlines()
    ):
        if line.startswith("#") or line.startswith("@") or not line.strip():
            continue
        values = [float(v) for v in line.split()]
        break

    if values is None or len(values) != 13:
        raise ValueError(
            "Failed to parse GROMACS term vector from terms_flex_full.xvg"
        )

    return {
        "bond": values[1] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "angle": values[2] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "urey_bradley": values[3] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "proper_dihedral": values[4] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "improper_dihedral": values[5] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "lj14": values[6] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "coulomb14": values[7] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "lj_sr": values[8] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "coulomb_sr": values[9] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "coulomb_recip": values[10] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "potential": values[11] * KJ_PER_MOL_TO_KCAL_PER_MOL,
        "pressure": values[12],
    }


def _read_natom_from_gro(gro_path):
    lines = Path(gro_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    return int(lines[1].split()[0])


def perturb_gro_inplace(
    gro_path,
    perturbation_angstrom,
    seed,
    perturb_non_water=True,
):
    gro_path = Path(gro_path)
    lines = gro_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")

    natom = int(lines[1].split()[0])
    if len(lines) < 3 + natom:
        raise ValueError(
            f"Invalid .gro file length in {gro_path}: natom={natom}, lines={len(lines)}"
        )
    if perturbation_angstrom <= 0:
        return

    perturbation_nm = perturbation_angstrom / 10.0
    rng = np.random.RandomState(seed)
    out_lines = [lines[0], lines[1]]
    water_resnames = {"SOL", "WAT", "HOH"}
    residue_delta = {}
    for idx in range(natom):
        line = lines[2 + idx]
        if len(line) < 44:
            raise ValueError(
                f"Invalid atom line in .gro at {idx + 1}: {line!r}"
            )
        resid = int(line[0:5])
        resname = line[5:10]
        resname_key = resname.strip()
        atomname = line[10:15]
        atomnr = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        tail = line[44:]

        if resname_key in water_resnames:
            key = (resid, resname_key)
            if key not in residue_delta:
                residue_delta[key] = (rng.rand(3) - 0.5) * 2.0 * perturbation_nm
            dx, dy, dz = residue_delta[key]
        else:
            if perturb_non_water:
                dx, dy, dz = (rng.rand(3) - 0.5) * 2.0 * perturbation_nm
            else:
                dx, dy, dz = 0.0, 0.0, 0.0
        x += dx
        y += dy
        z += dz

        out_lines.append(
            f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}{tail}"
        )

    out_lines.extend(lines[2 + natom :])
    gro_path.write_text("\n".join(out_lines) + "\n")


def extract_gromacs_forces(case_dir):
    _run_command(
        [
            "gmx",
            "traj",
            "-s",
            "topol_flex.tpr",
            "-f",
            "run0_flex.trr",
            "-of",
            "forces_flex.xvg",
        ],
        cwd=case_dir,
        input_text="System\n",
    )

    natom = _read_natom_from_gro(Path(case_dir) / "solv.gro")
    data = None
    for line in (Path(case_dir) / "forces_flex.xvg").read_text().splitlines():
        if line.startswith("#") or line.startswith("@") or not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 1 + 3 * natom:
            raise ValueError(
                "Invalid GROMACS force frame width in forces_flex.xvg: "
                f"expected {1 + 3 * natom}, got {len(values)}"
            )
        data = values[1:]

    if data is None:
        raise ValueError("No numeric force frame found in forces_flex.xvg")

    forces = np.asarray(data, dtype=np.float64).reshape(natom, 3)
    return forces * KJ_PER_MOL_PER_NM_TO_KCAL_PER_MOL_PER_A


def generate_sponge_inputs_from_gromacs(case_dir, output_prefix="sys_flexible"):
    script = f"""
import Xponge
import Xponge.forcefield.charmm as charmm
from Xponge.load import load_ffitp
from Xponge import AtomType, load_molitp, load_gro, save_sponge_input
from Xponge.forcefield.base import (
    bond_base,
    dihedral_base,
    lj_base,
    ub_angle_base,
    improper_base,
    nb14_extra_base,
    nb14_base,
    cmap_base,
)

output = load_ffitp("./charmm27.ff/forcefield.itp", macros={{"FLEXIBLE": ""}})
AtomType.New_From_String(output["atomtypes"])
bond_base.BondType.New_From_String(output["bonds"])
# tip3p.itp (FLEXIBLE branch, non-CHARMM_TIP3P constants)
bond_base.BondType.New_From_String(
    \"\"\"
name b[nm] k[kJ/mol\\u00b7nm^-2]
OWT3-HWT3 0.09572 251208.0
\"\"\"
)

dihedral_base.ProperType.New_From_String(output["dihedrals"])
lj_base.LJType.New_From_String(output["LJ"])
ub_angle_base.UreyBradleyType.New_From_String(output["Urey-Bradley"])
ub_angle_base.UreyBradleyType.New_From_String(
    \"\"\"
name b[degree] k[kJ/mol\\u00b7rad^-2] r13[nm] kUB[kJ/mol\\u00b7nm^-2]
HWT3-OWT3-HWT3 104.52 314.01 0 0
\"\"\"
)
improper_base.ImproperType.New_From_String(output["impropers"])
nb14_extra_base.NB14Type.New_From_String(output["nb14_extra"])
nb14_base.NB14Type.New_From_String(output["nb14"])
cmap_base.CMapType.New_From_Dict(output["cmaps"])

system, _ = load_molitp("topol.top", water_replace=False, macros={{"FLEXIBLE": ""}})
load_gro("solv.gro", system)
save_sponge_input(system, "{output_prefix}")
"""

    _run_command(
        ["python", "-c", script],
        cwd=case_dir,
        env=_clean_python_env(),
    )


def write_sponge_run0_mdin(case_dir, input_prefix="sys_flexible"):
    mdin = (
        textwrap.dedent(
            f"""
        gromacs charmm27 tip3p flexible run0
        mode = nve
        step_limit = 0
        dt = 0
        cutoff = 12.0
        default_in_file_prefix = {input_prefix}
        frc = frc.dat
        print_pressure = 1
        print_zeroth_frame = 1
        write_mdout_interval = 1
        """
        ).strip()
        + "\n"
    )
    Path(case_dir, "sponge.mdin").write_text(mdin)


def _resolve_sponge_command():
    sponge_bin = os.environ.get("SPONGE_BIN", "SPONGE")
    return shlex.split(sponge_bin)


def run_sponge_run0(case_dir):
    cmd = _resolve_sponge_command() + ["-mdin", "sponge.mdin"]
    return _run_command(cmd, cwd=case_dir)


def extract_sponge_terms(case_dir):
    mdout_path = Path(case_dir) / "mdout.txt"
    lines = mdout_path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid SPONGE mdout: {mdout_path}")

    headers = lines[0].split()
    data_line = None
    for line in reversed(lines[1:]):
        if not line.strip():
            continue
        if line.strip()[0].isdigit():
            data_line = line
            break

    if data_line is None:
        raise ValueError(f"Failed to find numeric data line in {mdout_path}")

    values = data_line.split()
    if len(values) != len(headers):
        raise ValueError(
            f"mdout header/value mismatch in {mdout_path}: {len(headers)} vs {len(values)}"
        )

    kv = dict(zip(headers, values))
    pxx = float(kv["Pxx"])
    pyy = float(kv["Pyy"])
    pzz = float(kv["Pzz"])

    return {
        "bond": float(kv["bond"]),
        "urey_bradley": float(kv["urey_bradley"]),
        "proper_dihedral": float(kv["dihedral"]),
        "improper_dihedral": float(kv["improper_dihedral"]),
        "lj_short": float(kv["LJ_short"]),
        "lj": float(kv["LJ"]),
        "nb14_lj": float(kv["nb14_LJ"]),
        "nb14_ee": float(kv["nb14_EE"]),
        "pm": float(kv["PM"]),
        "potential": float(kv["potential"]),
        "pressure": (pxx + pyy + pzz) / 3.0,
    }


def extract_sponge_forces(case_dir, natom):
    raw = np.fromfile(Path(case_dir) / "frc.dat", dtype=np.float32)
    frame_width = natom * 3
    if frame_width == 0 or raw.size % frame_width != 0:
        raise ValueError(
            "Invalid frc.dat length: "
            f"size={raw.size}, natom={natom}, path={Path(case_dir) / 'frc.dat'}"
        )
    return raw[-frame_width:].reshape(natom, 3).astype(np.float64)


def force_stats(reference, predicted):
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"Force shape mismatch: ref={reference.shape}, pred={predicted.shape}"
        )
    diff = predicted - reference
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    rms = float(np.sqrt(np.mean(diff * diff)))

    ref_norm = np.linalg.norm(reference, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    denom = ref_norm * pred_norm
    valid = denom > 1.0e-12
    if np.any(valid):
        cos = float(
            np.mean(
                np.sum(reference[valid] * predicted[valid], axis=1)
                / denom[valid]
            )
        )
    else:
        cos = 1.0
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "cosine_similarity": cos,
    }
