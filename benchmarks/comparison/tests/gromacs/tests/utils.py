import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from benchmarks.comparison.utils import (
    force_stats,
    get_reference_json_path,
    get_reference_root,
    get_reference_statics_case_dir,
    load_reference_npy,
)
from benchmarks.comparison.utils import (
    load_reference_entry as load_common_reference_entry,
)
from benchmarks.utils import Extractor, Runner


KJ_PER_MOL_TO_KCAL_PER_MOL = 0.2390057361376673
KJ_PER_MOL_PER_NM_TO_KCAL_PER_MOL_PER_A = 0.02390057361376673


def load_gromacs_reference_entry(statics_path, case_name, iteration):
    return load_common_reference_entry(
        get_reference_json_path(statics_path, "gromacs"),
        "GROMACS",
        case_name,
        iteration,
    )


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
    return load_reference_npy(
        get_reference_root(statics_path, "gromacs"),
        entry,
        key="forces_file",
        suite_label="GROMACS",
        expected_ndim=2,
        expected_last_dim=3,
    )


def copy_gromacs_reference_case_files(statics_path, case_dir, case_name):
    src_dir = get_reference_statics_case_dir(
        statics_path, "gromacs", case_name
    ).resolve()
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
    src_dir = (get_reference_root(statics_path, "gromacs") / rel_dir).resolve()
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
    Runner.run_command(
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
    Runner.run_command(
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
    Runner.run_command(
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
    Runner.run_command(
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

    Runner.run_command(
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
    return Extractor.extract_sponge_forces(case_dir, natom)
