import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

from benchmarks.comparison.utils import get_comparison_root_from_statics
from benchmarks.utils import Runner

HARTREE_TO_KCAL_MOL = 627.509474

SUPPORTED_DFT_METHODS = ["LDA", "PBE", "BLYP", "PBE0", "B3LYP"]
PYSCF_REFERENCE_ENERGY_REL_PATH = "reference/pyscf/reference.json"
PYSCF_REFERENCE_STATICS_REL_DIR = "reference/pyscf/statics"

QC_ENERGY_PATTERN = re.compile(
    r"(?:QC|SCF_Energy)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _sanitize_token(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace("*", "star")
        .replace("+", "plus")
        .replace(" ", "")
    )


def _read_non_empty_lines(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _read_atoms_from_xyz(xyz_path: Path):
    lines = _read_non_empty_lines(xyz_path)
    if len(lines) < 2:
        raise ValueError(f"Invalid xyz file (too short): {xyz_path}")

    natom = int(lines[0].split()[0])
    if len(lines) < natom + 2:
        raise ValueError(f"Invalid xyz file (atom lines mismatch): {xyz_path}")

    atoms = []
    for i in range(natom):
        parts = lines[2 + i].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid xyz atom line {i} in {xyz_path}")
        sym = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        atoms.append((sym, x, y, z))
    return natom, atoms


def get_reference_xyz_path(statics_path: Path, case_name: str):
    comparison_root = get_comparison_root_from_statics(statics_path)
    return (
        comparison_root / PYSCF_REFERENCE_STATICS_REL_DIR / f"{case_name}.xyz"
    ).resolve()


def load_case_definition(statics_path: Path, case_name: str):
    sponge_dir = statics_path / case_name / "sponge"
    if not sponge_dir.exists():
        raise FileNotFoundError(f"Static case not found: {sponge_dir}")

    qc_type_lines = _read_non_empty_lines(sponge_dir / "qc_type.txt")
    if len(qc_type_lines) < 1:
        raise ValueError(f"qc_type.txt is empty for case: {case_name}")

    head = qc_type_lines[0].split()
    if len(head) < 3:
        raise ValueError(f"Invalid qc_type header in case: {case_name}")
    natom = int(head[0])
    charge = int(head[1])
    multiplicity = int(head[2])

    symbols = {}
    for line in qc_type_lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        symbols[int(parts[0])] = parts[1]
    if len(symbols) != natom:
        raise ValueError(
            f"qc_type atom count mismatch for {case_name}: "
            f"expected {natom}, got {len(symbols)}"
        )

    xyz_file = get_reference_xyz_path(statics_path, case_name)
    if not xyz_file.exists():
        raise FileNotFoundError(
            f"Missing reference xyz for case {case_name}: {xyz_file}"
        )
    xyz_natom, atoms = _read_atoms_from_xyz(xyz_file)
    if xyz_natom != natom:
        raise ValueError(
            f"xyz atom count mismatch for {case_name}: "
            f"qc_type={natom}, xyz={xyz_natom}"
        )
    for i, atom in enumerate(atoms):
        if atom[0] != symbols[i]:
            raise ValueError(
                f"xyz symbol mismatch for {case_name} atom {i}: "
                f"qc_type={symbols[i]}, xyz={atom[0]}"
            )
    return {
        "case_name": case_name,
        "charge": charge,
        "multiplicity": multiplicity,
        "atoms": atoms,
        "static_sponge_dir": sponge_dir,
    }


def prepare_output_case(
    statics_path: Path,
    outputs_path: Path,
    case_name: str,
    run_tag: str,
    *,
    mpi_np=None,
):
    static_case = statics_path / case_name / "sponge"
    if not static_case.exists():
        raise FileNotFoundError(
            f"Static sponge directory not found: {static_case}"
        )

    run_dir = outputs_path / run_tag
    sponge_dir = run_dir / "sponge"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(static_case, sponge_dir)
    _normalize_sponge_input_files_for_platform(sponge_dir)
    return run_dir, sponge_dir


def _normalize_sponge_input_files_for_platform(sponge_dir: Path):
    # Windows SPONGE rejects LF-only text inputs from Linux checkouts.
    if os.name != "nt":
        return

    for input_file in sponge_dir.rglob("*.txt"):
        with open(input_file, "r", encoding="utf-8", newline=None) as f:
            content = f.read()
        with open(input_file, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(content)


def run_sponge_scf_energy_ha(
    sponge_dir: Path,
    model_chemistry: str,
    restricted: bool,
    mpi_np=None,
    extra_sponge_args=None,
):
    output = Runner.run_sponge(
        sponge_dir,
        mpi_np=mpi_np,
        mdin_name="mdin.txt",
        sponge_cmd=os.environ.get("SPONGE_BIN", "SPONGE"),
        extra_args=["-qc_model_chemistry", model_chemistry]
        + ([] if restricted else ["-qc_restricted", "0"])
        + (extra_sponge_args or []),
    )

    matches = QC_ENERGY_PATTERN.findall(output)
    if not matches:
        raise ValueError(
            "Failed to parse QC energy from SPONGE output.\n"
            f"Output tail:\n{output[-2000:]}"
        )

    raw_energy = float(matches[-1])
    output_lower = output.lower()

    # Prefer explicit unit tags if present; otherwise use a safe magnitude
    # heuristic to remain compatible with both Ha and kcal/mol outputs.
    if "kcal/mol" in output_lower:
        return raw_energy / HARTREE_TO_KCAL_MOL
    if "hartree" in output_lower or " (ha)" in output_lower:
        return raw_energy
    if abs(raw_energy) > 500.0:
        return raw_energy / HARTREE_TO_KCAL_MOL
    return raw_energy


def _build_pyscf_method(
    atoms,
    basis_name: str,
    charge: int,
    multiplicity: int,
    method_name: str,
    restricted: bool,
):
    from pyscf import dft, gto, scf

    atom_str = "\n".join(f"{sym} {x} {y} {z}" for sym, x, y, z in atoms)
    spin = multiplicity - 1
    mol = gto.M(
        atom=atom_str,
        basis=basis_name,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=0,
    )

    method = method_name.strip().upper()
    if method == "HF":
        mf = scf.RHF(mol) if restricted else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if restricted else dft.UKS(mol)
        if method == "LDA":
            mf.xc = "LDA,VWN5"
        elif method == "PBE":
            mf.xc = "PBE,PBE"
        elif method == "BLYP":
            mf.xc = "B88,LYP"
        elif method == "PBE0":
            mf.xc = "PBE0"
        elif method == "B3LYP":
            mf.xc = "0.2*HF + 0.08*SLATER + 0.72*B88, 0.81*LYP + 0.19*VWN5"
        else:
            raise ValueError(f"Unsupported DFT method for PySCF: {method_name}")

    # Open-shell systems can converge to different local solutions when relying
    # on PySCF's default initial guess. Force atomic guess to stabilize
    # unrestricted references across repeated runs.
    if not restricted:
        mf.init_guess = "atom"

    return mol, mf


def run_pyscf_energy_ha(
    atoms,
    basis_name: str,
    charge: int,
    multiplicity: int,
    method_name: str,
    restricted: bool,
):
    _mol, mf = _build_pyscf_method(
        atoms=atoms,
        basis_name=basis_name,
        charge=charge,
        multiplicity=multiplicity,
        method_name=method_name,
        restricted=restricted,
    )
    return float(mf.kernel())


def run_pyscf_energy_and_mulliken(
    atoms,
    basis_name: str,
    charge: int,
    multiplicity: int,
    method_name: str,
    restricted: bool,
):
    mol, mf = _build_pyscf_method(
        atoms=atoms,
        basis_name=basis_name,
        charge=charge,
        multiplicity=multiplicity,
        method_name=method_name,
        restricted=restricted,
    )
    energy_ha = float(mf.kernel())
    overlap = mol.intor_symmetric("int1e_ovlp")
    dm = mf.make_rdm1()
    _pop, atom_charges = mf.mulliken_pop(mol, dm, s=overlap, verbose=0)
    return energy_ha, [float(v) for v in atom_charges]


def run_pyscf_mulliken_charges(
    atoms,
    basis_name: str,
    charge: int,
    multiplicity: int,
    method_name: str,
    restricted: bool,
):
    _energy_ha, mulliken_charges = run_pyscf_energy_and_mulliken(
        atoms=atoms,
        basis_name=basis_name,
        charge=charge,
        multiplicity=multiplicity,
        method_name=method_name,
        restricted=restricted,
    )
    return mulliken_charges


def _to_bool_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean-like value: {value!r}")


def _build_pyscf_reference_key(
    case_name: str, method_name: str, basis_name: str, restricted
):
    return (
        str(case_name).strip().lower(),
        str(method_name).strip().upper(),
        str(basis_name).strip().lower(),
        _to_bool_flag(restricted),
    )


@lru_cache(maxsize=8)
def _load_pyscf_reference_table(reference_path_str: str):
    reference_path = Path(reference_path_str)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing PySCF reference file: {reference_path}"
        )

    with open(reference_path, "r") as f:
        payload = json.load(f)

    entries = payload
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Invalid PySCF reference format (entries should be a list): "
            f"{reference_path}"
        )

    table = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid PySCF reference entry #{idx}: expected object"
            )
        try:
            key = _build_pyscf_reference_key(
                case_name=entry["case_name"],
                method_name=entry["method_name"],
                basis_name=entry["basis_name"],
                restricted=entry["restricted"],
            )
            energy_ha = float(entry["energy_ha"])
        except KeyError as exc:
            raise ValueError(
                f"Invalid PySCF reference entry #{idx}: missing {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Invalid PySCF reference entry #{idx}: {exc}"
            ) from exc
        if key in table:
            raise ValueError(
                "Duplicate PySCF reference entry for "
                f"{key[0]} {key[1]}/{key[2]} restricted={int(key[3])}"
            )
        table[key] = energy_ha
    return table


def get_pyscf_reference_energy_ha(
    statics_path: Path,
    case_name: str,
    method_name: str,
    basis_name: str,
    restricted: bool,
):
    comparison_root = get_comparison_root_from_statics(statics_path)
    reference_path = (
        comparison_root / PYSCF_REFERENCE_ENERGY_REL_PATH
    ).resolve()
    table = _load_pyscf_reference_table(str(reference_path))
    key = _build_pyscf_reference_key(
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=restricted,
    )
    if key not in table:
        raise KeyError(
            "PySCF reference energy not found for "
            f"{case_name} {method_name}/{basis_name} "
            f"restricted={int(bool(restricted))} in {reference_path}"
        )
    return table[key]


def run_sponge_vs_pyscf(
    statics_path: Path,
    outputs_path: Path,
    case_name: str,
    method_name: str,
    basis_name: str,
    restricted: bool,
    run_prefix: str,
    mpi_np=None,
    extra_sponge_args=None,
):
    model_chemistry = f"{method_name}/{basis_name}"
    run_name = "_".join(
        [
            _sanitize_token(run_prefix),
            _sanitize_token(case_name),
            _sanitize_token(method_name),
            _sanitize_token(basis_name),
            f"r{int(restricted)}",
        ]
    )
    _run_dir, sponge_dir = prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        run_tag=run_name,
        mpi_np=mpi_np,
    )

    sponge_energy_ha = run_sponge_scf_energy_ha(
        sponge_dir=sponge_dir,
        model_chemistry=model_chemistry,
        restricted=restricted,
        mpi_np=mpi_np,
        extra_sponge_args=extra_sponge_args,
    )
    pyscf_energy_ha = get_pyscf_reference_energy_ha(
        statics_path=statics_path,
        case_name=case_name,
        method_name=method_name,
        basis_name=basis_name,
        restricted=restricted,
    )

    diff_ha = sponge_energy_ha - pyscf_energy_ha
    abs_diff_ha = abs(diff_ha)
    pyscf_energy_kcal = pyscf_energy_ha * HARTREE_TO_KCAL_MOL
    sponge_energy_kcal = sponge_energy_ha * HARTREE_TO_KCAL_MOL
    abs_diff_kcal = abs_diff_ha * HARTREE_TO_KCAL_MOL
    return {
        "case_name": case_name,
        "method": method_name,
        "basis": basis_name,
        "restricted": restricted,
        "pyscf_energy_ha": pyscf_energy_ha,
        "pyscf_energy_kcal_mol": pyscf_energy_kcal,
        "sponge_energy_ha": sponge_energy_ha,
        "sponge_energy_kcal_mol": sponge_energy_kcal,
        "abs_diff_ha": abs_diff_ha,
        "abs_diff_kcal_mol": abs_diff_kcal,
    }
