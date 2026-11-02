import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "h5_bundle" / "fixtures" / "input_matrix"
XPONGE_DEV_ROOT = REPO_ROOT.parent / "XPONGE"

FULL_CONTRACT_IDS = {
    "restart.coordinate",
    "restart.velocity",
    "restart.box",
    "trajectory.crd",
    "trajectory.box",
    "trajectory.vel",
    "topology.mass",
    "topology.charge",
    "topology.residue",
    "topology.bond",
    "topology.REAXFF",
    "topology.REAXFF_type",
    "protocol.cv",
    "protocol.constrain",
    "protocol.restrain",
    "protocol.restrain_cv",
    "protocol.soft_walls",
    "protocol.steer_cv",
    "restart.protocol_sidecar.cv_in_file",
    "restart.protocol_sidecar.SITS_in_file",
    "restart.protocol_sidecar.meta_potential_in_file",
    "protocol.SITS",
    "protocol.SITS_atom",
    "restart.SITS_nk",
    "protocol.restrain_atom_id",
    "protocol.restrain_weight",
    "restart.restrain_coordinate",
    "protocol.meta_edge",
    "restart.meta_potential",
    "restart.meta_scatter",
    "restart.hills",
    "restart.nose_hoover_chain_restart_input",
    "topology.pairwise_force",
    "topology.listed_forces",
    "topology.pairwise_force_data.custom_pair",
    "topology.listed_force_data.custom_bond",
    "output.h5.output_h5_trajectory_path",
    "output.h5.output_h5_trajectory_vds",
    "output.h5.output_h5_restart_path",
    "output.h5.output_h5_observable_path",
}

CORE_NORMAL_IDS = {
    "restart.coordinate",
    "restart.velocity",
    "restart.box",
    "topology.mass",
    "topology.charge",
    "protocol.cv",
    "protocol.restrain",
    "protocol.SITS",
    "restart.SITS_nk",
}

REQUIRED_FULL_BUNDLE_PATHS = {
    "restart.spgr.h5": {
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
        "/parameters/restart/thermostat/nose_hoover_chain",
        "/parameters/restart/bias/sits/SITS/nk",
        "/parameters/restart/bias/meta/default/potential_export",
        "/parameters/restart/bias/meta/default/scatter",
        "/parameters/restart/bias/meta/default/hills",
        "/parameters/restart/references/restraint/default/coordinate",
        "/parameters/restart/protocol_sidecars/cv_in_file",
        "/parameters/restart/protocol_sidecars/SITS_in_file",
        "/parameters/restart/protocol_sidecars/meta_potential_in_file",
    },
    "trajectory.spg.h5md": {
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
        "/particles/all/step",
        "/particles/all/time",
    },
    "topology.spgt.h5": {
        "/atoms/mass",
        "/atoms/charge",
        "/atoms/residue_index",
        "/residues/atom_offset",
        "/forcefield/bond",
        "/forcefield/custom_force/pairwise",
        "/forcefield/custom_force/pairwise/data/custom_pair",
        "/forcefield/custom_force/listed",
        "/forcefield/custom_force/listed/data/custom_bond",
        "/manybody/reaxff/parameters",
        "/manybody/reaxff/type",
    },
    "protocol.spgp.h5": {
        "/cv",
        "/cv/config/section/name",
        "/constraint/default/pairs/atoms",
        "/constraint/default/pairs/r0",
        "/sits",
        "/sits/atom_indices",
        "/restraint/config/section/name",
        "/restraint/cv/config/section/name",
        "/restraint/default/atom_indices",
        "/restraint/default/weight",
        "/meta/default/grid",
        "/wall/soft/potential",
        "/steer/config/section/name",
    },
}


def _xponge_python() -> Path:
    configured = os.environ.get("SPONGE_XPONGE_PYTHON")
    if configured:
        return Path(configured)
    return Path(sys.executable)


def _xponge_env() -> dict[str, str]:
    env = dict(os.environ)
    configured_root = os.environ.get("SPONGE_XPONGE_ROOT")
    xponge_root = Path(configured_root) if configured_root else XPONGE_DEV_ROOT
    if xponge_root.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(xponge_root)
            if not existing
            else str(xponge_root) + os.pathsep + existing
        )
    return env


def _run(cmd, *, cwd=REPO_ROOT, env=None):
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "command failed with code "
            f"{result.returncode}: {' '.join(str(part) for part in cmd)}\n"
            f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
        )
    return result


def _require_converter():
    python = _xponge_python()
    env = _xponge_env()
    try:
        _run(
            [python, "-m", "Xponge", "legacy-to-bundle", "--help"],
            env=env,
        )
    except AssertionError as err:
        raise AssertionError(
            "Xponge legacy-to-bundle converter is required. Set "
            "SPONGE_XPONGE_ROOT or SPONGE_XPONGE_PYTHON to the XPONGE "
            "development tree that provides Xponge.io_bundle."
        ) from err
    return python, env


def _convert_legacy_case(case_root: Path, output_dir: Path) -> Path:
    python, env = _require_converter()
    _run(
        [
            python,
            "-m",
            "Xponge",
            "legacy-to-bundle",
            case_root,
            "-o",
            output_dir,
            "-m",
            "mdin.spg.toml",
        ],
        env=env,
    )
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    assert (output_dir / "bundle" / "mdin.bundled.spg.toml").exists()
    return manifest_path


def _manifest_ids(manifest_path: Path) -> set[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["contract_id"] for entry in data["entries"]}


def _h5_paths(path: Path) -> set[str]:
    h5dump = shutil.which("h5dump")
    assert h5dump is not None, "h5dump is required for bundled I/O smoke tests"
    result = _run([h5dump, "-n", path])
    paths = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] in {"group", "dataset"}:
            paths.add(fields[1])
    return paths


def test_xponge_converter_core_normal_input_contract(tmp_path):
    manifest_path = _convert_legacy_case(
        FIXTURE_ROOT / "core_structural" / "legacy_input",
        tmp_path / "core_structural",
    )

    ids = _manifest_ids(manifest_path)
    missing = sorted(CORE_NORMAL_IDS - ids)
    assert not missing

    mdin = (
        tmp_path / "core_structural" / "bundle" / "mdin.bundled.spg.toml"
    ).read_text(encoding="utf-8")
    assert 'input_h5_topology_path = "topology.spgt.h5"' in mdin
    assert 'input_h5_protocol_path = "protocol.spgp.h5"' in mdin
    assert 'input_h5_restart_path = "restart.spgr.h5"' in mdin


def test_xponge_converter_full_contract_manifest(tmp_path):
    manifest_path = _convert_legacy_case(
        FIXTURE_ROOT / "full_contract_rerun" / "legacy_input",
        tmp_path / "full_contract_rerun",
    )

    ids = _manifest_ids(manifest_path)
    missing = sorted(FULL_CONTRACT_IDS - ids)
    assert not missing

    mdin = (
        tmp_path / "full_contract_rerun" / "bundle" / "mdin.bundled.spg.toml"
    ).read_text(encoding="utf-8")
    for expected in (
        'mode = "rerun"',
        'input_h5_topology_path = "topology.spgt.h5"',
        'input_h5_protocol_path = "protocol.spgp.h5"',
        'input_h5_restart_path = "restart.spgr.h5"',
        'input_h5_restart_load = "full"',
        'input_h5_trajectory_path = "trajectory.spg.h5md"',
        'input_h5_trajectory_particle_stream = "all"',
        'output_h5_trajectory_path = "prod.spg.h5md"',
        "output_h5_trajectory_vds = true",
        'output_h5_restart_path = "prod.spgr.h5"',
        'output_h5_observable_path = "prod.obs.spg.h5md"',
    ):
        assert expected in mdin


def test_xponge_converter_full_contract_h5_paths(tmp_path):
    _convert_legacy_case(
        FIXTURE_ROOT / "full_contract_rerun" / "legacy_input",
        tmp_path / "full_contract_rerun",
    )
    bundle_dir = tmp_path / "full_contract_rerun" / "bundle"

    for file_name, required_paths in REQUIRED_FULL_BUNDLE_PATHS.items():
        actual_paths = _h5_paths(bundle_dir / file_name)
        missing = sorted(required_paths - actual_paths)
        assert not missing, f"{file_name} missing paths: {missing}"
