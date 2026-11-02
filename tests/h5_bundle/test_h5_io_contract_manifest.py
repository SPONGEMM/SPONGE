#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REQUIRED_ENTRIES = {
    "restart.coordinate": "converted",
    "restart.velocity": "converted",
    "restart.box": "converted",
    "trajectory.crd": "typed_converted",
    "trajectory.box": "typed_converted",
    "trajectory.vel": "typed_converted",
    "topology.mass": "typed_converted",
    "topology.charge": "typed_converted",
    "topology.residue": "typed_converted",
    "topology.bond": "typed_converted",
    "topology.angle": "typed_converted",
    "topology.dihedral": "typed_converted",
    "topology.LJ": "typed_converted",
    "topology.nb14_extra": "typed_converted",
    "topology.urey_bradley": "typed_converted",
    "topology.cmap": "typed_converted",
    "topology.gb": "typed_converted",
    "topology.exclude": "typed_converted",
    "topology.virtual_atom": "typed_converted",
    "topology.LJ_soft_core": "typed_converted",
    "topology.subsys_division": "typed_converted",
    "topology.EAM": "typed_converted",
    "topology.EAM_atom_type": "typed_converted",
    "topology.SW": "typed_converted",
    "topology.EDIP": "typed_converted",
    "topology.TERSOFF": "typed_converted",
    "topology.REAXFF": "typed_converted",
    "topology.REAXFF_type": "typed_converted",
    "protocol.cv": "typed_converted",
    "protocol.constrain": "typed_converted",
    "protocol.restrain": "typed_converted",
    "protocol.SITS": "typed_converted",
    "protocol.SITS_atom": "typed_converted",
    "protocol.restrain_atom_id": "typed_converted",
    "protocol.restrain_weight": "typed_converted",
    "protocol.restrain_cv": "typed_converted",
    "protocol.meta_edge": "typed_converted",
    "protocol.soft_walls": "typed_converted",
    "protocol.steer_cv": "typed_converted",
    "restart.SITS_nk": "typed_converted",
    "restart.restrain_coordinate": "typed_converted",
    "restart.meta_potential": "typed_converted",
    "restart.meta_scatter": "typed_converted",
    "restart.hills": "typed_converted",
    "restart.nose_hoover_chain_restart_input": "typed_converted",
    "topology.pairwise_force": "typed_converted",
    "topology.listed_forces": "typed_converted",
    "restart.protocol_sidecar.cv_in_file": "sidecar_embedded",
    "restart.protocol_sidecar.SITS_in_file": "sidecar_embedded",
    "restart.protocol_sidecar.meta_potential_in_file": "sidecar_embedded",
    "topology.pairwise_force_data.custom_pair": "typed_converted",
    "topology.listed_force_data.custom_bond": "typed_converted",
    "output.h5.output_h5_trajectory_path": "output_plan_preserved",
    "output.h5.output_h5_trajectory_vds": "output_plan_preserved",
    "output.h5.output_h5_trajectory_chunk_size": "output_plan_preserved",
    "output.h5.output_h5_trajectory_repair_policy": "output_plan_preserved",
    "output.h5.output_h5_restart_path": "output_plan_preserved",
    "output.h5.output_h5_restart_topology_hash": "output_plan_preserved",
    "output.h5.output_h5_restart_atom_order_hash": "output_plan_preserved",
    "output.h5.output_h5_restart_protocol_hash": "output_plan_preserved",
    "output.h5.output_h5_observable_path": "output_plan_preserved",
    "output.legacy_sidecar.mdout": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.mdinfo": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.crd": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.box": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.vel": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.frc": "legacy_output_sidecar_preserved",
    "output.legacy_sidecar.rst": "legacy_output_sidecar_preserved",
    "run_mdin.rerun_start": "preserved_in_mdin",
    "run_mdin.rerun_strip": "preserved_in_mdin",
    "run_mdin.rerun_frame_limit": "preserved_in_mdin",
    "run_mdin.rerun_need_box_update": "preserved_in_mdin",
    "run_mdin.input_h5_trajectory_particle_stream": "preserved_in_mdin",
}

SEMANTIC_EQUIVALENCE_EVIDENCE = {
    "restart.coordinate": [
        "compare_restart_structural_to_legacy",
        "coordinate.txt",
        "/particles/all/position/value",
    ],
    "restart.velocity": [
        "compare_restart_structural_to_legacy",
        "velocity.txt",
        "/particles/all/velocity/value",
    ],
    "restart.box": [
        "compare_restart_structural_to_legacy",
        "coordinate.txt",
        "/particles/all/box/edges/value",
    ],
    "trajectory.crd": [
        "compare_trajectory_to_legacy",
        "traj.dat",
        "/particles/all/position/value",
    ],
    "trajectory.box": [
        "compare_trajectory_to_legacy",
        "traj_box.dat",
        "/particles/all/box/edges/value",
    ],
    "trajectory.vel": [
        "compare_trajectory_to_legacy",
        "traj_vel.dat",
        "/particles/all/velocity/value",
    ],
    "topology.mass": ["compare_group_mass_charge", "mass.txt", "/atoms/mass"],
    "topology.charge": [
        "compare_group_mass_charge",
        "charge.txt",
        "/atoms/charge",
    ],
    "topology.residue": [
        "compare_residue_to_h5",
        "residue.txt",
        "/atoms/residue_index",
    ],
    "topology.bond": [
        "compare_bonds_to_h5",
        "bond.txt",
        "/forcefield/bond/atoms",
    ],
    "topology.angle": [
        "compare_angles_to_h5",
        "angle.txt",
        "/forcefield/angle/atoms",
    ],
    "topology.dihedral": [
        "compare_dihedrals_to_h5",
        "dihedral.txt",
        "/forcefield/dihedral/atoms",
    ],
    "topology.LJ": ["compare_lj_to_h5", "lj.txt", "/forcefield/lj/type"],
    "topology.nb14_extra": [
        "compare_nb14_extra_to_h5",
        "nb14_extra.txt",
        "/forcefield/nb14_extra/atoms",
    ],
    "topology.urey_bradley": [
        "compare_urey_bradley_to_h5",
        "urey_bradley.txt",
        "/forcefield/urey_bradley/atoms",
    ],
    "topology.cmap": [
        "compare_cmap_to_h5",
        "cmap.txt",
        "/forcefield/cmap/atoms",
    ],
    "topology.gb": ["compare_gb_to_h5", "gb.txt", "/forcefield/gb/params"],
    "topology.exclude": [
        "compare_exclusions_to_h5",
        "exclude.txt",
        "/topology/exclusions/list",
    ],
    "topology.virtual_atom": [
        "compare_virtual_atoms_to_h5",
        "virtual_atom.txt",
        "/forcefield/virtual_atom/parameter",
    ],
    "topology.LJ_soft_core": [
        "compare_lj_soft_core_to_h5",
        "lj_soft_core.txt",
        "pair_AA",
    ],
    "topology.subsys_division": [
        "compare_lj_soft_core_to_h5",
        "subsys_division.txt",
        "/forcefield/subsys_division",
    ],
    "topology.EAM": ["compare_eam_to_h5", "eam.txt", "/manybody/eam"],
    "topology.EAM_atom_type": [
        "compare_eam_to_h5",
        "eam_atom_type.txt",
        'root + "/atom_type"',
    ],
    "topology.SW": [
        "compare_manybody_pair_triple_to_h5",
        "sw.txt",
        "/manybody/sw",
    ],
    "topology.EDIP": [
        "compare_manybody_pair_triple_to_h5",
        "edip.txt",
        "/manybody/edip",
    ],
    "topology.TERSOFF": [
        "compare_tersoff_to_h5",
        "tersoff.txt",
        "/manybody/tersoff",
    ],
    "topology.REAXFF": [
        "compare_reaxff_to_h5",
        "reaxff.txt",
        "/manybody/reaxff/parameters",
    ],
    "topology.REAXFF_type": [
        "compare_reaxff_to_h5",
        "reaxff_type.txt",
        "/manybody/reaxff/type",
    ],
    "protocol.cv": ["compare_config_sections_to_h5", "cv.txt", "/cv/config"],
    "protocol.constrain": [
        "compare_protocol_constraints_to_h5",
        "constrain.txt",
        "/constraint/default/pairs",
    ],
    "protocol.restrain": [
        "compare_config_sections_to_h5",
        "restrain.txt",
        "/restraint/config",
    ],
    "protocol.SITS": [
        "compare_config_sections_to_h5",
        "sits.txt",
        "/sits/config",
    ],
    "protocol.SITS_atom": [
        "compare_protocol_sits_atom_to_h5",
        "sits_atom.txt",
        "/sits/atom_indices",
    ],
    "protocol.restrain_atom_id": [
        "compare_protocol_restraint_default_to_h5",
        "restrain_atom_id.txt",
        "/restraint/default/atom_indices",
    ],
    "protocol.restrain_weight": [
        "compare_protocol_restraint_default_to_h5",
        "restrain_weight.txt",
        "/restraint/default/weight",
    ],
    "protocol.restrain_cv": [
        "compare_config_sections_to_h5",
        "restrain_cv.txt",
        "/restraint/cv/config",
    ],
    "protocol.meta_edge": [
        "compare_protocol_meta_edge_to_h5",
        "meta_edge.txt",
        "/meta/default/grid",
    ],
    "protocol.soft_walls": [
        "compare_soft_walls_to_h5",
        "soft_walls.txt",
        "/wall/soft",
    ],
    "protocol.steer_cv": [
        "compare_config_sections_to_h5",
        "steer_cv.txt",
        "/steer/config",
    ],
    "restart.SITS_nk": [
        "compare_restart_sits_nk_to_h5",
        "sits_nk.txt",
        "/parameters/restart/bias/sits/SITS/nk",
    ],
    "restart.restrain_coordinate": [
        "compare_restart_restraint_reference_to_h5",
        "restrain_coordinate.txt",
        "/parameters/restart/references/restraint/default/coordinate",
    ],
    "restart.meta_potential": [
        "compare_restart_meta_potential_to_h5",
        "meta_potential.txt",
        "/parameters/restart/bias/meta/default/potential",
    ],
    "restart.meta_scatter": [
        "compare_restart_meta_scatter_to_h5",
        "meta_scatter.txt",
        "/parameters/restart/bias/meta/default/scatter",
    ],
    "restart.hills": [
        "compare_restart_hills_to_h5",
        "hills.txt",
        "/parameters/restart/bias/meta/default/hills",
    ],
    "restart.nose_hoover_chain_restart_input": [
        "compare_restart_nhc_to_h5",
        "nhc_restart.txt",
        "/parameters/restart/thermostat/nose_hoover_chain",
    ],
    "topology.pairwise_force": [
        "compare_custom_pair_to_h5",
        "pairwise_force.txt",
        "/forcefield/custom_force/pairwise",
    ],
    "topology.listed_forces": [
        "compare_custom_bond_to_h5",
        "listed_forces.txt",
        "/forcefield/custom_force/listed",
    ],
    "topology.pairwise_force_data.custom_pair": [
        "compare_custom_pair_to_h5",
        "custom_pair.txt",
        'data_root + "/atom_type"',
    ],
    "topology.listed_force_data.custom_bond": [
        "compare_custom_bond_to_h5",
        "custom_bond.txt",
        'data_root + "/parameter/value"',
    ],
}

REQUIRED_PROTOCOL_SIDECARS = {
    "restart.protocol_sidecar.cv_in_file",
    "restart.protocol_sidecar.constrain_in_file",
    "restart.protocol_sidecar.restrain_in_file",
    "restart.protocol_sidecar.pairwise_force_in_file",
    "restart.protocol_sidecar.listed_forces_in_file",
    "restart.protocol_sidecar.soft_walls_in_file",
    "restart.protocol_sidecar.SITS_in_file",
    "restart.protocol_sidecar.SITS_atom_in_file",
    "restart.protocol_sidecar.SITS_nk_in_file",
    "restart.protocol_sidecar.restrain_atom_id",
    "restart.protocol_sidecar.restrain_coordinate_in_file",
    "restart.protocol_sidecar.restrain_weight_in_file",
    "restart.protocol_sidecar.meta_edge_in_file",
    "restart.protocol_sidecar.meta_potential_in_file",
    "restart.protocol_sidecar.meta_scatter_in_file",
    "restart.protocol_sidecar.restrain_cv_in_file",
    "restart.protocol_sidecar.steer_cv_in_file",
}

REQUIRED_H5_OUTPUT_KEYS = {
    "output.h5.output_h5_trajectory_path",
    "output.h5.output_h5_trajectory_vds",
    "output.h5.output_h5_trajectory_chunk_size",
    "output.h5.output_h5_trajectory_repair_policy",
    "output.h5.output_h5_restart_path",
    "output.h5.output_h5_restart_topology_hash",
    "output.h5.output_h5_restart_atom_order_hash",
    "output.h5.output_h5_restart_protocol_hash",
    "output.h5.output_h5_observable_path",
}

REQUIRED_LEGACY_OUTPUT_SIDECARS = {
    "output.legacy_sidecar.mdout",
    "output.legacy_sidecar.mdinfo",
    "output.legacy_sidecar.crd",
    "output.legacy_sidecar.box",
    "output.legacy_sidecar.vel",
    "output.legacy_sidecar.frc",
    "output.legacy_sidecar.rst",
}

REQUIRED_FULL_RUN_POLICY_KEYS = {
    "run_mdin.mode",
    "run_mdin.step_limit",
    "run_mdin.rerun_start",
    "run_mdin.rerun_strip",
    "run_mdin.rerun_frame_limit",
    "run_mdin.rerun_need_box_update",
    "run_mdin.input_h5_trajectory_particle_stream",
}

CORE_REQUIRED_ENTRIES = {
    "restart.coordinate": "converted",
    "restart.velocity": "converted",
    "restart.box": "converted",
    "topology.mass": "typed_converted",
    "topology.charge": "typed_converted",
    "protocol.cv": "typed_converted",
    "protocol.restrain": "typed_converted",
    "protocol.SITS": "typed_converted",
    "restart.SITS_nk": "typed_converted",
    "restart.protocol_sidecar.cv_in_file": "sidecar_embedded",
    "restart.protocol_sidecar.restrain_in_file": "sidecar_embedded",
    "restart.protocol_sidecar.SITS_in_file": "sidecar_embedded",
    "restart.protocol_sidecar.SITS_nk_in_file": "sidecar_embedded",
    "run_mdin.mode": "preserved_in_mdin",
    "run_mdin.dt": "preserved_in_mdin",
    "run_mdin.step_limit": "preserved_in_mdin",
    "run_mdin.target_temperature": "preserved_in_mdin",
    "run_mdin.write_trajectory_interval": "preserved_in_mdin",
    "run_mdin.write_mdout_interval": "preserved_in_mdin",
    "run_mdin.write_restart_file_interval": "preserved_in_mdin",
    "run_mdin.print_zeroth_frame": "preserved_in_mdin",
    "run_mdin.thermostat": "preserved_in_mdin",
    "output.legacy_sidecar.mdout": "legacy_output_sidecar_preserved",
}

CORE_PROTOCOL_SIDECARS = {
    "restart.protocol_sidecar.cv_in_file",
    "restart.protocol_sidecar.restrain_in_file",
    "restart.protocol_sidecar.SITS_in_file",
    "restart.protocol_sidecar.SITS_nk_in_file",
}

EXPECTED_ENTRY_FIELDS = {
    "contract_id",
    "status",
    "component",
    "direction",
    "payload_kind",
    "override_policy",
    "bundle_file",
    "bundle_path",
}

SIDECAR_ENTRY_FIELDS = {
    "sidecar_key",
    "sidecar_path",
}

TOP_LEVEL_INPUT_FILE_KEYS = {
    "coordinate_in_file",
    "velocity_in_file",
    "rst7",
    "crd",
    "box",
    "vel",
    "restrain_atom_id",
    "nose_hoover_chain_restart_input",
}

SECTION_INPUT_FILE_KEY_ALIASES = {
    ("EAM", "in_file"): "EAM_in_file",
    ("EAM", "atom_type_in_file"): "EAM_atom_type_in_file",
    ("TERSOFF", "in_file"): "TERSOFF_in_file",
    ("REAXFF", "in_file"): "REAXFF_in_file",
    ("REAXFF", "type_in_file"): "REAXFF_type_in_file",
}

LEGACY_SIDECAR_KEYS_PATH = "/parameters/sponge/files/legacy_sidecars/key"
LEGACY_SIDECAR_PATHS_PATH = "/parameters/sponge/files/legacy_sidecars/path"


def fail(message):
    raise AssertionError(message)


def require_file(path, label):
    if not path.is_file():
        fail(f"missing {label}: {path}")


def require_dir(path, label):
    if not path.is_dir():
        fail(f"missing {label}: {path}")


def require_no_path(path, label):
    if path.exists():
        fail(f"unexpected {label}: {path}")


def parse_flat_mdin_values(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line or line.startswith("["):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_mdin_toml(path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def collect_mdin_input_file_refs(path):
    data = load_mdin_toml(path)
    refs = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                source_key = SECTION_INPUT_FILE_KEY_ALIASES.get(
                    (key, child_key)
                )
                if source_key and isinstance(child_value, str):
                    refs[source_key] = child_value
            continue
        if not isinstance(value, str):
            continue
        if key in TOP_LEVEL_INPUT_FILE_KEYS or key.endswith("_in_file"):
            refs[key] = value
    return refs


def require_mdin_input_refs_exist(mdin_path, root, label):
    refs = collect_mdin_input_file_refs(mdin_path)
    for key, value in refs.items():
        path = Path(value)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            fail(f"{label} mdin input reference is missing: {key}={value!r}")
    return refs


def require_no_legacy_input_refs(mdin_path, label):
    refs = collect_mdin_input_file_refs(mdin_path)
    if refs:
        fail(
            f"{label} should not retain legacy input file refs: {sorted(refs)}"
        )


def require_sidecar_mdin_refs_are_materialized(mdin_path, bundle_root, label):
    refs = collect_mdin_input_file_refs(mdin_path)
    for key, value in refs.items():
        path = Path(value)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "legacy_sidecars"
        ):
            fail(
                f"{label} sidecar mdin input ref must stay below "
                f"legacy_sidecars/: {key}={value!r}"
            )
        if not (bundle_root / path).is_file():
            fail(f"{label} sidecar mdin input ref is missing: {key}={value!r}")
    return refs


def require_manifest_input_sources_match_legacy_mdin(entries, legacy_refs):
    for entry in entries:
        if entry.get("direction") != "input":
            continue
        if entry.get("payload_kind") != "file":
            continue
        contract_id = entry["contract_id"]
        source_key = entry.get("source_key")
        source_path = entry.get("source_path")
        if not source_key:
            fail(f"{contract_id} input file entry lacks source_key")
        if not source_path:
            fail(f"{contract_id} input file entry lacks source_path")
        if source_key not in legacy_refs:
            fail(
                f"{contract_id} source_key is not declared by legacy "
                f"mdin input refs: {source_key}"
            )
        expected_source_name = Path(legacy_refs[source_key]).name
        actual_source_name = Path(source_path).name
        if actual_source_name != expected_source_name:
            fail(
                f"{contract_id} source_path does not match source_key "
                f"{source_key}: actual={actual_source_name!r} "
                f"expected={expected_source_name!r}"
            )


def h5dump_string_dataset(h5dump, h5_path, dataset_path):
    result = subprocess.run(
        [str(h5dump), "-d", dataset_path, str(h5_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if "unable to get link info" in result.stderr:
            return None
        fail(
            f"h5dump failed for {h5_path.name}:{dataset_path}: "
            f"{result.stderr.strip()}"
        )

    match = re.search(r"DATA\s*\{(?P<data>.*?)\n\s*\}", result.stdout, re.S)
    if not match:
        fail(
            f"h5dump output lacks DATA block for {h5_path.name}:{dataset_path}"
        )
    return [
        json.loads(f'"{text}"')
        for text in re.findall(r'"((?:[^"\\]|\\.)*)"', match.group("data"))
    ]


def read_h5_legacy_sidecar_table(h5dump, h5_path):
    keys = h5dump_string_dataset(h5dump, h5_path, LEGACY_SIDECAR_KEYS_PATH)
    paths = h5dump_string_dataset(h5dump, h5_path, LEGACY_SIDECAR_PATHS_PATH)
    if keys is None and paths is None:
        return []
    if keys is None or paths is None:
        fail(
            f"{h5_path.name} legacy sidecar key/path datasets must be "
            "present together"
        )

    if len(keys) != len(paths):
        fail(f"{h5_path.name} legacy sidecar key/path length mismatch")

    pairs = []
    seen_keys = set()
    for key, raw_path in zip(keys, paths):
        if not key:
            fail(f"{h5_path.name} legacy sidecar key must not be empty")
        if not raw_path:
            fail(f"{h5_path.name} legacy sidecar path must not be empty")
        if key in seen_keys:
            fail(f"{h5_path.name} duplicate legacy sidecar key: {key}")
        seen_keys.add(key)
        pairs.append((key, raw_path))
    return pairs


def resolve_h5_container_relative_path(h5_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return h5_path.parent / path


def require_manifest_lists_h5_sidecar_tables(
    entries, bundle_root, legacy_root, h5dump
):
    input_file_entries = [
        entry
        for entry in entries
        if entry.get("direction") == "input"
        and entry.get("payload_kind") == "file"
    ]

    h5_files = sorted(
        path
        for path in bundle_root.iterdir()
        if path.is_file() and path.name.endswith((".h5", ".h5md"))
    )
    for h5_path in h5_files:
        for source_key, raw_sidecar_path in read_h5_legacy_sidecar_table(
            h5dump, h5_path
        ):
            raw_path = Path(raw_sidecar_path)
            if raw_path.is_absolute() or not raw_path.parts:
                fail(
                    f"{h5_path.name} sidecar path must be bundle-relative: "
                    f"{source_key}={raw_sidecar_path!r}"
                )
            if raw_path.parts[0] != "legacy_sidecars":
                fail(
                    f"{h5_path.name} sidecar path must stay below "
                    f"legacy_sidecars/: {source_key}={raw_sidecar_path!r}"
                )
            if len(raw_path.parts) != 3 or raw_path.parts[1] != source_key:
                fail(
                    f"{h5_path.name} sidecar path must be "
                    f"legacy_sidecars/<key>/<basename>: "
                    f"{source_key}={raw_sidecar_path!r}"
                )

            materialized = resolve_h5_container_relative_path(
                h5_path, raw_sidecar_path
            )
            if not materialized.is_file():
                fail(
                    f"{h5_path.name} sidecar table points at missing file: "
                    f"{source_key}={raw_sidecar_path!r}"
                )

            legacy_source = legacy_root / raw_path.name
            if not legacy_source.is_file():
                fail(
                    f"{h5_path.name} sidecar source is missing in legacy input: "
                    f"{source_key}={legacy_source}"
                )

            matches = [
                entry
                for entry in input_file_entries
                if entry.get("bundle_file") == h5_path.name
                and entry.get("source_key") == source_key
                and Path(entry.get("source_path", "")).name == raw_path.name
            ]
            if not matches:
                fail(
                    "manifest does not list H5 legacy sidecar table entry: "
                    f"bundle_file={h5_path.name} source_key={source_key} "
                    f"source_path={raw_path.name}"
                )
            for entry in matches:
                missing_sidecar_fields = SIDECAR_ENTRY_FIELDS - set(entry)
                if missing_sidecar_fields:
                    fail(
                        f"{entry['contract_id']} sidecar table manifest "
                        f"lacks fields {sorted(missing_sidecar_fields)}"
                    )
                if entry.get("sidecar_key") != source_key:
                    fail(
                        f"{entry['contract_id']} sidecar_key mismatch: "
                        f"actual={entry.get('sidecar_key')!r} "
                        f"expected={source_key!r}"
                    )
                if entry.get("sidecar_path") != raw_sidecar_path:
                    fail(
                        f"{entry['contract_id']} sidecar_path mismatch: "
                        f"actual={entry.get('sidecar_path')!r} "
                        f"expected={raw_sidecar_path!r}"
                    )
                if not entry.get("bundle_path", "").startswith("/"):
                    fail(
                        f"{entry['contract_id']} sidecar table manifest "
                        "bundle_path is not absolute inside HDF5"
                    )
                if not entry.get("override_policy"):
                    fail(
                        f"{entry['contract_id']} sidecar table manifest "
                        "override_policy is empty"
                    )


def require_manifest_lists_materialized_sidecar_files(entries, bundle_root):
    sidecar_root = bundle_root / "legacy_sidecars"
    by_sidecar = {}
    for entry in entries:
        sidecar_key = entry.get("sidecar_key")
        sidecar_path = entry.get("sidecar_path")
        if sidecar_key or sidecar_path:
            if not sidecar_key or not sidecar_path:
                fail(f"{entry['contract_id']} has incomplete sidecar fields")
            if Path(sidecar_path).is_absolute():
                fail(f"{entry['contract_id']} sidecar_path must be relative")
            raw_path = Path(sidecar_path)
            if (
                len(raw_path.parts) != 3
                or raw_path.parts[0] != "legacy_sidecars"
            ):
                fail(
                    f"{entry['contract_id']} sidecar_path must be "
                    f"legacy_sidecars/<key>/<basename>: {sidecar_path!r}"
                )
            if raw_path.parts[1] != sidecar_key:
                fail(
                    f"{entry['contract_id']} sidecar_key/path mismatch: "
                    f"key={sidecar_key!r} path={sidecar_path!r}"
                )
            if entry.get("source_key") != sidecar_key:
                fail(
                    f"{entry['contract_id']} source_key does not match "
                    f"sidecar_key: {entry.get('source_key')!r} vs {sidecar_key!r}"
                )
            if Path(entry.get("source_path", "")).name != raw_path.name:
                fail(
                    f"{entry['contract_id']} source_path does not match "
                    f"sidecar_path basename: {entry.get('source_path')!r} vs "
                    f"{sidecar_path!r}"
                )
            if not (bundle_root / raw_path).is_file():
                fail(
                    f"{entry['contract_id']} sidecar_path points at missing file: "
                    f"{sidecar_path!r}"
                )
            by_sidecar.setdefault(sidecar_path, []).append(entry["contract_id"])

    materialized_files = sorted(
        path for path in sidecar_root.rglob("*") if path.is_file()
    )
    if not materialized_files:
        fail(f"legacy_sidecars contains no materialized files: {sidecar_root}")
    for sidecar_file in materialized_files:
        sidecar_path = sidecar_file.relative_to(bundle_root).as_posix()
        if sidecar_path not in by_sidecar:
            fail(
                "materialized sidecar file is not listed by manifest "
                f"sidecar_path: {sidecar_path}"
            )


def require_sidecar_mdin_does_not_spell_h5_table_keys(bundle_root, h5dump):
    table_keys = set()
    h5_files = sorted(
        path
        for path in bundle_root.iterdir()
        if path.is_file() and path.name.endswith((".h5", ".h5md"))
    )
    for h5_path in h5_files:
        for source_key, _ in read_h5_legacy_sidecar_table(h5dump, h5_path):
            table_keys.add(source_key)

    refs = collect_mdin_input_file_refs(bundle_root / "mdin.bundled.spg.toml")
    explicit_table_keys = sorted(set(refs) & table_keys)
    if explicit_table_keys:
        fail(
            "sidecar-preserving bundled mdin must not explicitly spell H5 "
            "sidecar-table keys: "
            f"{explicit_table_keys}"
        )


def existing_source_path(entry, legacy_root):
    source_path = entry.get("source_path")
    if not source_path:
        return True
    path = Path(source_path)
    relocated = legacy_root / path.name
    return relocated.exists()


def require_manifest_path_relocates(value, expected_path, label):
    if not value:
        fail(f"manifest missing top-level {label}")
    raw_path = Path(value)
    if (
        raw_path.resolve() != expected_path.resolve()
        and raw_path.name != expected_path.name
    ):
        fail(
            f"manifest {label} cannot be relocated by basename: "
            f"actual={raw_path} expected={expected_path}"
        )
    if not expected_path.exists():
        fail(f"relocated manifest {label} is missing: {expected_path}")


def load_manifest(fixture_root, case_name):
    case_root = fixture_root / case_name / "bundled_input_with_legacy_sidecar"
    bundle_root = case_root / "bundle"
    legacy_root = fixture_root / case_name / "legacy_input"
    manifest_path = case_root / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema") != "xponge.legacy_to_bundle.manifest":
        fail(
            f"{case_name} manifest schema mismatch: {manifest.get('schema')!r}"
        )
    if manifest.get("schema_version") != 1:
        fail(
            f"{case_name} manifest schema_version mismatch: "
            f"{manifest.get('schema_version')!r}"
        )

    require_manifest_path_relocates(
        manifest.get("bundled_mdin"),
        bundle_root / "mdin.bundled.spg.toml",
        "bundled_mdin",
    )
    require_manifest_path_relocates(
        manifest.get("case_root"), legacy_root, "case_root"
    )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("manifest entries must be a non-empty list")

    by_contract = {}
    for entry in entries:
        missing_fields = EXPECTED_ENTRY_FIELDS - set(entry)
        if missing_fields:
            fail(
                f"manifest entry lacks required fields "
                f"{sorted(missing_fields)}: {entry!r}"
            )
        contract_id = entry.get("contract_id")
        if not contract_id:
            fail(f"manifest entry lacks contract_id: {entry!r}")
        if contract_id in by_contract:
            fail(f"duplicate manifest contract_id: {contract_id}")
        by_contract[contract_id] = entry

        bundle_file = entry.get("bundle_file")
        if bundle_file and bundle_file.endswith((".h5", ".h5md")):
            bundle_path = bundle_root / bundle_file
            if not bundle_path.exists():
                fail(f"bundle file for {contract_id} is missing: {bundle_file}")
        if not existing_source_path(entry, legacy_root):
            fail(f"source path for {contract_id} is missing after relocation")

    return manifest, by_contract, bundle_root, legacy_root


def require_fixture_group_structure(fixture_root, case_name, has_trajectory):
    case_root = fixture_root / case_name
    legacy_root = case_root / "legacy_input"
    bundled_root = case_root / "bundled_input" / "bundle"
    sidecar_case_root = case_root / "bundled_input_with_legacy_sidecar"
    sidecar_root = sidecar_case_root / "bundle"

    require_dir(case_root, f"{case_name} fixture group")
    require_dir(legacy_root, f"{case_name} legacy_input")
    require_dir(bundled_root, f"{case_name} bundled_input bundle")
    require_dir(sidecar_root, f"{case_name} sidecar bundle")

    require_file(legacy_root / "mdin.spg.toml", f"{case_name} legacy mdin")
    require_file(
        bundled_root / "mdin.bundled.spg.toml", f"{case_name} bundled mdin"
    )
    require_file(
        sidecar_root / "mdin.bundled.spg.toml", f"{case_name} sidecar mdin"
    )
    require_file(
        sidecar_case_root / "manifest.json", f"{case_name} sidecar manifest"
    )

    for bundle in (bundled_root, sidecar_root):
        require_file(bundle / "topology.spgt.h5", f"{case_name} topology H5")
        require_file(bundle / "protocol.spgp.h5", f"{case_name} protocol H5")
        require_file(bundle / "restart.spgr.h5", f"{case_name} restart H5")
        if has_trajectory:
            require_file(
                bundle / "trajectory.spg.h5md", f"{case_name} trajectory H5"
            )
        else:
            require_no_path(
                bundle / "trajectory.spg.h5md",
                f"{case_name} normal-mode trajectory H5",
            )

    require_no_path(
        bundled_root / "legacy_sidecars",
        f"{case_name} pure bundled legacy_sidecars",
    )
    require_dir(
        sidecar_root / "legacy_sidecars",
        f"{case_name} sidecar materialization root",
    )
    legacy_refs = require_mdin_input_refs_exist(
        legacy_root / "mdin.spg.toml", legacy_root, f"{case_name} legacy_input"
    )
    if not legacy_refs:
        fail(f"{case_name} legacy_input mdin declares no input file refs")
    require_no_legacy_input_refs(
        bundled_root / "mdin.bundled.spg.toml", f"{case_name} bundled_input"
    )
    require_sidecar_mdin_refs_are_materialized(
        sidecar_root / "mdin.bundled.spg.toml",
        sidecar_root,
        f"{case_name} bundled_input_with_legacy_sidecar",
    )

    if case_name == "core_structural":
        require_file(
            sidecar_root / "mdin.override_conflict.spg.toml",
            "core_structural override-conflict mdin",
        )
        require_file(
            sidecar_root / "mdin.override_same_path.spg.toml",
            "core_structural override same-path mdin",
        )
        require_sidecar_mdin_refs_are_materialized(
            sidecar_root / "mdin.override_same_path.spg.toml",
            sidecar_root,
            "core_structural override same-path mdin",
        )
    else:
        require_no_path(
            sidecar_root / "mdin.override_conflict.spg.toml",
            f"{case_name} override-conflict mdin",
        )
        require_no_path(
            sidecar_root / "mdin.override_same_path.spg.toml",
            f"{case_name} override same-path mdin",
        )


def require_entry_statuses(by_contract, required_entries):
    for contract_id, expected_status in required_entries.items():
        entry = by_contract.get(contract_id)
        if entry is None:
            fail(f"manifest missing contract_id: {contract_id}")
        actual_status = entry.get("status")
        if actual_status != expected_status:
            fail(
                f"{contract_id} status mismatch: "
                f"actual={actual_status!r} expected={expected_status!r}"
            )


def require_entry_semantics(entries):
    for entry in entries:
        contract_id = entry["contract_id"]
        if contract_id.startswith("restart.protocol_sidecar."):
            if entry.get("component") != "restart":
                fail(f"{contract_id} component is not restart")
            if entry.get("override_policy") != "legacy_protocol_sidecar":
                fail(f"{contract_id} override policy is not sidecar-specific")
            if entry.get("bundle_file") != "restart.spgr.h5":
                fail(f"{contract_id} is not stored in restart.spgr.h5")
        elif contract_id.startswith("output.h5."):
            if entry.get("component") != "output":
                fail(f"{contract_id} component is not output")
            if entry.get("direction") != "output":
                fail(f"{contract_id} direction is not output")
            if entry.get("payload_kind") not in {"path", "scalar"}:
                fail(f"{contract_id} payload kind is not output plan data")
        elif contract_id.startswith("output.legacy_sidecar."):
            if entry.get("component") != "output":
                fail(f"{contract_id} component is not output")
            if entry.get("direction") != "output":
                fail(f"{contract_id} direction is not output")
            if entry.get("override_policy") != "explicit":
                fail(f"{contract_id} override policy is not explicit")
        elif contract_id.startswith("run_mdin."):
            if entry.get("component") != "run_policy":
                fail(f"{contract_id} component is not run_policy")
            if entry.get("direction") != "input":
                fail(f"{contract_id} direction is not input")
            if entry.get("payload_kind") != "scalar":
                fail(f"{contract_id} payload kind is not scalar")
            if entry.get("override_policy") != "explicit":
                fail(f"{contract_id} override policy is not explicit")
        else:
            if entry.get("direction") != "input":
                fail(f"{contract_id} direction is not input")
            if not entry.get("bundle_path", "").startswith("/"):
                fail(f"{contract_id} bundle_path is not absolute inside HDF5")


def require_status_semantics(entries):
    for entry in entries:
        contract_id = entry["contract_id"]
        status = entry.get("status")
        direction = entry.get("direction")
        payload_kind = entry.get("payload_kind")
        override_policy = entry.get("override_policy")

        if status == "converted":
            if direction != "input":
                fail(f"{contract_id} converted entry is not input")
            if payload_kind != "file":
                fail(f"{contract_id} converted entry is not file payload")
            if override_policy != "forbidden":
                fail(
                    f"{contract_id} converted override policy is not forbidden"
                )
        elif status == "typed_converted":
            if direction != "input":
                fail(f"{contract_id} typed entry is not input")
            if payload_kind != "file":
                fail(f"{contract_id} typed entry is not file payload")
            if override_policy == "legacy_mdin_sidecar":
                if contract_id not in {
                    "topology.pairwise_force_data.custom_pair",
                    "topology.listed_force_data.custom_bond",
                }:
                    fail(f"{contract_id} unexpected mdin sidecar typed entry")
            elif override_policy not in {"allowed", "forbidden"}:
                fail(f"{contract_id} typed override policy is unexpected")
        elif status == "sidecar_embedded":
            if direction != "input":
                fail(f"{contract_id} sidecar entry is not input")
            if payload_kind != "file":
                fail(f"{contract_id} sidecar entry is not file payload")
            if override_policy != "legacy_protocol_sidecar":
                fail(f"{contract_id} sidecar override policy is unexpected")
        elif status == "preserved_in_mdin":
            if direction != "input":
                fail(f"{contract_id} mdin entry is not input")
            if payload_kind != "scalar":
                fail(f"{contract_id} mdin entry is not scalar")
            if override_policy != "explicit":
                fail(f"{contract_id} mdin override policy is not explicit")
        elif status == "output_plan_preserved":
            if direction != "output":
                fail(f"{contract_id} output H5 entry is not output")
            if payload_kind not in {"path", "scalar"}:
                fail(f"{contract_id} output H5 payload kind is unexpected")
            if override_policy != "explicit":
                fail(f"{contract_id} output H5 override policy is not explicit")
        elif status == "legacy_output_sidecar_preserved":
            if direction != "output":
                fail(f"{contract_id} legacy output sidecar is not output")
            if payload_kind != "path":
                fail(f"{contract_id} legacy output sidecar is not path payload")
            if override_policy != "explicit":
                fail(
                    f"{contract_id} legacy output sidecar policy is not explicit"
                )
        else:
            fail(f"{contract_id} has unknown manifest status: {status!r}")


def require_materialized_sidecars_match_legacy(bundle_root, legacy_root):
    sidecar_root = bundle_root / "legacy_sidecars"
    if not sidecar_root.is_dir():
        fail(f"missing legacy_sidecars directory: {sidecar_root}")
    sidecar_files = sorted(
        path for path in sidecar_root.rglob("*") if path.is_file()
    )
    if not sidecar_files:
        fail(f"legacy_sidecars contains no files: {sidecar_root}")
    for sidecar_file in sidecar_files:
        rel = sidecar_file.relative_to(sidecar_root)
        if len(rel.parts) != 2:
            fail(f"sidecar file is not stored as <key>/<basename>: {rel}")
        legacy_file = legacy_root / rel.name
        if not legacy_file.exists():
            fail(f"sidecar source missing for {rel}: {legacy_file}")
        if sidecar_file.read_bytes() != legacy_file.read_bytes():
            fail(f"sidecar file does not byte-match legacy source: {rel}")


def require_all_converted_inputs_are_declared(entries):
    declared = set(REQUIRED_ENTRIES)
    missing = sorted(
        entry["contract_id"]
        for entry in entries
        if entry.get("direction") == "input"
        and entry.get("status") in {"converted", "typed_converted"}
        and entry["contract_id"] not in declared
    )
    if missing:
        fail(
            "full-contract converted input manifest entries are not declared "
            f"as required: {missing}"
        )


def require_required_inputs_have_semantic_equivalence_evidence():
    equivalence_path = Path(__file__).with_name(
        "test_h5_input_fixture_equivalence.py"
    )
    equivalence_text = equivalence_path.read_text(encoding="utf-8")

    for contract_id, status in REQUIRED_ENTRIES.items():
        if status not in {"converted", "typed_converted", "sidecar_embedded"}:
            continue
        if contract_id.startswith("restart.protocol_sidecar."):
            for token in ["compare_embedded_sidecar_text", "sidecar_embedded"]:
                if token not in equivalence_text:
                    fail(
                        f"{contract_id} lacks embedded-sidecar semantic "
                        f"evidence token: {token}"
                    )
            continue

        tokens = SEMANTIC_EQUIVALENCE_EVIDENCE.get(contract_id)
        if tokens is None:
            fail(f"{contract_id} lacks semantic equivalence evidence mapping")
        missing = [token for token in tokens if token not in equivalence_text]
        if missing:
            fail(
                f"{contract_id} semantic equivalence evidence is stale: "
                f"missing tokens {missing}"
            )

    stale = sorted(
        set(SEMANTIC_EQUIVALENCE_EVIDENCE)
        - {
            contract_id
            for contract_id, status in REQUIRED_ENTRIES.items()
            if status in {"converted", "typed_converted"}
        }
    )
    if stale:
        fail(
            f"semantic equivalence evidence contains stale contract ids: {stale}"
        )


def validate_core_structural_manifest(fixture_root, h5dump):
    manifest, by_contract, bundle_root, legacy_root = load_manifest(
        fixture_root, "core_structural"
    )
    entries = manifest["entries"]

    if manifest.get("mode") != "nve":
        fail("core_structural manifest mode is not nve")

    require_entry_statuses(by_contract, CORE_REQUIRED_ENTRIES)
    require_entry_semantics(entries)
    require_status_semantics(entries)
    require_manifest_input_sources_match_legacy_mdin(
        entries, collect_mdin_input_file_refs(legacy_root / "mdin.spg.toml")
    )
    require_materialized_sidecars_match_legacy(bundle_root, legacy_root)
    require_manifest_lists_h5_sidecar_tables(
        entries, bundle_root, legacy_root, h5dump
    )
    require_manifest_lists_materialized_sidecar_files(entries, bundle_root)
    require_sidecar_mdin_does_not_spell_h5_table_keys(bundle_root, h5dump)

    protocol_sidecar_ids = {
        entry["contract_id"]
        for entry in entries
        if entry.get("status") == "sidecar_embedded"
    }
    if protocol_sidecar_ids != CORE_PROTOCOL_SIDECARS:
        fail(
            "core protocol sidecar set mismatch: "
            f"actual={sorted(protocol_sidecar_ids)}"
        )

    h5_output_ids = {
        entry["contract_id"]
        for entry in entries
        if entry.get("contract_id", "").startswith("output.h5.")
    }
    if h5_output_ids:
        fail(
            f"core fixture unexpectedly preserves H5 output keys: {h5_output_ids}"
        )

    legacy_output_sidecar_ids = {
        entry["contract_id"]
        for entry in entries
        if entry.get("status") == "legacy_output_sidecar_preserved"
    }
    if legacy_output_sidecar_ids != {"output.legacy_sidecar.mdout"}:
        fail(
            "core legacy output sidecar set mismatch: "
            f"actual={sorted(legacy_output_sidecar_ids)}"
        )


def validate_full_contract_manifest(fixture_root, h5dump):
    manifest, by_contract, bundle_root, legacy_root = load_manifest(
        fixture_root, "full_contract_rerun"
    )
    entries = manifest["entries"]

    if manifest.get("mode") != "rerun":
        fail("manifest mode is not rerun")

    require_entry_statuses(by_contract, REQUIRED_ENTRIES)
    require_all_converted_inputs_are_declared(entries)
    require_entry_semantics(entries)
    require_status_semantics(entries)
    require_manifest_input_sources_match_legacy_mdin(
        entries, collect_mdin_input_file_refs(legacy_root / "mdin.spg.toml")
    )
    require_materialized_sidecars_match_legacy(bundle_root, legacy_root)
    require_manifest_lists_h5_sidecar_tables(
        entries, bundle_root, legacy_root, h5dump
    )
    require_manifest_lists_materialized_sidecar_files(entries, bundle_root)
    require_sidecar_mdin_does_not_spell_h5_table_keys(bundle_root, h5dump)
    require_output_plan_matches_mdin(by_contract, bundle_root, legacy_root)

    for contract_id, expected_status in REQUIRED_ENTRIES.items():
        entry = by_contract.get(contract_id)
        if contract_id.startswith("restart.protocol_sidecar."):
            if entry.get("bundle_file") != "restart.spgr.h5":
                fail(f"{contract_id} is not stored in restart.spgr.h5")

    typed_components = {
        entry["component"]
        for entry in entries
        if entry.get("status") == "typed_converted"
    }
    for component in {"topology", "protocol", "restart", "trajectory"}:
        if component not in typed_components:
            fail(f"typed coverage missing component: {component}")

    output_h5_entries = [
        entry
        for entry in entries
        if entry.get("contract_id", "").startswith("output.h5.")
    ]
    output_h5_ids = {entry["contract_id"] for entry in output_h5_entries}
    if output_h5_ids != REQUIRED_H5_OUTPUT_KEYS:
        fail(f"bundled output H5 plan set mismatch: {sorted(output_h5_ids)}")

    protocol_sidecar_ids = {
        entry["contract_id"]
        for entry in entries
        if entry.get("status") == "sidecar_embedded"
    }
    if protocol_sidecar_ids != REQUIRED_PROTOCOL_SIDECARS:
        fail(
            "protocol sidecar set mismatch: "
            f"actual={sorted(protocol_sidecar_ids)}"
        )

    legacy_output_sidecar_ids = {
        entry["contract_id"]
        for entry in entries
        if entry.get("status") == "legacy_output_sidecar_preserved"
    }
    if legacy_output_sidecar_ids != REQUIRED_LEGACY_OUTPUT_SIDECARS:
        fail(
            "legacy output sidecar set mismatch: "
            f"actual={sorted(legacy_output_sidecar_ids)}"
        )


def require_output_plan_matches_mdin(by_contract, bundle_root, legacy_root):
    bundled_mdin = parse_flat_mdin_values(bundle_root / "mdin.bundled.spg.toml")
    legacy_mdin = parse_flat_mdin_values(legacy_root / "mdin.spg.toml")

    for contract_id in REQUIRED_FULL_RUN_POLICY_KEYS:
        entry = by_contract[contract_id]
        source_key = entry.get("source_key")
        if source_key not in legacy_mdin or source_key not in bundled_mdin:
            fail(f"{contract_id} run policy is missing from a paired mdin")
        if bundled_mdin[source_key] != legacy_mdin[source_key]:
            fail(
                f"{contract_id} run policy mismatch for {source_key}: "
                f"legacy={legacy_mdin[source_key]!r} "
                f"bundled={bundled_mdin[source_key]!r}"
            )

    for contract_id in REQUIRED_H5_OUTPUT_KEYS:
        entry = by_contract[contract_id]
        source_key = entry.get("source_key")
        if source_key not in legacy_mdin:
            fail(
                f"{contract_id} source key missing from legacy mdin: {source_key}"
            )
        if source_key not in bundled_mdin:
            fail(
                f"{contract_id} source key missing from bundled mdin: {source_key}"
            )
        if bundled_mdin[source_key] != legacy_mdin[source_key]:
            fail(
                f"{contract_id} mdin value mismatch for {source_key}: "
                f"legacy={legacy_mdin[source_key]!r} "
                f"bundled={bundled_mdin[source_key]!r}"
            )

    legacy_sidecar_keys = set()
    for contract_id in REQUIRED_LEGACY_OUTPUT_SIDECARS:
        entry = by_contract[contract_id]
        source_key = entry.get("source_key")
        legacy_sidecar_keys.add(source_key)
        if source_key not in legacy_mdin:
            fail(
                f"{contract_id} legacy output key missing from legacy mdin: "
                f"{source_key}"
            )

    active_bundled_legacy_sidecars = legacy_sidecar_keys & set(bundled_mdin)
    rerun_input_conflicts = {"crd", "box", "vel"}
    expected_active_sidecars = legacy_sidecar_keys - rerun_input_conflicts
    if active_bundled_legacy_sidecars != expected_active_sidecars:
        fail(
            "bundled rerun mdin should retain every non-conflicting explicit "
            "legacy output sidecar after H5 trajectory conversion: "
            f"{sorted(active_bundled_legacy_sidecars)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--h5dump", required=True)
    args = parser.parse_args()

    fixture_root = Path(args.fixture_root)
    h5dump = Path(args.h5dump)
    require_fixture_group_structure(fixture_root, "core_structural", False)
    require_fixture_group_structure(fixture_root, "full_contract_rerun", True)
    require_required_inputs_have_semantic_equivalence_evidence()
    validate_core_structural_manifest(fixture_root, h5dump)
    validate_full_contract_manifest(fixture_root, h5dump)


if __name__ == "__main__":
    main()
