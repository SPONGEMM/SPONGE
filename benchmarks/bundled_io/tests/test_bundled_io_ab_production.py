from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import h5py
import pytest

from benchmarks.bundled_io.ab_contracts import (
    AssertionEvidence,
    build_case_evidence,
    load_contract_registry,
    update_evidence_report,
    validate_complete_evidence_report,
    validate_contract_registry,
)
from benchmarks.bundled_io.ab_statistics import (
    StatisticalEquivalencePolicy,
    compare_replicas,
    holm_correct_equivalence_family,
    normal_cdf,
)
from benchmarks.bundled_io.input_semantics import (
    InputSemanticSpec,
    assert_module_semantics,
)
from benchmarks.bundled_io.trajectory_statistics import (
    trajectory_observable_series,
)
from benchmarks.utils import Extractor
from benchmarks.validation.thermostat.tests.utils import (
    read_mass_values,
    write_velocity_file_for_temperature,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "h5_bundle" / "fixtures" / "input_matrix"
XPONGE_DEV_ROOT = REPO_ROOT.parent / "XPONGE"
SITS_FF19SB_CMAP_FIXTURE = "sits_ff19sb_cmap_peptide"
FOCUSED_CORE_TOPOLOGY_FIXTURE = "focused_core_topology_two_atom"
FOCUSED_BOND_FIXTURE = "focused_bond_two_atom"
FOCUSED_ANGLE_FIXTURE = "focused_angle_three_atom"
FOCUSED_UREY_BRADLEY_FIXTURE = "focused_urey_bradley_three_atom"
FOCUSED_DIHEDRAL_FIXTURE = "focused_dihedral_four_atom"
FOCUSED_LISTED_FIXTURE = "focused_listed_two_atom"
FOCUSED_NB14_SCALED_FIXTURE = "focused_nb14_scaled_two_atom"
FOCUSED_NB14_EXTRA_FIXTURE = "focused_nb14_extra_two_atom"
FOCUSED_EDIP_FIXTURE = "focused_edip_two_atom"
FOCUSED_REAXFF_FIXTURE = "focused_reaxff_two_atom"
FOCUSED_EAM_FUNCFL_FIXTURE = "focused_eam_funcfl_two_atom"
FOCUSED_EAM_SETFL_FIXTURE = "focused_eam_setfl_two_atom"
FOCUSED_POSITIONAL_RESTRAINT_FIXTURE = "focused_positional_restraint_two_atom"
FOCUSED_POSITIONAL_RESTRAINT_SIDECAR_FIXTURE = (
    "focused_positional_restraint_sidecar_two_atom"
)
FOCUSED_SOFT_WALL_FIXTURE = "focused_soft_wall_two_atom"
FOCUSED_SOFT_WALL_SIDECAR_FIXTURE = "focused_soft_wall_sidecar_two_atom"
FOCUSED_CV_RESTRAINT_TYPED_FIXTURE = "focused_cv_restraint_typed_two_atom"
FOCUSED_CV_RESTRAINT_SIDECAR_FIXTURE = "focused_cv_restraint_sidecar_two_atom"
FOCUSED_SW_SIDECAR_FIXTURE = "focused_sw_sidecar_three_atom"
FOCUSED_SW_TYPED_FIXTURE = "focused_sw_typed_three_atom"
FOCUSED_TERSOFF_SIDECAR_FIXTURE = "focused_tersoff_sidecar_three_atom"
FOCUSED_TERSOFF_TYPED_FIXTURE = "focused_tersoff_typed_three_atom"
FOCUSED_CUSTOM_PAIR_FIXTURE = "focused_custom_pair_two_atom"
FOCUSED_EXCLUSIONS_FIXTURE = "focused_exclusions_three_atom"
FOCUSED_RESIDUE_SIDECAR_FIXTURE = "focused_residue_sidecar_pbc_four_atom"
FOCUSED_RESIDUE_COM_RES_FIXTURE = "focused_residue_sidecar_com_res_four_atom"
FOCUSED_RESIDUE_TYPED_PBC_FIXTURE = "focused_residue_typed_pbc_four_atom"
FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE = (
    "focused_residue_typed_com_res_four_atom"
)
FOCUSED_GB_HYBRID_FIXTURE = "focused_gb_hybrid_two_atom"
FOCUSED_GB_NATIVE_FIXTURE = "focused_gb_native_two_atom"
FOCUSED_IMPROPER_CONVERTED_CANONICAL_FIXTURE = (
    "focused_improper_converted_canonical_four_atom"
)
FOCUSED_IMPROPER_CONVERTED_ALIAS_FIXTURE = (
    "focused_improper_converted_alias_four_atom"
)
FOCUSED_IMPROPER_NATIVE_FIXTURE = "focused_improper_native_four_atom"
FOCUSED_LJ_SOFT_CORE_FIXTURE = "focused_lj_soft_core_two_atom"
FOCUSED_SUBSYSTEM_DIVISION_FIXTURE = "focused_subsystem_division_two_atom"
FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE = "focused_virtual_atoms_all_types"
FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE = "focused_virtual_atoms_plural_alias"
FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE = "focused_virtual_atoms_pbc_boundary"
FOCUSED_CONSTRAINT_SIDECAR_FIXTURE = "focused_constraint_sidecar_two_atom"
FOCUSED_CONSTRAINT_TYPED_FIXTURE = "focused_constraint_typed_two_atom"
FOCUSED_STEERING_CV_SIDECAR_FIXTURE = "focused_steering_cv_sidecar_two_atom"
FOCUSED_STEERING_CV_TYPED_FIXTURE = "focused_steering_cv_typed_two_atom"
FOCUSED_SITS_NK_TYPED_RESTART_FIXTURE = "focused_sits_nk_typed_restart_two_atom"
FOCUSED_SITS_TYPED_CONFIG_FIXTURE = "focused_sits_typed_config_two_atom"
FOCUSED_SITS_TYPED_INACTIVE_FIXTURE = "focused_sits_typed_inactive_two_atom"
SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS = (
    "0",
    "1",
    "xponge.legacy_to_bundle.v1",
)
SITS_FF19SB_CMAP_SOURCE = (
    REPO_ROOT / "benchmarks" / "performance" / "sits" / "statics" / "ala2_sits"
)
PROFILE = os.environ.get("SPONGE_BUNDLED_IO_AB_PROFILE", "medium").strip()
EVIDENCE_RUN_ID = os.environ.get(
    "SPONGE_BUNDLED_IO_AB_RUN_ID", f"{os.getpid()}-{time.time_ns()}"
)

PROFILE_LIMITS = {
    "medium": {
        "normal_step_limit": 1000,
        "normal_interval": 20,
        "normal_replicas": 10,
        "normal_burn_in_frames": 10,
        "normal_block_size": 8,
        "normal_relative_margin": 5.0e-2,
        "normal_absolute_margin": 1.5e-1,
        "rerun_frame_limit": 2,
    },
    "production": {
        "normal_step_limit": 10000,
        "normal_interval": 100,
        "normal_replicas": 5,
        "normal_burn_in_frames": 20,
        "normal_block_size": 10,
        "normal_relative_margin": 1.0e-2,
        "normal_absolute_margin": 5.0e-2,
        "rerun_frame_limit": 2,
    },
}

STATISTICAL_CONFIDENCE_Z = 3.0
STATISTICAL_MAXIMUM_STD_RATIO = 1.5
STATISTICAL_MINIMUM_BLOCKS_PER_REPLICA = 4

PHYSICAL_ABSOLUTE_MARGINS = {
    "medium": {
        "temperature": 1.0,
        "pressure": 5.0,
        "density": 2.0e-2,
        "position": 2.0e-2,
        "velocity": 2.0e-2,
        "force": 2.0e-1,
        "box_length": 5.0e-2,
        "box_angle": 5.0e-3,
        "box_volume": 1.0,
    },
    "production": {
        "temperature": 5.0e-1,
        "pressure": 2.0,
        "density": 1.0e-2,
        "position": 1.0e-2,
        "velocity": 1.0e-2,
        "force": 1.0e-1,
        "box_length": 2.0e-2,
        "box_angle": 2.0e-3,
        "box_volume": 5.0e-1,
    },
}

DETERMINISTIC_TOLERANCES = {
    "schedule": (0.0, 1.0e-12),
    "position": (1.0e-5, 1.0e-5),
    "velocity": (1.0e-5, 1.0e-4),
    "force": (1.0e-4, 3.0e-5),
    "box": (1.0e-6, 1.0e-7),
    "observable": (1.0e-4, 1.0e-8),
}

FULL_CONTRACT_INPUT_REQUIRED_PATHS = {
    "topology.spgt.h5": {
        "/atoms/mass",
        "/atoms/charge",
        "/atoms/residue_index",
        "/forcefield/bond/atoms",
        "/forcefield/angle/atoms",
        "/forcefield/dihedral/atoms",
        "/forcefield/cmap/grid_value",
        "/forcefield/custom_force/pairwise",
        "/forcefield/custom_force/pairwise/data/custom_pair",
        "/forcefield/custom_force/listed",
        "/forcefield/custom_force/listed/data/custom_bond",
        "/manybody/eam/atom_type",
        "/manybody/sw/atom_type",
        "/manybody/edip/atom_type",
        "/manybody/tersoff/atom_type",
        "/manybody/reaxff/parameters",
        "/manybody/reaxff/type",
    },
    "protocol.spgp.h5": {
        "/cv/config/section/name",
        "/constraint/default/pairs/atoms",
        "/sits/atom_indices",
        "/restraint/config/section/name",
        "/restraint/cv/config/section/name",
        "/restraint/default/atom_indices",
        "/restraint/default/weight",
        "/meta/default/grid",
        "/wall/soft/potential",
        "/steer/config/section/name",
    },
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
}

H5_COMPARE_DATASETS = (
    "/parameters/sponge/output/frame_count",
    "/parameters/sponge/output/last_complete_step",
    "/parameters/sponge/output/last_complete_time",
    "/particles/all/step",
    "/particles/all/time",
    "/particles/all/position/value",
    "/particles/all/velocity/value",
    "/particles/all/force/value",
    "/particles/all/box/edges/value",
)

RESTART_COMPARE_DATASETS = (
    "/particles/all/position/value",
    "/particles/all/velocity/value",
    "/particles/all/box/edges/value",
)

NHC_RESTART_DATASET = "/parameters/restart/thermostat/nose_hoover_chain"
NHC_OBSERVABLE_ROOT = "/observables/all/thermostat/nose_hoover_chain"
BUSSI_RNG_DATASET = "/parameters/restart/rng_state/bussi_thermostat"
BUSSI_RNG_ENGINE_DATASET = f"{BUSSI_RNG_DATASET}/engine"
BUSSI_RNG_STATE_WORDS_DATASET = f"{BUSSI_RNG_DATASET}/state_words"
BUSSI_LAMBDA_DATASET = "/parameters/restart/thermostat/bussi_thermostat/lambda"
BUSSI_CONTINUATION_TOLERANCES = {
    "/particles/all/velocity/value": (1.0e-5, 5.0e-4),
    "/particles/all/force/value": (1.0e-3, 2.0e-2),
}
BUSSI_INPUT_ROUTE_TOLERANCES = {
    **BUSSI_CONTINUATION_TOLERANCES,
    "/particles/all/velocity/value": (1.0e-5, 2.5e-4),
    "/particles/all/force/value": (1.0e-3, 2.0e-2),
}
PRESSURE_BAROSTAT_RNG_DATASET = (
    "/parameters/restart/rng_state/pressure_based_barostat"
)
PRESSURE_BAROSTAT_RNG_ENGINE_DATASET = f"{PRESSURE_BAROSTAT_RNG_DATASET}/engine"
PRESSURE_BAROSTAT_RNG_STATE_WORDS_DATASET = (
    f"{PRESSURE_BAROSTAT_RNG_DATASET}/state_words"
)
PRESSURE_BAROSTAT_G_DATASET = (
    "/parameters/restart/barostat/pressure_based_barostat/g"
)
PRESSURE_BAROSTAT_MDOUT_TOLERANCES = {
    column: (5.0e-4, 2.5e-1)
    for column in ("pressure", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
}
PRESSURE_BAROSTAT_G_TOLERANCE = (5.0e-3, 1.0e-5)
PRESSURE_BAROSTAT_CONTINUATION_TOLERANCES = {
    "/particles/all/velocity/value": (1.0e-5, 5.0e-4),
    "/particles/all/force/value": (1.0e-3, 2.0e-2),
}
PRESSURE_BAROSTAT_INPUT_ROUTE_TOLERANCES = {
    **PRESSURE_BAROSTAT_CONTINUATION_TOLERANCES,
    "/particles/all/force/value": (1.0e-3, 2.0e-2),
}
MONTE_CARLO_BAROSTAT_CONTINUATION_TOLERANCES = {
    "/particles/all/velocity/value": (1.0e-4, 2.0e-4),
}
META_RESTART_ROOT = "/parameters/restart/bias/meta/meta"
META_OBSERVABLE_ROOT = "/observables/all/metadynamics"

TRAJECTORY_REL = Path("output") / "ab.spg.h5md"
OBSERVABLE_REL = Path("output") / "ab.obs.spg.h5md"
RESTART_REL = Path("output") / "ab.spgr.h5"
OUTPUT_FAMILY_COMBINATIONS = (
    ("trajectory",),
    ("observable",),
    ("restart",),
    ("trajectory", "observable"),
    ("trajectory", "restart"),
    ("observable", "restart"),
    ("trajectory", "observable", "restart"),
)
OUTPUT_WRITER_FAILURE_POINTS = tuple(
    (family, phase)
    for family in ("trajectory", "observable", "restart")
    for phase in ("append", "finalize")
)

INPUT_SEMANTIC_SPECS_BY_CASE = {
    "normal_core_h5_output": (
        InputSemanticSpec("input.topology.mass", ("temperature",), 1.0),
        InputSemanticSpec("input.topology.charge", ("PM",), 1.0e-6),
        InputSemanticSpec(
            "input.topology.lj", ("LJ_short", "LJ_long", "LJ"), 1.0e-6
        ),
    ),
    "normal_core_topology_payload_sensitivity": (
        InputSemanticSpec("input.topology.mass", ("temperature",), 1.0e-6),
        InputSemanticSpec("input.topology.charge", ("Coulomb",), 1.0e-6),
        InputSemanticSpec("input.topology.lj", ("LJ",), 1.0e-6),
    ),
    "normal_bond_payload_sensitivity": (
        InputSemanticSpec("input.topology.bond", ("bond",), 1.0e-6),
    ),
    "normal_angle_payload_sensitivity": (
        InputSemanticSpec("input.topology.angle", ("angle",), 1.0e-6),
    ),
    "normal_urey_bradley_payload_sensitivity": (
        InputSemanticSpec(
            "input.topology.urey_bradley", ("urey_bradley",), 1.0e-6
        ),
    ),
    "normal_dihedral_payload_sensitivity": (
        InputSemanticSpec("input.topology.dihedral", ("dihedral",), 1.0e-6),
    ),
    "normal_listed_payload_sensitivity": (
        InputSemanticSpec("input.custom.listed", ("custom_bond",), 1.0e-6),
    ),
    "normal_nb14_scaled_nonzero": (
        InputSemanticSpec(
            "input.topology.nb14", ("nb14_LJ", "nb14_EE"), 1.0e-6
        ),
    ),
    "normal_nb14_extra_nonzero": (
        InputSemanticSpec(
            "input.topology.nb14_extra", ("nb14_LJ", "nb14_EE"), 1.0e-6
        ),
    ),
    "normal_sits_ff19sb_cmap_peptide": (
        InputSemanticSpec("input.topology.cmap", ("cmap",), 1.0e-6),
    ),
    "normal_sits_nk_typed_restart_nonzero": (
        InputSemanticSpec(
            "input.protocol.sits.nk_typed_restart",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
    ),
    "normal_sits_typed_configuration_nonzero": (
        InputSemanticSpec(
            "input.protocol.sits",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
        InputSemanticSpec(
            "input.protocol.sits.nk_typed_restart",
            ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
            1.0e-4,
        ),
    ),
    "normal_edip_nonzero": (
        InputSemanticSpec("input.manybody.edip", ("EDIP",), 1.0e-6),
    ),
    "normal_reaxff_payload_sensitivity": (
        InputSemanticSpec(
            "input.manybody.reaxff",
            ("REAXFF_BOND", "REAXFF_VDW", "REAXFF"),
            1.0e-6,
        ),
    ),
    "normal_eam_funcfl_nonzero": (
        InputSemanticSpec("input.manybody.eam", ("EAM",), 1.0e-6),
    ),
    "normal_eam_setfl_nonzero": (
        InputSemanticSpec("input.manybody.eam", ("EAM",), 1.0e-6),
    ),
    "normal_positional_restraint_typed_nonzero": (
        InputSemanticSpec(
            "input.protocol.positional_restraint", ("restrain",), 1.0e-6
        ),
    ),
    "normal_positional_restraint_sidecar_nonzero": (
        InputSemanticSpec(
            "input.protocol.positional_restraint.sidecar",
            ("restrain",),
            1.0e-6,
        ),
    ),
    "normal_soft_wall_typed_nonzero": (
        InputSemanticSpec("input.protocol.soft_wall", ("z_wall",), 1.0e-6),
    ),
    "normal_soft_wall_sidecar_nonzero": (
        InputSemanticSpec(
            "input.protocol.soft_wall.sidecar", ("z_wall",), 1.0e-6
        ),
    ),
    "normal_cv_restraint_typed_nonzero": (
        InputSemanticSpec(
            "input.protocol.cv_restraint", ("restrain_cv",), 1.0e-6
        ),
    ),
    "normal_cv_restraint_sidecar_nonzero": (
        InputSemanticSpec(
            "input.protocol.cv_restraint.sidecar",
            ("restrain_cv",),
            1.0e-6,
        ),
    ),
    "normal_sw_sidecar_pair_three_body": (
        InputSemanticSpec("input.manybody.sw.sidecar", ("SW",), 1.0e-6),
    ),
    "normal_sw_typed_pair_three_body": (
        InputSemanticSpec("input.manybody.sw", ("SW",), 1.0e-6),
    ),
    "normal_tersoff_sidecar_angular": (
        InputSemanticSpec(
            "input.manybody.tersoff.sidecar", ("potential",), 1.0e-6
        ),
    ),
    "normal_tersoff_typed_angular": (
        InputSemanticSpec("input.manybody.tersoff", ("potential",), 1.0e-6),
    ),
    "normal_custom_pair_nonzero": (
        InputSemanticSpec("input.custom.pairwise", ("custom_pair",), 1.0e-6),
    ),
    "normal_exclusions_coulomb_oracle": (
        InputSemanticSpec("input.topology.exclusions", ("Coulomb",), 1.0e-6),
    ),
    "normal_residue_sidecar_pbc_mapping": (
        InputSemanticSpec(
            "input.topology.residue.sidecar",
            ("bond",),
            1.0e-6,
        ),
    ),
    "normal_residue_sidecar_com_res_virial": (
        InputSemanticSpec(
            "input.topology.residue.sidecar",
            ("bond", "restrain", "pressure", "Pxx"),
            1.0e-6,
        ),
    ),
    "normal_residue_typed_pbc_mapping": (
        InputSemanticSpec(
            "input.topology.residue",
            ("bond",),
            1.0e-6,
        ),
    ),
    "normal_residue_typed_com_res_virial": (
        InputSemanticSpec(
            "input.topology.residue",
            ("bond", "restrain", "pressure", "Pxx"),
            1.0e-6,
        ),
    ),
    "normal_gb_hybrid_nonzero": (
        InputSemanticSpec(
            "input.topology.gb.hybrid_activation", ("gb",), 1.0e-6
        ),
    ),
    "normal_gb_native_nonzero": (
        InputSemanticSpec("input.topology.gb", ("gb",), 1.0e-6),
    ),
    "normal_improper_native_nonzero": (
        InputSemanticSpec(
            "input.topology.improper.native_runtime",
            ("improper_dihedral",),
            1.0e-6,
        ),
    ),
    "normal_improper_converted_canonical_nonzero": (
        InputSemanticSpec(
            "input.topology.improper",
            ("improper_dihedral",),
            1.0e-6,
        ),
    ),
    "normal_improper_converted_alias_nonzero": (
        InputSemanticSpec(
            "input.topology.improper",
            ("improper_dihedral",),
            1.0e-6,
        ),
    ),
    "normal_steering_cv_sidecar_nonzero": (
        InputSemanticSpec(
            "input.protocol.steering.cv_sidecar", ("steer_cv",), 1.0e-6
        ),
    ),
    "normal_steering_cv_typed_nonzero": (
        InputSemanticSpec("input.protocol.steering", ("steer_cv",), 1.0e-6),
        InputSemanticSpec("input.protocol.cv", ("steer_cv",), 1.0e-6),
    ),
    "normal_lj_soft_core_nonzero": (
        InputSemanticSpec("input.topology.lj_soft_core", ("LJ_soft",), 1.0e-6),
    ),
    "normal_subsystem_division_partition": (
        InputSemanticSpec(
            "input.topology.subsystem_division",
            ("LJ_soft_inter", "LJ_soft_intra"),
            1.0e-6,
        ),
    ),
    "normal_virtual_atoms_all_types": (
        InputSemanticSpec("input.topology.virtual_atoms", ("PM",), 1.0e-6),
    ),
    "normal_virtual_atoms_pbc_boundary": (
        InputSemanticSpec("input.topology.virtual_atoms_pbc", ("PM",), 1.0e-6),
    ),
    "normal_virtual_atoms_plural_alias": (
        InputSemanticSpec(
            "input.topology.virtual_atoms_alias", ("PM",), 1.0e-6
        ),
    ),
}

RERUN_INPUT_SEMANTIC_SPECS = (
    InputSemanticSpec(
        "input.topology.nb14_extra", ("nb14_LJ", "nb14_EE"), 1.0e-6
    ),
    InputSemanticSpec("input.topology.bond", ("bond",), 1.0e-6),
    InputSemanticSpec("input.topology.angle", ("angle",), 1.0e-6),
    InputSemanticSpec("input.topology.urey_bradley", ("urey_bradley",), 1.0e-6),
    InputSemanticSpec("input.topology.dihedral", ("dihedral",), 1.0e-6),
    InputSemanticSpec("input.custom.listed", ("custom_bond",), 1.0e-6),
    InputSemanticSpec("input.manybody.eam", ("EAM",), 1.0e-6),
    InputSemanticSpec(
        "input.manybody.reaxff",
        (
            "REAXFF_EEQ",
            "REAXFF_BOND",
            "REAXFF_VDW",
            "REAXFF_ELP",
            "REAXFF_OVUN",
            "REAXFF_ANG",
            "REAXFF_PEN",
            "REAXFF_COA",
            "REAXFF_TOR",
            "REAXFF_CONJ",
            "REAXFF_HB",
            "REAXFF",
        ),
        1.0e-6,
    ),
    InputSemanticSpec(
        "input.protocol.positional_restraint", ("restrain",), 1.0e-6
    ),
    InputSemanticSpec(
        "input.protocol.positional_restraint.sidecar", ("restrain",), 1.0e-6
    ),
    InputSemanticSpec("input.protocol.soft_wall", ("z_wall",), 1.0e-6),
    InputSemanticSpec("input.protocol.soft_wall.sidecar", ("z_wall",), 1.0e-6),
    InputSemanticSpec("input.protocol.cv_restraint", ("restrain_cv",), 1.0e-6),
    InputSemanticSpec(
        "input.protocol.cv_restraint.sidecar", ("restrain_cv",), 1.0e-6
    ),
    InputSemanticSpec("input.protocol.cv", ("distance",), 1.0e-6),
)


@dataclass(frozen=True)
class AbCase:
    name: str
    fixture_case: str
    legacy_subdir: str
    bundled_subdir: str
    mode: str
    vds: bool
    statistical_md: bool
    restart_load_policy: str
    contract_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]
    rerun_start: int = 0
    rerun_strip: int = 0
    rerun_frame_limit: int | None = 2
    rerun_need_box_update: bool = False
    rerun_velocity_present: bool = True
    rerun_force_output: bool = True
    trajectory_particle_stream: str = "all"
    trajectory_file_name: str = "trajectory.spg.h5md"
    failure_mutation: str | None = None
    failure_branches: tuple[str, ...] = ("legacy", "bundled")
    expected_error_category: str = ""
    expected_diagnostic_tokens: tuple[str, ...] = ()
    output_chunk_size: int = 1
    output_repair_policy: str = "strict"
    normal_step_limit: int | None = None
    normal_interval: int | None = None
    normal_dt: float | None = None
    expected_trajectory_frames: int | None = None
    input_behavior_only: bool = False
    evolution_contract_ids: tuple[str, ...] = ()
    evolution_cohort: str = ""
    evolution_dt: float = 0.0
    output_families: tuple[str, ...] = ()
    legacy_output_mode: str = "coexist"
    output_fault_family: str = ""
    output_fault_phase: str = ""


NORMAL_SUCCESS = "normal_success"
FAILURE_SEMANTICS = "failure_semantics"
NONFINITE_PROPAGATION = "nonfinite_propagation"


def _case_output_classification(case: AbCase) -> str:
    if case.failure_mutation is not None or case.mode == "output_fault":
        return FAILURE_SEMANTICS
    if case.mode == NONFINITE_PROPAGATION:
        return NONFINITE_PROPAGATION
    return NORMAL_SUCCESS


@dataclass
class AbRun:
    replica_index: int
    replica_seed: int
    legacy_dir: Path
    bundled_dir: Path
    legacy_metrics: dict[str, object]
    bundled_metrics: dict[str, object]
    legacy_output_contract: dict[str, object]
    bundled_output_contract: dict[str, object]


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


@dataclass(frozen=True)
class H5VirtualAtomRecord:
    atom: int
    kind: int
    source_atoms: tuple[int, ...]
    parameters: tuple[float, ...]


def _cases_for_profile() -> list[AbCase]:
    cases = [
        AbCase(
            name="normal_structural_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="structural_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=("input.restart_load.structural",),
            assertion_ids=("restart_continuation_equivalence",),
        ),
        AbCase(
            name="normal_core_h5_output",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=True,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.normal_md",
                "output.legacy.mdout",
                "output.legacy.mdinfo",
                "output.legacy.crd",
                "output.legacy.box",
                "output.legacy.velocity",
                "output.legacy.force",
                "output.legacy.restart",
                "output.trajectory",
                "output.observable",
                "output.restart",
                "output.trajectory.vds_off",
                "input.topology.mass",
                "input.topology.charge",
                "input.topology.lj",
            ),
            assertion_ids=(
                "mdout_statistical_equivalence",
                "mdinfo_structured_equivalence",
                "h5_statistical_equivalence",
                "particle_legacy_coexistence",
                "restart_structural_coexistence",
                "restart_continuation_equivalence",
                "input_semantic_equivalence",
            ),
        ),
        AbCase(
            name="normal_core_topology_payload_sensitivity",
            fixture_case=FOCUSED_CORE_TOPOLOGY_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.mass",
                "input.topology.charge",
                "input.topology.lj",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_bond_payload_sensitivity",
            fixture_case=FOCUSED_BOND_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=("output.legacy.mdout", "input.topology.bond"),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_angle_payload_sensitivity",
            fixture_case=FOCUSED_ANGLE_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=("output.legacy.mdout", "input.topology.angle"),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_urey_bradley_payload_sensitivity",
            fixture_case=FOCUSED_UREY_BRADLEY_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.urey_bradley",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_dihedral_payload_sensitivity",
            fixture_case=FOCUSED_DIHEDRAL_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=("output.legacy.mdout", "input.topology.dihedral"),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_listed_payload_sensitivity",
            fixture_case=FOCUSED_LISTED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=("output.legacy.mdout", "input.custom.listed"),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_nb14_scaled_nonzero",
            fixture_case=FOCUSED_NB14_SCALED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.nb14",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_nb14_extra_nonzero",
            fixture_case=FOCUSED_NB14_EXTRA_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.nb14_extra",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_nhc_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="dynamic_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=(
                "input.restart_load.dynamic",
                "input.restart.dynamic.integrator_state",
                "input.restart.dynamic.nose_hoover_chain",
                "input.bias.nhc",
                "output.restart.dynamic_continuation",
            ),
            assertion_ids=("restart_dynamic_continuation_equivalence",),
        ),
        AbCase(
            name="normal_bussi_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="bussi_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=("input.restart.dynamic.bussi_thermostat",),
            assertion_ids=("restart_bussi_continuation_equivalence",),
        ),
        AbCase(
            name="normal_pressure_barostat_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="pressure_barostat_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=("input.restart.dynamic.pressure_based_barostat",),
            assertion_ids=(
                "restart_pressure_barostat_continuation_equivalence",
            ),
        ),
        AbCase(
            name="normal_middle_langevin_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="portable_stochastic_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=("input.restart.dynamic.middle_langevin_rng",),
            assertion_ids=("restart_middle_langevin_continuation_equivalence",),
        ),
        AbCase(
            name="normal_andersen_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="portable_stochastic_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=("input.restart.dynamic.andersen_rng",),
            assertion_ids=("restart_andersen_continuation_equivalence",),
        ),
        AbCase(
            name="normal_monte_carlo_barostat_dynamic_restart_continuation",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="portable_stochastic_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="dynamic",
            contract_ids=("input.restart.dynamic.monte_carlo_barostat_rng",),
            assertion_ids=(
                "restart_monte_carlo_barostat_continuation_equivalence",
            ),
        ),
        AbCase(
            name="normal_meta_protocol_full_restart_continuation",
            fixture_case="focused_metadynamics_two_atom",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="protocol_full_continuation",
            vds=False,
            statistical_md=False,
            restart_load_policy="protocol/full",
            contract_ids=(
                "input.restart_load.protocol",
                "input.restart_load.full",
                "input.bias.metadynamics",
            ),
            assertion_ids=("restart_protocol_full_continuation_equivalence",),
        ),
        AbCase(
            name="normal_sits_ff19sb_cmap_peptide",
            fixture_case=SITS_FF19SB_CMAP_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=True,
            statistical_md=True,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.normal_md",
                "output.legacy.mdout",
                "output.legacy.mdinfo",
                "output.legacy.crd",
                "output.legacy.box",
                "output.legacy.velocity",
                "output.legacy.force",
                "output.legacy.restart",
                "output.trajectory",
                "output.observable",
                "output.restart",
                "output.trajectory.vds_on",
                "input.topology.cmap",
                "system.ff19sb_ace_ala_nme",
            ),
            assertion_ids=(
                "mdout_statistical_equivalence",
                "mdinfo_structured_equivalence",
                "h5_statistical_equivalence",
                "particle_legacy_coexistence",
                "restart_structural_coexistence",
                "restart_continuation_equivalence",
                "input_semantic_equivalence",
            ),
        ),
        AbCase(
            name="normal_sits_nk_typed_restart_nonzero",
            fixture_case=FOCUSED_SITS_NK_TYPED_RESTART_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="protocol",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.sits.nk_typed_restart",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_sits_typed_configuration_nonzero",
            fixture_case=FOCUSED_SITS_TYPED_CONFIG_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="protocol",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.sits",
                "input.protocol.sits.nk_typed_restart",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
            evolution_contract_ids=(
                "input.protocol.sits",
                "input.protocol.sits.nk_typed_restart",
            ),
            evolution_cohort="protocol_stateful",
            evolution_dt=1.0e-4,
        ),
        AbCase(
            name="normal_sits_typed_inactive_configuration",
            fixture_case=FOCUSED_SITS_TYPED_INACTIVE_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.sits",
            ),
            assertion_ids=("mdout_deterministic_equivalence",),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_edip_nonzero",
            fixture_case=FOCUSED_EDIP_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.edip",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
            evolution_contract_ids=("input.manybody.edip",),
            evolution_cohort="manybody_custom",
            evolution_dt=1.0e-5,
        ),
        AbCase(
            name="normal_reaxff_payload_sensitivity",
            fixture_case=FOCUSED_REAXFF_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.reaxff",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_eam_funcfl_nonzero",
            fixture_case=FOCUSED_EAM_FUNCFL_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.eam",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_eam_setfl_nonzero",
            fixture_case=FOCUSED_EAM_SETFL_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.eam",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_positional_restraint_typed_nonzero",
            fixture_case=FOCUSED_POSITIONAL_RESTRAINT_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.positional_restraint",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_positional_restraint_sidecar_nonzero",
            fixture_case=FOCUSED_POSITIONAL_RESTRAINT_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.positional_restraint.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_soft_wall_typed_nonzero",
            fixture_case=FOCUSED_SOFT_WALL_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.soft_wall",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_soft_wall_sidecar_nonzero",
            fixture_case=FOCUSED_SOFT_WALL_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.soft_wall.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_cv_restraint_typed_nonzero",
            fixture_case=FOCUSED_CV_RESTRAINT_TYPED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.cv_restraint",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_cv_restraint_sidecar_nonzero",
            fixture_case=FOCUSED_CV_RESTRAINT_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.cv_restraint.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_sw_sidecar_pair_three_body",
            fixture_case=FOCUSED_SW_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.sw.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_sw_typed_pair_three_body",
            fixture_case=FOCUSED_SW_TYPED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.sw",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_tersoff_sidecar_angular",
            fixture_case=FOCUSED_TERSOFF_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.tersoff.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_tersoff_typed_angular",
            fixture_case=FOCUSED_TERSOFF_TYPED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.manybody.tersoff",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_custom_pair_nonzero",
            fixture_case=FOCUSED_CUSTOM_PAIR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.custom.pairwise",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_exclusions_coulomb_oracle",
            fixture_case=FOCUSED_EXCLUSIONS_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.exclusions",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_residue_sidecar_pbc_mapping",
            fixture_case=FOCUSED_RESIDUE_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.residue.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_residue_sidecar_com_res_virial",
            fixture_case=FOCUSED_RESIDUE_COM_RES_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.residue.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.001,
            input_behavior_only=True,
            evolution_contract_ids=("input.topology.residue.sidecar",),
            evolution_cohort="topology_pbc",
            evolution_dt=1.0e-3,
        ),
        AbCase(
            name="normal_residue_typed_pbc_mapping",
            fixture_case=FOCUSED_RESIDUE_TYPED_PBC_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.residue",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_residue_typed_com_res_virial",
            fixture_case=FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.residue",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.001,
            input_behavior_only=True,
            evolution_contract_ids=("input.topology.residue",),
            evolution_cohort="topology_pbc",
            evolution_dt=1.0e-3,
        ),
        AbCase(
            name="normal_gb_hybrid_nonzero",
            fixture_case=FOCUSED_GB_HYBRID_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.gb.hybrid_activation",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_gb_native_nonzero",
            fixture_case=FOCUSED_GB_NATIVE_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.gb",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_improper_converted_canonical_nonzero",
            fixture_case=FOCUSED_IMPROPER_CONVERTED_CANONICAL_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.improper",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_improper_converted_alias_nonzero",
            fixture_case=FOCUSED_IMPROPER_CONVERTED_ALIAS_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.improper",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_improper_native_nonzero",
            fixture_case=FOCUSED_IMPROPER_NATIVE_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.improper.native_runtime",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_lj_soft_core_nonzero",
            fixture_case=FOCUSED_LJ_SOFT_CORE_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.lj_soft_core",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_subsystem_division_partition",
            fixture_case=FOCUSED_SUBSYSTEM_DIVISION_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.subsystem_division",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_virtual_atoms_all_types",
            fixture_case=FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.virtual_atoms",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_virtual_atoms_pbc_boundary",
            fixture_case=FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.virtual_atoms_pbc",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_virtual_atoms_plural_alias",
            fixture_case=FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.topology.virtual_atoms_alias",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_constraint_sidecar_projection",
            fixture_case=FOCUSED_CONSTRAINT_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.constraint.sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "constraint_geometry_equivalence",
            ),
            normal_step_limit=4,
            normal_interval=1,
            normal_dt=0.001,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_constraint_typed_projection",
            fixture_case=FOCUSED_CONSTRAINT_TYPED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.constraint",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "constraint_geometry_equivalence",
            ),
            normal_step_limit=4,
            normal_interval=1,
            normal_dt=0.001,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_steering_cv_sidecar_nonzero",
            fixture_case=FOCUSED_STEERING_CV_SIDECAR_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.steering.cv_sidecar",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="normal_steering_cv_typed_nonzero",
            fixture_case=FOCUSED_STEERING_CV_TYPED_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.legacy.mdout",
                "input.protocol.steering",
                "input.protocol.cv",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "input_semantic_equivalence",
            ),
            normal_step_limit=1,
            normal_interval=1,
            normal_dt=0.0,
            input_behavior_only=True,
        ),
        AbCase(
            name="rerun_full_contract_pure_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
                "input.full_contract.inventory",
                "input.full_contract.pure_native",
                "input.manybody.reaxff",
                "input.topology.nb14_extra",
                "input.topology.bond",
                "input.topology.angle",
                "input.topology.urey_bradley",
                "input.topology.dihedral",
                "input.custom.listed",
                "input.manybody.eam",
                "input.protocol.positional_restraint",
                "input.protocol.soft_wall",
                "input.protocol.cv_restraint",
                "input.protocol.cv",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "input_semantic_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_pure_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
                "input.full_contract.inventory",
                "input.full_contract.pure_native",
                "input.manybody.reaxff",
                "input.topology.nb14_extra",
                "input.topology.bond",
                "input.topology.angle",
                "input.topology.urey_bradley",
                "input.topology.dihedral",
                "input.custom.listed",
                "input.manybody.eam",
                "input.protocol.positional_restraint",
                "input.protocol.soft_wall",
                "input.protocol.cv_restraint",
                "input.protocol.cv",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "input_semantic_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_sidecar_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
                "input.full_contract.inventory",
                "input.full_contract.sidecar",
                "input.manybody.reaxff",
                "input.topology.nb14_extra",
                "input.topology.bond",
                "input.topology.angle",
                "input.topology.urey_bradley",
                "input.topology.dihedral",
                "input.custom.listed",
                "input.manybody.eam",
                "input.protocol.positional_restraint.sidecar",
                "input.protocol.soft_wall.sidecar",
                "input.protocol.cv_restraint.sidecar",
                "input.protocol.cv",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "input_semantic_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_sidecar_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
                "input.full_contract.inventory",
                "input.full_contract.sidecar",
                "input.manybody.reaxff",
                "input.topology.nb14_extra",
                "input.topology.bond",
                "input.topology.angle",
                "input.topology.urey_bradley",
                "input.topology.dihedral",
                "input.custom.listed",
                "input.manybody.eam",
                "input.protocol.positional_restraint.sidecar",
                "input.protocol.soft_wall.sidecar",
                "input.protocol.cv_restraint.sidecar",
                "input.protocol.cv",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "input_semantic_equivalence",
            ),
        ),
    ]
    cases.extend(_chunk_boundary_cases())
    cases.extend(_output_family_cases())
    cases.extend(_output_writer_failure_cases())
    cases.extend(_rerun_boundary_cases())
    cases.extend(_failure_cases())
    return cases


def _output_family_cases() -> list[AbCase]:
    cases = []
    for families in OUTPUT_FAMILY_COMBINATIONS:
        family_name = "_".join(families)
        for legacy_mode in ("suppressed", "coexist"):
            cases.append(
                AbCase(
                    name=f"normal_output_family_{family_name}_{legacy_mode}",
                    fixture_case="tip3p_validation_generated",
                    legacy_subdir="generated_legacy",
                    bundled_subdir="generated_bundled",
                    mode="output_family",
                    vds=False,
                    statistical_md=False,
                    restart_load_policy="structural",
                    contract_ids=("output.family.combinations",),
                    assertion_ids=("output_family_process_matrix",),
                    normal_step_limit=2,
                    normal_interval=1,
                    normal_dt=1.0e-3,
                    output_families=families,
                    legacy_output_mode=legacy_mode,
                )
            )
    return cases


def _audit_output_family_cases(cases: list[AbCase]) -> dict[str, object]:
    expected = {
        (families, legacy_mode)
        for families in OUTPUT_FAMILY_COMBINATIONS
        for legacy_mode in ("suppressed", "coexist")
    }
    output_cases = [case for case in cases if case.mode == "output_family"]
    actual = {
        (case.output_families, case.legacy_output_mode) for case in output_cases
    }
    if len(output_cases) != len(expected) or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(
            "output family process matrix is incomplete: "
            f"cases={len(output_cases)}, expected={len(expected)}, "
            f"missing={missing}, extra={extra}"
        )
    for case in output_cases:
        if case.contract_ids != ("output.family.combinations",):
            raise AssertionError(
                f"{case.name} has unexpected contracts {case.contract_ids}"
            )
        if case.assertion_ids != ("output_family_process_matrix",):
            raise AssertionError(
                f"{case.name} has unexpected assertions {case.assertion_ids}"
            )
        if case.vds or case.normal_step_limit != 2 or case.normal_interval != 1:
            raise AssertionError(
                f"{case.name} changed the bounded process-matrix schedule"
            )
    return {
        "case_count": len(output_cases),
        "combination_count": len(OUTPUT_FAMILY_COMBINATIONS),
        "legacy_modes": ("suppressed", "coexist"),
    }


def _output_writer_failure_cases() -> list[AbCase]:
    cases = []
    for family, phase in OUTPUT_WRITER_FAILURE_POINTS:
        cases.append(
            AbCase(
                name=f"failure_output_{family}_{phase}_isolation",
                fixture_case="tip3p_validation_generated",
                legacy_subdir="generated_legacy",
                bundled_subdir="generated_bundled",
                mode="output_fault",
                vds=False,
                statistical_md=False,
                restart_load_policy="structural",
                contract_ids=("output.writer.failure_isolation",),
                assertion_ids=("writer_failure_isolation",),
                normal_step_limit=2,
                normal_interval=1,
                normal_dt=1.0e-3,
                output_families=("trajectory", "observable", "restart"),
                legacy_output_mode="suppressed",
                output_fault_family=family,
                output_fault_phase=phase,
            )
        )
    return cases


def _chunk_boundary_cases() -> list[AbCase]:
    shared = {
        "fixture_case": "tip3p_validation_generated",
        "legacy_subdir": "generated_legacy",
        "bundled_subdir": "generated_bundled",
        "mode": "chunk_boundary",
        "vds": True,
        "statistical_md": False,
        "restart_load_policy": "structural",
        "contract_ids": (
            "output.trajectory",
            "output.trajectory.vds_on",
            "output.trajectory.chunk_size",
        ),
        "assertion_ids": ("h5_chunk_boundary_equivalence",),
        "output_chunk_size": 4,
        "normal_interval": 1,
        "normal_dt": 0.0001,
    }
    cases = [
        AbCase(
            name="normal_vds_chunk_minus_one",
            normal_step_limit=3,
            expected_trajectory_frames=3,
            **shared,
        ),
        AbCase(
            name="normal_vds_chunk_exact",
            normal_step_limit=4,
            expected_trajectory_frames=4,
            **shared,
        ),
        AbCase(
            name="normal_vds_chunk_plus_one",
            normal_step_limit=5,
            expected_trajectory_frames=5,
            **shared,
        ),
        AbCase(
            name="normal_vds_chunk_two_plus_one",
            normal_step_limit=9,
            expected_trajectory_frames=9,
            **shared,
        ),
    ]
    cases.append(
        AbCase(
            name="normal_vds_complete_prefix_noop",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="chunk_boundary",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "output.trajectory",
                "output.trajectory.vds_on",
                "output.trajectory.chunk_size",
                "output.vds.complete_prefix_repair",
            ),
            assertion_ids=(
                "h5_chunk_boundary_equivalence",
                "h5_complete_prefix_repair_equivalence",
            ),
            output_chunk_size=4,
            output_repair_policy="complete_prefix",
            normal_step_limit=5,
            normal_interval=1,
            normal_dt=0.0001,
            expected_trajectory_frames=5,
        )
    )
    return cases


def _rerun_boundary_cases() -> list[AbCase]:
    shared_contracts = (
        "runtime.rerun",
        "input.rerun.start",
        "input.rerun.strip",
        "input.rerun.frame_limit",
        "input.rerun.box_update",
        "input.trajectory.velocity_optional",
        "output.legacy.mdout",
    )
    shared_assertions = (
        "mdout_deterministic_equivalence",
        "rerun_selection_equivalence",
    )
    return [
        AbCase(
            name="rerun_restart_absent_same_bootstrap_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="absent",
            contract_ids=(
                "runtime.rerun",
                "input.rerun.start",
                "input.rerun.strip",
                "input.rerun.frame_limit",
                "input.rerun.box_update",
                "input.trajectory.velocity_optional",
                "input.restart_load.absent",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
            ),
            assertion_ids=(
                "mdout_deterministic_equivalence",
                "rerun_selection_equivalence",
                "h5_rerun_semantic_equivalence",
            ),
            rerun_frame_limit=2,
            rerun_need_box_update=False,
        ),
        AbCase(
            name="rerun_boundary_start0_strip0_limit1_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=shared_contracts,
            assertion_ids=shared_assertions,
            rerun_frame_limit=1,
        ),
        AbCase(
            name="rerun_boundary_start1_strip0_unlimited_no_velocity_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=shared_contracts,
            assertion_ids=shared_assertions,
            rerun_start=1,
            rerun_frame_limit=None,
            rerun_need_box_update=True,
            rerun_velocity_present=False,
            rerun_force_output=False,
            trajectory_file_name="trajectory.no_velocity.spg.h5md",
        ),
        AbCase(
            name="rerun_optional_no_velocity_no_force_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                *shared_contracts,
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
            ),
            assertion_ids=(
                *shared_assertions,
                "h5_rerun_semantic_equivalence",
            ),
            rerun_frame_limit=2,
            rerun_need_box_update=False,
            rerun_velocity_present=False,
            rerun_force_output=False,
            trajectory_file_name="trajectory.no_velocity.spg.h5md",
        ),
        AbCase(
            name="rerun_selected_stream_velocity_force_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                *shared_contracts,
                "input.trajectory.particle_stream",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
            ),
            assertion_ids=(
                *shared_assertions,
                "h5_rerun_semantic_equivalence",
            ),
            rerun_frame_limit=2,
            trajectory_particle_stream="selected",
            trajectory_file_name="trajectory.selected.spg.h5md",
        ),
        AbCase(
            name="rerun_boundary_start0_strip1_beyond_selected_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                *shared_contracts,
                "input.trajectory.particle_stream",
            ),
            assertion_ids=shared_assertions,
            rerun_strip=1,
            rerun_frame_limit=3,
            trajectory_particle_stream="selected",
            trajectory_file_name="trajectory.selected.spg.h5md",
        ),
        AbCase(
            name="rerun_boundary_start0_strip0_exact_eof_box_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                *shared_contracts,
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
            ),
            assertion_ids=(
                *shared_assertions,
                "h5_rerun_semantic_equivalence",
            ),
            rerun_frame_limit=2,
            rerun_need_box_update=True,
        ),
        AbCase(
            name="rerun_boundary_start1_strip1_limit1_selected_no_velocity_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                *shared_contracts,
                "input.trajectory.particle_stream",
            ),
            assertion_ids=shared_assertions,
            rerun_start=1,
            rerun_strip=1,
            rerun_frame_limit=1,
            rerun_velocity_present=False,
            trajectory_particle_stream="selected",
            trajectory_file_name="trajectory.selected_no_velocity.spg.h5md",
        ),
    ]


def _failure_cases() -> list[AbCase]:
    shared = {
        "fixture_case": "full_contract_rerun",
        "legacy_subdir": "legacy_input",
        "bundled_subdir": "bundled_input_with_legacy_sidecar/bundle",
        "mode": "failure",
        "vds": False,
        "statistical_md": False,
        "restart_load_policy": "structural",
        "contract_ids": ("failure.input_configuration",),
        "assertion_ids": ("stable_failure_semantics",),
    }
    sidecar_shared = {
        **shared,
        "contract_ids": ("failure.sidecar_table",),
    }
    metadata_shared = {
        **shared,
        "contract_ids": ("failure.h5_metadata",),
    }
    nb14_metadata_shared = {
        **metadata_shared,
        "bundled_subdir": "bundled_input/bundle",
    }
    restart_owner_shared = {
        key: value
        for key, value in shared.items()
        if key != "restart_load_policy"
    }
    restart_owner_shared["contract_ids"] = ("failure.restart_owner_state",)
    return [
        AbCase(
            name="failure_missing_trajectory_binding",
            failure_mutation="missing_trajectory",
            expected_error_category="spongeErrorMissingCommand",
            expected_diagnostic_tokens=("trajectory",),
            **shared,
        ),
        AbCase(
            name="failure_invalid_output_chunk_size",
            failure_mutation="invalid_chunk_size",
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("output_h5_trajectory_chunk_size",),
            **shared,
        ),
        AbCase(
            name="failure_invalid_output_vds_value",
            failure_mutation="invalid_vds_value",
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("output_h5_trajectory_vds",),
            **shared,
        ),
        AbCase(
            name="failure_invalid_output_repair_policy",
            failure_mutation="invalid_repair_policy",
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("output_h5_trajectory_repair_policy",),
            **shared,
        ),
        AbCase(
            name="failure_invalid_restart_policy",
            failure_mutation="invalid_restart_policy",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("input_h5_restart_load",),
            **shared,
        ),
        AbCase(
            name="failure_missing_topology_binding",
            failure_mutation="missing_topology",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("input_h5_topology_path",),
            **shared,
        ),
        AbCase(
            name="failure_missing_protocol_binding",
            failure_mutation="missing_protocol",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("input_h5_protocol_path",),
            **shared,
        ),
        AbCase(
            name="failure_mixed_legacy_h5_trajectory",
            failure_mutation="mixed_trajectory",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=("input_h5_trajectory_path", "crd"),
            **shared,
        ),
        AbCase(
            name="failure_mixed_legacy_h5_restart",
            failure_mutation="mixed_restart",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "input_h5_restart_path",
                "legacy coordinate/velocity restart inputs",
            ),
            **shared,
        ),
        AbCase(
            name="failure_sidecar_unsupported_key",
            failure_mutation="unsupported_sidecar_key",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "unsupported H5 legacy sidecar key",
                "input_h5_topology_path",
                "not_a_supported_sidecar_key",
            ),
            **sidecar_shared,
        ),
        AbCase(
            name="failure_sidecar_key_path_length_mismatch",
            failure_mutation="sidecar_length_mismatch",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "legacy sidecar key/path dataset length mismatch",
            ),
            **sidecar_shared,
        ),
        AbCase(
            name="failure_sidecar_path_conflict",
            failure_mutation="sidecar_path_conflict",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "H5 legacy sidecar key conflicts with existing command",
                "mass_in_file",
                "existing=",
                "h5=",
            ),
            **sidecar_shared,
        ),
        AbCase(
            name="failure_h5_topology_atom_count_mismatch",
            failure_mutation="h5_topology_atom_count_mismatch",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "input_h5_restart_path",
                "restart atom_count does not match topology",
            ),
            **metadata_shared,
        ),
        AbCase(
            name="failure_h5_topology_mass_shape",
            failure_mutation="h5_topology_mass_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Topology_Core",
                "atom mass dataset /atoms/mass must be one-dimensional",
            ),
            **metadata_shared,
        ),
        AbCase(
            name="failure_h5_topology_mass_dtype",
            failure_mutation="h5_topology_mass_dtype",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Topology_Core",
                "failed to read native topology H5 core state",
                "Unable to read the dataset",
            ),
            **metadata_shared,
        ),
        AbCase(
            name="failure_h5_topology_schema_version",
            failure_mutation="h5_topology_schema_version",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorValueErrorCommand",
            expected_diagnostic_tokens=(
                "Xponge::Validate_H5_Topology_Schema_Version",
                "input_h5_topology_path",
                "unsupported /schema/version",
                "unsupported.topology.v999",
            ),
            **metadata_shared,
        ),
        AbCase(
            name="failure_h5_nb14_dual_root",
            failure_mutation="h5_nb14_dual_root",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_NB14_Extra",
                "cannot define both /forcefield/nb14 and "
                "/forcefield/nb14_extra",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_nb14_extra_param_shape",
            failure_mutation="h5_nb14_extra_param_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_NB14_Extra",
                "/forcefield/nb14_extra/params must have shape [n,3]",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_eam_unknown_format",
            failure_mutation="h5_eam_unknown_format",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_EAM",
                "unsupported native EAM format",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_eam_embed_shape",
            failure_mutation="h5_eam_embed_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_EAM",
                "EAM raw embedding table",
                "must have shape [1,2]",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_positional_restraint_dual_owner",
            failure_mutation="h5_positional_restraint_dual_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Positional_Restraint",
                "cannot both own positional restraint state",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_positional_restraint_weight_shape",
            failure_mutation="h5_positional_restraint_weight_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Positional_Restraint",
                "restraint weight",
                "must have shape [2,3]",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_soft_wall_dual_owner",
            failure_mutation="h5_soft_wall_dual_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Soft_Wall",
                "cannot both own soft-wall state",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_soft_wall_count_shape",
            failure_mutation="h5_soft_wall_count_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Soft_Wall",
                "soft-wall count dataset /wall/soft/count must be scalar",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_soft_wall_name_shape",
            failure_mutation="h5_soft_wall_name_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Soft_Wall",
                "/wall/soft/name must have shape [1]",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_soft_wall_potential_shape",
            failure_mutation="h5_soft_wall_potential_shape",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_Native_Soft_Wall",
                "/wall/soft/potential must have shape [1]",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_cv_restraint_dual_owner",
            failure_mutation="h5_cv_restraint_dual_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Materialize_H5_CV_Config",
                "cannot both own CV-restraint state",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_cv_restraint_partial_owner",
            failure_mutation="h5_cv_restraint_partial_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_CV_Config",
                "typed CV restraint requires both /restraint/config and "
                "/restraint/cv/config",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_cv_restraint_offset_mismatch",
            failure_mutation="h5_cv_restraint_offset_mismatch",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_CV_Config",
                "/restraint/cv/config section offsets and key/value lengths "
                "are inconsistent",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_h5_cv_restraint_definition_conflict",
            failure_mutation="h5_cv_restraint_definition_conflict",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorBadFileFormat",
            expected_diagnostic_tokens=(
                "Materialize_H5_CV_Config",
                "/restraint/config conflicts with another typed CV "
                "definition for distance",
            ),
            **nb14_metadata_shared,
        ),
        AbCase(
            name="failure_restart_dynamic_without_owner",
            restart_load_policy="dynamic",
            failure_mutation="restart_dynamic_without_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Restart contains Nose-Hoover chain state",
                "nose_hoover_chain thermostat is not initialized",
            ),
            **restart_owner_shared,
        ),
        AbCase(
            name="failure_restart_protocol_without_owner",
            restart_load_policy="protocol",
            failure_mutation="restart_protocol_without_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Restart contains metadynamics state",
                "meta module is not initialized",
            ),
            **restart_owner_shared,
        ),
        AbCase(
            name="failure_restart_full_without_owner",
            restart_load_policy="full",
            failure_mutation="restart_full_without_owner",
            failure_branches=("bundled",),
            expected_error_category="spongeErrorConflictingCommand",
            expected_diagnostic_tokens=(
                "Restart contains Nose-Hoover chain state",
                "nose_hoover_chain thermostat is not initialized",
            ),
            **restart_owner_shared,
        ),
    ]


def _input_semantic_specs(case: AbCase) -> tuple[InputSemanticSpec, ...]:
    if case.mode == "rerun":
        candidates = RERUN_INPUT_SEMANTIC_SPECS
    else:
        candidates = INPUT_SEMANTIC_SPECS_BY_CASE.get(case.name, ())
    return tuple(
        spec for spec in candidates if spec.contract_id in case.contract_ids
    )


@pytest.fixture(scope="module", autouse=True)
def _require_complete_evidence_for_full_matrix(request):
    yield
    if request.session.testsfailed:
        return
    selected_case_ids = {
        item.callspec.params["case"].name
        for item in request.session.items
        if hasattr(item, "callspec")
        and isinstance(item.callspec.params.get("case"), AbCase)
    }
    expected_case_ids = {case.name for case in _cases_for_profile()}
    if selected_case_ids != expected_case_ids:
        return
    validate_complete_evidence_report(
        _output_root() / "ab_evidence.json",
        load_contract_registry(),
        EVIDENCE_RUN_ID,
    )


@pytest.mark.parametrize(
    "case", _cases_for_profile(), ids=lambda case: case.name
)
def test_legacy_and_bundled_ab_behavior(case: AbCase):
    if PROFILE not in PROFILE_LIMITS:
        raise AssertionError(
            "SPONGE_BUNDLED_IO_AB_PROFILE must be one of "
            f"{sorted(PROFILE_LIMITS)}"
        )
    contracts = load_contract_registry()
    validate_contract_registry(contracts, _cases_for_profile())
    output_classification = _case_output_classification(case)
    if case.failure_mutation is not None:
        if output_classification != FAILURE_SEMANTICS:
            raise AssertionError(f"{case.name} failure case is misclassified")
        _run_failure_case(case, contracts)
        return
    if case.mode == "structural_continuation":
        _run_structural_restart_case(case, contracts)
        return
    if case.mode == "bussi_continuation":
        _run_bussi_dynamic_restart_case(case, contracts)
        return
    if case.mode == "pressure_barostat_continuation":
        _run_pressure_barostat_dynamic_restart_case(case, contracts)
        return
    if case.mode == "portable_stochastic_continuation":
        _run_portable_stochastic_restart_case(case, contracts)
        return
    if case.mode == "dynamic_continuation":
        _run_nhc_dynamic_restart_case(case, contracts)
        return
    if case.mode == "protocol_full_continuation":
        _run_meta_protocol_full_restart_case(case, contracts)
        return
    if case.mode == "chunk_boundary":
        _run_chunk_boundary_case(case, contracts)
        return
    if case.mode == "output_family":
        _run_output_family_case(case, contracts)
        return
    if case.mode == "output_fault":
        if output_classification != FAILURE_SEMANTICS:
            raise AssertionError(f"{case.name} output fault is misclassified")
        _run_output_writer_failure_case(case, contracts)
        return
    if output_classification != NORMAL_SUCCESS:
        raise AssertionError(
            f"{case.name} cannot emit supported functionality evidence as "
            f"{output_classification}"
        )
    root = _output_root()
    case_root = root / case.name
    runs = []
    for replica_index in range(_replica_count(case)):
        replica_seed = _replica_seed(replica_index)
        replica_root = case_root / f"replica_{replica_index:02d}"
        legacy_dir, bundled_dir = _prepare_case_pair(
            case, replica_root, replica_seed
        )
        _prepare_mdin(
            legacy_dir,
            "mdin.spg.toml",
            case,
            branch="legacy",
            replica_seed=replica_seed,
        )
        _prepare_mdin(
            bundled_dir,
            "mdin.bundled.spg.toml",
            case,
            branch="bundled",
            replica_seed=replica_seed,
        )

        legacy_metrics = _run_sponge(legacy_dir, _mdin_name(legacy_dir))
        bundled_metrics = _run_sponge(bundled_dir, _mdin_name(bundled_dir))
        _assert_pure_bundle_has_no_native_bridge_artifacts(case, bundled_dir)
        runs.append(
            AbRun(
                replica_index=replica_index,
                replica_seed=replica_seed,
                legacy_dir=legacy_dir,
                bundled_dir=bundled_dir,
                legacy_metrics=legacy_metrics,
                bundled_metrics=bundled_metrics,
                legacy_output_contract=_validate_branch_output_contract(
                    case, legacy_dir, branch="legacy"
                ),
                bundled_output_contract=_validate_branch_output_contract(
                    case, bundled_dir, branch="bundled"
                ),
            )
        )

    comparison, assertion_evidence = _compare_outputs(case, runs)
    evidence = build_case_evidence(contracts, case, assertion_evidence)
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "vds": case.vds,
        "statistical_md": case.statistical_md,
        "restart_load_policy": case.restart_load_policy,
        "replica_count": len(runs),
        "replicas": [
            {
                "index": run.replica_index,
                "seed": run.replica_seed,
                "legacy": run.legacy_metrics,
                "bundled": run.bundled_metrics,
                "legacy_output_contract": run.legacy_output_contract,
                "bundled_output_contract": run.bundled_output_contract,
            }
            for run in runs
        ],
        "comparison": comparison,
        "contract_boundary": {
            "cross_process_vds_reopen_append_resume": "unsupported",
            "vds_resume_covered_semantics": (
                "complete-prefix terminal tail repair and no-op policy path"
            ),
        },
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        root / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "replica_seeds": [run.replica_seed for run in runs],
            "vds": case.vds,
            "statistical_md": case.statistical_md,
            "restart_load_policy": case.restart_load_policy,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            "mpi_rank_count": os.environ.get("OMPI_COMM_WORLD_SIZE", "1"),
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B metrics: {metrics_path}")


def _assert_pure_bundle_has_no_native_bridge_artifacts(
    case: AbCase, bundled_dir: Path
) -> None:
    if "input.full_contract.pure_native" not in case.contract_ids:
        return
    bridge_roots = (
        ".sponge_h5_restart_protocol",
        ".sponge_h5_native_protocol",
        ".sponge_h5_native_custom_force",
        ".sponge_h5_native_qc",
        ".sponge_h5_native_manybody",
    )
    unexpected = [
        root for root in bridge_roots if (bundled_dir / root).exists()
    ]
    if unexpected:
        raise AssertionError(
            f"{case.name} pure bundle produced native bridge artifacts: "
            f"{unexpected}"
        )


def _run_failure_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    legacy_dir, bundled_dir = _prepare_case_pair(case, case_root, 20260709)
    supported_schema_controls = {}
    for branch, case_dir, mdin_name in (
        ("legacy", legacy_dir, "mdin.spg.toml"),
        ("bundled", bundled_dir, "mdin.bundled.spg.toml"),
    ):
        _prepare_mdin(
            case_dir,
            mdin_name,
            case,
            branch=branch,
            replica_seed=20260709,
        )
        _mutate_failure_mdin(case, case_dir / mdin_name, branch)
        if (
            case.failure_mutation == "h5_topology_schema_version"
            and branch == "bundled"
        ):
            supported_schema_controls = _run_supported_topology_schema_controls(
                case, case_dir
            )
        _mutate_failure_h5(case, case_dir, branch)

    outcomes = {}
    for branch, case_dir in (("legacy", legacy_dir), ("bundled", bundled_dir)):
        if branch not in case.failure_branches:
            continue
        outcome = _run_sponge_process(case_dir, _mdin_name(case_dir))
        if outcome.returncode == 0:
            raise AssertionError(f"{case.name} {branch} unexpectedly succeeded")
        category = _failure_category(outcome.stdout + "\n" + outcome.stderr)
        if not category:
            raise AssertionError(f"{case.name} {branch} has no error category")
        if (
            case.expected_error_category
            and category != case.expected_error_category
        ):
            raise AssertionError(
                f"{case.name} {branch} category mismatch: "
                f"expected={case.expected_error_category}, actual={category}"
            )
        normalized = (outcome.stdout + "\n" + outcome.stderr).lower()
        missing_tokens = [
            token
            for token in case.expected_diagnostic_tokens
            if token.lower() not in normalized
        ]
        if missing_tokens:
            raise AssertionError(
                f"{case.name} {branch} diagnostics are missing tokens: "
                f"{missing_tokens}"
            )
        outcomes[branch] = {
            "exit_code": outcome.returncode,
            "category": category,
            "diagnostic_tokens": list(case.expected_diagnostic_tokens),
            "elapsed_s": outcome.elapsed_s,
        }

    if set(case.failure_branches) == {"legacy", "bundled"}:
        if outcomes["legacy"]["exit_code"] != outcomes["bundled"]["exit_code"]:
            raise AssertionError(
                f"{case.name} exit code mismatch: "
                f"legacy={outcomes['legacy']['exit_code']}, "
                f"bundled={outcomes['bundled']['exit_code']}"
            )
        if outcomes["legacy"]["category"] != outcomes["bundled"]["category"]:
            raise AssertionError(
                f"{case.name} error category mismatch: "
                f"legacy={outcomes['legacy']['category']}, "
                f"bundled={outcomes['bundled']['category']}"
            )

    assertion = AssertionEvidence(
        assertion_id="stable_failure_semantics",
        evidence_level="F1",
        details={
            "output_classification": FAILURE_SEMANTICS,
            "mutation": case.failure_mutation,
            "branches": list(case.failure_branches),
            "outcomes": outcomes,
            "supported_schema_controls": supported_schema_controls,
        },
    )
    evidence = build_case_evidence(contracts, case, (assertion,))
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "failure": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "failure_mutation": case.failure_mutation,
            "branches": list(case.failure_branches),
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B failure metrics: {metrics_path}")


def _run_structural_restart_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    legacy_dir, bundled_dir = _prepare_normal_tip3p_pair(case_root, 20260714)
    producer_case = replace(
        case,
        mode="normal",
        normal_step_limit=1,
        normal_interval=1,
        normal_dt=0.0001,
    )
    for branch, case_dir, mdin_name in (
        ("legacy", legacy_dir, "mdin.spg.toml"),
        ("bundled", bundled_dir, "mdin.bundled.spg.toml"),
    ):
        _prepare_mdin(
            case_dir,
            mdin_name,
            producer_case,
            branch=branch,
            replica_seed=20260714,
        )
        mdin_path = case_dir / mdin_name
        mdin_path.write_text(
            _remove_key_lines(
                mdin_path.read_text(encoding="utf-8"), {"constrain_mode"}
            )
            + "\n",
            encoding="utf-8",
        )

    producer_metrics = {
        "legacy": _run_sponge(legacy_dir, "mdin.spg.toml"),
        "bundled": _run_sponge(bundled_dir, "mdin.bundled.spg.toml"),
    }
    producer_state = {
        "legacy": _validate_restart_legacy_coexistence(
            case, legacy_dir, legacy_dir / RESTART_REL
        ),
        "bundled": _validate_restart_legacy_coexistence(
            case, bundled_dir, bundled_dir / RESTART_REL
        ),
    }
    run = AbRun(
        replica_index=0,
        replica_seed=20260714,
        legacy_dir=legacy_dir,
        bundled_dir=bundled_dir,
        legacy_metrics=producer_metrics["legacy"],
        bundled_metrics=producer_metrics["bundled"],
        legacy_output_contract={},
        bundled_output_contract={},
    )
    continuation = _compare_restart_continuation(case, run)
    assertion = AssertionEvidence(
        assertion_id="restart_continuation_equivalence",
        evidence_level="E4",
        details={
            "method": "independent_producers_and_branch_owned_restart_sources",
            "producer_state": producer_state,
            "continuation": continuation,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "restart_load_policy": case.restart_load_policy,
        "producer_metrics": producer_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_branches": ["legacy", "bundled"],
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B structural restart metrics: {metrics_path}")


def _write_bussi_mdin(
    case_dir: Path,
    *,
    input_route: str,
    step_limit: int,
    thermostat_seed: int,
    topology_hash: str = "",
    atom_order_hash: str = "",
) -> None:
    if input_route == "legacy":
        input_lines = [
            'default_in_file_prefix = "tip3p"',
            'velocity_in_file = "initial_velocity.txt"',
        ]
    elif input_route == "bundled_initial":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "restart.spgr.h5"',
            'input_h5_restart_load = "structural"',
        ]
    elif input_route == "bundled_dynamic":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "output/producer.spgr.h5"',
            'input_h5_restart_load = "dynamic"',
        ]
    else:
        raise AssertionError(f"unknown Bussi input route: {input_route}")

    lines = [
        f'md_name = "bundled io ab Bussi {input_route}"',
        'mode = "nvt"',
        f"step_limit = {step_limit}",
        "dt = 0.001",
        "cutoff = 8.0",
        *input_lines,
        'thermostat = "bussi_thermostat"',
        'thermostat_mode = "bussi_thermostat"',
        f"thermostat_seed = {thermostat_seed}",
        "thermostat_tau = 0.1",
        "target_temperature = 300.0",
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'crd = "output/legacy.crd"',
        'box = "output/legacy.box"',
        'vel = "output/legacy.vel"',
        'frc = "output/legacy.frc"',
        'rst = "output/legacy_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 4",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
    ]
    if topology_hash and atom_order_hash:
        lines.extend(
            [
                f'output_h5_restart_topology_hash = "{topology_hash}"',
                f'output_h5_restart_atom_order_hash = "{atom_order_hash}"',
            ]
        )
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_typed_rng_restart_state(path: Path, module: str) -> dict[str, object]:
    root = f"/parameters/restart/rng_state/{module}"
    with h5py.File(path, "r") as restart:
        required = {
            f"{root}/engine",
            f"{root}/schema_version",
            f"{root}/state_words",
        }
        missing = sorted(
            dataset for dataset in required if dataset not in restart
        )
        if missing:
            raise AssertionError(
                f"typed RNG restart state is missing {missing}: {path}"
            )
        engine = restart[f"{root}/engine"].asstr()[()]
        schema_version = int(
            restart[f"{root}/schema_version"][:].reshape(-1)[0]
        )
        state_words = [
            int(value)
            for value in restart[f"{root}/state_words"][:].reshape(-1)
        ]
        state_shape = list(restart[f"{root}/state_words"].shape)
    return {
        "engine": engine,
        "schema_version": schema_version,
        "state_words": state_words,
        "state_shape": state_shape,
    }


def _read_bussi_restart_state(path: Path) -> dict[str, object]:
    rng = _read_typed_rng_restart_state(path, "bussi_thermostat")
    lambda_values = _h5_numeric_values(path, BUSSI_LAMBDA_DATASET)
    steps = _h5_numeric_values(path, "/particles/all/step")
    times = _h5_numeric_values(path, "/particles/all/time")
    if (
        rng["engine"] != "sponge.philox4x32-10.counter.v1"
        or rng["schema_version"] != 1
        or rng["state_shape"] != [1, 4]
        or any(value < 0 or value > 0xFFFFFFFF for value in rng["state_words"])
        or len(lambda_values) != 1
        or len(steps) != 1
        or len(times) != 1
        or not all(
            math.isfinite(value) for value in (*lambda_values, *steps, *times)
        )
    ):
        raise AssertionError(f"invalid Bussi restart state: {path}")
    return {
        "rng": rng,
        "lambda": lambda_values[0],
        "step": int(steps[0]),
        "time": times[0],
        "sha256": _sha256_file(path),
    }


def _require_finite_values(
    case: AbCase, label: str, values: Sequence[float]
) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{case.name} {label} is empty or non-finite")


def _compare_h5_frame_subset(
    case: AbCase,
    label: str,
    reference_path: Path,
    candidate_path: Path,
    datasets: Sequence[str],
    *,
    time_dataset: str,
    tolerance_overrides: dict[str, tuple[float, float]] | None = None,
    periodic_position_dataset: str | None = None,
    periodic_box_dataset: str = "/particles/all/box/edges/value",
) -> dict[str, object]:
    reference_times = _h5_numeric_values(reference_path, time_dataset)
    candidate_times = _h5_numeric_values(candidate_path, time_dataset)
    reference_indices = []
    for candidate_time in candidate_times:
        matches = [
            index
            for index, reference_time in enumerate(reference_times)
            if math.isclose(
                reference_time, candidate_time, rel_tol=0.0, abs_tol=1.0e-12
            )
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"{case.name} {label} time {candidate_time} has "
                f"{len(matches)} reference matches"
            )
        reference_indices.append(matches[0])

    compared = []
    with h5py.File(reference_path, "r") as reference_h5:
        for dataset in datasets:
            if dataset not in reference_h5:
                raise AssertionError(
                    f"{case.name} {label} reference is missing {dataset}"
                )
            reference_values = reference_h5[dataset][reference_indices]
            candidate_values = _h5_numeric_values(candidate_path, dataset)
            flattened_reference = [
                float(value) for value in reference_values.reshape(-1)
            ]
            _require_finite_values(
                case, f"{label} reference {dataset}", flattened_reference
            )
            _require_finite_values(
                case, f"{label} candidate {dataset}", candidate_values
            )
            if dataset.endswith("/step"):
                reference_origin = flattened_reference[0]
                candidate_origin = candidate_values[0]
                flattened_reference = [
                    value - reference_origin for value in flattened_reference
                ]
                candidate_values = [
                    value - candidate_origin for value in candidate_values
                ]
            relative_tolerance, absolute_tolerance = (
                tolerance_overrides or {}
            ).get(dataset, _deterministic_tolerance(dataset))
            if dataset == periodic_position_dataset:
                if periodic_box_dataset not in reference_h5:
                    raise AssertionError(
                        f"{case.name} {label} reference is missing "
                        f"{periodic_box_dataset} for periodic position comparison"
                    )
                reference_boxes = reference_h5[periodic_box_dataset][
                    reference_indices
                ]
                _assert_periodic_positions_close(
                    f"{case.name} {label} {dataset}",
                    flattened_reference,
                    candidate_values,
                    tuple(reference_values.shape),
                    [float(value) for value in reference_boxes.reshape(-1)],
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
            else:
                _assert_numeric_sequences_close(
                    f"{case.name} {label} {dataset}",
                    flattened_reference,
                    candidate_values,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
            compared.append(dataset)
    return {
        "reference_indices": reference_indices,
        "times": candidate_times,
        "datasets": compared,
        "step_comparison": "relative_to_branch_start",
        "periodic_position_dataset": periodic_position_dataset,
    }


def _compare_dynamic_mdout_suffix(
    case: AbCase,
    module_label: str,
    reference_dir: Path,
    continuation_dir: Path,
    *,
    skip_initial_row: bool = False,
    tolerance_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    reference = _read_mdout(reference_dir / "mdout.txt")
    continuation = _read_mdout(continuation_dir / "mdout.txt")
    columns = _require_matching_mdout_columns(
        reference, continuation, f"{case.name} {module_label} continuation"
    )
    continuation_rows = continuation["rows"]
    if skip_initial_row:
        continuation_rows = continuation_rows[1:]
    reference_rows = reference["rows"][-len(continuation_rows) :]
    for column in columns:
        reference_values = [row[column] for row in reference_rows]
        continuation_values = [row[column] for row in continuation_rows]
        _require_finite_values(
            case,
            f"{module_label} mdout reference suffix {column}",
            reference_values,
        )
        _require_finite_values(
            case,
            f"{module_label} mdout continuation {column}",
            continuation_values,
        )
        if column == "step":
            reference_origin = reference_values[0]
            continuation_origin = continuation_values[0]
            reference_values = [
                value - reference_origin for value in reference_values
            ]
            continuation_values = [
                value - continuation_origin for value in continuation_values
            ]
        relative_tolerance, absolute_tolerance = (
            tolerance_overrides or {}
        ).get(column, _deterministic_tolerance(column))
        _assert_numeric_sequences_close(
            f"{case.name} {module_label} mdout suffix {column}",
            reference_values,
            continuation_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    return {
        "rows": len(continuation_rows),
        "columns": columns,
        "step_comparison": "relative_to_branch_start",
        "skipped_initial_row": skip_initial_row,
        "tolerance_overrides": tolerance_overrides or {},
    }


def _prepare_bussi_continuation(
    source_dir: Path,
    destination: Path,
    checkpoint: Path,
    *,
    thermostat_seed: int,
) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination)
    for relative_path in (TRAJECTORY_REL, OBSERVABLE_REL, RESTART_REL):
        output = destination / relative_path
        if output.exists():
            output.unlink()
    producer_copy = destination / "output/producer.spgr.h5"
    shutil.copy2(checkpoint, producer_copy)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if _sha256_file(producer_copy) != checkpoint_sha256:
        raise AssertionError("Bussi producer checkpoint changed while copied")
    topology_path = source_dir / "topology.spgt.h5"
    _write_bussi_mdin(
        destination,
        input_route="bundled_dynamic",
        step_limit=2,
        thermostat_seed=thermostat_seed,
        topology_hash=_h5_scalar_text(topology_path, "/topology/topology_hash"),
        atom_order_hash=_h5_scalar_text(
            topology_path, "/topology/atom_order_hash"
        ),
    )
    return checkpoint_sha256


def _run_bussi_mutation_controls(
    case: AbCase,
    bundled_producer_dir: Path,
    checkpoint: Path,
    expected_restart: Path,
    mutation_root: Path,
) -> dict[str, object]:
    outcomes = {}
    for mutation, diagnostic in (
        (
            "corrupt_rng",
            "counter Philox restart state has an incompatible schema",
        ),
        ("missing_lambda", "Bussi thermostat restart state is missing"),
    ):
        destination = mutation_root / mutation
        _prepare_bussi_continuation(
            bundled_producer_dir,
            destination,
            checkpoint,
            thermostat_seed=13579,
        )
        producer_copy = destination / "output/producer.spgr.h5"
        with h5py.File(producer_copy, "r+") as restart:
            if mutation == "corrupt_rng":
                _replace_h5_string_dataset(
                    restart, BUSSI_RNG_ENGINE_DATASET, "invalid_rng_engine"
                )
            else:
                del restart[BUSSI_LAMBDA_DATASET]
        _refresh_restart_state_hash(producer_copy)
        _refresh_restart_state_hash(producer_copy)
        outcome = _run_sponge_process(destination, "mdin.spg.toml")
        if outcome.returncode == 0 or diagnostic not in (
            outcome.stdout + outcome.stderr
        ):
            raise AssertionError(
                f"{case.name} {mutation} was not rejected with {diagnostic!r}"
            )
        outcomes[mutation] = {
            "exit_code": outcome.returncode,
            "diagnostic": diagnostic,
        }

    alternate = mutation_root / "alternate_rng"
    _prepare_bussi_continuation(
        bundled_producer_dir,
        alternate,
        checkpoint,
        thermostat_seed=13579,
    )
    with h5py.File(alternate / "output/producer.spgr.h5", "r+") as restart:
        words = restart[BUSSI_RNG_STATE_WORDS_DATASET][:]
        words.reshape(-1)[0] = (int(words.reshape(-1)[0]) + 1) & 0xFFFFFFFF
        restart[BUSSI_RNG_STATE_WORDS_DATASET][:] = words
    _refresh_restart_state_hash(alternate / "output/producer.spgr.h5")
    _refresh_restart_state_hash(alternate / "output/producer.spgr.h5")
    _run_sponge(alternate, "mdin.spg.toml")
    expected_state = _read_bussi_restart_state(expected_restart)
    alternate_state = _read_bussi_restart_state(alternate / RESTART_REL)
    expected_velocity = _h5_numeric_values(
        expected_restart, "/particles/all/velocity/value"
    )
    alternate_velocity = _h5_numeric_values(
        alternate / RESTART_REL, "/particles/all/velocity/value"
    )
    if len(expected_velocity) != len(alternate_velocity):
        raise AssertionError(
            f"{case.name} alternate Bussi RNG velocity shape changed"
        )
    maximum_velocity_delta = max(
        abs(left - right)
        for left, right in zip(expected_velocity, alternate_velocity)
    )
    if (
        alternate_state["rng"] == expected_state["rng"]
        or maximum_velocity_delta <= 1.0e-5
    ):
        raise AssertionError(
            f"{case.name} alternate valid Bussi RNG state was not observable"
        )
    outcomes["alternate_rng"] = {
        "exit_code": 0,
        "final_rng_differs": True,
        "maximum_velocity_delta": maximum_velocity_delta,
    }
    return outcomes


def _run_bussi_dynamic_restart_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    setup_root = case_root / "setup"
    legacy_template, bundled_producer_dir = _prepare_normal_tip3p_pair(
        setup_root, 20260714
    )
    topology_path = bundled_producer_dir / "topology.spgt.h5"
    topology_hash = _h5_scalar_text(topology_path, "/topology/topology_hash")
    atom_order_hash = _h5_scalar_text(
        topology_path, "/topology/atom_order_hash"
    )
    reference_dir = case_root / "reference"
    legacy_checkpoint_dir = case_root / "legacy_checkpoint"
    for destination in (reference_dir, legacy_checkpoint_dir):
        if destination.exists():
            shutil.rmtree(destination)
    shutil.copytree(bundled_producer_dir, reference_dir)
    shutil.copytree(legacy_template, legacy_checkpoint_dir)

    _write_bussi_mdin(
        reference_dir,
        input_route="bundled_initial",
        step_limit=8,
        thermostat_seed=20260714,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    _write_bussi_mdin(
        legacy_checkpoint_dir,
        input_route="legacy",
        step_limit=6,
        thermostat_seed=20260714,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    _write_bussi_mdin(
        bundled_producer_dir,
        input_route="bundled_initial",
        step_limit=6,
        thermostat_seed=20260714,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    producer_metrics = {
        "reference": _run_sponge(reference_dir, "mdin.spg.toml"),
        "legacy_checkpoint": _run_sponge(
            legacy_checkpoint_dir, "mdin.spg.toml"
        ),
        "bundled_checkpoint": _run_sponge(
            bundled_producer_dir, "mdin.spg.toml"
        ),
    }

    particle_datasets = (
        "/particles/all/step",
        "/particles/all/time",
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/force/value",
        "/particles/all/box/edges/value",
    )
    prefix = {
        "legacy_reference": _compare_h5_frame_subset(
            case,
            "legacy reference/checkpoint prefix",
            reference_dir / TRAJECTORY_REL,
            legacy_checkpoint_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            tolerance_overrides=BUSSI_INPUT_ROUTE_TOLERANCES,
            periodic_position_dataset="/particles/all/position/value",
        ),
        "bundled_producer": _compare_h5_frame_subset(
            case,
            "legacy/bundled checkpoint prefix",
            legacy_checkpoint_dir / TRAJECTORY_REL,
            bundled_producer_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            tolerance_overrides=BUSSI_INPUT_ROUTE_TOLERANCES,
            periodic_position_dataset="/particles/all/position/value",
        ),
    }
    checkpoint_states = {
        "legacy": _read_bussi_restart_state(
            legacy_checkpoint_dir / RESTART_REL
        ),
        "bundled": _read_bussi_restart_state(
            bundled_producer_dir / RESTART_REL
        ),
    }
    if (
        checkpoint_states["legacy"]["rng"]
        != checkpoint_states["bundled"]["rng"]
    ):
        raise AssertionError(f"{case.name} checkpoint Bussi RNG state differs")
    _assert_numeric_sequences_close(
        f"{case.name} checkpoint Bussi lambda",
        [checkpoint_states["legacy"]["lambda"]],
        [checkpoint_states["bundled"]["lambda"]],
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-7,
    )

    continuation_dir = case_root / "continuation"
    checkpoint = bundled_producer_dir / RESTART_REL
    checkpoint_sha256 = _prepare_bussi_continuation(
        bundled_producer_dir,
        continuation_dir,
        checkpoint,
        thermostat_seed=13579,
    )
    continuation_metrics = _run_sponge(continuation_dir, "mdin.spg.toml")
    mdout = _compare_dynamic_mdout_suffix(
        case, "Bussi", reference_dir, continuation_dir
    )
    suffix = _compare_h5_frame_subset(
        case,
        "continuous/restarted suffix",
        reference_dir / TRAJECTORY_REL,
        continuation_dir / TRAJECTORY_REL,
        particle_datasets,
        time_dataset="/particles/all/time",
        tolerance_overrides=BUSSI_CONTINUATION_TOLERANCES,
        periodic_position_dataset="/particles/all/position/value",
    )
    payload_summary = {}
    for quantity in ("velocity", "force"):
        values = _h5_numeric_values(
            continuation_dir / TRAJECTORY_REL,
            f"/particles/all/{quantity}/value",
        )
        _require_finite_values(case, f"continuation {quantity}", values)
        maximum_magnitude = max(abs(value) for value in values)
        required_nonzero = quantity == "velocity"
        if required_nonzero and maximum_magnitude <= 1.0e-6:
            raise AssertionError(
                f"{case.name} continuation {quantity} is trivial"
            )
        payload_summary[quantity] = {
            "maximum_magnitude": maximum_magnitude,
            "required_nonzero": required_nonzero,
        }
    final_states = {
        "reference": _read_bussi_restart_state(reference_dir / RESTART_REL),
        "continuation": _read_bussi_restart_state(
            continuation_dir / RESTART_REL
        ),
    }
    if final_states["reference"]["rng"] != final_states["continuation"]["rng"]:
        raise AssertionError(f"{case.name} final Bussi RNG state differs")
    _assert_numeric_sequences_close(
        f"{case.name} final Bussi lambda",
        [final_states["reference"]["lambda"]],
        [final_states["continuation"]["lambda"]],
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-7,
    )
    reference_step_delta = (
        final_states["reference"]["step"] - checkpoint_states["bundled"]["step"]
    )
    continuation_step_delta = (
        final_states["continuation"]["step"]
        - checkpoint_states["bundled"]["step"]
    )
    if reference_step_delta != continuation_step_delta:
        raise AssertionError(
            f"{case.name} continuation step-count mismatch: "
            f"reference={reference_step_delta}, "
            f"continuation={continuation_step_delta}"
        )
    for dataset in (
        "/particles/all/time",
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
    ):
        reference_values = _h5_numeric_values(
            reference_dir / RESTART_REL, dataset
        )
        continuation_values = _h5_numeric_values(
            continuation_dir / RESTART_REL, dataset
        )
        relative_tolerance, absolute_tolerance = (
            BUSSI_CONTINUATION_TOLERANCES.get(
                dataset, _deterministic_tolerance(dataset)
            )
        )
        if dataset == "/particles/all/position/value":
            _assert_periodic_positions_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                _h5_dataset_shape(reference_dir / RESTART_REL, dataset),
                _h5_numeric_values(
                    reference_dir / RESTART_REL,
                    "/particles/all/box/edges/value",
                ),
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            _assert_numeric_sequences_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )

    mutations = _run_bussi_mutation_controls(
        case,
        bundled_producer_dir,
        checkpoint,
        continuation_dir / RESTART_REL,
        case_root / "mutations",
    )
    assertion = AssertionEvidence(
        assertion_id="restart_bussi_continuation_equivalence",
        evidence_level="E4",
        details={
            "method": "continuous_reference_vs_bundled_checkpoint_restart",
            "producer_branch": "bundled",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_states": checkpoint_states,
            "prefix": prefix,
            "mdout": mdout,
            "suffix": suffix,
            "payload_summary": payload_summary,
            "final_states": final_states,
            "step_progression": {
                "comparison": "completed_steps_after_checkpoint",
                "reference": reference_step_delta,
                "continuation": continuation_step_delta,
            },
            "continuation_tolerances": BUSSI_CONTINUATION_TOLERANCES,
            "thermostat_seed_overridden_by_restart": {
                "producer": 20260714,
                "continuation_mdin": 13579,
            },
            "mutations": mutations,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation_metrics,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "producer_metrics": producer_metrics,
        "continuation_metrics": continuation_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_branch": "bundled",
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B Bussi restart metrics: {metrics_path}")


def _write_pressure_barostat_mdin(
    case_dir: Path,
    *,
    input_route: str,
    step_limit: int,
    barostat_seed: int = 20260714,
    topology_hash: str = "",
    atom_order_hash: str = "",
) -> None:
    if input_route == "legacy":
        input_lines = [
            'default_in_file_prefix = "tip3p"',
            'velocity_in_file = "initial_velocity.txt"',
        ]
    elif input_route == "bundled_initial":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "restart.spgr.h5"',
            'input_h5_restart_load = "structural"',
        ]
    elif input_route == "bundled_dynamic":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "output/producer.spgr.h5"',
            'input_h5_restart_load = "dynamic"',
        ]
    else:
        raise AssertionError(
            f"unknown pressure barostat input route: {input_route}"
        )

    lines = [
        f'md_name = "bundled io ab pressure barostat {input_route}"',
        'mode = "npt"',
        f"step_limit = {step_limit}",
        "dt = 0.0001",
        "cutoff = 8.0",
        *input_lines,
        'thermostat = "berendsen_thermostat"',
        'thermostat_mode = "berendsen_thermostat"',
        "thermostat_tau = 0.1",
        "target_temperature = 300.0",
        'barostat = "andersen_barostat"',
        'barostat_mode = "andersen_barostat"',
        'barostat_isotropy = "isotropic"',
        "barostat_tau = 0.1",
        "barostat_update_interval = 1",
        f"barostat_seed = {barostat_seed}",
        "target_pressure = 1000.0",
        "print_pressure = true",
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'crd = "output/legacy.crd"',
        'box = "output/legacy.box"',
        'vel = "output/legacy.vel"',
        'frc = "output/legacy.frc"',
        'rst = "output/legacy_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 4",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
    ]
    if topology_hash and atom_order_hash:
        lines.extend(
            [
                f'output_h5_restart_topology_hash = "{topology_hash}"',
                f'output_h5_restart_atom_order_hash = "{atom_order_hash}"',
            ]
        )
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_pressure_barostat_restart_state(path: Path) -> dict[str, object]:
    rng = _read_typed_rng_restart_state(path, "pressure_based_barostat")
    g = _h5_numeric_values(path, PRESSURE_BAROSTAT_G_DATASET)
    steps = _h5_numeric_values(path, "/particles/all/step")
    times = _h5_numeric_values(path, "/particles/all/time")
    if (
        rng["engine"] != "sponge.philox4x32-10.counter.v1"
        or rng["schema_version"] != 1
        or rng["state_shape"] != [1, 4]
        or any(value < 0 or value > 0xFFFFFFFF for value in rng["state_words"])
        or len(g) != 6
        or len(steps) != 1
        or len(times) != 1
        or not all(math.isfinite(value) for value in (*g, *steps, *times))
    ):
        raise AssertionError(f"invalid pressure barostat restart state: {path}")
    return {
        "rng": rng,
        "g": g,
        "step": int(steps[0]),
        "time": times[0],
        "sha256": _sha256_file(path),
    }


def _box_volume(box: Sequence[float]) -> float:
    if len(box) != 9:
        raise AssertionError(
            f"box matrix must contain 9 values, got {len(box)}"
        )
    a, b, c, d, e, f, g, h, i = box
    return abs(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def _prepare_pressure_barostat_continuation(
    source_dir: Path, destination: Path, checkpoint: Path
) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination)
    for relative_path in (TRAJECTORY_REL, OBSERVABLE_REL, RESTART_REL):
        output = destination / relative_path
        if output.exists():
            output.unlink()
    producer_copy = destination / "output/producer.spgr.h5"
    shutil.copy2(checkpoint, producer_copy)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if _sha256_file(producer_copy) != checkpoint_sha256:
        raise AssertionError(
            "pressure barostat checkpoint changed while copied"
        )
    topology_path = source_dir / "topology.spgt.h5"
    _write_pressure_barostat_mdin(
        destination,
        input_route="bundled_dynamic",
        step_limit=2,
        barostat_seed=13579,
        topology_hash=_h5_scalar_text(topology_path, "/topology/topology_hash"),
        atom_order_hash=_h5_scalar_text(
            topology_path, "/topology/atom_order_hash"
        ),
    )
    return checkpoint_sha256


def _run_pressure_barostat_mutation_controls(
    case: AbCase,
    bundled_producer_dir: Path,
    checkpoint: Path,
    expected_restart: Path,
    mutation_root: Path,
) -> dict[str, object]:
    outcomes = {}
    for mutation, diagnostic in (
        (
            "corrupt_rng",
            "counter Philox restart state has an incompatible schema",
        ),
        (
            "missing_g",
            "restart does not contain pressure-based barostat dynamic state",
        ),
        ("owner_mismatch", "current barostat is not pressure-based"),
    ):
        destination = mutation_root / mutation
        _prepare_pressure_barostat_continuation(
            bundled_producer_dir, destination, checkpoint
        )
        producer_copy = destination / "output/producer.spgr.h5"
        if mutation == "owner_mismatch":
            mdin = destination / "mdin.spg.toml"
            text = mdin.read_text(encoding="utf-8").replace(
                "andersen_barostat", "monte_carlo_barostat"
            )
            mdin.write_text(text, encoding="utf-8")
        else:
            with h5py.File(producer_copy, "r+") as restart:
                if mutation == "corrupt_rng":
                    _replace_h5_string_dataset(
                        restart,
                        PRESSURE_BAROSTAT_RNG_ENGINE_DATASET,
                        "invalid_rng_engine",
                    )
                else:
                    del restart[PRESSURE_BAROSTAT_G_DATASET]
            _refresh_restart_state_hash(producer_copy)
        outcome = _run_sponge_process(destination, "mdin.spg.toml")
        if outcome.returncode == 0 or diagnostic not in (
            outcome.stdout + outcome.stderr
        ):
            raise AssertionError(
                f"{case.name} {mutation} was not rejected with {diagnostic!r}"
            )
        outcomes[mutation] = {
            "exit_code": outcome.returncode,
            "diagnostic": diagnostic,
        }

    expected_state = _read_pressure_barostat_restart_state(expected_restart)
    for mutation in ("swapped_g", "alternate_rng"):
        destination = mutation_root / mutation
        _prepare_pressure_barostat_continuation(
            bundled_producer_dir, destination, checkpoint
        )
        producer_copy = destination / "output/producer.spgr.h5"
        with h5py.File(producer_copy, "r+") as restart:
            if mutation == "swapped_g":
                values = restart[PRESSURE_BAROSTAT_G_DATASET][:]
                restart[PRESSURE_BAROSTAT_G_DATASET][:] = values[::-1]
            else:
                words = restart[PRESSURE_BAROSTAT_RNG_STATE_WORDS_DATASET][:]
                words.reshape(-1)[0] = (
                    int(words.reshape(-1)[0]) + 1
                ) & 0xFFFFFFFF
                restart[PRESSURE_BAROSTAT_RNG_STATE_WORDS_DATASET][:] = words
        _refresh_restart_state_hash(producer_copy)
        _refresh_restart_state_hash(producer_copy)
        _run_sponge(destination, "mdin.spg.toml")
        candidate_state = _read_pressure_barostat_restart_state(
            destination / RESTART_REL
        )
        expected_box = _h5_numeric_values(
            expected_restart, "/particles/all/box/edges/value"
        )
        candidate_box = _h5_numeric_values(
            destination / RESTART_REL, "/particles/all/box/edges/value"
        )
        if len(expected_box) != len(candidate_box):
            raise AssertionError(
                f"{case.name} {mutation} changed restart box shape"
            )
        maximum_box_delta = max(
            abs(left - right)
            for left, right in zip(expected_box, candidate_box)
        )
        rng_differs = candidate_state["rng"] != expected_state["rng"]
        g_delta = max(
            abs(left - right)
            for left, right in zip(expected_state["g"], candidate_state["g"])
        )
        if maximum_box_delta <= 1.0e-7 or g_delta <= 1.0e-7:
            raise AssertionError(
                f"{case.name} {mutation} was not behaviorally observable"
            )
        if mutation == "alternate_rng" and not rng_differs:
            raise AssertionError(
                f"{case.name} alternate barostat RNG did not change final state"
            )
        outcomes[mutation] = {
            "exit_code": 0,
            "final_rng_differs": rng_differs,
            "maximum_g_delta": g_delta,
            "maximum_box_delta": maximum_box_delta,
        }
    return outcomes


def _run_pressure_barostat_dynamic_restart_case(
    case: AbCase, contracts
) -> None:
    case_root = _output_root() / case.name
    setup_root = case_root / "setup"
    legacy_template, bundled_producer_dir = _prepare_normal_tip3p_pair(
        setup_root, 20260714
    )
    topology_path = bundled_producer_dir / "topology.spgt.h5"
    topology_hash = _h5_scalar_text(topology_path, "/topology/topology_hash")
    atom_order_hash = _h5_scalar_text(
        topology_path, "/topology/atom_order_hash"
    )
    reference_dir = case_root / "reference"
    legacy_checkpoint_dir = case_root / "legacy_checkpoint"
    for destination in (reference_dir, legacy_checkpoint_dir):
        if destination.exists():
            shutil.rmtree(destination)
    shutil.copytree(bundled_producer_dir, reference_dir)
    shutil.copytree(legacy_template, legacy_checkpoint_dir)

    _write_pressure_barostat_mdin(
        reference_dir,
        input_route="bundled_initial",
        step_limit=8,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    _write_pressure_barostat_mdin(
        legacy_checkpoint_dir,
        input_route="legacy",
        step_limit=6,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    _write_pressure_barostat_mdin(
        bundled_producer_dir,
        input_route="bundled_initial",
        step_limit=6,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    producer_metrics = {
        "reference": _run_sponge(reference_dir, "mdin.spg.toml"),
        "legacy_checkpoint": _run_sponge(
            legacy_checkpoint_dir, "mdin.spg.toml"
        ),
        "bundled_checkpoint": _run_sponge(
            bundled_producer_dir, "mdin.spg.toml"
        ),
    }

    particle_datasets = (
        "/particles/all/step",
        "/particles/all/time",
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/force/value",
        "/particles/all/box/edges/value",
    )
    prefix = {
        "legacy_reference": _compare_h5_frame_subset(
            case,
            "legacy reference/checkpoint prefix",
            reference_dir / TRAJECTORY_REL,
            legacy_checkpoint_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            tolerance_overrides=PRESSURE_BAROSTAT_INPUT_ROUTE_TOLERANCES,
            periodic_position_dataset="/particles/all/position/value",
        ),
        "bundled_producer": _compare_h5_frame_subset(
            case,
            "legacy/bundled checkpoint prefix",
            legacy_checkpoint_dir / TRAJECTORY_REL,
            bundled_producer_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            tolerance_overrides=PRESSURE_BAROSTAT_INPUT_ROUTE_TOLERANCES,
            periodic_position_dataset="/particles/all/position/value",
        ),
    }
    checkpoint_states = {
        "legacy": _read_pressure_barostat_restart_state(
            legacy_checkpoint_dir / RESTART_REL
        ),
        "bundled": _read_pressure_barostat_restart_state(
            bundled_producer_dir / RESTART_REL
        ),
    }
    if (
        checkpoint_states["legacy"]["rng"]
        != checkpoint_states["bundled"]["rng"]
    ):
        raise AssertionError(
            f"{case.name} checkpoint pressure barostat RNG state differs"
        )
    _assert_numeric_sequences_close(
        f"{case.name} checkpoint pressure barostat g",
        checkpoint_states["legacy"]["g"],
        checkpoint_states["bundled"]["g"],
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-8,
    )

    continuation_dir = case_root / "continuation"
    checkpoint = bundled_producer_dir / RESTART_REL
    checkpoint_sha256 = _prepare_pressure_barostat_continuation(
        bundled_producer_dir, continuation_dir, checkpoint
    )
    continuation_metrics = _run_sponge(continuation_dir, "mdin.spg.toml")
    mdout = _compare_dynamic_mdout_suffix(
        case,
        "pressure barostat",
        reference_dir,
        continuation_dir,
        skip_initial_row=True,
        tolerance_overrides=PRESSURE_BAROSTAT_MDOUT_TOLERANCES,
    )
    required_pressure_columns = {"pressure", "Pxx", "Pyy", "Pzz"}
    if not required_pressure_columns.issubset(mdout["columns"]):
        raise AssertionError(
            f"{case.name} pressure tensor columns are incomplete"
        )
    suffix = _compare_h5_frame_subset(
        case,
        "continuous/restarted pressure suffix",
        reference_dir / TRAJECTORY_REL,
        continuation_dir / TRAJECTORY_REL,
        particle_datasets,
        time_dataset="/particles/all/time",
        tolerance_overrides=PRESSURE_BAROSTAT_CONTINUATION_TOLERANCES,
        periodic_position_dataset="/particles/all/position/value",
    )
    final_states = {
        "reference": _read_pressure_barostat_restart_state(
            reference_dir / RESTART_REL
        ),
        "continuation": _read_pressure_barostat_restart_state(
            continuation_dir / RESTART_REL
        ),
    }
    if final_states["reference"]["rng"] != final_states["continuation"]["rng"]:
        raise AssertionError(f"{case.name} final pressure barostat RNG differs")
    _assert_numeric_sequences_close(
        f"{case.name} final pressure barostat g",
        final_states["reference"]["g"],
        final_states["continuation"]["g"],
        relative_tolerance=PRESSURE_BAROSTAT_G_TOLERANCE[0],
        absolute_tolerance=PRESSURE_BAROSTAT_G_TOLERANCE[1],
    )
    reference_step_delta = (
        final_states["reference"]["step"] - checkpoint_states["bundled"]["step"]
    )
    continuation_step_delta = (
        final_states["continuation"]["step"]
        - checkpoint_states["bundled"]["step"]
    )
    if reference_step_delta != continuation_step_delta:
        raise AssertionError(
            f"{case.name} continuation step-count mismatch: "
            f"reference={reference_step_delta}, "
            f"continuation={continuation_step_delta}"
        )
    for dataset in (
        "/particles/all/time",
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
    ):
        reference_values = _h5_numeric_values(
            reference_dir / RESTART_REL, dataset
        )
        continuation_values = _h5_numeric_values(
            continuation_dir / RESTART_REL, dataset
        )
        _require_finite_values(
            case, f"final reference {dataset}", reference_values
        )
        _require_finite_values(
            case, f"final continuation {dataset}", continuation_values
        )
        relative_tolerance, absolute_tolerance = (
            PRESSURE_BAROSTAT_CONTINUATION_TOLERANCES.get(
                dataset, _deterministic_tolerance(dataset)
            )
        )
        if dataset == "/particles/all/position/value":
            _assert_periodic_positions_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                _h5_dataset_shape(reference_dir / RESTART_REL, dataset),
                _h5_numeric_values(
                    reference_dir / RESTART_REL,
                    "/particles/all/box/edges/value",
                ),
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            _assert_numeric_sequences_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )

    checkpoint_box = _h5_numeric_values(
        checkpoint, "/particles/all/box/edges/value"
    )
    final_box = _h5_numeric_values(
        continuation_dir / RESTART_REL, "/particles/all/box/edges/value"
    )
    box_delta = max(
        abs(left - right) for left, right in zip(checkpoint_box, final_box)
    )
    checkpoint_volume = _box_volume(checkpoint_box)
    final_volume = _box_volume(final_box)
    volume_delta = abs(final_volume - checkpoint_volume)
    g_delta = max(
        abs(left - right)
        for left, right in zip(
            checkpoint_states["bundled"]["g"], final_states["continuation"]["g"]
        )
    )
    if box_delta <= 1.0e-7 or volume_delta <= 1.0e-5 or g_delta <= 1.0e-7:
        raise AssertionError(
            f"{case.name} barostat box/g evolution is not observable"
        )
    evolution = {
        "maximum_box_delta": box_delta,
        "checkpoint_volume": checkpoint_volume,
        "final_volume": final_volume,
        "volume_delta": volume_delta,
        "maximum_g_delta": g_delta,
        "g_component_count": len(final_states["continuation"]["g"]),
        "nonzero_g_component_count": sum(
            abs(value) > 1.0e-12 for value in final_states["continuation"]["g"]
        ),
    }
    if (
        evolution["g_component_count"] != 6
        or evolution["nonzero_g_component_count"] == 0
    ):
        raise AssertionError(
            f"{case.name} does not exercise the six-value g state"
        )

    mutations = _run_pressure_barostat_mutation_controls(
        case,
        bundled_producer_dir,
        checkpoint,
        continuation_dir / RESTART_REL,
        case_root / "mutations",
    )
    assertion = AssertionEvidence(
        assertion_id="restart_pressure_barostat_continuation_equivalence",
        evidence_level="E4",
        details={
            "method": "continuous_reference_vs_bundled_checkpoint_restart",
            "producer_branch": "bundled",
            "algorithm": "andersen_barostat",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_states": checkpoint_states,
            "prefix": prefix,
            "mdout": mdout,
            "suffix": suffix,
            "final_states": final_states,
            "step_progression": {
                "comparison": "completed_steps_after_checkpoint",
                "reference": reference_step_delta,
                "continuation": continuation_step_delta,
            },
            "evolution": evolution,
            "continuation_tolerances": {
                "g": PRESSURE_BAROSTAT_G_TOLERANCE,
                "mdout": PRESSURE_BAROSTAT_MDOUT_TOLERANCES,
            },
            "barostat_seed_overridden_by_restart": {
                "producer": 20260714,
                "continuation_mdin": 13579,
            },
            "mutations": mutations,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation_metrics,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "producer_metrics": producer_metrics,
        "continuation_metrics": continuation_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_branch": "bundled",
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B pressure restart metrics: {metrics_path}")


def _portable_stochastic_spec(case: AbCase) -> dict[str, object]:
    if "middle_langevin" in case.name:
        return {
            "module": "middle_langevin",
            "assertion_id": "restart_middle_langevin_continuation_equivalence",
            "seed": 20260723,
            "continuation_seed": 13579,
            "engine": "sponge.philox4x32-10.counter.v1",
        }
    if "monte_carlo_barostat" in case.name:
        return {
            "module": "monte_carlo_barostat",
            "assertion_id": (
                "restart_monte_carlo_barostat_continuation_equivalence"
            ),
            "seed": 20260723,
            "continuation_seed": 13579,
            "engine": "sponge.splitmix64.v1",
        }
    if "andersen" in case.name:
        return {
            "module": "andersen",
            "assertion_id": "restart_andersen_continuation_equivalence",
            "seed": 20260723,
            "continuation_seed": 13579,
            "engine": "sponge.philox4x32-10.counter.v1",
        }
    raise AssertionError(f"unknown portable stochastic case: {case.name}")


def _write_portable_stochastic_mdin(
    case_dir: Path,
    *,
    module: str,
    input_route: str,
    step_limit: int,
    seed: int,
    topology_hash: str,
    atom_order_hash: str,
) -> None:
    if input_route == "legacy":
        input_lines = [
            'default_in_file_prefix = "tip3p"',
            'velocity_in_file = "initial_velocity.txt"',
        ]
    elif input_route == "bundled_initial":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "restart.spgr.h5"',
            'input_h5_restart_load = "structural"',
        ]
    elif input_route == "bundled_dynamic":
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "output/producer.spgr.h5"',
            'input_h5_restart_load = "dynamic"',
        ]
    else:
        raise AssertionError(
            f"unknown portable stochastic input route: {input_route}"
        )

    if module in {"middle_langevin", "andersen"}:
        module_lines = [
            'mode = "nvt"',
            f'thermostat = "{module}"',
            f'thermostat_mode = "{module}"',
            f"thermostat_seed = {seed}",
            "thermostat_tau = 0.001",
            "target_temperature = 300.0",
        ]
    elif module == "monte_carlo_barostat":
        module_lines = [
            'mode = "npt"',
            'thermostat = "berendsen_thermostat"',
            'thermostat_mode = "berendsen_thermostat"',
            "thermostat_tau = 0.1",
            "target_temperature = 300.0",
            'barostat_mode = "monte_carlo_barostat"',
            "target_pressure = 1000.0",
            "print_pressure = true",
        ]
    else:
        raise AssertionError(f"unknown portable stochastic module: {module}")

    lines = [
        f'md_name = "bundled io ab portable stochastic {module} {input_route}"',
        *module_lines,
        f"step_limit = {step_limit}",
        "dt = 0.001",
        "cutoff = 8.0",
        *input_lines,
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'crd = "output/legacy.crd"',
        'box = "output/legacy.box"',
        'vel = "output/legacy.vel"',
        'frc = "output/legacy.frc"',
        'rst = "output/legacy_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 4",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        f'output_h5_restart_topology_hash = "{topology_hash}"',
        f'output_h5_restart_atom_order_hash = "{atom_order_hash}"',
    ]
    if module == "monte_carlo_barostat":
        lines.extend(
            [
                "",
                "[barostat.monte_carlo]",
                f"seed = {seed}",
                "update_interval = 1",
                "check_interval = 10",
                'couple_dimension = "XYZ"',
            ]
        )
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_portable_stochastic_restart_state(
    path: Path, module: str
) -> dict[str, object]:
    root = f"/parameters/restart/rng_state/{module}"
    with h5py.File(path, "r") as restart:
        required = {
            f"{root}/engine",
            f"{root}/schema_version",
            f"{root}/state_words",
            "/particles/all/step",
            "/particles/all/time",
        }
        missing = sorted(item for item in required if item not in restart)
        if missing:
            raise AssertionError(
                f"portable stochastic restart is missing {missing}: {path}"
            )
        engine_value = restart[f"{root}/engine"][()]
        engine = (
            engine_value.decode("utf-8")
            if isinstance(engine_value, bytes)
            else str(engine_value)
        )
        state = {
            "engine": engine,
            "schema_version": int(
                restart[f"{root}/schema_version"][:].reshape(-1)[0]
            ),
            "state_words": [
                int(value)
                for value in restart[f"{root}/state_words"][:].reshape(-1)
            ],
            "state_shape": list(restart[f"{root}/state_words"].shape),
            "step": int(restart["/particles/all/step"][:].reshape(-1)[0]),
            "time": float(restart["/particles/all/time"][:].reshape(-1)[0]),
        }
        if module == "monte_carlo_barostat":
            barostat_root = f"/parameters/restart/barostat/{module}"
            for field in (
                "delta_box_length_max",
                "accept_rate",
                "total_count_int64",
                "accept_count_int64",
            ):
                dataset = f"{barostat_root}/{field}"
                if dataset not in restart:
                    raise AssertionError(
                        f"Monte Carlo restart is missing {dataset}: {path}"
                    )
                values = restart[dataset][:].reshape(-1)
                state[field] = [
                    int(value) if field.endswith("int64") else float(value)
                    for value in values
                ]
    return state


def _assert_portable_stochastic_states_equal(
    case: AbCase,
    label: str,
    left: dict[str, object],
    right: dict[str, object],
    module: str,
) -> None:
    for field in ("engine", "schema_version", "state_words", "state_shape"):
        if left[field] != right[field]:
            raise AssertionError(
                f"{case.name} {label} typed RNG {field} differs"
            )
    if module == "monte_carlo_barostat":
        for field in ("total_count_int64", "accept_count_int64"):
            if left[field] != right[field]:
                raise AssertionError(
                    f"{case.name} {label} Monte Carlo {field} differs"
                )
        for field in ("delta_box_length_max", "accept_rate"):
            _assert_numeric_sequences_close(
                f"{case.name} {label} Monte Carlo {field}",
                left[field],
                right[field],
                relative_tolerance=1.0e-6,
                absolute_tolerance=1.0e-7,
            )


def _prepare_portable_stochastic_continuation(
    source_dir: Path,
    destination: Path,
    checkpoint: Path,
    *,
    module: str,
    seed: int,
    step_limit: int,
    topology_hash: str,
    atom_order_hash: str,
) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination)
    for relative_path in (TRAJECTORY_REL, OBSERVABLE_REL, RESTART_REL):
        output = destination / relative_path
        if output.exists():
            output.unlink()
    producer_copy = destination / "output/producer.spgr.h5"
    shutil.copy2(checkpoint, producer_copy)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if _sha256_file(producer_copy) != checkpoint_sha256:
        raise AssertionError(
            f"{module} producer checkpoint changed while copied"
        )
    _write_portable_stochastic_mdin(
        destination,
        module=module,
        input_route="bundled_dynamic",
        step_limit=step_limit,
        seed=seed,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    return checkpoint_sha256


def _run_portable_stochastic_restart_case(case: AbCase, contracts) -> None:
    spec = _portable_stochastic_spec(case)
    module = str(spec["module"])
    seed = int(spec["seed"])
    continuation_seed = int(spec["continuation_seed"])
    assertion_id = str(spec["assertion_id"])
    case_root = _output_root() / case.name
    if case_root.exists():
        shutil.rmtree(case_root)
    legacy_template, bundled_checkpoint_dir = _prepare_normal_tip3p_pair(
        case_root / "setup", seed
    )
    topology_path = bundled_checkpoint_dir / "topology.spgt.h5"
    topology_hash = _h5_scalar_text(topology_path, "/topology/topology_hash")
    atom_order_hash = _h5_scalar_text(
        topology_path, "/topology/atom_order_hash"
    )
    reference_dir = case_root / "reference"
    legacy_checkpoint_dir = case_root / "legacy_checkpoint"
    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    shutil.copytree(bundled_checkpoint_dir, reference_dir)
    if legacy_checkpoint_dir.exists():
        shutil.rmtree(legacy_checkpoint_dir)
    shutil.copytree(legacy_template, legacy_checkpoint_dir)

    reference_steps = 4
    checkpoint_steps = 2
    continuation_steps = reference_steps - checkpoint_steps
    for destination, route, steps in (
        (reference_dir, "bundled_initial", reference_steps),
        (legacy_checkpoint_dir, "legacy", checkpoint_steps),
        (bundled_checkpoint_dir, "bundled_initial", checkpoint_steps),
    ):
        _write_portable_stochastic_mdin(
            destination,
            module=module,
            input_route=route,
            step_limit=steps,
            seed=seed,
            topology_hash=topology_hash,
            atom_order_hash=atom_order_hash,
        )
    producer_metrics = {
        "reference": _run_sponge(reference_dir, "mdin.spg.toml"),
        "legacy_checkpoint": _run_sponge(
            legacy_checkpoint_dir, "mdin.spg.toml"
        ),
        "bundled_checkpoint": _run_sponge(
            bundled_checkpoint_dir, "mdin.spg.toml"
        ),
    }

    particle_datasets = (
        "/particles/all/step",
        "/particles/all/time",
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
    )
    prefix = {
        "bundled_reference": _compare_h5_frame_subset(
            case,
            "bundled reference/checkpoint prefix",
            reference_dir / TRAJECTORY_REL,
            bundled_checkpoint_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            periodic_position_dataset="/particles/all/position/value",
        ),
        "legacy_producer": _compare_h5_frame_subset(
            case,
            "legacy/bundled checkpoint prefix",
            legacy_checkpoint_dir / TRAJECTORY_REL,
            bundled_checkpoint_dir / TRAJECTORY_REL,
            particle_datasets,
            time_dataset="/particles/all/time",
            periodic_position_dataset="/particles/all/position/value",
        ),
    }
    checkpoint_states = {
        "legacy": _read_portable_stochastic_restart_state(
            legacy_checkpoint_dir / RESTART_REL, module
        ),
        "bundled": _read_portable_stochastic_restart_state(
            bundled_checkpoint_dir / RESTART_REL, module
        ),
    }
    _assert_portable_stochastic_states_equal(
        case,
        "legacy/bundled checkpoint",
        checkpoint_states["legacy"],
        checkpoint_states["bundled"],
        module,
    )

    continuation_dir = case_root / "continuation"
    checkpoint = bundled_checkpoint_dir / RESTART_REL
    checkpoint_sha256 = _prepare_portable_stochastic_continuation(
        bundled_checkpoint_dir,
        continuation_dir,
        checkpoint,
        module=module,
        seed=continuation_seed,
        step_limit=continuation_steps,
        topology_hash=topology_hash,
        atom_order_hash=atom_order_hash,
    )
    continuation_metrics = _run_sponge(continuation_dir, "mdin.spg.toml")
    mdout = _compare_dynamic_mdout_suffix(
        case, module, reference_dir, continuation_dir
    )
    continuation_tolerances = (
        MONTE_CARLO_BAROSTAT_CONTINUATION_TOLERANCES
        if module == "monte_carlo_barostat"
        else {}
    )
    suffix = _compare_h5_frame_subset(
        case,
        "continuous/restarted suffix",
        reference_dir / TRAJECTORY_REL,
        continuation_dir / TRAJECTORY_REL,
        particle_datasets,
        time_dataset="/particles/all/time",
        tolerance_overrides=continuation_tolerances,
        periodic_position_dataset="/particles/all/position/value",
    )
    final_states = {
        "reference": _read_portable_stochastic_restart_state(
            reference_dir / RESTART_REL, module
        ),
        "continuation": _read_portable_stochastic_restart_state(
            continuation_dir / RESTART_REL, module
        ),
    }
    _assert_portable_stochastic_states_equal(
        case,
        "continuous/restarted final",
        final_states["reference"],
        final_states["continuation"],
        module,
    )
    if final_states["reference"]["engine"] != spec["engine"]:
        raise AssertionError(
            f"{case.name} emitted unexpected RNG engine "
            f"{final_states['reference']['engine']!r}"
        )
    if (
        final_states["reference"]["time"]
        != final_states["continuation"]["time"]
    ):
        raise AssertionError(f"{case.name} continuation time differs")
    for dataset in (
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
    ):
        relative_tolerance, absolute_tolerance = continuation_tolerances.get(
            dataset, _deterministic_tolerance(dataset)
        )
        reference_values = _h5_numeric_values(
            reference_dir / RESTART_REL, dataset
        )
        continuation_values = _h5_numeric_values(
            continuation_dir / RESTART_REL, dataset
        )
        if dataset == "/particles/all/position/value":
            with h5py.File(reference_dir / RESTART_REL, "r") as restart:
                shape = tuple(restart[dataset].shape)
                boxes = [
                    float(value)
                    for value in restart["/particles/all/box/edges/value"][
                        ...
                    ].reshape(-1)
                ]
            _assert_periodic_positions_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                shape,
                boxes,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            _assert_numeric_sequences_close(
                f"{case.name} final restart {dataset}",
                reference_values,
                continuation_values,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )

    assertion = AssertionEvidence(
        assertion_id=assertion_id,
        evidence_level="E4",
        details={
            "method": "continuous_reference_vs_bundled_checkpoint_restart",
            "producer_branches": ["legacy", "bundled"],
            "module": module,
            "engine": spec["engine"],
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_states": checkpoint_states,
            "prefix": prefix,
            "mdout": mdout,
            "suffix": suffix,
            "final_states": final_states,
            "configured_seed_overridden_by_restart": {
                "producer": seed,
                "continuation_mdin": continuation_seed,
            },
            "continuation_tolerances": continuation_tolerances,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation_metrics,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "producer_metrics": producer_metrics,
        "continuation_metrics": continuation_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_branch": "bundled",
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B {module} restart metrics: {metrics_path}")


def _run_nhc_dynamic_restart_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    producer_dir, bundled_template = _prepare_normal_tip3p_pair(
        case_root / "producer_setup", 20260709
    )
    _write_nhc_producer_mdin(producer_dir)
    producer_metrics = _run_sponge(producer_dir, "mdin.spg.toml")
    producer_state = _validate_nhc_producer_state(case, producer_dir)

    continuation_root = case_root / "continuations"
    continuation_dirs = {}
    continuation_metrics = {}
    for branch in ("legacy", "bundled"):
        destination = continuation_root / branch
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = producer_dir if branch == "legacy" else bundled_template
        shutil.copytree(source, destination)
        (destination / "output").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            producer_dir / RESTART_REL,
            destination / "output/producer.spgr.h5",
        )
        _write_nhc_continuation_mdin(destination, branch)
        continuation_dirs[branch] = destination
        continuation_metrics[branch] = _run_sponge(destination, "mdin.spg.toml")

    route_evidence = _validate_nhc_continuation_routes(continuation_dirs)
    comparison = _compare_nhc_dynamic_continuations(case, continuation_dirs)
    assertion = AssertionEvidence(
        assertion_id="restart_dynamic_continuation_equivalence",
        evidence_level="E4",
        details={
            "method": "one_checkpoint_forked_to_legacy_and_h5_dynamic_load",
            "producer": producer_state,
            "routes": route_evidence,
            "continuation": comparison,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation_metrics,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "restart_load_policy": case.restart_load_policy,
        "producer_metrics": producer_metrics,
        "continuation_metrics": continuation_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_count": 1,
            "continuation_branches": ["legacy", "bundled"],
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B NHC restart metrics: {metrics_path}")


def _write_nhc_producer_mdin(case_dir: Path) -> None:
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    lines = [
        'md_name = "bundled io ab nhc restart producer"',
        'mode = "nvt"',
        "step_limit = 10",
        "dt = 0.002",
        "cutoff = 8.0",
        'thermostat = "nose_hoover_chain"',
        "thermostat_tau = 0.2",
        "target_temperature = 300.0",
        'default_in_file_prefix = "tip3p"',
        'velocity_in_file = "initial_velocity.txt"',
        'constrain_mode = "SETTLE"',
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'rst = "output/legacy_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 4",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        "",
        "[nose_hoover_chain]",
        "length = 3",
        'restart_output = "output/legacy_nhc_restart.txt"',
        'crd = "output/producer_nhc.crd"',
        'vel = "output/producer_nhc.vel"',
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_nhc_continuation_mdin(case_dir: Path, branch: str) -> None:
    if branch not in {"legacy", "bundled"}:
        raise AssertionError(f"unknown NHC continuation branch: {branch}")
    restart_inputs = (
        [
            'coordinate_in_file = "output/legacy_restart_coordinate.txt"',
            'velocity_in_file = "output/legacy_restart_velocity.txt"',
        ]
        if branch == "legacy"
        else [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "output/producer.spgr.h5"',
            'input_h5_restart_load = "dynamic"',
        ]
    )
    topology_inputs = (
        ['default_in_file_prefix = "tip3p"'] if branch == "legacy" else []
    )
    nhc_restart_input = (
        ['restart_input = "output/legacy_nhc_restart.txt"']
        if branch == "legacy"
        else []
    )
    lines = [
        f'md_name = "bundled io ab nhc {branch} continuation"',
        'mode = "nvt"',
        "step_limit = 2",
        "dt = 0.002",
        "cutoff = 8.0",
        'thermostat = "nose_hoover_chain"',
        "thermostat_tau = 0.2",
        "target_temperature = 300.0",
        *topology_inputs,
        'constrain_mode = "SETTLE"',
        *restart_inputs,
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'crd = "output/continuation.crd"',
        'box = "output/continuation.box"',
        'vel = "output/continuation.vel"',
        'frc = "output/continuation.frc"',
        'rst = "output/continuation_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 4",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        "",
        "[nose_hoover_chain]",
        "length = 3",
        *nhc_restart_input,
        'restart_output = "output/continuation_nhc_restart.txt"',
        'crd = "output/continuation_nhc.crd"',
        'vel = "output/continuation_nhc.vel"',
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_nhc_text_rows(path: Path, chain_length: int) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != chain_length:
            raise AssertionError(
                f"NHC row width differs from chain length in {path}: {values}"
            )
        rows.append(values)
    if not rows:
        raise AssertionError(f"NHC text output is empty: {path}")
    return rows


def _read_nhc_restart_pairs(path: Path, chain_length: int) -> list[float]:
    rows = _read_nhc_text_rows(path, 2)
    if len(rows) != chain_length:
        raise AssertionError(
            f"NHC restart row count differs from chain length: {path}"
        )
    return [value for row in rows for value in row]


def _assert_nhc_text_matches_h5(
    label: str,
    text_path: Path,
    h5_path: Path,
    dataset: str,
    chain_length: int,
) -> dict[str, object]:
    rows = _read_nhc_text_rows(text_path, chain_length)
    text_values = [value for row in rows for value in row]
    h5_values = _h5_numeric_values(h5_path, dataset)
    _assert_numeric_sequences_close(
        label,
        text_values,
        h5_values,
        relative_tolerance=1.0e-6,
        absolute_tolerance=6.0e-7,
    )
    return {"frame_count": len(rows), "value_count": len(text_values)}


def _validate_nhc_producer_state(
    case: AbCase, producer_dir: Path
) -> dict[str, object]:
    restart_path = producer_dir / RESTART_REL
    nhc_text_path = producer_dir / "output/legacy_nhc_restart.txt"
    pairs = _read_nhc_restart_pairs(nhc_text_path, 3)
    h5_pairs = _h5_numeric_values(restart_path, NHC_RESTART_DATASET)
    _assert_numeric_sequences_close(
        f"{case.name} producer legacy/H5 NHC restart",
        pairs,
        h5_pairs,
        relative_tolerance=1.0e-6,
        absolute_tolerance=6.0e-7,
    )
    maximum_abs_state = max(abs(value) for value in h5_pairs)
    if maximum_abs_state <= 1.0e-8:
        raise AssertionError(f"{case.name} producer NHC restart is trivial")
    structural = _validate_restart_legacy_coexistence(
        case, producer_dir, restart_path
    )
    mode = _h5_string_values(
        restart_path, "/parameters/restart/integrator_state/mode"
    )
    if mode != ["nvt"]:
        raise AssertionError(
            f"{case.name} producer restart mode is not NVT: {mode}"
        )
    return {
        "producer_count": 1,
        "chain_length": 3,
        "maximum_abs_nhc_state": maximum_abs_state,
        "legacy_h5_nhc_tolerance": 6.0e-7,
        "structural_state": structural,
    }


def _validate_nhc_continuation_routes(
    continuation_dirs: dict[str, Path],
) -> dict[str, object]:
    legacy_text = (continuation_dirs["legacy"] / "mdin.spg.toml").read_text(
        encoding="utf-8"
    )
    bundled_text = (continuation_dirs["bundled"] / "mdin.spg.toml").read_text(
        encoding="utf-8"
    )
    legacy_required = {
        "coordinate_in_file",
        "velocity_in_file",
        "restart_input",
    }
    bundled_required = {
        "input_h5_topology_path",
        "input_h5_protocol_path",
        "input_h5_restart_path",
        "input_h5_restart_load",
    }
    if not all(_has_key_line(legacy_text, key) for key in legacy_required):
        raise AssertionError(
            "legacy NHC continuation restart route is incomplete"
        )
    if any(_has_key_line(legacy_text, key) for key in bundled_required):
        raise AssertionError(
            "legacy NHC continuation retained an H5 restart route"
        )
    if not all(_has_key_line(bundled_text, key) for key in bundled_required):
        raise AssertionError(
            "bundled NHC continuation restart route is incomplete"
        )
    if any(_has_key_line(bundled_text, key) for key in legacy_required):
        raise AssertionError(
            "bundled NHC continuation retained a legacy restart route"
        )
    return {
        "legacy": sorted(legacy_required),
        "bundled": sorted(bundled_required),
        "same_producer_checkpoint": True,
    }


def _compare_nhc_dynamic_continuations(
    case: AbCase, continuation_dirs: dict[str, Path]
) -> dict[str, object]:
    legacy_dir = continuation_dirs["legacy"]
    bundled_dir = continuation_dirs["bundled"]
    mdout = {
        branch: _read_mdout(directory / "mdout.txt")
        for branch, directory in continuation_dirs.items()
    }
    columns = _require_matching_mdout_columns(
        mdout["legacy"], mdout["bundled"], f"{case.name} continuation"
    )
    for column in columns:
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            column
        )
        _assert_numeric_sequences_close(
            f"{case.name} continuation mdout {column}",
            [row[column] for row in mdout["legacy"]["rows"]],
            [row[column] for row in mdout["bundled"]["rows"]],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )

    semantic_datasets = {
        "trajectory": (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/force/value",
            "/particles/all/box/edges/value",
            f"{NHC_OBSERVABLE_ROOT}/step",
            f"{NHC_OBSERVABLE_ROOT}/time",
            f"{NHC_OBSERVABLE_ROOT}/coordinate/value",
            f"{NHC_OBSERVABLE_ROOT}/velocity/value",
        ),
        "restart": (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/box/edges/value",
            NHC_RESTART_DATASET,
        ),
    }
    files = {
        "legacy": _output_h5_files(case, legacy_dir),
        "bundled": _output_h5_files(case, bundled_dir),
    }
    compared = {}
    for family, datasets in semantic_datasets.items():
        left_path = files["legacy"][family]
        right_path = files["bundled"][family]
        for dataset in datasets:
            left_values = _h5_numeric_values(left_path, dataset)
            right_values = _h5_numeric_values(right_path, dataset)
            _assert_matching_numeric_shape(
                f"{case.name} {family}:{dataset}",
                left_path,
                right_path,
                dataset,
                left_values,
                right_values,
            )
            if dataset == NHC_RESTART_DATASET:
                for offset, quantity in ((0, "position"), (1, "velocity")):
                    relative_tolerance, absolute_tolerance = (
                        _deterministic_tolerance(quantity)
                    )
                    _assert_numeric_sequences_close(
                        f"{case.name} {family}:{dataset} {quantity}",
                        left_values[offset::2],
                        right_values[offset::2],
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                    )
                continue
            tolerance_label = dataset.replace("coordinate", "position")
            relative_tolerance, absolute_tolerance = _deterministic_tolerance(
                tolerance_label
            )
            _assert_numeric_sequences_close(
                f"{case.name} {family}:{dataset}",
                left_values,
                right_values,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        compared[family] = list(datasets)

    branch_text_h5 = {}
    for branch, directory in continuation_dirs.items():
        trajectory_path = files[branch]["trajectory"]
        restart_path = files[branch]["restart"]
        _validate_observable_output(
            case.name,
            files[branch]["observable"],
            directory / "mdout.txt",
        )
        _validate_restart_output(case.name, restart_path)
        coordinate = _assert_nhc_text_matches_h5(
            f"{case.name} {branch} NHC coordinate text/H5",
            directory / "output/continuation_nhc.crd",
            trajectory_path,
            f"{NHC_OBSERVABLE_ROOT}/coordinate/value",
            3,
        )
        velocity = _assert_nhc_text_matches_h5(
            f"{case.name} {branch} NHC velocity text/H5",
            directory / "output/continuation_nhc.vel",
            trajectory_path,
            f"{NHC_OBSERVABLE_ROOT}/velocity/value",
            3,
        )
        restart_pairs = _read_nhc_restart_pairs(
            directory / "output/continuation_nhc_restart.txt", 3
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} NHC restart text/H5",
            restart_pairs,
            _h5_numeric_values(restart_path, NHC_RESTART_DATASET),
            relative_tolerance=1.0e-6,
            absolute_tolerance=6.0e-7,
        )
        branch_text_h5[branch] = {
            "coordinate": coordinate,
            "velocity": velocity,
            "restart_pair_count": len(restart_pairs) // 2,
        }

    for key in ("mode", "step", "time"):
        dataset = f"/parameters/restart/integrator_state/{key}"
        left = _h5_string_values(files["legacy"]["restart"], dataset)
        right = _h5_string_values(files["bundled"]["restart"], dataset)
        if left != right:
            raise AssertionError(
                f"{case.name} continuation integrator state differs for {key}: "
                f"legacy={left}, bundled={right}"
            )

    return {
        "mdout_rows": len(mdout["legacy"]["rows"]),
        "mdout_columns": columns,
        "semantic_h5_datasets": compared,
        "legacy_output_matches_h5": branch_text_h5,
        "nontrivial_nhc_state": max(
            abs(value)
            for value in _h5_numeric_values(
                files["bundled"]["restart"], NHC_RESTART_DATASET
            )
        ),
    }


def _run_meta_protocol_full_restart_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    producer_dir, legacy_template, bundled_template = (
        _prepare_meta_protocol_full_setup(case_root / "producer_setup")
    )
    producer_metrics = _run_sponge(producer_dir, "mdin.spg.toml")
    producer_state = _validate_meta_producer_state(case, producer_dir)

    continuation_dirs, projection = _prepare_meta_continuations(
        producer_dir,
        legacy_template,
        bundled_template,
        case_root / "continuations",
    )
    continuation_metrics = {
        branch: _run_sponge(directory, "mdin.spg.toml")
        for branch, directory in continuation_dirs.items()
    }
    routes = _validate_meta_continuation_routes(producer_dir, continuation_dirs)
    comparison = _compare_meta_continuations(case, continuation_dirs)
    assertion = AssertionEvidence(
        assertion_id="restart_protocol_full_continuation_equivalence",
        evidence_level="E4",
        details={
            "method": "one_checkpoint_forked_to_legacy_protocol_and_full",
            "producer": producer_state,
            "legacy_projection": projection,
            "routes": routes,
            "continuation": comparison,
        },
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts,
        case,
        (assertion,),
        producer_metrics,
        continuation_metrics,
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "restart_load_policy": case.restart_load_policy,
        "producer_metrics": producer_metrics,
        "continuation_metrics": continuation_metrics,
        "finite_scan": finite_scan,
        "comparison": assertion.details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "restart_load_policy": case.restart_load_policy,
            "producer_count": 1,
            "continuation_branches": ["legacy", "protocol", "full"],
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B metadynamics restart metrics: {metrics_path}")


def _prepare_meta_protocol_full_setup(
    setup_root: Path,
) -> tuple[Path, Path, Path]:
    source = setup_root / "source"
    producer = setup_root / "producer"
    legacy_template = setup_root / "legacy_template"
    converted = setup_root / "converted"
    bundled_template = setup_root / "bundled_template"
    if setup_root.exists():
        shutil.rmtree(setup_root)
    source.mkdir(parents=True)
    _write_meta_source_inputs(source)
    _write_meta_producer_mdin(source)
    shutil.copytree(source, producer)
    shutil.copytree(source, legacy_template)
    _convert_legacy_case(source, converted)
    shutil.copytree(converted / "bundle", bundled_template)
    return producer, legacy_template, bundled_template


def _write_meta_source_inputs(case_dir: Path) -> None:
    files = {
        "mass.txt": "2\n12.0\n12.0\n",
        "charge.txt": "2\n0.0\n0.0\n",
        "coordinate.txt": (
            "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n20.0 20.0 20.0\n90.0 90.0 90.0\n"
        ),
        "velocity.txt": "2\n0.15 0.0 0.0\n-0.15 0.0 0.0\n",
        "cv.txt": (
            "distance\n"
            "{\n"
            "    CV_type = distance\n"
            "    atom = 0 1\n"
            "}\n"
            "meta\n"
            "{\n"
            "    Ndim = 1\n"
            "    CV = distance\n"
            "    CV_minimal = 0.5\n"
            "    CV_maximum = 3.0\n"
            "    CV_period = 0\n"
            "    CV_grid = 50\n"
            "    CV_sigma = 0.1\n"
            "    height = 0.2\n"
            "    potential_update_interval = 1\n"
            "    welltemp_factor = 20\n"
            "}\n"
            "print\n"
            "{\n"
            "    CV = distance\n"
            "}\n"
        ),
    }
    for name, content in files.items():
        (case_dir / name).write_text(content, encoding="utf-8")


def _write_meta_producer_mdin(case_dir: Path) -> None:
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    lines = [
        'md_name = "bundled io ab metadynamics producer"',
        'mode = "nvt"',
        "pbc = true",
        "step_limit = 3",
        "dt = 0.001",
        "cutoff = 10.0",
        'mass_in_file = "mass.txt"',
        'charge_in_file = "charge.txt"',
        'coordinate_in_file = "coordinate.txt"',
        'velocity_in_file = "velocity.txt"',
        'cv_in_file = "cv.txt"',
        'thermostat = "nose_hoover_chain"',
        "thermostat_tau = 0.2",
        "target_temperature = 300.0",
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'rst = "output/legacy_restart"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        "",
        "[nose_hoover_chain]",
        "length = 3",
        'restart_output = "output/legacy_nhc_restart.txt"',
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _h5_scalar_text(path: Path, dataset: str) -> str:
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise AssertionError(f"H5 text state is missing: {path}:{dataset}")
        value = h5[dataset].asstr()[()]
    if not isinstance(value, str) or not value:
        raise AssertionError(f"H5 text state is empty: {path}:{dataset}")
    return value


def _meta_hill_values(text: str, label: str) -> list[float]:
    values = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = [float(value) for value in line.split()]
        if len(row) != 2 or not all(math.isfinite(value) for value in row):
            raise AssertionError(f"{label} has an invalid hill row: {line}")
        values.extend(row)
    if len(values) < 4 or max(abs(value) for value in values[1::2]) <= 0.0:
        raise AssertionError(f"{label} has no non-trivial hill state")
    return values


def _numeric_text_values(text: str, label: str) -> list[float]:
    values = [
        float(value)
        for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text
        )
    ]
    if not values or not all(map(math.isfinite, values)):
        raise AssertionError(f"{label} has no finite numeric state")
    return values


def _validate_meta_producer_state(
    case: AbCase, producer_dir: Path
) -> dict[str, object]:
    restart_path = producer_dir / RESTART_REL
    hills = _h5_scalar_text(restart_path, f"{META_RESTART_ROOT}/hills")
    hill_values = _meta_hill_values(hills, f"{case.name} producer H5 hills")
    legacy_hills = (producer_dir / "myhill.log").read_text(encoding="utf-8")
    if not legacy_hills.startswith(hills):
        raise AssertionError("producer H5 hills are not a legacy hill prefix")
    potential = _h5_scalar_text(
        restart_path, f"{META_RESTART_ROOT}/potential_export"
    )
    potential_values = _numeric_text_values(
        potential, f"{case.name} producer metadynamics potential"
    )
    if (
        len(potential_values) % 4 != 0
        or max(abs(value) for value in potential_values[1::4]) <= 0.1
    ):
        raise AssertionError("producer metadynamics potential is trivial")
    nhc = _validate_nhc_producer_state(case, producer_dir)
    return {
        "producer_count": 1,
        "hill_count": len(hill_values) // 2,
        "maximum_abs_hill_height": max(abs(v) for v in hill_values[1::2]),
        "potential_value_count": len(potential_values),
        "h5_checkpoint_precedes_terminal_legacy_hill": len(
            _meta_hill_values(legacy_hills, "producer legacy hills")
        )
        > len(hill_values),
        "nhc": nhc,
    }


def _prepare_meta_continuations(
    producer_dir: Path,
    legacy_template: Path,
    bundled_template: Path,
    continuation_root: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    checkpoint = producer_dir / RESTART_REL
    hills = _h5_scalar_text(checkpoint, f"{META_RESTART_ROOT}/hills")
    potential = _h5_scalar_text(
        checkpoint, f"{META_RESTART_ROOT}/potential_export"
    )
    directories = {}
    for branch in ("legacy", "protocol", "full"):
        destination = continuation_root / branch
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = legacy_template if branch == "legacy" else bundled_template
        shutil.copytree(source, destination)
        (destination / "output").mkdir(parents=True, exist_ok=True)
        if branch == "legacy":
            for name in (
                "legacy_restart_coordinate.txt",
                "legacy_restart_velocity.txt",
                "legacy_nhc_restart.txt",
            ):
                shutil.copy2(producer_dir / "output" / name, destination / name)
            (destination / "myhill.log").write_text(hills, encoding="utf-8")
            (destination / "Meta_Potential.txt").write_text(
                potential, encoding="utf-8"
            )
        else:
            shutil.copy2(checkpoint, destination / "output/producer.spgr.h5")
            if branch == "protocol":
                shutil.copy2(
                    producer_dir / "output/legacy_nhc_restart.txt",
                    destination / "legacy_nhc_restart.txt",
                )
        _write_meta_continuation_mdin(destination, branch)
        directories[branch] = destination
    return directories, {
        "source": "producer H5 restart protocol text state",
        "hill_count": len(_meta_hill_values(hills, "projected hills")) // 2,
        "potential_bytes": len(potential.encode("utf-8")),
    }


def _write_meta_continuation_mdin(case_dir: Path, branch: str) -> None:
    if branch not in {"legacy", "protocol", "full"}:
        raise AssertionError(
            f"unknown metadynamics continuation branch: {branch}"
        )
    if branch == "legacy":
        input_lines = [
            'mass_in_file = "mass.txt"',
            'charge_in_file = "charge.txt"',
            'coordinate_in_file = "legacy_restart_coordinate.txt"',
            'velocity_in_file = "legacy_restart_velocity.txt"',
            'cv_in_file = "cv.txt"',
        ]
    else:
        input_lines = [
            'input_h5_topology_path = "topology.spgt.h5"',
            'input_h5_protocol_path = "protocol.spgp.h5"',
            'input_h5_restart_path = "output/producer.spgr.h5"',
            f'input_h5_restart_load = "{branch}"',
        ]
    nhc_lines = (
        ['restart_input = "legacy_nhc_restart.txt"']
        if branch in {"legacy", "protocol"}
        else []
    )
    lines = [
        f'md_name = "bundled io ab meta {branch} continuation"',
        'mode = "nvt"',
        "pbc = true",
        "step_limit = 2",
        "dt = 0.001",
        "cutoff = 10.0",
        *input_lines,
        'thermostat = "nose_hoover_chain"',
        "thermostat_tau = 0.2",
        "target_temperature = 300.0",
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 1",
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'crd = "output/continuation.crd"',
        'box = "output/continuation.box"',
        'vel = "output/continuation.vel"',
        'frc = "output/continuation.frc"',
        'rst = "output/continuation_restart"',
        f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
        "output_h5_trajectory_vds = false",
        "output_h5_trajectory_chunk_size = 2",
        f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        "",
        "[nose_hoover_chain]",
        "length = 3",
        *nhc_lines,
        'restart_output = "output/continuation_nhc_restart.txt"',
        'crd = "output/continuation_nhc.crd"',
        'vel = "output/continuation_nhc.vel"',
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _validate_meta_continuation_routes(
    producer_dir: Path, continuation_dirs: dict[str, Path]
) -> dict[str, object]:
    texts = {
        branch: (directory / "mdin.spg.toml").read_text(encoding="utf-8")
        for branch, directory in continuation_dirs.items()
    }
    h5_keys = {
        "input_h5_topology_path",
        "input_h5_protocol_path",
        "input_h5_restart_path",
        "input_h5_restart_load",
    }
    legacy_keys = {
        "coordinate_in_file",
        "velocity_in_file",
        "cv_in_file",
        "restart_input",
    }
    if any(_has_key_line(texts["legacy"], key) for key in h5_keys):
        raise AssertionError("legacy meta continuation retained an H5 route")
    if not all(_has_key_line(texts["legacy"], key) for key in legacy_keys):
        raise AssertionError("legacy meta continuation route is incomplete")
    for branch in ("protocol", "full"):
        if not all(_has_key_line(texts[branch], key) for key in h5_keys):
            raise AssertionError(
                f"{branch} H5 continuation route is incomplete"
            )
        if f'input_h5_restart_load = "{branch}"' not in texts[branch]:
            raise AssertionError(f"{branch} selected the wrong H5 load policy")
        if any(
            _has_key_line(texts[branch], key)
            for key in {"coordinate_in_file", "velocity_in_file", "cv_in_file"}
        ):
            raise AssertionError(
                f"{branch} retained legacy structural/protocol input"
            )
    if not _has_key_line(texts["protocol"], "restart_input"):
        raise AssertionError(
            "protocol route lacks its legacy NHC dynamic state"
        )
    if _has_key_line(texts["full"], "restart_input"):
        raise AssertionError("full route retained a legacy NHC dynamic state")
    checkpoint = (producer_dir / RESTART_REL).read_bytes()
    for branch in ("protocol", "full"):
        if (
            continuation_dirs[branch] / "output/producer.spgr.h5"
        ).read_bytes() != checkpoint:
            raise AssertionError(
                f"{branch} did not use the producer checkpoint"
            )
        if not (continuation_dirs[branch] / "myhill.log").exists():
            raise AssertionError(
                f"{branch} did not materialize metadynamics hills"
            )
    return {
        "legacy": sorted(legacy_keys),
        "protocol": sorted(h5_keys | {"restart_input"}),
        "full": sorted(h5_keys),
        "same_producer_checkpoint": True,
        "protocol_dynamic_owner": "legacy NHC text",
        "full_dynamic_owner": "H5 NHC state",
    }


def _compare_meta_continuations(
    case: AbCase, continuation_dirs: dict[str, Path]
) -> dict[str, object]:
    mdout = {
        branch: _read_mdout(directory / "mdout.txt")
        for branch, directory in continuation_dirs.items()
    }
    columns = _require_matching_mdout_columns(
        mdout["legacy"], mdout["protocol"], f"{case.name} protocol"
    )
    _require_matching_mdout_columns(
        mdout["legacy"], mdout["full"], f"{case.name} full"
    )
    for branch in ("protocol", "full"):
        for column in columns:
            relative_tolerance, absolute_tolerance = _deterministic_tolerance(
                column
            )
            _assert_numeric_sequences_close(
                f"{case.name} {branch} mdout {column}",
                [row[column] for row in mdout["legacy"]["rows"]],
                [row[column] for row in mdout[branch]["rows"]],
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
    for name in ("meta", "rbias"):
        values = [row[name] for row in mdout["legacy"]["rows"]]
        if (
            not all(math.isfinite(value) for value in values)
            or max(abs(value) for value in values) <= 0.1
        ):
            raise AssertionError(f"{case.name} has trivial {name} behavior")

    files = {
        branch: _output_h5_files(case, directory)
        for branch, directory in continuation_dirs.items()
    }
    semantic_datasets = {
        "trajectory": (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/force/value",
            "/particles/all/box/edges/value",
            f"{NHC_OBSERVABLE_ROOT}/coordinate/step",
            f"{NHC_OBSERVABLE_ROOT}/coordinate/time",
            f"{NHC_OBSERVABLE_ROOT}/coordinate/value",
            f"{NHC_OBSERVABLE_ROOT}/velocity/value",
            f"{META_OBSERVABLE_ROOT}/meta/step",
            f"{META_OBSERVABLE_ROOT}/meta/time",
            f"{META_OBSERVABLE_ROOT}/meta/value",
            f"{META_OBSERVABLE_ROOT}/rbias/value",
            f"{META_OBSERVABLE_ROOT}/rct/value",
        ),
        "observable": (
            "/observables/all/step",
            "/observables/all/time",
            "/observables/all/meta/value",
            "/observables/all/rbias/value",
            "/observables/all/rct/value",
            "/observables/all/distance/value",
            f"{NHC_OBSERVABLE_ROOT}/coordinate/value",
            f"{NHC_OBSERVABLE_ROOT}/velocity/value",
            f"{META_OBSERVABLE_ROOT}/meta/value",
            f"{META_OBSERVABLE_ROOT}/rbias/value",
            f"{META_OBSERVABLE_ROOT}/rct/value",
        ),
        "restart": (
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/box/edges/value",
            NHC_RESTART_DATASET,
        ),
    }
    for branch in ("protocol", "full"):
        for family, datasets in semantic_datasets.items():
            left_path = files["legacy"][family]
            right_path = files[branch][family]
            for dataset in datasets:
                left = _h5_numeric_values(left_path, dataset)
                right = _h5_numeric_values(right_path, dataset)
                _assert_matching_numeric_shape(
                    f"{case.name} {branch} {family}:{dataset}",
                    left_path,
                    right_path,
                    dataset,
                    left,
                    right,
                )
                if dataset == NHC_RESTART_DATASET:
                    tolerance = (1.0e-6, 6.0e-7)
                else:
                    tolerance = _deterministic_tolerance(
                        dataset.replace("coordinate", "position")
                    )
                _assert_numeric_sequences_close(
                    f"{case.name} {branch} {family}:{dataset}",
                    left,
                    right,
                    relative_tolerance=tolerance[0],
                    absolute_tolerance=tolerance[1],
                )

    legacy_hills = _meta_hill_values(
        (continuation_dirs["legacy"] / "myhill.log").read_text(
            encoding="utf-8"
        ),
        f"{case.name} legacy final hills",
    )
    legacy_restart_hills = _meta_hill_values(
        _h5_scalar_text(
            files["legacy"]["restart"], f"{META_RESTART_ROOT}/hills"
        ),
        f"{case.name} legacy restart hills",
    )
    legacy_potential = _numeric_text_values(
        (continuation_dirs["legacy"] / "Meta_Potential.txt").read_text(
            encoding="utf-8"
        ),
        f"{case.name} legacy final potential",
    )
    legacy_restart_potential = _numeric_text_values(
        _h5_scalar_text(
            files["legacy"]["restart"],
            f"{META_RESTART_ROOT}/potential_export",
        ),
        f"{case.name} legacy restart potential",
    )
    final_hill_counts = {}
    for branch, directory in continuation_dirs.items():
        branch_hills = _meta_hill_values(
            (directory / "myhill.log").read_text(encoding="utf-8"),
            f"{case.name} {branch} final hills",
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} final hill history",
            legacy_hills,
            branch_hills,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-7,
        )
        restart_hills = _meta_hill_values(
            _h5_scalar_text(
                files[branch]["restart"], f"{META_RESTART_ROOT}/hills"
            ),
            f"{case.name} {branch} restart hills",
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} restart hill history",
            legacy_restart_hills,
            restart_hills,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-7,
        )
        branch_potential = _numeric_text_values(
            (directory / "Meta_Potential.txt").read_text(encoding="utf-8"),
            f"{case.name} {branch} final potential",
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} final potential",
            legacy_potential,
            branch_potential,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-7,
        )
        branch_restart_potential = _numeric_text_values(
            _h5_scalar_text(
                files[branch]["restart"],
                f"{META_RESTART_ROOT}/potential_export",
            ),
            f"{case.name} {branch} restart potential",
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} restart potential",
            legacy_restart_potential,
            branch_restart_potential,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-7,
        )
        final_hill_counts[branch] = {
            "sidecar": len(branch_hills) // 2,
            "restart": len(restart_hills) // 2,
        }
        _validate_observable_output(
            case.name, files[branch]["observable"], directory / "mdout.txt"
        )
        _validate_restart_output(case.name, files[branch]["restart"])
        _assert_nhc_text_matches_h5(
            f"{case.name} {branch} NHC coordinate text/H5",
            directory / "output/continuation_nhc.crd",
            files[branch]["trajectory"],
            f"{NHC_OBSERVABLE_ROOT}/coordinate/value",
            3,
        )
        _assert_nhc_text_matches_h5(
            f"{case.name} {branch} NHC velocity text/H5",
            directory / "output/continuation_nhc.vel",
            files[branch]["trajectory"],
            f"{NHC_OBSERVABLE_ROOT}/velocity/value",
            3,
        )
        restart_pairs = _read_nhc_restart_pairs(
            directory / "output/continuation_nhc_restart.txt", 3
        )
        _assert_numeric_sequences_close(
            f"{case.name} {branch} NHC restart text/H5",
            restart_pairs,
            _h5_numeric_values(files[branch]["restart"], NHC_RESTART_DATASET),
            relative_tolerance=1.0e-6,
            absolute_tolerance=6.0e-7,
        )

    for key in ("mode", "step", "time"):
        dataset = f"/parameters/restart/integrator_state/{key}"
        expected = _h5_string_values(files["legacy"]["restart"], dataset)
        for branch in ("protocol", "full"):
            actual = _h5_string_values(files[branch]["restart"], dataset)
            if actual != expected:
                raise AssertionError(
                    f"{case.name} {branch} integrator {key} differs: "
                    f"legacy={expected}, bundled={actual}"
                )
    return {
        "mdout_rows": len(mdout["legacy"]["rows"]),
        "mdout_columns": columns,
        "maximum_abs_meta": max(
            abs(row["meta"]) for row in mdout["legacy"]["rows"]
        ),
        "maximum_abs_rbias": max(
            abs(row["rbias"]) for row in mdout["legacy"]["rows"]
        ),
        "semantic_h5_datasets": {
            family: list(datasets)
            for family, datasets in semantic_datasets.items()
        },
        "final_hill_counts": final_hill_counts,
    }


def _run_chunk_boundary_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    legacy_dir, bundled_dir = _prepare_case_pair(case, case_root, 20260709)
    run_metrics = {}
    for branch, case_dir, mdin_name in (
        ("legacy", legacy_dir, "mdin.spg.toml"),
        ("bundled", bundled_dir, "mdin.bundled.spg.toml"),
    ):
        _prepare_mdin(
            case_dir,
            mdin_name,
            case,
            branch=branch,
            replica_seed=20260709,
        )
        run_metrics[branch] = _run_sponge(case_dir, mdin_name)

    run = AbRun(
        replica_index=0,
        replica_seed=20260709,
        legacy_dir=legacy_dir,
        bundled_dir=bundled_dir,
        legacy_metrics={},
        bundled_metrics={},
        legacy_output_contract={},
        bundled_output_contract={},
    )
    mdout = _compare_mdout_deterministically(case, run)
    h5 = _compare_h5_outputs_deterministically(case, run)
    layouts = {
        "legacy": _assert_chunk_boundary_layout(case, legacy_dir),
        "bundled": _assert_chunk_boundary_layout(case, bundled_dir),
    }
    details = {
        "method": "same_semantic_deterministic_h5_and_vds_layout",
        "chunk_size": case.output_chunk_size,
        "expected_frames": case.expected_trajectory_frames,
        "mdout_rows": mdout["rows"],
        "h5_families": sorted(h5),
        "layouts": layouts,
    }
    assertions = [
        AssertionEvidence(
            assertion_id="h5_chunk_boundary_equivalence",
            evidence_level="E3",
            details=details,
        )
    ]
    if "h5_complete_prefix_repair_equivalence" in case.assertion_ids:
        noops = {
            "legacy": _assert_complete_prefix_noop_layout(
                f"{case.name} legacy",
                legacy_dir / TRAJECTORY_REL,
                expected_frame_count=case.expected_trajectory_frames,
                expected_shard_count=math.ceil(
                    case.expected_trajectory_frames / case.output_chunk_size
                ),
            ),
            "bundled": _assert_complete_prefix_noop_layout(
                f"{case.name} bundled",
                bundled_dir / TRAJECTORY_REL,
                expected_frame_count=case.expected_trajectory_frames,
                expected_shard_count=math.ceil(
                    case.expected_trajectory_frames / case.output_chunk_size
                ),
            ),
        }
        terminal_repair = _run_vds_terminal_repair_smoke()
        repair_details = {
            "production_noop": noops,
            "terminal_tail_repair": terminal_repair,
            "cross_process_append_resume": "unsupported",
        }
        details["complete_prefix"] = repair_details
        assertions.append(
            AssertionEvidence(
                assertion_id="h5_complete_prefix_repair_equivalence",
                evidence_level="E3",
                details=repair_details,
            )
        )
    evidence, finite_scan = _build_success_case_evidence(
        contracts, case, tuple(assertions), run_metrics
    )
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "run_metrics": run_metrics,
        "finite_scan": finite_scan,
        "comparison": details,
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "chunk_size": case.output_chunk_size,
            "expected_trajectory_frames": case.expected_trajectory_frames,
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B chunk metrics: {metrics_path}")


def _run_output_family_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    legacy_dir, bundled_dir = _prepare_case_pair(case, case_root, 20260709)
    run_metrics = {}
    for branch, case_dir in (("legacy", legacy_dir), ("bundled", bundled_dir)):
        _prepare_output_family_mdin(case, case_dir, branch=branch)
        run_metrics[branch] = _run_sponge(case_dir, _mdin_name(case_dir))

    run = AbRun(
        replica_index=0,
        replica_seed=20260709,
        legacy_dir=legacy_dir,
        bundled_dir=bundled_dir,
        legacy_metrics={},
        bundled_metrics={},
        legacy_output_contract={},
        bundled_output_contract={},
    )
    mdout = _compare_mdout_deterministically(case, run)
    h5 = _compare_h5_outputs_deterministically(case, run)
    branches = {
        branch: _validate_output_family_layout(case, directory)
        for branch, directory in (
            ("legacy", legacy_dir),
            ("bundled", bundled_dir),
        )
    }
    continuation = None
    if case.output_families == ("restart",):
        continuation = _run_restart_only_output_continuation(case, bundled_dir)
    details = {
        "families": list(case.output_families),
        "legacy_output_mode": case.legacy_output_mode,
        "mdout_rows": mdout["rows"],
        "h5_families": sorted(h5),
        "branches": branches,
        "restart_only_continuation": continuation,
    }
    evidence_level = "E4" if continuation is not None else "E3"
    assertions = (
        AssertionEvidence(
            assertion_id="output_family_process_matrix",
            evidence_level=evidence_level,
            details=details,
        ),
    )
    evidence, finite_scan = _build_success_case_evidence(
        contracts, case, assertions, run_metrics, continuation
    )
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "profile": PROFILE,
                "case": case.name,
                "contract_ids": list(case.contract_ids),
                "assertion_ids": list(case.assertion_ids),
                "evidence": [record.as_dict() for record in evidence],
                "run_metrics": run_metrics,
                "finite_scan": finite_scan,
                "comparison": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )


def _prepare_output_family_mdin(
    case: AbCase, case_dir: Path, *, branch: str
) -> None:
    normal_case = replace(case, mode="normal", output_families=())
    _prepare_mdin(
        case_dir,
        _mdin_name(case_dir),
        normal_case,
        branch=branch,
        replica_seed=20260709,
    )
    mdin_path = case_dir / _mdin_name(case_dir)
    text = mdin_path.read_text(encoding="utf-8")
    remove_keys = {"print_zeroth_frame"}
    enabled = set(case.output_families)
    if "trajectory" not in enabled:
        remove_keys.update(
            {
                "output_h5_trajectory_path",
                "output_h5_trajectory_vds",
                "output_h5_trajectory_chunk_size",
                "output_h5_trajectory_repair_policy",
            }
        )
    if "observable" not in enabled:
        remove_keys.add("output_h5_observable_path")
    if "restart" not in enabled:
        remove_keys.add("output_h5_restart_path")
    if case.legacy_output_mode == "suppressed":
        remove_keys.update({"crd", "box", "vel", "frc", "rst"})
    elif case.legacy_output_mode == "coexist":
        if "trajectory" not in enabled:
            remove_keys.update({"crd", "box", "vel", "frc"})
        if "restart" not in enabled:
            remove_keys.add("rst")
    else:
        raise AssertionError(
            f"{case.name} has unknown legacy output mode {case.legacy_output_mode}"
        )
    mdin_path.write_text(
        _insert_root_toml_keys(
            _remove_key_lines(text, remove_keys), ["print_zeroth_frame = 0"]
        ),
        encoding="utf-8",
    )


def _validate_output_family_layout(
    case: AbCase, case_dir: Path
) -> dict[str, object]:
    expected = _output_h5_files(case, case_dir)
    all_paths = {
        "trajectory": case_dir / TRAJECTORY_REL,
        "observable": case_dir / OBSERVABLE_REL,
        "restart": case_dir / RESTART_REL,
    }
    for family, path in all_paths.items():
        if path.exists() != (family in expected):
            raise AssertionError(
                f"{case.name} {family} file presence changed: {path.exists()}"
            )
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise AssertionError(
                f"{case.name} left temporary output {temporary}"
            )

    summary = {}
    if "trajectory" in expected:
        _validate_trajectory_output(case, expected["trajectory"], case_dir)
        summary["trajectory"] = {
            "status": _h5_string_values(
                expected["trajectory"], "/parameters/sponge/output/status"
            ),
            "frame_count": int(
                _h5_numeric_values(
                    expected["trajectory"],
                    "/parameters/sponge/output/frame_count",
                )[-1]
            ),
        }
    if "observable" in expected:
        summary["observable"] = _validate_observable_output(
            case.name, expected["observable"], case_dir / "mdout.txt"
        )
    if "restart" in expected:
        _validate_restart_output(case.name, expected["restart"])
        summary["restart"] = {"readable": True}

    legacy_paths = {
        "crd": case_dir / "output/legacy.crd",
        "box": case_dir / "output/legacy.box",
        "vel": case_dir / "output/legacy.vel",
        "frc": case_dir / "output/legacy.frc",
        "rst_coordinate": case_dir / "output/legacy_restart_coordinate.txt",
        "rst_velocity": case_dir / "output/legacy_restart_velocity.txt",
    }
    if case.legacy_output_mode == "suppressed":
        unexpected = sorted(
            name for name, path in legacy_paths.items() if path.exists()
        )
        if unexpected:
            raise AssertionError(
                f"{case.name} default legacy suppression failed: {unexpected}"
            )
    else:
        if "trajectory" in expected:
            _validate_output_family_trajectory_coexistence(
                case, case_dir, expected["trajectory"]
            )
        if "restart" in expected:
            _validate_restart_legacy_coexistence(
                case, case_dir, expected["restart"]
            )
    forbidden_provenance = {"crd", "box", "vel", "frc", "rst"}
    if case.legacy_output_mode == "suppressed":
        for family, path in expected.items():
            sidecar_root = "/parameters/sponge/files/legacy_sidecars/key"
            keys = (
                _h5_string_values(path, sidecar_root)
                if sidecar_root in _h5_paths(path)
                else []
            )
            overlap = sorted(forbidden_provenance.intersection(keys))
            if overlap:
                raise AssertionError(
                    f"{case.name} {family} has suppressed provenance: {overlap}"
                )
    summary["legacy_artifacts"] = {
        name: path.exists() for name, path in legacy_paths.items()
    }
    return summary


def _validate_output_family_trajectory_coexistence(
    case: AbCase, case_dir: Path, trajectory: Path
) -> None:
    keys = _h5_string_values(
        trajectory, "/parameters/sponge/files/legacy_sidecars/key"
    )
    paths = _h5_string_values(
        trajectory, "/parameters/sponge/files/legacy_sidecars/path"
    )
    provenance = dict(zip(keys, paths))
    expected = {
        "crd": "output/legacy.crd",
        "box": "output/legacy.box",
        "vel": "output/legacy.vel",
        "frc": "output/legacy.frc",
    }
    if {key: provenance.get(key) for key in expected} != expected:
        raise AssertionError(
            f"{case.name} trajectory coexistence provenance differs: "
            f"{provenance}"
        )
    missing_or_empty = sorted(
        key
        for key, relative_path in expected.items()
        if not (case_dir / relative_path).is_file()
        or (case_dir / relative_path).stat().st_size == 0
    )
    if missing_or_empty:
        raise AssertionError(
            f"{case.name} trajectory coexistence artifacts are absent or "
            f"empty: {missing_or_empty}"
        )


def _run_restart_only_output_continuation(
    case: AbCase, producer_dir: Path
) -> dict[str, object]:
    continuation = producer_dir.parent / "bundled_restart_only_continuation"
    if continuation.exists():
        shutil.rmtree(continuation)
    shutil.copytree(producer_dir, continuation)
    checkpoint = continuation / "family_checkpoint.spgr.h5"
    shutil.copy2(continuation / RESTART_REL, checkpoint)
    output = continuation / "output"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = continuation / file_name
        if path.exists():
            path.unlink()
    mdin_path = continuation / _mdin_name(continuation)
    text = _remove_key_lines(
        mdin_path.read_text(encoding="utf-8"),
        {
            "input_h5_restart_path",
            "input_h5_restart_load",
            "output_h5_trajectory_path",
            "output_h5_trajectory_vds",
            "output_h5_trajectory_chunk_size",
            "output_h5_trajectory_repair_policy",
            "output_h5_observable_path",
            "output_h5_restart_path",
            "crd",
            "box",
            "vel",
            "frc",
            "rst",
            "step_limit",
            "write_mdout_interval",
            "write_trajectory_interval",
            "write_restart_file_interval",
            "mdout",
            "mdinfo",
        },
    )
    mdin_path.write_text(
        _insert_root_toml_keys(
            text,
            [
                'input_h5_restart_path = "family_checkpoint.spgr.h5"',
                'input_h5_restart_load = "structural"',
                "step_limit = 1",
                "write_mdout_interval = 1",
                'mdout = "mdout.txt"',
                'mdinfo = "mdinfo.txt"',
            ],
        ),
        encoding="utf-8",
    )
    outcome = _run_sponge_process(continuation, mdin_path.name)
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} restart-only continuation failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    finite_scan = _scan_success_outputs(continuation)
    rows = _read_mdout(continuation / "mdout.txt")["rows"]
    if len(rows) != 1 or [row.get("step") for row in rows] != [1.0]:
        raise AssertionError(
            f"{case.name} restart-only continuation schedule changed: {rows}"
        )
    result = {
        "exit_code": outcome.returncode,
        "mdout_rows": len(rows),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "finite_scan": finite_scan,
    }
    shutil.rmtree(continuation)
    return result


def _run_output_writer_failure_case(case: AbCase, contracts) -> None:
    case_root = _output_root() / case.name
    bundled_dir = _copy_case(case, "bundled", case.bundled_subdir, case_root)
    _prepare_output_family_mdin(case, bundled_dir, branch="bundled")
    outcome = _run_sponge_process(
        bundled_dir,
        _mdin_name(bundled_dir),
        extra_env={
            "SPONGE_H5_TEST_FAULT": (
                f"{case.output_fault_family}:{case.output_fault_phase}"
            )
        },
    )
    combined = outcome.stdout + "\n" + outcome.stderr
    expected_reason = (
        f"injected H5 {case.output_fault_family} "
        f"{case.output_fault_phase} failure"
    )
    if outcome.returncode == 0:
        raise AssertionError(f"{case.name} swallowed the injected failure")
    for token in (
        "spongeErrorValueErrorCommand raised by Main_Clear",
        "H5 output failure isolation",
        f"family={case.output_fault_family}",
        f"phase={case.output_fault_phase}",
        expected_reason,
    ):
        if token not in combined:
            raise AssertionError(
                f"{case.name} has unstable failure diagnostics: missing {token!r}"
            )

    family_paths = {
        "trajectory": bundled_dir / TRAJECTORY_REL,
        "observable": bundled_dir / OBSERVABLE_REL,
        "restart": bundled_dir / RESTART_REL,
    }
    files = {
        family: _validate_writer_failure_file(
            case,
            family,
            path,
            failed=family == case.output_fault_family,
            expected_reason=expected_reason,
        )
        for family, path in family_paths.items()
    }
    for path in family_paths.values():
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise AssertionError(f"{case.name} left temporary file {temporary}")

    details = {
        "output_classification": FAILURE_SEMANTICS,
        "fault_family": case.output_fault_family,
        "fault_phase": case.output_fault_phase,
        "exit_code": outcome.returncode,
        "stable_reason": expected_reason,
        "files": files,
    }
    assertions = (
        AssertionEvidence(
            assertion_id="writer_failure_isolation",
            evidence_level="F1",
            details=details,
        ),
    )
    evidence = build_case_evidence(contracts, case, assertions)
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "profile": PROFILE,
                "case": case.name,
                "contract_ids": list(case.contract_ids),
                "assertion_ids": list(case.assertion_ids),
                "evidence": [record.as_dict() for record in evidence],
                "comparison": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    update_evidence_report(
        _output_root() / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )


def _validate_writer_failure_file(
    case: AbCase,
    family: str,
    path: Path,
    *,
    failed: bool,
    expected_reason: str,
) -> dict[str, object]:
    if not path.is_file():
        raise AssertionError(f"{case.name} did not preserve {family} output")
    paths = _h5_paths(path)
    required = {
        "/parameters/sponge/output/status",
        "/parameters/sponge/output/frame_count",
        "/parameters/sponge/output/last_complete_step",
        "/parameters/sponge/output/last_complete_time",
    }
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"{case.name} {family} completion metadata is missing: {missing}"
        )
    status = _h5_string_values(path, "/parameters/sponge/output/status")[-1]
    expected_status = "failed" if failed else "finalized"
    if status != expected_status:
        raise AssertionError(
            f"{case.name} {family} status={status!r}, expected={expected_status!r}"
        )
    if failed:
        error_path = "/parameters/sponge/output/error"
        if error_path not in paths:
            raise AssertionError(f"{case.name} {family} has no failure reason")
        error = _h5_string_values(path, error_path)[-1]
        if expected_reason not in error:
            raise AssertionError(
                f"{case.name} {family} failure reason differs: {error!r}"
            )

    frame_count = int(
        _h5_numeric_values(path, "/parameters/sponge/output/frame_count")[-1]
    )
    stream_path = {
        "trajectory": "/particles/all/step",
        "observable": "/observables/all/step",
        "restart": "/particles/all/step",
    }[family]
    stream = (
        _h5_numeric_values(path, stream_path) if stream_path in paths else []
    )
    if not failed and frame_count != len(stream):
        raise AssertionError(
            f"{case.name} {family} completion count differs: "
            f"metadata={frame_count}, stream={len(stream)}"
        )
    if failed and frame_count > len(stream):
        raise AssertionError(
            f"{case.name} {family} failed completion prefix exceeds stream"
        )
    if not failed and frame_count == 0:
        raise AssertionError(
            f"{case.name} unaffected {family} has no complete frames"
        )
    return {
        "status": status,
        "frame_count": frame_count,
        "stream_count": len(stream),
        "readable": True,
    }


def _assert_chunk_boundary_layout(
    case: AbCase, case_dir: Path
) -> dict[str, int]:
    if case.expected_trajectory_frames is None:
        raise AssertionError(f"{case.name} has no expected frame count")
    trajectory = case_dir / TRAJECTORY_REL
    frame_count = int(
        _h5_numeric_values(trajectory, "/parameters/sponge/output/frame_count")[
            -1
        ]
    )
    if frame_count != case.expected_trajectory_frames:
        raise AssertionError(
            f"{case.name} frame count mismatch: "
            f"expected={case.expected_trajectory_frames}, actual={frame_count}"
        )
    shape = _h5_dataset_shape(trajectory, "/particles/all/position/value")
    if not shape or shape[0] != frame_count:
        raise AssertionError(
            f"{case.name} position frame dimension differs: {shape}"
        )
    shard_count = _vds_shard_count(trajectory)
    expected_shards = math.ceil(frame_count / case.output_chunk_size)
    if shard_count != expected_shards:
        raise AssertionError(
            f"{case.name} shard count mismatch: "
            f"expected={expected_shards}, actual={shard_count}"
        )
    return {
        "frame_count": frame_count,
        "shard_count": shard_count,
        "expected_shard_count": expected_shards,
    }


def _assert_complete_prefix_noop_layout(
    label: str,
    trajectory: Path,
    *,
    expected_frame_count: int | None,
    expected_shard_count: int,
) -> dict[str, object]:
    if expected_frame_count is None:
        raise AssertionError(f"{label} has no expected frame count")
    policy = _h5_string_values(
        trajectory, "/parameters/sponge/output/repair_policy"
    )
    status = _h5_string_values(
        trajectory, "/parameters/sponge/output/repair_status"
    )
    repaired_count = _h5_numeric_values(
        trajectory, "/parameters/sponge/output/repaired_shard_count"
    )
    frame_count = _h5_numeric_values(
        trajectory, "/parameters/sponge/output/frame_count"
    )
    manifest_status = _h5_string_values(
        trajectory, "/parameters/sponge/output/shard_manifest/status"
    )
    if policy != ["complete_prefix"]:
        raise AssertionError(f"{label} repair policy differs: {policy}")
    if status != ["not_applied"]:
        raise AssertionError(f"{label} repair status differs: {status}")
    if repaired_count != [0.0]:
        raise AssertionError(
            f"{label} repaired shard count differs: {repaired_count}"
        )
    if not frame_count or int(frame_count[-1]) != expected_frame_count:
        raise AssertionError(f"{label} completion frame count differs")
    if len(manifest_status) != expected_shard_count or any(
        value != "complete" for value in manifest_status
    ):
        raise AssertionError(
            f"{label} manifest is not a complete {expected_shard_count}-shard "
            f"prefix: {manifest_status}"
        )
    return {
        "policy": policy[0],
        "status": status[0],
        "repaired_shard_count": int(repaired_count[0]),
        "frame_count": int(frame_count[-1]),
        "manifest_status": manifest_status,
    }


def _run_vds_terminal_repair_smoke() -> dict[str, object]:
    configured = os.environ.get("SPONGE_BUNDLED_IO_AB_VDS_REPAIR_SMOKE")
    executable = (
        Path(configured)
        if configured
        else Path(_sponge_executable()).parent
        / "tests"
        / "h5_bundle"
        / "test_h5_vds_terminal_resume_smoke"
    )
    if not executable.is_file():
        raise AssertionError(
            "VDS terminal repair smoke executable is missing; set "
            "SPONGE_BUNDLED_IO_AB_VDS_REPAIR_SMOKE or build "
            "test_h5_vds_terminal_resume_smoke"
        )
    env = dict(os.environ)
    env["SPONGE_H5_ENABLE_RUNTIME_SMOKE"] = "1"
    outcome = subprocess.run(
        [str(executable)],
        cwd=executable.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if outcome.returncode != 0:
        raise AssertionError(
            "VDS terminal repair smoke failed: "
            f"returncode={outcome.returncode}, stdout={outcome.stdout!r}, "
            f"stderr={outcome.stderr!r}"
        )
    return {
        "executable": str(executable),
        "terminal_shard_finalize_failure_injected": True,
        "repaired_to_complete_prefix": True,
        "complete_prefix_noop_checked": True,
    }


def test_sidecar_rerun_cmap_potential_is_branch_and_vds_invariant():
    cases = {
        case.vds: case
        for case in _cases_for_profile()
        if case.name.startswith("rerun_full_contract_sidecar_vds_")
    }
    if set(cases) != {False, True}:
        raise AssertionError(
            "sidecar rerun A/B cases must cover VDS off and on"
        )

    root = _output_root() / "sidecar_rerun_cmap_potential_regression"
    potential_by_branch: dict[str, dict[bool, list[float]]] = {
        "legacy": {},
        "bundled": {},
    }
    for vds, case in cases.items():
        case_root = root / ("vds_on" if vds else "vds_off")
        legacy_dir, bundled_dir = _prepare_case_pair(case, case_root, 20260709)
        _prepare_mdin(
            legacy_dir,
            "mdin.spg.toml",
            case,
            branch="legacy",
            replica_seed=20260709,
        )
        _prepare_mdin(
            bundled_dir,
            "mdin.bundled.spg.toml",
            case,
            branch="bundled",
            replica_seed=20260709,
        )
        _run_sponge(legacy_dir, _mdin_name(legacy_dir))
        _run_sponge(bundled_dir, _mdin_name(bundled_dir))

        legacy_mdout = _read_mdout(legacy_dir / "mdout.txt")
        bundled_mdout = _read_mdout(bundled_dir / "mdout.txt")
        _require_matching_mdout_columns(
            legacy_mdout, bundled_mdout, f"{case.name} potential regression"
        )
        if "potential" not in legacy_mdout["columns"]:
            raise AssertionError(f"{case.name} mdout is missing potential")
        legacy_potential = [row["potential"] for row in legacy_mdout["rows"]]
        bundled_potential = [row["potential"] for row in bundled_mdout["rows"]]
        if not any(math.isfinite(value) for value in legacy_potential):
            raise AssertionError(
                f"{case.name} legacy potential has no finite frame"
            )
        if not any(math.isfinite(value) for value in bundled_potential):
            raise AssertionError(
                f"{case.name} bundled potential has no finite frame"
            )
        _assert_numeric_sequences_close(
            f"{case.name} potential",
            legacy_potential,
            bundled_potential,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )
        potential_by_branch["legacy"][vds] = legacy_potential
        potential_by_branch["bundled"][vds] = bundled_potential

    for branch, potentials in potential_by_branch.items():
        _assert_numeric_sequences_close(
            f"sidecar rerun {branch} potential VDS off/on",
            potentials[False],
            potentials[True],
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )


def _output_root() -> Path:
    configured = os.environ.get("SPONGE_BUNDLED_IO_AB_OUTPUT_DIR")
    if configured:
        root = Path(configured)
        root.mkdir(parents=True, exist_ok=True)
        return root
    cached = os.environ.get("SPONGE_BUNDLED_IO_AB_RUN_ROOT")
    if cached:
        return Path(cached)
    root = Path(tempfile.mkdtemp(prefix=f"sponge_bundled_io_ab_{PROFILE}_"))
    os.environ["SPONGE_BUNDLED_IO_AB_RUN_ROOT"] = str(root)
    return root


def _copy_case(case: AbCase, branch: str, subdir: str, case_root: Path) -> Path:
    source = FIXTURE_ROOT / case.fixture_case / subdir
    destination = case_root / branch
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def _replica_count(case: AbCase) -> int:
    if not case.statistical_md:
        return 1
    return int(PROFILE_LIMITS[PROFILE]["normal_replicas"])


def _replica_seed(replica_index: int) -> int:
    return 20260709 + replica_index * 104729


def _prepare_case_pair(
    case: AbCase, case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    if case.mode in {"normal", "chunk_boundary", "output_family"}:
        if case.fixture_case == FOCUSED_CORE_TOPOLOGY_FIXTURE:
            return _prepare_focused_core_topology_pair(case_root)
        if case.fixture_case in {
            FOCUSED_BOND_FIXTURE,
            FOCUSED_ANGLE_FIXTURE,
            FOCUSED_UREY_BRADLEY_FIXTURE,
            FOCUSED_DIHEDRAL_FIXTURE,
            FOCUSED_LISTED_FIXTURE,
        }:
            return _prepare_focused_bonded_pair(case.fixture_case, case_root)
        if case.fixture_case == FOCUSED_NB14_SCALED_FIXTURE:
            return _prepare_focused_nb14_pair(case_root, "nb14_in_file")
        if case.fixture_case == FOCUSED_NB14_EXTRA_FIXTURE:
            return _prepare_focused_nb14_pair(case_root, "nb14_extra_in_file")
        if case.fixture_case == SITS_FF19SB_CMAP_FIXTURE:
            return _prepare_sits_ff19sb_cmap_pair(case_root, replica_seed)
        if case.fixture_case == FOCUSED_SITS_NK_TYPED_RESTART_FIXTURE:
            return _prepare_focused_sits_nk_typed_restart_pair(case_root)
        if case.fixture_case == FOCUSED_SITS_TYPED_CONFIG_FIXTURE:
            return _prepare_focused_sits_typed_config_pair(case_root)
        if case.fixture_case == FOCUSED_SITS_TYPED_INACTIVE_FIXTURE:
            return _prepare_focused_sits_typed_inactive_pair(case_root)
        if case.fixture_case == FOCUSED_EDIP_FIXTURE:
            return _prepare_focused_edip_pair(case_root)
        if case.fixture_case == FOCUSED_REAXFF_FIXTURE:
            return _prepare_focused_reaxff_pair(case_root)
        if case.fixture_case in {
            FOCUSED_EAM_FUNCFL_FIXTURE,
            FOCUSED_EAM_SETFL_FIXTURE,
        }:
            return _prepare_focused_eam_pair(case.fixture_case, case_root)
        if case.fixture_case == FOCUSED_POSITIONAL_RESTRAINT_FIXTURE:
            return _prepare_focused_positional_restraint_pair(
                case_root, typed=True
            )
        if case.fixture_case == FOCUSED_POSITIONAL_RESTRAINT_SIDECAR_FIXTURE:
            return _prepare_focused_positional_restraint_pair(
                case_root, typed=False
            )
        if case.fixture_case == FOCUSED_SOFT_WALL_FIXTURE:
            return _prepare_focused_soft_wall_pair(case_root, typed=True)
        if case.fixture_case == FOCUSED_SOFT_WALL_SIDECAR_FIXTURE:
            return _prepare_focused_soft_wall_pair(case_root, typed=False)
        if case.fixture_case == FOCUSED_CV_RESTRAINT_TYPED_FIXTURE:
            return _prepare_focused_cv_restraint_pair(case_root, typed=True)
        if case.fixture_case == FOCUSED_CV_RESTRAINT_SIDECAR_FIXTURE:
            return _prepare_focused_cv_restraint_pair(case_root, typed=False)
        if case.fixture_case == FOCUSED_SW_SIDECAR_FIXTURE:
            return _prepare_focused_sw_sidecar_pair(case_root)
        if case.fixture_case == FOCUSED_SW_TYPED_FIXTURE:
            return _prepare_focused_sw_typed_pair(case_root)
        if case.fixture_case == FOCUSED_TERSOFF_SIDECAR_FIXTURE:
            return _prepare_focused_tersoff_sidecar_pair(case_root)
        if case.fixture_case == FOCUSED_TERSOFF_TYPED_FIXTURE:
            return _prepare_focused_tersoff_typed_pair(case_root)
        if case.fixture_case == FOCUSED_CUSTOM_PAIR_FIXTURE:
            return _prepare_focused_custom_pair_pair(case_root)
        if case.fixture_case == FOCUSED_EXCLUSIONS_FIXTURE:
            return _prepare_focused_exclusions_pair(case_root)
        if case.fixture_case == FOCUSED_RESIDUE_SIDECAR_FIXTURE:
            return _prepare_focused_residue_sidecar_pair(case_root)
        if case.fixture_case == FOCUSED_RESIDUE_COM_RES_FIXTURE:
            return _prepare_focused_residue_com_res_pair(case_root)
        if case.fixture_case == FOCUSED_RESIDUE_TYPED_PBC_FIXTURE:
            return _prepare_focused_residue_typed_pair(case_root)
        if case.fixture_case == FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE:
            return _prepare_focused_residue_typed_com_res_pair(case_root)
        if case.fixture_case == FOCUSED_GB_HYBRID_FIXTURE:
            return _prepare_focused_gb_hybrid_pair(case_root)
        if case.fixture_case == FOCUSED_GB_NATIVE_FIXTURE:
            return _prepare_focused_gb_native_pair(case_root)
        if case.fixture_case == FOCUSED_IMPROPER_CONVERTED_CANONICAL_FIXTURE:
            return _prepare_focused_improper_converted_pair(
                case_root, "improper_dihedral_in_file"
            )
        if case.fixture_case == FOCUSED_IMPROPER_CONVERTED_ALIAS_FIXTURE:
            return _prepare_focused_improper_converted_pair(
                case_root, "improper_in_file"
            )
        if case.fixture_case == FOCUSED_IMPROPER_NATIVE_FIXTURE:
            return _prepare_focused_improper_native_pair(case_root)
        if case.fixture_case == FOCUSED_LJ_SOFT_CORE_FIXTURE:
            return _prepare_focused_lj_soft_core_pair(case_root)
        if case.fixture_case == FOCUSED_SUBSYSTEM_DIVISION_FIXTURE:
            return _prepare_focused_subsystem_division_pair(case_root)
        if case.fixture_case in {
            FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
            FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
            FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
        }:
            return _prepare_focused_virtual_atoms_pair(
                case.fixture_case, case_root
            )
        if case.fixture_case == FOCUSED_CONSTRAINT_SIDECAR_FIXTURE:
            return _prepare_focused_constraint_sidecar_pair(case_root)
        if case.fixture_case == FOCUSED_CONSTRAINT_TYPED_FIXTURE:
            return _prepare_focused_constraint_typed_pair(case_root)
        if case.fixture_case == FOCUSED_STEERING_CV_SIDECAR_FIXTURE:
            return _prepare_focused_steering_cv_sidecar_pair(case_root)
        if case.fixture_case == FOCUSED_STEERING_CV_TYPED_FIXTURE:
            return _prepare_focused_steering_cv_typed_pair(case_root)
        return _prepare_normal_tip3p_pair(case_root, replica_seed)

    legacy_dir = _copy_case(case, "legacy", case.legacy_subdir, case_root)
    bundled_dir = _copy_case(case, "bundled", case.bundled_subdir, case_root)
    _prepare_full_contract_nb14_owner(case, bundled_dir)
    _prepare_full_contract_eam_owner(case, bundled_dir)
    _prepare_full_contract_positional_restraint_owner(case, bundled_dir)
    _prepare_full_contract_soft_wall_owner(case, bundled_dir)
    _prepare_full_contract_cv_restraint_owner(case, bundled_dir)
    _validate_full_contract_input(case, bundled_dir)
    if "input.restart_load.absent" in case.contract_ids:
        _prepare_restart_absent_inputs(legacy_dir, bundled_dir)
        _validate_restart_absent_routes(legacy_dir, bundled_dir)
    _prepare_full_contract_finite_rerun_frames(case, legacy_dir, bundled_dir)
    _prepare_rerun_trajectory_variant(case, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_full_contract_nb14_owner(case: AbCase, bundled_dir: Path) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_root = "/forcefield/nb14_extra"
    with h5py.File(topology_path, "r+") as topology:
        typed_owner = typed_root in topology
        sidecar_owner = False
        if sidecar_root in topology:
            sidecar_keys = topology[f"{sidecar_root}/key"].asstr()[...]
            sidecar_owner = "nb14_extra_in_file" in sidecar_keys

        if "input.full_contract.pure_native" in case.contract_ids:
            if not typed_owner or sidecar_owner:
                raise AssertionError(
                    f"{case.name} pure full-contract fixture requires only "
                    "the typed nb14_extra owner"
                )
            return
        if (
            "input.full_contract.sidecar" not in case.contract_ids
            and not sidecar_owner
        ):
            return
        if not typed_owner or not sidecar_owner:
            raise AssertionError(
                f"{case.name} sidecar source fixture must contain both NB14 "
                "representations before owner selection"
            )
        del topology[typed_root]


def _prepare_full_contract_eam_owner(case: AbCase, bundled_dir: Path) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_root = "/manybody/eam"
    with h5py.File(topology_path, "r+") as topology:
        typed_owner = typed_root in topology
        sidecar_owner = False
        if sidecar_root in topology:
            sidecar_keys = topology[f"{sidecar_root}/key"].asstr()[...]
            sidecar_owner = "EAM_in_file" in sidecar_keys

        if "input.full_contract.pure_native" in case.contract_ids:
            if not typed_owner or sidecar_owner:
                raise AssertionError(
                    f"{case.name} pure full-contract fixture requires only "
                    "the typed EAM owner"
                )
            return
        if (
            "input.full_contract.sidecar" not in case.contract_ids
            and not sidecar_owner
        ):
            return
        if not typed_owner or not sidecar_owner:
            raise AssertionError(
                f"{case.name} sidecar source fixture must contain both EAM "
                "representations before owner selection"
            )
        del topology[typed_root]


def _prepare_full_contract_positional_restraint_owner(
    case: AbCase, bundled_dir: Path
) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    protocol_path = bundled_dir / "protocol.spgp.h5"
    restart_path = bundled_dir / "restart.spgr.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    protocol_typed_paths = (
        "/restraint/default/atom_indices",
        "/restraint/default/weight",
    )
    reference_path = (
        "/parameters/restart/references/restraint/default/coordinate"
    )
    with (
        h5py.File(protocol_path, "r+") as protocol,
        h5py.File(restart_path, "r+") as restart,
    ):
        typed_parts = [path in protocol for path in protocol_typed_paths]
        typed_parts.append(reference_path in restart)
        typed_owner = all(typed_parts)
        sidecar_keys: set[str] = set()
        if sidecar_root in protocol:
            sidecar_keys = set(protocol[f"{sidecar_root}/key"].asstr()[...])
        if sidecar_root in restart:
            sidecar_keys.update(restart[f"{sidecar_root}/key"].asstr()[...])
        required_sidecar_keys = {
            "restrain_atom_id",
            "restrain_weight_in_file",
            "restrain_coordinate_in_file",
        }
        sidecar_owner = required_sidecar_keys.issubset(sidecar_keys)
        partial_sidecar_owner = bool(required_sidecar_keys & sidecar_keys)

        if "input.full_contract.pure_native" in case.contract_ids:
            if not typed_owner or partial_sidecar_owner:
                raise AssertionError(
                    f"{case.name} pure full-contract fixture requires only "
                    "the typed positional-restraint owner"
                )
            return
        if (
            "input.full_contract.sidecar" not in case.contract_ids
            and not partial_sidecar_owner
        ):
            return
        if not typed_owner or not sidecar_owner:
            raise AssertionError(
                f"{case.name} sidecar source fixture must contain complete "
                "typed and sidecar positional-restraint representations "
                "before owner selection"
            )
        del protocol["/restraint/default"]
        del restart[reference_path]


def _prepare_full_contract_soft_wall_owner(
    case: AbCase, bundled_dir: Path
) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    protocol_path = bundled_dir / "protocol.spgp.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_root = "/wall/soft"
    with h5py.File(protocol_path, "r+") as protocol:
        typed_owner = typed_root in protocol
        sidecar_owner = False
        if sidecar_root in protocol:
            sidecar_keys = protocol[f"{sidecar_root}/key"].asstr()[...]
            sidecar_owner = "soft_walls_in_file" in sidecar_keys

        if "input.full_contract.pure_native" in case.contract_ids:
            if not typed_owner or sidecar_owner:
                raise AssertionError(
                    f"{case.name} pure full-contract fixture requires only "
                    "the typed soft-wall owner"
                )
            return
        if (
            "input.full_contract.sidecar" not in case.contract_ids
            and not sidecar_owner
        ):
            return
        if not typed_owner or not sidecar_owner:
            raise AssertionError(
                f"{case.name} sidecar source fixture must contain both "
                "soft-wall representations before owner selection"
            )
        del protocol[typed_root]


def _prepare_full_contract_cv_restraint_owner(
    case: AbCase, bundled_dir: Path
) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    protocol_path = bundled_dir / "protocol.spgp.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_roots = ("/restraint/config", "/restraint/cv/config")
    required_sidecar_keys = {"restrain_in_file", "restrain_cv_in_file"}
    with h5py.File(protocol_path, "r+") as protocol:
        typed_parts = [path in protocol for path in typed_roots]
        sidecar_keys: set[str] = set()
        if sidecar_root in protocol:
            sidecar_keys = set(protocol[f"{sidecar_root}/key"].asstr()[...])
        sidecar_parts = [key in sidecar_keys for key in required_sidecar_keys]
        typed_owner = all(typed_parts)
        partial_typed_owner = any(typed_parts)
        sidecar_owner = all(sidecar_parts)
        partial_sidecar_owner = any(sidecar_parts)

        if partial_typed_owner and not typed_owner:
            raise AssertionError(
                f"{case.name} source fixture has an incomplete typed "
                "CV-restraint owner"
            )
        if partial_sidecar_owner and not sidecar_owner:
            raise AssertionError(
                f"{case.name} source fixture has an incomplete sidecar "
                "CV-restraint owner"
            )
        if "input.full_contract.pure_native" in case.contract_ids:
            if not typed_owner or partial_sidecar_owner:
                raise AssertionError(
                    f"{case.name} pure full-contract fixture requires only "
                    "the typed CV-restraint owner"
                )
            return
        if (
            "input.full_contract.sidecar" not in case.contract_ids
            and not partial_sidecar_owner
        ):
            return
        if not typed_owner or not sidecar_owner:
            raise AssertionError(
                f"{case.name} sidecar source fixture must contain both "
                "CV-restraint representations before owner selection"
            )
        del protocol["/restraint/cv"]
        del protocol["/restraint/config"]


def _prepare_restart_absent_inputs(legacy_dir: Path, bundled_dir: Path) -> None:
    for file_name in ("coordinate.txt", "velocity.txt"):
        shutil.copy2(legacy_dir / file_name, bundled_dir / file_name)

    bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
    text = _remove_key_lines(
        bundled_mdin.read_text(encoding="utf-8"),
        {"input_h5_restart_path", "input_h5_restart_load"},
    )
    text = _insert_root_toml_keys(
        text,
        [
            'coordinate_in_file = "coordinate.txt"',
            'velocity_in_file = "velocity.txt"',
        ],
    )
    bundled_mdin.write_text(text, encoding="utf-8")

    restart_path = bundled_dir / "restart.spgr.h5"
    if restart_path.exists():
        restart_path.unlink()


def _validate_restart_absent_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    for branch, text in (("legacy", legacy_mdin), ("bundled", bundled_mdin)):
        for key in ("coordinate_in_file", "velocity_in_file"):
            if not _has_key_line(text, key):
                raise AssertionError(
                    f"restart-absent {branch} branch is missing {key} bootstrap"
                )
        for key in ("input_h5_restart_path", "input_h5_restart_load"):
            if _has_key_line(text, key):
                raise AssertionError(
                    f"restart-absent {branch} branch retained {key}"
                )

    for file_name in ("coordinate.txt", "velocity.txt"):
        legacy_payload = (legacy_dir / file_name).read_bytes()
        bundled_payload = (bundled_dir / file_name).read_bytes()
        if legacy_payload != bundled_payload:
            raise AssertionError(
                f"restart-absent bootstrap differs for {file_name}"
            )
    if (bundled_dir / "restart.spgr.h5").exists():
        raise AssertionError(
            "restart-absent bundled branch retained restart H5"
        )


def _prepare_rerun_trajectory_variant(case: AbCase, bundled_dir: Path) -> None:
    if (
        case.rerun_velocity_present
        and case.trajectory_particle_stream == "all"
        and case.trajectory_file_name == "trajectory.spg.h5md"
    ):
        return

    source = bundled_dir / "trajectory.spg.h5md"
    destination = bundled_dir / case.trajectory_file_name
    if destination == source:
        raise AssertionError(
            f"{case.name} trajectory variant must not overwrite its source"
        )
    if destination.exists():
        destination.unlink()
    h5copy = shutil.which("h5copy")
    if h5copy is None:
        raise AssertionError("h5copy is required for rerun trajectory variants")

    def copy(source_path: str, destination_path: str, *, parents: bool = False):
        command = [
            h5copy,
            "-i",
            source,
            "-o",
            destination,
            "-s",
            source_path,
            "-d",
            destination_path,
        ]
        if parents:
            command.insert(1, "-p")
        _run(command)

    copy("/h5md", "/h5md")
    copy("/parameters", "/parameters")
    source_stream = "/particles/all"
    destination_stream = f"/particles/{case.trajectory_particle_stream}"
    copy(
        f"{source_stream}/step",
        f"{destination_stream}/step",
        parents=True,
    )
    copy(
        f"{source_stream}/time",
        f"{destination_stream}/time",
        parents=True,
    )
    copy(
        f"{source_stream}/box",
        f"{destination_stream}/box",
        parents=True,
    )
    copy(
        f"{source_stream}/position",
        f"{destination_stream}/position",
        parents=True,
    )
    if case.rerun_velocity_present:
        copy(
            f"{source_stream}/velocity",
            f"{destination_stream}/velocity",
            parents=True,
        )


def _prepare_full_contract_finite_rerun_frames(
    case: AbCase, legacy_dir: Path, bundled_dir: Path
) -> None:
    if case.fixture_case != "full_contract_rerun":
        return
    legacy_path = legacy_dir / "traj.dat"
    payload = legacy_path.read_bytes()
    if len(payload) != 2 * 2 * 3 * 4:
        raise AssertionError(
            f"{case.name} full-contract trajectory shape changed: "
            f"bytes={len(payload)}"
        )
    legacy_values = [item[0] for item in struct.iter_unpack("f", payload)]
    legacy_values[8] = legacy_values[2]
    legacy_values[11] = legacy_values[5]
    legacy_path.write_bytes(
        struct.pack(f"{len(legacy_values)}f", *legacy_values)
    )

    trajectory_path = bundled_dir / "trajectory.spg.h5md"
    with h5py.File(trajectory_path, "r+") as trajectory:
        position = trajectory["/particles/all/position/value"]
        if position.shape != (2, 2, 3):
            raise AssertionError(
                f"{case.name} bundled full-contract trajectory shape changed: "
                f"{position.shape}"
            )
        position[1, :, 2] = position[0, :, 2]
        bundled_values = [float(value) for value in position[...].flat]
    _assert_numeric_sequences_close(
        f"{case.name} finite full-contract trajectory synchronization",
        legacy_values,
        bundled_values,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )


def _prepare_normal_tip3p_pair(
    case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    source = (
        REPO_ROOT
        / "benchmarks"
        / "validation"
        / "thermostat"
        / "statics"
        / "tip3p_water"
    )
    legacy_source = case_root / "generated_legacy_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(source, legacy_source)
    _reset_xponge_coordinate_start_time(legacy_source / "tip3p_coordinate.txt")
    masses = read_mass_values(legacy_source / "tip3p_mass.txt")
    constrain_pair_count = Extractor.read_first_field_int(
        legacy_source / "tip3p_bond.txt"
    )
    write_velocity_file_for_temperature(
        legacy_source / "initial_velocity.txt",
        masses,
        temperature=300.0,
        seed=replica_seed,
        degrees_of_freedom=3 * len(masses) - constrain_pair_count,
    )
    _write_tip3p_mdin(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_sits_ff19sb_cmap_pair(
    case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    legacy_source = case_root / "sits_ff19sb_cmap_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_sits_ff19sb_cmap_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(SITS_FF19SB_CMAP_SOURCE, legacy_source)
    _write_sits_ff19sb_cmap_mdin(legacy_source, replica_seed)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    return legacy_dir, bundled_dir


FOCUSED_BONDED_MODULES = {
    FOCUSED_BOND_FIXTURE: "bond",
    FOCUSED_ANGLE_FIXTURE: "angle",
    FOCUSED_UREY_BRADLEY_FIXTURE: "urey_bradley",
    FOCUSED_DIHEDRAL_FIXTURE: "dihedral",
    FOCUSED_LISTED_FIXTURE: "listed",
}


def _prepare_focused_bonded_pair(
    fixture_case: str, case_root: Path
) -> tuple[Path, Path]:
    module = FOCUSED_BONDED_MODULES[fixture_case]
    legacy_source = case_root / f"focused_{module}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_focused_{module}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_bonded_input(legacy_source, module)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    if module == "listed":
        bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
        bundled_mdin.write_text(
            _remove_key_lines(
                bundled_mdin.read_text(encoding="utf-8"),
                {"listed_forces_in_file", "custom_bond_in_file"},
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_bonded_routes(module, legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_bonded_input(case_dir: Path, module: str) -> None:
    coordinates = {
        "bond": ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        "listed": ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        "angle": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        "urey_bradley": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        "dihedral": (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
    }[module]
    case_dir.mkdir(parents=True, exist_ok=True)
    atom_count = len(coordinates)
    (case_dir / "mass.txt").write_text(
        f"{atom_count}\n" + "12.0\n" * atom_count, encoding="utf-8"
    )
    coordinate_lines = [f"{atom_count} 0.0"]
    coordinate_lines.extend(" ".join(map(str, xyz)) for xyz in coordinates)
    coordinate_lines.extend(("1000.0 1000.0 1000.0", "90.0 90.0 90.0"))
    (case_dir / "coordinate.txt").write_text(
        "\n".join(coordinate_lines) + "\n", encoding="utf-8"
    )
    (case_dir / "velocity.txt").write_text(
        f"{atom_count}\n" + "0.0 0.0 0.0\n" * atom_count,
        encoding="utf-8",
    )

    module_lines = []
    if module == "bond":
        (case_dir / "bond.txt").write_text(
            "1\n0 1 10.0 1.5\n", encoding="utf-8"
        )
        module_lines.append('bond_in_file = "bond.txt"')
    elif module == "angle":
        (case_dir / "angle.txt").write_text(
            "1\n0 1 2 2.0 0.5\n", encoding="utf-8"
        )
        module_lines.append('angle_in_file = "angle.txt"')
    elif module == "urey_bradley":
        (case_dir / "urey_bradley.txt").write_text(
            "1\n0 1 2 0.0 1.5707963268 5.0 1.0\n", encoding="utf-8"
        )
        module_lines.append('urey_bradley_in_file = "urey_bradley.txt"')
    elif module == "dihedral":
        (case_dir / "dihedral.txt").write_text(
            "1\n0 1 2 3 1 2.0 0.0\n", encoding="utf-8"
        )
        module_lines.append('dihedral_in_file = "dihedral.txt"')
    elif module == "listed":
        (case_dir / "listed_forces.txt").write_text(
            "[[[ custom_bond ]]]\n"
            "[[ potential ]]\n"
            "E = k * (r_ab - r0) * (r_ab - r0);\n"
            "[[ parameters ]]\n"
            "int atom_a, int atom_b, float k, float r0\n"
            "[[ connected_atoms ]]\n"
            "ab\n"
            "[[ constrain_distance ]]\n"
            "r0\n"
            "[[ end ]]\n",
            encoding="utf-8",
        )
        (case_dir / "custom_bond.txt").write_text(
            "1\n0 1 7.0 1.5\n", encoding="utf-8"
        )
        module_lines.extend(
            (
                'listed_forces_in_file = "listed_forces.txt"',
                'custom_bond_in_file = "custom_bond.txt"',
            )
        )
    else:
        raise AssertionError(f"unknown focused bonded module: {module}")

    mdin_lines = [
        f'md_name = "bundled io ab focused {module}"',
        'mode = "nve"',
        "pbc = false",
        "step_limit = 1",
        "dt = 0.0",
        "cutoff = 8.0",
        'mass_in_file = "mass.txt"',
        'coordinate_in_file = "coordinate.txt"',
        'velocity_in_file = "velocity.txt"',
        *module_lines,
        "force_whole_output = true",
        "print_zeroth_frame = 0",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(mdin_lines) + "\n", encoding="utf-8"
    )


def _validate_focused_bonded_routes(
    module: str, legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_keys = {
        "bond": ("bond_in_file",),
        "angle": ("angle_in_file",),
        "urey_bradley": ("urey_bradley_in_file",),
        "dihedral": ("dihedral_in_file",),
        "listed": ("listed_forces_in_file", "custom_bond_in_file"),
    }[module]
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    for key in legacy_keys:
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(
                f"focused {module} legacy route is missing {key}"
            )
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(
                f"focused {module} bundled route retained {key}"
            )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError(f"focused {module} bundle retained sidecars")

    root = (
        "/forcefield/custom_force/listed"
        if module == "listed"
        else f"/forcefield/{module}"
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    paths = _h5_paths(topology_path)
    required = {root, f"{root}/count"}
    if module == "listed":
        required.update(
            {
                f"{root}/potential",
                f"{root}/data/custom_bond/parameter/value",
            }
        )
    else:
        required.add(f"{root}/atoms")
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"focused {module} typed topology is incomplete: {missing}"
        )


def _prepare_focused_core_topology_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_core_topology_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_core_topology_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_core_topology_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_core_topology_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_nb14_pair(
    case_root: Path, legacy_key: str
) -> tuple[Path, Path]:
    if legacy_key not in {"nb14_in_file", "nb14_extra_in_file"}:
        raise AssertionError(f"unsupported focused NB14 key: {legacy_key}")
    suffix = "scaled" if legacy_key == "nb14_in_file" else "extra"
    legacy_source = case_root / f"focused_nb14_{suffix}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_focused_nb14_{suffix}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_nb14_input(legacy_source, legacy_key)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_nb14_routes(legacy_dir, bundled_dir, legacy_key)
    return legacy_dir, bundled_dir


def _write_focused_nb14_input(case_dir: Path, legacy_key: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n1.0\n-1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n2.0 0.0 0.0\n1000.0 1000.0 1000.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    if legacy_key == "nb14_in_file":
        lj_payload = "2 1\n2048.0\n128.0\n0\n0\n"
        nb14_name = "nb14.txt"
        nb14_payload = "1\n0 1 0.5 0.5\n"
    else:
        lj_payload = "2 1\n4096.0\n128.0\n0\n0\n"
        nb14_name = "nb14_extra.txt"
        nb14_payload = "1\n0 1 1.0 2.0 0.5\n"
    (case_dir / "lj.txt").write_text(lj_payload, encoding="utf-8")
    (case_dir / nb14_name).write_text(nb14_payload, encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused NB14"\n'
        'mode = "nve"\n'
        "pbc = false\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'LJ_in_file = "lj.txt"\n'
        f'{legacy_key} = "{nb14_name}"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_nb14_routes(
    legacy_dir: Path, bundled_dir: Path, legacy_key: str
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, legacy_key):
        raise AssertionError(f"focused NB14 legacy route lost {legacy_key}")
    if _has_key_line(bundled_mdin, legacy_key):
        raise AssertionError(
            f"focused NB14 bundled route retained {legacy_key}"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused NB14 bundled route retained sidecars")

    root = (
        "/forcefield/nb14"
        if legacy_key == "nb14_in_file"
        else "/forcefield/nb14_extra"
    )
    other_root = (
        "/forcefield/nb14_extra"
        if root == "/forcefield/nb14"
        else "/forcefield/nb14"
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r") as topology:
        if other_root in topology:
            raise AssertionError(
                f"focused NB14 retained alternate root {other_root}"
            )
        atoms = topology[f"{root}/atoms"][...].tolist()
        params = topology[f"{root}/params"][...].tolist()
        if "/parameters/sponge/files/legacy_sidecars" in topology:
            raise AssertionError("focused NB14 topology retained sidecar table")
    if atoms != [[0, 1]]:
        raise AssertionError(f"focused NB14 atoms changed: {atoms}")
    expected_params = (
        [[0.5, 0.5]] if legacy_key == "nb14_in_file" else [[1.0, 2.0, 0.5]]
    )
    if params != expected_params:
        raise AssertionError(f"focused NB14 params changed: {params}")


def _write_focused_core_topology_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n1.0\n4.0\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n1.0\n-1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n1.0 0.0 0.0\n3.0 0.0 0.0\n1000.0 1000.0 1000.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.25 0.0 0.0\n-0.25 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "lj.txt").write_text("2 1\n1.0\n1.0\n0\n0\n", encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused core topology"\n'
        'mode = "nve"\n'
        "pbc = false\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'LJ_in_file = "lj.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_core_topology_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    for key in ("mass_in_file", "charge_in_file", "LJ_in_file"):
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(f"focused core legacy route is missing {key}")
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(f"focused core bundled mdin retained {key}")
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused core bundle retained legacy sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    required = {
        "/atoms/mass",
        "/atoms/charge",
        "/forcefield/lj/type",
        "/forcefield/lj/pair_A_12",
        "/forcefield/lj/pair_B_6",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused core topology is missing typed datasets: {missing}"
        )
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused core topology retained a sidecar table")
    with h5py.File(topology_path, "r") as topology:
        masses = topology["/atoms/mass"][...].tolist()
        charges = topology["/atoms/charge"][...].tolist()
        lj_types = topology["/forcefield/lj/type"][...].tolist()
        pair_a = topology["/forcefield/lj/pair_A_12"][...].tolist()
        pair_b = topology["/forcefield/lj/pair_B_6"][...].tolist()
    _assert_numeric_sequences_close(
        "focused core typed masses",
        (1.0, 4.0),
        masses,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-7,
    )
    _assert_numeric_sequences_close(
        "focused core typed charges",
        (1.0, -1.0),
        charges,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-7,
    )
    if lj_types != [0, 0]:
        raise AssertionError(f"focused core LJ types changed: {lj_types}")
    if len(pair_a) != 1 or len(pair_b) != 1:
        raise AssertionError("focused core LJ pair payload changed shape")
    if pair_a[0] <= 0.0 or pair_b[0] <= 0.0:
        raise AssertionError("focused core LJ pair payload is trivial")


def _prepare_focused_eam_pair(
    fixture_case: str, case_root: Path
) -> tuple[Path, Path]:
    if fixture_case not in {
        FOCUSED_EAM_FUNCFL_FIXTURE,
        FOCUSED_EAM_SETFL_FIXTURE,
    }:
        raise AssertionError(f"unsupported focused EAM fixture: {fixture_case}")
    format_name = (
        "funcfl" if fixture_case == FOCUSED_EAM_FUNCFL_FIXTURE else "setfl"
    )
    legacy_source = case_root / f"focused_eam_{format_name}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_focused_eam_{format_name}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_eam_input(legacy_source, format_name)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_eam_routes(legacy_dir, bundled_dir, format_name)
    return legacy_dir, bundled_dir


def _write_focused_eam_input(case_dir: Path, format_name: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n63.55\n63.55\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    if format_name == "funcfl":
        eam_text = (
            "Focused Cu funcfl\n"
            "29 63.55 3.615 FCC\n"
            "4 1.0 8 0.5 4.0\n"
            "0.0 1.0 1.0 1.0\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
        )
        atom_type_text = "0\n0\n"
    elif format_name == "setfl":
        pair_row = " ".join(["1.0"] * 8)
        eam_text = (
            "Focused AB setfl\n"
            "comment\n"
            "comment\n"
            "2 A B\n"
            "4 1.0 8 0.5 4.0\n"
            "29 63.55 3.615 FCC\n"
            "0.0 1.0 1.0 1.0\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
            "28 58.69 3.52 FCC\n"
            "0.0 1.0 1.0 1.0\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
            f"{pair_row}\n{pair_row}\n{pair_row}\n"
        )
        atom_type_text = "0\n1\n"
    else:
        raise AssertionError(f"unsupported focused EAM format: {format_name}")
    (case_dir / "eam.txt").write_text(eam_text, encoding="utf-8")
    (case_dir / "eam_atom_type.txt").write_text(
        atom_type_text, encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused EAM"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'EAM_in_file = "eam.txt"\n'
        'EAM_atom_type_in_file = "eam_atom_type.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_eam_routes(
    legacy_dir: Path, bundled_dir: Path, format_name: str
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    for key in ("EAM_in_file", "EAM_atom_type_in_file"):
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(f"focused EAM legacy branch lost {key}")
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(f"focused EAM bundled branch retained {key}")
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused EAM bundled branch retained sidecars")
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r") as topology:
        actual_format = topology["/manybody/eam/format"].asstr()[()]
        if actual_format != format_name:
            raise AssertionError(
                f"focused EAM format changed: {actual_format!r}"
            )
        required = {
            "/manybody/eam/atom_type",
            "/manybody/eam/embed/raw_ev",
            "/manybody/eam/electron_density/value",
            "/manybody/eam/pair_potential/value",
        }
        missing = sorted(path for path in required if path not in topology)
        if missing:
            raise AssertionError(
                f"focused EAM bundled topology is missing datasets: {missing}"
            )


def _prepare_focused_positional_restraint_pair(
    case_root: Path, *, typed: bool
) -> tuple[Path, Path]:
    route = "typed" if typed else "sidecar"
    legacy_source = case_root / f"focused_positional_restraint_{route}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_positional_restraint_{route}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_positional_restraint_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    reference_path = (
        "/parameters/restart/references/restraint/default/coordinate"
    )
    if typed:
        for file_name in ("protocol.spgp.h5", "restart.spgr.h5"):
            with h5py.File(bundled_dir / file_name, "r+") as bundle:
                for sidecar_root in (
                    "/parameters/sponge/files/legacy_sidecars",
                    "/parameters/restart/protocol_sidecars",
                ):
                    if sidecar_root in bundle:
                        del bundle[sidecar_root]
        sidecar_dir = bundled_dir / "legacy_sidecars"
        if sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
    else:
        with h5py.File(bundled_dir / "protocol.spgp.h5", "r+") as protocol:
            if "/restraint/default" in protocol:
                del protocol["/restraint/default"]
        with h5py.File(bundled_dir / "restart.spgr.h5", "r+") as restart:
            if reference_path in restart:
                del restart[reference_path]
    _validate_focused_positional_restraint_routes(
        legacy_dir, bundled_dir, typed=typed
    )
    return legacy_dir, bundled_dir


def _write_focused_positional_restraint_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n1.0\n1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n2.0 0.0 0.0\n0.0 1.5 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "restrain_atom_id.txt").write_text("0\n1\n", encoding="utf-8")
    (case_dir / "restrain_weight.txt").write_text(
        "10.0 0.0 0.0\n0.0 20.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "restrain_coordinate.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused positional restraint"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'restrain_atom_id = "restrain_atom_id.txt"\n'
        'restrain_weight_in_file = "restrain_weight.txt"\n'
        'restrain_coordinate_in_file = "restrain_coordinate.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_positional_restraint_routes(
    legacy_dir: Path, bundled_dir: Path, *, typed: bool
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    legacy_keys = (
        "restrain_atom_id",
        "restrain_weight_in_file",
        "restrain_coordinate_in_file",
    )
    for key in legacy_keys:
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(
                f"focused positional restraint legacy branch lost {key}"
            )
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(
                f"focused positional restraint bundled branch retained {key}"
            )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    restart_path = bundled_dir / "restart.spgr.h5"
    reference_path = (
        "/parameters/restart/references/restraint/default/coordinate"
    )
    if not typed:
        with (
            h5py.File(protocol_path, "r") as protocol,
            h5py.File(restart_path, "r") as restart,
        ):
            if "/restraint/default" in protocol or reference_path in restart:
                raise AssertionError(
                    "focused positional-restraint sidecar route retained "
                    "typed payload"
                )
            bindings = {}
            sidecar_root = "/parameters/sponge/files/legacy_sidecars"
            for bundle in (protocol, restart):
                if sidecar_root not in bundle:
                    continue
                keys = bundle[f"{sidecar_root}/key"].asstr()[...].tolist()
                paths = bundle[f"{sidecar_root}/path"].asstr()[...].tolist()
                bindings.update(zip(keys, paths, strict=True))
        expected = {
            "restrain_atom_id": (
                "legacy_sidecars/restrain_atom_id/restrain_atom_id.txt"
            ),
            "restrain_weight_in_file": (
                "legacy_sidecars/restrain_weight_in_file/restrain_weight.txt"
            ),
            "restrain_coordinate_in_file": (
                "legacy_sidecars/restrain_coordinate_in_file/"
                "restrain_coordinate.txt"
            ),
        }
        actual = {key: bindings.get(key) for key in expected}
        if actual != expected:
            raise AssertionError(
                "focused positional-restraint sidecar bindings changed: "
                f"{actual}"
            )
        for key, relative_path in expected.items():
            source_name = Path(relative_path).name
            if (legacy_dir / source_name).read_bytes() != (
                bundled_dir / relative_path
            ).read_bytes():
                raise AssertionError(
                    "focused positional-restraint sidecar payload changed "
                    f"for {key}"
                )
        return

    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError(
            "focused positional restraint bundled branch retained sidecars"
        )
    with h5py.File(protocol_path, "r") as protocol:
        atom_indices = protocol["/restraint/default/atom_indices"][...].tolist()
        weight = protocol["/restraint/default/weight"][...].tolist()
        if atom_indices != [0, 1]:
            raise AssertionError(
                f"focused positional restraint atom IDs changed: {atom_indices}"
            )
        if weight != [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]:
            raise AssertionError(
                f"focused positional restraint weights changed: {weight}"
            )
        if "/parameters/sponge/files/legacy_sidecars" in protocol:
            raise AssertionError(
                "focused positional restraint protocol retained sidecars"
            )
    with h5py.File(restart_path, "r") as restart:
        reference = restart[reference_path][...].tolist()
        if reference != [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]:
            raise AssertionError(
                "focused positional restraint reference coordinates changed: "
                f"{reference}"
            )


def _prepare_focused_soft_wall_pair(
    case_root: Path, *, typed: bool
) -> tuple[Path, Path]:
    route = "typed" if typed else "sidecar"
    legacy_source = case_root / f"focused_soft_wall_{route}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_soft_wall_{route}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_soft_wall_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    with h5py.File(bundled_dir / "protocol.spgp.h5", "r+") as protocol:
        if typed:
            for sidecar_root in (
                "/parameters/sponge/files/legacy_sidecars",
                "/parameters/restart/protocol_sidecars",
            ):
                if sidecar_root in protocol:
                    del protocol[sidecar_root]
        elif "/wall/soft" in protocol:
            del protocol["/wall/soft"]
    if typed:
        sidecar_dir = bundled_dir / "legacy_sidecars"
        if sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
    _validate_focused_soft_wall_routes(legacy_dir, bundled_dir, typed=typed)
    return legacy_dir, bundled_dir


def _write_focused_soft_wall_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n1.0\n1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 2.0\n0.0 0.0 4.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "soft_walls.txt").write_text(
        "[[[ z_wall ]]]\n"
        "[[ potential ]]\n"
        "E = (z - 1.0f) * (z - 1.0f);\n"
        "[[ end ]]\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused soft wall"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'soft_walls_in_file = "soft_walls.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_soft_wall_routes(
    legacy_dir: Path, bundled_dir: Path, *, typed: bool
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "soft_walls_in_file"):
        raise AssertionError("focused soft-wall legacy branch lost its owner")
    if _has_key_line(bundled_mdin, "soft_walls_in_file"):
        raise AssertionError(
            "focused soft-wall bundled branch retained sidecar"
        )
    with h5py.File(bundled_dir / "protocol.spgp.h5", "r") as protocol:
        sidecar_root = "/parameters/sponge/files/legacy_sidecars"
        if not typed:
            if "/wall/soft" in protocol:
                raise AssertionError(
                    "focused soft-wall sidecar route retained typed payload"
                )
            keys = protocol[f"{sidecar_root}/key"].asstr()[...].tolist()
            paths = protocol[f"{sidecar_root}/path"].asstr()[...].tolist()
            bindings = dict(zip(keys, paths, strict=True))
            expected = "legacy_sidecars/soft_walls_in_file/soft_walls.txt"
            if bindings.get("soft_walls_in_file") != expected:
                raise AssertionError(
                    "focused soft-wall sidecar binding changed: "
                    f"{bindings.get('soft_walls_in_file')!r}"
                )
            if (legacy_dir / "soft_walls.txt").read_bytes() != (
                bundled_dir / expected
            ).read_bytes():
                raise AssertionError(
                    "focused soft-wall sidecar payload changed"
                )
            return

        if (bundled_dir / "legacy_sidecars").exists():
            raise AssertionError(
                "focused soft-wall bundled branch retained sidecars"
            )
        count = int(protocol["/wall/soft/count"][()])
        names = protocol["/wall/soft/name"].asstr()[...].tolist()
        potentials = protocol["/wall/soft/potential"].asstr()[...].tolist()
        if count != 1 or names != ["z_wall"]:
            raise AssertionError(
                "focused soft-wall typed identity changed: "
                f"count={count}, names={names}"
            )
        expected = "E = (z - 1.0f) * (z - 1.0f);"
        if potentials != [expected]:
            raise AssertionError(
                f"focused soft-wall potential changed: {potentials}"
            )
        if sidecar_root in protocol:
            raise AssertionError("focused soft-wall protocol retained sidecars")


def _prepare_focused_cv_restraint_pair(
    case_root: Path, *, typed: bool
) -> tuple[Path, Path]:
    route = "typed" if typed else "sidecar"
    legacy_source = case_root / f"focused_cv_restraint_{route}_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / f"converted_cv_restraint_{route}_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_cv_restraint_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_sidecars = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if topology_sidecars in topology:
            del topology[topology_sidecars]
    mass_sidecar = bundled_dir / "legacy_sidecars" / "mass_in_file"
    if mass_sidecar.exists():
        shutil.rmtree(mass_sidecar)

    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_sidecars = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(protocol_path, "r+") as protocol:
        if typed:
            if protocol_sidecars in protocol:
                del protocol[protocol_sidecars]
        elif "/restraint" in protocol:
            del protocol["/restraint"]
    if typed:
        sidecar_dir = bundled_dir / "legacy_sidecars"
        if sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
    _validate_focused_cv_restraint_routes(legacy_dir, bundled_dir, typed=typed)
    return legacy_dir, bundled_dir


def _write_focused_cv_restraint_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n1.0\n1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n2.0 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "restrain.txt").write_text(
        "distance\n{\n    CV_type = distance\n    atom = 0 1\n}\n",
        encoding="utf-8",
    )
    (case_dir / "restrain_cv.txt").write_text(
        "restrain\n"
        "{\n"
        "    CV = distance\n"
        "    weight = 4.0\n"
        "    reference = 1.5\n"
        "}\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused CV restraint"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'restrain_in_file = "restrain.txt"\n'
        'restrain_cv_in_file = "restrain_cv.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_cv_restraint_routes(
    legacy_dir: Path, bundled_dir: Path, *, typed: bool
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    owner_keys = ("restrain_in_file", "restrain_cv_in_file")
    for key in owner_keys:
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(f"focused CV-restraint legacy lost {key}")
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(f"focused CV-restraint bundled retained {key}")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_paths = {
        "/restraint/config/section/count",
        "/restraint/config/section/name",
        "/restraint/config/section/key_offset",
        "/restraint/config/key",
        "/restraint/config/value",
        "/restraint/cv/config/section/count",
        "/restraint/cv/config/section/name",
        "/restraint/cv/config/section/key_offset",
        "/restraint/cv/config/key",
        "/restraint/cv/config/value",
    }
    protocol_paths = _h5_paths(protocol_path)
    if typed:
        missing = sorted(typed_paths - protocol_paths)
        if missing:
            raise AssertionError(
                f"focused typed CV restraint is missing datasets: {missing}"
            )
        if sidecar_root in protocol_paths:
            raise AssertionError("focused typed CV restraint retained sidecars")
        if (bundled_dir / "legacy_sidecars").exists():
            raise AssertionError(
                "focused typed CV restraint retained sidecar files"
            )
        return

    if typed_paths & protocol_paths:
        raise AssertionError("focused sidecar CV restraint retained typed data")
    keys = _h5_string_values(protocol_path, f"{sidecar_root}/key")
    paths = _h5_string_values(protocol_path, f"{sidecar_root}/path")
    expected = {
        "restrain_in_file": ("legacy_sidecars/restrain_in_file/restrain.txt"),
        "restrain_cv_in_file": (
            "legacy_sidecars/restrain_cv_in_file/restrain_cv.txt"
        ),
    }
    actual = dict(zip(keys, paths, strict=True))
    if actual != expected:
        raise AssertionError(
            f"focused CV-restraint sidecar bindings changed: {actual}"
        )
    for key, relative_path in expected.items():
        source_name = (
            "restrain.txt" if key == "restrain_in_file" else "restrain_cv.txt"
        )
        if (legacy_dir / source_name).read_bytes() != (
            bundled_dir / relative_path
        ).read_bytes():
            raise AssertionError(
                f"focused CV-restraint sidecar payload changed for {key}"
            )


def _prepare_focused_edip_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_edip_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_edip_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_edip_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_edip_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_edip_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "2\n28.0855\n28.0855\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "edip.txt").write_text(
        "2 1\n"
        "# pair\n"
        "0 0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n"
        "# triple\n"
        "0 0 0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0\n"
        "# atom types\n"
        "0\n"
        "0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused edip"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'EDIP_in_file = "edip.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_edip_routes(legacy_dir: Path, bundled_dir: Path) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "EDIP_in_file"):
        raise AssertionError("focused EDIP legacy branch lost EDIP_in_file")
    if _has_key_line(bundled_mdin, "EDIP_in_file"):
        raise AssertionError(
            "focused EDIP bundled branch retained EDIP_in_file"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused EDIP bundled branch retained sidecars")
    topology_paths = _h5_paths(bundled_dir / "topology.spgt.h5")
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused EDIP bundled topology retained sidecars")
    required = {
        "/manybody/edip/atom_type",
        "/manybody/edip/pair/type",
        "/manybody/edip/pair/parameters",
        "/manybody/edip/triple/type",
        "/manybody/edip/triple/parameters",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused EDIP bundled topology is missing datasets: {missing}"
        )


def _prepare_focused_reaxff_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_reaxff_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_reaxff_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_reaxff_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_reaxff_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_reaxff_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    source = FIXTURE_ROOT / "full_contract_rerun" / "legacy_input"
    shutil.copy2(source / "reaxff.txt", case_dir / "reaxff.txt")
    shutil.copy2(source / "reaxff_type.txt", case_dir / "reaxff_type.txt")
    (case_dir / "mass.txt").write_text("2\n15.999\n1.008\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n0.0\n0.0\n", encoding="utf-8")
    (case_dir / "exclude.txt").write_text("2 0\n0\n0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused ReaxFF"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 2.0\n"
        "skin = 0.5\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'exclude_in_file = "exclude.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'REAXFF_in_file = "reaxff.txt"\n'
        'REAXFF_type_in_file = "reaxff_type.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_reaxff_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "REAXFF_in_file") or not _has_key_line(
        legacy_mdin, "REAXFF_type_in_file"
    ):
        raise AssertionError("focused ReaxFF legacy branch lost its owner")
    for key in ("REAXFF_in_file", "REAXFF_type_in_file"):
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(
                f"focused ReaxFF bundled branch retained {key}"
            )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused ReaxFF bundled branch retained sidecars")
    paths = _h5_paths(bundled_dir / "topology.spgt.h5")
    required = {
        "/manybody/reaxff/parameters",
        "/manybody/reaxff/parameters/general/value",
        "/manybody/reaxff/parameters/atom/value",
        "/manybody/reaxff/parameters/bond/value",
        "/manybody/reaxff/type/count",
        "/manybody/reaxff/type/name",
    }
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"focused ReaxFF bundled topology is missing datasets: {missing}"
        )


def _prepare_focused_sw_sidecar_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_sw_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_sw_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_sw_sidecar_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_table = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_table not in topology:
            raise AssertionError("focused SW conversion did not emit sidecars")
        sidecars = topology[sidecar_table]
        keys = sidecars["key"].asstr()[...].tolist()
        paths = sidecars["path"].asstr()[...].tolist()
        try:
            sw_index = keys.index("SW_in_file")
        except ValueError as error:
            raise AssertionError(
                "focused SW conversion did not bind SW_in_file"
            ) from error
        del sidecars["key"]
        del sidecars["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset("key", data=["SW_in_file"], dtype=string_dtype)
        sidecars.create_dataset(
            "path", data=[paths[sw_index]], dtype=string_dtype
        )
        if "/manybody/sw" in topology:
            del topology["/manybody/sw"]

    mass_sidecar = bundled_dir / "legacy_sidecars" / "mass_in_file"
    if mass_sidecar.exists():
        shutil.rmtree(mass_sidecar)
    _validate_focused_sw_sidecar_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_sw_typed_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_sw_sidecar_pair(case_root)
    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_root not in topology:
            raise AssertionError(
                "focused SW conversion did not emit the source sidecar"
            )
        del topology[sidecar_root]
        sw = topology.require_group("/manybody/sw")
        sw.create_dataset("atom_type_count", data=1, dtype="i4")
        sw.create_dataset("atom_type", data=[0, 0, 0], dtype="i4")
        pair = sw.require_group("pair")
        pair.create_dataset("type", data=[[0, 0]], dtype="i4")
        pair.create_dataset(
            "parameters",
            data=[
                [
                    7.917,
                    0.767446,
                    27.2658,
                    4.0,
                    0.0,
                    1.527956,
                    1.2,
                    2.663951,
                ]
            ],
            dtype="f4",
        )
        triple = sw.require_group("triple")
        triple.create_dataset("type", data=[[0, 0, 0]], dtype="i4")
        triple.create_dataset(
            "parameters",
            data=[[32.5, 27.2658, -0.333333333333]],
            dtype="f4",
        )

    sidecars = bundled_dir / "legacy_sidecars"
    if sidecars.exists():
        shutil.rmtree(sidecars)
    _validate_focused_sw_typed_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_sw_sidecar_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "3\n28.0855\n28.0855\n28.0855\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "3 0.0\n"
        "0.0 0.0 0.0\n"
        "2.0 0.0 0.0\n"
        "0.0 2.0 0.0\n"
        "10.0 10.0 10.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "3\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "sw.txt").write_text(
        "3 1\n"
        "# two-body\n"
        "0 0 7.917 0.767446 27.2658 4.0 0.0 1.527956 1.2 2.663951\n"
        "# three-body\n"
        "0 0 0 32.5 27.2658 -0.333333333333\n"
        "# atom types\n"
        "0 0 0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused SW sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'SW_in_file = "sw.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_sw_sidecar_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "SW_in_file"):
        raise AssertionError("focused SW legacy branch lost SW_in_file")
    if _has_key_line(bundled_mdin, "SW_in_file"):
        raise AssertionError("focused SW bundled mdin retained SW_in_file")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if any(path.startswith("/manybody/sw") for path in topology_paths):
        raise AssertionError("focused SW bundled topology retained typed SW")
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
    paths = _h5_string_values(topology_path, f"{sidecar_root}/path")
    if keys != ["SW_in_file"]:
        raise AssertionError(f"focused SW bundled sidecar keys changed: {keys}")
    if len(paths) != 1 or not paths[0].endswith("/SW_in_file/sw.txt"):
        raise AssertionError(
            f"focused SW bundled sidecar path changed: {paths}"
        )
    sw_sidecar = bundled_dir / paths[0]
    if not sw_sidecar.is_file() or sw_sidecar.stat().st_size == 0:
        raise AssertionError("focused SW bundled sidecar payload is missing")
    if (bundled_dir / "legacy_sidecars" / "mass_in_file").exists():
        raise AssertionError("focused SW bundled branch retained mass sidecar")


def _validate_focused_sw_typed_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "SW_in_file"):
        raise AssertionError("focused typed SW legacy branch lost SW_in_file")
    if _has_key_line(bundled_mdin, "SW_in_file"):
        raise AssertionError("focused typed SW bundle retained SW_in_file")
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused typed SW bundle retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r") as topology:
        if sidecar_root in topology:
            raise AssertionError(
                "focused typed SW topology retained a sidecar table"
            )
        atom_type_count = int(topology["/manybody/sw/atom_type_count"][()])
        atom_type = topology["/manybody/sw/atom_type"][...].tolist()
        pair_type = topology["/manybody/sw/pair/type"][...].tolist()
        pair_parameters = topology["/manybody/sw/pair/parameters"][...].tolist()
        triple_type = topology["/manybody/sw/triple/type"][...].tolist()
        triple_parameters = topology["/manybody/sw/triple/parameters"][
            ...
        ].tolist()
    if atom_type_count != 1 or atom_type != [0, 0, 0]:
        raise AssertionError(
            "focused typed SW atom-type payload changed: "
            f"count={atom_type_count}, values={atom_type}"
        )
    if pair_type != [[0, 0]] or len(pair_parameters) != 1:
        raise AssertionError("focused typed SW pair payload changed")
    if triple_type != [[0, 0, 0]] or len(triple_parameters) != 1:
        raise AssertionError("focused typed SW triple payload changed")
    _assert_numeric_sequences_close(
        "focused typed SW pair parameters",
        (7.917, 0.767446, 27.2658, 4.0, 0.0, 1.527956, 1.2, 2.663951),
        pair_parameters[0],
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-7,
    )
    _assert_numeric_sequences_close(
        "focused typed SW triple parameters",
        (32.5, 27.2658, -0.333333333333),
        triple_parameters[0],
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-7,
    )


def _prepare_focused_tersoff_sidecar_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_tersoff_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_tersoff_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_tersoff_sidecar_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_table = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_table not in topology:
            raise AssertionError(
                "focused Tersoff conversion did not emit sidecars"
            )
        sidecars = topology[sidecar_table]
        keys = sidecars["key"].asstr()[...].tolist()
        paths = sidecars["path"].asstr()[...].tolist()
        try:
            tersoff_index = keys.index("TERSOFF_in_file")
        except ValueError as error:
            raise AssertionError(
                "focused Tersoff conversion did not bind TERSOFF_in_file"
            ) from error
        del sidecars["key"]
        del sidecars["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset(
            "key", data=["TERSOFF_in_file"], dtype=string_dtype
        )
        sidecars.create_dataset(
            "path", data=[paths[tersoff_index]], dtype=string_dtype
        )
        if "/manybody/tersoff" in topology:
            del topology["/manybody/tersoff"]

    mass_sidecar = bundled_dir / "legacy_sidecars" / "mass_in_file"
    if mass_sidecar.exists():
        shutil.rmtree(mass_sidecar)
    _validate_focused_tersoff_sidecar_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_tersoff_typed_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_tersoff_sidecar_pair(case_root)
    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_root not in topology:
            raise AssertionError(
                "focused Tersoff conversion did not emit source sidecars"
            )
        del topology[sidecar_root]
        tersoff = topology.require_group("/manybody/tersoff")
        tersoff.create_dataset("atom_type_count", data=1, dtype="i4")
        tersoff.create_dataset("atom_type", data=[0, 0, 0], dtype="i4")
        tersoff.create_dataset("map", data=[0], dtype="i4")
        tersoff.create_dataset("type_name", data=["Si"], dtype=string_dtype)
        entry = tersoff.require_group("entry")
        entry.create_dataset("count", data=1, dtype="i8")
        entry.create_dataset("type", data=[[0, 0, 0]], dtype="i4")
        entry.create_dataset(
            "type_name",
            data=[["Si", "Si", "Si"]],
            dtype=string_dtype,
        )
        entry.create_dataset(
            "parameters_raw",
            data=[
                [
                    3.0,
                    1.0,
                    0.0,
                    25000.0,
                    4.3484,
                    -0.89,
                    0.72751,
                    0.000000125724,
                    2.199,
                    340.0,
                    1.95,
                    0.05,
                    3.568,
                    1380.0,
                ]
            ],
            dtype="f4",
        )
        entry.create_dataset(
            "parameters",
            data=[
                [
                    3.0,
                    1.0,
                    0.0,
                    25000.0,
                    4.3484,
                    -0.89,
                    0.72751,
                    0.000000125724,
                    2.199,
                    7840.58642578125,
                    1.95,
                    0.05,
                    3.568,
                    31823.556640625,
                    5.874280307059868e21,
                    59229962240.0,
                    1.6883347150886685e-11,
                    1.702336211264742e-22,
                ]
            ],
            dtype="f4",
        )

    sidecars = bundled_dir / "legacy_sidecars"
    if sidecars.exists():
        shutil.rmtree(sidecars)
    _validate_focused_tersoff_typed_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_tersoff_sidecar_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "3\n28.0855\n28.0855\n28.0855\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "3 0.0\n"
        "0.0 0.0 0.0\n"
        "1.8 0.0 0.0\n"
        "0.0 1.8 0.0\n"
        "10.0 10.0 10.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "3\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "tersoff.txt").write_text(
        "3 1\n"
        "Si\n"
        "Si Si Si 3.0 1.0 0.0 25000.0 4.3484 -0.89 0.72751 "
        "0.000000125724 2.199 340.0 1.95 0.05 3.568 1380.0\n"
        "# Atom types\n"
        "0 0 0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused Tersoff sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'TERSOFF_in_file = "tersoff.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_tersoff_sidecar_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "TERSOFF_in_file"):
        raise AssertionError(
            "focused Tersoff legacy branch lost TERSOFF_in_file"
        )
    if _has_key_line(bundled_mdin, "TERSOFF_in_file"):
        raise AssertionError(
            "focused Tersoff bundled mdin retained TERSOFF_in_file"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if any(path.startswith("/manybody/tersoff") for path in topology_paths):
        raise AssertionError(
            "focused Tersoff bundled topology retained typed Tersoff"
        )
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
    paths = _h5_string_values(topology_path, f"{sidecar_root}/path")
    if keys != ["TERSOFF_in_file"]:
        raise AssertionError(
            f"focused Tersoff bundled sidecar keys changed: {keys}"
        )
    if len(paths) != 1 or not paths[0].endswith("/TERSOFF_in_file/tersoff.txt"):
        raise AssertionError(
            f"focused Tersoff bundled sidecar path changed: {paths}"
        )
    tersoff_sidecar = bundled_dir / paths[0]
    if not tersoff_sidecar.is_file() or tersoff_sidecar.stat().st_size == 0:
        raise AssertionError(
            "focused Tersoff bundled sidecar payload is missing"
        )
    if (bundled_dir / "legacy_sidecars" / "mass_in_file").exists():
        raise AssertionError(
            "focused Tersoff bundled branch retained mass sidecar"
        )


def _validate_focused_tersoff_typed_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "TERSOFF_in_file"):
        raise AssertionError("focused typed Tersoff legacy route is missing")
    if _has_key_line(bundled_mdin, "TERSOFF_in_file"):
        raise AssertionError("focused typed Tersoff retained TERSOFF_in_file")
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused typed Tersoff retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r") as topology:
        if sidecar_root in topology:
            raise AssertionError(
                "focused typed Tersoff topology retained a sidecar table"
            )
        atom_type_count = int(topology["/manybody/tersoff/atom_type_count"][()])
        atom_type = topology["/manybody/tersoff/atom_type"][...].tolist()
        type_map = topology["/manybody/tersoff/map"][...].tolist()
        type_name = (
            topology["/manybody/tersoff/type_name"].asstr()[...].tolist()
        )
        entry_count = int(topology["/manybody/tersoff/entry/count"][()])
        entry_type = topology["/manybody/tersoff/entry/type"][...].tolist()
        entry_type_name = (
            topology["/manybody/tersoff/entry/type_name"].asstr()[...].tolist()
        )
        parameters = topology["/manybody/tersoff/entry/parameters_raw"][
            ...
        ].tolist()
        native_parameters = topology["/manybody/tersoff/entry/parameters"][
            ...
        ].tolist()
    if atom_type_count != 1 or atom_type != [0, 0, 0]:
        raise AssertionError("focused typed Tersoff atom-type payload changed")
    if type_map != [0] or type_name != ["Si"]:
        raise AssertionError("focused typed Tersoff type mapping changed")
    if (
        entry_count != 1
        or entry_type != [[0, 0, 0]]
        or entry_type_name != [["Si", "Si", "Si"]]
        or len(parameters) != 1
        or len(native_parameters) != 1
    ):
        raise AssertionError("focused typed Tersoff entry payload changed")
    _assert_numeric_sequences_close(
        "focused typed Tersoff parameters",
        (
            3.0,
            1.0,
            0.0,
            25000.0,
            4.3484,
            -0.89,
            0.72751,
            0.000000125724,
            2.199,
            340.0,
            1.95,
            0.05,
            3.568,
            1380.0,
        ),
        parameters[0],
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-7,
    )
    if len(native_parameters[0]) != 18:
        raise AssertionError(
            "focused typed Tersoff native parameter payload changed"
        )


def _prepare_focused_custom_pair_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_custom_pair_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_custom_pair_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_custom_pair_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
    bundled_mdin.write_text(
        _remove_key_lines(
            bundled_mdin.read_text(encoding="utf-8"),
            {"pairwise_force_in_file", "custom_pair_in_file"},
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_custom_pair_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_custom_pair_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "pairwise_force.txt").write_text(
        "[[[ custom_pair ]]]\n"
        "[[ potential ]]\n"
        "E = epsilon_ij * powf(sigma_ij / r_ij, 12.0f);\n"
        "[[ parameters ]]\n"
        "float epsilon_ij, float sigma_ij\n"
        "[[ with_ele ]]\n"
        "false\n"
        "[[ end ]]\n",
        encoding="utf-8",
    )
    (case_dir / "custom_pair.txt").write_text(
        "2 1\n1.0\n2.0\n0\n0\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused custom pair"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'pairwise_force_in_file = "pairwise_force.txt"\n'
        'custom_pair_in_file = "custom_pair.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_custom_pair_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    legacy_keys = {"pairwise_force_in_file", "custom_pair_in_file"}
    if not all(_has_key_line(legacy_mdin, key) for key in legacy_keys):
        raise AssertionError("focused custom-pair legacy routes are incomplete")
    retained = sorted(
        key for key in legacy_keys if _has_key_line(bundled_mdin, key)
    )
    if retained:
        raise AssertionError(
            f"focused custom-pair bundled branch retained legacy keys: {retained}"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused custom-pair bundle retained sidecars")
    topology_paths = _h5_paths(bundled_dir / "topology.spgt.h5")
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused custom-pair topology retained sidecars")
    required = {
        "/forcefield/custom_force/pairwise/name",
        "/forcefield/custom_force/pairwise/potential",
        "/forcefield/custom_force/pairwise/parameters/name",
        "/forcefield/custom_force/pairwise/parameters/type",
        "/forcefield/custom_force/pairwise/data/custom_pair/atom_type",
        "/forcefield/custom_force/pairwise/data/custom_pair/parameter/name",
        "/forcefield/custom_force/pairwise/data/custom_pair/parameter/type",
        "/forcefield/custom_force/pairwise/data/custom_pair/parameter/value",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused custom-pair topology is missing datasets: {missing}"
        )


def _prepare_focused_exclusions_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_exclusions_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_exclusions_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_exclusions_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_exclusions_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_exclusions_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "3\n12.0\n12.0\n12.0\n", encoding="utf-8"
    )
    (case_dir / "charge.txt").write_text(
        "3\n1.0\n-1.0\n1.0\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "3 0.0\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "4.0 0.0 0.0\n"
        "1000.0 1000.0 1000.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "3\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "exclude.txt").write_text("3 1\n1 1\n0\n0\n", encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused exclusions"\n'
        'mode = "nve"\n'
        "pbc = false\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 100.0\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'exclude_in_file = "exclude.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_exclusions_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "exclude_in_file"):
        raise AssertionError("focused exclusions legacy route is missing")
    if _has_key_line(bundled_mdin, "exclude_in_file"):
        raise AssertionError(
            "focused exclusions bundle retained exclude_in_file"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused exclusions bundle retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused exclusions topology retained sidecars")
    required = {
        "/topology/exclusions/offset",
        "/topology/exclusions/list",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused exclusions topology is missing datasets: {missing}"
        )
    with h5py.File(topology_path, "r") as topology:
        offsets = topology["/topology/exclusions/offset"][...].tolist()
        excluded = topology["/topology/exclusions/list"][...].tolist()
    if offsets != [0, 1, 1, 1] or excluded != [1]:
        raise AssertionError(
            "focused exclusions native payload changed: "
            f"offsets={offsets}, list={excluded}"
        )


def _prepare_focused_residue_sidecar_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_residue_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_residue_sidecar_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_residue_sidecar_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    _replace_typed_residue_with_focused_sidecar(legacy_dir, bundled_dir)

    sidecar_dir = bundled_dir / "legacy_sidecars"
    for child in sidecar_dir.iterdir():
        if child.name not in {"residue_in_file", "constrain_in_file"}:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    _validate_focused_residue_sidecar_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _replace_typed_residue_with_focused_sidecar(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    relative_sidecar = Path("legacy_sidecars/residue_in_file/residue.txt")
    sidecar_path = bundled_dir / relative_sidecar
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_dir / "residue.txt", sidecar_path)

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    typed_paths = ("/atoms/residue_index", "/residues/atom_offset")
    with h5py.File(topology_path, "r+") as topology:
        missing = [path for path in typed_paths if path not in topology]
        if missing:
            raise AssertionError(
                "focused residue conversion did not emit typed residue "
                f"datasets: {missing}"
            )
        for typed_path in typed_paths:
            del topology[typed_path]

        sidecars = topology.require_group(sidecar_root)
        for dataset_name in ("key", "path"):
            if dataset_name in sidecars:
                del sidecars[dataset_name]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset(
            "key", data=["residue_in_file"], dtype=string_dtype
        )
        sidecars.create_dataset(
            "path", data=[relative_sidecar.as_posix()], dtype=string_dtype
        )


def _write_focused_residue_sidecar_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "4\n1.0\n1.0\n1.0\n1.0\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "4 0.0\n"
        "19.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "5.0 0.0 0.0\n"
        "8.0 0.0 0.0\n"
        "20.0 20.0 20.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "4\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "residue.txt").write_text("4 2\n2\n2\n", encoding="utf-8")
    (case_dir / "bond.txt").write_text(
        "3\n0 1 2.0 1.0\n1 2 0.0 4.0\n2 3 0.0 3.0\n",
        encoding="utf-8",
    )
    (case_dir / "constrain.txt").write_text(
        "2\n0 1 2.0\n2 3 3.0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused residue sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'residue_in_file = "residue.txt"\n'
        'bond_in_file = "bond.txt"\n'
        'constrain_in_file = "constrain.txt"\n'
        'constrain_mode = "SHAKE"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_residue_sidecar_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "residue_in_file"):
        raise AssertionError("focused residue legacy route is missing")
    if _has_key_line(bundled_mdin, "residue_in_file"):
        raise AssertionError(
            "focused residue bundled mdin retained residue_in_file"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    for typed_path in ("/atoms/residue_index", "/residues/atom_offset"):
        if typed_path in topology_paths:
            raise AssertionError(
                f"focused residue sidecar route retained {typed_path}"
            )
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
    paths = _h5_string_values(topology_path, f"{sidecar_root}/path")
    if keys != ["residue_in_file"]:
        raise AssertionError(
            f"focused residue bundled sidecar keys changed: {keys}"
        )
    if len(paths) != 1 or not paths[0].endswith("/residue_in_file/residue.txt"):
        raise AssertionError(
            f"focused residue bundled sidecar path changed: {paths}"
        )
    bundled_payload = bundled_dir / paths[0]
    if (
        bundled_payload.read_bytes()
        != (legacy_dir / "residue.txt").read_bytes()
    ):
        raise AssertionError(
            "focused residue sidecar payload differs from legacy"
        )
    expected_sidecars = ["constrain_in_file", "residue_in_file"]
    if (
        sorted(
            path.name for path in (bundled_dir / "legacy_sidecars").iterdir()
        )
        != expected_sidecars
    ):
        raise AssertionError(
            "focused residue bundle retained unrelated sidecars"
        )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_root = "/parameters/sponge/files/legacy_sidecars"
    protocol_keys = _h5_string_values(protocol_path, f"{protocol_root}/key")
    if protocol_keys != ["constrain_in_file"]:
        raise AssertionError(
            f"focused residue support constraint route changed: {protocol_keys}"
        )


def _prepare_focused_residue_com_res_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_residue_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_residue_sidecar_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_residue_com_res_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    _replace_typed_residue_with_focused_sidecar(legacy_dir, bundled_dir)

    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    coordinate_sidecar = (
        bundled_dir
        / "legacy_sidecars"
        / "restrain_coordinate_in_file"
        / "restrain_coordinate.txt"
    )
    coordinate_sidecar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_dir / "restrain_coordinate.txt", coordinate_sidecar)
    protocol_path = bundled_dir / "protocol.spgp.h5"
    with h5py.File(protocol_path, "r+") as protocol:
        if sidecar_root not in protocol:
            raise AssertionError(
                "focused residue conversion did not emit restraint sidecars"
            )
        sidecars = protocol[sidecar_root]
        keys = sidecars["key"].asstr()[...].tolist()
        paths = sidecars["path"].asstr()[...].tolist()
        try:
            constraint_path = paths[keys.index("constrain_in_file")]
            atom_id_path = paths[keys.index("restrain_atom_id")]
        except ValueError as error:
            raise AssertionError(
                "focused residue conversion did not bind support sidecars"
            ) from error
        del sidecars["key"]
        del sidecars["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset(
            "key",
            data=[
                "constrain_in_file",
                "restrain_atom_id",
                "restrain_coordinate_in_file",
            ],
            dtype=string_dtype,
        )
        sidecars.create_dataset(
            "path",
            data=[
                constraint_path,
                atom_id_path,
                "legacy_sidecars/restrain_coordinate_in_file/"
                "restrain_coordinate.txt",
            ],
            dtype=string_dtype,
        )
        if "/restraint/default" not in protocol:
            raise AssertionError(
                "focused residue conversion lost typed restraint data"
            )
        del protocol["/restraint/default"]

    reference_path = (
        "/parameters/restart/references/restraint/default/coordinate"
    )
    with h5py.File(bundled_dir / "restart.spgr.h5", "r+") as restart:
        if reference_path not in restart:
            raise AssertionError(
                "focused residue conversion lost typed restraint reference"
            )
        del restart[reference_path]

    sidecar_dir = bundled_dir / "legacy_sidecars"
    for child in sidecar_dir.iterdir():
        if child.name not in {
            "residue_in_file",
            "constrain_in_file",
            "restrain_atom_id",
            "restrain_coordinate_in_file",
        }:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    _validate_focused_residue_com_res_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _promote_focused_residue_bundle_to_typed(bundled_dir: Path) -> None:
    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_root not in topology:
            raise AssertionError(
                "focused residue conversion did not emit topology sidecars"
            )
        del topology[sidecar_root]
        atoms = topology.require_group("/atoms")
        residues = topology.require_group("/residues")
        atoms.create_dataset("residue_index", data=[0, 0, 1, 1], dtype="i4")
        residues.create_dataset("atom_offset", data=[0, 2, 4], dtype="i8")

    residue_sidecar = bundled_dir / "legacy_sidecars" / "residue_in_file"
    if not residue_sidecar.exists():
        raise AssertionError(
            "focused residue bundle lost residue sidecar input"
        )
    shutil.rmtree(residue_sidecar)


def _prepare_focused_residue_typed_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_residue_sidecar_pair(case_root)
    _promote_focused_residue_bundle_to_typed(bundled_dir)
    _validate_focused_residue_typed_pbc_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_residue_typed_com_res_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_residue_com_res_pair(case_root)
    _promote_focused_residue_bundle_to_typed(bundled_dir)
    _validate_focused_residue_typed_com_res_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_residue_com_res_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "4\n1.0\n1.0\n1.0\n1.0\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "4 0.0\n"
        "19.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "5.0 0.0 0.0\n"
        "8.0 0.0 0.0\n"
        "20.0 20.0 20.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "4\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "residue.txt").write_text("4 2\n2\n2\n", encoding="utf-8")
    (case_dir / "bond.txt").write_text(
        "2\n0 1 2.0 1.0\n2 3 0.0 3.0\n",
        encoding="utf-8",
    )
    (case_dir / "constrain.txt").write_text(
        "2\n0 1 2.0\n2 3 3.0\n",
        encoding="utf-8",
    )
    (case_dir / "restrain_atom_id.txt").write_text("0\n", encoding="utf-8")
    (case_dir / "restrain_coordinate.txt").write_text(
        "4\n18.0 0.0 0.0\n1.0 0.0 0.0\n5.0 0.0 0.0\n8.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused residue sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.001\n"
        "cutoff = 8.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'residue_in_file = "residue.txt"\n'
        'bond_in_file = "bond.txt"\n'
        'constrain_in_file = "constrain.txt"\n'
        'constrain_mode = "SHAKE"\n'
        'restrain_atom_id = "restrain_atom_id.txt"\n'
        'restrain_coordinate_in_file = "restrain_coordinate.txt"\n'
        "restrain_single_weight = 4.0\n"
        'restrain_refcoord_scaling = "com_res"\n'
        "restrain_calc_virial = true\n"
        "force_whole_output = true\n"
        "print_pressure = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_residue_com_res_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "residue_in_file"):
        raise AssertionError("focused residue legacy route is missing")
    if _has_key_line(bundled_mdin, "residue_in_file"):
        raise AssertionError(
            "focused residue bundled mdin retained residue_in_file"
        )
    for key in (
        "constrain_in_file",
        "restrain_atom_id",
        "restrain_coordinate_in_file",
    ):
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(
                f"focused residue legacy route is missing {key}"
            )
        if _has_key_line(bundled_mdin, key):
            raise AssertionError(f"focused residue bundled mdin retained {key}")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    for typed_path in ("/atoms/residue_index", "/residues/atom_offset"):
        if typed_path in topology_paths:
            raise AssertionError(
                f"focused residue sidecar route retained {typed_path}"
            )
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
    paths = _h5_string_values(topology_path, f"{sidecar_root}/path")
    if keys != ["residue_in_file"]:
        raise AssertionError(
            f"focused residue bundled sidecar keys changed: {keys}"
        )
    if len(paths) != 1 or not paths[0].endswith("/residue_in_file/residue.txt"):
        raise AssertionError(
            f"focused residue bundled sidecar path changed: {paths}"
        )
    bundled_payload = bundled_dir / paths[0]
    if (
        bundled_payload.read_bytes()
        != (legacy_dir / "residue.txt").read_bytes()
    ):
        raise AssertionError(
            "focused residue sidecar payload differs from legacy"
        )
    expected_sidecars = [
        "constrain_in_file",
        "residue_in_file",
        "restrain_atom_id",
        "restrain_coordinate_in_file",
    ]
    if (
        sorted(
            path.name for path in (bundled_dir / "legacy_sidecars").iterdir()
        )
        != expected_sidecars
    ):
        raise AssertionError(
            "focused residue bundle retained unrelated sidecars"
        )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_root = "/parameters/sponge/files/legacy_sidecars"
    protocol_keys = _h5_string_values(protocol_path, f"{protocol_root}/key")
    protocol_paths = _h5_string_values(protocol_path, f"{protocol_root}/path")
    expected_protocol_keys = [
        "constrain_in_file",
        "restrain_atom_id",
        "restrain_coordinate_in_file",
    ]
    if protocol_keys != expected_protocol_keys:
        raise AssertionError(
            f"focused residue support routes changed: {protocol_keys}"
        )
    expected_files = {
        "constrain_in_file": "constrain.txt",
        "restrain_atom_id": "restrain_atom_id.txt",
        "restrain_coordinate_in_file": "restrain_coordinate.txt",
    }
    for key, path in zip(protocol_keys, protocol_paths, strict=True):
        expected_name = expected_files[key]
        if not path.endswith(f"/{key}/{expected_name}"):
            raise AssertionError(
                f"focused residue {key} sidecar path changed: {path}"
            )
        if (bundled_dir / path).read_bytes() != (
            legacy_dir / expected_name
        ).read_bytes():
            raise AssertionError(
                f"focused residue {key} payload differs from legacy"
            )


def _validate_focused_residue_typed_payload(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "residue_in_file"):
        raise AssertionError("focused typed residue legacy route is missing")
    if _has_key_line(bundled_mdin, "residue_in_file"):
        raise AssertionError(
            "focused typed residue bundled mdin retained residue_in_file"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r") as topology:
        if sidecar_root in topology:
            raise AssertionError(
                "focused typed residue retained topology sidecar table"
            )
        residue_index = topology["/atoms/residue_index"][...].tolist()
        atom_offset = topology["/residues/atom_offset"][...].tolist()
    if residue_index != [0, 0, 1, 1]:
        raise AssertionError(
            f"focused typed residue membership changed: {residue_index}"
        )
    if atom_offset != [0, 2, 4]:
        raise AssertionError(
            f"focused typed residue offsets changed: {atom_offset}"
        )


def _validate_focused_residue_typed_pbc_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    _validate_focused_residue_typed_payload(legacy_dir, bundled_dir)
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    expected_sidecars = ["constrain_in_file"]
    sidecar_dir = bundled_dir / "legacy_sidecars"
    if sorted(path.name for path in sidecar_dir.iterdir()) != expected_sidecars:
        raise AssertionError(
            "focused typed residue PBC bundle retained unrelated sidecars"
        )
    protocol_keys = _h5_string_values(
        bundled_dir / "protocol.spgp.h5", f"{sidecar_root}/key"
    )
    if protocol_keys != expected_sidecars:
        raise AssertionError(
            f"focused typed residue PBC support routes changed: {protocol_keys}"
        )


def _validate_focused_residue_typed_com_res_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    _validate_focused_residue_typed_payload(legacy_dir, bundled_dir)
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"

    expected_sidecars = [
        "constrain_in_file",
        "restrain_atom_id",
        "restrain_coordinate_in_file",
    ]
    sidecar_dir = bundled_dir / "legacy_sidecars"
    if sorted(path.name for path in sidecar_dir.iterdir()) != expected_sidecars:
        raise AssertionError(
            "focused typed residue bundle retained unrelated sidecars"
        )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_keys = _h5_string_values(protocol_path, f"{sidecar_root}/key")
    if protocol_keys != expected_sidecars:
        raise AssertionError(
            f"focused typed residue support routes changed: {protocol_keys}"
        )


def _prepare_focused_gb_hybrid_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_gb_hybrid_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_gb_hybrid_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_gb_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecars = topology["/parameters/sponge/files/legacy_sidecars"]
        del sidecars["key"]
        del sidecars["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset("key", data=["gb_in_file"], dtype=string_dtype)
        sidecars.create_dataset(
            "path",
            data=["legacy_sidecars/gb_in_file/gb.txt"],
            dtype=string_dtype,
        )
    for key in ("mass_in_file", "charge_in_file"):
        sidecar_dir = bundled_dir / "legacy_sidecars" / key
        if sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
    _validate_focused_gb_hybrid_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_gb_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n1.0\n-1.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n"
        "0.0 0.0 0.0\n"
        "2.0 0.0 0.0\n"
        "1000.0 1000.0 1000.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "gb.txt").write_text("2\n1.5 0.8\n1.5 0.8\n", encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused gb hybrid"\n'
        'mode = "nve"\n'
        "pbc = false\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 10.0\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'gb_in_file = "gb.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_gb_hybrid_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "gb_in_file"):
        raise AssertionError("focused GB legacy route lost gb_in_file")
    if _has_key_line(bundled_mdin, "gb_in_file"):
        raise AssertionError("focused GB bundled mdin retained gb_in_file")

    topology_path = bundled_dir / "topology.spgt.h5"
    required = {
        "/forcefield/gb/params",
        "/parameters/sponge/files/legacy_sidecars/key",
        "/parameters/sponge/files/legacy_sidecars/path",
    }
    missing = sorted(required - _h5_paths(topology_path))
    if missing:
        raise AssertionError(
            f"focused GB topology is missing datasets: {missing}"
        )
    with h5py.File(topology_path, "r") as topology:
        params = topology["/forcefield/gb/params"][...].tolist()
        keys = (
            topology["/parameters/sponge/files/legacy_sidecars/key"]
            .asstr()[...]
            .tolist()
        )
        paths = (
            topology["/parameters/sponge/files/legacy_sidecars/path"]
            .asstr()[...]
            .tolist()
        )
    if len(params) != 2 or any(
        len(row) != 2
        or not math.isclose(row[0], 1.5, rel_tol=0.0, abs_tol=1.0e-7)
        or not math.isclose(row[1], 0.8, rel_tol=0.0, abs_tol=1.0e-7)
        for row in params
    ):
        raise AssertionError(f"focused GB native parameters changed: {params}")
    expected_path = "legacy_sidecars/gb_in_file/gb.txt"
    if dict(zip(keys, paths, strict=True)) != {"gb_in_file": expected_path}:
        raise AssertionError(
            f"focused GB sidecar activation binding changed: {keys}, {paths}"
        )
    legacy_payload = (legacy_dir / "gb.txt").read_bytes()
    bundled_payload = (bundled_dir / expected_path).read_bytes()
    if legacy_payload != bundled_payload:
        raise AssertionError("focused GB sidecar payload differs from legacy")


def _prepare_focused_gb_native_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_gb_native_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_gb_native_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_gb_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as topology:
        if sidecar_root in topology:
            del topology[sidecar_root]
    sidecar_dir = bundled_dir / "legacy_sidecars"
    if sidecar_dir.exists():
        shutil.rmtree(sidecar_dir)

    _validate_focused_gb_native_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _validate_focused_gb_native_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "gb_in_file"):
        raise AssertionError("focused native GB legacy route lost gb_in_file")
    if _has_key_line(bundled_mdin, "gb_in_file"):
        raise AssertionError(
            "focused native GB bundled mdin retained gb_in_file"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if "/forcefield/gb/params" not in topology_paths:
        raise AssertionError("focused native GB topology lost typed parameters")
    if any(
        path.startswith("/parameters/sponge/files/legacy_sidecars")
        for path in topology_paths
    ):
        raise AssertionError(
            "focused native GB topology retained sidecar bindings"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused native GB bundle retained sidecar files")

    with h5py.File(topology_path, "r") as topology:
        params = topology["/forcefield/gb/params"][...].tolist()
    if len(params) != 2 or any(
        len(row) != 2
        or not math.isclose(row[0], 1.5, rel_tol=0.0, abs_tol=1.0e-7)
        or not math.isclose(row[1], 0.8, rel_tol=0.0, abs_tol=1.0e-7)
        for row in params
    ):
        raise AssertionError(f"focused native GB parameters changed: {params}")


def _prepare_focused_improper_converted_pair(
    case_root: Path, legacy_key: str
) -> tuple[Path, Path]:
    if legacy_key not in {
        "improper_dihedral_in_file",
        "improper_in_file",
    }:
        raise AssertionError(f"unsupported improper legacy key: {legacy_key}")
    legacy_source = case_root / "focused_improper_converted_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_improper_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_improper_native_input(legacy_source)
    if legacy_key == "improper_in_file":
        mdin_path = legacy_source / "mdin.spg.toml"
        mdin_path.write_text(
            mdin_path.read_text(encoding="utf-8").replace(
                "improper_dihedral_in_file", "improper_in_file"
            ),
            encoding="utf-8",
        )
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    _validate_focused_improper_converted_routes(
        legacy_dir,
        bundled_dir,
        converted_dir / "manifest.json",
        legacy_key,
    )
    return legacy_dir, bundled_dir


def _validate_focused_improper_converted_routes(
    legacy_dir: Path,
    bundled_dir: Path,
    manifest_path: Path,
    legacy_key: str,
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, legacy_key):
        raise AssertionError(
            f"focused converted improper legacy route lost {legacy_key}"
        )
    other_key = (
        "improper_in_file"
        if legacy_key == "improper_dihedral_in_file"
        else "improper_dihedral_in_file"
    )
    if _has_key_line(legacy_mdin, other_key):
        raise AssertionError(
            f"focused converted improper legacy route retained {other_key}"
        )
    retained = sorted(
        key
        for key in ("improper_dihedral_in_file", "improper_in_file")
        if _has_key_line(bundled_mdin, key)
    )
    if retained:
        raise AssertionError(
            f"focused converted improper bundle retained legacy keys: {retained}"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    required = {
        "/forcefield/improper/atoms",
        "/forcefield/improper/count",
        "/forcefield/improper/pk",
        "/forcefield/improper/phi0",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused converted improper topology is missing datasets: {missing}"
        )
    if "/forcefield/improper/k" in topology_paths:
        raise AssertionError(
            "focused converted improper topology emitted legacy k"
        )
    sidecar_keys = _h5_string_values(
        topology_path, "/parameters/sponge/files/legacy_sidecars/key"
    )
    sidecar_paths = _h5_string_values(
        topology_path, "/parameters/sponge/files/legacy_sidecars/path"
    )
    if sidecar_keys != ["mass_in_file"] or sidecar_paths != [
        "legacy_sidecars/mass_in_file/mass.txt"
    ]:
        raise AssertionError(
            "focused converted improper emitted unexpected compatibility "
            f"sidecars: keys={sidecar_keys}, paths={sidecar_paths}"
        )
    if not (bundled_dir / sidecar_paths[0]).is_file():
        raise AssertionError(
            "focused converted improper mass sidecar is absent"
        )

    with h5py.File(topology_path, "r") as topology:
        count = int(topology["/forcefield/improper/count"][()])
        atoms = topology["/forcefield/improper/atoms"][...].tolist()
        pk = topology["/forcefield/improper/pk"][...].tolist()
        phi0 = topology["/forcefield/improper/phi0"][...].tolist()
    if count != 1 or atoms != [[0, 1, 2, 3]]:
        raise AssertionError(
            "focused converted improper atom payload changed: "
            f"count={count}, atoms={atoms}"
        )
    if len(pk) != 1 or len(phi0) != 1:
        raise AssertionError(
            "focused converted improper parameters changed shape"
        )
    if not math.isclose(
        pk[0], 10.0, rel_tol=0.0, abs_tol=1.0e-7
    ) or not math.isclose(phi0[0], 0.2, rel_tol=0.0, abs_tol=1.0e-7):
        raise AssertionError(
            "focused converted improper parameters changed: "
            f"pk={pk}, phi0={phi0}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest["entries"]
        if entry.get("source_key") == legacy_key
    ]
    expected_contract = (
        "topology.improper_dihedral"
        if legacy_key == "improper_dihedral_in_file"
        else "topology.improper"
    )
    if len(entries) != 1 or entries[0].get("status") != "typed_converted":
        raise AssertionError(
            f"focused converted improper manifest entry changed: {entries}"
        )
    if (
        entries[0].get("contract_id") != expected_contract
        or entries[0].get("bundle_path") != "/forcefield/improper"
    ):
        raise AssertionError(
            f"focused converted improper manifest route changed: {entries[0]}"
        )


def _prepare_focused_improper_native_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_improper_native_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_improper_native_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_improper_native_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        improper = topology["/forcefield/improper"]
        if "pk" not in improper:
            if "k" not in improper:
                raise AssertionError(
                    "focused improper conversion emitted neither k nor pk"
                )
            improper.create_dataset("pk", data=improper["k"][...])
        if "k" in improper:
            del improper["k"]
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_improper_native_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_improper_native_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text(
        "4\n12.0\n12.0\n12.0\n12.0\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "4 0.0\n"
        "1.0 0.0 0.0\n"
        "0.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "10.0 10.0 10.0\n"
        "90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "4\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (case_dir / "improper.txt").write_text(
        "1\n0 1 2 3 10.0 0.2\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused improper native"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'improper_dihedral_in_file = "improper.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_improper_native_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "improper_dihedral_in_file"):
        raise AssertionError(
            "focused improper legacy route lost improper_dihedral_in_file"
        )
    retained = sorted(
        key
        for key in ("improper_dihedral_in_file", "improper_in_file")
        if _has_key_line(bundled_mdin, key)
    )
    if retained:
        raise AssertionError(
            f"focused improper bundle retained legacy keys: {retained}"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused improper bundle retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused improper topology retained sidecars")
    required = {
        "/forcefield/improper/atoms",
        "/forcefield/improper/count",
        "/forcefield/improper/pk",
        "/forcefield/improper/phi0",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused improper topology is missing datasets: {missing}"
        )
    if "/forcefield/improper/k" in topology_paths:
        raise AssertionError("focused improper topology retained legacy k")
    with h5py.File(topology_path, "r") as topology:
        count = int(topology["/forcefield/improper/count"][()])
        atoms = topology["/forcefield/improper/atoms"][...].tolist()
        pk = topology["/forcefield/improper/pk"][...].tolist()
        phi0 = topology["/forcefield/improper/phi0"][...].tolist()
    if count != 1 or atoms != [[0, 1, 2, 3]]:
        raise AssertionError(
            "focused improper native atom payload changed: "
            f"count={count}, atoms={atoms}"
        )
    if len(pk) != 1 or len(phi0) != 1:
        raise AssertionError("focused improper native parameters changed shape")
    if not math.isclose(
        pk[0], 10.0, rel_tol=0.0, abs_tol=1.0e-7
    ) or not math.isclose(phi0[0], 0.2, rel_tol=0.0, abs_tol=1.0e-7):
        raise AssertionError(
            f"focused improper native parameters changed: pk={pk}, phi0={phi0}"
        )


def _prepare_focused_lj_soft_core_pair(case_root: Path) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_lj_soft_core_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_lj_soft_core_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_lj_soft_core_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
    bundled_mdin.write_text(
        _remove_key_lines(
            bundled_mdin.read_text(encoding="utf-8"),
            {"LJ_soft_core_in_file"},
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_lj_soft_core_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_subsystem_division_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_subsystem_division_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_subsystem_division_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_subsystem_division_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
    bundled_mdin.write_text(
        _remove_key_lines(
            bundled_mdin.read_text(encoding="utf-8"),
            {"LJ_soft_core_in_file", "subsys_division_in_file"},
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_lj_soft_core_routes(
        legacy_dir, bundled_dir, expect_subsystem=True
    )
    return legacy_dir, bundled_dir


def _write_focused_lj_soft_core_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n0.0\n0.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n20.0 20.0 20.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "lj_soft_core.txt").write_text(
        "2 1 1\n0.0\n0.0\n2.0\n1.5\n0 0\n0 0\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused lj soft core"\n'
        'mode = "nve"\n'
        "pbc = true\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        "lambda_lj = 0.5\n"
        "soft_core_alpha = 0.5\n"
        "soft_core_powfer = 1.0\n"
        "soft_core_sigma = 3.0\n"
        "soft_core_sigma_min = 0.0\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'LJ_soft_core_in_file = "lj_soft_core.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _write_focused_subsystem_division_input(case_dir: Path) -> None:
    _write_focused_lj_soft_core_input(case_dir)
    (case_dir / "subsys_division.txt").write_text("2\n0\n1\n", encoding="utf-8")
    mdin = case_dir / "mdin.spg.toml"
    mdin.write_text(
        mdin.read_text(encoding="utf-8")
        + 'subsys_division_in_file = "subsys_division.txt"\n',
        encoding="utf-8",
    )


def _validate_focused_lj_soft_core_routes(
    legacy_dir: Path, bundled_dir: Path, *, expect_subsystem: bool = False
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "LJ_soft_core_in_file"):
        raise AssertionError("focused LJ soft-core legacy route is missing")
    if _has_key_line(bundled_mdin, "LJ_soft_core_in_file"):
        raise AssertionError(
            "focused LJ soft-core bundle retained LJ_soft_core_in_file"
        )
    if _has_key_line(bundled_mdin, "subsys_division_in_file"):
        raise AssertionError(
            "focused LJ soft-core bundle unexpectedly retained subsystem input"
        )
    if expect_subsystem != _has_key_line(
        legacy_mdin, "subsys_division_in_file"
    ):
        raise AssertionError(
            "focused LJ soft-core legacy subsystem route does not match fixture"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused LJ soft-core bundle retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused LJ soft-core topology retained sidecars")
    required = {
        "/forcefield/lj_soft_core/atom_type_A",
        "/forcefield/lj_soft_core/atom_type_B",
        "/forcefield/lj_soft_core/atom_type_count_A",
        "/forcefield/lj_soft_core/atom_type_count_B",
        "/forcefield/lj_soft_core/pair_AA",
        "/forcefield/lj_soft_core/pair_AB",
        "/forcefield/lj_soft_core/pair_BA",
        "/forcefield/lj_soft_core/pair_BB",
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused LJ soft-core topology is missing datasets: {missing}"
        )
    subsystem_path = "/forcefield/subsys_division"
    if expect_subsystem != (subsystem_path in topology_paths):
        raise AssertionError(
            "focused LJ soft-core native subsystem route does not match fixture"
        )
    with h5py.File(topology_path, "r") as topology:
        payload = {
            "atom_type_A": topology["/forcefield/lj_soft_core/atom_type_A"][
                ...
            ].tolist(),
            "atom_type_B": topology["/forcefield/lj_soft_core/atom_type_B"][
                ...
            ].tolist(),
            "pair_AA": topology["/forcefield/lj_soft_core/pair_AA"][
                ...
            ].tolist(),
            "pair_AB": topology["/forcefield/lj_soft_core/pair_AB"][
                ...
            ].tolist(),
            "pair_BA": topology["/forcefield/lj_soft_core/pair_BA"][
                ...
            ].tolist(),
            "pair_BB": topology["/forcefield/lj_soft_core/pair_BB"][
                ...
            ].tolist(),
        }
        type_counts = (
            int(topology["/forcefield/lj_soft_core/atom_type_count_A"][()]),
            int(topology["/forcefield/lj_soft_core/atom_type_count_B"][()]),
        )
        subsystem = (
            topology[subsystem_path][...].tolist() if expect_subsystem else None
        )
    expected_payload = {
        "atom_type_A": [0, 0],
        "atom_type_B": [0, 0],
        "pair_AA": [0.0],
        "pair_AB": [0.0],
        "pair_BA": [2.0],
        "pair_BB": [1.5],
    }
    if payload != expected_payload or type_counts != (1, 1):
        raise AssertionError(
            "focused LJ soft-core native payload changed: "
            f"payload={payload}, type_counts={type_counts}"
        )
    if expect_subsystem and subsystem != [0, 1]:
        raise AssertionError(
            f"focused subsystem native payload changed: {subsystem}"
        )


def _prepare_focused_virtual_atoms_pair(
    fixture_case: str, case_root: Path
) -> tuple[Path, Path]:
    if fixture_case not in {
        FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
        FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
        FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    }:
        raise AssertionError(
            f"unknown focused virtual-atom fixture: {fixture_case}"
        )
    legacy_source = case_root / "focused_virtual_atoms_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_virtual_atoms_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_virtual_atoms_input(legacy_source, fixture_case)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    bundled_mdin = bundled_dir / "mdin.bundled.spg.toml"
    bundled_mdin.write_text(
        _remove_key_lines(
            bundled_mdin.read_text(encoding="utf-8"),
            {"virtual_atom_in_file", "virtual_atoms_in_file"},
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    legacy_sidecars = bundled_dir / "legacy_sidecars"
    if legacy_sidecars.exists():
        shutil.rmtree(legacy_sidecars)
    _validate_focused_virtual_atoms_routes(
        legacy_dir, bundled_dir, fixture_case
    )
    return legacy_dir, bundled_dir


def _write_focused_virtual_atoms_input(
    case_dir: Path, fixture_case: str
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    if fixture_case == FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE:
        atom_count = 8
        masses = [1.0] * atom_count
        charges = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0, -1.0]
        coordinates = [
            (0.0, 0.0, 1.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (2.0, 0.0, 1.0),
            (0.0, 2.0, 1.0),
            (4.0, 4.0, 1.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
        ]
        box = (20.0, 20.0, 20.0)
        virtual_atoms = (
            "2 1 0 3 4 0.25 0.5\n3 2 1 3 4 1.0 0.5\n0 6 5 2.0\n1 7 3 4 0.25\n"
        )
        pbc = True
        cutoff = 8.0
    elif fixture_case in {
        FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
        FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    }:
        atom_count = 4
        masses = [1.0] * atom_count
        charges = [0.0, 0.0, 1.0, -1.0]
        coordinates = [
            (9.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ]
        box = (10.0, 10.0, 10.0)
        virtual_atoms = "1 2 0 1 0.25\n"
        pbc = True
        cutoff = 4.0
    else:
        raise AssertionError(
            f"unknown focused virtual-atom fixture: {fixture_case}"
        )
    virtual_atom_key = (
        "virtual_atoms_in_file"
        if fixture_case == FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE
        else "virtual_atom_in_file"
    )

    (case_dir / "mass.txt").write_text(
        f"{atom_count}\n" + "".join(f"{value}\n" for value in masses),
        encoding="utf-8",
    )
    (case_dir / "charge.txt").write_text(
        f"{atom_count}\n" + "".join(f"{value}\n" for value in charges),
        encoding="utf-8",
    )
    coordinate_text = [f"{atom_count} 0.0"]
    coordinate_text.extend(
        " ".join(str(value) for value in xyz) for xyz in coordinates
    )
    coordinate_text.extend(
        [" ".join(str(value) for value in box), "90.0 90.0 90.0"]
    )
    (case_dir / "coordinate.txt").write_text(
        "\n".join(coordinate_text) + "\n", encoding="utf-8"
    )
    (case_dir / "velocity.txt").write_text(
        f"{atom_count}\n" + "0.0 0.0 0.0\n" * atom_count,
        encoding="utf-8",
    )
    (case_dir / "virtual_atom.txt").write_text(virtual_atoms, encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused virtual atoms"\n'
        'mode = "nve"\n'
        f"pbc = {'true' if pbc else 'false'}\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        f"cutoff = {cutoff}\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        f'{virtual_atom_key} = "virtual_atom.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _focused_virtual_atom_payload(fixture_case: str) -> dict[str, list[float]]:
    if fixture_case == FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE:
        return {
            "type": [2, 3, 0, 1],
            "atom": [1, 2, 6, 7],
            "from_offset": [0, 3, 6, 7, 9],
            "from": [0, 3, 4, 1, 3, 4, 5, 3, 4],
            "parameter_offset": [0, 2, 4, 5, 6],
            "parameter": [0.25, 0.5, 1.0, 0.5, 2.0, 0.25],
        }
    if fixture_case in {
        FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
        FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    }:
        return {
            "type": [1],
            "atom": [2],
            "from_offset": [0, 2],
            "from": [0, 1],
            "parameter_offset": [0, 1],
            "parameter": [0.25],
        }
    raise AssertionError(
        f"unknown focused virtual-atom fixture: {fixture_case}"
    )


def _validate_focused_virtual_atoms_routes(
    legacy_dir: Path, bundled_dir: Path, fixture_case: str
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    legacy_key = (
        "virtual_atoms_in_file"
        if fixture_case == FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE
        else "virtual_atom_in_file"
    )
    if not _has_key_line(legacy_mdin, legacy_key):
        raise AssertionError(
            f"focused virtual-atom legacy route is missing: {legacy_key}"
        )
    alternate_legacy_key = (
        "virtual_atom_in_file"
        if legacy_key == "virtual_atoms_in_file"
        else "virtual_atoms_in_file"
    )
    if _has_key_line(legacy_mdin, alternate_legacy_key):
        raise AssertionError(
            "focused virtual-atom legacy route retained alternate key: "
            f"{alternate_legacy_key}"
        )
    retained = sorted(
        key
        for key in ("virtual_atom_in_file", "virtual_atoms_in_file")
        if _has_key_line(bundled_mdin, key)
    )
    if retained:
        raise AssertionError(
            f"focused virtual-atom bundle retained legacy keys: {retained}"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused virtual-atom bundle retained sidecars")

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_paths = _h5_paths(topology_path)
    if "/parameters/sponge/files/legacy_sidecars" in topology_paths:
        raise AssertionError("focused virtual-atom topology retained sidecars")
    expected = _focused_virtual_atom_payload(fixture_case)
    required = {
        f"/forcefield/virtual_atom/{name}" for name in (*expected, "count")
    }
    missing = sorted(required - topology_paths)
    if missing:
        raise AssertionError(
            f"focused virtual-atom topology is missing datasets: {missing}"
        )
    with h5py.File(topology_path, "r") as topology:
        actual = {
            name: topology[f"/forcefield/virtual_atom/{name}"][...].tolist()
            for name in expected
        }
        count = int(topology["/forcefield/virtual_atom/count"][()])
    if actual != expected or count != len(expected["type"]):
        raise AssertionError(
            "focused virtual-atom native payload changed: "
            f"payload={actual}, count={count}"
        )


def _prepare_focused_constraint_sidecar_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_constraint_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_constraint_sidecar_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_constraint_sidecar_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    _validate_focused_constraint_sidecar_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_constraint_sidecar_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n-1.0 0.0 0.0\n1.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "constrain.txt").write_text("1\n0 1 1.5\n", encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused constraint sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 4\n"
        "dt = 0.001\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'constrain_in_file = "constrain.txt"\n'
        'constrain_mode = "SHAKE"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_trajectory_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_constraint_sidecar_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "constrain_in_file"):
        raise AssertionError(
            "focused constraint legacy route lost constrain_in_file"
        )
    if _has_key_line(bundled_mdin, "constrain_in_file"):
        raise AssertionError(
            "focused constraint bundled mdin retained constrain_in_file"
        )
    for branch, mdin in (("legacy", legacy_mdin), ("bundled", bundled_mdin)):
        if not _has_key_line(mdin, "constrain_mode"):
            raise AssertionError(
                f"focused constraint {branch} route lost constrain_mode"
            )

    protocol_path = bundled_dir / "protocol.spgp.h5"
    required = {
        "/constraint/default/pairs/atoms",
        "/constraint/default/pairs/r0",
        "/parameters/sponge/files/legacy_sidecars/key",
        "/parameters/sponge/files/legacy_sidecars/path",
    }
    missing = sorted(required - _h5_paths(protocol_path))
    if missing:
        raise AssertionError(
            f"focused constraint protocol is missing datasets: {missing}"
        )
    with h5py.File(protocol_path, "r") as protocol:
        atoms = protocol["/constraint/default/pairs/atoms"][...].tolist()
        distances = protocol["/constraint/default/pairs/r0"][...].tolist()
        keys = (
            protocol["/parameters/sponge/files/legacy_sidecars/key"]
            .asstr()[...]
            .tolist()
        )
        paths = (
            protocol["/parameters/sponge/files/legacy_sidecars/path"]
            .asstr()[...]
            .tolist()
        )
    if atoms != [[0, 1]] or distances != [1.5]:
        raise AssertionError(
            "focused constraint typed payload changed: "
            f"atoms={atoms}, r0={distances}"
        )
    bindings = dict(zip(keys, paths, strict=True))
    expected_path = "legacy_sidecars/constrain_in_file/constrain.txt"
    if bindings != {"constrain_in_file": expected_path}:
        raise AssertionError(
            f"focused constraint sidecar binding changed: {bindings}"
        )
    legacy_payload = (legacy_dir / "constrain.txt").read_bytes()
    bundled_payload = (bundled_dir / expected_path).read_bytes()
    if legacy_payload != bundled_payload:
        raise AssertionError(
            "focused constraint sidecar payload differs from legacy input"
        )


def _prepare_focused_constraint_typed_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_constraint_sidecar_pair(
        case_root
    )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(protocol_path, "r+") as protocol:
        if sidecar_root not in protocol:
            raise AssertionError(
                "focused constraint conversion lost protocol sidecars"
            )
        del protocol[sidecar_root]
    sidecar_dir = bundled_dir / "legacy_sidecars"
    if not sidecar_dir.exists():
        raise AssertionError("focused constraint bundle lost sidecar payload")
    shutil.rmtree(sidecar_dir)
    _validate_focused_constraint_typed_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _validate_focused_constraint_typed_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "constrain_in_file"):
        raise AssertionError("focused typed constraint legacy route is missing")
    if _has_key_line(bundled_mdin, "constrain_in_file"):
        raise AssertionError(
            "focused typed constraint retained constrain_in_file"
        )
    if not _has_key_line(bundled_mdin, "constrain_mode"):
        raise AssertionError("focused typed constraint lost constrain_mode")
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused typed constraint retained sidecar files")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_paths = _h5_paths(protocol_path)
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    if sidecar_root in protocol_paths:
        raise AssertionError("focused typed constraint retained sidecar table")
    required = {
        "/constraint/default/pairs/atoms",
        "/constraint/default/pairs/r0",
    }
    missing = sorted(required - protocol_paths)
    if missing:
        raise AssertionError(
            f"focused typed constraint is missing datasets: {missing}"
        )
    with h5py.File(protocol_path, "r") as protocol:
        atoms = protocol["/constraint/default/pairs/atoms"][...].tolist()
        distances = protocol["/constraint/default/pairs/r0"][...].tolist()
    if atoms != [[0, 1]] or distances != [1.5]:
        raise AssertionError(
            "focused typed constraint payload changed: "
            f"atoms={atoms}, r0={distances}"
        )


def _prepare_focused_steering_cv_sidecar_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_steering_cv_sidecar_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_steering_cv_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_steering_cv_sidecar_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    topology_path = bundled_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in topology:
            del topology[sidecar_table]
    protocol_path = bundled_dir / "protocol.spgp.h5"
    with h5py.File(protocol_path, "r+") as protocol:
        if "/cv" in protocol:
            del protocol["/cv"]
    mass_sidecar = bundled_dir / "legacy_sidecars" / "mass_in_file"
    if mass_sidecar.exists():
        shutil.rmtree(mass_sidecar)
    _validate_focused_steering_cv_sidecar_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _write_focused_steering_cv_sidecar_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n1.5 0.0 0.0\n10.0 10.0 10.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "cv.txt").write_text(
        "steer\n"
        "{\n"
        "    CV = distance\n"
        "    weight = 2.0\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 0 1\n"
        "}\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused steering CV sidecar"\n'
        'mode = "nve"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 4.0\n"
        "skin = 0.4\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'cv_in_file = "cv.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _prepare_focused_steering_cv_typed_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_steering_cv_sidecar_pair(
        case_root
    )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(protocol_path, "r+") as protocol:
        if sidecar_root not in protocol:
            raise AssertionError(
                "focused steering conversion lost protocol sidecars"
            )
        del protocol[sidecar_root]
        config = protocol.require_group("/cv/config")
        section = config.require_group("section")
        section.create_dataset("count", data=2, dtype="i8")
        section.create_dataset(
            "name", data=["steer", "distance"], dtype=string_dtype
        )
        section.create_dataset("key_offset", data=[0, 2, 4], dtype="i8")
        config.create_dataset(
            "key",
            data=["CV", "weight", "CV_type", "atom"],
            dtype=string_dtype,
        )
        config.create_dataset(
            "value",
            data=["distance", "2.0", "distance", "0 1"],
            dtype=string_dtype,
        )
    sidecar_dir = bundled_dir / "legacy_sidecars"
    if not sidecar_dir.exists():
        raise AssertionError("focused steering bundle lost sidecar payload")
    shutil.rmtree(sidecar_dir)
    _validate_focused_steering_cv_typed_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _validate_focused_steering_cv_typed_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "cv_in_file"):
        raise AssertionError("focused typed steering legacy route is missing")
    if _has_key_line(bundled_mdin, "cv_in_file"):
        raise AssertionError("focused typed steering retained cv_in_file")
    if _has_key_line(bundled_mdin, "steer_cv_in_file"):
        raise AssertionError(
            "focused typed steering used unconsumed steer_cv_in_file"
        )
    if (bundled_dir / "legacy_sidecars").exists():
        raise AssertionError("focused typed steering retained sidecar files")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_paths = _h5_paths(protocol_path)
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    if sidecar_root in protocol_paths:
        raise AssertionError("focused typed steering retained sidecar table")
    required = {
        "/cv/config/section/count",
        "/cv/config/section/name",
        "/cv/config/section/key_offset",
        "/cv/config/key",
        "/cv/config/value",
    }
    missing = sorted(required - protocol_paths)
    if missing:
        raise AssertionError(
            f"focused typed steering is missing datasets: {missing}"
        )
    with h5py.File(protocol_path, "r") as protocol:
        count = int(protocol["/cv/config/section/count"][()])
        names = protocol["/cv/config/section/name"].asstr()[...].tolist()
        offsets = protocol["/cv/config/section/key_offset"][...].tolist()
        keys = protocol["/cv/config/key"].asstr()[...].tolist()
        values = protocol["/cv/config/value"].asstr()[...].tolist()
    expected = (
        2,
        ["steer", "distance"],
        [0, 2, 4],
        ["CV", "weight", "CV_type", "atom"],
        ["distance", "2.0", "distance", "0 1"],
    )
    actual = (count, names, offsets, keys, values)
    if actual != expected:
        raise AssertionError(
            f"focused typed steering payload changed: {actual}"
        )


def _validate_focused_steering_cv_sidecar_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "cv_in_file"):
        raise AssertionError("focused steering legacy route lost cv_in_file")
    if _has_key_line(legacy_mdin, "steer_cv_in_file"):
        raise AssertionError(
            "focused steering legacy route used unconsumed steer_cv_in_file"
        )
    if _has_key_line(bundled_mdin, "cv_in_file"):
        raise AssertionError(
            "focused steering bundled mdin retained cv_in_file"
        )

    topology_path = bundled_dir / "topology.spgt.h5"
    topology_sidecars = "/parameters/sponge/files/legacy_sidecars"
    if topology_sidecars in _h5_paths(topology_path):
        raise AssertionError(
            "focused steering bundled topology retained sidecars"
        )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_paths = _h5_paths(protocol_path)
    if any(path.startswith("/cv") for path in protocol_paths):
        raise AssertionError(
            "focused steering bundled protocol retained typed CV data"
        )
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    keys = _h5_string_values(protocol_path, f"{sidecar_root}/key")
    paths = _h5_string_values(protocol_path, f"{sidecar_root}/path")
    expected_path = "legacy_sidecars/cv_in_file/cv.txt"
    if keys != ["cv_in_file"] or paths != [expected_path]:
        raise AssertionError(
            "focused steering protocol sidecar binding changed: "
            f"keys={keys}, paths={paths}"
        )
    legacy_payload = (legacy_dir / "cv.txt").read_bytes()
    bundled_payload = (bundled_dir / expected_path).read_bytes()
    if legacy_payload != bundled_payload:
        raise AssertionError(
            "focused steering CV sidecar payload differs from legacy input"
        )
    if (bundled_dir / "legacy_sidecars" / "mass_in_file").exists():
        raise AssertionError(
            "focused steering bundled branch retained mass sidecar"
        )


def _prepare_focused_sits_nk_typed_restart_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_sits_nk_typed_restart_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_sits_nk_restart_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_sits_nk_typed_restart_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    bundled_mdin_path = bundled_dir / "mdin.bundled.spg.toml"
    bundled_mdin = _remove_key_lines(
        bundled_mdin_path.read_text(encoding="utf-8"), {"SITS_nk_rest"}
    )
    bundled_mdin = _insert_root_toml_keys(
        bundled_mdin, ["SITS_nk_rest = false"]
    )
    bundled_mdin_path.write_text(bundled_mdin, encoding="utf-8")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    with h5py.File(protocol_path, "r+") as protocol:
        if "/sits" in protocol:
            del protocol["/sits"]
    restart_path = bundled_dir / "restart.spgr.h5"
    with h5py.File(restart_path, "r+") as restart:
        embedded_nk = "/parameters/restart/protocol_sidecars/SITS_nk_in_file"
        if embedded_nk in restart:
            del restart[embedded_nk]
        sidecar_table = "/parameters/sponge/files/legacy_sidecars"
        if sidecar_table in restart:
            del restart[sidecar_table]
    nk_sidecar = bundled_dir / "legacy_sidecars" / "SITS_nk_in_file"
    if nk_sidecar.exists():
        shutil.rmtree(nk_sidecar)
    _validate_focused_sits_nk_typed_restart_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_sits_typed_config_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_dir, bundled_dir = _prepare_focused_sits_nk_typed_restart_pair(
        case_root
    )
    bundled_mdin_path = bundled_dir / "mdin.bundled.spg.toml"
    typed_keys = (
        "mode",
        "k_numbers",
        "T",
        "nk_rest",
        "nk_fix",
        "pe_a",
        "pe_b",
        "fb_bias",
        "fb_interval",
    )
    bundled_mdin = _remove_key_lines(
        bundled_mdin_path.read_text(encoding="utf-8"),
        {f"SITS_{key}" for key in typed_keys} | {"SITS_atom_numbers"},
    )
    bundled_mdin_path.write_text(bundled_mdin, encoding="utf-8")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    typed_values = (
        "production",
        "2",
        "300/600",
        "false",
        "true",
        "1.0",
        "0.0",
        "0.0",
        "1",
    )
    with h5py.File(protocol_path, "r+") as protocol:
        sits = protocol.create_group("/sits")
        sits.create_dataset("atom_indices", data=[0, 1], dtype="i4")
        config = sits.create_group("config")
        config.create_dataset("key", data=typed_keys, dtype=string_dtype)
        config.create_dataset("value", data=typed_values, dtype=string_dtype)
        section = config.create_group("section")
        section.create_dataset("count", data=1, dtype="i8")
        section.create_dataset(
            "key_offset", data=[0, len(typed_keys)], dtype="i8"
        )
        section.create_dataset("name", data=["SITS"], dtype=string_dtype)

    _validate_focused_sits_typed_config_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_focused_sits_typed_inactive_pair(
    case_root: Path,
) -> tuple[Path, Path]:
    legacy_source = case_root / "focused_sits_typed_inactive_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_focused_sits_typed_inactive"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_focused_sits_typed_inactive_input(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)

    protocol_path = bundled_dir / "protocol.spgp.h5"
    with h5py.File(protocol_path, "r+") as protocol:
        if "/sits/config" not in protocol:
            raise AssertionError(
                "inactive SITS conversion did not emit typed configuration"
            )
    _remove_h5_sidecar_binding(protocol_path, "SITS_in_file")

    restart_path = bundled_dir / "restart.spgr.h5"
    with h5py.File(restart_path, "r+") as restart:
        embedded = "/parameters/restart/protocol_sidecars/SITS_in_file"
        if embedded in restart:
            del restart[embedded]
    _remove_h5_sidecar_binding(restart_path, "SITS_in_file")

    sits_sidecar = bundled_dir / "legacy_sidecars" / "SITS_in_file"
    if sits_sidecar.exists():
        shutil.rmtree(sits_sidecar)
    _validate_focused_sits_typed_inactive_routes(legacy_dir, bundled_dir)
    return legacy_dir, bundled_dir


def _remove_h5_sidecar_binding(h5_path: Path, removed_key: str) -> None:
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(h5_path, "r+") as h5_file:
        if sidecar_root not in h5_file:
            return
        sidecars = h5_file[sidecar_root]
        keys = sidecars["key"].asstr()[...].tolist()
        paths = sidecars["path"].asstr()[...].tolist()
        retained = [
            item
            for item in zip(keys, paths, strict=True)
            if item[0] != removed_key
        ]
        if len(retained) == len(keys):
            return
        if not retained:
            del h5_file[sidecar_root]
            return

        del sidecars["key"]
        del sidecars["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        sidecars.create_dataset(
            "key", data=[item[0] for item in retained], dtype=string_dtype
        )
        sidecars.create_dataset(
            "path", data=[item[1] for item in retained], dtype=string_dtype
        )


def _write_focused_sits_typed_inactive_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n2.0 0.0 0.0\n1000.0 1000.0 1000.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "sits.txt").write_text(
        "SITS\n{\n    CV = distance\n    nk = 2\n}\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab inactive typed SITS"\n'
        'mode = "nve"\n'
        "pbc = false\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        'mass_in_file = "mass.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'SITS_in_file = "sits.txt"\n'
        "force_whole_output = true\n"
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_sits_typed_inactive_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "SITS_in_file"):
        raise AssertionError("inactive SITS legacy route lost SITS_in_file")
    if _has_key_line(bundled_mdin, "SITS_in_file"):
        raise AssertionError("inactive typed SITS route retained SITS_in_file")

    protocol_path = bundled_dir / "protocol.spgp.h5"
    keys = _h5_string_values(protocol_path, "/sits/config/key")
    values = _h5_string_values(protocol_path, "/sits/config/value")
    if keys != ["CV", "nk"] or values != ["distance", "2"]:
        raise AssertionError(
            f"inactive typed SITS payload changed: keys={keys}, values={values}"
        )
    if "mode" in keys:
        raise AssertionError(
            "inactive typed SITS fixture unexpectedly has mode"
        )
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    if sidecar_root in _h5_paths(protocol_path):
        sidecar_keys = _h5_string_values(protocol_path, f"{sidecar_root}/key")
        if "SITS_in_file" in sidecar_keys:
            raise AssertionError(
                "inactive typed SITS retained protocol sidecar"
            )
    if (bundled_dir / "legacy_sidecars" / "SITS_in_file").exists():
        raise AssertionError("inactive typed SITS retained external sidecar")


def _write_focused_sits_nk_typed_restart_input(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "mass.txt").write_text("2\n12.0\n12.0\n", encoding="utf-8")
    (case_dir / "charge.txt").write_text("2\n1.0\n-1.0\n", encoding="utf-8")
    (case_dir / "lj.txt").write_text(
        "2 1\n4096.0\n128.0\n0\n0\n", encoding="utf-8"
    )
    (case_dir / "coordinate.txt").write_text(
        "2 0.0\n0.0 0.0 0.0\n2.0 0.0 0.0\n20.0 20.0 20.0\n90.0 90.0 90.0\n",
        encoding="utf-8",
    )
    (case_dir / "velocity.txt").write_text(
        "2\n0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    (case_dir / "sits_nk.txt").write_text("1.0 4.0\n", encoding="utf-8")
    (case_dir / "mdin.spg.toml").write_text(
        'md_name = "bundled io ab focused SITS nk typed restart"\n'
        'mode = "nvt"\n'
        "pbc = true\n"
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 10.0\n"
        "target_temperature = 300.0\n"
        'thermostat = "middle_langevin"\n'
        "thermostat_tau = 0.01\n"
        "thermostat_seed = 20260713\n"
        'mass_in_file = "mass.txt"\n'
        'charge_in_file = "charge.txt"\n'
        'LJ_in_file = "lj.txt"\n'
        'coordinate_in_file = "coordinate.txt"\n'
        'velocity_in_file = "velocity.txt"\n'
        'SITS_nk_in_file = "sits_nk.txt"\n'
        "print_zeroth_frame = 0\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n"
        'SITS_mode = "production"\n'
        "SITS_atom_numbers = 2\n"
        "SITS_k_numbers = 2\n"
        'SITS_T = "300/600"\n'
        "SITS_nk_rest = true\n"
        "SITS_nk_fix = true\n"
        "SITS_pe_a = 1.0\n"
        "SITS_pe_b = 0.0\n"
        "SITS_fb_bias = 0.0\n"
        "SITS_fb_interval = 1\n",
        encoding="utf-8",
    )


def _validate_focused_sits_nk_typed_restart_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    if not _has_key_line(legacy_mdin, "SITS_nk_in_file"):
        raise AssertionError("focused SITS legacy route lost SITS_nk_in_file")
    if _has_key_line(bundled_mdin, "SITS_nk_in_file"):
        raise AssertionError(
            "focused SITS bundled mdin retained SITS_nk_in_file"
        )
    if (
        re.search(r"(?m)^SITS_nk_rest\s*=\s*(?:false|0)\s*$", bundled_mdin)
        is None
    ):
        raise AssertionError(
            "focused SITS bundled route did not disable text Nk"
        )
    for text, label in ((legacy_mdin, "legacy"), (bundled_mdin, "bundled")):
        for key in ("SITS_mode", "SITS_atom_numbers", "SITS_k_numbers"):
            if not _has_key_line(text, key):
                raise AssertionError(
                    f"focused SITS {label} route lost inline key {key}"
                )

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    topology_paths = _h5_paths(topology_path)
    for typed_path in (
        "/atoms/mass",
        "/atoms/charge",
        "/forcefield/lj/type",
        "/forcefield/lj/pair_A_12",
        "/forcefield/lj/pair_B_6",
    ):
        if typed_path not in topology_paths:
            raise AssertionError(
                f"focused SITS topology lost typed support {typed_path}"
            )
    topology_keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
    topology_sidecar_paths = _h5_string_values(
        topology_path, f"{sidecar_root}/path"
    )
    expected_topology_sidecars = {
        "mass_in_file": "legacy_sidecars/mass_in_file/mass.txt",
        "charge_in_file": "legacy_sidecars/charge_in_file/charge.txt",
        "LJ_in_file": "legacy_sidecars/LJ_in_file/lj.txt",
    }
    if dict(zip(topology_keys, topology_sidecar_paths, strict=True)) != (
        expected_topology_sidecars
    ):
        raise AssertionError(
            "focused SITS topology support routes changed: "
            f"keys={topology_keys}, paths={topology_sidecar_paths}"
        )
    protocol_path = bundled_dir / "protocol.spgp.h5"
    protocol_paths = _h5_paths(protocol_path)
    if any(path.startswith("/sits") for path in protocol_paths):
        raise AssertionError(
            "focused SITS bundled protocol retained typed data"
        )
    if sidecar_root in protocol_paths:
        raise AssertionError(
            "focused SITS protocol file unexpectedly retained sidecars"
        )
    restart_path = bundled_dir / "restart.spgr.h5"
    restart_paths = _h5_paths(restart_path)
    typed_nk_path = "/parameters/restart/bias/sits/SITS/nk"
    if typed_nk_path not in restart_paths:
        raise AssertionError("focused SITS bundled restart lost typed Nk state")
    typed_nk = _h5_numeric_values(restart_path, typed_nk_path)
    _assert_numeric_sequences_close(
        "focused SITS typed Nk payload",
        (1.0, 4.0),
        typed_nk,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    if sidecar_root in restart_paths:
        raise AssertionError("focused SITS restart retained sidecar path table")
    embedded_path = "/parameters/restart/protocol_sidecars/SITS_nk_in_file"
    if embedded_path in restart_paths:
        raise AssertionError("focused SITS restart retained embedded Nk text")
    if (bundled_dir / "legacy_sidecars" / "SITS_nk_in_file").exists():
        raise AssertionError("focused SITS retained external Nk sidecar file")
    for key in ("mass_in_file", "charge_in_file", "LJ_in_file"):
        if not (bundled_dir / "legacy_sidecars" / key).exists():
            raise AssertionError(
                f"focused SITS bundled branch lost {key} support sidecar"
            )


def _validate_focused_sits_typed_config_routes(
    legacy_dir: Path, bundled_dir: Path
) -> None:
    legacy_mdin = (legacy_dir / "mdin.spg.toml").read_text(encoding="utf-8")
    bundled_mdin = (bundled_dir / "mdin.bundled.spg.toml").read_text(
        encoding="utf-8"
    )
    for key in ("SITS_mode", "SITS_atom_numbers", "SITS_k_numbers", "SITS_T"):
        if not _has_key_line(legacy_mdin, key):
            raise AssertionError(f"focused typed SITS legacy route lost {key}")
    if re.search(r"(?m)^\s*SITS_[A-Za-z0-9_]+\s*=", bundled_mdin):
        raise AssertionError(
            "focused typed SITS bundled route retained inline SITS ownership"
        )

    protocol_path = bundled_dir / "protocol.spgp.h5"
    expected_keys = (
        "mode",
        "k_numbers",
        "T",
        "nk_rest",
        "nk_fix",
        "pe_a",
        "pe_b",
        "fb_bias",
        "fb_interval",
    )
    expected_values = (
        "production",
        "2",
        "300/600",
        "false",
        "true",
        "1.0",
        "0.0",
        "0.0",
        "1",
    )
    if (
        tuple(_h5_string_values(protocol_path, "/sits/config/key"))
        != expected_keys
    ):
        raise AssertionError("focused typed SITS config keys changed")
    if (
        tuple(_h5_string_values(protocol_path, "/sits/config/value"))
        != expected_values
    ):
        raise AssertionError("focused typed SITS config values changed")
    if _h5_numeric_values(protocol_path, "/sits/atom_indices") != [0, 1]:
        raise AssertionError("focused typed SITS atom indices changed")
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    if sidecar_root in _h5_paths(protocol_path):
        raise AssertionError("focused typed SITS protocol retained sidecars")


def _write_sits_ff19sb_cmap_mdin(case_dir: Path, replica_seed: int):
    _assert_sits_ff19sb_cmap_fixture(case_dir / "ALA_cmap.txt")
    masses = read_mass_values(case_dir / "ALA_mass.txt")
    constrain_pair_count = Extractor.read_first_field_int(
        case_dir / "ALA_bond.txt"
    )
    write_velocity_file_for_temperature(
        case_dir / "initial_velocity.txt",
        masses,
        temperature=300.0,
        seed=replica_seed,
        degrees_of_freedom=3 * len(masses) - constrain_pair_count,
    )
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(
            [
                'md_name = "bundled io ab sits ff19sb cmap"',
                'mode = "nve"',
                "step_limit = 10",
                "dt = 0.002",
                "cutoff = 8.0",
                "dont_check_input = 1",
                'default_in_file_prefix = "ALA"',
                'velocity_in_file = "initial_velocity.txt"',
                "print_zeroth_frame = 1",
                "write_mdout_interval = 1",
                "write_information_interval = 1",
                'constrain_mode = "SHAKE"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _assert_sits_ff19sb_cmap_fixture(path: Path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 4:
        raise AssertionError(f"SITS ff19SB CMAP fixture is too short: {path}")
    if lines[0] != "1 1":
        raise AssertionError(
            f"SITS ff19SB ACE-ALA-NME must contain one CMAP entry: {path}"
        )
    if lines[1] != "24":
        raise AssertionError(
            f"SITS ff19SB ACE-ALA-NME CMAP resolution must be 24: {path}"
        )
    if lines[-1] != "4 6 8 14 16 0":
        raise AssertionError(
            "SITS ff19SB ACE-ALA-NME CMAP atom/type entry changed: "
            f"{lines[-1]!r}"
        )


def _reset_xponge_coordinate_start_time(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"empty coordinate file: {path}")
    fields = lines[0].split()
    if len(fields) >= 2:
        fields[1] = "0.0000000000"
        lines[0] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tip3p_mdin(case_dir: Path):
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(
            [
                'md_name = "bundled io ab tip3p normal"',
                'mode = "nve"',
                "step_limit = 10",
                "dt = 0.002",
                "cutoff = 8.0",
                'default_in_file_prefix = "tip3p"',
                "print_zeroth_frame = 1",
                "write_mdout_interval = 1",
                "write_information_interval = 1",
                'constrain_mode = "SETTLE"',
                'velocity_in_file = "initial_velocity.txt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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


def _convert_legacy_case(case_root: Path, output_dir: Path):
    _run(
        [
            _xponge_python(),
            "-m",
            "Xponge",
            "legacy-to-bundle",
            case_root,
            "-o",
            output_dir,
            "-m",
            "mdin.spg.toml",
        ],
        env=_xponge_env(),
    )
    assert (output_dir / "bundle" / "mdin.bundled.spg.toml").exists()


def _mdin_name(case_dir: Path) -> str:
    bundled = case_dir / "mdin.bundled.spg.toml"
    if bundled.exists():
        return bundled.name
    return "mdin.spg.toml"


def _prepare_mdin(
    case_dir: Path,
    mdin_name: str,
    case: AbCase,
    *,
    branch: str,
    replica_seed: int,
):
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    mdin_path = case_dir / mdin_name
    text = mdin_path.read_text(encoding="utf-8")
    remove_keys = {
        "step_limit",
        "write_mdout_interval",
        "write_trajectory_interval",
        "write_restart_file_interval",
        "output_h5_trajectory_path",
        "output_h5_trajectory_vds",
        "output_h5_trajectory_chunk_size",
        "output_h5_trajectory_repair_policy",
        "output_h5_restart_path",
        "output_h5_observable_path",
        "rerun_start",
        "rerun_strip",
        "rerun_frame_limit",
        "rerun_need_box_update",
        "input_h5_restart_load",
        "input_h5_trajectory_path",
        "input_h5_trajectory_particle_stream",
        "mdout",
        "mdinfo",
        "crd",
        "box",
        "vel",
        "frc",
        "rst",
        "rerun_output_crd",
        "rerun_output_box",
        "rerun_output_vel",
    }
    if case.mode in {"normal", "chunk_boundary"}:
        remove_keys.update(
            {
                "mode",
                "thermostat",
                "thermostat_seed",
                "thermostat_tau",
                "target_temperature",
            }
        )
        if case.normal_dt is not None:
            remove_keys.add("dt")
    text = _remove_key_lines(
        text,
        remove_keys,
    )
    limits = PROFILE_LIMITS[PROFILE]
    if case.mode in {"normal", "chunk_boundary"}:
        step_limit = case.normal_step_limit or limits["normal_step_limit"]
        interval = case.normal_interval or limits["normal_interval"]
        additions = []
        if case.statistical_md:
            additions.extend(
                [
                    'mode = "nvt"',
                    'thermostat = "middle_langevin"',
                    f"thermostat_seed = {replica_seed}",
                    "thermostat_tau = 0.1",
                    "target_temperature = 300.0",
                ]
            )
        else:
            additions.append('mode = "nve"')
        if case.normal_dt is not None:
            additions.append(f"dt = {case.normal_dt}")
        additions.extend(
            [
                f"step_limit = {step_limit}",
                f"write_mdout_interval = {interval}",
                f"write_trajectory_interval = {interval}",
                f"write_restart_file_interval = {step_limit}",
                'crd = "output/legacy.crd"',
                'box = "output/legacy.box"',
                'vel = "output/legacy.vel"',
                'frc = "output/legacy.frc"',
                'rst = "output/legacy_restart"',
            ]
        )
        if (
            branch == "bundled"
            and case.restart_load_policy != "structural"
            and _has_key_line(text, "input_h5_restart_path")
        ):
            additions.append(
                f'input_h5_restart_load = "{case.restart_load_policy}"'
            )
    else:
        additions = [
            f"rerun_start = {case.rerun_start}",
            f"rerun_strip = {case.rerun_strip}",
            f"rerun_need_box_update = {1 if case.rerun_need_box_update else 0}",
            "write_mdout_interval = 1",
            "write_trajectory_interval = 1",
            "write_restart_file_interval = 0",
        ]
        if case.rerun_frame_limit is not None:
            additions.append(f"rerun_frame_limit = {case.rerun_frame_limit}")
        if branch == "legacy":
            additions.extend(
                [
                    'crd = "traj.dat"',
                    'box = "traj_box.dat"',
                ]
            )
            if case.rerun_velocity_present:
                additions.append('vel = "traj_vel.dat"')
        else:
            additions.extend(
                [
                    f'input_h5_trajectory_path = "{case.trajectory_file_name}"',
                    "input_h5_trajectory_particle_stream = "
                    f'"{case.trajectory_particle_stream}"',
                ]
            )
        additions.extend(
            [
                f'rerun_output_crd = "output/{branch}_rerun.crd"',
                f'rerun_output_box = "output/{branch}_rerun.box"',
            ]
        )
        if case.rerun_velocity_present:
            additions.append(f'rerun_output_vel = "output/{branch}_rerun.vel"')
        if case.rerun_force_output:
            additions.append(f'frc = "output/{branch}_rerun.frc"')
        if _has_key_line(text, "input_h5_restart_path"):
            additions.append(
                f'input_h5_restart_load = "{case.restart_load_policy}"'
            )
    additions.extend(
        [
            'mdout = "mdout.txt"',
            'mdinfo = "mdinfo.txt"',
            f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
            f"output_h5_trajectory_vds = {'true' if case.vds else 'false'}",
            f"output_h5_trajectory_chunk_size = {case.output_chunk_size}",
            "output_h5_trajectory_repair_policy = "
            f'"{case.output_repair_policy}"',
            f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        ]
    )
    if case.mode == "normal":
        additions.append(f'output_h5_restart_path = "{RESTART_REL.as_posix()}"')
    mdin_path.write_text(
        _insert_root_toml_keys(text, additions),
        encoding="utf-8",
    )
    assert branch in {"legacy", "bundled"}


def _insert_root_toml_keys(text: str, additions: Sequence[str]) -> str:
    lines = text.rstrip().splitlines()
    table_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("[")
        ),
        len(lines),
    )
    updated = [
        *lines[:table_index],
        *additions,
        *lines[table_index:],
    ]
    return "\n".join(updated) + "\n"


def _mutate_failure_mdin(case: AbCase, mdin_path: Path, branch: str) -> None:
    mutation = case.failure_mutation
    if mutation is None or branch not in case.failure_branches:
        return
    remove_keys: set[str] = set()
    additions: list[str] = []
    if mutation == "missing_trajectory":
        remove_keys.update({"crd", "box", "vel", "input_h5_trajectory_path"})
    elif mutation == "invalid_chunk_size":
        remove_keys.add("output_h5_trajectory_chunk_size")
        additions.append("output_h5_trajectory_chunk_size = 0")
    elif mutation == "invalid_vds_value":
        remove_keys.add("output_h5_trajectory_vds")
        additions.append('output_h5_trajectory_vds = "invalid"')
    elif mutation == "invalid_repair_policy":
        remove_keys.add("output_h5_trajectory_repair_policy")
        additions.append('output_h5_trajectory_repair_policy = "invalid"')
    elif mutation == "invalid_restart_policy":
        remove_keys.add("input_h5_restart_load")
        additions.append('input_h5_restart_load = "invalid"')
    elif mutation == "missing_topology":
        remove_keys.add("input_h5_topology_path")
    elif mutation == "missing_protocol":
        remove_keys.add("input_h5_protocol_path")
    elif mutation == "mixed_trajectory":
        additions.extend(['crd = "traj.dat"', 'box = "traj_box.dat"'])
    elif mutation == "mixed_restart":
        additions.extend(
            [
                'coordinate_in_file = "coordinate.txt"',
                'velocity_in_file = "velocity.txt"',
            ]
        )
    elif mutation == "h5_positional_restraint_dual_owner":
        additions.append('restrain_atom_id = "duplicate_restrain_atom_id.txt"')
    elif mutation == "h5_soft_wall_dual_owner":
        additions.append('soft_walls_in_file = "duplicate_soft_walls.txt"')
    elif mutation == "h5_cv_restraint_dual_owner":
        additions.append('restrain_cv_in_file = "duplicate_restrain_cv.txt"')
    elif mutation in {
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
        "h5_topology_atom_count_mismatch",
        "h5_topology_mass_shape",
        "h5_topology_mass_dtype",
        "h5_topology_schema_version",
        "h5_nb14_dual_root",
        "h5_nb14_extra_param_shape",
        "h5_eam_unknown_format",
        "h5_eam_embed_shape",
        "h5_positional_restraint_weight_shape",
        "h5_soft_wall_count_shape",
        "h5_soft_wall_name_shape",
        "h5_soft_wall_potential_shape",
        "h5_cv_restraint_partial_owner",
        "h5_cv_restraint_offset_mismatch",
        "h5_cv_restraint_definition_conflict",
        "restart_dynamic_without_owner",
        "restart_protocol_without_owner",
        "restart_full_without_owner",
    }:
        return
    else:
        raise AssertionError(f"unknown failure mutation: {mutation}")

    text = mdin_path.read_text(encoding="utf-8")
    text = _remove_key_lines(text, remove_keys)
    mdin_path.write_text(
        _insert_root_toml_keys(text, additions), encoding="utf-8"
    )


def _mutate_failure_h5(case: AbCase, case_dir: Path, branch: str) -> None:
    mutation = case.failure_mutation
    sidecar_mutations = {
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
    }
    metadata_mutations = {
        "h5_topology_atom_count_mismatch",
        "h5_topology_mass_shape",
        "h5_topology_mass_dtype",
        "h5_topology_schema_version",
        "h5_nb14_dual_root",
        "h5_nb14_extra_param_shape",
        "h5_eam_unknown_format",
        "h5_eam_embed_shape",
    }
    protocol_mutations = {
        "h5_positional_restraint_weight_shape",
        "h5_soft_wall_count_shape",
        "h5_soft_wall_name_shape",
        "h5_soft_wall_potential_shape",
        "h5_cv_restraint_partial_owner",
        "h5_cv_restraint_offset_mismatch",
        "h5_cv_restraint_definition_conflict",
    }
    if (
        mutation
        not in sidecar_mutations | metadata_mutations | protocol_mutations
        or branch not in case.failure_branches
    ):
        return
    if branch != "bundled":
        raise AssertionError(f"{mutation} requires the bundled H5 branch")

    topology_path = case_dir / "topology.spgt.h5"
    if mutation in protocol_mutations:
        protocol_path = case_dir / "protocol.spgp.h5"
        with h5py.File(protocol_path, "r+") as protocol:
            if mutation == "h5_positional_restraint_weight_shape":
                del protocol["/restraint/default/weight"]
                protocol.create_dataset(
                    "/restraint/default/weight",
                    data=[[10.0, 0.0], [0.0, 20.0]],
                    dtype="f4",
                )
            elif mutation == "h5_soft_wall_count_shape":
                del protocol["/wall/soft/count"]
                protocol.create_dataset(
                    "/wall/soft/count", data=[1], dtype="i8"
                )
            elif mutation in {
                "h5_soft_wall_name_shape",
                "h5_soft_wall_potential_shape",
            }:
                dataset_name = (
                    "name"
                    if mutation == "h5_soft_wall_name_shape"
                    else "potential"
                )
                path = f"/wall/soft/{dataset_name}"
                del protocol[path]
                protocol.create_dataset(
                    path,
                    data=[],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            elif mutation == "h5_cv_restraint_partial_owner":
                del protocol["/restraint/cv"]
            elif mutation == "h5_cv_restraint_offset_mismatch":
                protocol["/restraint/cv/config/section/key_offset"][...] = [
                    0,
                    2,
                ]
            else:
                root = "/restraint/config"
                keys = protocol[f"{root}/key"].asstr()[...].tolist()
                values = protocol[f"{root}/value"].asstr()[...].tolist()
                values[keys.index("atom")] = "0 0"
                del protocol[f"{root}/value"]
                protocol.create_dataset(
                    f"{root}/value",
                    data=values,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
        return
    if mutation in metadata_mutations:
        with h5py.File(topology_path, "r+") as topology:
            if mutation == "h5_topology_atom_count_mismatch":
                topology["/topology/atom_count"][...] = 3
                return
            if mutation == "h5_topology_schema_version":
                _replace_h5_string_dataset(
                    topology,
                    "/schema/version",
                    "unsupported.topology.v999",
                )
                return
            if mutation == "h5_nb14_dual_root":
                canonical = topology.create_group("/forcefield/nb14")
                canonical.create_dataset("atoms", data=[[0, 1]], dtype="i4")
                canonical.create_dataset(
                    "params", data=[[12.0, 6.0, 0.5]], dtype="f4"
                )
                return
            if mutation == "h5_nb14_extra_param_shape":
                del topology["/forcefield/nb14_extra/params"]
                topology.create_dataset(
                    "/forcefield/nb14_extra/params",
                    data=[[1.0, 0.5]],
                    dtype="f4",
                )
                return
            if mutation == "h5_eam_unknown_format":
                _replace_h5_string_dataset(
                    topology,
                    "/manybody/eam/format",
                    "unsupported-eam-format",
                )
                return
            if mutation == "h5_eam_embed_shape":
                del topology["/manybody/eam/embed/raw_ev"]
                topology.create_dataset(
                    "/manybody/eam/embed/raw_ev",
                    data=[[0.0, 1.0, 2.0]],
                    dtype="f4",
                )
                return
            del topology["/atoms/mass"]
            if mutation == "h5_topology_mass_shape":
                topology.create_dataset(
                    "/atoms/mass",
                    data=[[12.011], [15.999]],
                    dtype="f4",
                )
            else:
                topology.create_dataset(
                    "/atoms/mass",
                    data=["12.011", "15.999"],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
        return

    group_path = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r+") as h5:
        if group_path not in h5:
            raise AssertionError(f"sidecar table is missing: {topology_path}")
        group = h5[group_path]
        keys = group["key"].asstr()[...].tolist()
        paths = group["path"].asstr()[...].tolist()
        if len(keys) != len(paths):
            raise AssertionError(
                f"source sidecar table is already malformed: {topology_path}"
            )

        if mutation == "unsupported_sidecar_key":
            keys.append("not_a_supported_sidecar_key")
            paths.append("legacy_sidecars/mass_in_file/mass.txt")
        elif mutation == "sidecar_length_mismatch":
            paths.pop()
        else:
            if "mass_in_file" not in keys:
                raise AssertionError(
                    f"mass_in_file is missing from sidecar table: {topology_path}"
                )
            keys.append("mass_in_file")
            paths.append("legacy_sidecars/charge_in_file/charge.txt")

        del group["key"]
        del group["path"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("key", data=keys, dtype=string_dtype)
        group.create_dataset("path", data=paths, dtype=string_dtype)


def _run_supported_topology_schema_controls(
    case: AbCase, bundled_dir: Path
) -> dict[str, object]:
    results = {}
    for version in SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS:
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", version).strip("_")
        control_dir = bundled_dir.parent / f"bundled_schema_{suffix}"
        if control_dir.exists():
            shutil.rmtree(control_dir)
        shutil.copytree(bundled_dir, control_dir)
        with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
            _replace_h5_string_dataset(topology, "/schema/version", version)
        outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
        if outcome.returncode != 0:
            raise AssertionError(
                f"{case.name} rejected supported topology schema {version!r} "
                f"with code {outcome.returncode}\n"
                f"{outcome.stdout}\n{outcome.stderr}"
            )
        results[version] = {
            "exit_code": outcome.returncode,
            "elapsed_s": outcome.elapsed_s,
        }
        shutil.rmtree(control_dir)
    return results


def _replace_h5_string_dataset(h5: h5py.File, path: str, value: str) -> None:
    if path in h5:
        del h5[path]
    h5.create_dataset(
        path,
        data=value,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _canonical_h5_shape_text(shape: Sequence[int]) -> str:
    if not shape:
        return "()"
    values = ", ".join(str(int(value)) for value in shape)
    if len(shape) == 1:
        values += ","
    return f"({values})"


def _canonical_restart_state_hash(path: Path) -> str:
    datasets: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    with h5py.File(path, "r") as restart:
        particle_paths = (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/box/edges/value",
            "/particles/all/force/value",
        )
        paths = [dataset for dataset in particle_paths if dataset in restart]

        def collect(_name: str, item) -> None:
            if isinstance(item, h5py.Dataset):
                paths.append(item.name)

        if "/parameters/restart" in restart:
            restart["/parameters/restart"].visititems(collect)

        for dataset_path in sorted(set(paths)):
            dataset = restart[dataset_path]
            shape = tuple(int(value) for value in dataset.shape)
            if h5py.check_string_dtype(dataset.dtype) is not None:
                values = dataset.asstr()[()]
                if isinstance(values, str):
                    payload = values.encode("utf-8")
                else:
                    payload = b"\0".join(
                        str(value).encode("utf-8")
                        for value in values.reshape(-1)
                    )
                datasets[dataset_path] = ("object", shape, payload)
                continue
            kind = dataset.dtype.kind
            itemsize = dataset.dtype.itemsize
            dtype_name = {
                ("f", 4): "float32",
                ("f", 8): "float64",
                ("i", 1): "int8",
                ("i", 4): "int32",
                ("i", 8): "int64",
                ("u", 1): "uint8",
                ("u", 4): "uint32",
                ("u", 8): "uint64",
            }.get((kind, itemsize))
            if dtype_name is None:
                raise AssertionError(
                    f"unsupported restart hash dtype at {dataset_path}: "
                    f"{dataset.dtype}"
                )
            datasets[dataset_path] = (
                dtype_name,
                shape,
                dataset[...].tobytes(order="C"),
            )

    digest = hashlib.sha256()
    digest.update(b"restart.spgr.h5")
    for dataset_path, (dtype_name, shape, payload) in datasets.items():
        digest.update(b"\0")
        digest.update(dataset_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canonical_h5_shape_text(shape).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _refresh_restart_state_hash(path: Path) -> None:
    state_hash = _canonical_restart_state_hash(path)
    with h5py.File(path, "r+") as restart:
        _replace_h5_string_dataset(restart, "/run/state_hash", state_hash)


def _remove_key_lines(text: str, keys: set[str]) -> str:
    output = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in keys:
            output.append(line)
    return "\n".join(output)


def _has_key_line(text: str, key: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split("=", 1)[0].strip() == key:
            return True
    return False


def _run_sponge(case_dir: Path, mdin_name: str) -> dict[str, object]:
    outcome = _run_sponge_process(case_dir, mdin_name)
    if outcome.returncode != 0:
        raise AssertionError(
            f"SPONGE failed in {case_dir} with code {outcome.returncode}\n"
            f"[stdout]\n{outcome.stdout}\n[stderr]\n{outcome.stderr}"
        )
    metrics = _collect_metrics(case_dir, outcome.elapsed_s)
    metrics["finite_scan"] = _scan_success_outputs(case_dir)
    return metrics


def _scan_success_outputs(case_dir: Path) -> dict[str, object]:
    files = []
    dataset_count = 0
    value_count = 0

    mdout_paths = []
    for candidate in (
        case_dir / "mdout.txt",
        *sorted(case_dir.glob("*.mdout")),
    ):
        if candidate.is_file() and candidate not in mdout_paths:
            mdout_paths.append(candidate)
    for path in mdout_paths:
        mdout = _read_mdout(path)
        values = [value for row in mdout["rows"] for value in row.values()]
        _assert_finite_values(f"normal-success mdout {path}", values)
        files.append(
            {
                "path": str(path.relative_to(case_dir)),
                "kind": "mdout",
                "dataset_count": len(mdout["columns"]),
                "value_count": len(values),
            }
        )
        dataset_count += len(mdout["columns"])
        value_count += len(values)

    output_dir = case_dir / "output"
    h5_paths = []
    if output_dir.is_dir():
        h5_paths = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix in {".h5", ".h5md"}
        )
    for path in h5_paths:
        scanned = _scan_success_h5_file(case_dir, path)
        files.append(scanned)
        dataset_count += scanned["dataset_count"]
        value_count += scanned["value_count"]

    numeric_sidecars = []
    if output_dir.is_dir():
        numeric_sidecars = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.suffix in {".crd", ".vel", ".frc", ".box"}
        )
    for path in numeric_sidecars:
        payload = path.read_bytes()
        text_payload = None
        try:
            candidate = payload.decode("ascii")
            if re.fullmatch(r"[\s0-9eE+.\-]*", candidate):
                text_payload = candidate
        except UnicodeDecodeError:
            pass
        if path.suffix == ".box" or text_payload is not None:
            if text_payload is None:
                text_payload = payload.decode("utf-8")
            values = [float(token) for token in text_payload.split()]
        else:
            if len(payload) % 4 != 0:
                raise AssertionError(
                    f"normal-success binary output is not float32-aligned: {path}"
                )
            values = [item[0] for item in struct.iter_unpack("f", payload)]
        _assert_finite_values(f"normal-success sidecar {path}", values)
        files.append(
            {
                "path": str(path.relative_to(case_dir)),
                "kind": f"legacy_{path.suffix.removeprefix('.')}",
                "dataset_count": 1,
                "value_count": len(values),
            }
        )
        dataset_count += 1
        value_count += len(values)

    if not files:
        raise AssertionError(
            f"normal-success run has no scannable outputs: {case_dir}"
        )
    return {
        "scan_kind": "normal_success_finite_scan",
        "run_directory": str(case_dir.resolve()),
        "file_count": len(files),
        "dataset_count": dataset_count,
        "value_count": value_count,
        "files": files,
    }


def _scan_success_h5_file(case_dir: Path, path: Path) -> dict[str, object]:
    dataset_count = 0
    value_count = 0
    with h5py.File(path, "r") as handle:
        numeric_datasets = []

        def collect(name: str, item: object) -> None:
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "biuf":
                numeric_datasets.append((name, item))

        handle.visititems(collect)
        for name, dataset in numeric_datasets:
            payload = dataset[()]
            values = payload.flat if hasattr(payload, "flat") else (payload,)
            values = [float(value) for value in values]
            _assert_finite_values(f"normal-success H5 {path}:{name}", values)
            dataset_count += 1
            value_count += len(values)
    return {
        "path": str(path.relative_to(case_dir)),
        "kind": "h5",
        "dataset_count": dataset_count,
        "value_count": value_count,
    }


def _assert_finite_values(label: str, values: Sequence[float]) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise AssertionError(
                f"{label} contains non-finite value at index {index}: {value}"
            )


def _summarize_finite_scans(*sources: object) -> dict[str, object]:
    scans: dict[str, Mapping[str, object]] = {}

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            if value.get("scan_kind") == "normal_success_finite_scan":
                scans[str(value["run_directory"])] = value
                return
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    for source in sources:
        collect(source)
    if not scans:
        raise AssertionError("normal-success evidence has no finite scan")
    ordered = [scans[key] for key in sorted(scans)]
    return {
        "classification": NORMAL_SUCCESS,
        "run_directory_count": len(ordered),
        "file_count": sum(int(scan["file_count"]) for scan in ordered),
        "dataset_count": sum(int(scan["dataset_count"]) for scan in ordered),
        "value_count": sum(int(scan["value_count"]) for scan in ordered),
        "run_directories": [scan["run_directory"] for scan in ordered],
    }


def _attach_finite_scan(
    case: AbCase,
    assertions: Sequence[AssertionEvidence],
    finite_scan: Mapping[str, object],
) -> tuple[AssertionEvidence, ...]:
    if _case_output_classification(case) != NORMAL_SUCCESS:
        raise AssertionError(
            f"{case.name} cannot attach normal-success finite evidence"
        )
    return tuple(
        replace(
            assertion,
            details={
                **dict(assertion.details),
                "output_classification": NORMAL_SUCCESS,
                "finite_scan": dict(finite_scan),
            },
        )
        for assertion in assertions
    )


def _build_success_case_evidence(
    contracts: Mapping[str, object],
    case: AbCase,
    assertions: Sequence[AssertionEvidence],
    *scan_sources: object,
) -> tuple[list[object], dict[str, object]]:
    finite_scan = _summarize_finite_scans(*scan_sources)
    enriched = _attach_finite_scan(case, assertions, finite_scan)
    return build_case_evidence(contracts, case, enriched), finite_scan


def _run_sponge_process(
    case_dir: Path,
    mdin_name: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> ProcessOutcome:
    start = time.perf_counter()
    environment = os.environ.copy()
    if extra_env:
        environment.update(extra_env)
    result = subprocess.run(
        [_sponge_executable(), "-mdin", mdin_name],
        cwd=case_dir,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=int(os.environ.get("SPONGE_BUNDLED_IO_AB_TIMEOUT", "7200")),
    )
    elapsed_s = time.perf_counter() - start
    (case_dir / "run.stdout").write_text(result.stdout, encoding="utf-8")
    (case_dir / "run.stderr").write_text(result.stderr, encoding="utf-8")
    return ProcessOutcome(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_s=elapsed_s,
    )


def _failure_category(text: str) -> str:
    match = re.search(r"\b(spongeError[A-Za-z0-9_]+) raised by\b", text)
    return match.group(1) if match else ""


def _sponge_executable() -> str:
    configured = os.environ.get("SPONGE_BUNDLED_IO_AB_SPONGE")
    if configured:
        return configured
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if not env_name:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            env_name = Path(conda_prefix).name
    candidates = []
    if env_name:
        candidates.append(REPO_ROOT / f"build-{env_name}" / "SPONGE")
    candidates.extend(
        [
            REPO_ROOT / "build-dev-cuda13" / "SPONGE",
            REPO_ROOT / "build-dev-cpu-h5-2" / "SPONGE",
            REPO_ROOT / "build" / "SPONGE",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    path_executable = shutil.which("SPONGE")
    if path_executable:
        return path_executable
    raise AssertionError(
        "SPONGE executable was not found. Set "
        "SPONGE_BUNDLED_IO_AB_SPONGE=/path/to/SPONGE or build SPONGE first."
    )


def _collect_metrics(case_dir: Path, elapsed_s: float) -> dict[str, object]:
    trajectory = case_dir / TRAJECTORY_REL
    observable = case_dir / OBSERVABLE_REL
    restart = case_dir / RESTART_REL
    h5_finalize = _parse_h5_finalize_timing(case_dir / "run.stdout")
    return {
        "elapsed_s": elapsed_s,
        "mdout_rows": len(_read_mdout(case_dir / "mdout.txt")["rows"]),
        "trajectory_h5_bytes": _file_size(trajectory),
        "observable_h5_bytes": _file_size(observable),
        "restart_h5_bytes": _file_size(restart),
        "vds_shard_count": _vds_shard_count(trajectory),
        "h5_files_total_bytes": sum(
            _file_size(path) for path in (trajectory, observable, restart)
        ),
        "flush_finalize_elapsed_s": h5_finalize["total_s"],
        "h5_finalize_timing": h5_finalize,
    }


def _parse_h5_finalize_timing(stdout_path: Path) -> dict[str, float]:
    pattern = re.compile(
        r"H5 I/O finalize timing: trajectory=(?P<trajectory>[0-9.eE+-]+) s, "
        r"observable=(?P<observable>[0-9.eE+-]+) s, "
        r"restart=(?P<restart>[0-9.eE+-]+) s, total=(?P<total>[0-9.eE+-]+) s"
    )
    text = stdout_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise AssertionError(
            f"H5 finalize timing was not reported in {stdout_path}"
        )
    return {
        "trajectory_s": float(match.group("trajectory")),
        "observable_s": float(match.group("observable")),
        "restart_s": float(match.group("restart")),
        "total_s": float(match.group("total")),
    }


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _vds_shard_count(wrapper_path: Path) -> int:
    shard_root = wrapper_path.with_suffix("").with_suffix(".shards")
    if not shard_root.exists():
        # ab.spg.h5md -> ab.spg.shards
        shard_root = wrapper_path.parent / "ab.spg.shards"
    if not shard_root.exists():
        return 0
    return len(list(shard_root.glob("segment_*.spg.h5md")))


def _compare_outputs(
    case: AbCase, runs: Sequence[AbRun]
) -> tuple[dict[str, object], tuple[AssertionEvidence, ...]]:
    if _case_output_classification(case) != NORMAL_SUCCESS:
        raise AssertionError(
            f"{case.name} non-success case reached the success comparator"
        )
    finite_scan = _summarize_finite_scans(
        [run.legacy_metrics for run in runs],
        [run.bundled_metrics for run in runs],
    )
    evidence = []
    if case.statistical_md:
        mdout_comparison = _compare_mdout_statistically(case, runs)
        h5_comparison = _compare_h5_outputs_statistically(case, runs)
        evidence.extend(
            (
                AssertionEvidence(
                    assertion_id="mdout_statistical_equivalence",
                    evidence_level="E3",
                    details={
                        "method": mdout_comparison["method"],
                        "column_count": len(mdout_comparison["columns"]),
                    },
                ),
                AssertionEvidence(
                    assertion_id="h5_statistical_equivalence",
                    evidence_level="E3",
                    details={
                        "families": sorted(h5_comparison),
                        "method": "all_dataset_schema_and_statistical_values",
                    },
                ),
            )
        )
    else:
        mdout_comparison = _compare_mdout_deterministically(case, runs[0])
        h5_comparison: dict[str, object] = {"method": "not_requested"}
        evidence.append(
            AssertionEvidence(
                assertion_id="mdout_deterministic_equivalence",
                evidence_level="E3",
                details={
                    "method": mdout_comparison["method"],
                    "row_count": mdout_comparison["rows"],
                    "columns": mdout_comparison["columns"],
                },
            )
        )
        if "h5_rerun_semantic_equivalence" in case.assertion_ids:
            h5_comparison = _compare_h5_outputs_deterministically(case, runs[0])
            evidence.append(
                AssertionEvidence(
                    assertion_id="h5_rerun_semantic_equivalence",
                    evidence_level="E3",
                    details={
                        "method": h5_comparison["method"],
                        "trajectory_frame_count": h5_comparison[
                            "trajectory_frame_count"
                        ],
                        "payloads": h5_comparison["payloads"],
                    },
                )
            )

    comparison: dict[str, object] = {
        "mdout": mdout_comparison,
        "h5": h5_comparison,
        "output_classification": NORMAL_SUCCESS,
        "finite_scan": finite_scan,
    }
    if case.fixture_case == FOCUSED_SITS_TYPED_INACTIVE_FIXTURE:
        comparison["inactive_sits"] = _compare_sits_inactive_configuration(
            runs[0]
        )
    if "rerun_selection_equivalence" in case.assertion_ids:
        rerun_selection = _compare_rerun_selection(case, runs[0])
        comparison["rerun_selection"] = rerun_selection
        evidence.append(
            AssertionEvidence(
                assertion_id="rerun_selection_equivalence",
                evidence_level="E3",
                details=rerun_selection,
            )
        )
    input_semantics = _compare_input_semantics(case, runs)
    if input_semantics:
        comparison["input_semantics"] = input_semantics
        contract_oracles = _build_input_contract_oracles(input_semantics)
        evidence.append(
            AssertionEvidence(
                assertion_id="input_semantic_equivalence",
                evidence_level="E3",
                details={
                    "contracts": [
                        item["contract_id"] for item in input_semantics
                    ],
                    "criterion": "present_nontrivial_module_owned_result",
                    "results": input_semantics,
                    "contract_oracles": contract_oracles,
                },
            )
        )
    if {
        "input.topology.nb14",
        "input.topology.nb14_extra",
    }.intersection(case.contract_ids):
        comparison["nb14"] = _assert_nb14_oracle(case, runs[0])
    if "constraint_geometry_equivalence" in case.assertion_ids:
        constraint = _compare_focused_constraint_projection(case, runs[0])
        comparison["constraint"] = constraint
        evidence.append(
            AssertionEvidence(
                assertion_id="constraint_geometry_equivalence",
                evidence_level="E3",
                details=constraint,
            )
        )
    if "output.legacy.mdinfo" in case.contract_ids:
        mdinfo_comparison = _compare_mdinfo_structured(case, runs)
        comparison["mdinfo"] = mdinfo_comparison
        evidence.append(
            AssertionEvidence(
                assertion_id="mdinfo_structured_equivalence",
                evidence_level="E3",
                details=mdinfo_comparison,
            )
        )
    if case.mode == "normal" and not case.input_behavior_only:
        restart_continuation = _compare_restart_continuation(case, runs[0])
        comparison["restart_continuation"] = restart_continuation
        evidence.extend(
            (
                AssertionEvidence(
                    assertion_id="particle_legacy_coexistence",
                    evidence_level="E3",
                    details={
                        "routes": ["crd", "box", "vel", "frc"],
                        "branches": ["legacy", "bundled"],
                        "same_run_h5_payload_comparison": True,
                        "cross_branch_assertion": "h5_statistical_equivalence",
                    },
                ),
                AssertionEvidence(
                    assertion_id="restart_structural_coexistence",
                    evidence_level="E3",
                    details={
                        "routes": ["rst", "output_h5_restart_path"],
                        "branches": ["legacy", "bundled"],
                        "comparison": "position_velocity_box_step_time",
                    },
                ),
                AssertionEvidence(
                    assertion_id="restart_continuation_equivalence",
                    evidence_level="E4",
                    details=restart_continuation,
                ),
            )
        )
    if "input.full_contract.inventory" in case.contract_ids:
        inventory = _validate_full_contract_input(case, runs[0].bundled_dir)
        comparison["full_contract_input_inventory"] = inventory
        evidence.append(
            AssertionEvidence(
                assertion_id="full_contract_input_inventory",
                evidence_level="E0",
                details=inventory,
            )
        )
    if "input.topology.cmap" in case.contract_ids:
        comparison["cmap"] = [
            _compare_cmap_materialization(run.legacy_dir, run.bundled_dir)
            for run in runs
        ]
    return comparison, _attach_finite_scan(case, evidence, finite_scan)


def _compare_sits_inactive_configuration(run: AbRun) -> dict[str, object]:
    legacy_mdout = _read_mdout(run.legacy_dir / "mdout.txt")
    bundled_mdout = _read_mdout(run.bundled_dir / "mdout.txt")
    for branch, mdout in (
        ("legacy", legacy_mdout),
        ("bundled", bundled_mdout),
    ):
        sits_columns = [
            column for column in mdout["columns"] if column.startswith("SITS")
        ]
        if sits_columns:
            raise AssertionError(
                f"inactive SITS {branch} route emitted columns: {sits_columns}"
            )

    materialized = run.bundled_dir / ".sponge_h5_native_protocol" / "sits.txt"
    if not materialized.exists():
        raise AssertionError("inactive typed SITS config was not materialized")
    legacy_config = run.legacy_dir / "sits.txt"
    if materialized.read_bytes() != legacy_config.read_bytes():
        raise AssertionError(
            "inactive typed SITS materialization differs from legacy config"
        )
    return {
        "mode_present": False,
        "module_initialized": False,
        "materialization": "exact_legacy_text",
    }


def _assert_nb14_oracle(case: AbCase, run: AbRun) -> dict[str, object]:
    expected_by_case = {
        "normal_nb14_scaled_nonzero": {
            "nb14_LJ": -0.75,
            "nb14_EE": -0.25,
            "force": (4.875, 0.0, 0.0, -4.875, 0.0, 0.0),
        },
        "normal_nb14_extra_nonzero": {
            "nb14_LJ": -0.03,
            "nb14_EE": -0.25,
            "force": (
                0.46728515625,
                0.0,
                0.0,
                -0.46728515625,
                0.0,
                0.0,
            ),
        },
    }
    expected = expected_by_case.get(case.name)
    if expected is None:
        return {"full_contract": "covered_by_input_semantic_equivalence"}

    for branch, case_dir in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        mdout = _read_mdout(case_dir / "mdout.txt")
        if len(mdout["rows"]) != 1:
            raise AssertionError(
                f"{case.name} {branch} NB14 oracle expected one mdout row"
            )
        row = mdout["rows"][0]
        for column in ("nb14_LJ", "nb14_EE"):
            _assert_numeric_sequences_close(
                f"{case.name} {branch} {column} oracle",
                (expected[column],),
                (row[column],),
                relative_tolerance=0.0,
                absolute_tolerance=1.0e-7,
            )
        force = _read_native_float32_file(case_dir / "output" / "legacy.frc")
        _assert_numeric_sequences_close(
            f"{case.name} {branch} force oracle",
            expected["force"],
            force,
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-7,
        )

    control_dir = run.bundled_dir.parent / "bundled_nb14_zero_lj_control"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    root = (
        "/forcefield/nb14"
        if case.name == "normal_nb14_scaled_nonzero"
        else "/forcefield/nb14_extra"
    )
    with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
        topology[f"{root}/params"][0, 0] = 0.0
        if root.endswith("nb14_extra"):
            topology[f"{root}/params"][0, 1] = 0.0

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} zero-LJ control failed with code {outcome.returncode}\n"
            f"{outcome.stdout}\n{outcome.stderr}"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1:
        raise AssertionError(f"{case.name} zero-LJ control expected one row")
    _assert_numeric_sequences_close(
        f"{case.name} zero-LJ energy",
        (0.0, expected["nb14_EE"]),
        (control_rows[0]["nb14_LJ"], control_rows[0]["nb14_EE"]),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-7,
    )
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    maximum_force_delta = max(
        abs(baseline - control)
        for baseline, control in zip(
            expected["force"], control_force, strict=True
        )
    )
    if maximum_force_delta <= 1.0e-2:
        raise AssertionError(
            f"{case.name} typed NB14 LJ parameters did not change force"
        )
    shutil.rmtree(control_dir)
    return {
        "nb14_LJ": expected["nb14_LJ"],
        "nb14_EE": expected["nb14_EE"],
        "force": list(expected["force"]),
        "zero_lj_control": {
            "nb14_LJ": control_rows[0]["nb14_LJ"],
            "nb14_EE": control_rows[0]["nb14_EE"],
            "maximum_force_delta": maximum_force_delta,
        },
    }


def _expected_rerun_frame_indices(case: AbCase, frame_count: int) -> list[int]:
    indices = []
    next_index = 0
    strip = max(0, case.rerun_start)
    while (
        case.rerun_frame_limit is None or len(indices) < case.rerun_frame_limit
    ):
        frame_index = next_index + strip
        if frame_index >= frame_count:
            break
        indices.append(frame_index)
        next_index = frame_index + 1
        strip = max(0, case.rerun_strip)
    return indices


def _compare_rerun_selection(case: AbCase, run: AbRun) -> dict[str, object]:
    input_trajectory = run.bundled_dir / case.trajectory_file_name
    stream_root = f"/particles/{case.trajectory_particle_stream}"
    input_steps = _h5_numeric_values(input_trajectory, f"{stream_root}/step")
    expected_indices = _expected_rerun_frame_indices(case, len(input_steps))
    expected_frames = [
        float(index) for index in range(1, len(expected_indices) + 1)
    ]

    branch_frames = {}
    for branch, case_dir in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        mdout = _read_mdout(case_dir / "mdout.txt")
        if "frame" not in mdout["columns"]:
            raise AssertionError(f"{case.name} {branch} mdout is missing frame")
        actual_frames = [row["frame"] for row in mdout["rows"]]
        _assert_numeric_sequences_close(
            f"{case.name} {branch} selected rerun frames",
            expected_frames,
            actual_frames,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
        branch_frames[branch] = actual_frames

    legacy_stdout = (run.legacy_dir / "run.stdout").read_text(encoding="utf-8")
    bundled_stdout = (run.bundled_dir / "run.stdout").read_text(
        encoding="utf-8"
    )
    for token in (
        "Open rerun coordinate trajectory 'traj.dat'",
        "Open rerun box trajectory 'traj_box.dat'",
    ):
        if token not in legacy_stdout:
            raise AssertionError(
                f"{case.name} legacy route is missing {token!r}"
            )
    velocity_token = "Open rerun velocity trajectory 'traj_vel.dat'"
    if (velocity_token in legacy_stdout) != case.rerun_velocity_present:
        raise AssertionError(
            f"{case.name} legacy velocity route does not match the case"
        )
    h5_open_token = f"Open rerun H5MD trajectory '{case.trajectory_file_name}'"
    if h5_open_token not in bundled_stdout:
        raise AssertionError(
            f"{case.name} bundled route is missing {h5_open_token!r}"
        )
    velocity_path = f"{stream_root}/velocity/value"
    bundled_has_velocity = velocity_path in _h5_paths(input_trajectory)
    if bundled_has_velocity != case.rerun_velocity_present:
        raise AssertionError(
            f"{case.name} bundled velocity payload does not match the case"
        )

    return {
        "method": "input_step_selection_and_module_owned_mdout_frames",
        "start": case.rerun_start,
        "strip": case.rerun_strip,
        "frame_limit": case.rerun_frame_limit,
        "input_frame_count": len(input_steps),
        "selected_indices": expected_indices,
        "selected_frames": expected_frames,
        "branch_frames": branch_frames,
        "box_update": case.rerun_need_box_update,
        "velocity_present": case.rerun_velocity_present,
        "particle_stream": case.trajectory_particle_stream,
    }


def _compare_input_semantics(
    case: AbCase, runs: Sequence[AbRun]
) -> list[dict[str, object]]:
    specs = _input_semantic_specs(case)
    results = []
    evolution_by_replica: dict[int, dict[str, object]] = {}
    for spec in specs:
        replica_results = []
        for run in runs:
            legacy = _read_mdout(run.legacy_dir / "mdout.txt")
            bundled = _read_mdout(run.bundled_dir / "mdout.txt")
            replica_result = assert_module_semantics(
                f"{case.name} replica {run.replica_index} {spec.contract_id}",
                legacy["rows"],
                bundled["rows"],
                spec,
                deterministic=not case.statistical_md,
            )
            if case.name == "normal_core_topology_payload_sensitivity":
                replica_result["oracle"] = (
                    _compare_focused_core_topology_sensitivity(
                        case, run, spec.contract_id
                    )
                )
            elif case.fixture_case in FOCUSED_BONDED_MODULES:
                replica_result["oracle"] = (
                    _compare_focused_bonded_payload_sensitivity(
                        case, run, spec.contract_id
                    )
                )
            elif spec.contract_id in {
                "input.manybody.tersoff",
                "input.manybody.tersoff.sidecar",
            }:
                replica_result["oracle"] = _compare_focused_tersoff_angular(
                    case, run
                )
            elif spec.contract_id in {
                "input.manybody.sw",
                "input.manybody.sw.sidecar",
            }:
                replica_result["oracle"] = _compare_focused_sw_pair_three_body(
                    case, run
                )
            elif spec.contract_id == "input.manybody.edip":
                replica_result["oracle"] = _compare_focused_edip_behavior(
                    case, run
                )
            elif (
                spec.contract_id == "input.manybody.reaxff"
                and case.fixture_case == FOCUSED_REAXFF_FIXTURE
            ):
                replica_result["oracle"] = _compare_focused_reaxff_behavior(
                    case, run
                )
            elif spec.contract_id == "input.manybody.eam" and case.name in {
                "normal_eam_funcfl_nonzero",
                "normal_eam_setfl_nonzero",
            }:
                replica_result["oracle"] = _compare_focused_eam_behavior(
                    case, run
                )
            elif spec.contract_id in {
                "input.protocol.positional_restraint",
                "input.protocol.positional_restraint.sidecar",
            } and case.fixture_case in {
                FOCUSED_POSITIONAL_RESTRAINT_FIXTURE,
                FOCUSED_POSITIONAL_RESTRAINT_SIDECAR_FIXTURE,
            }:
                replica_result["oracle"] = (
                    _compare_focused_positional_restraint_behavior(case, run)
                )
            elif spec.contract_id in {
                "input.protocol.soft_wall",
                "input.protocol.soft_wall.sidecar",
            } and case.fixture_case in {
                FOCUSED_SOFT_WALL_FIXTURE,
                FOCUSED_SOFT_WALL_SIDECAR_FIXTURE,
            }:
                replica_result["oracle"] = _compare_focused_soft_wall_behavior(
                    case, run
                )
            elif spec.contract_id in {
                "input.protocol.cv_restraint",
                "input.protocol.cv_restraint.sidecar",
            } and case.fixture_case in {
                FOCUSED_CV_RESTRAINT_TYPED_FIXTURE,
                FOCUSED_CV_RESTRAINT_SIDECAR_FIXTURE,
            }:
                replica_result["oracle"] = (
                    _compare_focused_cv_restraint_behavior(case, run)
                )
            elif spec.contract_id == "input.custom.pairwise":
                replica_result["force"] = _compare_focused_custom_pair_forces(
                    case, run
                )
            elif spec.contract_id == "input.topology.exclusions":
                replica_result["oracle"] = _compare_focused_exclusions_oracle(
                    case, run
                )
            elif spec.contract_id in {
                "input.topology.residue",
                "input.topology.residue.sidecar",
            }:
                if case.name in {
                    "normal_residue_sidecar_pbc_mapping",
                    "normal_residue_typed_pbc_mapping",
                }:
                    replica_result["oracle"] = (
                        _compare_focused_residue_pbc_mapping(case, run)
                    )
                else:
                    replica_result["oracle"] = (
                        _compare_focused_residue_com_res_virial(case, run)
                    )
            elif spec.contract_id in {
                "input.topology.gb",
                "input.topology.gb.hybrid_activation",
            }:
                replica_result["force"] = _compare_focused_gb_forces(case, run)
            elif spec.contract_id in {
                "input.topology.improper",
                "input.topology.improper.native_runtime",
            }:
                replica_result["force"] = _compare_focused_improper_forces(
                    case, run
                )
            elif spec.contract_id in {
                "input.protocol.steering",
                "input.protocol.steering.cv_sidecar",
            }:
                replica_result["oracle"] = _compare_focused_steering_cv(
                    case, run
                )
            elif (
                spec.contract_id == "input.protocol.cv"
                and case.fixture_case == FOCUSED_STEERING_CV_TYPED_FIXTURE
            ):
                replica_result["oracle"] = (
                    _compare_focused_cv_definition_payload(case, run)
                )
            elif spec.contract_id == "input.protocol.sits.nk_typed_restart":
                replica_result["oracle"] = (
                    _compare_focused_sits_nk_typed_restart(case, run)
                )
            elif spec.contract_id == "input.protocol.sits":
                replica_result["oracle"] = _compare_focused_sits_typed_config(
                    case, run
                )
            elif spec.contract_id == "input.topology.lj_soft_core":
                replica_result["force"] = _compare_focused_lj_soft_core_forces(
                    case, run
                )
            elif spec.contract_id == "input.topology.subsystem_division":
                replica_result["oracle"] = _compare_focused_subsystem_partition(
                    case, run
                )
            elif spec.contract_id in {
                "input.topology.virtual_atoms",
                "input.topology.virtual_atoms_pbc",
            }:
                replica_result["oracle"] = (
                    _compare_focused_virtual_atoms_oracle(case, run)
                )
            if spec.contract_id in case.evolution_contract_ids:
                if run.replica_index not in evolution_by_replica:
                    evolution_by_replica[run.replica_index] = (
                        _compare_nonzero_dt_input_evolution(case, run)
                    )
                replica_result["nonzero_dt_evolution"] = evolution_by_replica[
                    run.replica_index
                ]
            replica_results.append(replica_result)
        results.append(
            {
                "contract_id": spec.contract_id,
                "observables": list(spec.observables),
                "replicas": replica_results,
                "cross_branch_comparison": (
                    "mdout_statistical_equivalence"
                    if case.statistical_md
                    else "module_owned_deterministic_rows"
                ),
            }
        )
    return results


def _build_input_contract_oracles(
    input_semantics: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Scope the shared input assertion to one executable oracle per contract."""

    contract_oracles = {}
    for result in input_semantics:
        contract_id = result.get("contract_id")
        observables = result.get("observables")
        replicas = result.get("replicas")
        if not isinstance(contract_id, str) or not contract_id:
            raise AssertionError("input semantic result has no contract ID")
        if not isinstance(observables, list) or not observables:
            raise AssertionError(
                f"{contract_id} input semantic result has no observables"
            )
        if not isinstance(replicas, list) or not replicas:
            raise AssertionError(
                f"{contract_id} input semantic result has no replicas"
            )
        focused_controls = []
        for replica_index, replica in enumerate(replicas):
            if not isinstance(replica, Mapping):
                raise AssertionError(
                    f"{contract_id} input semantic replica is not an object"
                )
            for control_kind in ("oracle", "force", "nonzero_dt_evolution"):
                control = replica.get(control_kind)
                if isinstance(control, Mapping) and control:
                    focused_controls.append(
                        {
                            "replica_index": replica_index,
                            "control_kind": control_kind,
                            "result": dict(control),
                        }
                    )
        contract_oracles[contract_id] = {
            "oracle_contract_id": contract_id,
            "control_mutation": {
                "kind": "legacy_surface_to_bundled_h5_substitution",
                "changed_variable": contract_id,
                "focused_payload_controls": focused_controls,
            },
            "expected_delta": {
                "cross_branch": "within_configured_tolerance",
                "module_owned_payload": "present_and_nontrivial",
                "configured_tolerances": [
                    {
                        "relative": replica.get("relative_tolerance"),
                        "absolute": replica.get("absolute_tolerance"),
                    }
                    for replica in replicas
                ],
            },
            "actual_delta": {
                "cross_branch": result.get("cross_branch_comparison"),
                "replica_count": len(replicas),
                "observable_deltas": [
                    replica.get("observable_deltas") for replica in replicas
                ],
                "legacy_nontrivial": all(
                    replica.get("legacy_nontrivial") is True
                    for replica in replicas
                ),
                "bundled_nontrivial": all(
                    replica.get("bundled_nontrivial") is True
                    for replica in replicas
                ),
                "focused_payload_control_count": len(focused_controls),
            },
            "compared_payload": list(observables),
        }
    if len(contract_oracles) != len(input_semantics):
        raise AssertionError(
            "input semantic results contain duplicate contracts"
        )
    return contract_oracles


def _compare_nonzero_dt_input_evolution(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    evolution_case = replace(
        case,
        normal_step_limit=3,
        normal_interval=1,
        normal_dt=case.evolution_dt,
    )
    directories = {}
    for branch, source in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        target = source.parent / f"{branch}_nonzero_dt_evolution"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        output = target / "output"
        if output.exists():
            shutil.rmtree(output)
        for materialized in target.glob(".sponge_h5_*"):
            if materialized.is_dir():
                shutil.rmtree(materialized)
            else:
                materialized.unlink()
        for file_name in (
            "mdout.txt",
            "mdinfo.txt",
            "run.stdout",
            "run.stderr",
        ):
            path = target / file_name
            if path.exists():
                path.unlink()
        _prepare_mdin(
            target,
            _mdin_name(target),
            evolution_case,
            branch=branch,
            replica_seed=run.replica_seed,
        )
        _run_sponge(target, _mdin_name(target))
        directories[branch] = target

    payloads = {
        branch: _read_nonzero_dt_evolution_payload(directory)
        for branch, directory in directories.items()
    }
    result = _assert_nonzero_dt_evolution(
        case.name,
        payloads["legacy"],
        payloads["bundled"],
        dt=case.evolution_dt,
    )
    result.update(
        {
            "cohort": case.evolution_cohort,
            "contract_ids": list(case.evolution_contract_ids),
            "step_limit": 3,
        }
    )
    for directory in directories.values():
        shutil.rmtree(directory)
    return result


def _read_nonzero_dt_evolution_payload(case_dir: Path) -> dict[str, object]:
    trajectory = case_dir / TRAJECTORY_REL
    mdout = _read_mdout(case_dir / "mdout.txt")
    return {
        "mdout": mdout,
        "step": _h5_numeric_values(trajectory, "/particles/all/step"),
        "time": _h5_numeric_values(trajectory, "/particles/all/time"),
        "position": _h5_numeric_values(
            trajectory, "/particles/all/position/value"
        ),
        "velocity": _h5_numeric_values(
            trajectory, "/particles/all/velocity/value"
        ),
        "force": _h5_numeric_values(trajectory, "/particles/all/force/value"),
        "box": _h5_numeric_values(trajectory, "/particles/all/box/edges/value"),
    }


def _assert_nonzero_dt_evolution(
    label: str,
    legacy: dict[str, object],
    bundled: dict[str, object],
    *,
    dt: float,
) -> dict[str, object]:
    if not math.isfinite(dt) or dt <= 0.0:
        raise AssertionError(
            f"{label} evolution dt must be positive and finite"
        )
    expected_steps = [1.0, 2.0, 3.0]
    expected_times = [0.0, dt, 2.0 * dt]
    for branch, payload in (("legacy", legacy), ("bundled", bundled)):
        steps = payload["step"]
        times = payload["time"]
        if steps != expected_steps:
            raise AssertionError(
                f"{label} {branch} evolution schedule changed: {steps}"
            )
        _assert_numeric_sequences_close(
            f"{label} {branch} evolution time schedule",
            expected_times,
            times,
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-12,
        )
        mdout = payload["mdout"]
        if len(mdout["rows"]) != 3:
            raise AssertionError(
                f"{label} {branch} evolution emitted "
                f"{len(mdout['rows'])} mdout rows instead of 3"
            )
        for field in ("position", "velocity", "force", "box"):
            values = payload[field]
            if not values or not all(math.isfinite(value) for value in values):
                raise AssertionError(
                    f"{label} {branch} evolution {field} is empty or non-finite"
                )

    columns = _require_matching_mdout_columns(
        legacy["mdout"], bundled["mdout"], f"{label} evolution"
    )
    for column in columns:
        legacy_values = [row[column] for row in legacy["mdout"]["rows"]]
        bundled_values = [row[column] for row in bundled["mdout"]["rows"]]
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            column
        )
        _assert_numeric_sequences_close(
            f"{label} evolution mdout {column}",
            legacy_values,
            bundled_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    for field in ("step", "time", "position", "velocity", "force", "box"):
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(field)
        _assert_numeric_sequences_close(
            f"{label} evolution {field}",
            legacy[field],
            bundled[field],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )

    positions = bundled["position"]
    if len(positions) % 3 != 0:
        raise AssertionError(f"{label} evolution position payload is malformed")
    values_per_frame = len(positions) // 3
    if values_per_frame == 0:
        raise AssertionError(f"{label} evolution position payload is empty")
    maximum_coordinate_delta = max(
        abs(final - initial)
        for initial, final in zip(
            positions[:values_per_frame],
            positions[-values_per_frame:],
            strict=True,
        )
    )
    if maximum_coordinate_delta <= 1.0e-8:
        raise AssertionError(f"{label} evolution coordinates are frozen")
    return {
        "frame_count": 3,
        "steps": expected_steps,
        "times": expected_times,
        "maximum_coordinate_delta": maximum_coordinate_delta,
        "maximum_abs_recorded_force": max(
            abs(value) for value in bundled["force"]
        ),
        "compared_particle_fields": ["position", "velocity", "force", "box"],
        "mdout_column_count": len(columns),
    }


def _compare_focused_sits_nk_typed_restart(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_sits_nk_typed_restart_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} SITS force", forces["legacy"], forces["bundled"]
    )
    return {
        "route": "isolated_h5_typed_SITS_nk_restart_state",
        "branches": branch_results,
        "cross_branch_force": cross_branch_force,
    }


def _compare_focused_sits_typed_config(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    baseline = _compare_focused_sits_nk_typed_restart(case, run)
    materialized_root = run.bundled_dir / ".sponge_h5_native_protocol"
    expected_config = (
        "SITS\n"
        "{\n"
        "    mode = production\n"
        "    k_numbers = 2\n"
        "    T = 300/600\n"
        "    nk_rest = false\n"
        "    nk_fix = true\n"
        "    pe_a = 1.0\n"
        "    pe_b = 0.0\n"
        "    fb_bias = 0.0\n"
        "    fb_interval = 1\n"
        "}\n"
    )
    config_path = materialized_root / "sits.txt"
    atom_path = materialized_root / "sits_atom.txt"
    if config_path.read_text(encoding="utf-8") != expected_config:
        raise AssertionError(f"{case.name} materialized SITS config changed")
    if atom_path.read_text(encoding="utf-8") != "0\n1\n":
        raise AssertionError(f"{case.name} materialized SITS atom list changed")

    baseline_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    baseline_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    control_results: dict[str, object] = {}
    for mutation in (
        "pe_a_half",
        "single_selected_atom",
        "explicit_config_precedence",
    ):
        control_dir = (
            run.bundled_dir.parent / f"bundled_sits_{mutation}_control"
        )
        if control_dir.exists():
            shutil.rmtree(control_dir)
        shutil.copytree(run.bundled_dir, control_dir)
        for path in (
            control_dir / "output",
            control_dir / ".sponge_h5_native_protocol",
        ):
            if path.exists():
                shutil.rmtree(path)
        (control_dir / "output").mkdir(parents=True)
        for file_name in (
            "mdout.txt",
            "mdinfo.txt",
            "run.stdout",
            "run.stderr",
        ):
            path = control_dir / file_name
            if path.exists():
                path.unlink()

        protocol_path = control_dir / "protocol.spgp.h5"
        if mutation == "explicit_config_precedence":
            mdin_path = control_dir / _mdin_name(control_dir)
            mdin = _insert_root_toml_keys(
                mdin_path.read_text(encoding="utf-8"),
                [
                    'SITS_mode = "production"',
                    "SITS_k_numbers = 2",
                    'SITS_T = "300/600"',
                    "SITS_nk_rest = false",
                    "SITS_nk_fix = true",
                    "SITS_pe_a = 0.5",
                    "SITS_pe_b = 0.0",
                    "SITS_fb_bias = 0.0",
                    "SITS_fb_interval = 1",
                ],
            )
            mdin_path.write_text(mdin, encoding="utf-8")
        else:
            with h5py.File(protocol_path, "r+") as protocol:
                if mutation == "pe_a_half":
                    keys = protocol["/sits/config/key"].asstr()[...].tolist()
                    pe_a_index = keys.index("pe_a")
                    protocol["/sits/config/value"][pe_a_index] = "0.5"
                else:
                    del protocol["/sits/atom_indices"]
                    protocol["/sits"].create_dataset(
                        "atom_indices", data=[0], dtype="i4"
                    )

        outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
        if outcome.returncode != 0:
            raise AssertionError(
                f"{case.name} {mutation} control failed with "
                f"code {outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
            )
        control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
        control_force = _read_native_float32_file(
            control_dir / "output" / "legacy.frc"
        )
        if len(control_rows) != len(baseline_rows) or len(control_rows) != 1:
            raise AssertionError(
                f"{case.name} {mutation} control changed mdout shape"
            )
        observables = ("SITS_AA_kAB", "SITS_bias", "SITS_fb", "eff_pot")
        observable_deltas = {
            observable: abs(
                float(control_rows[0].get(observable, math.nan))
                - float(baseline_rows[0].get(observable, math.nan))
            )
            for observable in observables
        }
        if not all(
            math.isfinite(value) for value in observable_deltas.values()
        ):
            raise AssertionError(
                f"{case.name} {mutation} control produced non-finite evidence"
            )
        if len(control_force) != len(baseline_force) or not all(
            math.isfinite(value) for value in control_force
        ):
            raise AssertionError(
                f"{case.name} {mutation} control produced invalid force"
            )
        maximum_force_delta = max(
            abs(control - reference)
            for control, reference in zip(
                control_force, baseline_force, strict=True
            )
        )
        _assert_sits_typed_control_response(
            case.name, mutation, control_rows[0], maximum_force_delta
        )
        if mutation == "explicit_config_precedence":
            if (
                control_dir / ".sponge_h5_native_protocol" / "sits.txt"
            ).exists():
                raise AssertionError(
                    f"{case.name} typed config overrode explicit SITS config"
                )
            if not (
                control_dir / ".sponge_h5_native_protocol" / "sits_atom.txt"
            ).exists():
                raise AssertionError(
                    f"{case.name} explicit config suppressed typed atom indices"
                )
        control_results[mutation] = {
            "observables": {
                key: float(control_rows[0][key]) for key in observables
            },
            "observable_deltas": observable_deltas,
            "maximum_force_delta": maximum_force_delta,
        }
        shutil.rmtree(control_dir)

    failure_results = {}
    for mutation, diagnostic in (
        ("duplicate_atom", "/sits/atom_indices row 1 duplicates atom 0"),
        ("duplicate_config_key", "/sits/config/key contains a duplicate key"),
    ):
        failure_dir = (
            run.bundled_dir.parent / f"bundled_sits_{mutation}_failure"
        )
        if failure_dir.exists():
            shutil.rmtree(failure_dir)
        shutil.copytree(run.bundled_dir, failure_dir)
        for path in (
            failure_dir / "output",
            failure_dir / ".sponge_h5_native_protocol",
        ):
            if path.exists():
                shutil.rmtree(path)
        (failure_dir / "output").mkdir(parents=True)
        with h5py.File(failure_dir / "protocol.spgp.h5", "r+") as protocol:
            if mutation == "duplicate_atom":
                protocol["/sits/atom_indices"][...] = (0, 0)
            else:
                protocol["/sits/config/key"][1] = "mode"
        failure = _run_sponge_process(failure_dir, _mdin_name(failure_dir))
        failure_text = failure.stdout + failure.stderr
        if failure.returncode == 0 or not all(
            token in failure_text
            for token in (
                "spongeErrorBadFileFormat",
                "Materialize_H5_SITS_Input",
                diagnostic,
            )
        ):
            raise AssertionError(
                f"{case.name} {mutation} control was not rejected\n"
                f"{failure.stdout}\n{failure.stderr}"
            )
        failure_results[mutation] = {
            "exit_code": failure.returncode,
            "failure_category": "spongeErrorBadFileFormat",
            "diagnostic": diagnostic,
        }
        shutil.rmtree(failure_dir)

    return {
        "route": "pure_typed_SITS_config_atoms_and_nk_restart",
        "baseline": baseline,
        "materialized_config": str(config_path.relative_to(run.bundled_dir)),
        "materialized_atom_list": str(atom_path.relative_to(run.bundled_dir)),
        "controls": control_results,
        "invalid_payloads_rejected": failure_results,
    }


def _assert_sits_typed_control_response(
    label: str,
    mutation: str,
    row: dict[str, float],
    maximum_force_delta: float,
) -> dict[str, float]:
    expected_by_mutation = {
        "pe_a_half": {"SITS_bias": -1.4591, "SITS_fb": 0.6471},
        "explicit_config_precedence": {
            "SITS_bias": -1.4591,
            "SITS_fb": 0.6471,
        },
        "single_selected_atom": {
            "SITS_AA_kAB": -0.61,
            "SITS_bias": -0.7295,
            "SITS_fb": 0.6471,
        },
    }
    if mutation not in expected_by_mutation:
        raise AssertionError(
            f"{label} has unknown typed SITS mutation {mutation}"
        )
    for observable, expected in expected_by_mutation[mutation].items():
        _assert_numeric_sequences_close(
            f"{label} {mutation} {observable} oracle",
            (expected,),
            (float(row.get(observable, math.nan)),),
            relative_tolerance=0.0,
            absolute_tolerance=1.5e-4,
        )
    minimum_force_delta = (
        0.01
        if mutation in {"pe_a_half", "explicit_config_precedence"}
        else 0.02
    )
    if not math.isfinite(maximum_force_delta) or (
        maximum_force_delta <= minimum_force_delta
    ):
        raise AssertionError(
            f"{label} {mutation} force response is too small: "
            f"{maximum_force_delta}"
        )
    return {key: float(row[key]) for key in expected_by_mutation[mutation]} | {
        "maximum_force_delta": maximum_force_delta
    }


def _assert_sits_nk_typed_restart_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    enhancing_energy = -1.22
    expected_bias = -0.5317
    expected_factor = 0.7049
    expected_potential = -1.28
    expected_effective_potential = -1.2829471
    expected_force = (
        0.1828715056180954,
        1.096548518653151e-09,
        -1.2317579178855453e-09,
        -0.2489061951637268,
        -7.1972111603813e-10,
        1.3064306303434137e-09,
    )
    control_bias = -0.1833
    control_factor = 0.8677
    control_force_x = 0.21928882598876953

    module_energy = [row["SITS_AA_kAB"] for row in rows if "SITS_AA_kAB" in row]
    bias = [row["SITS_bias"] for row in rows if "SITS_bias" in row]
    factor = [row["SITS_fb"] for row in rows if "SITS_fb" in row]
    lj_short = [row["LJ_short"] for row in rows if "LJ_short" in row]
    lj = [row["LJ"] for row in rows if "LJ" in row]
    particle_mesh = [row["PM"] for row in rows if "PM" in row]
    potential = [row["potential"] for row in rows if "potential" in row]
    effective_potential = [row["eff_pot"] for row in rows if "eff_pot" in row]
    for observable, expected, values, tolerance in (
        ("SITS_AA_kAB", enhancing_energy, module_energy, 1.0e-6),
        ("SITS_bias", expected_bias, bias, 1.5e-4),
        ("SITS_fb", expected_factor, factor, 1.5e-4),
        ("LJ_short", -1.0, lj_short, 1.0e-6),
        ("LJ", -1.0, lj, 1.0e-6),
        ("PM", -0.5, particle_mesh, 1.0e-6),
        ("potential", expected_potential, potential, 1.1e-2),
        (
            "eff_pot",
            expected_effective_potential,
            effective_potential,
            1.0e-5,
        ),
    ):
        _assert_numeric_sequences_close(
            f"{label} {observable} oracle",
            (expected,),
            values,
            relative_tolerance=0.0,
            absolute_tolerance=tolerance,
        )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} SITS force oracle",
        expected_force,
        forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if abs(bias[0] - control_bias) <= 0.3:
        raise AssertionError(f"{label} SITS bias does not distinguish typed Nk")
    if abs(factor[0] - control_factor) <= 0.1:
        raise AssertionError(
            f"{label} SITS force factor does not distinguish typed Nk"
        )
    if abs(forces[0] - control_force_x) <= 0.03:
        raise AssertionError(
            f"{label} SITS force does not distinguish typed Nk"
        )
    return {
        "enhancing_energy_at_update": enhancing_energy,
        "nk": [1.0, 4.0],
        "temperatures": [300.0, 600.0],
        "bias": bias[0],
        "force_factor": factor[0],
        "potential": potential[0],
        "effective_potential": effective_potential[0],
        "maximum_abs_force": max(abs(value) for value in forces),
        "initialization_only_rejected": True,
    }


def _compare_focused_steering_cv(case: AbCase, run: AbRun) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_steering_cv_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} steering force", forces["legacy"], forces["bundled"]
    )
    result = {
        "route": (
            "typed_h5_cv_config"
            if case.fixture_case == FOCUSED_STEERING_CV_TYPED_FIXTURE
            else "isolated_h5_cv_in_file_protocol_sidecar"
        ),
        "branches": branch_results,
        "cross_branch_force": cross_branch_force,
    }
    if case.fixture_case == FOCUSED_STEERING_CV_TYPED_FIXTURE:
        materialized = run.bundled_dir / ".sponge_h5_native_protocol" / "cv.txt"
        if not materialized.exists() or materialized.stat().st_size == 0:
            raise AssertionError(
                f"{case.name} did not materialize bundled CV payload"
            )
        result["bundled_materialized_path"] = str(
            materialized.relative_to(run.bundled_dir)
        )
        result["weight_zero_control"] = _run_typed_steering_weight_zero_control(
            case, run, forces["bundled"]
        )
    return result


def _run_typed_steering_weight_zero_control(
    case: AbCase, run: AbRun, correct_forces: Sequence[float]
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_steering_weight_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    protocol_path = control_dir / "protocol.spgp.h5"
    with h5py.File(protocol_path, "r+") as protocol:
        keys = protocol["/cv/config/key"].asstr()[...].tolist()
        values = protocol["/cv/config/value"].asstr()[...].tolist()
        if keys != ["CV", "weight", "CV_type", "atom"] or values != [
            "distance",
            "2.0",
            "distance",
            "0 1",
        ]:
            raise AssertionError(
                f"{case.name} weight control lost canonical CV payload"
            )
        protocol["/cv/config/value"][1] = "0.0"

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} weight=0 control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} weight=0 control expected one mdout row"
        )
    zero_observables = tuple(
        rows[0].get(name, math.nan)
        for name in ("steer_cv", "potential", "eff_pot")
    )
    _assert_numeric_sequences_close(
        f"{case.name} weight=0 energy",
        (0.0, 0.0, 0.0),
        zero_observables,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-8,
    )
    zero_forces = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    _assert_numeric_sequences_close(
        f"{case.name} weight=0 force",
        (0.0,) * 6,
        zero_forces,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-8,
    )
    maximum_force_delta = max(
        abs(full - control)
        for full, control in zip(correct_forces, zero_forces, strict=True)
    )
    if maximum_force_delta <= 1.0:
        raise AssertionError(
            f"{case.name} typed weight did not change steering force"
        )
    result = {
        "typed_weight": 0.0,
        "steering_energy": zero_observables[0],
        "maximum_abs_force": max(abs(value) for value in zero_forces),
        "maximum_force_delta": maximum_force_delta,
        "exit_code": outcome.returncode,
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_cv_definition_payload(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    if case.fixture_case != FOCUSED_STEERING_CV_TYPED_FIXTURE:
        raise AssertionError(f"{case.name} CV definition oracle is not typed")
    baseline_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    baseline_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    baseline = _assert_steering_cv_oracle(
        f"{case.name} typed CV definition baseline",
        baseline_rows,
        baseline_force,
    )

    control_dir = run.bundled_dir.parent / "bundled_cv_position_y"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    mutation_path = "/cv/config/value"
    with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
        keys = protocol["/cv/config/key"].asstr()[...].tolist()
        values = protocol[mutation_path].asstr()[...].tolist()
        if keys != ["CV", "weight", "CV_type", "atom"] or values != [
            "distance",
            "2.0",
            "distance",
            "0 1",
        ]:
            raise AssertionError(
                f"{case.name} CV definition control lost canonical payload"
            )
        protocol[mutation_path][2] = "position_y"
        protocol[mutation_path][3] = "1"

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} position-y CV control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} CV definition control expected one row"
        )
    observables = tuple(
        rows[0].get(name, math.nan)
        for name in ("steer_cv", "potential", "eff_pot")
    )
    _assert_numeric_sequences_close(
        f"{case.name} position-y CV energy",
        (0.0, 0.0, 0.0),
        observables,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-8,
    )
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    expected_force = (0.0, 0.0, 0.0, 0.0, -2.0, 0.0)
    _assert_numeric_sequences_close(
        f"{case.name} position-y CV force",
        expected_force,
        control_force,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-6,
    )
    maximum_force_delta = max(
        abs(reference - mutated)
        for reference, mutated in zip(
            baseline_force, control_force, strict=True
        )
    )
    if maximum_force_delta <= 1.0:
        raise AssertionError(
            f"{case.name} CV definition did not change its consumer force"
        )
    result = {
        "route": "typed_h5_cv_config",
        "baseline": baseline,
        "control": {
            "mutation_path": mutation_path,
            "cv_type": ["distance", "position_y"],
            "atom": ["0 1", "1"],
            "steering_energy": observables[0],
            "force": control_force,
            "maximum_force_delta": maximum_force_delta,
        },
    }
    shutil.rmtree(control_dir)
    return result


def _assert_steering_cv_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    expected_energy = 3.0
    expected_force = (2.0, 0.0, 0.0, -2.0, 0.0, 0.0)
    steering_energy = [row["steer_cv"] for row in rows if "steer_cv" in row]
    total_potential = [row["potential"] for row in rows if "potential" in row]
    effective_potential = [row["eff_pot"] for row in rows if "eff_pot" in row]
    for observable, values in (
        ("steer_cv", steering_energy),
        ("potential", total_potential),
        ("eff_pot", effective_potential),
    ):
        _assert_numeric_sequences_close(
            f"{label} steering {observable} oracle",
            (expected_energy,),
            values,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-6,
        )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} steering force oracle",
        expected_force,
        forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    non_steering_maximum = max(
        abs(row.get(observable, 0.0))
        for row in rows
        for observable in ("PM", "temperature")
    )
    if non_steering_maximum > 1.0e-8:
        raise AssertionError(
            f"{label} isolated steering fixture has another non-zero result"
        )
    return {
        "distance": 1.5,
        "weight": 2.0,
        "steering_energy": steering_energy[0],
        "maximum_abs_force": max(abs(value) for value in forces),
        "weight_zero_energy": 0.0,
        "weight_zero_maximum_abs_force": 0.0,
        "isolated_module_owned_result": True,
    }


def _compare_focused_tersoff_angular(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_tersoff_angular_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} Tersoff force", forces["legacy"], forces["bundled"]
    )
    result = {
        "route": (
            "typed_h5_manybody_tersoff"
            if case.fixture_case == FOCUSED_TERSOFF_TYPED_FIXTURE
            else "isolated_h5_TERSOFF_in_file_sidecar"
        ),
        "module_owned_energy": "isolated_total_potential",
        "branches": branch_results,
        "cross_branch_force": cross_branch_force,
    }
    if case.fixture_case == FOCUSED_TERSOFF_TYPED_FIXTURE:
        materialized = (
            run.bundled_dir / ".sponge_h5_native_manybody" / "tersoff.txt"
        )
        if not materialized.exists() or materialized.stat().st_size == 0:
            raise AssertionError(
                f"{case.name} did not materialize bundled Tersoff payload"
            )
        result["bundled_materialized_path"] = str(
            materialized.relative_to(run.bundled_dir)
        )
        result["gamma_zero_control"] = _run_typed_tersoff_gamma_zero_control(
            case, run, forces["bundled"]
        )
        result["typed_validation_controls"] = (
            _run_typed_tersoff_validation_controls(case, run)
        )
    return result


def _run_typed_tersoff_gamma_zero_control(
    case: AbCase, run: AbRun, correct_forces: Sequence[float]
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_tersoff_gamma_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        raw_parameters = topology["/manybody/tersoff/entry/parameters_raw"]
        native_parameters = topology["/manybody/tersoff/entry/parameters"]
        if raw_parameters.shape != (1, 14) or native_parameters.shape != (
            1,
            18,
        ):
            raise AssertionError(
                f"{case.name} gamma control lost Tersoff parameter shape"
            )
        if not math.isclose(
            float(raw_parameters[0, 1]), 1.0, rel_tol=0.0, abs_tol=1.0e-7
        ) or not math.isclose(
            float(native_parameters[0, 1]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise AssertionError(
                f"{case.name} gamma control lost canonical gamma"
            )
        raw_parameters[0, 1] = 0.0
        native_parameters[0, 1] = 0.0

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} gamma=0 control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} gamma=0 control expected one mdout row"
        )
    gamma_zero_potential = rows[0].get("potential", math.nan)
    gamma_zero_effective_potential = rows[0].get("eff_pot", math.nan)
    _assert_numeric_sequences_close(
        f"{case.name} gamma=0 potential",
        (-196.06, -196.05984),
        (gamma_zero_potential, gamma_zero_effective_potential),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-6,
    )
    gamma_zero_forces = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    expected_gamma_zero_force = (
        144.78313,
        144.78313,
        0.0,
        -144.78313,
        0.0,
        0.0,
        0.0,
        -144.78313,
        0.0,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{case.name} gamma=0 force",
        expected_gamma_zero_force,
        gamma_zero_forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    maximum_force_delta = max(
        abs(full - control)
        for full, control in zip(correct_forces, gamma_zero_forces, strict=True)
    )
    if maximum_force_delta <= 1.0:
        raise AssertionError(
            f"{case.name} typed gamma did not change Tersoff force"
        )
    result = {
        "typed_gamma": 0.0,
        "potential": gamma_zero_potential,
        "effective_potential": gamma_zero_effective_potential,
        "maximum_abs_force": max(abs(value) for value in gamma_zero_forces),
        "maximum_force_delta": maximum_force_delta,
        "exit_code": outcome.returncode,
    }
    shutil.rmtree(control_dir)
    return result


def _run_typed_tersoff_validation_controls(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    controls = (
        (
            "map_mismatch",
            "/manybody/tersoff/map",
            "inconsistent with entry/type",
        ),
        (
            "parameter_mismatch",
            "/manybody/tersoff/entry/parameters",
            "inconsistent with parameters_raw",
        ),
    )
    results = {}
    for name, dataset_token, reason_token in controls:
        control_dir = run.bundled_dir.parent / f"bundled_tersoff_{name}"
        if control_dir.exists():
            shutil.rmtree(control_dir)
        shutil.copytree(run.bundled_dir, control_dir)
        for path in (
            control_dir / "output",
            control_dir / ".sponge_h5_native_manybody",
        ):
            if path.exists():
                shutil.rmtree(path)
        (control_dir / "output").mkdir()
        for file_name in (
            "mdout.txt",
            "mdinfo.txt",
            "run.stdout",
            "run.stderr",
        ):
            path = control_dir / file_name
            if path.exists():
                path.unlink()

        with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
            if name == "map_mismatch":
                topology["/manybody/tersoff/map"][0] = -1
            else:
                topology["/manybody/tersoff/entry/parameters"][0, 1] = 0.5

        outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
        combined = outcome.stdout + outcome.stderr
        category = _failure_category(combined)
        expected_tokens = (
            "Materialize_H5_Tersoff_Input",
            dataset_token,
            reason_token,
        )
        missing = [token for token in expected_tokens if token not in combined]
        if (
            outcome.returncode == 0
            or category != "spongeErrorBadFileFormat"
            or missing
        ):
            raise AssertionError(
                f"{case.name} {name} was not rejected: "
                f"code={outcome.returncode}, category={category}, "
                f"missing={missing}\n{combined}"
            )
        results[name] = {
            "exit_code": outcome.returncode,
            "failure_category": category,
            "diagnostic_tokens": list(expected_tokens),
        }
        shutil.rmtree(control_dir)
    return results


def _assert_tersoff_angular_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    expected_potential = -173.23
    expected_effective_potential = -173.23468
    gamma_zero_potential = -196.06
    expected_force = (
        135.94907,
        135.94907,
        0.0,
        -119.686844,
        -16.262218,
        0.0,
        -16.262218,
        -119.686844,
        0.0,
    )
    gamma_zero_force = (
        144.78313,
        144.78313,
        0.0,
        -144.78313,
        0.0,
        0.0,
        0.0,
        -144.78313,
        0.0,
    )
    potential = [row["potential"] for row in rows if "potential" in row]
    effective_potential = [row["eff_pot"] for row in rows if "eff_pot" in row]
    _assert_numeric_sequences_close(
        f"{label} angular Tersoff potential oracle",
        (expected_potential,),
        potential,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{label} angular Tersoff effective-potential oracle",
        (expected_effective_potential,),
        effective_potential,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-6,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} angular Tersoff force oracle",
        expected_force,
        forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    force_delta_from_gamma_zero = max(
        abs(actual - control)
        for actual, control in zip(forces, gamma_zero_force, strict=True)
    )
    if force_delta_from_gamma_zero < 1.0:
        raise AssertionError(
            f"{label} did not distinguish the gamma=0 Tersoff force"
        )
    non_tersoff_maximum = max(
        abs(row.get(observable, 0.0))
        for row in rows
        for observable in ("PM", "temperature")
    )
    if non_tersoff_maximum > 1.0e-8:
        raise AssertionError(
            f"{label} isolated Tersoff fixture has another non-zero result"
        )
    return {
        "potential": potential[0],
        "effective_potential": effective_potential[0],
        "gamma_zero_potential": gamma_zero_potential,
        "angular_energy_contribution": potential[0] - gamma_zero_potential,
        "maximum_abs_force": max(abs(value) for value in forces),
        "maximum_force_delta_from_gamma_zero": force_delta_from_gamma_zero,
        "three_body_parameter_gamma": 1.0,
        "isolated_module_owned_potential": True,
        "angular_bond_order_required": True,
    }


def _compare_focused_sw_pair_three_body(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_sw_pair_three_body_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} SW force", forces["legacy"], forces["bundled"]
    )
    result = {
        "route": (
            "typed_h5_manybody_sw"
            if case.fixture_case == FOCUSED_SW_TYPED_FIXTURE
            else "isolated_h5_SW_in_file_sidecar"
        ),
        "branches": branch_results,
        "cross_branch_force": cross_branch_force,
    }
    if case.fixture_case == FOCUSED_SW_TYPED_FIXTURE:
        materialized = run.bundled_dir / ".sponge_h5_native_manybody" / "sw.txt"
        if not materialized.exists() or materialized.stat().st_size == 0:
            raise AssertionError(
                f"{case.name} did not materialize bundled SW native payload"
            )
        result["bundled_materialized_path"] = str(
            materialized.relative_to(run.bundled_dir)
        )
        result["lambda_zero_control"] = _run_sw_typed_lambda_zero_control(
            case, run, forces["bundled"]
        )
    return result


def _run_sw_typed_lambda_zero_control(
    case: AbCase, run: AbRun, correct_forces: Sequence[float]
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_sw_lambda_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        triple_parameters = topology["/manybody/sw/triple/parameters"]
        if triple_parameters.shape != (1, 3):
            raise AssertionError(
                f"{case.name} typed SW control lost triple payload shape"
            )
        if not math.isclose(
            float(triple_parameters[0, 0]),
            32.5,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise AssertionError(
                f"{case.name} typed SW control lost canonical lambda"
            )
        triple_parameters[0, 0] = 0.0

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} lambda=0 control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} lambda=0 control expected one mdout row"
        )
    pair_only_energy = rows[0].get("SW", math.nan)
    if not math.isclose(pair_only_energy, 158.79, rel_tol=0.0, abs_tol=1.0e-6):
        raise AssertionError(
            f"{case.name} lambda=0 pair-only energy changed: {pair_only_energy}"
        )
    pair_only_forces = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    expected_pair_only_force = (
        -340.4832,
        -340.4832,
        0.0,
        343.52106,
        -3.037885,
        0.0,
        -3.037885,
        343.52106,
        0.0,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{case.name} lambda=0 pair-only force",
        expected_pair_only_force,
        pair_only_forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    maximum_force_delta = max(
        abs(full - pair_only)
        for full, pair_only in zip(
            correct_forces, pair_only_forces, strict=True
        )
    )
    if maximum_force_delta <= 1.0:
        raise AssertionError(
            f"{case.name} typed lambda payload did not change SW force"
        )
    result = {
        "typed_lambda": 0.0,
        "pair_only_energy": pair_only_energy,
        "maximum_abs_force": max(abs(value) for value in pair_only_forces),
        "maximum_force_delta": maximum_force_delta,
        "exit_code": outcome.returncode,
    }
    shutil.rmtree(control_dir)
    return result


def _assert_sw_pair_three_body_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    expected_energy = 194.50
    pair_only_energy = 158.79
    expected_force = (
        -352.62115,
        -352.62115,
        0.0,
        404.2786,
        -51.657455,
        0.0,
        -51.657455,
        404.2786,
        0.0,
    )
    pair_only_force = (
        -340.4832,
        -340.4832,
        0.0,
        343.52106,
        -3.037885,
        0.0,
        -3.037885,
        343.52106,
        0.0,
    )
    sw_energy = [row["SW"] for row in rows if "SW" in row]
    _assert_numeric_sequences_close(
        f"{label} pair+three-body SW energy oracle",
        (expected_energy,),
        sw_energy,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-6,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} pair+three-body SW force oracle",
        expected_force,
        forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    force_delta_from_pair_only = max(
        abs(actual - pair_only)
        for actual, pair_only in zip(forces, pair_only_force, strict=True)
    )
    if force_delta_from_pair_only < 1.0:
        raise AssertionError(
            f"{label} did not distinguish the lambda=0 pair-only force"
        )
    return {
        "energy": sw_energy[0],
        "pair_only_energy": pair_only_energy,
        "three_body_energy_contribution": sw_energy[0] - pair_only_energy,
        "maximum_abs_force": max(abs(value) for value in forces),
        "maximum_force_delta_from_pair_only": force_delta_from_pair_only,
        "three_body_parameter_lambda": 32.5,
        "pair_and_three_body_required": True,
    }


def _compare_focused_edip_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    materialized = run.bundled_dir / ".sponge_h5_native_manybody" / "edip.txt"
    if not materialized.exists() or materialized.stat().st_size == 0:
        raise AssertionError(
            f"{case.name} did not materialize bundled EDIP native payload"
        )
    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} EDIP force", legacy_force, bundled_force
    )
    baseline_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(baseline_rows) != 1:
        raise AssertionError(f"{case.name} expected one EDIP row")
    baseline_energy = baseline_rows[0].get("EDIP", math.nan)
    if not math.isfinite(baseline_energy) or abs(baseline_energy) <= 1.0e-8:
        raise AssertionError(f"{case.name} EDIP baseline energy is trivial")

    control_dir = run.bundled_dir.parent / "bundled_edip_pair_a_half"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    mutation_path = "/manybody/edip/pair/parameters"
    with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
        parameters = topology[mutation_path]
        original_pair_a = float(parameters[0, 3])
        if not math.isclose(original_pair_a, 4.0):
            raise AssertionError(
                f"{case.name} EDIP pair A changed: {original_pair_a}"
            )
        parameters[0, 3] = 0.5 * original_pair_a

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} EDIP pair-A control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1:
        raise AssertionError(f"{case.name} EDIP control expected one row")
    control_energy = control_rows[0].get("EDIP", math.nan)
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    _require_finite_values(case, "EDIP pair-A control force", control_force)
    energy_delta = abs(baseline_energy - control_energy)
    maximum_force_delta = max(
        abs(baseline - control)
        for baseline, control in zip(bundled_force, control_force, strict=True)
    )
    if (
        not math.isfinite(control_energy)
        or energy_delta <= 1.0e-3
        or maximum_force_delta <= 1.0e-3
    ):
        raise AssertionError(
            f"{case.name} EDIP pair A did not change energy and force"
        )
    result = {
        "energy": baseline_energy,
        "force": force_result,
        "bundled_materialized_path": str(
            materialized.relative_to(run.bundled_dir)
        ),
        "control": {
            "mutation_path": mutation_path,
            "pair_a": [original_pair_a, 0.5 * original_pair_a],
            "energy": control_energy,
            "energy_delta": energy_delta,
            "maximum_force_delta": maximum_force_delta,
        },
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_reaxff_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    materialized_root = run.bundled_dir / ".sponge_h5_native_manybody"
    materialized = (
        materialized_root / "reaxff.txt",
        materialized_root / "reaxff_type.txt",
    )
    missing = [
        path
        for path in materialized
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise AssertionError(
            f"{case.name} did not materialize ReaxFF payloads: {missing}"
        )
    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} ReaxFF force", legacy_force, bundled_force
    )
    baseline_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(baseline_rows) != 1:
        raise AssertionError(f"{case.name} expected one ReaxFF row")
    observables = ("REAXFF_BOND", "REAXFF_VDW", "REAXFF")
    baseline = {
        name: baseline_rows[0].get(name, math.nan) for name in observables
    }
    _require_finite_values(
        case, "ReaxFF baseline observables", baseline.values()
    )

    control_dir = run.bundled_dir.parent / "bundled_reaxff_oh_to_oo"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    mutation_path = "/manybody/reaxff/type/name"
    with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
        atom_types = topology[mutation_path]
        original_types = atom_types.asstr()[...].tolist()
        if original_types != ["O", "H"]:
            raise AssertionError(
                f"{case.name} ReaxFF atom types changed: {original_types}"
            )
        atom_types[1] = "O"

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} ReaxFF O-O control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1:
        raise AssertionError(f"{case.name} ReaxFF control expected one row")
    control = {
        name: control_rows[0].get(name, math.nan) for name in observables
    }
    _require_finite_values(case, "ReaxFF O-O observables", control.values())
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    _require_finite_values(case, "ReaxFF O-O force", control_force)
    observable_deltas = {
        name: abs(baseline[name] - control[name]) for name in observables
    }
    maximum_force_delta = max(
        abs(reference - mutated)
        for reference, mutated in zip(bundled_force, control_force, strict=True)
    )
    if (
        max(observable_deltas.values()) <= 1.0e-3
        or maximum_force_delta <= 1.0e-3
    ):
        raise AssertionError(
            f"{case.name} ReaxFF atom-type payload did not affect behavior"
        )
    result = {
        "observables": baseline,
        "force": force_result,
        "bundled_materialized_paths": [
            str(path.relative_to(run.bundled_dir)) for path in materialized
        ],
        "control": {
            "mutation_path": mutation_path,
            "atom_types": [original_types, ["O", "O"]],
            "observables": control,
            "observable_deltas": observable_deltas,
            "maximum_force_delta": maximum_force_delta,
        },
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_eam_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    expected = {
        FOCUSED_EAM_FUNCFL_FIXTURE: {
            "format": "funcfl",
            "energy": 267.33,
            "force": (-165.90681, 0.0, 0.0, 165.90681, 0.0, 0.0),
        },
        FOCUSED_EAM_SETFL_FIXTURE: {
            "format": "setfl",
            "energy": 61.49,
            "force": (-11.530274, 0.0, 0.0, 11.530274, 0.0, 0.0),
        },
    }.get(case.fixture_case)
    if expected is None:
        raise AssertionError(f"{case.name} has no focused EAM oracle")
    materialized_root = run.bundled_dir / ".sponge_h5_native_manybody"
    materialized = (
        materialized_root / "eam.txt",
        materialized_root / "eam_atom_type.txt",
    )
    missing = [
        path
        for path in materialized
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise AssertionError(
            f"{case.name} did not materialize bundled EAM payloads: {missing}"
        )
    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} EAM force", legacy_force, bundled_force
    )

    rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(rows) != 1 or not math.isfinite(rows[0].get("EAM", math.nan)):
        raise AssertionError(f"{case.name} did not emit one finite EAM result")
    if abs(rows[0]["EAM"]) <= 1.0e-3:
        raise AssertionError(f"{case.name} EAM result is trivial")
    _assert_numeric_sequences_close(
        f"{case.name} EAM energy oracle",
        (expected["energy"],),
        (rows[0]["EAM"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-2,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{case.name} EAM force oracle",
        expected["force"],
        bundled_force,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    control_dir = run.bundled_dir.parent / "bundled_eam_pair_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_manybody",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        format_name = topology["/manybody/eam/format"].asstr()[()]
        if format_name != expected["format"]:
            raise AssertionError(
                f"{case.name} EAM format changed: {format_name!r}"
            )
        if format_name == "funcfl":
            topology["/manybody/eam/funcfl/z"][...] = 0.0
        elif format_name == "setfl":
            topology["/manybody/eam/pair_potential/value"][...] = 0.0
        else:
            raise AssertionError(
                f"{case.name} has unsupported EAM format {format_name!r}"
            )

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} zero-pair control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1:
        raise AssertionError(
            f"{case.name} zero-pair control expected one mdout row"
        )
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    _assert_numeric_sequences_close(
        f"{case.name} zero-pair EAM energy oracle",
        (46.12,),
        (control_rows[0]["EAM"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-2,
    )
    _assert_numeric_sequences_close(
        f"{case.name} zero-pair EAM force oracle",
        (0.0,) * 6,
        control_force,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    maximum_force_delta = max(
        abs(full - zero)
        for full, zero in zip(bundled_force, control_force, strict=True)
    )
    energy_delta = abs(rows[0]["EAM"] - control_rows[0]["EAM"])
    if maximum_force_delta <= 1.0e-3 or energy_delta <= 1.0e-3:
        raise AssertionError(
            f"{case.name} EAM pair table did not affect energy and force"
        )
    result = {
        "format": format_name,
        "energy": rows[0]["EAM"],
        "zero_pair_energy": control_rows[0]["EAM"],
        "energy_delta": energy_delta,
        "maximum_force_delta": maximum_force_delta,
        "force": force_result,
        "bundled_materialized_paths": [
            str(path.relative_to(run.bundled_dir)) for path in materialized
        ],
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_positional_restraint_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    typed = case.fixture_case == FOCUSED_POSITIONAL_RESTRAINT_FIXTURE
    if (
        not typed
        and case.fixture_case != FOCUSED_POSITIONAL_RESTRAINT_SIDECAR_FIXTURE
    ):
        raise AssertionError(
            f"{case.name} has no focused positional-restraint oracle"
        )
    materialized_root = (
        run.bundled_dir / ".sponge_h5_native_protocol" / "restraint"
    )
    materialized = (
        materialized_root / "restrain_atom_id.txt",
        materialized_root / "restrain_weight.txt",
        materialized_root / "restrain_coordinate.txt",
    )
    if typed:
        missing = [
            path
            for path in materialized
            if not path.exists() or path.stat().st_size == 0
        ]
        if missing:
            raise AssertionError(
                f"{case.name} did not materialize positional restraint "
                f"payloads: {missing}"
            )
    elif any(path.exists() for path in materialized):
        raise AssertionError(
            f"{case.name} sidecar route materialized typed restraint input"
        )

    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} positional restraint force", legacy_force, bundled_force
    )
    rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(rows) != 1 or not math.isfinite(rows[0].get("restrain", math.nan)):
        raise AssertionError(
            f"{case.name} did not emit one finite restraint result"
        )
    _assert_numeric_sequences_close(
        f"{case.name} positional restraint energy oracle",
        (42.5,),
        (rows[0]["restrain"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{case.name} positional restraint force oracle",
        (-20.0, 0.0, 0.0, 0.0, -30.0, 0.0),
        bundled_force,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )

    control_results = {}
    control_names = ["swapped_atom_ids", "zero_weight", "matching_reference"]
    for control_name in control_names:
        control_results[control_name] = _run_positional_restraint_control(
            case, run, control_name, typed=typed
        )
        _assert_numeric_sequences_close(
            f"{case.name} {control_name} energy oracle",
            (0.0,),
            (control_results[control_name]["energy"],),
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-6,
        )
        _assert_numeric_sequences_close(
            f"{case.name} {control_name} force oracle",
            (0.0,) * 6,
            control_results[control_name]["force"],
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-6,
        )

    result = {
        "energy": rows[0]["restrain"],
        "force": force_result,
        "controls": control_results,
        "route": (
            "typed_h5" if typed else "isolated_protocol_and_restart_sidecars"
        ),
    }
    if typed:
        result["bundled_materialized_paths"] = [
            str(path.relative_to(run.bundled_dir)) for path in materialized
        ]
    else:
        result["bundled_sidecar_paths"] = [
            "legacy_sidecars/restrain_atom_id/restrain_atom_id.txt",
            "legacy_sidecars/restrain_weight_in_file/restrain_weight.txt",
            (
                "legacy_sidecars/restrain_coordinate_in_file/"
                "restrain_coordinate.txt"
            ),
        ]
    return result


def _run_positional_restraint_control(
    case: AbCase, run: AbRun, control_name: str, *, typed: bool
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / f"bundled_restraint_{control_name}"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    if typed:
        protocol_path = control_dir / "protocol.spgp.h5"
        restart_path = control_dir / "restart.spgr.h5"
        if control_name == "swapped_atom_ids":
            with h5py.File(protocol_path, "r+") as protocol:
                protocol["/restraint/default/atom_indices"][...] = [1, 0]
        elif control_name == "zero_weight":
            with h5py.File(protocol_path, "r+") as protocol:
                protocol["/restraint/default/weight"][...] = 0.0
        elif control_name == "matching_reference":
            reference_path = (
                "/parameters/restart/references/restraint/default/coordinate"
            )
            with h5py.File(restart_path, "r+") as restart:
                restart[reference_path][...] = [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0],
                ]
        else:
            raise AssertionError(
                f"unsupported positional restraint control: {control_name}"
            )
    else:
        sidecar_root = control_dir / "legacy_sidecars"
        if control_name == "swapped_atom_ids":
            path = sidecar_root / "restrain_atom_id" / "restrain_atom_id.txt"
            path.write_text("1\n0\n", encoding="utf-8")
        elif control_name == "zero_weight":
            path = (
                sidecar_root / "restrain_weight_in_file" / "restrain_weight.txt"
            )
            path.write_text("0.0 0.0 0.0\n0.0 0.0 0.0\n", encoding="utf-8")
        elif control_name == "matching_reference":
            reference_text = "2\n2.0 0.0 0.0\n0.0 1.5 0.0\n"
            path = (
                sidecar_root
                / "restrain_coordinate_in_file"
                / "restrain_coordinate.txt"
            )
            path.write_text(reference_text, encoding="utf-8")
            with h5py.File(control_dir / "restart.spgr.h5", "r+") as restart:
                restart[
                    "/parameters/restart/protocol_sidecars/"
                    "restrain_coordinate_in_file"
                ][()] = reference_text
        else:
            raise AssertionError(
                f"unsupported positional restraint control: {control_name}"
            )

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} {control_name} control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} {control_name} emitted {len(rows)} rows"
        )
    force = _read_native_float32_file(control_dir / "output" / "legacy.frc")
    result = {"energy": rows[0]["restrain"], "force": force}
    shutil.rmtree(control_dir)
    return result


def _compare_focused_soft_wall_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    typed = case.fixture_case == FOCUSED_SOFT_WALL_FIXTURE
    if not typed and case.fixture_case != FOCUSED_SOFT_WALL_SIDECAR_FIXTURE:
        raise AssertionError(f"{case.name} has no focused soft-wall oracle")
    materialized = (
        run.bundled_dir / ".sponge_h5_native_protocol" / "soft_walls.txt"
    )
    if typed:
        if not materialized.exists() or materialized.stat().st_size == 0:
            raise AssertionError(
                f"{case.name} did not materialize the bundled soft-wall payload"
            )
        materialized_text = materialized.read_text(encoding="utf-8")
        for token in (
            "[[[ z_wall ]]]",
            "[[ potential ]]",
            "E = (z - 1.0f) * (z - 1.0f);",
            "[[ end ]]",
        ):
            if token not in materialized_text:
                raise AssertionError(
                    f"{case.name} materialized soft wall is missing {token!r}"
                )
    elif materialized.exists():
        raise AssertionError(
            f"{case.name} sidecar route materialized typed soft-wall input"
        )

    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} soft-wall force", legacy_force, bundled_force
    )
    rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(rows) != 1 or not math.isfinite(rows[0].get("z_wall", math.nan)):
        raise AssertionError(f"{case.name} did not emit one finite soft wall")
    _assert_numeric_sequences_close(
        f"{case.name} soft-wall energy oracle",
        (10.0,),
        (rows[0]["z_wall"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{case.name} soft-wall force oracle",
        (0.0, 0.0, -2.0, 0.0, 0.0, -6.0),
        bundled_force,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )

    control = _run_soft_wall_zero_potential_control(case, run, typed=typed)
    _assert_numeric_sequences_close(
        f"{case.name} zero-potential energy oracle",
        (0.0,),
        (control["energy"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{case.name} zero-potential force oracle",
        (0.0,) * 6,
        control["force"],
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    result = {
        "energy": rows[0]["z_wall"],
        "force": force_result,
        "zero_potential_control": control,
        "route": "typed_h5" if typed else "isolated_h5_sidecar",
    }
    if typed:
        result["bundled_materialized_path"] = str(
            materialized.relative_to(run.bundled_dir)
        )
    else:
        result["bundled_sidecar_path"] = (
            "legacy_sidecars/soft_walls_in_file/soft_walls.txt"
        )
    return result


def _run_soft_wall_zero_potential_control(
    case: AbCase, run: AbRun, *, typed: bool
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_soft_wall_zero_potential"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    if typed:
        with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
            del protocol["/wall/soft/potential"]
            protocol.create_dataset(
                "/wall/soft/potential",
                data=["E = 0.0f * z;"],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
    else:
        sidecar = (
            control_dir
            / "legacy_sidecars"
            / "soft_walls_in_file"
            / "soft_walls.txt"
        )
        text = sidecar.read_text(encoding="utf-8")
        expected = "E = (z - 1.0f) * (z - 1.0f);"
        if expected not in text:
            raise AssertionError(
                f"{case.name} soft-wall control lost its sidecar potential"
            )
        sidecar.write_text(
            text.replace(expected, "E = 0.0f * z;"), encoding="utf-8"
        )
    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} zero-potential control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} zero-potential control emitted {len(rows)} rows"
        )
    result = {
        "energy": rows[0]["z_wall"],
        "force": _read_native_float32_file(
            control_dir / "output" / "legacy.frc"
        ),
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_cv_restraint_behavior(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    typed = case.fixture_case == FOCUSED_CV_RESTRAINT_TYPED_FIXTURE
    if not typed and case.fixture_case != FOCUSED_CV_RESTRAINT_SIDECAR_FIXTURE:
        raise AssertionError(f"{case.name} has no focused CV-restraint oracle")
    materialized = run.bundled_dir / ".sponge_h5_native_protocol" / "cv.txt"
    if typed:
        if not materialized.exists() or materialized.stat().st_size == 0:
            raise AssertionError(
                f"{case.name} did not materialize typed CV-restraint config"
            )
        materialized_text = materialized.read_text(encoding="utf-8")
        for token in (
            "distance",
            "CV_type = distance",
            "atom = 0 1",
            "restrain",
            "weight = 4.0",
            "reference = 1.5",
        ):
            if token not in materialized_text:
                raise AssertionError(
                    f"{case.name} materialized config is missing {token!r}"
                )
    elif materialized.exists():
        raise AssertionError(
            f"{case.name} sidecar route unexpectedly materialized typed config"
        )

    legacy_force = _read_native_float32_file(
        run.legacy_dir / "output" / "legacy.frc"
    )
    bundled_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_result = _assert_nontrivial_equivalent_forces(
        f"{case.name} CV-restraint force", legacy_force, bundled_force
    )
    rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(rows) != 1 or not math.isfinite(
        rows[0].get("restrain_cv", math.nan)
    ):
        raise AssertionError(
            f"{case.name} did not emit one finite CV-restraint result"
        )
    _assert_numeric_sequences_close(
        f"{case.name} CV-restraint energy oracle",
        (1.0,),
        (rows[0]["restrain_cv"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{case.name} CV-restraint force oracle",
        (4.0, 0.0, 0.0, -4.0, 0.0, 0.0),
        bundled_force,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )

    control = _run_cv_restraint_zero_weight_control(case, run, typed=typed)
    _assert_numeric_sequences_close(
        f"{case.name} zero-weight energy oracle",
        (0.0,),
        (control["energy"],),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    _assert_numeric_sequences_close(
        f"{case.name} zero-weight force oracle",
        (0.0,) * 6,
        control["force"],
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    result = {
        "energy": rows[0]["restrain_cv"],
        "force": force_result,
        "zero_weight_control": control,
        "route": "typed" if typed else "sidecar",
    }
    if typed:
        reordered = _run_cv_restraint_reordered_definition_control(case, run)
        _assert_numeric_sequences_close(
            f"{case.name} reordered-definition energy oracle",
            (1.0,),
            (reordered["energy"],),
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-6,
        )
        _assert_numeric_sequences_close(
            f"{case.name} reordered-definition force oracle",
            (4.0, 0.0, 0.0, -4.0, 0.0, 0.0),
            reordered["force"],
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-6,
        )
        result["reordered_definition_control"] = reordered
        result["bundled_materialized_path"] = str(
            materialized.relative_to(run.bundled_dir)
        )
    return result


def _run_cv_restraint_zero_weight_control(
    case: AbCase, run: AbRun, *, typed: bool
) -> dict[str, object]:
    route = "typed" if typed else "sidecar"
    control_dir = run.bundled_dir.parent / f"bundled_cv_restraint_{route}_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    if typed:
        with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
            root = "/restraint/cv/config"
            keys = protocol[f"{root}/key"].asstr()[...].tolist()
            values = protocol[f"{root}/value"].asstr()[...].tolist()
            weight_index = keys.index("weight")
            values[weight_index] = "0.0"
            del protocol[f"{root}/value"]
            protocol.create_dataset(
                f"{root}/value",
                data=values,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
    else:
        sidecar_path = (
            control_dir
            / "legacy_sidecars"
            / "restrain_cv_in_file"
            / "restrain_cv.txt"
        )
        text = sidecar_path.read_text(encoding="utf-8")
        if "weight = 4.0" not in text:
            raise AssertionError(
                f"{case.name} sidecar control cannot find the weight"
            )
        sidecar_path.write_text(
            text.replace("weight = 4.0", "weight = 0.0"), encoding="utf-8"
        )

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} zero-weight control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} zero-weight control emitted {len(rows)} rows"
        )
    result = {
        "energy": rows[0]["restrain_cv"],
        "force": _read_native_float32_file(
            control_dir / "output" / "legacy.frc"
        ),
    }
    shutil.rmtree(control_dir)
    return result


def _run_cv_restraint_reordered_definition_control(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    control_dir = (
        run.bundled_dir.parent / "bundled_cv_restraint_reordered_definition"
    )
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    for path in (
        control_dir / "output",
        control_dir / ".sponge_h5_native_protocol",
    ):
        if path.exists():
            shutil.rmtree(path)
    (control_dir / "output").mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
        config = protocol.create_group("/cv/config")
        section = config.create_group("section")
        section.create_dataset("count", data=1, dtype="i8")
        section.create_dataset("name", data=["distance"], dtype=string_dtype)
        section.create_dataset("key_offset", data=[0, 2], dtype="i8")
        config.create_dataset(
            "key", data=["atom", "CV_type"], dtype=string_dtype
        )
        config.create_dataset(
            "value", data=["0 1", "distance"], dtype=string_dtype
        )

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} reordered-definition control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(rows) != 1:
        raise AssertionError(
            f"{case.name} reordered-definition control emitted {len(rows)} rows"
        )
    result = {
        "energy": rows[0]["restrain_cv"],
        "force": _read_native_float32_file(
            control_dir / "output" / "legacy.frc"
        ),
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_custom_pair_forces(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    materialized_root = run.bundled_dir / ".sponge_h5_native_custom_force"
    materialized = (
        materialized_root / "pairwise_force.txt",
        materialized_root / "custom_pair.txt",
    )
    missing = [
        path
        for path in materialized
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise AssertionError(
            f"{case.name} did not materialize bundled custom-pair payloads: "
            f"{missing}"
        )
    legacy = _read_native_float32_file(run.legacy_dir / "output" / "legacy.frc")
    bundled = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    result = _assert_nontrivial_equivalent_forces(
        f"{case.name} custom-pair force", legacy, bundled
    )
    result["bundled_materialized_paths"] = [
        str(path.relative_to(run.bundled_dir)) for path in materialized
    ]
    return result


def _compare_focused_core_topology_sensitivity(
    case: AbCase, run: AbRun, contract_id: str
) -> dict[str, object]:
    controls = {
        "input.topology.mass": {
            "datasets": {"/atoms/mass": (2.0, 8.0)},
            "observable": "temperature",
            "minimum_delta": 1.0e-2,
            "force_changes": False,
        },
        "input.topology.charge": {
            "datasets": {"/atoms/charge": (0.0, 0.0)},
            "observable": "Coulomb",
            "minimum_delta": 1.0e-2,
            "force_changes": True,
        },
        "input.topology.lj": {
            "datasets": {
                "/forcefield/lj/pair_A_12": (0.0,),
                "/forcefield/lj/pair_B_6": (0.0,),
            },
            "observable": "LJ",
            "minimum_delta": 1.0e-3,
            "force_changes": True,
        },
    }
    if contract_id not in controls:
        raise AssertionError(
            f"{case.name} has no payload control for {contract_id}"
        )
    control = controls[contract_id]
    suffix = contract_id.rsplit(".", 1)[-1]
    control_dir = run.bundled_dir.parent / f"bundled_{suffix}_payload_control"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        for dataset_path, values in control["datasets"].items():
            if dataset_path not in topology:
                raise AssertionError(
                    f"{case.name} control is missing {dataset_path}"
                )
            topology[dataset_path][...] = values

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} {contract_id} payload control failed with "
            f"code {outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    correct_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(correct_rows) != 1 or len(control_rows) != 1:
        raise AssertionError(
            f"{case.name} {contract_id} control expected one mdout row"
        )
    observable = control["observable"]
    correct_value = correct_rows[0].get(observable, math.nan)
    control_value = control_rows[0].get(observable, math.nan)
    correct_force = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    response = _assert_core_topology_payload_response(
        f"{case.name} {contract_id}",
        observable,
        correct_value,
        control_value,
        minimum_observable_delta=float(control["minimum_delta"]),
        correct_force=correct_force,
        control_force=control_force,
        force_must_change=bool(control["force_changes"]),
    )

    result = {
        "contract_id": contract_id,
        "typed_datasets": sorted(control["datasets"]),
        "observable": observable,
        "correct_value": correct_value,
        "control_value": control_value,
        "force_changes": control["force_changes"],
        "exit_code": outcome.returncode,
        **response,
    }
    shutil.rmtree(control_dir)
    return result


FOCUSED_BONDED_ORACLES = {
    "input.topology.bond": {
        "module": "bond",
        "observable": "bond",
        "mutation_paths": ("/forcefield/bond/k",),
        "expected_energy": 2.5,
        "expected_force": (10.0, 0.0, 0.0, -10.0, 0.0, 0.0),
    },
    "input.topology.angle": {
        "module": "angle",
        "observable": "angle",
        "mutation_paths": ("/forcefield/angle/k",),
        "expected_energy": 2.0 * (math.pi / 2.0 - 0.5) ** 2,
        "expected_force": (
            0.0,
            4.0 * (math.pi / 2.0 - 0.5),
            0.0,
            -4.0 * (math.pi / 2.0 - 0.5),
            -4.0 * (math.pi / 2.0 - 0.5),
            0.0,
            4.0 * (math.pi / 2.0 - 0.5),
            0.0,
            0.0,
        ),
    },
    "input.topology.urey_bradley": {
        "module": "urey_bradley",
        "observable": "urey_bradley",
        "mutation_paths": ("/forcefield/urey_bradley/bond_k",),
        "expected_energy": 5.0 * (math.sqrt(2.0) - 1.0) ** 2,
        "expected_force": (
            -5.0 * (2.0 - math.sqrt(2.0)),
            5.0 * (2.0 - math.sqrt(2.0)),
            0.0,
            0.0,
            0.0,
            0.0,
            5.0 * (2.0 - math.sqrt(2.0)),
            -5.0 * (2.0 - math.sqrt(2.0)),
            0.0,
        ),
    },
    "input.topology.dihedral": {
        "module": "dihedral",
        "observable": "dihedral",
        "mutation_paths": ("/forcefield/dihedral/k",),
        "expected_energy": 2.0 * (1.0 + 1.0 / math.sqrt(2.0)),
        "expected_force": (
            0.0,
            0.0,
            -math.sqrt(2.0),
            0.0,
            0.0,
            math.sqrt(2.0),
            0.0,
            1.0 / math.sqrt(2.0),
            -1.0 / math.sqrt(2.0),
            0.0,
            -1.0 / math.sqrt(2.0),
            1.0 / math.sqrt(2.0),
        ),
    },
    "input.custom.listed": {
        "module": "listed",
        "observable": "custom_bond",
        "mutation_paths": (
            "/forcefield/custom_force/listed/data/custom_bond/parameter/value",
            "/forcefield/custom_force/listed/data/custom_bond/parameter/float_value",
        ),
        "expected_energy": 1.75,
        "expected_force": (7.0, 0.0, 0.0, -7.0, 0.0, 0.0),
    },
}


def _compare_focused_bonded_payload_sensitivity(
    case: AbCase, run: AbRun, contract_id: str
) -> dict[str, object]:
    if contract_id not in FOCUSED_BONDED_ORACLES:
        raise AssertionError(
            f"{case.name} has no bonded oracle for {contract_id}"
        )
    oracle = FOCUSED_BONDED_ORACLES[contract_id]
    expected_module = FOCUSED_BONDED_MODULES[case.fixture_case]
    if oracle["module"] != expected_module:
        raise AssertionError(
            f"{case.name} fixture/oracle mismatch: "
            f"{expected_module} != {oracle['module']}"
        )
    observable = str(oracle["observable"])
    branch_results = {}
    branch_forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        if len(rows) != 1:
            raise AssertionError(f"{case.name} {branch} expected one mdout row")
        energy = rows[0].get(observable, math.nan)
        if not math.isfinite(energy) or abs(energy) <= 1.0e-8:
            raise AssertionError(
                f"{case.name} {branch} has trivial {observable} energy"
            )
        force = _read_native_float32_file(directory / "output" / "legacy.frc")
        _require_finite_values(case, f"{branch} {observable} force", force)
        expected_force = oracle["expected_force"]
        if expected_force is not None:
            _assert_numeric_sequences_close(
                f"{case.name} {branch} precomputed force oracle",
                expected_force,
                force,
                relative_tolerance=1.0e-5,
                absolute_tolerance=1.0e-5,
            )
        if not math.isclose(
            energy,
            float(oracle["expected_energy"]),
            rel_tol=0.0,
            abs_tol=5.1e-3,
        ):
            raise AssertionError(
                f"{case.name} {branch} energy oracle mismatch: {energy}"
            )
        branch_results[branch] = {"energy": energy, "force": force}
        branch_forces[branch] = force

    _assert_numeric_sequences_close(
        f"{case.name} bonded energy",
        [branch_results["legacy"]["energy"]],
        [branch_results["bundled"]["energy"]],
        relative_tolerance=1.0e-5,
        absolute_tolerance=1.0e-6,
    )
    force_equivalence = _assert_nontrivial_equivalent_forces(
        f"{case.name} bonded force",
        branch_forces["legacy"],
        branch_forces["bundled"],
    )

    control_dir = run.bundled_dir.parent / f"bundled_{expected_module}_k_zero"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
        for dataset_path in oracle["mutation_paths"]:
            if dataset_path not in topology:
                raise AssertionError(
                    f"{case.name} control is missing {dataset_path}"
                )
            dataset = topology[dataset_path]
            if expected_module == "listed":
                dataset[0, 2] = 0.0
            else:
                dataset[0] = 0.0
    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} zero-k control failed with {outcome.returncode}\n"
            f"{outcome.stdout}\n{outcome.stderr}"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1:
        raise AssertionError(f"{case.name} zero-k control expected one row")
    control_energy = control_rows[0].get(observable, math.nan)
    control_force = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    _require_finite_values(case, "zero-k control force", control_force)
    if not math.isfinite(control_energy) or abs(control_energy) > 1.0e-6:
        raise AssertionError(
            f"{case.name} zero-k control retained energy {control_energy}"
        )
    if max(abs(value) for value in control_force) > 1.0e-5:
        raise AssertionError(f"{case.name} zero-k control retained force")
    maximum_force_delta = max(
        abs(left - right)
        for left, right in zip(
            branch_forces["bundled"], control_force, strict=True
        )
    )
    shutil.rmtree(control_dir)
    return {
        "method": "isolated_typed_payload_with_k_zero_mutation",
        "module": expected_module,
        "observable": observable,
        "typed_mutation_paths": list(oracle["mutation_paths"]),
        "expected_energy": oracle["expected_energy"],
        "expected_force": oracle["expected_force"],
        "branches": branch_results,
        "force_equivalence": force_equivalence,
        "control": {
            "energy": control_energy,
            "maximum_force_magnitude": max(
                abs(value) for value in control_force
            ),
            "maximum_force_delta": maximum_force_delta,
        },
    }


def _assert_core_topology_payload_response(
    label: str,
    observable: str,
    correct_value: float,
    control_value: float,
    *,
    minimum_observable_delta: float,
    correct_force: Sequence[float],
    control_force: Sequence[float],
    force_must_change: bool,
) -> dict[str, float]:
    if not math.isfinite(correct_value) or not math.isfinite(control_value):
        raise AssertionError(
            f"{label} has no finite {observable} payload fingerprint"
        )
    if abs(correct_value) <= 1.0e-8:
        raise AssertionError(f"{label} correct {observable} is trivial")
    observable_delta = abs(correct_value - control_value)
    if observable_delta <= minimum_observable_delta:
        raise AssertionError(
            f"{label} payload did not change {observable}: "
            f"correct={correct_value}, control={control_value}"
        )
    if len(correct_force) != 6 or len(control_force) != 6:
        raise AssertionError(f"{label} force control has wrong shape")
    if not all(
        math.isfinite(value) for value in (*correct_force, *control_force)
    ):
        raise AssertionError(f"{label} force control is non-finite")
    force_delta = max(
        abs(correct - mutated)
        for correct, mutated in zip(correct_force, control_force, strict=True)
    )
    if force_must_change and force_delta <= 1.0e-4:
        raise AssertionError(f"{label} payload did not change force")
    if not force_must_change and force_delta > 1.0e-6:
        raise AssertionError(
            f"{label} mass-only control unexpectedly changed force"
        )
    return {
        "observable_delta": observable_delta,
        "force_delta": force_delta,
    }


def _compare_focused_exclusions_oracle(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_exclusion_coulomb_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch = _assert_nontrivial_equivalent_forces(
        f"{case.name} exclusion force",
        forces["legacy"],
        forces["bundled"],
    )
    return {
        "excluded_pair": [0, 1],
        "branches": branch_results,
        "cross_branch_force": cross_branch,
    }


def _compare_focused_residue_pbc_mapping(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branches = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        mdinfo = (directory / "mdinfo.txt").read_text(encoding="utf-8")
        residue_match = re.search(r"\bresidue_numbers is\s+(\d+)\b", mdinfo)
        if residue_match is None:
            raise AssertionError(
                f"{case.name} {branch} mdinfo has no residue count"
            )
        stdout = (directory / "run.stdout").read_text(encoding="utf-8")
        runtime_match = re.search(
            r"rank_id=0, atom_numbers=4, residue_numbers=(\d+)", stdout
        )
        if runtime_match is None:
            raise AssertionError(
                f"{case.name} {branch} has no runtime domain residue count"
            )
        with h5py.File(directory / TRAJECTORY_REL, "r") as trajectory:
            positions = (
                trajectory["/particles/all/position/value"][...]
                .reshape(-1)
                .tolist()
            )
        branches[branch] = _assert_residue_pbc_mapping_oracle(
            f"{case.name} {branch}",
            rows,
            branch_forces,
            residue_numbers=int(residue_match.group(1)),
            runtime_residue_numbers=int(runtime_match.group(1)),
            positions=positions,
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} residue PBC support force",
        forces["legacy"],
        forces["bundled"],
    )
    result = {
        "route": (
            "typed_h5_atoms_residue_index"
            if case.fixture_case == FOCUSED_RESIDUE_TYPED_PBC_FIXTURE
            else "isolated_h5_residue_in_file_topology_sidecar"
        ),
        "residue_atom_counts": [2, 2],
        "runtime_consumer": "force_whole_output molecule coordinate mapping",
        "branches": branches,
        "cross_branch_force": cross_branch_force,
    }
    if case.fixture_case == FOCUSED_RESIDUE_TYPED_PBC_FIXTURE:
        result["typed_validation_controls"] = (
            _run_typed_residue_validation_controls(case, run)
        )
    return result


def _run_typed_residue_validation_controls(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    controls = (
        (
            "inconsistent_offset",
            "spongeErrorBadFileFormat",
            (
                "/atoms/residue_index",
                "/residues/atom_offset",
                "different residue mappings",
            ),
        ),
        (
            "legacy_conflict",
            "spongeErrorConflictingCommand",
            ("native residue mapping", "residue_in_file"),
        ),
    )
    results = {}
    for name, expected_category, expected_tokens in controls:
        control_dir = run.bundled_dir.parent / f"bundled_residue_{name}"
        if control_dir.exists():
            shutil.rmtree(control_dir)
        shutil.copytree(run.bundled_dir, control_dir)
        output_dir = control_dir / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        for file_name in (
            "mdout.txt",
            "mdinfo.txt",
            "run.stdout",
            "run.stderr",
        ):
            path = control_dir / file_name
            if path.exists():
                path.unlink()

        if name == "inconsistent_offset":
            with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
                topology["/residues/atom_offset"][...] = [0, 1, 4]
        else:
            shutil.copy2(
                run.legacy_dir / "residue.txt", control_dir / "residue.txt"
            )
            mdin_path = control_dir / _mdin_name(control_dir)
            text = mdin_path.read_text(encoding="utf-8")
            mdin_path.write_text(
                _insert_root_toml_keys(
                    text, ['residue_in_file = "residue.txt"']
                ),
                encoding="utf-8",
            )

        outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
        combined = outcome.stdout + outcome.stderr
        category = _failure_category(combined)
        if outcome.returncode == 0 or category != expected_category:
            raise AssertionError(
                f"{case.name} {name} control was not rejected as "
                f"{expected_category}: code={outcome.returncode}, "
                f"category={category}\n{combined}"
            )
        missing = [token for token in expected_tokens if token not in combined]
        if missing:
            raise AssertionError(
                f"{case.name} {name} control lost diagnostics: {missing}"
            )
        results[name] = {
            "exit_code": outcome.returncode,
            "failure_category": category,
            "diagnostic_tokens": list(expected_tokens),
        }
    return results


def _assert_residue_pbc_mapping_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
    *,
    residue_numbers: int,
    runtime_residue_numbers: int,
    positions: Sequence[float],
) -> dict[str, object]:
    if len(rows) != 1:
        raise AssertionError(f"{label} expected one mdout row, got {len(rows)}")
    row = rows[0]
    if "bond" not in row or not math.isfinite(row["bond"]) or row["bond"] <= 0:
        raise AssertionError(
            f"{label} has no nonzero support force-field result"
        )
    if residue_numbers != 2:
        raise AssertionError(
            f"{label} residue state mismatch: expected 2, got {residue_numbers}"
        )
    if runtime_residue_numbers != 2:
        raise AssertionError(
            f"{label} runtime residue partition mismatch: expected 2, "
            f"got {runtime_residue_numbers}"
        )
    expected_positions = (
        19.0,
        0.0,
        0.0,
        21.0,
        0.0,
        0.0,
        25.0,
        0.0,
        0.0,
        28.0,
        0.0,
        0.0,
    )
    _assert_numeric_sequences_close(
        f"{label} residue-owned PBC mapping oracle",
        expected_positions,
        positions,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-6,
    )
    if len(forces) != 12 or max(abs(value) for value in forces) <= 1.0:
        raise AssertionError(f"{label} support force is absent or incomplete")
    return {
        "bond": row["bond"],
        "residue_numbers": residue_numbers,
        "runtime_residue_numbers": runtime_residue_numbers,
        "mapped_positions": list(positions),
        "maximum_abs_force": max(abs(value) for value in forces),
    }


def _compare_focused_residue_com_res_virial(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branches = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        mdinfo = (directory / "mdinfo.txt").read_text(encoding="utf-8")
        residue_match = re.search(r"\bresidue_numbers is\s+(\d+)\b", mdinfo)
        if residue_match is None:
            raise AssertionError(
                f"{case.name} {branch} mdinfo has no residue count"
            )
        with h5py.File(directory / TRAJECTORY_REL, "r") as trajectory:
            positions = (
                trajectory["/particles/all/position/value"][...]
                .reshape(-1)
                .tolist()
            )
        branches[branch] = _assert_residue_com_res_virial_oracle(
            f"{case.name} {branch}",
            rows,
            branch_forces,
            residue_numbers=int(residue_match.group(1)),
            positions=positions,
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} com_res restraint force",
        forces["legacy"],
        forces["bundled"],
    )
    return {
        "route": (
            "typed_h5_atoms_residue_index"
            if case.fixture_case == FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE
            else "isolated_h5_residue_in_file_topology_sidecar"
        ),
        "residue_atom_counts": [2, 2],
        "runtime_consumer": "restrain com_res virial and pressure",
        "wrong_partition_control": _run_residue_wrong_partition_control(
            case, run
        ),
        "branches": branches,
        "cross_branch_force": cross_branch_force,
    }


def _assert_residue_com_res_virial_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
    *,
    residue_numbers: int,
    positions: Sequence[float],
) -> dict[str, object]:
    if len(rows) != 1:
        raise AssertionError(f"{label} expected one mdout row, got {len(rows)}")
    row = rows[0]
    for key in ("bond", "restrain", "pressure", "Pxx"):
        if key not in row or not math.isfinite(row[key]):
            raise AssertionError(f"{label} has no finite {key} result")
    if not math.isclose(row["bond"], 2.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise AssertionError(f"{label} bond energy mismatch: {row['bond']}")
    if not math.isclose(row["restrain"], 2.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise AssertionError(
            f"{label} restraint energy mismatch: {row['restrain']}"
        )
    if not math.isclose(row["pressure"], 0.04, rel_tol=0.0, abs_tol=1.0e-2):
        raise AssertionError(
            f"{label} com_res pressure mismatch: {row['pressure']}"
        )
    if not math.isclose(row["Pxx"], 0.11, rel_tol=0.0, abs_tol=1.0e-2):
        raise AssertionError(f"{label} com_res Pxx mismatch: {row['Pxx']}")
    if residue_numbers != 2:
        raise AssertionError(
            f"{label} residue state mismatch: expected 2, got {residue_numbers}"
        )
    if len(positions) != 12 or not all(
        math.isfinite(value) for value in positions
    ):
        raise AssertionError(
            f"{label} trajectory positions are absent or non-finite"
        )
    if len(forces) != 12 or not all(math.isfinite(value) for value in forces):
        raise AssertionError(
            f"{label} restraint force payload is absent or non-finite"
        )
    if max(abs(value) for value in forces) <= 1.0:
        raise AssertionError(f"{label} restraint force payload is trivial")
    return {
        "bond": row["bond"],
        "restrain": row["restrain"],
        "pressure": row["pressure"],
        "Pxx": row["Pxx"],
        "residue_numbers": residue_numbers,
        "mapped_positions": list(positions),
        "maximum_abs_force": max(abs(value) for value in forces),
    }


def _run_residue_wrong_partition_control(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_wrong_residue_partition"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    if case.fixture_case == FOCUSED_RESIDUE_TYPED_COM_RES_FIXTURE:
        with h5py.File(topology_path, "r+") as topology:
            if sidecar_root in topology:
                raise AssertionError(
                    f"{case.name} typed control retained topology sidecars"
                )
            residue_index = topology["/atoms/residue_index"]
            atom_offset = topology["/residues/atom_offset"]
            if residue_index[...].tolist() != [0, 0, 1, 1]:
                raise AssertionError(
                    f"{case.name} typed control lost canonical membership"
                )
            residue_index[...] = [0, 1, 1, 1]
            atom_offset[...] = [0, 1, 4]
    else:
        keys = _h5_string_values(topology_path, f"{sidecar_root}/key")
        paths = _h5_string_values(topology_path, f"{sidecar_root}/path")
        if keys != ["residue_in_file"] or len(paths) != 1:
            raise AssertionError(
                f"{case.name} wrong-partition control lost isolated residue route"
            )
        (control_dir / paths[0]).write_text("4 2\n1\n3\n", encoding="utf-8")

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} wrong-partition control failed with "
            f"code {outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    mdinfo = (control_dir / "mdinfo.txt").read_text(encoding="utf-8")
    declared_match = re.search(r"\bresidue_numbers is\s+(\d+)\b", mdinfo)
    if declared_match is None or int(declared_match.group(1)) != 2:
        raise AssertionError(
            f"{case.name} wrong-partition control did not load two residues"
        )
    split_diagnostic = (
        "Residue 1 is disconnected (components=2, atoms=3). "
        "Splitting into contiguous segments."
    )
    if split_diagnostic not in outcome.stdout:
        raise AssertionError(
            f"{case.name} wrong-partition control did not split the "
            "cross-molecule residue"
        )
    control_rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    correct_rows = _read_mdout(run.bundled_dir / "mdout.txt")["rows"]
    if len(control_rows) != 1 or len(correct_rows) != 1:
        raise AssertionError(
            f"{case.name} wrong-partition control expected one mdout row"
        )
    control_row = control_rows[0]
    correct_row = correct_rows[0]
    if not math.isclose(
        control_row.get("restrain", math.nan),
        2.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise AssertionError(
            f"{case.name} wrong-partition control changed restraint energy"
        )
    for key in ("bond", "restrain"):
        if not math.isclose(
            control_row.get(key, math.nan),
            correct_row.get(key, math.nan),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise AssertionError(
                f"{case.name} wrong-partition control changed {key} energy"
            )
    if not math.isclose(
        control_row.get("pressure", math.nan),
        -11.53,
        rel_tol=0.0,
        abs_tol=1.0e-2,
    ):
        raise AssertionError(
            f"{case.name} wrong-partition pressure fingerprint changed"
        )
    if not math.isclose(
        control_row.get("Pxx", math.nan),
        -34.60,
        rel_tol=0.0,
        abs_tol=1.0e-2,
    ):
        raise AssertionError(
            f"{case.name} wrong-partition Pxx fingerprint changed"
        )
    if (
        abs(correct_row["pressure"] - control_row["pressure"]) <= 1.0
        or abs(correct_row["Pxx"] - control_row["Pxx"]) <= 1.0
    ):
        raise AssertionError(
            f"{case.name} residue membership did not distinguish virial pressure"
        )
    control_forces = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    correct_forces = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    force_equivalence = _assert_nontrivial_equivalent_forces(
        f"{case.name} wrong-partition control force",
        correct_forces,
        control_forces,
    )
    result = {
        "input_residue_atom_counts": [1, 3],
        "declared_residue_numbers": 2,
        "split_diagnostic": split_diagnostic,
        "correct_pressure": correct_row["pressure"],
        "control_pressure": control_row["pressure"],
        "correct_Pxx": correct_row["Pxx"],
        "control_Pxx": control_row["Pxx"],
        "restrain": control_row["restrain"],
        "force_equivalence": force_equivalence,
        "exit_code": outcome.returncode,
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_gb_forces(case: AbCase, run: AbRun) -> dict[str, object]:
    forces = {}
    branches = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branches[branch] = _assert_gb_force_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch = _assert_nontrivial_equivalent_forces(
        f"{case.name} GB force", forces["legacy"], forces["bundled"]
    )
    return {
        "route": (
            "pure_native_gb_state"
            if case.name == "normal_gb_native_nonzero"
            else "native_gb_state_plus_h5_sidecar_activation"
        ),
        "branches": branches,
        "cross_branch_force": cross_branch,
    }


def _assert_gb_force_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    if len(rows) != 1:
        raise AssertionError(f"{label} expected one mdout row, got {len(rows)}")
    row = rows[0]
    for observable, expected_value in (
        ("Coulomb", -0.50),
        ("gb", -0.25),
        ("potential", -0.75),
    ):
        actual = row.get(observable, math.nan)
        if not math.isclose(
            actual, expected_value, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise AssertionError(
                f"{label} {observable} oracle mismatch: {actual}"
            )
    expected = (
        0.10313021,
        0.0,
        0.0,
        -0.10313021,
        0.0,
        0.0,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} GB+Coulomb force oracle",
        expected,
        forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    maximum_abs_force = max(abs(value) for value in forces)
    coulomb_only_maximum = 0.25
    if math.isclose(
        maximum_abs_force,
        coulomb_only_maximum,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AssertionError(f"{label} retained the Coulomb-only force")
    return {
        "Coulomb": row["Coulomb"],
        "gb": row["gb"],
        "potential": row["potential"],
        "maximum_abs_force": maximum_abs_force,
        "coulomb_only_maximum": coulomb_only_maximum,
        "gb_force_contribution_required": True,
    }


def _compare_focused_improper_forces(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        rows = _read_mdout(directory / "mdout.txt")["rows"]
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_focused_improper_oracle(
            f"{case.name} {branch}", rows, branch_forces
        )
        forces[branch] = branch_forces
    cross_branch_force = _assert_nontrivial_equivalent_forces(
        f"{case.name} improper force", forces["legacy"], forces["bundled"]
    )
    return {
        "route": (
            "pure_native_typed_h5"
            if case.fixture_case == FOCUSED_IMPROPER_NATIVE_FIXTURE
            else "unmodified_legacy_to_bundle_output"
        ),
        "branches": branch_results,
        "cross_branch_force": cross_branch_force,
        "bundled_native_paths": [
            "/forcefield/improper/atoms",
            "/forcefield/improper/pk",
            "/forcefield/improper/phi0",
        ],
        "typed_pk_half_control": _run_improper_pk_half_control(
            case, run, forces["bundled"]
        ),
    }


def _assert_focused_improper_oracle(
    label: str,
    rows: Sequence[dict[str, float]],
    forces: Sequence[float],
    *,
    parameter_scale: float = 1.0,
) -> dict[str, float]:
    if len(rows) != 1:
        raise AssertionError(
            f"{label} expected one mdout row, found {len(rows)}"
        )
    energy = rows[0].get("improper_dihedral", math.nan)
    expected_energy = 31.36 * parameter_scale
    if not math.isfinite(energy) or not math.isclose(
        energy, expected_energy, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise AssertionError(
            f"{label} improper energy changed: {energy}, "
            f"expected {expected_energy}"
        )
    if len(forces) != 12 or not all(math.isfinite(value) for value in forces):
        raise AssertionError(f"{label} improper force payload is invalid")
    maximum_abs_force = max(abs(value) for value in forces)
    expected_maximum = 35.415924072265625 * parameter_scale
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    if not math.isclose(
        maximum_abs_force,
        expected_maximum,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AssertionError(
            f"{label} improper maximum force changed: {maximum_abs_force}, "
            f"expected {expected_maximum}"
        )
    return {
        "improper_dihedral": energy,
        "maximum_abs_force": maximum_abs_force,
        "parameter_scale": parameter_scale,
    }


def _run_improper_pk_half_control(
    case: AbCase,
    run: AbRun,
    baseline_forces: Sequence[float],
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_improper_pk_half"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    topology_path = control_dir / "topology.spgt.h5"
    with h5py.File(topology_path, "r+") as topology:
        pk = topology["/forcefield/improper/pk"]
        if pk.shape != (1,) or not math.isclose(
            float(pk[0]), 10.0, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise AssertionError(
                f"{case.name} improper pk control lost canonical payload"
            )
        pk[...] = [5.0]

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} pk=5 control failed with code {outcome.returncode}\n"
            f"{outcome.stdout}\n{outcome.stderr}"
        )
    rows = _read_mdout(control_dir / "mdout.txt")["rows"]
    half_forces = _read_native_float32_file(
        control_dir / "output" / "legacy.frc"
    )
    oracle = _assert_focused_improper_oracle(
        f"{case.name} pk=5 control",
        rows,
        half_forces,
        parameter_scale=0.5,
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{case.name} pk=5 scaled force",
        [0.5 * value for value in baseline_forces],
        half_forces,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    maximum_force_delta = max(
        abs(full - half)
        for full, half in zip(baseline_forces, half_forces, strict=True)
    )
    if maximum_force_delta <= 1.0:
        raise AssertionError(
            f"{case.name} typed pk did not change improper force"
        )
    result = {
        **oracle,
        "typed_pk": 5.0,
        "maximum_force_delta": maximum_force_delta,
        "exit_code": outcome.returncode,
    }
    shutil.rmtree(control_dir)
    return result


def _compare_focused_lj_soft_core_forces(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    legacy = _read_native_float32_file(run.legacy_dir / "output" / "legacy.frc")
    bundled = _read_native_float32_file(
        run.bundled_dir / "output" / "legacy.frc"
    )
    result = _assert_nontrivial_equivalent_forces(
        f"{case.name} LJ soft-core force", legacy, bundled
    )
    result["bundled_native_path"] = "/forcefield/lj_soft_core"
    result["subsystem_division_present"] = False
    return result


def _assert_subsystem_partition_response(
    label: str, values: dict[str, float]
) -> dict[str, float]:
    if not all(math.isfinite(value) for value in values.values()):
        raise AssertionError(
            f"{label} has non-finite partition values: {values}"
        )
    if abs(values["baseline_inter"]) <= 1.0e-2:
        raise AssertionError(f"{label} cross-subsystem energy is trivial")
    if abs(values["baseline_intra"]) > 1.0e-2:
        raise AssertionError(f"{label} baseline intra energy is not empty")
    if abs(values["control_intra"]) <= 1.0e-2:
        raise AssertionError(f"{label} all-intra control energy is trivial")
    if abs(values["control_inter"]) > 1.0e-2:
        raise AssertionError(f"{label} all-intra control retained inter energy")
    for prefix in ("baseline", "control"):
        partition_sum = values[f"{prefix}_inter"] + values[f"{prefix}_intra"]
        if not math.isclose(
            partition_sum,
            values[f"{prefix}_total"],
            rel_tol=0.0,
            abs_tol=1.0e-2,
        ):
            raise AssertionError(
                f"{label} {prefix} partition does not conserve "
                f"LJ_soft_short: {values}"
            )
    total_delta = abs(values["baseline_total"] - values["control_total"])
    if total_delta > 1.0e-2:
        raise AssertionError(
            f"{label} mask changed total soft-core energy: {values}"
        )
    return {
        "inter_response": abs(
            values["baseline_inter"] - values["control_inter"]
        ),
        "intra_response": abs(
            values["baseline_intra"] - values["control_intra"]
        ),
        "total_delta": total_delta,
    }


def _compare_focused_subsystem_partition(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    controls: dict[str, Path] = {}
    for branch, source_dir in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        control_dir = (
            source_dir.parent / f"{branch}_subsystem_partition_control"
        )
        if control_dir.exists():
            shutil.rmtree(control_dir)
        shutil.copytree(source_dir, control_dir)
        output_dir = control_dir / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        for file_name in (
            "mdout.txt",
            "mdinfo.txt",
            "run.stdout",
            "run.stderr",
        ):
            path = control_dir / file_name
            if path.exists():
                path.unlink()
        if branch == "legacy":
            (control_dir / "subsys_division.txt").write_text(
                "2\n0\n0\n", encoding="utf-8"
            )
        else:
            with h5py.File(control_dir / "topology.spgt.h5", "r+") as topology:
                topology["/forcefield/subsys_division"][...] = (0, 0)
        outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
        if outcome.returncode != 0:
            raise AssertionError(
                f"{case.name} {branch} partition control failed with "
                f"code {outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
            )
        controls[branch] = control_dir

    observable_spec = InputSemanticSpec(
        "input.topology.subsystem_division",
        ("LJ_soft_inter", "LJ_soft_intra"),
        1.0e-6,
    )
    control_rows = {
        branch: _read_mdout(path / "mdout.txt")["rows"]
        for branch, path in controls.items()
    }
    assert_module_semantics(
        f"{case.name} all-intra control",
        control_rows["legacy"],
        control_rows["bundled"],
        observable_spec,
        deterministic=True,
    )

    partition_values: dict[str, dict[str, float]] = {}
    for branch, source_dir in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        baseline_rows = _read_mdout(source_dir / "mdout.txt")["rows"]
        if len(baseline_rows) != 1 or len(control_rows[branch]) != 1:
            raise AssertionError(
                f"{case.name} {branch} partition oracle expected one mdout row"
            )
        baseline = baseline_rows[0]
        control = control_rows[branch][0]
        values = {
            "baseline_inter": float(baseline.get("LJ_soft_inter", math.nan)),
            "baseline_intra": float(baseline.get("LJ_soft_intra", math.nan)),
            "baseline_total": float(baseline.get("LJ_soft_short", math.nan)),
            "control_inter": float(control.get("LJ_soft_inter", math.nan)),
            "control_intra": float(control.get("LJ_soft_intra", math.nan)),
            "control_total": float(control.get("LJ_soft_short", math.nan)),
        }
        _assert_subsystem_partition_response(f"{case.name} {branch}", values)
        _assert_nontrivial_equivalent_forces(
            f"{case.name} {branch} mask-invariant force",
            _read_native_float32_file(source_dir / "output" / "legacy.frc"),
            _read_native_float32_file(
                controls[branch] / "output" / "legacy.frc"
            ),
        )
        partition_values[branch] = values

    _assert_nontrivial_equivalent_forces(
        f"{case.name} all-intra control A/B force",
        _read_native_float32_file(controls["legacy"] / "output" / "legacy.frc"),
        _read_native_float32_file(
            controls["bundled"] / "output" / "legacy.frc"
        ),
    )
    for observable in partition_values["legacy"]:
        if not math.isclose(
            partition_values["legacy"][observable],
            partition_values["bundled"][observable],
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise AssertionError(
                f"{case.name} partition A/B mismatch for {observable}: "
                f"{partition_values}"
            )

    for path in controls.values():
        shutil.rmtree(path)
    return {
        "bundled_native_path": "/forcefield/subsys_division",
        "baseline_mask": [0, 1],
        "control_mask": [0, 0],
        "partition_values": partition_values,
        "total_energy_invariant": True,
        "force_invariant": True,
    }


def _compare_focused_constraint_projection(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    branch_results = {}
    positions = {}
    velocities = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        branch_positions = _read_native_float32_file(
            directory / "output" / "legacy.crd"
        )
        branch_velocities = _read_native_float32_file(
            directory / "output" / "legacy.vel"
        )
        branch_results[branch] = _assert_constraint_projection_oracle(
            f"{case.name} {branch}", branch_positions, branch_velocities
        )
        if branch_results[branch]["frame_count"] != case.normal_step_limit:
            raise AssertionError(
                f"{case.name} {branch} constrained frame count changed: "
                f"expected={case.normal_step_limit}, "
                f"actual={branch_results[branch]['frame_count']}"
            )
        positions[branch] = branch_positions
        velocities[branch] = branch_velocities

    position_relative, position_absolute = _deterministic_tolerance("position")
    velocity_relative, velocity_absolute = _deterministic_tolerance("velocity")
    frame_count = int(branch_results["legacy"]["frame_count"])
    _assert_periodic_positions_close(
        f"{case.name} constrained positions",
        positions["legacy"],
        positions["bundled"],
        (frame_count, 2, 3),
        (10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0),
        relative_tolerance=position_relative,
        absolute_tolerance=position_absolute,
    )
    _assert_numeric_sequences_close(
        f"{case.name} constrained velocities",
        velocities["legacy"],
        velocities["bundled"],
        relative_tolerance=velocity_relative,
        absolute_tolerance=velocity_absolute,
    )
    position_error = max(
        abs((left - right) - round((left - right) / 10.0) * 10.0)
        for left, right in zip(
            positions["legacy"], positions["bundled"], strict=True
        )
    )
    velocity_error = max(
        abs(left - right)
        for left, right in zip(
            velocities["legacy"], velocities["bundled"], strict=True
        )
    )
    result = {
        "method": "per_frame_distance_and_radial_velocity_projection",
        "route": (
            "typed_h5_constraint_pairs"
            if case.fixture_case == FOCUSED_CONSTRAINT_TYPED_FIXTURE
            else "isolated_h5_constrain_in_file_protocol_sidecar"
        ),
        "target_distance": 1.5,
        "initial_relative_radial_speed": 2.0,
        "branches": branch_results,
        "cross_branch_position_max_absolute_error": position_error,
        "cross_branch_velocity_max_absolute_error": velocity_error,
    }
    if case.fixture_case == FOCUSED_CONSTRAINT_TYPED_FIXTURE:
        result["target_distance_control"] = (
            _run_typed_constraint_target_control(case, run)
        )
        result["invalid_pair_control"] = (
            _run_typed_constraint_invalid_pair_control(case, run)
        )
    return result


def _run_typed_constraint_target_control(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_constraint_r0_control"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()

    with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
        distances = protocol["/constraint/default/pairs/r0"]
        if distances[...].tolist() != [1.5]:
            raise AssertionError(f"{case.name} target control lost baseline r0")
        distances[...] = [2.0]
    with h5py.File(control_dir / "restart.spgr.h5", "r+") as restart:
        position = restart["/particles/all/position/value"]
        if position.shape != (1, 2, 3) or not math.isclose(
            float(position[0, 1, 0]), 1.5, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise AssertionError(
                f"{case.name} target control lost baseline coordinate"
            )
        position[0, 1, 0] = 2.0

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    if outcome.returncode != 0:
        raise AssertionError(
            f"{case.name} typed r0 control failed with code "
            f"{outcome.returncode}\n{outcome.stdout}\n{outcome.stderr}"
        )
    control_result = _assert_constraint_projection_oracle(
        f"{case.name} typed r0 control",
        _read_native_float32_file(control_dir / "output" / "legacy.crd"),
        _read_native_float32_file(control_dir / "output" / "legacy.vel"),
        target_distance=2.0,
    )
    return {
        "baseline_target_distance": 1.5,
        "control_target_distance": 2.0,
        "control_initial_distance": 2.0,
        "target_delta": 0.5,
        "exit_code": outcome.returncode,
        **control_result,
    }


def _run_typed_constraint_invalid_pair_control(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    control_dir = run.bundled_dir.parent / "bundled_constraint_invalid_pair"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(run.bundled_dir, control_dir)
    output_dir = control_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("mdout.txt", "mdinfo.txt", "run.stdout", "run.stderr"):
        path = control_dir / file_name
        if path.exists():
            path.unlink()
    with h5py.File(control_dir / "protocol.spgp.h5", "r+") as protocol:
        protocol["/constraint/default/pairs/atoms"][...] = [[0, 2]]

    outcome = _run_sponge_process(control_dir, _mdin_name(control_dir))
    combined = outcome.stdout + outcome.stderr
    category = _failure_category(combined)
    expected_tokens = (
        "/constraint/default/pairs/atoms",
        "invalid pair [0, 2]",
        "row 0",
    )
    missing = [token for token in expected_tokens if token not in combined]
    if (
        outcome.returncode == 0
        or category != "spongeErrorBadFileFormat"
        or missing
    ):
        raise AssertionError(
            f"{case.name} invalid typed constraint was not rejected: "
            f"code={outcome.returncode}, category={category}, "
            f"missing={missing}\n{combined}"
        )
    return {
        "mutated_pair": [0, 2],
        "exit_code": outcome.returncode,
        "failure_category": category,
        "diagnostic_tokens": list(expected_tokens),
    }


def _assert_constraint_projection_oracle(
    label: str,
    positions: Sequence[float],
    velocities: Sequence[float],
    *,
    target_distance: float = 1.5,
    initial_relative_radial_speed: float = 2.0,
    box_length: float = 10.0,
) -> dict[str, object]:
    frame_width = 6
    if not positions or len(positions) != len(velocities):
        raise AssertionError(
            f"{label} constraint trajectory length mismatch: "
            f"position={len(positions)}, velocity={len(velocities)}"
        )
    if len(positions) % frame_width != 0:
        raise AssertionError(
            f"{label} constraint trajectory is not two-atom XYZ data"
        )
    if (
        not math.isfinite(initial_relative_radial_speed)
        or abs(initial_relative_radial_speed) < 1.0
    ):
        raise AssertionError(
            f"{label} initial radial motion is not non-trivial"
        )
    if not math.isfinite(box_length) or box_length <= 2.0 * target_distance:
        raise AssertionError(f"{label} periodic box is invalid: {box_length}")

    distance_residuals = []
    radial_velocity_residuals = []
    for offset in range(0, len(positions), frame_width):
        displacement = tuple(
            delta - round(delta / box_length) * box_length
            for delta in (
                positions[offset + 3 + axis] - positions[offset + axis]
                for axis in range(3)
            )
        )
        distance = math.sqrt(sum(value * value for value in displacement))
        if not math.isfinite(distance) or distance <= 0.0:
            raise AssertionError(
                f"{label} has invalid constrained distance at frame "
                f"{offset // frame_width}: {distance}"
            )
        relative_velocity = tuple(
            velocities[offset + 3 + axis] - velocities[offset + axis]
            for axis in range(3)
        )
        radial_velocity = sum(
            relative_velocity[axis] * displacement[axis] / distance
            for axis in range(3)
        )
        if not math.isfinite(radial_velocity):
            raise AssertionError(
                f"{label} has non-finite radial velocity at frame "
                f"{offset // frame_width}: {radial_velocity}"
            )
        distance_residuals.append(abs(distance - target_distance))
        radial_velocity_residuals.append(abs(radial_velocity))

    maximum_distance_residual = max(distance_residuals)
    maximum_radial_velocity_residual = max(radial_velocity_residuals)
    if maximum_distance_residual > 1.0e-5:
        raise AssertionError(
            f"{label} constraint distance residual exceeds tolerance: "
            f"{maximum_distance_residual}"
        )
    if maximum_radial_velocity_residual > 1.0e-4:
        raise AssertionError(
            f"{label} constraint radial velocity residual exceeds tolerance: "
            f"{maximum_radial_velocity_residual}"
        )
    return {
        "frame_count": len(positions) // frame_width,
        "periodic_box_length": box_length,
        "maximum_distance_residual": maximum_distance_residual,
        "maximum_radial_velocity_residual": maximum_radial_velocity_residual,
        "radial_speed_reduction_factor": (
            abs(initial_relative_radial_speed)
            / max(maximum_radial_velocity_residual, 1.0e-30)
        ),
    }


def _focused_virtual_atom_coordinate_oracle(
    fixture_case: str,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if fixture_case == FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE:
        return (
            (
                0.0,
                0.0,
                1.0,
                0.5,
                1.0,
                1.0,
                1.5,
                1.0,
                1.0,
                2.0,
                0.0,
                1.0,
                0.0,
                2.0,
                1.0,
                4.0,
                4.0,
                1.0,
                4.0,
                4.0,
                7.0,
                1.5,
                0.5,
                1.0,
            ),
            (1, 2, 6, 7),
        )
    if fixture_case in {
        FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
        FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    }:
        return (
            (
                9.5,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                9.75,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ),
            (2,),
        )
    raise AssertionError(
        f"unknown focused virtual-atom fixture: {fixture_case}"
    )


def _compare_focused_virtual_atoms_oracle(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    expected_coordinates, virtual_indices = (
        _focused_virtual_atom_coordinate_oracle(case.fixture_case)
    )
    branch_results = {}
    coordinates = {}
    forces = {}
    for branch, directory in (
        ("legacy", run.legacy_dir),
        ("bundled", run.bundled_dir),
    ):
        branch_coordinates = _read_native_float32_file(
            directory / "output" / "legacy.crd"
        )
        branch_forces = _read_native_float32_file(
            directory / "output" / "legacy.frc"
        )
        branch_results[branch] = _assert_virtual_atom_oracle(
            f"{case.name} {branch}",
            expected_coordinates,
            virtual_indices,
            branch_coordinates,
            branch_forces,
        )
        coordinates[branch] = branch_coordinates
        forces[branch] = branch_forces
    position_relative, position_absolute = _deterministic_tolerance("position")
    _assert_numeric_sequences_close(
        f"{case.name} virtual-atom coordinates",
        coordinates["legacy"],
        coordinates["bundled"],
        relative_tolerance=position_relative,
        absolute_tolerance=position_absolute,
    )
    force = _assert_nontrivial_equivalent_forces(
        f"{case.name} redistributed force",
        forces["legacy"],
        forces["bundled"],
    )
    return {
        "virtual_types": (
            [0, 1, 2, 3]
            if case.fixture_case == FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE
            else [1]
        ),
        "periodic_boundary_crossing": (
            case.fixture_case
            in {
                FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
                FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
            }
        ),
        "branches": branch_results,
        "cross_branch_force": force,
    }


def _assert_virtual_atom_oracle(
    label: str,
    expected_coordinates: Sequence[float],
    virtual_indices: Sequence[int],
    coordinates: Sequence[float],
    forces: Sequence[float],
) -> dict[str, object]:
    position_relative, position_absolute = _deterministic_tolerance("position")
    _assert_numeric_sequences_close(
        f"{label} coordinate oracle",
        expected_coordinates,
        coordinates,
        relative_tolerance=position_relative,
        absolute_tolerance=position_absolute,
    )
    if len(forces) != len(expected_coordinates):
        raise AssertionError(
            f"{label} force value count mismatch: "
            f"expected={len(expected_coordinates)}, actual={len(forces)}"
        )
    virtual_components = [
        forces[3 * atom_index + axis]
        for atom_index in virtual_indices
        for axis in range(3)
    ]
    virtual_set = set(virtual_indices)
    real_components = [
        value
        for atom_index in range(len(forces) // 3)
        if atom_index not in virtual_set
        for value in forces[3 * atom_index : 3 * atom_index + 3]
    ]
    if not any(
        math.isfinite(value) and abs(value) > 1.0e-8
        for value in real_components
    ):
        raise AssertionError(
            f"{label} redistributed real-atom force is all trivial"
        )
    if any(
        not math.isfinite(value) or abs(value) > 1.0e-6
        for value in virtual_components
    ):
        raise AssertionError(
            f"{label} retained force on a virtual atom: "
            f"maximum={max(abs(value) for value in virtual_components)}"
        )
    return {
        "coordinate_value_count": len(coordinates),
        "virtual_atom_indices": list(virtual_indices),
        "maximum_abs_real_force": max(abs(value) for value in real_components),
        "maximum_abs_virtual_force": max(
            abs(value) for value in virtual_components
        ),
    }


def _assert_exclusion_coulomb_oracle(
    label: str,
    mdout_rows: Sequence[dict[str, float]],
    forces: Sequence[float],
) -> dict[str, object]:
    if len(mdout_rows) != 1:
        raise AssertionError(f"{label} oracle requires exactly one mdout row")
    expected_energy = 1.0 / 4.0 - 1.0 / 3.0
    actual_energy = mdout_rows[0].get("eff_pot")
    if actual_energy is None:
        raise AssertionError(f"{label} oracle requires eff_pot")
    relative_tolerance, absolute_tolerance = _deterministic_tolerance(
        "observable"
    )
    if not math.isclose(
        actual_energy,
        expected_energy,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AssertionError(
            f"{label} exclusion energy mismatch: "
            f"expected={expected_energy}, actual={actual_energy}"
        )

    expected_forces = (
        -1.0 / 16.0,
        0.0,
        0.0,
        1.0 / 9.0,
        0.0,
        0.0,
        1.0 / 16.0 - 1.0 / 9.0,
        0.0,
        0.0,
    )
    force_relative, force_absolute = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        f"{label} exclusion force oracle",
        expected_forces,
        forces,
        relative_tolerance=force_relative,
        absolute_tolerance=force_absolute,
    )
    return {
        "expected_energy": expected_energy,
        "actual_energy": actual_energy,
        "unexcluded_energy": -1.0 + expected_energy,
        "force_value_count": len(expected_forces),
        "maximum_abs_force": max(abs(value) for value in forces),
    }


def _assert_nontrivial_equivalent_forces(
    label: str, legacy: Sequence[float], bundled: Sequence[float]
) -> dict[str, object]:
    for branch, values in (("legacy", legacy), ("bundled", bundled)):
        if not any(
            math.isfinite(value) and abs(value) > 1.0e-8 for value in values
        ):
            raise AssertionError(f"{label} {branch} force is all trivial")
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("force")
    _assert_numeric_sequences_close(
        label,
        legacy,
        bundled,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return {
        "route": "frc",
        "value_count": len(legacy),
        "legacy_max_abs": max(abs(value) for value in legacy),
        "bundled_max_abs": max(abs(value) for value in bundled),
    }


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


MDINFO_CONTRACT_KEYS = {
    "mode",
    "skin",
    "cutoff",
    "dt",
    "atom_numbers",
    "target temperature",
    "friction coefficient",
    "random seed",
    "residue_numbers",
    "fftx",
    "ffty",
    "fftz",
    "beta",
}


def _parse_mdinfo_key_values(path: Path) -> dict[str, list[str]]:
    parsed = {key: [] for key in MDINFO_CONTRACT_KEYS}
    pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9 _-]*?)"
        r"(?:\s+(?:is|set to)\s+|\s*:\s*)(.+?)\s*$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        key = " ".join(match.group(1).lower().split())
        if key in parsed:
            value = " ".join(match.group(2).split())
            if value not in parsed[key]:
                parsed[key].append(value)
    missing = sorted(key for key, values in parsed.items() if not values)
    if missing:
        raise AssertionError(
            f"mdinfo is missing structured keys {missing}: {path}"
        )
    return parsed


def _compare_mdinfo_structured(
    case: AbCase, runs: Sequence[AbRun]
) -> dict[str, object]:
    for run in runs:
        legacy = _parse_mdinfo_key_values(run.legacy_dir / "mdinfo.txt")
        bundled = _parse_mdinfo_key_values(run.bundled_dir / "mdinfo.txt")
        if legacy != bundled:
            differing = sorted(
                key
                for key in MDINFO_CONTRACT_KEYS
                if legacy[key] != bundled[key]
            )
            raise AssertionError(
                f"{case.name} replica {run.replica_index} structured mdinfo "
                f"differs for keys: {differing}"
            )
    return {
        "method": "structured_key_value",
        "keys": sorted(MDINFO_CONTRACT_KEYS),
        "replicas": len(runs),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restart_state_summary(state: dict[str, object]) -> dict[str, object]:
    values = {
        quantity: list(state[quantity])
        for quantity in ("position", "velocity", "box")
    }
    return {
        "atom_count": state["atom_count"],
        "step": state["step"],
        "time": state["time"],
        "value_counts": {
            quantity: len(payload) for quantity, payload in values.items()
        },
        "maximum_absolute_values": {
            quantity: max(map(abs, payload), default=0.0)
            for quantity, payload in values.items()
        },
    }


def _restart_continuation_source(
    run: AbRun, producer_branch: str, source_dir: Path
) -> tuple[dict[str, object], dict[str, object]]:
    expected_dirs = {
        "legacy": run.legacy_dir,
        "bundled": run.bundled_dir,
    }
    if producer_branch not in expected_dirs:
        raise AssertionError(
            f"unknown structural restart producer branch: {producer_branch}"
        )
    expected_dir = expected_dirs[producer_branch]
    if source_dir.resolve() != expected_dir.resolve():
        raise AssertionError(
            f"{producer_branch} structural restart producer source mismatch: "
            f"expected={expected_dir}, actual={source_dir}"
        )

    if producer_branch == "legacy":
        state = _read_legacy_restart_state(source_dir)
        source_paths = {
            "coordinate": source_dir / "output/legacy_restart_coordinate.txt",
            "velocity": source_dir / "output/legacy_restart_velocity.txt",
        }
        route = "legacy_coordinate_velocity_text"
    else:
        restart_path = source_dir / RESTART_REL
        positions = _h5_numeric_values(
            restart_path, "/particles/all/position/value"
        )
        velocities = _h5_numeric_values(
            restart_path, "/particles/all/velocity/value"
        )
        box = _h5_numeric_values(restart_path, "/particles/all/box/edges/value")
        steps = _h5_numeric_values(restart_path, "/particles/all/step")
        times = _h5_numeric_values(restart_path, "/particles/all/time")
        if len(steps) != 1 or len(times) != 1 or len(positions) % 3 != 0:
            raise AssertionError(
                "bundled structural restart must contain one complete state"
            )
        state = {
            "atom_count": len(positions) // 3,
            "step": int(steps[0]),
            "time": times[0],
            "position": positions,
            "velocity": velocities,
            "box": box,
        }
        source_paths = {"restart": restart_path}
        route = "bundled_h5_restart"

    missing = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing:
        raise AssertionError(
            f"{producer_branch} structural restart source is missing: {missing}"
        )
    evidence = {
        "producer_branch": producer_branch,
        "source_directory": str(source_dir.resolve()),
        "route": route,
        "sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
        "structural_payload": _restart_state_summary(state),
    }
    return evidence, state


def _assert_restart_source_states_close(
    case: AbCase, states: dict[str, dict[str, object]]
) -> None:
    legacy = states["legacy"]
    bundled = states["bundled"]
    if legacy["atom_count"] != bundled["atom_count"]:
        raise AssertionError(f"{case.name} producer restart atom count differs")
    if legacy["step"] != bundled["step"]:
        raise AssertionError(f"{case.name} producer restart step differs")
    _assert_numeric_sequences_close(
        f"{case.name} producer restart time",
        [legacy["time"]],
        [bundled["time"]],
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-12,
    )
    for quantity in ("position", "velocity", "box"):
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            quantity
        )
        _assert_numeric_sequences_close(
            f"{case.name} producer restart {quantity}",
            legacy[quantity],
            bundled[quantity],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=max(absolute_tolerance, 5.0e-5),
        )


def _compare_restart_continuation(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    continuation_trajectory = Path("output/continuation.spg.h5md")
    continuation_restart = Path("output/continuation.spgr.h5")
    sources = {}
    source_states = {}
    continuations = {}
    step_limit = 2
    for branch, source_dir, source_mdin in (
        ("legacy", run.legacy_dir, "mdin.spg.toml"),
        ("bundled", run.bundled_dir, "mdin.bundled.spg.toml"),
    ):
        source_evidence, source_state = _restart_continuation_source(
            run, branch, source_dir
        )
        sources[branch] = source_evidence
        source_states[branch] = source_state

        destination = source_dir.parent / f"continuation_{branch}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_dir, destination)
        for name, expected_sha256 in source_evidence["sha256"].items():
            copied_path = (
                destination / RESTART_REL
                if name == "restart"
                else destination / f"output/legacy_restart_{name}.txt"
            )
            if _sha256_file(copied_path) != expected_sha256:
                raise AssertionError(
                    f"{case.name} {branch} restart source changed while copied"
                )

        mdin_path = destination / source_mdin
        text = _remove_key_lines(
            mdin_path.read_text(encoding="utf-8"),
            {
                "mode",
                "thermostat",
                "thermostat_seed",
                "thermostat_tau",
                "target_temperature",
                "constrain_mode",
                "step_limit",
                "dt",
                "print_zeroth_frame",
                "write_mdout_interval",
                "write_trajectory_interval",
                "write_restart_file_interval",
                "coordinate_in_file",
                "velocity_in_file",
                "input_h5_restart_path",
                "input_h5_restart_load",
                "output_h5_trajectory_path",
                "output_h5_trajectory_vds",
                "output_h5_trajectory_chunk_size",
                "output_h5_restart_path",
                "output_h5_observable_path",
                "mdout",
                "mdinfo",
                "crd",
                "box",
                "vel",
                "frc",
                "rst",
            },
        )
        additions = [
            'mode = "nve"',
            f"step_limit = {step_limit}",
            "dt = 0.0001",
            "print_zeroth_frame = 1",
            "write_mdout_interval = 1",
            "write_trajectory_interval = 1",
            "write_restart_file_interval = 1",
            'mdout = "continuation.mdout"',
            'mdinfo = "continuation.mdinfo"',
            'crd = "output/continuation.crd"',
            'box = "output/continuation.box"',
            'vel = "output/continuation.vel"',
            'frc = "output/continuation.frc"',
            'rst = "output/continuation_restart"',
            f'output_h5_trajectory_path = "{continuation_trajectory.as_posix()}"',
            "output_h5_trajectory_vds = false",
            "output_h5_trajectory_chunk_size = 2",
            f'output_h5_restart_path = "{continuation_restart.as_posix()}"',
        ]
        if branch == "legacy":
            additions.extend(
                (
                    'coordinate_in_file = "output/legacy_restart_coordinate.txt"',
                    'velocity_in_file = "output/legacy_restart_velocity.txt"',
                )
            )
        else:
            additions.extend(
                (
                    'input_h5_restart_path = "output/ab.spgr.h5"',
                    'input_h5_restart_load = "structural"',
                )
            )
        mdin_path.write_text(
            text.rstrip() + "\n" + "\n".join(additions) + "\n",
            encoding="utf-8",
        )
        outcome = _run_sponge_process(destination, source_mdin)
        if outcome.returncode != 0:
            raise AssertionError(
                f"SPONGE continuation failed in {destination} with code "
                f"{outcome.returncode}\n[stdout]\n{outcome.stdout}\n"
                f"[stderr]\n{outcome.stderr}"
            )
        finite_scan = _scan_success_outputs(destination)
        trajectory_path = destination / continuation_trajectory
        restart_path = destination / continuation_restart
        if not trajectory_path.is_file() or not restart_path.is_file():
            raise AssertionError(
                f"{case.name} {branch} continuation outputs are incomplete"
            )
        _validate_restart_output(case.name, restart_path)
        continuations[branch] = {
            "mdout": _read_mdout(destination / "continuation.mdout"),
            "trajectory": trajectory_path,
            "restart": restart_path,
            "exit_code": outcome.returncode,
            "elapsed_s": outcome.elapsed_s,
            "finite_scan": finite_scan,
        }

    _assert_restart_source_states_close(case, source_states)
    columns = _require_matching_mdout_columns(
        continuations["legacy"]["mdout"],
        continuations["bundled"]["mdout"],
        f"{case.name} restart continuation",
    )
    legacy_rows = continuations["legacy"]["mdout"]["rows"]
    bundled_rows = continuations["bundled"]["mdout"]["rows"]
    if len(legacy_rows) != len(bundled_rows):
        raise AssertionError(
            f"{case.name} restart continuation row count differs"
        )
    for column in columns:
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            column
        )
        _assert_numeric_sequences_close(
            f"{case.name} restart continuation {column}",
            [row[column] for row in legacy_rows],
            [row[column] for row in bundled_rows],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )

    semantic_datasets = {
        "trajectory": (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/force/value",
            "/particles/all/box/edges/value",
        ),
        "restart": (
            "/particles/all/step",
            "/particles/all/time",
            "/particles/all/position/value",
            "/particles/all/velocity/value",
            "/particles/all/box/edges/value",
        ),
    }
    for family, datasets in semantic_datasets.items():
        legacy_path = continuations["legacy"][family]
        bundled_path = continuations["bundled"][family]
        for dataset in datasets:
            legacy_values = _h5_numeric_values(legacy_path, dataset)
            bundled_values = _h5_numeric_values(bundled_path, dataset)
            _assert_matching_numeric_shape(
                f"{case.name} continuation {family}:{dataset}",
                legacy_path,
                bundled_path,
                dataset,
                legacy_values,
                bundled_values,
            )
            relative_tolerance, absolute_tolerance = _deterministic_tolerance(
                dataset
            )
            _assert_numeric_sequences_close(
                f"{case.name} continuation {family}:{dataset}",
                legacy_values,
                bundled_values,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )

    return {
        "method": "branch_owned_restart_two_step_nve_continuation",
        "dt": 0.0001,
        "constraint_mode": "disabled_to_isolate_restart_loading",
        "sources": sources,
        "rows": len(legacy_rows),
        "columns": columns,
        "semantic_h5_datasets": {
            family: list(datasets)
            for family, datasets in semantic_datasets.items()
        },
        "finite_scans": {
            branch: continuations[branch]["finite_scan"]
            for branch in ("legacy", "bundled")
        },
        "end_status": {
            branch: {
                "exit_code": continuations[branch]["exit_code"],
                "elapsed_s": continuations[branch]["elapsed_s"],
            }
            for branch in ("legacy", "bundled")
        },
    }


def _compare_cmap_materialization(
    legacy_dir: Path, bundled_dir: Path
) -> dict[str, object]:
    legacy_stdout = (legacy_dir / "run.stdout").read_text(encoding="utf-8")
    bundled_stdout = (bundled_dir / "run.stdout").read_text(encoding="utf-8")
    if "START INITIALIZING CMAP" not in legacy_stdout:
        raise AssertionError("legacy run did not initialize CMAP")
    if "START INITIALIZING CMAP" not in bundled_stdout:
        raise AssertionError("bundled run did not initialize CMAP")

    topology = bundled_dir / "topology.spgt.h5"
    paths = _h5_paths(topology)
    required = {
        "/forcefield/cmap/atoms",
        "/forcefield/cmap/type",
        "/forcefield/cmap/resolution",
        "/forcefield/cmap/grid_value",
    }
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"bundled CMAP native H5 datasets missing: {missing}"
        )
    return {
        "legacy_initialized": True,
        "bundled_initialized": True,
        "bundled_required_datasets": sorted(required),
    }


def _read_mdout(path: Path) -> dict[str, object]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise AssertionError(f"mdout has no data rows: {path}")
    columns = lines[0].split()
    rows = []
    for raw in lines[1:]:
        fields = raw.split()
        if len(fields) != len(columns):
            continue
        row = {}
        for column, field in zip(columns, fields):
            row[column] = _parse_float(field)
        rows.append(row)
    if not rows:
        raise AssertionError(f"mdout has no parseable rows: {path}")
    return {"columns": columns, "rows": rows}


def _parse_float(value: str) -> float:
    normalized = value.lower().replace("-nan(ind)", "nan")
    normalized = normalized.replace("nan(ind)", "nan")
    return float(normalized)


def _observable_quantity(label: str) -> str:
    normalized = label.lower()
    for quantity, aliases in {
        "temperature": ("temperature", "temp"),
        "pressure": ("pressure", "press"),
        "density": ("density",),
        "position": ("position", "squared_displacement", "pair_distance"),
        "velocity": ("velocity", "speed"),
        "force": ("force",),
        "box_volume": ("volume",),
        "box_angle": ("angle",),
        "box_length": ("box", "length", "matrix"),
    }.items():
        if any(alias in normalized for alias in aliases):
            return quantity
    return "energy"


def _statistical_policy(
    label: str = "energy", *, inference_unit: str = "block"
) -> StatisticalEquivalencePolicy:
    limits = PROFILE_LIMITS[PROFILE]
    quantity = _observable_quantity(label)
    absolute_margin = PHYSICAL_ABSOLUTE_MARGINS[PROFILE].get(
        quantity, float(limits["normal_absolute_margin"])
    )
    return StatisticalEquivalencePolicy(
        burn_in_frames=int(limits["normal_burn_in_frames"]),
        block_size=int(limits["normal_block_size"]),
        minimum_blocks_per_replica=STATISTICAL_MINIMUM_BLOCKS_PER_REPLICA,
        confidence_z=STATISTICAL_CONFIDENCE_Z,
        relative_margin=float(limits["normal_relative_margin"]),
        absolute_margin=absolute_margin,
        maximum_std_ratio=STATISTICAL_MAXIMUM_STD_RATIO,
        inference_unit=inference_unit,
    )


def _holm_alpha(policy: StatisticalEquivalencePolicy) -> float:
    return 1.0 - normal_cdf(policy.confidence_z)


def _deterministic_tolerance(label: str) -> tuple[float, float]:
    normalized = label.lower()
    if any(token in normalized for token in ("step", "time", "frame")):
        return DETERMINISTIC_TOLERANCES["schedule"]
    for quantity in ("position", "velocity", "force", "box"):
        if quantity in normalized:
            return DETERMINISTIC_TOLERANCES[quantity]
    return DETERMINISTIC_TOLERANCES["observable"]


def _require_matching_mdout_columns(
    left: dict[str, object], right: dict[str, object], label: str
) -> list[str]:
    left_columns = list(left["columns"])
    right_columns = list(right["columns"])
    if left_columns != right_columns:
        raise AssertionError(
            f"{label} mdout columns differ: legacy={left_columns}, "
            f"bundled={right_columns}"
        )
    if not left_columns:
        raise AssertionError(f"{label} has no mdout columns")
    return left_columns


def _same_nonfinite_value(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isinf(left) and math.isinf(right) and left == right


def _assert_numeric_sequences_close(
    label: str,
    left: Sequence[float],
    right: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if len(left) != len(right):
        raise AssertionError(
            f"{label} length mismatch: legacy={len(left)}, bundled={len(right)}"
        )
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if not _same_nonfinite_value(left_value, right_value):
                raise AssertionError(
                    f"{label} non-finite mismatch at index {index}: "
                    f"legacy={left_value}, bundled={right_value}"
                )
            raise AssertionError(
                f"{label} normal-success comparison contains matching "
                f"non-finite values at index {index}: {left_value}"
            )
        if not math.isclose(
            left_value,
            right_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(
                f"{label} mismatch at index {index}: legacy={left_value}, "
                f"bundled={right_value}"
            )


def _assert_periodic_positions_close(
    label: str,
    left: Sequence[float],
    right: Sequence[float],
    shape: tuple[int, ...],
    boxes: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if len(shape) != 3 or shape[2] != 3:
        raise AssertionError(
            f"{label} requires a frame/atom/xyz position shape"
        )
    frame_count, atom_count, _ = shape
    if len(left) != len(right) or len(left) != frame_count * atom_count * 3:
        raise AssertionError(f"{label} position value count differs from shape")
    if len(boxes) not in {9, frame_count * 9}:
        raise AssertionError(
            f"{label} box value count differs from frame count"
        )

    for frame_index in range(frame_count):
        box_offset = 0 if len(boxes) == 9 else frame_index * 9
        box = boxes[box_offset : box_offset + 9]
        inverse = _inverse_3x3(box, label)
        for atom_index in range(atom_count):
            offset = (frame_index * atom_count + atom_index) * 3
            left_xyz = left[offset : offset + 3]
            right_xyz = right[offset : offset + 3]
            if any(
                not math.isfinite(value) for value in (*left_xyz, *right_xyz)
            ):
                _assert_nonfinite_patterns_match(
                    f"{label} frame {frame_index} atom {atom_index}",
                    left_xyz,
                    right_xyz,
                )
                continue
            delta = [left_xyz[axis] - right_xyz[axis] for axis in range(3)]
            fractional = [
                sum(
                    delta[axis] * inverse[axis * 3 + lattice]
                    for axis in range(3)
                )
                for lattice in range(3)
            ]
            lattice = [round(value) for value in fractional]
            shift = [
                sum(
                    lattice[basis] * box[basis * 3 + axis] for basis in range(3)
                )
                for axis in range(3)
            ]
            for axis in range(3):
                adjusted = left_xyz[axis] - shift[axis]
                if not math.isclose(
                    adjusted,
                    right_xyz[axis],
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                ):
                    raise AssertionError(
                        f"{label} periodic mismatch at frame {frame_index}, "
                        f"atom {atom_index}, axis {axis}: "
                        f"legacy={left_xyz[axis]}, bundled={right_xyz[axis]}, "
                        f"lattice_shift={shift[axis]}"
                    )


def _inverse_3x3(values: Sequence[float], label: str) -> tuple[float, ...]:
    if len(values) != 9:
        raise AssertionError(f"{label} box matrix must have nine values")
    a, b, c, d, e, f, g, h, i = values
    determinant = (
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    )
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-20:
        raise AssertionError(f"{label} box matrix is singular")
    scale = 1.0 / determinant
    return (
        (e * i - f * h) * scale,
        (c * h - b * i) * scale,
        (b * f - c * e) * scale,
        (f * g - d * i) * scale,
        (a * i - c * g) * scale,
        (c * d - a * f) * scale,
        (d * h - e * g) * scale,
        (b * g - a * h) * scale,
        (a * e - b * d) * scale,
    )


def _assert_nonfinite_patterns_match(
    label: str, left: Sequence[float], right: Sequence[float]
) -> None:
    if len(left) != len(right):
        raise AssertionError(
            f"{label} length mismatch: legacy={len(left)}, bundled={len(right)}"
        )
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if math.isfinite(left_value) and math.isfinite(right_value):
            continue
        if not _same_nonfinite_value(left_value, right_value):
            raise AssertionError(
                f"{label} non-finite mismatch at index {index}: "
                f"legacy={left_value}, bundled={right_value}"
            )


def _compare_mdout_statistically(
    case: AbCase,
    runs: Sequence[AbRun],
    *,
    inference_unit: str = "block",
) -> dict[str, object]:
    parsed = [
        (
            _read_mdout(run.legacy_dir / "mdout.txt"),
            _read_mdout(run.bundled_dir / "mdout.txt"),
        )
        for run in runs
    ]
    columns = _require_matching_mdout_columns(
        parsed[0][0], parsed[0][1], f"{case.name} replica 0"
    )
    for replica_index, (left, right) in enumerate(parsed):
        replica_columns = _require_matching_mdout_columns(
            left, right, f"{case.name} replica {replica_index}"
        )
        if replica_columns != columns:
            raise AssertionError(
                f"{case.name} mdout columns changed between replicas: "
                f"first={columns}, replica_{replica_index}={replica_columns}"
            )
        if len(left["rows"]) != len(right["rows"]):
            raise AssertionError(
                f"{case.name} replica {replica_index} mdout row count mismatch: "
                f"legacy={len(left['rows'])}, bundled={len(right['rows'])}"
            )

    policy = _statistical_policy(inference_unit=inference_unit)
    comparison: dict[str, object] = {
        "method": "independent_replicas_block_mean_equivalence",
        "replicas": len(runs),
        "policy": {
            "burn_in_frames": policy.burn_in_frames,
            "block_size": policy.block_size,
            "minimum_blocks_per_replica": policy.minimum_blocks_per_replica,
            "confidence_z": policy.confidence_z,
            "relative_margin": policy.relative_margin,
            "absolute_margin": policy.absolute_margin,
            "maximum_std_ratio": policy.maximum_std_ratio,
        },
        "columns": {},
    }
    finite_results: dict[str, dict[str, float | int]] = {}
    for column in columns:
        legacy_replicas = [
            [row[column] for row in left["rows"]] for left, _ in parsed
        ]
        bundled_replicas = [
            [row[column] for row in right["rows"]] for _, right in parsed
        ]
        if column in {"step", "frame", "time"}:
            for replica_index, (legacy, bundled) in enumerate(
                zip(legacy_replicas, bundled_replicas)
            ):
                _assert_numeric_sequences_close(
                    f"{case.name} mdout {column} replica {replica_index}",
                    legacy,
                    bundled,
                    relative_tolerance=0.0,
                    absolute_tolerance=1.0e-12,
                )
            comparison["columns"][column] = {"method": "exact_schedule"}
            continue

        all_values = [
            value
            for replica in (*legacy_replicas, *bundled_replicas)
            for value in replica
        ]
        if any(not math.isfinite(value) for value in all_values):
            for replica_index, (legacy, bundled) in enumerate(
                zip(legacy_replicas, bundled_replicas)
            ):
                _assert_nonfinite_patterns_match(
                    f"{case.name} mdout {column} replica {replica_index}",
                    legacy,
                    bundled,
                )
            finite_legacy = [
                [value for value in replica if math.isfinite(value)]
                for replica in legacy_replicas
            ]
            finite_bundled = [
                [value for value in replica if math.isfinite(value)]
                for replica in bundled_replicas
            ]
            nonfinite_result: dict[str, object] = {
                "method": "exact_nonfinite_pattern",
                "nonfinite_count": sum(
                    not math.isfinite(value) for value in all_values
                ),
            }
            column_policy = _statistical_policy(
                column, inference_unit=inference_unit
            )
            if finite_legacy[0] and _can_use_statistics(
                finite_legacy, column_policy
            ):
                finite_result = compare_replicas(
                    f"{case.name} mdout {column} finite values",
                    finite_legacy,
                    finite_bundled,
                    column_policy,
                )
                nonfinite_result["finite_values"] = finite_result
                finite_results[column] = finite_result
            comparison["columns"][column] = nonfinite_result
            continue

        column_policy = _statistical_policy(
            column, inference_unit=inference_unit
        )
        result = compare_replicas(
            f"{case.name} mdout {column}",
            legacy_replicas,
            bundled_replicas,
            column_policy,
        )
        comparison["columns"][column] = result
        finite_results[column] = result
    if finite_results:
        holm_correct_equivalence_family(
            f"{case.name} mdout observable family",
            finite_results,
            alpha=_holm_alpha(policy),
        )
    return comparison


def _compare_mdout_deterministically(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    left = _read_mdout(run.legacy_dir / "mdout.txt")
    right = _read_mdout(run.bundled_dir / "mdout.txt")
    columns = _require_matching_mdout_columns(left, right, case.name)
    left_rows = left["rows"]
    right_rows = right["rows"]
    if len(left_rows) != len(right_rows):
        raise AssertionError(
            f"{case.name} mdout row count mismatch: legacy={len(left_rows)}, "
            f"bundled={len(right_rows)}"
        )
    maximum_abs_error = 0.0
    for column in columns:
        legacy_values = [row[column] for row in left_rows]
        bundled_values = [row[column] for row in right_rows]
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            column
        )
        _assert_numeric_sequences_close(
            f"{case.name} deterministic mdout {column}",
            legacy_values,
            bundled_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for legacy_value, bundled_value in zip(legacy_values, bundled_values):
            if math.isfinite(legacy_value) and math.isfinite(bundled_value):
                maximum_abs_error = max(
                    maximum_abs_error, abs(legacy_value - bundled_value)
                )
    return {
        "method": "deterministic_all_rows_all_columns",
        "rows": len(left_rows),
        "columns": columns,
        "maximum_abs_error": maximum_abs_error,
    }


H5_OBSERVABLE_REQUIRED_PATHS = {
    "/observables/all/step",
    "/observables/all/time",
    "/parameters/sponge/mdout/columns/original_name",
    "/parameters/sponge/mdout/columns/hdf5_name",
}


def _validate_full_contract_input(
    case: AbCase, bundled_dir: Path
) -> dict[str, object]:
    if case.fixture_case != "full_contract_rerun":
        return {"file_count": 0, "path_count": 0, "has_sidecars": False}
    all_paths: set[str] = set()
    for file_name, required_paths in FULL_CONTRACT_INPUT_REQUIRED_PATHS.items():
        file_path = bundled_dir / file_name
        paths = _h5_paths(file_path)
        effective_required_paths = set(required_paths)
        if file_name == "topology.spgt.h5" and "/manybody/eam" not in paths:
            effective_required_paths.discard("/manybody/eam/atom_type")
        if (
            file_name == "protocol.spgp.h5"
            and "/restraint/default" not in paths
        ):
            effective_required_paths.discard("/restraint/default/atom_indices")
            effective_required_paths.discard("/restraint/default/weight")
        if file_name == "protocol.spgp.h5" and "/wall/soft" not in paths:
            effective_required_paths.discard("/wall/soft/potential")
        if file_name == "protocol.spgp.h5" and "/restraint/config" not in paths:
            effective_required_paths.discard("/restraint/config/section/name")
            effective_required_paths.discard(
                "/restraint/cv/config/section/name"
            )
        if file_name == "restart.spgr.h5":
            reference_path = (
                "/parameters/restart/references/restraint/default/coordinate"
            )
            if reference_path not in paths:
                effective_required_paths.discard(reference_path)
        missing = sorted(effective_required_paths - paths)
        if missing:
            raise AssertionError(
                f"{case.name} full-contract input {file_name} is missing "
                f"required paths: {missing}"
            )
        all_paths.update(paths)

    topology_path = bundled_dir / "topology.spgt.h5"
    sidecar_root = "/parameters/sponge/files/legacy_sidecars"
    with h5py.File(topology_path, "r") as topology:
        typed_paths = ("/atoms/residue_index", "/residues/atom_offset")
        typed_owner = all(path in topology for path in typed_paths)
        partial_typed_owner = any(path in topology for path in typed_paths)
        sidecar_owner = False
        if sidecar_root in topology:
            sidecar_keys = topology[f"{sidecar_root}/key"].asstr()[...]
            sidecar_owner = "residue_in_file" in sidecar_keys
        typed_nb14_owner = all(
            path in topology
            for path in (
                "/forcefield/nb14_extra/atoms",
                "/forcefield/nb14_extra/params",
            )
        )
        partial_typed_nb14_owner = any(
            path in topology
            for path in (
                "/forcefield/nb14_extra/atoms",
                "/forcefield/nb14_extra/params",
            )
        )
        sidecar_nb14_owner = (
            sidecar_root in topology
            and "nb14_extra_in_file"
            in topology[f"{sidecar_root}/key"].asstr()[...]
        )
        typed_eam_owner = "/manybody/eam" in topology
        sidecar_eam_owner = (
            sidecar_root in topology
            and "EAM_in_file" in topology[f"{sidecar_root}/key"].asstr()[...]
        )
    restraint_protocol_paths = (
        "/restraint/default/atom_indices",
        "/restraint/default/weight",
    )
    restraint_reference_path = (
        "/parameters/restart/references/restraint/default/coordinate"
    )
    restraint_sidecar_keys = {
        "restrain_atom_id",
        "restrain_weight_in_file",
        "restrain_coordinate_in_file",
    }
    with (
        h5py.File(bundled_dir / "protocol.spgp.h5", "r") as protocol,
        h5py.File(bundled_dir / "restart.spgr.h5", "r") as restart,
    ):
        typed_restraint_parts = [
            path in protocol for path in restraint_protocol_paths
        ]
        typed_restraint_parts.append(restraint_reference_path in restart)
        protocol_sidecar_keys: set[str] = set()
        if sidecar_root in protocol:
            protocol_sidecar_keys = set(
                protocol[f"{sidecar_root}/key"].asstr()[...]
            )
        if sidecar_root in restart:
            protocol_sidecar_keys.update(
                restart[f"{sidecar_root}/key"].asstr()[...]
            )
        sidecar_restraint_parts = [
            key in protocol_sidecar_keys for key in restraint_sidecar_keys
        ]
        soft_wall_typed_paths = (
            "/wall/soft/count",
            "/wall/soft/name",
            "/wall/soft/potential",
        )
        typed_soft_wall_parts = [
            path in protocol for path in soft_wall_typed_paths
        ]
        sidecar_soft_wall_owner = "soft_walls_in_file" in protocol_sidecar_keys
        cv_restraint_typed_paths = (
            "/restraint/config",
            "/restraint/cv/config",
        )
        typed_cv_restraint_parts = [
            path in protocol for path in cv_restraint_typed_paths
        ]
        cv_restraint_sidecar_keys = (
            "restrain_in_file",
            "restrain_cv_in_file",
        )
        sidecar_cv_restraint_parts = [
            key in protocol_sidecar_keys for key in cv_restraint_sidecar_keys
        ]
    typed_restraint_owner = all(typed_restraint_parts)
    partial_typed_restraint_owner = any(typed_restraint_parts)
    sidecar_restraint_owner = all(sidecar_restraint_parts)
    partial_sidecar_restraint_owner = any(sidecar_restraint_parts)
    typed_soft_wall_owner = all(typed_soft_wall_parts)
    partial_typed_soft_wall_owner = any(typed_soft_wall_parts)
    typed_cv_restraint_owner = all(typed_cv_restraint_parts)
    partial_typed_cv_restraint_owner = any(typed_cv_restraint_parts)
    sidecar_cv_restraint_owner = all(sidecar_cv_restraint_parts)
    partial_sidecar_cv_restraint_owner = any(sidecar_cv_restraint_parts)
    if partial_typed_owner and not typed_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete typed residue "
            "owner"
        )
    if typed_owner == sidecar_owner:
        owner_count = int(typed_owner) + int(sidecar_owner)
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one residue "
            f"owner, found {owner_count}"
        )
    if partial_typed_nb14_owner and not typed_nb14_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete typed "
            "nb14_extra owner"
        )
    if typed_nb14_owner == sidecar_nb14_owner:
        owner_count = int(typed_nb14_owner) + int(sidecar_nb14_owner)
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one NB14 "
            f"owner, found {owner_count}"
        )
    if typed_eam_owner == sidecar_eam_owner:
        owner_count = int(typed_eam_owner) + int(sidecar_eam_owner)
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one EAM "
            f"owner, found {owner_count}"
        )
    if partial_typed_restraint_owner and not typed_restraint_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete typed "
            "positional-restraint owner"
        )
    if partial_sidecar_restraint_owner and not sidecar_restraint_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete sidecar "
            "positional-restraint owner"
        )
    if typed_restraint_owner == sidecar_restraint_owner:
        owner_count = int(typed_restraint_owner) + int(sidecar_restraint_owner)
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one positional "
            f"restraint owner, found {owner_count}"
        )
    if partial_typed_soft_wall_owner and not typed_soft_wall_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete typed "
            "soft-wall owner"
        )
    if typed_soft_wall_owner == sidecar_soft_wall_owner:
        owner_count = int(typed_soft_wall_owner) + int(sidecar_soft_wall_owner)
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one soft-wall "
            f"owner, found {owner_count}"
        )
    if partial_typed_cv_restraint_owner and not typed_cv_restraint_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete typed "
            "CV-restraint owner"
        )
    if partial_sidecar_cv_restraint_owner and not sidecar_cv_restraint_owner:
        raise AssertionError(
            f"{case.name} full-contract input has an incomplete sidecar "
            "CV-restraint owner"
        )
    if typed_cv_restraint_owner == sidecar_cv_restraint_owner:
        owner_count = int(typed_cv_restraint_owner) + int(
            sidecar_cv_restraint_owner
        )
        raise AssertionError(
            f"{case.name} full-contract input requires exactly one "
            f"CV-restraint owner, found {owner_count}"
        )

    has_sidecars = any("legacy_sidecars" in path for path in all_paths)
    if "input.full_contract.pure_native" in case.contract_ids and has_sidecars:
        raise AssertionError(
            f"{case.name} pure bundle retains legacy sidecar tables"
        )
    if "input.full_contract.sidecar" in case.contract_ids and not has_sidecars:
        raise AssertionError(
            f"{case.name} sidecar bundle has no legacy sidecar tables"
        )
    return {
        "file_count": len(FULL_CONTRACT_INPUT_REQUIRED_PATHS),
        "path_count": len(all_paths),
        "has_sidecars": has_sidecars,
        "validation": "required_paths_present",
    }


def _output_h5_files(case: AbCase, root: Path) -> dict[str, Path]:
    if case.output_families:
        enabled = set(case.output_families)
    else:
        enabled = {"trajectory", "observable"}
        if case.mode in {
            "normal",
            "dynamic_continuation",
            "protocol_full_continuation",
        }:
            enabled.add("restart")
    paths = {
        "trajectory": root / TRAJECTORY_REL,
        "observable": root / OBSERVABLE_REL,
        "restart": root / RESTART_REL,
    }
    return {family: paths[family] for family in paths if family in enabled}


def _validate_branch_output_contract(
    case: AbCase, case_dir: Path, *, branch: str
) -> dict[str, object]:
    if branch not in {"legacy", "bundled"}:
        raise AssertionError(f"unknown A/B branch: {branch}")
    if case.input_behavior_only:
        mdout = case_dir / "mdout.txt"
        force = case_dir / "output" / "legacy.frc"
        if not mdout.exists() or not force.exists():
            raise AssertionError(
                f"{case.name} {branch} input behavior artifacts are missing"
            )
        return {
            "scope": "input_behavior_only",
            "mdout_rows": len(_read_mdout(mdout)["rows"]),
            "legacy_force_value_count": len(_read_native_float32_file(force)),
        }
    if case.mode == "rerun" and branch == "legacy":
        if not (case_dir / "mdout.txt").exists():
            raise AssertionError(f"{case.name} legacy rerun did not emit mdout")
        expected = {
            "position": case_dir / "output/legacy_rerun.crd",
            "box": case_dir / "output/legacy_rerun.box",
        }
        if case.rerun_velocity_present:
            expected["velocity"] = case_dir / "output/legacy_rerun.vel"
        if case.rerun_force_output:
            expected["force"] = case_dir / "output/legacy_rerun.frc"
        missing = sorted(
            name for name, path in expected.items() if not path.exists()
        )
        if missing:
            raise AssertionError(
                f"{case.name} legacy rerun outputs are missing: {missing}"
            )
        return {
            "direct_output_fields": sorted(expected),
            "mdout_rows": len(_read_mdout(case_dir / "mdout.txt")["rows"]),
        }
    files = _output_h5_files(case, case_dir)
    summary: dict[str, object] = {}
    for name, path in files.items():
        if not path.exists():
            raise AssertionError(
                f"{case.name} did not emit {name} H5 output: {path}"
            )
        summary[name] = {
            "dataset_count": len(_h5_dataset_paths(path)),
            "path_count": len(_h5_paths(path)),
            "bytes": _file_size(path),
        }

    _validate_trajectory_output(case, files["trajectory"], case_dir)
    if case.mode == "normal":
        summary["particle_legacy_coexistence"] = (
            _validate_particle_legacy_coexistence(case, case_dir, files)
        )
    observable_summary = _validate_observable_output(
        case.name, files["observable"], case_dir / "mdout.txt"
    )
    summary["observable"].update(observable_summary)
    if "restart" in files:
        _validate_restart_output(case.name, files["restart"])
        if case.mode == "normal":
            summary["restart_legacy_coexistence"] = (
                _validate_restart_legacy_coexistence(
                    case, case_dir, files["restart"]
                )
            )
    trajectory_frame_count = int(
        _h5_numeric_values(
            files["trajectory"], "/parameters/sponge/output/frame_count"
        )[-1]
    )
    if (
        case.vds
        and trajectory_frame_count > 0
        and _vds_shard_count(files["trajectory"]) <= 0
    ):
        raise AssertionError(
            f"{case.name} VDS run did not create trajectory shards"
        )
    return summary


def _read_native_float32_file(path: Path) -> list[float]:
    payload = path.read_bytes()
    if not payload or len(payload) % 4 != 0:
        raise AssertionError(
            f"legacy float32 output must be non-empty and 4-byte aligned: {path}"
        )
    return [value[0] for value in struct.iter_unpack("=f", payload)]


def _read_legacy_restart_state(case_dir: Path) -> dict[str, object]:
    coordinate_path = case_dir / "output/legacy_restart_coordinate.txt"
    velocity_path = case_dir / "output/legacy_restart_velocity.txt"
    coordinate_lines = coordinate_path.read_text(encoding="utf-8").splitlines()
    velocity_lines = velocity_path.read_text(encoding="utf-8").splitlines()
    coordinate_header = coordinate_lines[0].split()
    velocity_header = velocity_lines[0].split()
    if coordinate_header != velocity_header or len(coordinate_header) != 3:
        raise AssertionError(
            "legacy restart coordinate/velocity headers differ"
        )
    atom_count = int(coordinate_header[0])
    if (
        len(coordinate_lines) != atom_count + 2
        or len(velocity_lines) != atom_count + 1
    ):
        raise AssertionError(
            "legacy restart row count does not match atom count"
        )
    positions = [
        float(value)
        for line in coordinate_lines[1 : atom_count + 1]
        for value in line.split()
    ]
    velocities = [
        float(value)
        for line in velocity_lines[1 : atom_count + 1]
        for value in line.split()
    ]
    box_fields = [float(value) for value in coordinate_lines[-1].split()]
    if len(box_fields) != 6:
        raise AssertionError("legacy restart box row must have six values")
    box = [
        box_fields[0],
        0.0,
        0.0,
        0.0,
        box_fields[1],
        0.0,
        0.0,
        0.0,
        box_fields[2],
    ]
    return {
        "atom_count": atom_count,
        "time": float(coordinate_header[1]),
        "step": int(coordinate_header[2]),
        "position": positions,
        "velocity": velocities,
        "box": box,
    }


def _validate_restart_legacy_coexistence(
    case: AbCase, case_dir: Path, restart_path: Path
) -> dict[str, object]:
    legacy = _read_legacy_restart_state(case_dir)
    comparisons = {
        "position": "/particles/all/position/value",
        "velocity": "/particles/all/velocity/value",
        "box": "/particles/all/box/edges/value",
    }
    for quantity, dataset in comparisons.items():
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            quantity
        )
        absolute_tolerance = max(absolute_tolerance, 5.0e-5)
        _assert_numeric_sequences_close(
            f"{case.name} legacy/H5 restart {quantity}",
            legacy[quantity],
            _h5_numeric_values(restart_path, dataset),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    steps = _h5_numeric_values(restart_path, "/particles/all/step")
    times = _h5_numeric_values(restart_path, "/particles/all/time")
    if steps != [legacy["step"]] or times != [legacy["time"]]:
        raise AssertionError(f"{case.name} legacy/H5 restart schedule differs")
    keys = _h5_string_values(
        restart_path, "/parameters/sponge/files/legacy_sidecars/key"
    )
    paths = _h5_string_values(
        restart_path, "/parameters/sponge/files/legacy_sidecars/path"
    )
    if dict(zip(keys, paths)).get("rst") != "output/legacy_restart":
        raise AssertionError(f"{case.name} restart provenance is missing rst")
    return {
        "method": "parsed_structural_state",
        "atom_count": legacy["atom_count"],
        "step": legacy["step"],
        "time": legacy["time"],
    }


def _validate_particle_legacy_coexistence(
    case: AbCase, case_dir: Path, files: dict[str, Path]
) -> dict[str, object]:
    trajectory = files["trajectory"]
    provenance_keys = _h5_string_values(
        trajectory, "/parameters/sponge/files/legacy_sidecars/key"
    )
    provenance_paths = _h5_string_values(
        trajectory, "/parameters/sponge/files/legacy_sidecars/path"
    )
    provenance = dict(zip(provenance_keys, provenance_paths))
    compared = {}
    for route, relative_path, dataset in (
        ("crd", "output/legacy.crd", "/particles/all/position/value"),
        ("vel", "output/legacy.vel", "/particles/all/velocity/value"),
        ("frc", "output/legacy.frc", "/particles/all/force/value"),
    ):
        if provenance.get(route) != relative_path:
            raise AssertionError(
                f"{case.name} {route} provenance mismatch: {provenance.get(route)!r}"
            )
        legacy_values = _read_native_float32_file(case_dir / relative_path)
        h5_values = _h5_numeric_values(trajectory, dataset)
        if route == "frc" and len(legacy_values) != len(h5_values):
            shape = _h5_dataset_shape(trajectory, dataset)
            frame_width = math.prod(shape[1:])
            h5_steps = _h5_numeric_values(trajectory, "/particles/all/step")
            raise AssertionError(
                f"{case.name} legacy/H5 force schedule mismatch: "
                f"legacy_frames={len(legacy_values) // frame_width}, "
                f"h5_frames={len(h5_values) // frame_width}, "
                f"h5_first_step={h5_steps[0]}, h5_last_step={h5_steps[-1]}"
            )
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            dataset
        )
        _assert_numeric_sequences_close(
            f"{case.name} legacy {route} coexistence",
            legacy_values,
            h5_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        compared[route] = {
            "value_count": len(legacy_values),
            "h5_dataset": dataset,
            "provenance_path": relative_path,
        }
    box_relative_path = "output/legacy.box"
    if provenance.get("box") != box_relative_path:
        raise AssertionError(
            f"{case.name} box provenance mismatch: {provenance.get('box')!r}"
        )
    legacy_box_values = []
    for line in (
        (case_dir / box_relative_path).read_text(encoding="utf-8").splitlines()
    ):
        fields = [float(field) for field in line.split()]
        if len(fields) != 6:
            raise AssertionError(
                f"{case.name} legacy box row must have 6 values"
            )
        legacy_box_values.extend(
            (fields[0], 0.0, 0.0, 0.0, fields[1], 0.0, 0.0, 0.0, fields[2])
        )
    h5_box_values = _h5_numeric_values(
        trajectory, "/particles/all/box/edges/value"
    )
    relative_tolerance, absolute_tolerance = _deterministic_tolerance("box")
    _assert_numeric_sequences_close(
        f"{case.name} legacy box coexistence",
        legacy_box_values,
        h5_box_values,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    compared["box"] = {
        "value_count": len(legacy_box_values),
        "h5_dataset": "/particles/all/box/edges/value",
        "provenance_path": box_relative_path,
    }
    return compared


def _validate_trajectory_output(
    case: AbCase, path: Path, case_dir: Path
) -> None:
    paths = _h5_paths(path)
    expected_output_frames: int | None = None
    if case.mode == "rerun":
        input_trajectory = case_dir / case.trajectory_file_name
        stream_root = f"/particles/{case.trajectory_particle_stream}"
        input_steps = _h5_numeric_values(
            input_trajectory, f"{stream_root}/step"
        )
        selected = _expected_rerun_frame_indices(case, len(input_steps))
        expected_output_frames = max(0, len(selected) - 1)
        if expected_output_frames == 0:
            frame_counts = _h5_numeric_values(
                path, "/parameters/sponge/output/frame_count"
            )
            if frame_counts != [0.0]:
                raise AssertionError(
                    f"{case.name} empty rerun trajectory has invalid completion "
                    f"frame count: {frame_counts}"
                )
            return

    required = set(H5_COMPARE_DATASETS)
    optional_expectations = {
        "/particles/all/velocity/value": (
            case.mode == "normal"
            or (case.mode == "rerun" and case.rerun_velocity_present)
        ),
        "/particles/all/force/value": (
            case.mode == "normal"
            or (case.mode == "rerun" and case.rerun_force_output)
        ),
    }
    for dataset, expected in optional_expectations.items():
        if not expected:
            required.discard(dataset)
            if dataset in paths:
                raise AssertionError(
                    f"{case.name} unexpectedly retained optional {dataset}"
                )
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"{case.name} trajectory output is missing required datasets: {missing}"
        )
    steps = _h5_numeric_values(path, "/particles/all/step")
    times = _h5_numeric_values(path, "/particles/all/time")
    frame_counts = _h5_numeric_values(
        path, "/parameters/sponge/output/frame_count"
    )
    if len(steps) != len(times):
        raise AssertionError(
            f"{case.name} trajectory timeline is inconsistent: "
            f"steps={len(steps)}, times={len(times)}"
        )
    if expected_output_frames is not None:
        if len(steps) != expected_output_frames:
            raise AssertionError(
                f"{case.name} rerun trajectory frame count mismatch: "
                f"expected={expected_output_frames}, actual={len(steps)}"
            )
    elif not steps:
        raise AssertionError(f"{case.name} trajectory timeline is empty")
    if not frame_counts or int(frame_counts[-1]) != len(steps):
        raise AssertionError(
            f"{case.name} trajectory completion frame count does not match "
            f"timeline: completion={frame_counts}, frames={len(steps)}"
        )
    for dataset in (
        "/particles/all/position/value",
        "/particles/all/box/edges/value",
    ):
        shape = _h5_dataset_shape(path, dataset)
        if not shape or shape[0] != len(steps):
            raise AssertionError(
                f"{case.name} trajectory {dataset} frame shape mismatch: "
                f"shape={shape}, frames={len(steps)}"
            )
    if case.mode in {"normal", "rerun"}:
        for dataset, expected in optional_expectations.items():
            if not expected:
                continue
            shape = _h5_dataset_shape(path, dataset)
            if not shape or shape[0] != len(steps):
                raise AssertionError(
                    f"{case.name} trajectory {dataset} frame shape mismatch: "
                    f"shape={shape}, frames={len(steps)}"
                )


def _validate_restart_output(case_name: str, path: Path) -> None:
    paths = _h5_paths(path)
    missing = sorted(set(RESTART_COMPARE_DATASETS) - paths)
    if missing:
        raise AssertionError(
            f"{case_name} restart output is missing required datasets: {missing}"
        )
    for dataset in RESTART_COMPARE_DATASETS:
        values = _h5_numeric_values(path, dataset)
        if not values:
            raise AssertionError(
                f"{case_name} restart output is empty: {dataset}"
            )


def _validate_observable_output(
    case_name: str, path: Path, mdout_path: Path
) -> dict[str, object]:
    paths = _h5_paths(path)
    missing = sorted(H5_OBSERVABLE_REQUIRED_PATHS - paths)
    if missing:
        raise AssertionError(
            f"{case_name} observable output is missing required paths: {missing}"
        )
    original_names = _h5_string_values(
        path, "/parameters/sponge/mdout/columns/original_name"
    )
    hdf5_names = _h5_string_values(
        path, "/parameters/sponge/mdout/columns/hdf5_name"
    )
    if len(original_names) != len(hdf5_names) or not original_names:
        raise AssertionError(
            f"{case_name} observable column mapping is invalid: "
            f"original={original_names}, hdf5={hdf5_names}"
        )
    if len(set(hdf5_names)) != len(hdf5_names):
        raise AssertionError(
            f"{case_name} observable HDF5 names are not unique"
        )

    mdout = _read_mdout(mdout_path)
    mdout_columns = list(mdout["columns"])
    expected_original_names = mdout_columns
    if original_names != expected_original_names:
        raise AssertionError(
            f"{case_name} observable mapping does not exactly match mdout "
            f"semantic order: mapping={original_names}, "
            f"expected={expected_original_names}"
        )
    rows = mdout["rows"]
    steps = _h5_numeric_values(path, "/observables/all/step")
    times = _h5_numeric_values(path, "/observables/all/time")
    if "step" in mdout_columns:
        _assert_numeric_sequences_close(
            f"{case_name} observable step stream",
            [row["step"] for row in rows],
            steps,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
    elif len(steps) != len(rows):
        raise AssertionError(
            f"{case_name} observable step stream length mismatch: "
            f"h5={len(steps)}, mdout={len(rows)}"
        )
    if "time" in mdout_columns:
        _assert_numeric_sequences_close(
            f"{case_name} observable time stream",
            [row["time"] for row in rows],
            times,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-8,
        )
    elif len(times) != len(rows):
        raise AssertionError(
            f"{case_name} observable time stream length mismatch: "
            f"h5={len(times)}, mdout={len(rows)}"
        )
    for original_name, hdf5_name in zip(original_names, hdf5_names):
        dataset = f"/observables/all/{hdf5_name}/value"
        if dataset not in paths:
            raise AssertionError(
                f"{case_name} observable mapping has no value dataset: {dataset}"
            )
        _assert_numeric_sequences_close(
            f"{case_name} observable {original_name}",
            [row[original_name] for row in rows],
            _h5_numeric_values(path, dataset),
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )
    return {"mapped_mdout_columns": original_names}


def _compare_h5_outputs_deterministically(
    case: AbCase,
    run: AbRun,
    families: set[str] | None = None,
) -> dict[str, object]:
    if case.mode == "rerun":
        return _compare_rerun_h5_output(case, run)

    compared: dict[str, object] = {}
    legacy_files = _output_h5_files(case, run.legacy_dir)
    bundled_files = _output_h5_files(case, run.bundled_dir)
    if families is not None:
        legacy_files = {
            name: path
            for name, path in legacy_files.items()
            if name in families
        }
        bundled_files = {
            name: path
            for name, path in bundled_files.items()
            if name in families
        }
    if set(legacy_files) != set(bundled_files):
        raise AssertionError(
            f"{case.name} H5 output families differ: legacy={sorted(legacy_files)}, "
            f"bundled={sorted(bundled_files)}"
        )
    for name in legacy_files:
        datasets = _assert_h5_schema_equivalent(
            legacy_files[name], bundled_files[name], f"{case.name} {name}"
        )
        for dataset in datasets:
            if _h5_dataset_kind(legacy_files[name], dataset) == "numeric":
                left_values = _h5_numeric_values(legacy_files[name], dataset)
                right_values = _h5_numeric_values(bundled_files[name], dataset)
                _assert_matching_numeric_shape(
                    f"{case.name} deterministic {name}:{dataset}",
                    legacy_files[name],
                    bundled_files[name],
                    dataset,
                    left_values,
                    right_values,
                )
                relative_tolerance, absolute_tolerance = (
                    _deterministic_tolerance(dataset)
                )
                particle_root = dataset.removesuffix("/position/value")
                box_dataset = f"{particle_root}/box/edges/value"
                if (
                    dataset.endswith("/position/value")
                    and box_dataset in datasets
                ):
                    _assert_periodic_positions_close(
                        f"{case.name} deterministic H5 {name}:{dataset}",
                        left_values,
                        right_values,
                        _h5_dataset_shape(legacy_files[name], dataset),
                        _h5_numeric_values(legacy_files[name], box_dataset),
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                    )
                else:
                    _assert_numeric_sequences_close(
                        f"{case.name} deterministic H5 {name}:{dataset}",
                        left_values,
                        right_values,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                    )
            else:
                left = _normalize_h5dump(
                    _h5dump_dataset(legacy_files[name], dataset)
                )
                right = _normalize_h5dump(
                    _h5dump_dataset(bundled_files[name], dataset)
                )
                if left != right:
                    raise AssertionError(
                        f"{case.name} deterministic H5 metadata differs: "
                        f"{name}:{dataset}"
                    )
        compared[name] = {
            "method": "deterministic_all_datasets",
            "dataset_count": len(datasets),
            "legacy_shard_count": _vds_shard_count(legacy_files[name]),
            "bundled_shard_count": _vds_shard_count(bundled_files[name]),
        }
    if case.vds:
        for branch, path in (
            ("legacy", legacy_files["trajectory"]),
            ("bundled", bundled_files["trajectory"]),
        ):
            if _vds_shard_count(path) <= 0:
                raise AssertionError(
                    f"{case.name} {branch} VDS output has no shards"
                )
    return compared


def _compare_rerun_h5_output(case: AbCase, run: AbRun) -> dict[str, object]:
    """Bridge the bundled H5 writer to legacy rerun observables and frames."""

    output_files = _output_h5_files(case, run.bundled_dir)
    trajectory_output = output_files["trajectory"]
    observable_output = output_files["observable"]
    input_trajectory = run.bundled_dir / case.trajectory_file_name
    if not input_trajectory.exists():
        raise AssertionError(
            f"{case.name} bundled rerun input trajectory is missing: {input_trajectory}"
        )

    output_steps = _h5_numeric_values(trajectory_output, "/particles/all/step")
    stream_root = f"/particles/{case.trajectory_particle_stream}"
    input_steps = _h5_numeric_values(input_trajectory, f"{stream_root}/step")
    if not output_steps:
        raise AssertionError(
            f"{case.name} bundled rerun output has no trajectory frames"
        )
    matching_indices = []
    for output_step in output_steps:
        try:
            matching_indices.append(input_steps.index(output_step))
        except ValueError as error:
            raise AssertionError(
                f"{case.name} output step {output_step} is absent from rerun input"
            ) from error

    input_position_dataset = f"{stream_root}/position/value"
    input_box_dataset = f"{stream_root}/box/edges/value"
    input_position_values = _h5_numeric_values(
        input_trajectory, input_position_dataset
    )
    input_position_shape = _h5_dataset_shape(
        input_trajectory, input_position_dataset
    )
    input_box_values = _h5_numeric_values(input_trajectory, input_box_dataset)
    input_box_shape = _h5_dataset_shape(input_trajectory, input_box_dataset)
    expected_positions = _rerun_expected_position_values(
        run,
        matching_indices,
        input_position_values,
        input_position_shape,
        input_box_values,
        input_box_shape,
    )
    expected_boxes = _rerun_expected_box_values(
        run, matching_indices, input_box_values, input_box_shape
    )

    for input_dataset, output_dataset in (
        (input_position_dataset, "/particles/all/position/value"),
        (input_box_dataset, "/particles/all/box/edges/value"),
    ):
        output_values = _h5_numeric_values(trajectory_output, output_dataset)
        input_shape = (
            input_position_shape
            if input_dataset == input_position_dataset
            else input_box_shape
        )
        output_shape = _h5_dataset_shape(trajectory_output, output_dataset)
        if len(input_shape) < 2 or len(output_shape) < 2:
            raise AssertionError(
                f"{case.name} rerun trajectory payload is not frame-shaped: "
                f"{input_dataset}"
            )
        input_frame_width = math.prod(input_shape[1:])
        output_frame_width = math.prod(output_shape[1:])
        if input_frame_width != output_frame_width:
            raise AssertionError(
                f"{case.name} rerun {input_dataset} frame width mismatch: "
                f"input={input_frame_width}, output={output_frame_width}"
            )
        expected = (
            expected_positions
            if input_dataset == input_position_dataset
            else expected_boxes
        )
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            output_dataset
        )
        _assert_numeric_sequences_close(
            f"{case.name} rerun output matches bundled input {input_dataset}",
            expected,
            output_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )

    output_times = _h5_numeric_values(trajectory_output, "/particles/all/time")
    input_time_path = f"{stream_root}/time"
    expected_times = output_times
    if input_time_path in _h5_paths(input_trajectory):
        input_times = _h5_numeric_values(input_trajectory, input_time_path)
        expected_times = [input_times[index] for index in matching_indices]
        _assert_numeric_sequences_close(
            f"{case.name} rerun time schedule",
            expected_times,
            output_times,
            relative_tolerance=0.0,
            absolute_tolerance=1.0e-12,
        )

    payload_datasets = {
        "position": "/particles/all/position/value",
        "box": "/particles/all/box/edges/value",
        "velocity": "/particles/all/velocity/value",
        "force": "/particles/all/force/value",
    }
    expected_presence = {
        "position": True,
        "box": True,
        "velocity": case.rerun_velocity_present,
        "force": case.rerun_force_output,
    }
    output_paths = _h5_paths(trajectory_output)
    payloads = {}
    for name, dataset in payload_datasets.items():
        present = dataset in output_paths
        legacy_path = run.legacy_dir / f"output/legacy_rerun.{name}"
        if name == "position":
            legacy_path = run.legacy_dir / "output/legacy_rerun.crd"
        elif name == "velocity":
            legacy_path = run.legacy_dir / "output/legacy_rerun.vel"
        elif name == "force":
            legacy_path = run.legacy_dir / "output/legacy_rerun.frc"
        _assert_rerun_optional_presence(
            case.name,
            name,
            expected=expected_presence[name],
            h5_present=present,
            legacy_present=legacy_path.exists(),
        )
        if not present:
            payloads[name] = {"present": False, "frame_count": 0}
            continue
        legacy_values = _read_legacy_rerun_payload(name, legacy_path)
        output_values = _h5_numeric_values(trajectory_output, dataset)
        shape = _h5_dataset_shape(trajectory_output, dataset)
        relative_tolerance, absolute_tolerance = _deterministic_tolerance(
            dataset
        )
        payloads[name] = _assert_rerun_direct_payload(
            case.name,
            name,
            legacy_values,
            output_values,
            shape,
            output_steps,
            output_times,
            [input_steps[index] for index in matching_indices],
            expected_times,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        payloads[name]["legacy_path"] = str(
            legacy_path.relative_to(run.legacy_dir)
        )
        payloads[name]["h5_dataset"] = dataset

    # _compare_mdout_deterministically already proves legacy and bundled mdout
    # equivalence. The bundled observable output was checked against bundled
    # mdout in _validate_observable_output, which gives a transitive semantic
    # comparison for every legacy observable column.
    observable_columns = _validate_observable_output(
        case.name,
        observable_output,
        run.bundled_dir / "mdout.txt",
    )
    if case.vds and _vds_shard_count(trajectory_output) <= 0:
        raise AssertionError(f"{case.name} bundled VDS output has no shards")
    return {
        "method": "direct_legacy_sidecar_to_bundled_h5_rerun_bridge",
        "trajectory_frame_count": len(output_steps),
        "payloads": payloads,
        "observable": observable_columns,
        "bundled_shard_count": _vds_shard_count(trajectory_output),
    }


def _assert_rerun_optional_presence(
    case_name: str,
    name: str,
    *,
    expected: bool,
    h5_present: bool,
    legacy_present: bool,
) -> None:
    if h5_present != expected or legacy_present != expected:
        raise AssertionError(
            f"{case_name} rerun optional field {name} presence differs: "
            f"expected={expected}, h5={h5_present}, legacy={legacy_present}"
        )


def _assert_rerun_direct_payload(
    case_name: str,
    name: str,
    legacy_values: Sequence[float],
    h5_values: Sequence[float],
    h5_shape: tuple[int, ...],
    h5_steps: Sequence[float],
    h5_times: Sequence[float],
    expected_steps: Sequence[float],
    expected_times: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    if len(h5_shape) < 2:
        raise AssertionError(
            f"{case_name} rerun {name} payload is not frame-shaped: "
            f"shape={h5_shape}"
        )
    frame_count = h5_shape[0]
    frame_width = math.prod(h5_shape[1:])
    expected_value_count = frame_count * frame_width
    if len(h5_values) != expected_value_count or len(legacy_values) != (
        expected_value_count
    ):
        raise AssertionError(
            f"{case_name} rerun {name} value count mismatch: "
            f"shape={h5_shape}, legacy={len(legacy_values)}, "
            f"h5={len(h5_values)}"
        )
    if not (
        len(h5_steps)
        == len(h5_times)
        == len(expected_steps)
        == len(expected_times)
        == frame_count
    ):
        raise AssertionError(
            f"{case_name} rerun {name} frame schedule length mismatch: "
            f"frames={frame_count}, steps={len(h5_steps)}, "
            f"times={len(h5_times)}, expected_steps={len(expected_steps)}, "
            f"expected_times={len(expected_times)}"
        )
    _assert_numeric_sequences_close(
        f"{case_name} rerun {name} step schedule",
        expected_steps,
        h5_steps,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    _assert_numeric_sequences_close(
        f"{case_name} rerun {name} time schedule",
        expected_times,
        h5_times,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-12,
    )
    _assert_numeric_sequences_close(
        f"{case_name} direct legacy/H5 rerun {name}",
        legacy_values,
        h5_values,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return {
        "present": True,
        "frame_count": frame_count,
        "step": list(h5_steps),
        "time": list(h5_times),
    }


def _read_legacy_rerun_payload(name: str, path: Path) -> list[float]:
    if name != "box":
        return _read_native_float32_file(path)
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [float(field) for field in line.split()]
        if len(fields) != 6:
            raise AssertionError(
                f"legacy rerun box row must have 6 values: {line}"
            )
        values.extend(
            (fields[0], 0.0, 0.0, 0.0, fields[1], 0.0, 0.0, 0.0, fields[2])
        )
    return values


def _h5_integer_values(path: Path, dataset: str) -> list[int]:
    values = _h5_numeric_values(path, dataset)
    integers = []
    for value in values:
        if not value.is_integer():
            raise AssertionError(
                f"H5 integer dataset has non-integer value: {path}:{dataset}"
            )
        integers.append(int(value))
    return integers


def _read_h5_virtual_atoms(topology: Path) -> list[H5VirtualAtomRecord]:
    root = "/forcefield/virtual_atom"
    required = {
        f"{root}/atom",
        f"{root}/type",
        f"{root}/from",
        f"{root}/from_offset",
        f"{root}/parameter",
        f"{root}/parameter_offset",
    }
    paths = _h5_paths(topology)
    if not required & paths:
        return []
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(f"virtual atom topology is incomplete: {missing}")

    atoms = _h5_integer_values(topology, f"{root}/atom")
    kinds = _h5_integer_values(topology, f"{root}/type")
    source_atoms = _h5_integer_values(topology, f"{root}/from")
    source_offsets = _h5_integer_values(topology, f"{root}/from_offset")
    parameters = _h5_numeric_values(topology, f"{root}/parameter")
    parameter_offsets = _h5_integer_values(topology, f"{root}/parameter_offset")
    record_count = len(atoms)
    if len(kinds) != record_count:
        raise AssertionError(
            "virtual atom type count does not match atom count"
        )
    if len(source_offsets) != record_count + 1:
        raise AssertionError(
            "virtual atom source offsets have an invalid length"
        )
    if len(parameter_offsets) != record_count + 1:
        raise AssertionError(
            "virtual atom parameter offsets have an invalid length"
        )
    if source_offsets[0] != 0 or source_offsets[-1] != len(source_atoms):
        raise AssertionError(
            "virtual atom source offsets do not cover all values"
        )
    if parameter_offsets[0] != 0 or parameter_offsets[-1] != len(parameters):
        raise AssertionError(
            "virtual atom parameter offsets do not cover all values"
        )

    records = []
    for index, (atom, kind) in enumerate(zip(atoms, kinds)):
        source_begin, source_end = source_offsets[index : index + 2]
        parameter_begin, parameter_end = parameter_offsets[index : index + 2]
        if source_begin > source_end or parameter_begin > parameter_end:
            raise AssertionError("virtual atom offsets must be monotonic")
        expected_counts = {0: (1, 1), 1: (2, 1), 2: (3, 2), 3: (3, 2)}
        if kind not in expected_counts:
            raise AssertionError(
                f"unsupported virtual atom type in H5 topology: {kind}"
            )
        expected_sources, expected_parameters = expected_counts[kind]
        if source_end - source_begin != expected_sources:
            raise AssertionError(
                f"virtual atom type {kind} has an invalid source count"
            )
        if parameter_end - parameter_begin != expected_parameters:
            raise AssertionError(
                f"virtual atom type {kind} has an invalid parameter count"
            )
        records.append(
            H5VirtualAtomRecord(
                atom=atom,
                kind=kind,
                source_atoms=tuple(source_atoms[source_begin:source_end]),
                parameters=tuple(parameters[parameter_begin:parameter_end]),
            )
        )
    return records


def _rerun_updates_box(case_dir: Path) -> bool:
    mdin_path = case_dir / _mdin_name(case_dir)
    for raw_line in mdin_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key != "rerun_need_box_update":
            continue
        normalized = value.strip("\"'").lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise AssertionError(f"invalid rerun_need_box_update value: {value!r}")
    return False


def _lower_triangular_periodic_displacement(
    left: Sequence[float], right: Sequence[float], box_edges: Sequence[float]
) -> tuple[float, float, float]:
    if len(box_edges) != 9:
        raise AssertionError("rerun box frame must contain nine edge values")
    a11, a21, a22, a31, a32, a33 = (
        box_edges[0],
        box_edges[3],
        box_edges[4],
        box_edges[6],
        box_edges[7],
        box_edges[8],
    )
    if min(abs(a11), abs(a22), abs(a33)) == 0.0:
        raise AssertionError(
            "rerun virtual atom calculation requires an invertible box"
        )
    dr_x = left[0] - right[0]
    dr_y = left[1] - right[1]
    dr_z = left[2] - right[2]
    r11 = 1.0 / a11
    r21 = -a21 / (a11 * a22)
    r22 = 1.0 / a22
    r31 = (a21 * a32 - a22 * a31) / (a11 * a22 * a33)
    r32 = -a32 / (a22 * a33)
    r33 = 1.0 / a33
    shift_x = math.floor(dr_x * r11 + dr_y * r21 + dr_z * r31 + 0.5)
    shift_y = math.floor(dr_y * r22 + dr_z * r32 + 0.5)
    shift_z = math.floor(dr_z * r33 + 0.5)
    return (
        dr_x - shift_x * a11 - shift_y * a21 - shift_z * a31,
        dr_y - shift_y * a22 - shift_z * a32,
        dr_z - shift_z * a33,
    )


def _canonicalize_virtual_atom_positions(
    positions: Sequence[float],
    atom_count: int,
    box_edges: Sequence[float],
    records: Sequence[H5VirtualAtomRecord],
) -> list[float]:
    if len(positions) != atom_count * 3:
        raise AssertionError(
            "rerun position frame does not match topology atom count"
        )
    coordinates = [
        list(positions[index : index + 3])
        for index in range(0, len(positions), 3)
    ]
    levels = [0] * atom_count
    records_by_level: dict[int, list[H5VirtualAtomRecord]] = {}
    for record in records:
        if record.atom < 0 or record.atom >= atom_count:
            raise AssertionError(
                f"virtual atom index is out of range: {record.atom}"
            )
        if any(
            source < 0 or source >= atom_count for source in record.source_atoms
        ):
            raise AssertionError("virtual atom source index is out of range")
        level = max(levels[source] for source in record.source_atoms) + 1
        levels[record.atom] = level
        records_by_level.setdefault(level, []).append(record)

    for level in sorted(records_by_level):
        for record in records_by_level[level]:
            source = [coordinates[index] for index in record.source_atoms]
            if record.kind == 0:
                h_double = 2.0 * record.parameters[0]
                coordinates[record.atom] = [
                    source[0][0],
                    source[0][1],
                    2.0 * h_double - source[0][2],
                ]
            elif record.kind == 1:
                delta = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                coordinates[record.atom] = [
                    source[0][axis] + record.parameters[0] * delta[axis]
                    for axis in range(3)
                ]
            elif record.kind == 2:
                delta_2 = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                delta_3 = _lower_triangular_periodic_displacement(
                    source[2], source[0], box_edges
                )
                coordinates[record.atom] = [
                    source[0][axis]
                    + record.parameters[0] * delta_2[axis]
                    + record.parameters[1] * delta_3[axis]
                    for axis in range(3)
                ]
            else:
                delta_21 = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                delta_32 = _lower_triangular_periodic_displacement(
                    source[2], source[1], box_edges
                )
                direction = [
                    delta_21[axis] + record.parameters[1] * delta_32[axis]
                    for axis in range(3)
                ]
                direction_norm = math.sqrt(
                    sum(value * value for value in direction)
                )
                if direction_norm == 0.0:
                    raise AssertionError(
                        "virtual atom type 3 has a zero-length direction"
                    )
                coordinates[record.atom] = [
                    source[0][axis]
                    + record.parameters[0] * direction[axis] / direction_norm
                    for axis in range(3)
                ]
    return [value for coordinate in coordinates for value in coordinate]


def _rerun_expected_position_values(
    run: AbRun,
    matching_indices: Sequence[int],
    input_positions: Sequence[float],
    input_position_shape: Sequence[int],
    input_boxes: Sequence[float],
    input_box_shape: Sequence[int],
) -> list[float]:
    if len(input_position_shape) != 3 or input_position_shape[2] != 3:
        raise AssertionError(
            "rerun position input must have shape [frame,atom,3]"
        )
    if len(input_box_shape) != 3 or input_box_shape[1:] != (3, 3):
        raise AssertionError("rerun box input must have shape [frame,3,3]")
    atom_count = input_position_shape[1]
    position_width = math.prod(input_position_shape[1:])
    box_width = math.prod(input_box_shape[1:])
    records = _read_h5_virtual_atoms(run.bundled_dir / "topology.spgt.h5")
    use_trajectory_box = _rerun_updates_box(run.bundled_dir)
    if use_trajectory_box:
        restart_box = []
    else:
        restart_box = _rerun_bootstrap_box_values(run, box_width)

    expected = []
    for frame_index in matching_indices:
        position_begin = frame_index * position_width
        box_begin = frame_index * box_width
        frame_box = (
            input_boxes[box_begin : box_begin + box_width]
            if use_trajectory_box
            else restart_box
        )
        expected.extend(
            _canonicalize_virtual_atom_positions(
                input_positions[
                    position_begin : position_begin + position_width
                ],
                atom_count,
                frame_box,
                records,
            )
        )
    return expected


def _rerun_expected_box_values(
    run: AbRun,
    matching_indices: Sequence[int],
    input_boxes: Sequence[float],
    input_box_shape: Sequence[int],
) -> list[float]:
    if len(input_box_shape) < 2:
        raise AssertionError("rerun box input is not frame-shaped")
    box_width = math.prod(input_box_shape[1:])
    return [
        value
        for frame_index in matching_indices
        for value in input_boxes[
            frame_index * box_width : (frame_index + 1) * box_width
        ]
    ]


def _rerun_bootstrap_box_values(run: AbRun, box_width: int) -> list[float]:
    restart_path = run.bundled_dir / "restart.spgr.h5"
    if restart_path.exists():
        values = _h5_numeric_values(
            restart_path, "/particles/all/box/edges/value"
        )
    else:
        coordinate_path = run.bundled_dir / "coordinate.txt"
        lines = [
            line.strip()
            for line in coordinate_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        atom_count = int(lines[0].split()[0])
        box_fields = lines[1 + atom_count :]
        if len(box_fields) != 1 or len(box_fields[0].split()) != 6:
            raise AssertionError(
                "restart-absent coordinate bootstrap requires one six-value box row"
            )
        length_x, length_y, length_z, alpha, beta, gamma = (
            float(value) for value in box_fields[0].split()
        )
        if not all(
            math.isclose(angle, 90.0, rel_tol=0.0, abs_tol=1.0e-6)
            for angle in (alpha, beta, gamma)
        ):
            raise AssertionError(
                "restart-absent coordinate bootstrap must be orthogonal"
            )
        values = [
            length_x,
            0.0,
            0.0,
            0.0,
            length_y,
            0.0,
            0.0,
            0.0,
            length_z,
        ]
    if len(values) != box_width:
        raise AssertionError("rerun bootstrap box has an invalid shape")
    return values


def _compare_h5_outputs_statistically(
    case: AbCase,
    runs: Sequence[AbRun],
    families: set[str] | None = None,
    *,
    inference_unit: str = "block",
) -> dict[str, object]:
    expected_families = set(_output_h5_files(case, runs[0].legacy_dir))
    if families is not None:
        expected_families.intersection_update(families)
    summaries: dict[str, object] = {}
    for name in sorted(expected_families):
        baseline_datasets: set[str] | None = None
        for run in runs:
            legacy_path = _output_h5_files(case, run.legacy_dir)[name]
            bundled_path = _output_h5_files(case, run.bundled_dir)[name]
            datasets = _assert_h5_schema_equivalent(
                legacy_path,
                bundled_path,
                f"{case.name} {name} replica {run.replica_index}",
            )
            if baseline_datasets is None:
                baseline_datasets = datasets
            elif baseline_datasets != datasets:
                raise AssertionError(
                    f"{case.name} {name} schema changed across replicas"
                )

        assert baseline_datasets is not None
        dataset_summaries: dict[str, object] = {}
        family_results: dict[str, dict[str, float | int]] = {}
        for dataset in sorted(baseline_datasets):
            legacy_path = _output_h5_files(case, runs[0].legacy_dir)[name]
            kind = _h5_dataset_kind(legacy_path, dataset)
            if kind == "numeric":
                numeric_summary = _compare_h5_numeric_dataset_statistics(
                    case,
                    runs,
                    name,
                    dataset,
                    inference_unit=inference_unit,
                )
                dataset_summaries[dataset] = numeric_summary
                equivalence_result = _primary_equivalence_result(
                    numeric_summary
                )
                if equivalence_result is not None:
                    family_results[dataset] = equivalence_result
            else:
                for run in runs:
                    legacy_file = _output_h5_files(case, run.legacy_dir)[name]
                    bundled_file = _output_h5_files(case, run.bundled_dir)[name]
                    left = _normalize_h5dump(
                        _h5dump_dataset(legacy_file, dataset)
                    )
                    right = _normalize_h5dump(
                        _h5dump_dataset(bundled_file, dataset)
                    )
                    if left != right:
                        raise AssertionError(
                            f"{case.name} {name} metadata dataset differs: {dataset}"
                        )
                dataset_summaries[dataset] = {"method": "exact_metadata"}
        if family_results:
            holm_correct_equivalence_family(
                f"{case.name} {name} H5 dataset family",
                family_results,
                alpha=_holm_alpha(
                    _statistical_policy(name, inference_unit=inference_unit)
                ),
            )
        summaries[name] = {
            "method": "all_dataset_schema_and_statistical_values",
            "dataset_count": len(baseline_datasets),
            "datasets": dataset_summaries,
        }
    return summaries


def _primary_equivalence_result(
    summary: dict[str, object],
) -> dict[str, float | int] | None:
    for key in ("flat_values", "finite_values"):
        candidate = summary.get(key)
        if isinstance(candidate, dict) and "equivalence_p_value" in candidate:
            return candidate
    return None


def _compare_h5_numeric_dataset_statistics(
    case: AbCase,
    runs: Sequence[AbRun],
    name: str,
    dataset: str,
    *,
    inference_unit: str = "block",
) -> dict[str, object]:
    legacy_replicas = []
    bundled_replicas = []
    shape: tuple[int, ...] | None = None
    for run in runs:
        legacy_path = _output_h5_files(case, run.legacy_dir)[name]
        bundled_path = _output_h5_files(case, run.bundled_dir)[name]
        legacy_values = _h5_numeric_values(legacy_path, dataset)
        bundled_values = _h5_numeric_values(bundled_path, dataset)
        _assert_matching_numeric_shape(
            f"{case.name} {name}:{dataset} replica {run.replica_index}",
            legacy_path,
            bundled_path,
            dataset,
            legacy_values,
            bundled_values,
        )
        if shape is None:
            shape = _h5_dataset_shape(legacy_path, dataset)
        legacy_replicas.append(legacy_values)
        bundled_replicas.append(bundled_values)

    if _is_deterministic_timeline_dataset(dataset):
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_numeric_sequences_close(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
                relative_tolerance=0.0,
                absolute_tolerance=1.0e-12,
            )
        return {"method": "exact_timeline_or_completion_metadata"}

    if any(
        not math.isfinite(value)
        for replica in (*legacy_replicas, *bundled_replicas)
        for value in replica
    ):
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_nonfinite_patterns_match(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
            )
        finite_legacy = [
            [value for value in replica if math.isfinite(value)]
            for replica in legacy_replicas
        ]
        finite_bundled = [
            [value for value in replica if math.isfinite(value)]
            for replica in bundled_replicas
        ]
        result: dict[str, object] = {"method": "exact_nonfinite_pattern"}
        finite_policy = replace(
            _statistical_policy(dataset, inference_unit=inference_unit),
            burn_in_frames=0,
        )
        if finite_legacy[0] and _can_use_statistics(
            finite_legacy, finite_policy
        ):
            result["finite_values"] = compare_replicas(
                f"{case.name} {name}:{dataset} finite values",
                finite_legacy,
                finite_bundled,
                finite_policy,
            )
        return result

    policy = _statistical_policy(dataset, inference_unit=inference_unit)
    flat_policy = replace(policy, burn_in_frames=0)
    result: dict[str, object] = {}
    atom_weights = (
        _trajectory_atom_weights(runs[0], dataset, shape) if shape else None
    )
    legacy_observables = (
        [
            trajectory_observable_series(
                dataset,
                values,
                shape,
                atom_weights=atom_weights,
                include_atom_features=inference_unit != "replica",
                maximum_pair_samples=4096,
                **_trajectory_box_arguments(
                    _output_h5_files(case, run.legacy_dir)[name], dataset
                ),
            )
            for run, values in zip(runs, legacy_replicas)
        ]
        if shape
        else []
    )
    bundled_observables = (
        [
            trajectory_observable_series(
                dataset,
                values,
                shape,
                atom_weights=atom_weights,
                include_atom_features=inference_unit != "replica",
                maximum_pair_samples=4096,
                **_trajectory_box_arguments(
                    _output_h5_files(case, run.bundled_dir)[name], dataset
                ),
            )
            for run, values in zip(runs, bundled_replicas)
        ]
        if shape
        else []
    )
    has_physical_observables = bool(
        legacy_observables and legacy_observables[0]
    )
    compare_flat_values = not (
        inference_unit == "replica" and has_physical_observables
    )
    if compare_flat_values and _can_use_statistics(
        legacy_replicas, flat_policy
    ):
        result["flat_values"] = compare_replicas(
            f"{case.name} {name}:{dataset} flat values",
            legacy_replicas,
            bundled_replicas,
            flat_policy,
        )
    elif compare_flat_values:
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_numeric_sequences_close(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
                relative_tolerance=1.0e-4,
                absolute_tolerance=1.0e-8,
            )
        result["flat_values"] = {"method": "exact_short_series"}

    if compare_flat_values and shape and len(shape) >= 1 and shape[0] > 1:
        legacy_frame_means = [
            _frame_summary_series(values, shape, rms=False)
            for values in legacy_replicas
        ]
        bundled_frame_means = [
            _frame_summary_series(values, shape, rms=False)
            for values in bundled_replicas
        ]
        if _can_use_statistics(legacy_frame_means, policy):
            result["frame_means"] = compare_replicas(
                f"{case.name} {name}:{dataset} frame means",
                legacy_frame_means,
                bundled_frame_means,
                policy,
            )
            legacy_frame_rms = [
                _frame_summary_series(values, shape, rms=True)
                for values in legacy_replicas
            ]
            bundled_frame_rms = [
                _frame_summary_series(values, shape, rms=True)
                for values in bundled_replicas
            ]
            result["frame_rms"] = compare_replicas(
                f"{case.name} {name}:{dataset} frame RMS",
                legacy_frame_rms,
                bundled_frame_rms,
                policy,
            )
    if shape:
        if has_physical_observables:
            all_feature_names = set(legacy_observables[0])
            feature_names = _trajectory_feature_names_for_inference(
                legacy_observables[0], inference_unit
            )
            if any(
                set(item) != all_feature_names for item in legacy_observables
            ):
                raise AssertionError(
                    f"{case.name} {dataset} legacy features differ"
                )
            if any(
                set(item) != all_feature_names for item in bundled_observables
            ):
                raise AssertionError(
                    f"{case.name} {dataset} bundled features differ"
                )
            feature_results: dict[str, dict[str, float | int]] = {}
            for feature in sorted(feature_names):
                legacy_series = [item[feature] for item in legacy_observables]
                bundled_series = [item[feature] for item in bundled_observables]
                feature_policy = _statistical_policy(
                    f"{dataset} {feature}",
                    inference_unit=inference_unit,
                )
                if not _can_use_statistics(legacy_series, feature_policy):
                    continue
                feature_results[feature] = compare_replicas(
                    f"{case.name} {name}:{dataset} {feature}",
                    legacy_series,
                    bundled_series,
                    feature_policy,
                )
            if feature_results:
                holm_correct_equivalence_family(
                    f"{case.name} {name}:{dataset} trajectory observable family",
                    feature_results,
                    alpha=_holm_alpha(policy),
                )
                result["trajectory_observables"] = feature_results
    return result


def _trajectory_feature_names_for_inference(
    observables: dict[str, list[float]], inference_unit: str
) -> set[str]:
    feature_names = set(observables)
    if inference_unit == "replica":
        feature_names = {
            name for name in feature_names if not name.startswith("atom_")
        }
    return feature_names


def _trajectory_atom_weights(
    run: AbRun, dataset: str, shape: tuple[int, ...]
) -> list[float] | None:
    if (
        not dataset.endswith(("/position/value", "/velocity/value"))
        or len(shape) != 3
    ):
        return None
    mass_files = sorted(run.legacy_dir.glob("*_mass.txt"))
    if len(mass_files) != 1:
        return None
    masses = read_mass_values(mass_files[0])
    if len(masses) != shape[1]:
        raise AssertionError(
            f"trajectory atom count {shape[1]} differs from mass count {len(masses)}"
        )
    return masses


def _trajectory_box_arguments(path: Path, dataset: str) -> dict[str, object]:
    if not dataset.endswith("/position/value"):
        return {}
    particle_root = dataset.removesuffix("/position/value")
    box_dataset = f"{particle_root}/box/edges/value"
    return {
        "box_values": _h5_numeric_values(path, box_dataset),
        "box_shape": _h5_dataset_shape(path, box_dataset),
    }


def _can_use_statistics(
    replicas: Sequence[Sequence[float]], policy: StatisticalEquivalencePolicy
) -> bool:
    required = policy.burn_in_frames + (
        policy.minimum_blocks_per_replica * policy.block_size
    )
    return all(len(replica) >= required for replica in replicas)


def _frame_summary_series(
    values: Sequence[float], shape: tuple[int, ...], *, rms: bool
) -> list[float]:
    frame_count = shape[0]
    if frame_count <= 0 or len(values) % frame_count != 0:
        raise AssertionError(
            f"cannot build frame statistics for shape={shape}, values={len(values)}"
        )
    width = len(values) // frame_count
    series = []
    for frame_index in range(frame_count):
        frame = values[frame_index * width : (frame_index + 1) * width]
        if rms:
            series.append(
                math.sqrt(statistics.fmean(value * value for value in frame))
            )
        else:
            series.append(statistics.fmean(frame))
    return series


def _assert_matching_numeric_shape(
    label: str,
    legacy_path: Path,
    bundled_path: Path,
    dataset: str,
    legacy_values: Sequence[float],
    bundled_values: Sequence[float],
) -> None:
    legacy_shape = _h5_dataset_shape(legacy_path, dataset)
    bundled_shape = _h5_dataset_shape(bundled_path, dataset)
    if legacy_shape != bundled_shape or len(legacy_values) != len(
        bundled_values
    ):
        raise AssertionError(
            f"{label} numeric shape mismatch: legacy_shape={legacy_shape}, "
            f"bundled_shape={bundled_shape}, legacy_values={len(legacy_values)}, "
            f"bundled_values={len(bundled_values)}"
        )


def _is_deterministic_timeline_dataset(dataset: str) -> bool:
    return (
        dataset.endswith("/step")
        or dataset.endswith("/time")
        or dataset.endswith("/frame_count")
        or dataset.endswith("/last_complete_step")
        or dataset.endswith("/last_complete_time")
    )


def _assert_h5_schema_equivalent(
    legacy_path: Path, bundled_path: Path, label: str
) -> set[str]:
    legacy_paths = _h5_paths(legacy_path)
    bundled_paths = _h5_paths(bundled_path)
    if legacy_paths != bundled_paths:
        raise AssertionError(
            f"{label} H5 path sets differ: "
            f"legacy_only={sorted(legacy_paths - bundled_paths)}, "
            f"bundled_only={sorted(bundled_paths - legacy_paths)}"
        )
    legacy_datasets = _h5_dataset_paths(legacy_path)
    bundled_datasets = _h5_dataset_paths(bundled_path)
    if legacy_datasets != bundled_datasets:
        raise AssertionError(
            f"{label} H5 dataset sets differ: "
            f"legacy_only={sorted(legacy_datasets - bundled_datasets)}, "
            f"bundled_only={sorted(bundled_datasets - legacy_datasets)}"
        )
    for dataset in legacy_datasets:
        legacy_header = _h5_dataset_header(legacy_path, dataset)
        bundled_header = _h5_dataset_header(bundled_path, dataset)
        if legacy_header != bundled_header:
            raise AssertionError(f"{label} H5 schema differs at {dataset}")
    return legacy_datasets


def _h5_paths(path: Path) -> set[str]:
    output = _run(["h5dump", "-n", path]).stdout
    paths = set()
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] in {"group", "dataset"}:
            paths.add(fields[1])
    return paths


def _h5_dataset_paths(path: Path) -> set[str]:
    output = _run(["h5dump", "-n", path]).stdout
    datasets = set()
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] == "dataset":
            datasets.add(fields[1])
    return datasets


def _h5dump_dataset(path: Path, dataset: str) -> str:
    return _run(["h5dump", "-d", dataset, path]).stdout


def _h5_dataset_header(path: Path, dataset: str) -> str:
    return _normalize_h5dump(_run(["h5dump", "-H", "-d", dataset, path]).stdout)


def _h5_dataset_kind(path: Path, dataset: str) -> str:
    header = _h5_dataset_header(path, dataset)
    match = re.search(
        r"DATASET\s+.*?\{\s*DATATYPE\s+([^\n]+)", header, re.DOTALL
    )
    if match is None:
        raise AssertionError(
            f"cannot determine H5 dataset type: {path}:{dataset}"
        )
    datatype = match.group(1)
    if "H5T_STRING" in datatype:
        return "string"
    if any(
        marker in datatype
        for marker in ("H5T_IEEE", "H5T_STD_I", "H5T_STD_U", "H5T_NATIVE_")
    ):
        return "numeric"
    return "other"


def _h5_dataset_shape(path: Path, dataset: str) -> tuple[int, ...]:
    header = _h5_dataset_header(path, dataset)
    match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*([^)]*?)\s*\)", header)
    if match is None:
        return ()
    dimensions = [item.strip() for item in match.group(1).split(",")]
    if not dimensions or dimensions == [""]:
        return ()
    return tuple(int(item) for item in dimensions)


def _h5_data_text(path: Path, dataset: str) -> str:
    dump = _run(
        [
            "h5dump",
            "-d",
            dataset,
            "-y",
            "-m",
            "%.17g",
            "-L",
            "%.21Lg",
            "-w",
            "0",
            path,
        ]
    ).stdout
    marker = re.search(r"\bDATA\s*\{", dump)
    if marker is None:
        raise AssertionError(
            f"H5 dataset has no data section: {path}:{dataset}"
        )
    start = marker.end()
    depth = 1
    in_string = False
    escaped = False
    for index in range(start, len(dump)):
        character = dump[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return dump[start:index]
    raise AssertionError(
        f"H5 dataset has unterminated data section: {path}:{dataset}"
    )


def _h5_numeric_values(path: Path, dataset: str) -> list[float]:
    if _h5_dataset_kind(path, dataset) != "numeric":
        raise AssertionError(f"H5 dataset is not numeric: {path}:{dataset}")
    data = re.sub(r'"(?:\\.|[^"\\])*"', "", _h5_data_text(path, dataset))
    tokens = re.findall(
        r"(?<![A-Za-z0-9_])[-+]?(?:nan|inf(?:inity)?|"
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?![A-Za-z0-9_])",
        data,
        flags=re.IGNORECASE,
    )
    return [float(token) for token in tokens]


def _h5_string_values(path: Path, dataset: str) -> list[str]:
    if _h5_dataset_kind(path, dataset) != "string":
        raise AssertionError(
            f"H5 dataset is not a string array: {path}:{dataset}"
        )
    with h5py.File(path, "r") as h5:
        values = h5[dataset].asstr()[...]
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values.flat]


def _normalize_h5dump(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("HDF5 "):
            continue
        normalized = re.sub(r'FILE\s+"[^"]+"', 'FILE "<vds-source>"', line)
        lines.append(normalized.strip())
    return "\n".join(lines)


def _run(
    cmd: list[object], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with code {result.returncode}: "
            f"{' '.join(str(part) for part in cmd)}\n"
            f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
        )
    return result
