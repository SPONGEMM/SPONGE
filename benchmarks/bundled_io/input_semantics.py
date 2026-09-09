"""Module-owned behavior assertions for bundled input A/B cases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

REQUIRED_INPUT_SEMANTIC_CONTRACTS = frozenset(
    {
        "input.topology.mass",
        "input.topology.charge",
        "input.topology.residue",
        "input.topology.residue.sidecar",
        "input.topology.exclusions",
        "input.topology.lj",
        "input.topology.nb14",
        "input.topology.nb14_extra",
        "input.topology.bond",
        "input.topology.angle",
        "input.topology.dihedral",
        "input.topology.improper",
        "input.topology.improper.native_runtime",
        "input.topology.urey_bradley",
        "input.topology.cmap",
        "input.topology.gb",
        "input.topology.gb.hybrid_activation",
        "input.topology.lj_soft_core",
        "input.topology.subsystem_division",
        "input.topology.virtual_atoms",
        "input.topology.virtual_atoms_alias",
        "input.topology.virtual_atoms_pbc",
        "input.manybody.eam",
        "input.manybody.sw",
        "input.manybody.sw.sidecar",
        "input.manybody.edip",
        "input.manybody.tersoff",
        "input.manybody.tersoff.sidecar",
        "input.manybody.reaxff",
        "input.custom.pairwise",
        "input.custom.listed",
        "input.protocol.constraint",
        "input.protocol.constraint.sidecar",
        "input.protocol.cv",
        "input.protocol.cv_restraint",
        "input.protocol.cv_restraint.sidecar",
        "input.protocol.positional_restraint",
        "input.protocol.positional_restraint.sidecar",
        "input.protocol.steering",
        "input.protocol.steering.cv_sidecar",
        "input.protocol.soft_wall",
        "input.protocol.soft_wall.sidecar",
        "input.protocol.sits",
        "input.protocol.sits.nk_typed_restart",
        "input.bias.metadynamics",
        "input.bias.nhc",
    }
)


@dataclass(frozen=True)
class InputSemanticSpec:
    contract_id: str
    observables: tuple[str, ...]
    minimum_magnitude: float = 1.0e-8


def assert_module_semantics(
    label: str,
    legacy_rows: Sequence[Mapping[str, float]],
    bundled_rows: Sequence[Mapping[str, float]],
    spec: InputSemanticSpec,
    *,
    deterministic: bool,
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-8,
) -> dict[str, object]:
    """Require a present, non-trivial, equivalent module-owned result."""

    if not legacy_rows or not bundled_rows:
        raise AssertionError(f"{label} has no runtime rows")
    if deterministic and len(legacy_rows) != len(bundled_rows):
        raise AssertionError(
            f"{label} row count mismatch: legacy={len(legacy_rows)}, "
            f"bundled={len(bundled_rows)}"
        )

    nontrivial = {"legacy": False, "bundled": False}
    observable_deltas = {}
    for observable in spec.observables:
        legacy = _observable_values(label, "legacy", legacy_rows, observable)
        bundled = _observable_values(label, "bundled", bundled_rows, observable)
        nontrivial["legacy"] |= any(
            math.isfinite(value) and abs(value) > spec.minimum_magnitude
            for value in legacy
        )
        nontrivial["bundled"] |= any(
            math.isfinite(value) and abs(value) > spec.minimum_magnitude
            for value in bundled
        )
        if deterministic:
            _assert_deterministic_values(
                f"{label} {observable}",
                legacy,
                bundled,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        paired_deltas = [
            abs(left - right)
            for left, right in zip(legacy, bundled)
            if math.isfinite(left) and math.isfinite(right)
        ]
        observable_deltas[observable] = {
            "legacy_value_count": len(legacy),
            "bundled_value_count": len(bundled),
            "finite_pair_count": len(paired_deltas),
            "maximum_absolute_delta": (
                max(paired_deltas) if paired_deltas else None
            ),
        }

    for branch, has_nontrivial_result in nontrivial.items():
        if not has_nontrivial_result:
            raise AssertionError(
                f"{label} {branch} module-owned observables are all trivial: "
                f"{list(spec.observables)}"
            )
    return {
        "contract_id": spec.contract_id,
        "observables": list(spec.observables),
        "minimum_magnitude": spec.minimum_magnitude,
        "deterministic": deterministic,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "observable_deltas": observable_deltas,
        "legacy_nontrivial": nontrivial["legacy"],
        "bundled_nontrivial": nontrivial["bundled"],
    }


def _observable_values(
    label: str,
    branch: str,
    rows: Sequence[Mapping[str, float]],
    observable: str,
) -> list[float]:
    if any(observable not in row for row in rows):
        raise AssertionError(
            f"{label} {branch} is missing module-owned observable {observable}"
        )
    return [float(row[observable]) for row in rows]


def _same_nonfinite(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isinf(left) and math.isinf(right) and left == right


def _assert_deterministic_values(
    label: str,
    legacy: Sequence[float],
    bundled: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if len(legacy) != len(bundled):
        raise AssertionError(
            f"{label} length mismatch: legacy={len(legacy)}, bundled={len(bundled)}"
        )
    for index, (left, right) in enumerate(zip(legacy, bundled)):
        if not math.isfinite(left) or not math.isfinite(right):
            if not _same_nonfinite(left, right):
                raise AssertionError(
                    f"{label} non-finite mismatch at row {index}: "
                    f"legacy={left}, bundled={right}"
                )
            continue
        if not math.isclose(
            left, right, rel_tol=relative_tolerance, abs_tol=absolute_tolerance
        ):
            raise AssertionError(
                f"{label} mismatch at row {index}: legacy={left}, bundled={right}"
            )
