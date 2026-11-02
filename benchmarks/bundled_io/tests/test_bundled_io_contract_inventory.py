from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.bundled_io.ab_contracts import (
    load_contract_registry,
    load_implementation_inventory,
    validate_implementation_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _quoted_values_between(path: Path, start: str, end: str) -> set[str]:
    source = path.read_text(encoding="utf-8")
    begin = source.index(start)
    finish = source.index(end, begin)
    return set(re.findall(r'"([A-Za-z0-9_]+)"', source[begin:finish]))


def _contract_key_constants(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return set(
        re.findall(
            r"k[A-Za-z0-9]+Key\s*=\s*\n?\s*\"([A-Za-z0-9_]+)\"",
            source,
        )
    )


def test_implementation_key_inventory_matches_runtime_sources():
    inventory = load_implementation_inventory()

    assert set(inventory["h5_input_keys"]) == _contract_key_constants(
        REPO_ROOT / "SPONGE" / "utils" / "control" / "h5_input_contract.hpp"
    )
    assert set(inventory["h5_output_keys"]) == _contract_key_constants(
        REPO_ROOT / "SPONGE" / "utils" / "control" / "h5_output_contract.hpp"
    )
    sidecar_contract = (
        REPO_ROOT
        / "SPONGE"
        / "utils"
        / "h5md"
        / "h5_legacy_sidecar_contract.hpp"
    )
    assert set(inventory["topology_sidecar_keys"]) == _quoted_values_between(
        sidecar_contract, "H5_Topology_Sidecar_Command_Keys", "return keys;"
    )
    assert set(inventory["protocol_sidecar_keys"]) == _quoted_values_between(
        sidecar_contract, "H5_Protocol_Sidecar_Command_Keys", "return keys;"
    )
    assert set(inventory["legacy_output_keys"]) == _quoted_values_between(
        REPO_ROOT / "SPONGE" / "utils" / "h5md" / "output_plan.hpp",
        "legacy_keys = {",
        "};",
    )

    mdin_schema = json.loads(
        (REPO_ROOT / "schemas" / "mdin.schema.json").read_text(encoding="utf-8")
    )
    rerun_schema_keys = {
        key for key in mdin_schema["properties"] if key.startswith("rerun_")
    }
    assert set(inventory["rerun_control_keys"]) == rerun_schema_keys


def test_every_implementation_key_has_exactly_one_contract_owner():
    contracts = load_contract_registry()
    inventory = load_implementation_inventory()
    owners = validate_implementation_inventory(contracts, inventory)

    expected_count = sum(len(values) for values in inventory.values())
    assert len(owners) == expected_count


def test_improper_conversion_is_promoted_with_module_owned_runtime_evidence():
    contract = load_contract_registry()["input.topology.improper"]

    assert contract.status == "supported"
    assert contract.minimum_evidence == "E3"
    assert contract.case_ids == (
        "normal_improper_converted_canonical_nonzero",
        "normal_improper_converted_alias_nonzero",
    )
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_nb14_is_promoted_only_with_module_owned_runtime_evidence():
    contract = load_contract_registry()["input.topology.nb14"]

    assert contract.status == "supported"
    assert contract.minimum_evidence == "E3"
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_removing_an_inventory_owner_fails_the_gate():
    contracts = load_contract_registry()
    inventory = load_implementation_inventory()
    contract_id = "input.topology.improper"
    contracts[contract_id] = replace(contracts[contract_id], inventory_refs=())

    with pytest.raises(AssertionError, match="refs have no contract"):
        validate_implementation_inventory(contracts, inventory)
