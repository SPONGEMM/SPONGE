#!/usr/bin/env python3
"""Validate representative mdin TOML examples against SPONGE's mdin schema.

It implements the small JSON Schema subset used by schemas/mdin.schema.json so
the test target does not depend on Mokda, node_modules, or Python jsonschema.
Python 3.11+ uses stdlib tomllib; Python 3.10 can use tomli when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "validate_mdin_schema_examples.py requires Python 3.11+ "
            "or the tomli package"
        ) from exc


EXAMPLES = {
    "h5_npt_bundle": """
mode = "npt"
step_limit = 2500000
dt = 0.002
target_temperature = 300.0
target_pressure = 1.0

[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/metadyn.protocol.spgp.h5"

[input.h5.restart]
path = "runs/prod_0007.restart.spgr.h5"
load = "structural"

[thermostat]
mode = "middle_langevin"
seed = 123456
tau = 1.0

[barostat]
mode = "monte_carlo_barostat"

[barostat.monte_carlo]
update_interval = 100

[output.h5.trajectory]
path = "prod.spg.h5md"
vds = true
chunk_size = 20
repair_policy = "strict"

[output.h5.restart]
path = "prod.spgr.h5"

[output.h5.observable]
path = "prod.obs.spg.h5md"

[write.interval]
trajectory = 5000
information = 500
mdout = 500
restart = 500000
""",
    "h5_rerun_trajectory": """
mode = "rerun"
step_limit = 100
rerun_frame_limit = 100
rerun_start = 0
rerun_strip = 1
rerun_need_box_update = 0

[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/analysis.protocol.spgp.h5"

[input.h5.trajectory]
path = "runs/prod.spg.h5md"
particle_stream = "all"

[write.interval]
information = 1
mdout = 1
""",
    "xtc_trajectory": """
mode = "npt"
step_limit = 500000
dt = 0.002
pbc = true
target_temperature = 300.0
target_pressure = 1.0

trajectory_format = "xtc"

[xtc]
path = "prod.xtc"
precision = 1000.0
flush = true
buffer_frames = 10
frame_index = true

[write.interval]
trajectory = 1000
""",
}


class ValidationError(RuntimeError):
    pass


def type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def matches_type(value: Any, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate(schema: dict[str, Any], value: Any, path: str) -> None:
    if "anyOf" in schema:
        errors = []
        for candidate in schema["anyOf"]:
            try:
                validate(candidate, value, path)
                return
            except ValidationError as err:
                errors.append(str(err))
        raise ValidationError(f"{path}: value matches none of anyOf: {errors}")

    if "type" in schema and not matches_type(value, schema["type"]):
        raise ValidationError(
            f"{path}: expected {schema['type']}, got {type_name(value)}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(
            f"{path}: {value!r} is not in enum {schema['enum']!r}"
        )

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValidationError(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValidationError(
                    f"{path}: unknown key(s): {', '.join(extra)}"
                )
        for key, child in value.items():
            if key in properties:
                validate(properties[key], child, f"{path}.{key}")
        return

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate(item_schema, item, f"{path}[{index}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/mdin.schema.json"),
        help="Path to SPONGE mdin JSON schema.",
    )
    args = parser.parse_args()

    schema_path = args.schema.resolve()
    schema = json.loads(schema_path.read_text())
    for name, toml_text in EXAMPLES.items():
        data = tomllib.loads(toml_text)
        validate(schema, data, name)
        print(f"[PASS] {name}")
    repo_root = schema_path.parent.parent
    examples_dir = repo_root / "examples" / "h5_input"
    if examples_dir.exists():
        for example_path in sorted(examples_dir.glob("*.toml")):
            data = tomllib.loads(example_path.read_text())
            validate(schema, data, str(example_path.relative_to(repo_root)))
            print(f"[PASS] {example_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
