#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def normalize_ws(text):
    return re.sub(r"\s+", " ", text)


def parse_cmake_targets(cmake_text):
    targets = {}
    for match in re.finditer(
        r"add_sponge_h5_bundle_test\(\s*(\w+)\s+([^\s\)]+)\s*\)",
        cmake_text,
        re.MULTILINE,
    ):
        targets[match.group(1)] = {
            "source": match.group(2),
            "labels": None,
        }

    for smoke_match in re.finditer(
        r"add_executable\(\s*"
        r"(test_h5_(?:input_output_smoke_matrix|"
        r"reaxff_edip_runtime_parity|"
        r"restart_load_runtime_closure))"
        r"\s+([^\s\)]+)",
        cmake_text,
        re.MULTILINE,
    ):
        targets[smoke_match.group(1)] = {
            "source": smoke_match.group(2),
            "labels": None,
        }

    for match in re.finditer(
        r"add_test\(\s*NAME\s+(\w+)\s+COMMAND\s+\$\{Python3_EXECUTABLE\}\s+"
        r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^\s\)]+)",
        cmake_text,
        re.MULTILINE,
    ):
        targets[match.group(1)] = {
            "source": match.group(2),
            "labels": None,
        }

    for match in re.finditer(
        r"set_tests_properties\(\s*(\w+)\s+PROPERTIES\s+LABELS\s+\"([^\"]+)\"",
        cmake_text,
        re.MULTILINE | re.DOTALL,
    ):
        name = match.group(1)
        if name in targets:
            targets[name]["labels"] = match.group(2)

    missing_labels = sorted(
        name for name, spec in targets.items() if spec["labels"] is None
    )
    if missing_labels:
        fail(f"CMake targets missing labels: {missing_labels}")
    return targets


def parse_cmake_build_targets(cmake_text):
    build_targets = {}
    for match in re.finditer(
        r"add_custom_target\(\s*(sponge_h5_bundle_\w+)\s+DEPENDS\s+(.*?)\)",
        cmake_text,
    ):
        name = match.group(1)
        depends = match.group(2).split()
        if "${SPONGE_H5_BUNDLE_TEST_TARGETS}" in depends:
            build_targets[name] = {"all_h5_bundle_tests": True, "depends": []}
        else:
            build_targets[name] = {
                "all_h5_bundle_tests": False,
                "depends": sorted(depends),
            }
    if not build_targets:
        fail("CMake build-only targets were not parsed")
    return build_targets


def parse_manifest_build_targets(manifest_text):
    rows = {}
    in_build_table = False
    for line in manifest_text.splitlines():
        if line.strip() == "## Build-only targets":
            in_build_table = True
            continue
        if in_build_table and line.startswith("## "):
            break
        if not in_build_table:
            continue
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        name = cells[0].strip("`")
        depends_cell = cells[1]
        if depends_cell == "all H5 bundle tests":
            rows[name] = {"all_h5_bundle_tests": True, "depends": []}
        else:
            rows[name] = {
                "all_h5_bundle_tests": False,
                "depends": sorted(re.findall(r"`([^`]+)`", depends_cell)),
            }
    return rows


def parse_manifest_targets(manifest_text):
    rows = {}
    in_ctest_table = False
    for line in manifest_text.splitlines():
        if line.strip() == "## CTest targets":
            in_ctest_table = True
            continue
        if in_ctest_table and line.startswith("## "):
            break
        if not in_ctest_table:
            continue
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        name = cells[0].strip("`")
        source = cells[1].strip("`")
        labels = cells[2].strip("`")
        rows[name] = {"source": source, "labels": labels}
    return rows


def require_manifest_matches_cmake(cmake_targets, manifest_targets):
    cmake_names = set(cmake_targets)
    manifest_names = set(manifest_targets)
    if cmake_names != manifest_names:
        fail(
            "CTest target manifest mismatch: "
            f"missing={sorted(cmake_names - manifest_names)} "
            f"extra={sorted(manifest_names - cmake_names)}"
        )

    for name, cmake_spec in sorted(cmake_targets.items()):
        manifest_spec = manifest_targets[name]
        if manifest_spec["source"] != cmake_spec["source"]:
            fail(
                f"{name} source mismatch: "
                f"manifest={manifest_spec['source']} cmake={cmake_spec['source']}"
            )
        if manifest_spec["labels"] != cmake_spec["labels"]:
            fail(
                f"{name} label mismatch: "
                f"manifest={manifest_spec['labels']} cmake={cmake_spec['labels']}"
            )
        if "h5_bundle" not in cmake_spec["labels"].split(";"):
            fail(f"{name} lacks h5_bundle label")


def require_build_targets_match_cmake(
    cmake_build_targets, manifest_build_targets, cmake_targets
):
    cmake_names = set(cmake_build_targets)
    manifest_names = set(manifest_build_targets)
    if cmake_names != manifest_names:
        fail(
            "Build-only target manifest mismatch: "
            f"missing={sorted(cmake_names - manifest_names)} "
            f"extra={sorted(manifest_names - cmake_names)}"
        )

    for name, cmake_spec in sorted(cmake_build_targets.items()):
        manifest_spec = manifest_build_targets[name]
        if (
            manifest_spec["all_h5_bundle_tests"]
            != cmake_spec["all_h5_bundle_tests"]
        ):
            fail(f"{name} all-H5-bundle-tests marker mismatch")
        if manifest_spec["depends"] != cmake_spec["depends"]:
            fail(
                f"{name} dependency mismatch: "
                f"manifest={manifest_spec['depends']} "
                f"cmake={cmake_spec['depends']}"
            )
        for dep in cmake_spec["depends"]:
            if dep not in cmake_targets:
                fail(f"{name} depends on unknown CTest target {dep}")
            if cmake_targets[dep]["source"].endswith(".py"):
                fail(
                    f"{name} should not depend on Python-only CTest target {dep}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    cmake_text = normalize_ws(args.cmake.read_text())
    manifest_text = args.manifest.read_text()
    cmake_targets = parse_cmake_targets(cmake_text)
    manifest_targets = parse_manifest_targets(manifest_text)
    cmake_build_targets = parse_cmake_build_targets(cmake_text)
    manifest_build_targets = parse_manifest_build_targets(manifest_text)
    require_manifest_matches_cmake(cmake_targets, manifest_targets)
    require_build_targets_match_cmake(
        cmake_build_targets, manifest_build_targets, cmake_targets
    )


if __name__ == "__main__":
    main()
