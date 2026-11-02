#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def parse_plan_labels(plan_text):
    match = re.search(
        r"## CI Labels\s+Use labels that allow staged execution:\s+```text\n"
        r"(?P<body>.*?)```",
        plan_text,
        re.DOTALL,
    )
    if not match:
        fail("failed to parse CI Labels block from plan")

    labels = []
    for line in match.group("body").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        labels.extend(
            part.strip() for part in stripped.split(";") if part.strip()
        )
    return sorted(set(labels))


def parse_current_runtime_boundary(plan_text):
    match = re.search(
        r"## Current Runtime Boundary\s+"
        r"(?P<body>.*?)(?:\n## |\Z)",
        plan_text,
        re.DOTALL,
    )
    if not match:
        fail("failed to parse Current Runtime Boundary from plan")
    return match.group("body")


def parse_phased_execution_checklist(plan_text):
    match = re.search(
        r"## Phased Execution Checklist\s+"
        r"(?P<body>.*?)\n## Input Fixture Generation",
        plan_text,
        re.DOTALL,
    )
    if not match:
        fail("failed to parse Phased Execution Checklist from plan")
    return match.group("body")


def parse_cmake_test_labels(cmake_text):
    labels_by_test = {}
    normalized = re.sub(r"\s+", " ", cmake_text)
    for match in re.finditer(
        r"set_tests_properties\(\s*(?P<name>\w+)\s+PROPERTIES\s+LABELS\s+"
        r'"(?P<labels>[^"]+)"',
        normalized,
    ):
        labels_by_test[match.group("name")] = set(
            match.group("labels").split(";")
        )
    if not labels_by_test:
        fail("failed to parse CTest labels from CMakeLists.txt")
    return labels_by_test


def parse_cmake_smoke_properties(cmake_text):
    normalized = re.sub(r"\s+", " ", cmake_text)
    properties = {}
    for name in [
        "test_h5_input_output_smoke_matrix",
        "test_h5_reaxff_edip_runtime_parity",
        "test_h5_restart_load_runtime_closure",
        "test_h5_vds_terminal_resume_smoke",
    ]:
        match = re.search(
            r"set_tests_properties\(\s*" + re.escape(name) + r"\s+"
            r"PROPERTIES\s+(?P<body>.*?)\)",
            normalized,
        )
        if not match:
            fail(f"failed to parse runtime smoke test properties for {name}")
        properties[name] = match.group("body")
    return properties


def parse_script_commands(script_text):
    match = re.search(
        r'case "\$\{command\}" in(?P<body>.*?)esac', script_text, re.DOTALL
    )
    if not match:
        fail("failed to parse command dispatch from run_h5_bundle_tests.sh")
    commands = set()
    for line in match.group("body").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        command_match = re.match(r"([A-Za-z0-9_|-]+)\)", stripped)
        if command_match:
            for command in command_match.group(1).split("|"):
                commands.add(command)
    if not commands:
        fail("failed to parse script commands")
    return commands


def parse_staged_labels(script_text):
    match = re.search(
        r"run_staged\(\)\s*\{(?P<body>.*?)\n\}", script_text, re.DOTALL
    )
    if not match:
        fail("failed to parse run_staged")
    labels = []
    for label in re.findall(
        r"run_label\s+([A-Za-z0-9_-]+)", match.group("body")
    ):
        labels.append(label)
    if not labels:
        fail("run_staged does not run any labels")
    return labels


def parse_script_build_targets(script_text):
    build_targets = {}
    for match in re.finditer(
        r"(?P<name>build_[A-Za-z0-9_]+)\(\)\s*\{(?P<body>.*?)\n\}",
        script_text,
        re.DOTALL,
    ):
        target_match = re.search(
            r"--target\s+([A-Za-z0-9_]+)", match.group("body")
        )
        if target_match:
            build_targets[match.group("name")] = target_match.group(1)
    return build_targets


def parse_script_staged_steps(script_text):
    match = re.search(
        r"run_staged\(\)\s*\{(?P<body>.*?)\n\}", script_text, re.DOTALL
    )
    if not match:
        fail("failed to parse run_staged")
    build_targets = parse_script_build_targets(script_text)
    steps = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "configure":
            steps.append(("configure",))
        elif line in build_targets:
            steps.append(("build", build_targets[line]))
        else:
            label_match = re.match(r"run_label\s+([A-Za-z0-9_-]+)$", line)
            if label_match:
                steps.append(("test", label_match.group(1)))
            else:
                fail(f"unrecognized run_staged step: {line}")
    return steps


def parse_expected_validation_steps(manifest_text):
    match = re.search(
        r"## Expected validation sequence\s+```bash\n(?P<body>.*?)```",
        manifest_text,
        re.DOTALL,
    )
    if not match:
        fail(
            "failed to parse Expected validation sequence from TEST_TARGETS.md"
        )
    steps = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("cmake -S "):
            steps.append(("configure",))
        elif line.startswith("cmake --build "):
            target_match = re.search(r"--target\s+([A-Za-z0-9_]+)", line)
            if not target_match:
                fail(f"build command lacks --target: {line}")
            steps.append(("build", target_match.group(1)))
        elif line.startswith("ctest "):
            labels = re.findall(r"-L\s+([A-Za-z0-9_-]+)", line)
            if not labels:
                fail(f"ctest command lacks -L label: {line}")
            if "h5_bundle" not in labels:
                fail(f"ctest command must include -L h5_bundle: {line}")
            stage_labels = [label for label in labels if label != "h5_bundle"]
            if len(stage_labels) != 1:
                fail(f"ctest command must include one staged label: {line}")
            steps.append(("test", stage_labels[0]))
        else:
            fail(f"unrecognized Expected validation sequence line: {line}")
    return steps


def require_plan_labels_registered(plan_labels, labels_by_test):
    all_cmake_labels = set()
    for name, labels in labels_by_test.items():
        if "h5_bundle" not in labels:
            fail(f"{name} lacks h5_bundle label")
        all_cmake_labels.update(labels)

    missing = sorted(set(plan_labels) - all_cmake_labels)
    if missing:
        fail(f"plan labels missing from CMake tests: {missing}")


def require_script_covers_staged_labels(staged_labels, script_commands):
    missing_commands = [
        f"test-{label}"
        for label in staged_labels
        if f"test-{label}" not in script_commands
    ]
    if missing_commands:
        fail(
            f"run_h5_bundle_tests.sh missing label commands: {missing_commands}"
        )


def require_script_label_filter_is_h5_bundle_scoped(script_text):
    match = re.search(
        r"run_label\(\)\s*\{(?P<body>.*?)\n\}", script_text, re.DOTALL
    )
    if not match:
        fail("failed to parse run_label")
    body = match.group("body")
    if "-L h5_bundle" not in body:
        fail("run_label must filter staged labels to h5_bundle tests")
    if '-L "${label}"' not in body:
        fail("run_label must still apply the requested staged label")


def require_runtime_smoke_gate(script_text, smoke_properties):
    if "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1" not in script_text:
        fail(
            "runtime smoke helper does not set SPONGE_H5_ENABLE_RUNTIME_SMOKE=1"
        )
    if (
        "-R" not in script_text
        or "test_h5_input_output_smoke_matrix" not in script_text
    ):
        fail(
            "runtime smoke helper does not run test_h5_input_output_smoke_matrix"
        )
    if "test_h5_reaxff_edip_runtime_parity" not in script_text:
        fail("runtime smoke helper does not run REAXFF/EDIP parity closure")
    if "test_h5_restart_load_runtime_closure" not in script_text:
        fail("runtime smoke helper does not run restart-load runtime closure")
    if "test_h5_vds_terminal_resume_smoke" not in script_text:
        fail("runtime smoke helper does not run VDS terminal/resume smoke")
    for name, properties in smoke_properties.items():
        if "SKIP_RETURN_CODE 77" not in properties:
            fail(f"{name} CTest does not use SKIP_RETURN_CODE 77")


def require_current_runtime_boundary_matches_ci(
    plan_text, script_text, smoke_text
):
    boundary = re.sub(r"\s+", " ", parse_current_runtime_boundary(plan_text))
    required_plan_tokens = [
        "default suite can prove parser, contract, fixture, reader, writer, "
        "static matrix, sidecar provenance, and manifest behavior",
        "final legacy-alignment claim still requires the gated runtime smoke",
        "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1",
        "ctest --test-dir build-h5-tests",
        "test_h5_input_output_smoke_matrix",
        "test_h5_reaxff_edip_runtime_parity",
        "test_h5_restart_load_runtime_closure",
        "test_h5_vds_terminal_resume_smoke",
        "--output-on-failure",
        "tests/h5_bundle/run_h5_bundle_tests.sh test-smoke-runtime",
    ]
    missing_plan_tokens = [
        token for token in required_plan_tokens if token not in boundary
    ]
    if missing_plan_tokens:
        fail(
            f"Current Runtime Boundary missing required tokens: {missing_plan_tokens}"
        )

    if "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1" not in script_text:
        fail(
            "runtime smoke helper command drifted from Current Runtime Boundary"
        )
    if (
        "-R" not in script_text
        or "test_h5_input_output_smoke_matrix" not in script_text
    ):
        fail(
            "runtime smoke helper target drifted from Current Runtime Boundary"
        )
    if "test_h5_reaxff_edip_runtime_parity" not in script_text:
        fail("runtime smoke helper omits REAXFF/EDIP parity target")
    if "test_h5_restart_load_runtime_closure" not in script_text:
        fail("runtime smoke helper omits restart-load closure target")
    if "test_h5_vds_terminal_resume_smoke" not in script_text:
        fail("runtime smoke helper omits VDS terminal/resume target")
    if "Validate_Runtime_Smoke_Preparation();" not in smoke_text:
        fail(
            "runtime smoke no longer validates default-suite wiring before gate"
        )
    if "Require_Runtime_Smoke_Enabled();" not in smoke_text:
        fail("runtime smoke no longer enforces Current Runtime Boundary gate")


def require_phased_execution_checklist_matches_ci(
    plan_text, labels_by_test, smoke_text
):
    checklist = parse_phased_execution_checklist(plan_text)
    normalized = re.sub(r"\s+", " ", checklist)

    required_phase_tokens = [
        "Fixture contract first",
        "Input behavior second",
        "Minimal normal smoke third",
        "Sidecar and override smoke fourth",
        "Rerun and VDS fifth",
        "Legal matrix last",
    ]
    missing_phase_tokens = [
        token for token in required_phase_tokens if token not in checklist
    ]
    if missing_phase_tokens:
        fail(
            f"Phased Execution Checklist missing phases: {missing_phase_tokens}"
        )

    required_test_targets = [
        "test_h5_io_contract_manifest",
        "test_h5_input_fixture_equivalence",
        "test_h5_input_matrix_contract",
        "test_h5_legacy_sidecar",
        "test_h5_input_validation",
        "test_h5_input_output_smoke_matrix",
    ]
    missing_checklist_targets = [
        target for target in required_test_targets if target not in checklist
    ]
    if missing_checklist_targets:
        fail(
            "Phased Execution Checklist missing test targets: "
            f"{missing_checklist_targets}"
        )

    missing_cmake_targets = [
        target
        for target in required_test_targets
        if target not in labels_by_test
    ]
    if missing_cmake_targets:
        fail(
            "Phased Execution Checklist references unknown CTest targets: "
            f"{missing_cmake_targets}"
        )

    required_command_tokens = [
        "ctest --test-dir build-h5-tests",
        "--output-on-failure",
        "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1",
        "test_h5_input_output_smoke_matrix",
        "test_h5_reaxff_edip_runtime_parity",
        "test_h5_restart_load_runtime_closure",
        "test_h5_vds_terminal_resume_smoke",
    ]
    missing_command_tokens = [
        token for token in required_command_tokens if token not in checklist
    ]
    if missing_command_tokens:
        fail(
            "Phased Execution Checklist missing command tokens: "
            f"{missing_command_tokens}"
        )

    required_behavior_tokens = [
        "Compare legacy and bundled output content",
        "process success alone is not enough",
        "Internal state checks are necessary",
        "every input family must have an end-to-end smoke row",
    ]
    missing_behavior_tokens = [
        token for token in required_behavior_tokens if token not in normalized
    ]
    if missing_behavior_tokens:
        fail(
            "Phased Execution Checklist missing behavioral requirements: "
            f"{missing_behavior_tokens}"
        )

    if "Require_Core_Mdout_Equivalent(" not in smoke_text:
        fail("runtime smoke lacks legacy/bundled core output comparison")
    if "Require_H5_Observable_Stream_Matches_Mdout(" not in smoke_text:
        fail("runtime smoke lacks bundled observable output comparison")


def require_staged_sequence_matches_manifest(
    script_text, targets_manifest_text
):
    script_steps = parse_script_staged_steps(script_text)
    manifest_steps = parse_expected_validation_steps(targets_manifest_text)
    if script_steps != manifest_steps:
        fail(
            "run_staged does not match TEST_TARGETS Expected validation sequence:\n"
            f"script={script_steps}\nmanifest={manifest_steps}"
        )


def require_smoke_preparation_before_gate(smoke_text):
    main_match = re.search(
        r"int\s+main\s*\([^)]*\)\s*\{(?P<body>.*)\n\}",
        smoke_text,
        re.DOTALL,
    )
    if not main_match:
        fail("failed to parse runtime smoke main()")
    main_body = main_match.group("body")
    prepare_pos = main_body.find("Validate_Runtime_Smoke_Preparation();")
    gate_pos = main_body.find("Require_Runtime_Smoke_Enabled();")
    normal_pos = main_body.find("Run_Normal_Mode_Matrix(")
    rerun_pos = main_body.find("Run_Rerun_Mode_Matrix(")
    if prepare_pos < 0:
        fail("runtime smoke main() does not validate preparation")
    if gate_pos < 0:
        fail("runtime smoke main() does not check runtime gate")
    if not prepare_pos < gate_pos:
        fail("runtime smoke gate runs before preparation validation")
    if normal_pos < 0 or rerun_pos < 0:
        fail("runtime smoke main() does not run both normal and rerun matrices")
    if not gate_pos < normal_pos < rerun_pos:
        fail("runtime smoke matrices do not run after the runtime gate")

    gate_match = re.search(
        r"void\s+Require_Runtime_Smoke_Enabled\s*\(\)\s*\{(?P<body>.*?)\n\}",
        smoke_text,
        re.DOTALL,
    )
    if not gate_match:
        fail("failed to parse Require_Runtime_Smoke_Enabled")
    gate_body = gate_match.group("body")
    if "SPONGE_H5_ENABLE_RUNTIME_SMOKE" not in gate_body:
        fail("runtime gate does not read SPONGE_H5_ENABLE_RUNTIME_SMOKE")
    if 'std::string(enabled) != "1"' not in gate_body:
        fail("runtime gate does not require SPONGE_H5_ENABLE_RUNTIME_SMOKE=1")
    if "std::exit(kSkipReturnCode)" not in gate_body:
        fail("runtime gate does not exit with the configured skip return code")


def require_manifest_guard_sources_are_wired(
    matrix_manifest_text,
    io_contract_manifest_text,
    cmake_text,
):
    matrix_tokens = [
        "parse_runtime_matrix_cases",
        "runtime_input_family_from_source",
        "Normal_Smoke_Cases",
        "Rerun_Legacy_Output_Cases",
        "Rerun_Bundled_Output_Cases",
        "parse_static_matrix_evidence",
        "require_static_matrix_evidence_is_explicit",
        "require_normal_smoke_branch_assertions",
        "require_rerun_smoke_branch_assertions",
        "require_sidecar_smoke_branch_assertions",
        "require_fixture_helper_describe_case_validates_inputs",
        "require_manybody_parity_closure",
        "Require_Path_Exists(paths.trajectory_h5)",
    ]
    missing_matrix_tokens = [
        token for token in matrix_tokens if token not in matrix_manifest_text
    ]
    if missing_matrix_tokens:
        fail(
            "matrix manifest source no longer protects runtime dimensions "
            f"or branch/helper assertions: {missing_matrix_tokens}"
        )

    if "--fixture-helper" not in cmake_text:
        fail("matrix manifest CTest command lacks --fixture-helper argument")
    if "h5_input_matrix_fixture.hpp" not in cmake_text:
        fail("matrix manifest CTest command lacks h5_input_matrix_fixture.hpp")
    if "--manybody-parity-source" not in cmake_text:
        fail(
            "matrix manifest CTest command lacks --manybody-parity-source argument"
        )
    if "test_h5_reaxff_edip_runtime_parity.cpp" not in cmake_text:
        fail("matrix manifest CTest command lacks REAXFF/EDIP parity source")

    io_contract_tokens = [
        "SEMANTIC_EQUIVALENCE_EVIDENCE",
        "require_required_inputs_have_semantic_equivalence_evidence",
        "compare_embedded_sidecar_text",
        "test_h5_input_fixture_equivalence.py",
    ]
    missing_io_tokens = [
        token
        for token in io_contract_tokens
        if token not in io_contract_manifest_text
    ]
    if missing_io_tokens:
        fail(
            "I/O contract manifest source no longer binds manifest inputs to "
            f"semantic equivalence evidence: {missing_io_tokens}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--smoke-matrix", required=True, type=Path)
    parser.add_argument("--matrix-manifest", required=True, type=Path)
    parser.add_argument("--io-contract-manifest", required=True, type=Path)
    parser.add_argument("--targets-manifest", required=True, type=Path)
    args = parser.parse_args()

    plan_text = args.plan.read_text()
    plan_labels = parse_plan_labels(plan_text)
    cmake_text = args.cmake.read_text()
    script_text = args.script.read_text()
    smoke_text = args.smoke_matrix.read_text()
    matrix_manifest_text = args.matrix_manifest.read_text()
    io_contract_manifest_text = args.io_contract_manifest.read_text()
    targets_manifest_text = args.targets_manifest.read_text()
    labels_by_test = parse_cmake_test_labels(cmake_text)
    script_commands = parse_script_commands(script_text)
    staged_labels = parse_staged_labels(script_text)
    smoke_properties = parse_cmake_smoke_properties(cmake_text)

    require_plan_labels_registered(plan_labels, labels_by_test)
    require_script_covers_staged_labels(staged_labels, script_commands)
    require_script_label_filter_is_h5_bundle_scoped(script_text)
    require_staged_sequence_matches_manifest(script_text, targets_manifest_text)
    require_runtime_smoke_gate(script_text, smoke_properties)
    require_current_runtime_boundary_matches_ci(
        plan_text, script_text, smoke_text
    )
    require_phased_execution_checklist_matches_ci(
        plan_text, labels_by_test, smoke_text
    )
    require_smoke_preparation_before_gate(smoke_text)
    require_manifest_guard_sources_are_wired(
        matrix_manifest_text,
        io_contract_manifest_text,
        cmake_text,
    )

    expected_core_commands = {
        "configure",
        "build-contract",
        "build-matrix",
        "build-smoke",
        "test-contract",
        "test-coverage",
        "test-matrix",
        "test-smoke",
        "test-smoke-runtime",
        "test-all",
        "staged",
    }
    missing_core_commands = sorted(expected_core_commands - script_commands)
    if missing_core_commands:
        fail(
            f"run_h5_bundle_tests.sh missing core commands: {missing_core_commands}"
        )


if __name__ == "__main__":
    main()
