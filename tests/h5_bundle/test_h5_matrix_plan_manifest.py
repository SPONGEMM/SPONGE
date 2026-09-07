#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class MatrixCase:
    name: str
    mode: str
    input_family: str
    output_family: str
    vds: str


def fail(message):
    raise AssertionError(message)


def normalize_input_family(text):
    mapping = {
        "legacy": "legacy",
        "bundled": "bundled",
        "bundled with sidecar": "bundled_with_sidecar",
    }
    try:
        return mapping[text.strip()]
    except KeyError:
        fail(f"unknown input family in plan: {text}")


def parse_plan_cases(plan_text):
    cases = []
    in_table = False
    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped == "Minimum legal case set:":
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("This is the complete support matrix"):
            break
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or stripped.startswith("| Case "):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 5:
            continue
        cases.append(
            MatrixCase(
                name=cells[0].strip("`"),
                mode=cells[1],
                input_family=normalize_input_family(cells[2]),
                output_family=cells[3],
                vds=cells[4],
            )
        )

    if not cases:
        fail("failed to parse Minimum legal case set from plan")
    return cases


def cpp_vds_value(mode, output_family, vds_applicable, vds):
    if not vds_applicable:
        if output_family == "legacy":
            return "N/A"
        if mode == "normal" and output_family == "bundled":
            return "off"
        fail(
            "unexpected non-applicable VDS for "
            f"mode={mode} output={output_family}"
        )
    return "on" if vds else "off"


def static_matrix_case_matches(cpp_text):
    function_match = re.search(
        r"std::vector<MatrixCase>\s+Legal_Matrix_Cases\(\)\s*\{"
        r".*?return\s*\{(?P<body>.*?)\};\s*\}",
        cpp_text,
        re.DOTALL,
    )
    if not function_match:
        fail("failed to locate Legal_Matrix_Cases in static matrix test")

    matches = list(
        re.finditer(
            r'\{\s*"(?P<name>[^"]+)"\s*,\s*Mode::(?P<mode>\w+)\s*,'
            r"\s*InputFamily::(?P<input>\w+)\s*,"
            r"\s*OutputFamily::(?P<output>\w+)\s*,"
            r"\s*(?P<vds_applicable>true|false)\s*,"
            r"\s*(?P<vds>true|false)\s*,"
            r"\s*EvidenceType::(?P<evidence>\w+)\s*\}",
            function_match.group("body"),
            re.DOTALL,
        )
    )
    if not matches:
        fail("failed to parse cases from Legal_Matrix_Cases")
    return matches


def parse_static_matrix_cases(cpp_text):
    cases = []
    for match in static_matrix_case_matches(cpp_text):
        mode = match.group("mode")
        output_family = match.group("output")
        vds_applicable = match.group("vds_applicable") == "true"
        vds = match.group("vds") == "true"
        cases.append(
            MatrixCase(
                name=match.group("name"),
                mode=mode,
                input_family=match.group("input"),
                output_family=output_family,
                vds=cpp_vds_value(mode, output_family, vds_applicable, vds),
            )
        )

    return cases


def parse_static_matrix_evidence(cpp_text):
    evidence_by_name = {}
    for match in static_matrix_case_matches(cpp_text):
        name = match.group("name")
        if name in evidence_by_name:
            fail(f"duplicate static matrix evidence entry: {name}")
        evidence_by_name[name] = match.group("evidence")
    return evidence_by_name


def require_static_matrix_evidence_is_explicit(static_evidence, plan_cases):
    plan_names = {case.name for case in plan_cases}
    if set(static_evidence) != plan_names:
        fail(
            "static matrix evidence names do not match plan cases: "
            f"actual={sorted(static_evidence)} expected={sorted(plan_names)}"
        )
    allowed = {
        "static_contract",
        "prepared_mdin_guard",
        "runtime_smoke",
        "semantic_fixture_comparison",
    }
    unknown = sorted(set(static_evidence.values()) - allowed)
    if unknown:
        fail(f"static matrix evidence contains unknown types: {unknown}")
    not_runtime_smoke = {
        name: evidence
        for name, evidence in static_evidence.items()
        if evidence != "runtime_smoke"
    }
    if not_runtime_smoke:
        fail(
            "current legal matrix cases must be backed by runtime smoke: "
            f"{not_runtime_smoke}"
        )


def parse_runtime_expected_names(cpp_text):
    set_match = re.search(
        r"expected_runtime_case_names\s*=\s*\{(?P<body>.*?)\};",
        cpp_text,
        re.DOTALL,
    )
    if not set_match:
        fail("failed to locate expected_runtime_case_names in smoke test")
    names = sorted(set(re.findall(r'"([^"]+)"', set_match.group("body"))))
    if not names:
        fail("failed to parse expected_runtime_case_names")
    return names


def runtime_input_family_from_source(source):
    mapping = {
        "legacy_source": "legacy",
        "bundled_source": "bundled",
        "sidecar_source": "bundled_with_sidecar",
    }
    try:
        return mapping[source]
    except KeyError:
        fail(f"unknown runtime case source family: {source}")


def runtime_vds_value(mode, output_family, bundled_output, vds):
    if output_family == "legacy":
        return "N/A"
    if mode == "normal":
        if not bundled_output or vds != "false":
            fail(
                "normal runtime bundled output must use explicit VDS off: "
                f"bundled_output={bundled_output} vds={vds}"
            )
        return "off"
    return "on" if vds == "true" else "off"


def parse_runtime_matrix_cases(cpp_text):
    cases = [
        MatrixCase(
            "normal_legacy_in_legacy_out",
            "normal",
            "legacy",
            "legacy",
            "N/A",
        ),
        MatrixCase(
            "rerun_legacy_in_legacy_out",
            "rerun",
            "legacy",
            "legacy",
            "N/A",
        ),
    ]

    normal_body = extract_function_body(cpp_text, "Normal_Smoke_Cases")
    for match in re.finditer(
        r'\{\s*"(?P<name>[^"]+)"\s*,\s*(?P<source>\w+_source)\s*,'
        r'\s*"(?P<mdin>[^"]+)"\s*,\s*(?P<bundled>true|false)\s*\}',
        normal_body,
        re.DOTALL,
    ):
        bundled_output = match.group("bundled") == "true"
        cases.append(
            MatrixCase(
                match.group("name"),
                "normal",
                runtime_input_family_from_source(match.group("source")),
                "bundled" if bundled_output else "legacy",
                runtime_vds_value(
                    "normal",
                    "bundled" if bundled_output else "legacy",
                    bundled_output,
                    "false",
                ),
            )
        )

    rerun_legacy_body = extract_function_body(
        cpp_text, "Rerun_Legacy_Output_Cases"
    )
    for match in re.finditer(
        r'\{\s*"(?P<name>[^"]+)"\s*,\s*(?P<source>\w+_source)\s*,'
        r'\s*"(?P<mdin>[^"]+)"\s*\}',
        rerun_legacy_body,
        re.DOTALL,
    ):
        cases.append(
            MatrixCase(
                match.group("name"),
                "rerun",
                runtime_input_family_from_source(match.group("source")),
                "legacy",
                "N/A",
            )
        )

    rerun_bundled_body = extract_function_body(
        cpp_text, "Rerun_Bundled_Output_Cases"
    )
    for match in re.finditer(
        r'\{\s*"(?P<name>[^"]+)"\s*,\s*(?P<source>\w+_source)\s*,'
        r'\s*"(?P<mdin>[^"]+)"\s*,\s*(?P<vds>true|false)\s*\}',
        rerun_bundled_body,
        re.DOTALL,
    ):
        cases.append(
            MatrixCase(
                match.group("name"),
                "rerun",
                runtime_input_family_from_source(match.group("source")),
                "bundled",
                runtime_vds_value("rerun", "bundled", True, match.group("vds")),
            )
        )

    if len(cases) != 15:
        fail(f"runtime matrix should expose 15 legal cases, found {len(cases)}")
    seen = set()
    duplicates = []
    for case in cases:
        if case.name in seen:
            duplicates.append(case.name)
        seen.add(case.name)
    if duplicates:
        fail(f"runtime matrix exposes duplicate case names: {duplicates}")
    return sorted(cases, key=lambda case: case.name)


def parse_plan_runtime_exceptions(plan_text):
    exceptions = {}
    in_table = False
    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped == "Current runtime native-parity exceptions:":
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped:
            if exceptions:
                break
            continue
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or stripped.startswith("| Exception "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3:
            fail(
                f"runtime native-parity exception row must have 3 cells: {line}"
            )
        exception = cells[0].strip("`")
        scrub_behavior = cells[1]
        reason = cells[2]
        if not exception:
            continue
        if not scrub_behavior:
            fail(f"runtime exception lacks scrub behavior: {exception}")
        if not reason:
            fail(f"runtime exception lacks reason: {exception}")
        exceptions[exception] = {
            "scrub_behavior": scrub_behavior,
            "reason": reason,
        }
    if not exceptions:
        fail("failed to parse runtime native-parity exception table from plan")
    return exceptions


def parse_plan_pure_bundled_core_columns(plan_text):
    match = re.search(
        r"Pure bundled rerun native core mdout columns:\s+```text\n"
        r"(?P<body>.*?)```",
        plan_text,
        re.DOTALL,
    )
    if not match:
        fail("failed to parse pure bundled rerun native core mdout columns")
    columns = [
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip()
    ]
    if not columns:
        fail("pure bundled rerun native core mdout column list is empty")
    return columns


def parse_smoke_runtime_exceptions(cpp_text):
    function_match = re.search(
        r"void\s+Scrub_Runtime_Unstable_Rerun_Features\([^)]*\)\s*\{"
        r"(?P<body>.*?)\nvoid\s+Require_No_Legacy_Sidecar_Directories",
        cpp_text,
        re.DOTALL,
    )
    if not function_match:
        fail("failed to locate Scrub_Runtime_Unstable_Rerun_Features")
    body = function_match.group("body")
    exceptions = set()
    if re.search(r'Remove_Toml_Section\(\s*mdin,\s*"REAXFF"\s*\)', body):
        exceptions.add("REAXFF")
    if re.search(
        r'Remove_Key_Lines\(\s*mdin,\s*\{\s*"EDIP_in_file"\s*\}', body
    ):
        exceptions.add("EDIP_in_file")
    if (
        "input_h5_restart_path" in body
        and "input_h5_restart_load" in body
        and "structural" in body
    ):
        exceptions.add("input_h5_restart_load")
    if not exceptions:
        fail("failed to parse runtime native-parity exceptions from smoke test")
    return exceptions


def parse_smoke_pure_bundled_core_columns(smoke_text):
    body = extract_void_function_body(
        smoke_text, "Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent"
    )
    call_match = re.search(
        r"Require_Mdout_Columns_Equivalent\([^;]*?\{(?P<body>.*?)\}\s*\)",
        body,
        re.DOTALL,
    )
    if not call_match:
        fail("failed to parse pure bundled rerun mdout column list from smoke")
    columns = re.findall(r'"([^"]+)"', call_match.group("body"))
    if not columns:
        fail("pure bundled rerun smoke column list is empty")
    return columns


def require_pure_bundled_core_columns_match_plan(plan_text, smoke_text):
    require_equal(
        "pure bundled rerun native core mdout columns",
        parse_smoke_pure_bundled_core_columns(smoke_text),
        parse_plan_pure_bundled_core_columns(plan_text),
    )


def require_runtime_exception_scrub_behavior(plan_text, smoke_text):
    exceptions = parse_plan_runtime_exceptions(plan_text)
    expected_plan_phrases = {
        "input_h5_restart_load": [
            "Force H5 rerun restart loading to `structural` in prepared broad runtime smoke mdin",
            "Restart-load policy runtime closure is covered by `test_h5_restart_load_runtime_closure.cpp`",
        ],
    }
    if set(exceptions) != set(expected_plan_phrases):
        fail(f"unexpected runtime exception set: {sorted(exceptions)}")
    for exception, phrases in expected_plan_phrases.items():
        row_text = (
            exceptions[exception]["scrub_behavior"]
            + " "
            + exceptions[exception]["reason"]
        )
        missing = [phrase for phrase in phrases if phrase not in row_text]
        if missing:
            fail(
                f"runtime exception {exception} missing plan phrases: {missing}"
            )

    scrub_body = extract_void_function_body(
        smoke_text, "Scrub_Runtime_Unstable_Rerun_Features"
    )
    required_scrub_tokens = {
        "input_h5_restart_load": [
            'Remove_Key_Lines(mdin, {"input_h5_restart_load"})',
            'Append_If_Missing(&mdin, "input_h5_restart_load"',
            "structural",
            'Has_Key_Line(mdin, "input_h5_restart_path")',
        ],
    }
    for exception, tokens in required_scrub_tokens.items():
        missing = [token for token in tokens if token not in scrub_body]
        if missing:
            fail(f"runtime scrub for {exception} missing tokens: {missing}")

    preparation_body = extract_void_function_body(
        smoke_text, "Validate_Runtime_Smoke_Preparation"
    )
    require_tokens(
        "runtime scrub preparation validation",
        preparation_body,
        [
            "rerun_runtime_scrub_prepare_check",
            "Scrub_Runtime_Unstable_Rerun_Features(",
            "Require_Runtime_Unstable_Rerun_Features_Scrubbed(",
        ],
    )


def extract_function_body(cpp_text, function_name):
    match = re.search(
        r"\b" + re.escape(function_name) + r"\s*\([^)]*\)\s*\{",
        cpp_text,
    )
    if not match:
        fail(f"failed to locate {function_name}")

    body_start = match.end()
    depth = 1
    pos = body_start
    while pos < len(cpp_text):
        char = cpp_text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cpp_text[body_start:pos]
        pos += 1
    fail(f"failed to parse body for {function_name}")


def extract_void_function_body(cpp_text, function_name):
    return extract_function_body(cpp_text, function_name)


def extract_sidecar_switch_case_body(sidecar_body, kind):
    start_token = f"case SidecarSmokeKind::{kind}:"
    start = sidecar_body.find(start_token)
    if start < 0:
        fail(f"failed to locate sidecar smoke branch: {kind}")
    start += len(start_token)
    next_start = sidecar_body.find("case SidecarSmokeKind::", start)
    if next_start >= 0:
        return sidecar_body[start:next_start]
    switch_end = sidecar_body.rfind("}")
    if switch_end < start:
        fail(f"failed to locate end of sidecar smoke branch: {kind}")
    return sidecar_body[start:switch_end]


def require_tokens(label, body, tokens):
    missing = [token for token in tokens if token not in body]
    if missing:
        fail(f"{label} missing tokens: {missing}")


def require_tokens_ignoring_whitespace(label, body, tokens):
    compact_body = "".join(body.split())
    missing = [
        token for token in tokens if "".join(token.split()) not in compact_body
    ]
    if missing:
        fail(f"{label} missing tokens: {missing}")


def require_runtime_smoke_execution_links(smoke_text):
    normal_body = extract_void_function_body(
        smoke_text, "Run_Normal_Mode_Matrix"
    )
    if "Normal_Smoke_Cases(" not in normal_body:
        fail("normal-mode smoke matrix no longer uses Normal_Smoke_Cases")
    if "Run_Legacy_Sidecar_Smoke_Cases(" not in normal_body:
        fail("Phase 3 sidecar smoke is not executed from normal-mode matrix")

    sidecar_body = extract_void_function_body(
        smoke_text, "Run_Legacy_Sidecar_Smoke_Cases"
    )
    if "Sidecar_Smoke_Cases(" not in sidecar_body:
        fail("sidecar smoke execution is not bound to Sidecar_Smoke_Cases")
    for token in [
        "injected_without_explicit_keys",
        "same_key_same_path",
        "same_key_different_path",
        "pure_bundled_without_sidecar_files",
    ]:
        if token not in sidecar_body:
            fail(f"sidecar smoke execution no longer handles {token}")

    rerun_body = extract_void_function_body(smoke_text, "Run_Rerun_Mode_Matrix")
    if "Rerun_Legacy_Output_Cases(" not in rerun_body:
        fail("rerun smoke matrix no longer uses Rerun_Legacy_Output_Cases")
    if "Rerun_Bundled_Output_Cases(" not in rerun_body:
        fail("rerun smoke matrix no longer uses Rerun_Bundled_Output_Cases")
    if "Run_Rerun_Frame_Selection_Smoke_Cases(" not in rerun_body:
        fail("Phase 4 frame-selection smoke is not executed from rerun matrix")
    if "if (spec.vds)" not in rerun_body:
        fail("rerun bundled-output smoke no longer branches on VDS")
    if "Require_VDS_Shards_Are_Complete(" not in rerun_body:
        fail("VDS-on rerun smoke no longer checks complete shard output")

    selection_body = extract_void_function_body(
        smoke_text, "Run_Rerun_Frame_Selection_Smoke_Cases"
    )
    if "Rerun_Selection_Cases(" not in selection_body:
        fail(
            "rerun frame-selection smoke is not bound to Rerun_Selection_Cases"
        )
    if "rerun_legacy_second_frame_only_legacy_out" not in selection_body:
        fail("rerun frame-selection smoke lacks legacy second-frame baseline")


def require_sidecar_smoke_branch_assertions(smoke_text):
    sidecar_body = extract_void_function_body(
        smoke_text, "Run_Legacy_Sidecar_Smoke_Cases"
    )
    branch_requirements = {
        "injected_without_explicit_keys": [
            "Run_SPONGE(",
            "Require_Normal_Legacy_Restart_Output(",
            "Require_Text_Equivalent(",
        ],
        "same_key_same_path": [
            "Run_SPONGE(",
            "Require_Normal_Legacy_Restart_Output(",
            "Require_Text_Equivalent(",
        ],
        "same_key_different_path": [
            "Run_SPONGE_Expect_Failure(",
            "mass_in_file is also set",
            "Native H5 topology data and ",
            "legacy text topology input cannot both own atom masses",
        ],
        "pure_bundled_without_sidecar_files": [
            "Remove_Legacy_Sidecar_Directories(",
            "Run_SPONGE(",
            "Require_Normal_Legacy_Restart_Output(",
            "Require_Core_Mdout_Equivalent(",
        ],
    }
    for kind, tokens in branch_requirements.items():
        require_tokens(
            f"sidecar smoke branch {kind}",
            extract_sidecar_switch_case_body(sidecar_body, kind),
            tokens,
        )


def require_normal_smoke_branch_assertions(smoke_text):
    normal_body = extract_void_function_body(
        smoke_text, "Run_Normal_Mode_Matrix"
    )
    require_tokens(
        "normal smoke baseline",
        normal_body,
        [
            '"normal_legacy_in_legacy_out"',
            "legacy_source",
            '"mdin.spg.toml"',
            "false);",
            "Run_SPONGE(sponge_executable, baseline)",
            "Require_Normal_Legacy_Restart_Output(baseline)",
        ],
    )
    require_tokens(
        "normal smoke bundled-input comparison branch",
        normal_body,
        [
            'Starts_With(std::string(spec.name), "normal_bundled_in_")',
            "Require_Core_Mdout_Equivalent(baseline.mdout, test_case.mdout)",
        ],
    )
    require_tokens(
        "normal smoke strict text comparison branch",
        normal_body,
        [
            "Require_Text_Equivalent(baseline.mdout, test_case.mdout)",
        ],
    )
    require_tokens(
        "normal smoke bundled-output readback branch",
        normal_body,
        [
            "if (spec.bundled_output)",
            "Require_H5_Restart_Matches_Core_State(test_case.h5_restart)",
            "Require_H5_Trajectory_Has_Frames(test_case.h5_trajectory, {1}",
            "Require_H5_Trajectory_First_Frame_Matches_Core_State(",
            "Require_H5_Observable_Stream_Matches_Mdout(",
            "test_case.h5_observable",
        ],
    )
    require_tokens(
        "normal smoke legacy-output restart branch",
        normal_body,
        [
            "else",
            "Require_Normal_Legacy_Restart_Output(test_case)",
        ],
    )


def require_rerun_smoke_branch_assertions(smoke_text):
    rerun_body = extract_void_function_body(smoke_text, "Run_Rerun_Mode_Matrix")
    require_tokens(
        "rerun smoke baseline",
        rerun_body,
        [
            '"rerun_legacy_in_legacy_out"',
            "Scrub_Runtime_Unstable_Rerun_Features(baseline)",
            "Run_SPONGE(sponge_executable, baseline)",
        ],
    )
    require_tokens(
        "rerun legacy-output branch",
        rerun_body,
        [
            "Rerun_Legacy_Output_Cases(",
            'Starts_With(std::string(spec.name), "rerun_bundled_")',
            "Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(baseline.mdout",
            "Require_Rerun_Mdout_Equivalent(baseline.mdout, test_case.mdout)",
        ],
    )
    require_tokens(
        "rerun bundled-output branch",
        rerun_body,
        [
            "Rerun_Bundled_Output_Cases(",
            "spec.vds",
            'Starts_With(std::string(spec.name), "rerun_bundled_")',
            'Starts_With(std::string(spec.name), "rerun_sidecar_")',
            "Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(",
            "Require_Rerun_Mdout_Equivalent(baseline.mdout",
            "Require_H5_Trajectory_Has_Frames(sidecar_bundled.h5_trajectory",
            "Require_H5_Trajectory_Frame_Matches_Rerun_Runtime_State(",
            "Require_H5_Observable_Stream_Matches_Mdout(",
            "sidecar_bundled.h5_observable",
        ],
    )
    require_tokens(
        "rerun VDS branch",
        rerun_body,
        [
            "if (spec.vds)",
            "Require_VDS_Shards_Are_Complete(sidecar_bundled.h5_trajectory",
            "trajectory_times, {0}",
            "{observable_times.front()}",
        ],
    )
    require_tokens(
        "rerun VDS and non-VDS trajectory observables",
        rerun_body,
        [
            "Require_H5_Observable_Stream_Matches_Mdout(",
            "sidecar_bundled.h5_trajectory, sidecar_bundled.mdout, {0, 1}",
            "observable_times",
        ],
    )


def require_runtime_smoke_output_assertions(smoke_text):
    legacy_restart_body = extract_void_function_body(
        smoke_text, "Require_Normal_Legacy_Restart_Output"
    )
    require_tokens(
        "normal legacy-output restart existence helper",
        legacy_restart_body,
        [
            '"restart_coordinate.txt"',
            '"restart_velocity.txt"',
            "SpongeH5InputMatrix::Require_Path_Exists(",
        ],
    )

    normal_body = extract_void_function_body(
        smoke_text, "Run_Normal_Mode_Matrix"
    )
    for token in [
        "Require_Core_Mdout_Equivalent(",
        "Require_Text_Equivalent(",
        "Require_Normal_Legacy_Restart_Output(",
        "Require_H5_Restart_Matches_Core_State(",
        "Require_H5_Trajectory_Has_Frames(",
        "Require_H5_Trajectory_First_Frame_Matches_Core_State(",
        "Require_H5_Observable_Stream_Matches_Mdout(",
    ]:
        if token not in normal_body:
            fail(f"normal-mode smoke lacks output assertion {token}")

    sidecar_body = extract_void_function_body(
        smoke_text, "Run_Legacy_Sidecar_Smoke_Cases"
    )
    for token in [
        "Require_Text_Equivalent(",
        "Require_Core_Mdout_Equivalent(",
        "Run_SPONGE_Expect_Failure(",
        "mass_in_file is also set",
        "Native H5 topology data and ",
        "legacy text topology input cannot both own atom masses",
        "Require_Normal_Legacy_Restart_Output(",
    ]:
        if token not in sidecar_body:
            fail(f"sidecar smoke lacks output/failure assertion {token}")

    rerun_body = extract_void_function_body(smoke_text, "Run_Rerun_Mode_Matrix")
    for token in [
        "Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(",
        "Require_Rerun_Mdout_Equivalent(",
        "Require_H5_Trajectory_Has_Frames(",
        "Require_H5_Observable_Stream_Matches_Mdout(",
        "Require_VDS_Shards_Are_Complete(",
    ]:
        if token not in rerun_body:
            fail(f"rerun smoke lacks output assertion {token}")

    vds_body = extract_void_function_body(
        smoke_text, "Require_VDS_Shards_Are_Complete"
    )
    for token in [
        "output_status",
        "output_vds_status",
        "output_repair_policy",
        "output_repair_status",
        "output_repaired_shard_count",
        "finalized",
        "particle, observable, and module virtual datasets",
        "not_applied",
        "HighFive::File shard_file",
        'Read_Int64_Vector(shard_file, "/particles/all/step")',
        'Read_Float64_Vector(shard_file, "/particles/all/time")',
        'Read_Float_Vector(file, "/particles/all/position/value")',
        'Read_Float_Vector(shard_file, "/particles/all/position/value")',
        'Read_Float_Vector(shard_file, "/particles/all/box/edges/value")',
        "expected_steps[expected_index]",
        "expected_shard_step_starts[i]",
        "expected_shard_time_starts[i]",
        '"VDS shard position"',
        '"VDS shard box"',
    ]:
        if token not in vds_body:
            fail(f"VDS runtime smoke lacks wrapper status assertion {token}")

    selection_body = extract_void_function_body(
        smoke_text, "Run_Rerun_Frame_Selection_Smoke_Cases"
    )
    for token in [
        "Require_Mdout_Row_Count(",
        "Require_Rerun_Selection_Mdout_Equivalent(",
    ]:
        if token not in selection_body:
            fail(f"rerun frame-selection smoke lacks output assertion {token}")


def require_sidecar_override_preparation_locks_native_mass_conflict(smoke_text):
    prep_body = extract_void_function_body(
        smoke_text, "Validate_Runtime_Smoke_Preparation"
    )
    required_tokens = [
        "same_key_different_path",
        'mass_in_file = \\"override_mass.txt\\"',
        "override_mass.txt",
        "Legacy_Sidecar_Path_For_Key(",
        '"mass_in_file"',
        'topology.exist("/atoms/mass")',
    ]
    for token in required_tokens:
        if token not in prep_body:
            fail(
                "sidecar different-path preparation no longer locks native "
                f"mass conflict token: {token}"
            )


def require_sidecar_preparation_locks_success_paths(smoke_text):
    prep_body = extract_void_function_body(
        smoke_text, "Validate_Runtime_Smoke_Preparation"
    )
    require_tokens(
        "sidecar absent-key injection preparation",
        prep_body,
        [
            "injected_without_explicit_keys",
            "!Has_Key_Line(mdin, key)",
            '"mass_in_file"',
            '"charge_in_file"',
            '"qc_type_in_file"',
            '"cv_in_file"',
            '"restrain_in_file"',
            '"SITS_in_file"',
            "Legacy_Sidecar_Path_For_Key(",
            "topology.spgt.h5",
            "protocol.spgp.h5",
            "Require_Materialized_Sidecars_Are_Exactly_H5_Referenced(",
        ],
    )
    require_tokens(
        "sidecar same-path preparation",
        prep_body,
        [
            "same_key_same_path",
            "qc_type_in_file = ",
            "legacy_sidecars/qc_type_in_file/qc_type.txt",
            "Legacy_Sidecar_Path_For_Key(",
            '"qc_type_in_file"',
        ],
    )
    require_tokens(
        "pure bundled without sidecars preparation",
        prep_body,
        [
            "pure_bundled_without_sidecar_files",
            "Remove_Legacy_Sidecar_Directories(",
            "Require_No_Legacy_Sidecar_Directories(",
        ],
    )


def require_normal_preparation_locks_bundled_output_sidecar_defaults(
    smoke_text,
):
    body = extract_void_function_body(
        smoke_text, "Require_Normal_Prepared_Mdin"
    )
    require_tokens(
        "normal bundled-output preparation legacy sidecar defaults",
        body,
        [
            'Require_Contains(mdin, "mdout = \\"mdout.txt\\"")',
            'Require_Contains(mdin, "mdinfo = \\"mdinfo.txt\\"")',
            'Require_Contains(mdin, "write_trajectory_interval = 1")',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "crd"))',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "box"))',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "vel"))',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "frc"))',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "rst"))',
            'REQUIRE_TRUE(!Has_Key_Line(mdin, "qc_scf_output"))',
            "output_h5_restart_path",
            "output_h5_trajectory_path",
            "output_h5_observable_path",
        ],
    )


def require_bundled_output_runtime_rejects_default_legacy_files(smoke_text):
    helper_body = extract_void_function_body(
        smoke_text, "Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs"
    )
    require_tokens(
        "bundled-output default legacy runtime output rejection helper",
        helper_body,
        [
            '"mdcrd.dat"',
            '"mdbox.txt"',
            '"restart"',
            "std::filesystem::exists(test_case.root / filename)",
        ],
    )
    normal_body = extract_void_function_body(
        smoke_text, "Run_Normal_Mode_Matrix"
    )
    require_tokens(
        "normal bundled-output default legacy runtime output rejection",
        normal_body,
        [
            "if (spec.bundled_output)",
            "Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs(test_case)",
            "Require_H5_Restart_Matches_Core_State(",
            "Require_H5_Trajectory_First_Frame_Matches_Core_State(",
        ],
    )
    rerun_body = extract_void_function_body(smoke_text, "Run_Rerun_Mode_Matrix")
    require_tokens_ignoring_whitespace(
        "rerun bundled-output default legacy runtime output rejection",
        rerun_body,
        [
            "Rerun_Bundled_Output_Cases(",
            "Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs(sidecar_bundled)",
            "Require_H5_Trajectory_Frame_Matches_Rerun_Runtime_State(",
            "Require_H5_Observable_Stream_Matches_Mdout(",
        ],
    )


def require_plan_rerun_restart_policy_matches_smoke(plan_text, smoke_text):
    expected_plan_text = (
        "Bundled rerun output writes trajectory and observable H5 artifacts; "
        "normal bundled-output smoke covers restart H5 artifacts."
    )
    if expected_plan_text not in re.sub(r"\s+", " ", plan_text):
        fail(
            "plan must state that rerun bundled output does not write restart H5"
        )

    function_match = re.search(
        r"void\s+Require_Rerun_Prepared_Mdin\([^)]*\)\s*\{"
        r"(?P<body>.*?)\n\}",
        smoke_text,
        re.DOTALL,
    )
    if not function_match:
        fail("failed to locate Require_Rerun_Prepared_Mdin")
    body = function_match.group("body")
    if 'Has_Key_Line(mdin, "output_h5_restart_path")' not in body:
        fail("rerun preparation test does not check output_h5_restart_path")
    if (
        'REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_restart_path"))'
        not in body
    ):
        fail("rerun bundled-output preparation must reject H5 restart output")


def require_bundled_input_preparation_rejects_legacy_input_keys(smoke_text):
    restart_body = extract_void_function_body(
        smoke_text, "Require_No_Legacy_Restart_Input_Keys"
    )
    require_tokens(
        "legacy restart input key rejection helper",
        restart_body,
        [
            '"coordinate_in_file"',
            '"velocity_in_file"',
            '"rst7"',
            "REQUIRE_TRUE(!Has_Key_Line(mdin, key))",
        ],
    )
    if '"rst"' in restart_body:
        fail(
            "legacy restart output key rst must not be treated as an input key"
        )
    rerun_body = extract_void_function_body(
        smoke_text, "Require_No_Legacy_Rerun_Input_Keys"
    )
    require_tokens(
        "legacy rerun input key rejection helper",
        rerun_body,
        [
            '"crd"',
            '"box"',
            '"vel"',
            "REQUIRE_TRUE(!Has_Key_Line(mdin, key))",
        ],
    )
    for function_name in [
        "Require_Normal_Bundled_Input_Mdin",
        "Require_Rerun_Bundled_Input_Mdin",
    ]:
        body = extract_void_function_body(smoke_text, function_name)
        require_tokens(
            f"{function_name} legacy input key rejection calls",
            body,
            [
                "Require_No_Legacy_Restart_Input_Keys(mdin)",
                "Require_No_Legacy_Rerun_Input_Keys(mdin)",
            ],
        )


def require_fixture_helper_describe_case_validates_inputs(helper_text):
    match = re.search(
        r"inline\s+FixtureCasePaths\s+Describe_Case\([^)]*\)\s*\{"
        r"(?P<body>.*?)\n\}",
        helper_text,
        re.DOTALL,
    )
    if not match:
        fail("failed to locate Describe_Case in fixture helper")
    require_tokens(
        "fixture helper Describe_Case input validation",
        match.group("body"),
        [
            "Require_Path_Exists(root)",
            "Require_Path_Exists(paths.mdin)",
            "Require_Path_Exists(paths.topology_h5)",
            "Require_Path_Exists(paths.protocol_h5)",
            "Require_Path_Exists(paths.restart_h5)",
            "Require_Path_Exists(paths.trajectory_h5)",
            "Normal_Output_Paths(root)",
            "Rerun_Output_Paths(root)",
        ],
    )


def require_manybody_parity_closure(plan_text, manybody_text):
    require_tokens(
        "REAXFF/EDIP runtime parity plan",
        plan_text,
        [
            "test_h5_reaxff_edip_runtime_parity",
            "Bundled-with-sidecar and pure bundled native REAXFF/EDIP rerun input",
            "no-many-body-scrub closure tests",
        ],
    )
    require_tokens(
        "REAXFF/EDIP runtime parity test",
        manybody_text,
        [
            "Manybody_Columns()",
            '"EDIP"',
            '"REAXFF_BOND"',
            '"REAXFF_VDW"',
            '"REAXFF_EEQ"',
            '"REAXFF_ELP"',
            '"REAXFF_OVUN"',
            '"REAXFF_ANG"',
            '"REAXFF_PEN"',
            '"REAXFF_COA"',
            '"REAXFF_TOR"',
            '"REAXFF_CONJ"',
            '"REAXFF_HB"',
            '"REAXFF"',
            "manybody_legacy_in_legacy_out_no_manybody_scrub",
            "manybody_sidecar_in_legacy_out_no_manybody_scrub",
            "manybody_sidecar_in_bundled_out_vds_",
            "manybody_pure_bundled_in_legacy_out_no_sidecar",
            "manybody_pure_bundled_in_bundled_out_vds_",
            "Require_No_Legacy_Sidecar_Directories(",
            "Remove_Legacy_Sidecar_Directories(",
            "Require_Mdout_Columns_Equivalent(",
            "Require_H5_Observable_Columns_Match_Mdout(",
            "Require_VDS_Wrapper_Finalized(",
            'input_h5_restart_load = \\"structural\\"',
            "Require_Manybody_Not_Scrubbed(",
        ],
    )
    forbidden_tokens = [
        "Remove_Toml_Section(",
        'Remove_Key_Lines(mdin, {"EDIP_in_file"})',
        '"REAXFF_in_file", "REAXFF_type_in_file"',
    ]
    found = [token for token in forbidden_tokens if token in manybody_text]
    if found:
        fail(
            f"REAXFF/EDIP sidecar parity test contains many-body scrub: {found}"
        )


def require_equal(label, actual, expected):
    if actual != expected:
        fail(f"{label} mismatch:\nactual={actual}\nexpected={expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--matrix-spec", required=True, type=Path)
    parser.add_argument("--smoke-matrix", required=True, type=Path)
    parser.add_argument("--manybody-parity-source", required=True, type=Path)
    parser.add_argument("--fixture-helper", required=True, type=Path)
    args = parser.parse_args()

    plan_text = args.plan.read_text()
    matrix_spec_text = args.matrix_spec.read_text()
    smoke_text = args.smoke_matrix.read_text()
    manybody_text = args.manybody_parity_source.read_text()
    helper_text = args.fixture_helper.read_text()
    plan_cases = parse_plan_cases(plan_text)
    static_cases = parse_static_matrix_cases(matrix_spec_text)
    runtime_names = parse_runtime_expected_names(smoke_text)
    plan_names = sorted(case.name for case in plan_cases)

    require_equal("static matrix cases", static_cases, plan_cases)
    require_static_matrix_evidence_is_explicit(
        parse_static_matrix_evidence(matrix_spec_text),
        plan_cases,
    )
    require_equal(
        "runtime matrix cases",
        parse_runtime_matrix_cases(smoke_text),
        sorted(plan_cases, key=lambda case: case.name),
    )
    allowed_extra_runtime = {
        "rerun_legacy_second_frame_only_legacy_out",
        "rerun_sidecar_second_frame_only_legacy_out",
    }
    unexpected_runtime = sorted(
        set(runtime_names) - set(plan_names) - allowed_extra_runtime
    )
    if unexpected_runtime:
        fail(
            f"runtime smoke includes cases outside legal matrix: {unexpected_runtime}"
        )
    required_smoke_names = {
        "normal_legacy_in_legacy_out",
        "normal_legacy_in_bundled_out",
        "normal_bundled_in_legacy_out",
        "normal_sidecar_in_legacy_out",
        "normal_bundled_in_bundled_out",
        "normal_sidecar_in_bundled_out",
        "rerun_legacy_in_legacy_out",
        "rerun_bundled_in_legacy_out",
        "rerun_legacy_in_bundled_out_vds_off",
        "rerun_legacy_in_bundled_out_vds_on",
        "rerun_bundled_in_bundled_out_vds_off",
        "rerun_bundled_in_bundled_out_vds_on",
        "rerun_sidecar_in_bundled_out_vds_off",
        "rerun_sidecar_in_bundled_out_vds_on",
        "rerun_sidecar_in_legacy_out",
        "rerun_legacy_second_frame_only_legacy_out",
        "rerun_sidecar_second_frame_only_legacy_out",
    }
    missing_runtime = sorted(required_smoke_names - set(runtime_names))
    if missing_runtime:
        fail(f"runtime smoke missing required staged cases: {missing_runtime}")

    require_equal(
        "runtime native-parity exceptions",
        parse_smoke_runtime_exceptions(smoke_text),
        set(parse_plan_runtime_exceptions(plan_text)),
    )
    require_pure_bundled_core_columns_match_plan(plan_text, smoke_text)
    require_runtime_exception_scrub_behavior(plan_text, smoke_text)
    require_runtime_smoke_execution_links(smoke_text)
    require_normal_smoke_branch_assertions(smoke_text)
    require_rerun_smoke_branch_assertions(smoke_text)
    require_sidecar_smoke_branch_assertions(smoke_text)
    require_runtime_smoke_output_assertions(smoke_text)
    require_sidecar_override_preparation_locks_native_mass_conflict(smoke_text)
    require_sidecar_preparation_locks_success_paths(smoke_text)
    require_normal_preparation_locks_bundled_output_sidecar_defaults(smoke_text)
    require_bundled_output_runtime_rejects_default_legacy_files(smoke_text)
    require_plan_rerun_restart_policy_matches_smoke(plan_text, smoke_text)
    require_manybody_parity_closure(plan_text, manybody_text)
    require_bundled_input_preparation_rejects_legacy_input_keys(smoke_text)
    require_fixture_helper_describe_case_validates_inputs(helper_text)


if __name__ == "__main__":
    main()
