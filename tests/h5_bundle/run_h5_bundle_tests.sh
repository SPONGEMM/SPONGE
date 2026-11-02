#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${SPONGE_H5_BUNDLE_TEST_BUILD_DIR:-}" ]]; then
  BUILD_DIR="${SPONGE_H5_BUNDLE_TEST_BUILD_DIR}"
elif [[ -n "${PIXI_ENVIRONMENT_NAME:-}" ]]; then
  BUILD_DIR="${ROOT_DIR}/build-${PIXI_ENVIRONMENT_NAME}"
else
  BUILD_DIR="${ROOT_DIR}/build-h5-tests"
fi
CMAKE_BIN="${CMAKE:-cmake}"
CTEST_BIN="${CTEST:-ctest}"

configure() {
  "${CMAKE_BIN}" -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DSPONGE_BUILD_TESTS=ON
}

build_contract() {
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target sponge_h5_bundle_contract_tests
}

build_mock() {
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target sponge_h5_bundle_mock_tests
}

build_backend_io() {
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target sponge_h5_bundle_backend_io_tests
}

build_smoke() {
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target sponge_h5_bundle_smoke_tests
}

build_matrix() {
  build_contract
  build_smoke
}

build_all() {
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target sponge_h5_bundle_tests
}

run_label() {
  local label="$1"
  "${CTEST_BIN}" --test-dir "${BUILD_DIR}" -L h5_bundle -L "${label}" \
    --output-on-failure
}

run_runtime_smoke() {
  SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 \
    "${CTEST_BIN}" --test-dir "${BUILD_DIR}" \
    -R 'test_h5_input_output_smoke_matrix|test_h5_reaxff_edip_runtime_parity|test_h5_restart_load_runtime_closure|test_h5_vds_terminal_resume_smoke' \
    --output-on-failure
}

run_staged() {
  configure
  build_contract
  run_label contract
  run_label coverage
  build_smoke
  run_label matrix
  build_mock
  run_label mock-backend
  run_label module
  build_backend_io
  run_label vds
  run_label backend-io
  run_label rerun
  run_label smoke
}

usage() {
  cat <<USAGE
Usage: $0 [command]

Commands:
  configure       Configure build directory with SPONGE_BUILD_TESTS=ON
  build-contract Build contract/completion tests
  build-mock     Build mock-backend writer/module/VDS tests
  build-backend  Build HighFive/HDF5 backend-io tests
  build-matrix   Build static matrix and runtime smoke matrix tests
  build-smoke    Build runtime smoke matrix tests
  build-all      Build all H5 bundle tests
  test-contract  Run CTest label: contract
  test-coverage  Run CTest label: coverage
  test-matrix    Run CTest label: matrix
  test-mock      Run CTest label: mock-backend
  test-mock-backend
                 Run CTest label: mock-backend
  test-module    Run CTest label: module
  test-vds       Run CTest label: vds
  test-rerun     Run CTest label: rerun
  test-backend   Run CTest label: backend-io
  test-backend-io
                 Run CTest label: backend-io
  test-smoke     Run CTest label: smoke
  test-smoke-runtime
                 Run runtime smoke matrix with SPONGE_H5_ENABLE_RUNTIME_SMOKE=1
  test-failure   Run CTest label: failure
  test-all       Run CTest label: h5_bundle
  staged         Configure, build, and run tests in recommended staged order

Environment:
  SPONGE_H5_BUNDLE_TEST_BUILD_DIR  Build directory override
  CMAKE                            CMake executable, default: cmake
  CTEST                            CTest executable, default: ctest

Default build directory:
  pixi env: ./build-\$PIXI_ENVIRONMENT_NAME
  outside pixi: ./build-h5-tests

Examples:
  $0 staged
  pixi run -e dev-cpu $0 staged
USAGE
}

command="${1:-staged}"
case "${command}" in
  configure) configure ;;
  build-contract) build_contract ;;
  build-mock) build_mock ;;
  build-backend) build_backend_io ;;
  build-matrix) build_matrix ;;
  build-smoke) build_smoke ;;
  build-all) build_all ;;
  test-contract) run_label contract ;;
  test-coverage) run_label coverage ;;
  test-matrix) run_label matrix ;;
  test-mock|test-mock-backend) run_label mock-backend ;;
  test-module) run_label module ;;
  test-vds) run_label vds ;;
  test-rerun) run_label rerun ;;
  test-backend|test-backend-io) run_label backend-io ;;
  test-smoke) run_label smoke ;;
  test-smoke-runtime) run_runtime_smoke ;;
  test-failure) run_label failure ;;
  test-all) run_label h5_bundle ;;
  staged) run_staged ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
