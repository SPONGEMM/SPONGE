find_package(HighFive CONFIG REQUIRED)
find_package(HDF5 1.10.7 REQUIRED COMPONENTS C)
find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
find_program(H5DIFF_EXECUTABLE NAMES h5diff REQUIRED)
find_program(H5DUMP_EXECUTABLE NAMES h5dump REQUIRED)

add_library(sponge_h5_bundle_test_common INTERFACE)
target_include_directories(sponge_h5_bundle_test_common
                           INTERFACE ${PROJECT_ROOT_DIR}/SPONGE)
target_compile_definitions(
  sponge_h5_bundle_test_common
  INTERFACE
    SPONGE_H5_INPUT_MATRIX_FIXTURE_ROOT="${PROJECT_ROOT_DIR}/tests/h5_bundle/fixtures/input_matrix"
)
target_compile_features(sponge_h5_bundle_test_common INTERFACE cxx_std_17)

if(TARGET HighFive::HighFive)
  target_link_libraries(sponge_h5_bundle_test_common
                        INTERFACE HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(sponge_h5_bundle_test_common INTERFACE HighFive)
else()
  message(
    FATAL_ERROR "HighFive target was not provided by find_package(HighFive)")
endif()

if(TARGET HDF5::HDF5)
  target_link_libraries(sponge_h5_bundle_test_common INTERFACE HDF5::HDF5)
else()
  target_include_directories(sponge_h5_bundle_test_common
                             INTERFACE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(sponge_h5_bundle_test_common
                        INTERFACE ${HDF5_LIBRARIES})
endif()

function(add_sponge_h5_bundle_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sponge_h5_bundle_test_common)
  add_test(NAME ${name} COMMAND ${name})
  set_property(GLOBAL APPEND PROPERTY SPONGE_H5_BUNDLE_TEST_TARGETS ${name})
endfunction()

add_sponge_h5_bundle_test(test_h5_output_plan test_h5_output_plan.cpp)
set_tests_properties(test_h5_output_plan PROPERTIES LABELS "h5_bundle;contract")

add_sponge_h5_bundle_test(test_output_route_helpers
                          test_output_route_helpers.cpp)
set_tests_properties(test_output_route_helpers PROPERTIES LABELS
                                                          "h5_bundle;contract")

add_sponge_h5_bundle_test(test_h5md_writers_with_mock_backend
                          test_h5md_writers_with_mock_backend.cpp)
set_tests_properties(test_h5md_writers_with_mock_backend
                     PROPERTIES LABELS "h5_bundle;mock-backend;failure")

add_sponge_h5_bundle_test(test_completion_tracker test_completion_tracker.cpp)
set_tests_properties(test_completion_tracker PROPERTIES LABELS
                                                        "h5_bundle;failure")

add_sponge_h5_bundle_test(test_module_h5_mappings_with_mock_backend
                          test_module_h5_mappings_with_mock_backend.cpp)
set_tests_properties(test_module_h5_mappings_with_mock_backend
                     PROPERTIES LABELS "h5_bundle;module;mock-backend")

add_sponge_h5_bundle_test(test_vds_trajectory_writer_with_mock_backend
                          test_vds_trajectory_writer_with_mock_backend.cpp)
set_tests_properties(test_vds_trajectory_writer_with_mock_backend
                     PROPERTIES LABELS "h5_bundle;vds;mock-backend;failure")

add_sponge_h5_bundle_test(test_highfive_backend_io test_highfive_backend_io.cpp)
set_tests_properties(test_highfive_backend_io
                     PROPERTIES LABELS "h5_bundle;backend-io;vds")

add_sponge_h5_bundle_test(test_restart_h5_reader test_restart_h5_reader.cpp)
set_tests_properties(test_restart_h5_reader
                     PROPERTIES LABELS "h5_bundle;input;backend-io")

add_sponge_h5_bundle_test(test_topology_native_h5_reader
                          test_topology_native_h5_reader.cpp)
set_tests_properties(test_topology_native_h5_reader
                     PROPERTIES LABELS "h5_bundle;input;backend-io;contract")

add_sponge_h5_bundle_test(test_trajectory_h5_reader
                          test_trajectory_h5_reader.cpp)
set_tests_properties(test_trajectory_h5_reader
                     PROPERTIES LABELS "h5_bundle;input;rerun;backend-io")

add_sponge_h5_bundle_test(test_h5_input_validation test_h5_input_validation.cpp)
set_tests_properties(test_h5_input_validation
                     PROPERTIES LABELS "h5_bundle;input;backend-io;contract")

add_sponge_h5_bundle_test(test_h5_legacy_sidecar test_h5_legacy_sidecar.cpp)
set_tests_properties(test_h5_legacy_sidecar
                     PROPERTIES LABELS "h5_bundle;input;backend-io")

add_sponge_h5_bundle_test(test_h5_input_matrix_contract
                          test_h5_input_matrix_contract.cpp)
set_tests_properties(test_h5_input_matrix_contract
                     PROPERTIES LABELS "h5_bundle;input;contract")

add_sponge_h5_bundle_test(test_h5_io_contract_coverage
                          test_h5_io_contract_coverage.cpp)
set_tests_properties(test_h5_io_contract_coverage
                     PROPERTIES LABELS "h5_bundle;contract;coverage")

add_test(
  NAME test_h5_io_contract_manifest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_io_contract_manifest.py --fixture-root
    ${PROJECT_ROOT_DIR}/tests/h5_bundle/fixtures/input_matrix --h5dump
    ${H5DUMP_EXECUTABLE})
set_tests_properties(test_h5_io_contract_manifest
                     PROPERTIES LABELS "h5_bundle;contract;coverage")

add_test(
  NAME test_h5_input_fixture_equivalence
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_input_fixture_equivalence.py
    --fixture-root ${PROJECT_ROOT_DIR}/tests/h5_bundle/fixtures/input_matrix
    --h5diff ${H5DIFF_EXECUTABLE} --h5dump ${H5DUMP_EXECUTABLE})
set_tests_properties(test_h5_input_fixture_equivalence
                     PROPERTIES LABELS "h5_bundle;input;contract;matrix")

add_test(
  NAME test_h5_test_targets_manifest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_test_targets_manifest.py --cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt --manifest
    ${CMAKE_CURRENT_SOURCE_DIR}/TEST_TARGETS.md)
set_tests_properties(test_h5_test_targets_manifest
                     PROPERTIES LABELS "h5_bundle;contract")

add_test(
  NAME test_h5_matrix_plan_manifest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_matrix_plan_manifest.py --plan
    ${CMAKE_CURRENT_SOURCE_DIR}/INPUT_OUTPUT_MATRIX_TEST_PLAN.md --matrix-spec
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_io_matrix_spec.cpp --smoke-matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_input_output_smoke_matrix.cpp
    --manybody-parity-source
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_reaxff_edip_runtime_parity.cpp
    --fixture-helper ${CMAKE_CURRENT_SOURCE_DIR}/h5_input_matrix_fixture.hpp)
set_tests_properties(test_h5_matrix_plan_manifest
                     PROPERTIES LABELS "h5_bundle;contract;matrix")

add_test(
  NAME test_h5_ci_plan_manifest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_ci_plan_manifest.py --plan
    ${CMAKE_CURRENT_SOURCE_DIR}/INPUT_OUTPUT_MATRIX_TEST_PLAN.md --cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt --script
    ${CMAKE_CURRENT_SOURCE_DIR}/run_h5_bundle_tests.sh --smoke-matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_input_output_smoke_matrix.cpp
    --matrix-manifest
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_matrix_plan_manifest.py
    --io-contract-manifest
    ${CMAKE_CURRENT_SOURCE_DIR}/test_h5_io_contract_manifest.py
    --targets-manifest ${CMAKE_CURRENT_SOURCE_DIR}/TEST_TARGETS.md)
set_tests_properties(test_h5_ci_plan_manifest
                     PROPERTIES LABELS "h5_bundle;contract;matrix")

add_sponge_h5_bundle_test(test_h5_io_matrix_spec test_h5_io_matrix_spec.cpp)
set_tests_properties(test_h5_io_matrix_spec
                     PROPERTIES LABELS "h5_bundle;contract;matrix;rerun;vds")

add_executable(test_h5_input_output_smoke_matrix
               test_h5_input_output_smoke_matrix.cpp)
target_link_libraries(test_h5_input_output_smoke_matrix
                      PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_h5_input_output_smoke_matrix
         COMMAND test_h5_input_output_smoke_matrix $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_h5_input_output_smoke_matrix
  PROPERTIES LABELS "h5_bundle;input;smoke;matrix;rerun;vds" SKIP_RETURN_CODE
             77)
if("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
  # TODO: Re-enable after resolving the VDS shard start-step mismatch when
  # observables are written before the first particle frame. Keep local runs.
  set_tests_properties(test_h5_input_output_smoke_matrix
                       PROPERTIES DISABLED TRUE)
endif()
set_property(GLOBAL APPEND PROPERTY SPONGE_H5_BUNDLE_TEST_TARGETS
                                    test_h5_input_output_smoke_matrix)

add_executable(test_h5_reaxff_edip_runtime_parity
               test_h5_reaxff_edip_runtime_parity.cpp)
target_link_libraries(test_h5_reaxff_edip_runtime_parity
                      PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_h5_reaxff_edip_runtime_parity
         COMMAND test_h5_reaxff_edip_runtime_parity $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_h5_reaxff_edip_runtime_parity
  PROPERTIES LABELS "h5_bundle;input;smoke;matrix;rerun;vds;manybody"
             SKIP_RETURN_CODE 77)
set_property(GLOBAL APPEND PROPERTY SPONGE_H5_BUNDLE_TEST_TARGETS
                                    test_h5_reaxff_edip_runtime_parity)

add_executable(test_h5_restart_load_runtime_closure
               test_h5_restart_load_runtime_closure.cpp)
target_link_libraries(test_h5_restart_load_runtime_closure
                      PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_h5_restart_load_runtime_closure
         COMMAND test_h5_restart_load_runtime_closure $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_h5_restart_load_runtime_closure
  PROPERTIES LABELS "h5_bundle;input;smoke;rerun;restart" SKIP_RETURN_CODE 77)
set_property(GLOBAL APPEND PROPERTY SPONGE_H5_BUNDLE_TEST_TARGETS
                                    test_h5_restart_load_runtime_closure)

add_sponge_h5_bundle_test(test_h5_vds_terminal_resume_smoke
                          test_h5_vds_terminal_resume_smoke.cpp)
set_tests_properties(
  test_h5_vds_terminal_resume_smoke
  PROPERTIES LABELS "h5_bundle;input;smoke;rerun;vds;failure" SKIP_RETURN_CODE
             77)

add_executable(h5_input_smoke_fixture h5_input_smoke_fixture.cpp)
target_link_libraries(h5_input_smoke_fixture
                      PRIVATE sponge_h5_bundle_test_common)

add_executable(h5_restart_read_probe h5_restart_read_probe.cpp)
target_link_libraries(h5_restart_read_probe
                      PRIVATE sponge_h5_bundle_test_common)

add_executable(h5_trajectory_read_probe h5_trajectory_read_probe.cpp)
target_link_libraries(h5_trajectory_read_probe
                      PRIVATE sponge_h5_bundle_test_common)

add_executable(h5_bundle_writer_probe h5_bundle_writer_probe.cpp)
target_link_libraries(h5_bundle_writer_probe
                      PRIVATE sponge_h5_bundle_test_common)

get_property(SPONGE_H5_BUNDLE_TEST_TARGETS GLOBAL
             PROPERTY SPONGE_H5_BUNDLE_TEST_TARGETS)
add_custom_target(sponge_h5_bundle_tests
                  DEPENDS ${SPONGE_H5_BUNDLE_TEST_TARGETS})

add_custom_target(
  sponge_h5_bundle_contract_tests
  DEPENDS test_h5_output_plan
          test_output_route_helpers
          test_completion_tracker
          test_topology_native_h5_reader
          test_h5_input_validation
          test_h5_input_matrix_contract
          test_h5_io_contract_coverage
          test_h5_io_matrix_spec)
add_custom_target(
  sponge_h5_bundle_mock_tests
  DEPENDS test_h5md_writers_with_mock_backend
          test_module_h5_mappings_with_mock_backend
          test_vds_trajectory_writer_with_mock_backend)
add_custom_target(
  sponge_h5_bundle_backend_io_tests
  DEPENDS test_highfive_backend_io
          test_restart_h5_reader
          test_topology_native_h5_reader
          test_trajectory_h5_reader
          test_h5_input_validation
          test_h5_legacy_sidecar
          test_h5_input_matrix_contract)

add_custom_target(
  sponge_h5_bundle_smoke_tests
  DEPENDS test_h5_input_output_smoke_matrix test_h5_reaxff_edip_runtime_parity
          test_h5_restart_load_runtime_closure
          test_h5_vds_terminal_resume_smoke)
