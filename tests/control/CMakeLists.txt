add_executable(test_toml_command_aliases test_toml_command_aliases.cpp)
add_executable(test_coupling_config test_coupling_config.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/external_force/drive_config.cpp)
target_include_directories(test_coupling_config PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
target_link_libraries(test_coupling_config PRIVATE sponge_toml)
add_test(NAME test_coupling_config COMMAND test_coupling_config)
set_tests_properties(test_coupling_config PROPERTIES LABELS "control;nonequilibrium;contract")
add_executable(test_group_load test_group_load.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/external_force/group_load.cpp)
target_include_directories(test_group_load PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_group_load COMMAND test_group_load)
set_tests_properties(test_group_load PROPERTIES LABELS "nonequilibrium;numerics")
add_executable(test_work_ledger test_work_ledger.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/MD_core/energy_ledger.cpp)
target_include_directories(test_work_ledger PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_work_ledger COMMAND test_work_ledger)
set_tests_properties(test_work_ledger PROPERTIES LABELS "nonequilibrium;restart;numerics")
target_include_directories(test_toml_command_aliases
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
target_link_libraries(test_toml_command_aliases PRIVATE sponge_toml)
add_test(NAME test_toml_command_aliases COMMAND test_toml_command_aliases)
set_tests_properties(test_toml_command_aliases
                     PROPERTIES LABELS "control;toml;contract")

add_executable(test_h5_input_plan test_h5_input_plan.cpp)
target_include_directories(test_h5_input_plan
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_h5_input_plan COMMAND test_h5_input_plan)
set_tests_properties(test_h5_input_plan PROPERTIES LABELS
                                                   "control;h5_input;contract")

add_executable(test_h5_input_metadata test_h5_input_metadata.cpp)
target_include_directories(test_h5_input_metadata
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_h5_input_metadata COMMAND test_h5_input_metadata)
set_tests_properties(test_h5_input_metadata
                     PROPERTIES LABELS "control;h5_input;contract")

add_executable(test_h5_input_assembler test_h5_input_assembler.cpp)
target_include_directories(test_h5_input_assembler
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  if(MSVC)
    target_compile_options(test_h5_input_assembler PRIVATE /UUSE_CUDA /UUSE_GPU)
  else()
    target_compile_options(test_h5_input_assembler PRIVATE -UUSE_CUDA -UUSE_GPU)
  endif()
endif()
add_test(NAME test_h5_input_assembler COMMAND test_h5_input_assembler)
set_tests_properties(test_h5_input_assembler
                     PROPERTIES LABELS "control;h5_input;assembly")

add_executable(test_h5_dynamic_state_modules test_h5_dynamic_state_modules.cpp)
target_include_directories(test_h5_dynamic_state_modules
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_h5_dynamic_state_modules.cpp
                              PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME test_h5_dynamic_state_modules
         COMMAND test_h5_dynamic_state_modules)
set_tests_properties(test_h5_dynamic_state_modules
                     PROPERTIES LABELS "control;h5_input;restart")

add_executable(test_h5_rerun_frame_selection test_h5_rerun_frame_selection.cpp)
target_include_directories(test_h5_rerun_frame_selection
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_h5_rerun_frame_selection
         COMMAND test_h5_rerun_frame_selection)
set_tests_properties(test_h5_rerun_frame_selection
                     PROPERTIES LABELS "control;h5_input;rerun")

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
add_test(NAME test_coupling_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_coupling_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_coupling_runtime PROPERTIES
  LABELS "nonequilibrium;integration" TIMEOUT 180)
add_test(NAME test_coupling_restart COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_coupling_restart.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_coupling_restart PROPERTIES
  LABELS "nonequilibrium;integration;restart" TIMEOUT 180 SKIP_RETURN_CODE 77)
add_test(
  NAME test_mdin_schema_examples
  COMMAND
    ${Python3_EXECUTABLE}
    ${PROJECT_ROOT_DIR}/scripts/validate_mdin_schema_examples.py --schema
    ${PROJECT_ROOT_DIR}/schemas/mdin.schema.json)
set_tests_properties(test_mdin_schema_examples PROPERTIES LABELS
                                                          "control;schema;mdin")

add_executable(test_rerun_output_guard test_rerun_output_guard.cpp)
target_include_directories(test_rerun_output_guard
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_rerun_output_guard COMMAND test_rerun_output_guard)
set_tests_properties(test_rerun_output_guard PROPERTIES LABELS "control;rerun")
add_executable(test_flow_math test_flow_math.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/thermostat/velocity_bias/mode_math.cpp)
target_include_directories(test_flow_math PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_flow_math COMMAND test_flow_math)
set_tests_properties(test_flow_math PROPERTIES LABELS "nonequilibrium;numerics")
add_test(NAME test_cosine_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_cosine_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_cosine_runtime PROPERTIES LABELS "nonequilibrium;integration" TIMEOUT 180)

add_test(NAME test_green_kubo_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_green_kubo_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_green_kubo_runtime PROPERTIES
  LABELS "nonequilibrium;integration" TIMEOUT 180 SKIP_RETURN_CODE 77)
add_test(NAME test_dihedral_stress_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_dihedral_stress_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_dihedral_stress_runtime PROPERTIES
  LABELS "nonequilibrium;integration;numerics" TIMEOUT 60 SKIP_RETURN_CODE 77)
add_test(NAME test_green_kubo_analysis COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_green_kubo_analysis.py)
set_tests_properties(test_green_kubo_analysis PROPERTIES
  LABELS "nonequilibrium;numerics" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_executable(test_rnemd_math test_rnemd_math.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/momentum_exchange/exchange_math.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/MD_core/energy_ledger.cpp)
target_include_directories(test_rnemd_math PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_rnemd_math COMMAND test_rnemd_math)
set_tests_properties(test_rnemd_math PROPERTIES LABELS "nonequilibrium;numerics;restart")
add_test(NAME test_rnemd_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_rnemd_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_rnemd_runtime PROPERTIES
  LABELS "nonequilibrium;integration" TIMEOUT 180 SKIP_RETURN_CODE 77)
add_test(NAME test_rnemd_analysis COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_rnemd_analysis.py)
set_tests_properties(test_rnemd_analysis PROPERTIES
  LABELS "nonequilibrium;numerics" TIMEOUT 60 SKIP_RETURN_CODE 77)
add_test(NAME test_lj_benchmark COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_lj_benchmark.py)
set_tests_properties(test_lj_benchmark PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_cosine_analysis COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_cosine_analysis.py)
set_tests_properties(test_cosine_analysis PROPERTIES
  LABELS "nonequilibrium;numerics" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_lj_gk_compare COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_lj_gk_compare.py)
set_tests_properties(test_lj_gk_compare PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_lj_rnemd_compare COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_lj_rnemd_compare.py)
set_tests_properties(test_lj_rnemd_compare PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_lj_cosine_compare COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_lj_cosine_compare.py)
set_tests_properties(test_lj_cosine_compare PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_experimental_reference COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_experimental_reference.py)
set_tests_properties(test_experimental_reference PROPERTIES
  LABELS "nonequilibrium;numerics;reference" TIMEOUT 30)

add_test(NAME test_hexane_cosine_analysis COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_hexane_cosine_analysis.py)
set_tests_properties(test_hexane_cosine_analysis PROPERTIES
  LABELS "nonequilibrium;numerics;reference" TIMEOUT 30)

add_test(NAME test_lj_native_cosine COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_lj_native_cosine.py)
set_tests_properties(test_lj_native_cosine PROPERTIES
  LABELS "nonequilibrium;numerics;reference" TIMEOUT 30)

add_executable(test_constraint_stress test_constraint_stress.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/constrain/tangent_frame.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/observables/stress_math.cpp)
target_include_directories(test_constraint_stress PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_constraint_stress COMMAND test_constraint_stress)
set_tests_properties(test_constraint_stress PROPERTIES
  LABELS "nonequilibrium;numerics;constraints" TIMEOUT 30)
add_executable(test_constrained_flow test_constrained_flow.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/MD_core/integrator/constrained_middle_step.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/constrain/tangent_frame.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/observables/stress_math.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/thermostat/velocity_bias/mode_math.cpp)
target_include_directories(test_constrained_flow PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_constrained_flow COMMAND test_constrained_flow)
set_tests_properties(test_constrained_flow PROPERTIES
  LABELS "nonequilibrium;numerics;constraints" TIMEOUT 30)
add_test(NAME test_constrained_flow_covariance COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_constrained_flow_covariance.py
  --probe $<TARGET_FILE:test_constrained_flow>)
set_tests_properties(test_constrained_flow_covariance PROPERTIES
  LABELS "nonequilibrium;numerics;constraints" TIMEOUT 60 SKIP_RETURN_CODE 77)
add_executable(test_constrained_flow_step test_constrained_flow_step.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/MD_core/integrator/constrained_middle_step.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/constrain/tangent_frame.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/observables/stress_math.cpp)
target_include_directories(test_constrained_flow_step PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_constrained_flow_step COMMAND test_constrained_flow_step)
set_tests_properties(test_constrained_flow_step PROPERTIES
  LABELS "nonequilibrium;numerics;constraints" TIMEOUT 30)
add_executable(test_constrained_nve test_constrained_nve.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/MD_core/integrator/rattle_step.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/constrain/tangent_frame.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/observables/stress_math.cpp)
target_include_directories(test_constrained_nve PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_constrained_nve COMMAND test_constrained_nve)
set_tests_properties(test_constrained_nve PROPERTIES
  LABELS "nonequilibrium;numerics;constraints" TIMEOUT 30)
add_test(NAME test_constraint_observer_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_constraint_observer_runtime.py
  --sponge $<TARGET_FILE:SPONGE> --observer $<TARGET_FILE:test_constraint_stress>)
add_test(NAME test_constrained_cosine_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_constrained_cosine_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_constrained_cosine_runtime PROPERTIES
  LABELS "nonequilibrium;numerics;constraints;restart" TIMEOUT 90 SKIP_RETURN_CODE 77)
add_test(NAME test_constrained_nve_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_constrained_nve_runtime.py
  --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_constrained_nve_runtime PROPERTIES
  LABELS "nonequilibrium;numerics;constraints;restart" TIMEOUT 120 SKIP_RETURN_CODE 77)
set_tests_properties(test_constraint_observer_runtime PROPERTIES
  LABELS "nonequilibrium;integration;constraints" TIMEOUT 180 SKIP_RETURN_CODE 77)

add_test(NAME test_water_model COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_water_model.py)
set_tests_properties(test_water_model PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_hexane_pilot COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_hexane_pilot.py)
set_tests_properties(test_hexane_pilot PROPERTIES
  LABELS "nonequilibrium;numerics;benchmark" TIMEOUT 60 SKIP_RETURN_CODE 77)

add_test(NAME test_water_pilot COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_water_pilot.py --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_water_pilot PROPERTIES
  LABELS "nonequilibrium;integration;benchmark" TIMEOUT 180 SKIP_RETURN_CODE 77)

add_test(NAME test_thermal_components_runtime COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_thermal_components_runtime.py --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_thermal_components_runtime PROPERTIES LABELS "dynamics;integration;thermal" TIMEOUT 120 SKIP_RETURN_CODE 77)

add_test(NAME test_generic_analysis COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_ROOT_DIR}/tests/nonequilibrium/test_generic_analysis.py)
set_tests_properties(test_generic_analysis PROPERTIES LABELS "dynamics;analysis;contract" TIMEOUT 60 SKIP_RETURN_CODE 77)
