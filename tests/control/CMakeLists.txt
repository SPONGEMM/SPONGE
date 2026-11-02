add_executable(test_toml_command_aliases test_toml_command_aliases.cpp)
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
add_test(
  NAME test_mdin_schema_examples
  COMMAND
    ${Python3_EXECUTABLE}
    ${PROJECT_ROOT_DIR}/scripts/validate_mdin_schema_examples.py --schema
    ${PROJECT_ROOT_DIR}/schemas/mdin.schema.json)
set_tests_properties(test_mdin_schema_examples PROPERTIES LABELS
                                                          "control;schema;mdin")
