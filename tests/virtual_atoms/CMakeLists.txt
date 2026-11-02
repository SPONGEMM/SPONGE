add_executable(test_virtual_atom_core test_virtual_atom_core.cpp)
target_include_directories(test_virtual_atom_core
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_virtual_atom_core.cpp PROPERTIES LANGUAGE
                                                                    CUDA)
else()
  target_sources(test_virtual_atom_core
                 PRIVATE ${PROJECT_ROOT_DIR}/SPONGE/common.cpp)
endif()
add_test(NAME test_virtual_atom_core COMMAND test_virtual_atom_core)
set_tests_properties(test_virtual_atom_core PROPERTIES LABELS
                                                       "virtual_atoms;contract")
