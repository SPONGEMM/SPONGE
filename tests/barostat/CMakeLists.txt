add_executable(test_mc_barostat_math test_mc_barostat_math.cpp)
target_include_directories(test_mc_barostat_math
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_mc_barostat_math.cpp PROPERTIES LANGUAGE
                                                                   CUDA)
endif()
add_test(NAME test_mc_barostat_math COMMAND test_mc_barostat_math)
set_tests_properties(test_mc_barostat_math PROPERTIES LABELS
                                                      "barostat;contract")
