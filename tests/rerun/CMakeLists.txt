find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)

foreach(case cell_history pme_boundary)
  add_test(
    NAME test_rerun_${case}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_regressions.py
      --sponge $<TARGET_FILE:SPONGE> --case ${case} --output
      ${CMAKE_CURRENT_BINARY_DIR}/${case})
  set_tests_properties(test_rerun_${case} PROPERTIES LABELS "rerun;regression"
                                                     TIMEOUT 300)
endforeach()

add_executable(test_pme_grid_coordinate test_pme_grid_coordinate.cpp)
target_include_directories(test_pme_grid_coordinate
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_pme_grid_coordinate.cpp PROPERTIES LANGUAGE
                                                                      CUDA)
endif()
add_test(NAME test_pme_grid_coordinate COMMAND test_pme_grid_coordinate)
set_tests_properties(test_pme_grid_coordinate PROPERTIES LABELS
                                                         "pme;regression")
