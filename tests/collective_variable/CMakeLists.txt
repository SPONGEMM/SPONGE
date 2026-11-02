find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)

foreach(case centers rmsd_zero rmsd_rotation tabulated)
  add_test(
    NAME test_cv_${case}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_regressions.py
      --sponge $<TARGET_FILE:SPONGE> --case ${case} --output
      ${CMAKE_CURRENT_BINARY_DIR}/${case})
  set_tests_properties(test_cv_${case} PROPERTIES LABELS "cv;regression"
                                                  TIMEOUT 300)
endforeach()
