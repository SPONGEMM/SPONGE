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

add_test(
  NAME test_meta_subhill_snapshot
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_meta_snapshot.py
    --sponge $<TARGET_FILE:SPONGE> --output
    ${CMAKE_CURRENT_BINARY_DIR}/meta_subhill_snapshot)
set_tests_properties(test_meta_subhill_snapshot
                     PROPERTIES LABELS "cv;regression;h5_bundle" TIMEOUT 300)
