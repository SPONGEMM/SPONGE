find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME test_pressure_virial_regressions
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pressure_virial.py
    --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_pressure_virial_regressions
  PROPERTIES LABELS
             "numerics;pressure;pme;dihedral;runtime"
             TIMEOUT
             180
             SKIP_RETURN_CODE
             77
             ENVIRONMENT
             "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_test(
  NAME test_constraint_precision
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_constraint_precision.py --sponge
    $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_constraint_precision
  PROPERTIES LABELS
             "numerics;constraints;precision;runtime"
             TIMEOUT
             180
             SKIP_RETURN_CODE
             77
             ENVIRONMENT
             "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_executable(test_nb14_softcore test_nb14_softcore.cpp)
add_test(NAME test_nb14_softcore COMMAND test_nb14_softcore)
set_tests_properties(test_nb14_softcore PROPERTIES LABELS "numerics;nb14;softcore")

add_executable(test_nb14_softcore_h5 test_nb14_softcore_h5.cpp)
target_link_libraries(test_nb14_softcore_h5 PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_nb14_softcore_h5 COMMAND test_nb14_softcore_h5)
set_tests_properties(test_nb14_softcore_h5 PROPERTIES LABELS "numerics;nb14;softcore;h5")

add_test(NAME test_nb14_softcore_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_nb14_softcore_runtime.py --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_nb14_softcore_runtime PROPERTIES LABELS "numerics;nb14;softcore;runtime" TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_executable(test_independent_sterics test_independent_sterics.cpp)
add_test(NAME test_independent_sterics COMMAND test_independent_sterics)
set_tests_properties(test_independent_sterics PROPERTIES LABELS "numerics;lj;softcore")

add_executable(test_independent_sterics_input test_independent_sterics_input.cpp)
target_link_libraries(test_independent_sterics_input PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_independent_sterics_input COMMAND test_independent_sterics_input)
set_tests_properties(test_independent_sterics_input PROPERTIES LABELS "numerics;lj;softcore;h5")

add_test(NAME test_independent_sterics_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_independent_sterics_runtime.py --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_independent_sterics_runtime PROPERTIES LABELS "numerics;lj;softcore;runtime" TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_executable(test_ordinary_restoration test_ordinary_restoration.cpp)
add_test(NAME test_ordinary_restoration COMMAND test_ordinary_restoration)
set_tests_properties(test_ordinary_restoration PROPERTIES LABELS "numerics;fep;exclusions")

add_test(NAME test_ordinary_restoration_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_ordinary_restoration_runtime.py $<TARGET_FILE:SPONGE>)
set_tests_properties(test_ordinary_restoration_runtime PROPERTIES LABELS "numerics;fep;exclusions;runtime" TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_executable(test_ordinary_restoration_h5 test_ordinary_restoration_h5.cpp)
target_link_libraries(test_ordinary_restoration_h5 PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_ordinary_restoration_h5 COMMAND test_ordinary_restoration_h5)
set_tests_properties(test_ordinary_restoration_h5 PROPERTIES LABELS "numerics;fep;exclusions;h5")

add_test(NAME test_ordinary_restoration_h5_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_ordinary_restoration_h5_runtime.py $<TARGET_FILE:SPONGE>)
set_tests_properties(test_ordinary_restoration_h5_runtime PROPERTIES LABELS "numerics;fep;exclusions;h5;runtime" TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
