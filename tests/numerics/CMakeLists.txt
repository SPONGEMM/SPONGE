find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME test_pressure_virial_regressions
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pressure_virial.py
          --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(test_pressure_virial_regressions PROPERTIES
  LABELS "numerics;pressure;pme;dihedral;runtime" TIMEOUT 180 SKIP_RETURN_CODE 77
  ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
