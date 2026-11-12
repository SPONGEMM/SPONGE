find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME test_nopbc_atom_order
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_atom_order.py
          --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_nopbc_atom_order
  PROPERTIES LABELS
             "nopbc;gb;regression;runtime"
             TIMEOUT
             180
             SKIP_RETURN_CODE
             77
             ENVIRONMENT
             "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
