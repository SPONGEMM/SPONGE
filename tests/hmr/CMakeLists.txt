add_executable(test_hmr test_hmr.cpp)
target_include_directories(test_hmr PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
add_test(NAME test_hmr COMMAND test_hmr)
set_tests_properties(test_hmr PROPERTIES LABELS "hmr;unit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME test_hmr_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_hmr_runtime.py
          --sponge $<TARGET_FILE:SPONGE>)
set_tests_properties(
  test_hmr_runtime
  PROPERTIES LABELS
             "hmr;runtime"
             TIMEOUT
             180
             SKIP_RETURN_CODE
             77
             ENVIRONMENT
             "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
