# Also runnable without SPONGE's GPU, math or HDF5 dependencies.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(SpongeSitsTests LANGUAGES CXX)
  enable_testing()
endif()

add_executable(test_sits_accumulator test_sits_accumulator.cpp)
target_compile_features(test_sits_accumulator PRIVATE cxx_std_11)
add_test(NAME test_sits_accumulator COMMAND test_sits_accumulator)
set_tests_properties(test_sits_accumulator PROPERTIES LABELS "sits;numerics")
