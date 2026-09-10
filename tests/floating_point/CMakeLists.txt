# Standalone configuration also allows testing without SPONGE's math/HDF5 deps.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(SpongeFloatTests LANGUAGES CXX)
  enable_testing()
endif()

set(float_cases
    # Positive/negative zero, ordinary values, minimum/maximum subnormals,
    # minimum normals, maximum finite values, infinities, quiet/signaling NaNs.
    00000000
    0000000000000000
    1
    80000000
    8000000000000000
    1
    3fc00000
    3ff8000000000000
    1
    bfc00000
    bff8000000000000
    1
    00000001
    0000000000000001
    1
    80000001
    8000000000000001
    1
    007fffff
    000fffffffffffff
    1
    807fffff
    800fffffffffffff
    1
    00800000
    0010000000000000
    1
    80800000
    8010000000000000
    1
    7f7fffff
    7fefffffffffffff
    1
    ff7fffff
    ffefffffffffffff
    1
    7f800000
    7ff0000000000000
    4
    ff800000
    fff0000000000000
    4
    7fc00000
    7ff8000000000000
    2
    ffc00000
    fff8000000000000
    2
    7f800001
    7ff0000000000001
    2
    ff800001
    fff0000000000001
    2)

include(CheckIPOSupported)
check_ipo_supported(
  RESULT float_ipo_supported
  OUTPUT float_ipo_error
  LANGUAGES CXX)
foreach(mode strict fast fast_lto)
  if(mode STREQUAL "fast_lto" AND NOT float_ipo_supported)
    message(STATUS "Skipping float LTO test: ${float_ipo_error}")
    continue()
  endif()
  foreach(test_name float_classification h5_input_assembler)
    set(target test_${test_name}_${mode})
    if(test_name STREQUAL "float_classification")
      add_executable(${target} test_float_classification.cpp)
    else()
      add_executable(${target} ../control/test_h5_input_assembler.cpp)
      target_include_directories(${target} PRIVATE ../../SPONGE)
      if(MSVC)
        target_compile_options(${target} PRIVATE /UUSE_CUDA /UUSE_GPU)
      else()
        target_compile_options(${target} PRIVATE -UUSE_CUDA -UUSE_GPU)
      endif()
    endif()
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(MSVC)
      target_compile_options(${target} PRIVATE /O2)
      if(mode STREQUAL "strict")
        target_compile_options(${target} PRIVATE /fp:strict)
      else()
        target_compile_options(${target} PRIVATE /fp:fast)
      endif()
    else()
      target_compile_options(${target} PRIVATE -O3)
      if(mode STREQUAL "strict")
        target_compile_options(${target} PRIVATE -fno-fast-math)
        target_link_options(${target} PRIVATE -fno-fast-math)
      else()
        target_compile_options(${target} PRIVATE -ffast-math)
        target_link_options(${target} PRIVATE -ffast-math)
      endif()
    endif()
    if(mode STREQUAL "fast_lto")
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    if(test_name STREQUAL "float_classification")
      add_test(NAME ${target} COMMAND ${target} ${float_cases})
    else()
      add_test(NAME ${target} COMMAND ${target})
    endif()
    set_tests_properties(${target} PROPERTIES LABELS "float;contract")
  endforeach()
endforeach()

# Test host and device passes with the backend's actual fast-math flags.
if(PARALLEL_BACKEND STREQUAL "cuda" OR PARALLEL_BACKEND STREQUAL "hip")
  if(PARALLEL_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
    set(gpu_language CUDA)
    set(gpu_flags --use_fast_math)
  else()
    enable_language(HIP)
    set(gpu_language HIP)
    set(gpu_flags -ffast-math)
  endif()
  configure_file(test_float_classification.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/test_float_device.cpp COPYONLY)
  # Resolve the copied source's relative header include through the source tree.
  add_executable(test_float_classification_device
                 ${CMAKE_CURRENT_BINARY_DIR}/test_float_device.cpp)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/test_float_device.cpp
                              PROPERTIES LANGUAGE ${gpu_language})
  target_include_directories(test_float_classification_device
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(test_float_classification_device
                        PROPERTIES CUDA_STANDARD 17 HIP_STANDARD 17)
  target_compile_options(test_float_classification_device PRIVATE -O3
                                                                  ${gpu_flags})
  add_test(NAME test_float_classification_device
           COMMAND test_float_classification_device ${float_cases})
  set_tests_properties(test_float_classification_device
                       PROPERTIES LABELS "float;gpu" SKIP_RETURN_CODE 77)
endif()
