message(STATUS "Set compilation warnings from ${WARNING_FILE}")
if(PARALLEL_BACKEND STREQUAL "cuda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=177")
  message(
    STATUS "-- Disable warning 177-D for Launch kernel unreferenced warning")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
  message(STATUS "-- Disable deprecated GPU target warnings from nvcc")
  add_compile_options(/wd4819)
  message(STATUS "-- Disable warning d4819 for GBK encoding")
endif()
message(
  STATUS "------------------------------------------------------------------")
