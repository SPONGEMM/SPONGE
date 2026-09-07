cmake_minimum_required(VERSION 3.24)
project(SpongeConstraintProbes LANGUAGES CXX)

get_filename_component(SPONGE_SOURCE_DIR
                       "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
find_package(OpenMP REQUIRED COMPONENTS CXX)
# MKL supplies FFTW compatibility headers in include/fftw (Library/include/fftw
# on Windows). Open-source FFTW installs fftw3.h directly in include.
find_path(
  PROBE_FFTW_INCLUDE_DIR fftw3.h
  HINTS "$ENV{CONDA_PREFIX}"
  PATH_SUFFIXES include include/fftw Library/include Library/include/fftw
                REQUIRED)

add_executable(
  constraint_velocity_projection_probe
  ../constraint_velocity_projection_probe.cpp
  "${SPONGE_SOURCE_DIR}/SPONGE/constrain/settle.cpp"
  "${SPONGE_SOURCE_DIR}/SPONGE/constrain/shake.cpp")
# The guard probe includes settle.cpp to exercise its internal guard directly.
add_executable(settle_pair_guard_probe ../settle_pair_guard_probe.cpp)

foreach(target constraint_velocity_projection_probe settle_pair_guard_probe)
  target_sources(${target} PRIVATE "${SPONGE_SOURCE_DIR}/SPONGE/common.cpp")
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE USE_CPU NO_GLOBAL_CONTROLLER)
  target_include_directories(${target} PRIVATE "${SPONGE_SOURCE_DIR}/SPONGE"
                                               "${PROBE_FFTW_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
               RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
  # The probes link only the exercised functions, not the full simulation.
  if(MSVC)
    target_compile_options(${target} PRIVATE /O2 /fp:fast /Gy /Gw /Zc:inline)
    target_link_options(${target} PRIVATE /OPT:REF)
  else()
    target_compile_options(
      ${target} PRIVATE -O3 -march=native -ffast-math -ffunction-sections
                        -fdata-sections)
    target_link_options(${target} PRIVATE -ffast-math)
    if(APPLE)
      target_link_options(${target} PRIVATE -Wl,-dead_strip)
    else()
      target_link_options(${target} PRIVATE -Wl,--gc-sections)
    endif()
  endif()
endforeach()
