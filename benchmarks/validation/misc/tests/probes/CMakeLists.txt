cmake_minimum_required(VERSION 3.24)
project(SpongeConstraintProbes LANGUAGES CXX)

get_filename_component(SPONGE_SOURCE_DIR
                       "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MKL CONFIG QUIET)
# MKL supplies FFTW compatibility headers in include/fftw (Library/include/fftw
# on Windows). Open-source FFTW installs fftw3.h directly in include.
find_path(
  PROBE_FFTW_INCLUDE_DIR fftw3.h
  HINTS "$ENV{CONDA_PREFIX}" "${MKL_ROOT}"
  PATH_SUFFIXES include include/fftw Library/include Library/include/fftw
                REQUIRED)

add_executable(
  constraint_velocity_projection_probe
  ../constraint_velocity_projection_probe.cpp
  "${SPONGE_SOURCE_DIR}/SPONGE/constrain/constrain.cpp"
  "${SPONGE_SOURCE_DIR}/SPONGE/constrain/settle.cpp"
  "${SPONGE_SOURCE_DIR}/SPONGE/constrain/shake.cpp")
# The guard probe includes settle.cpp to exercise its internal guard directly.
add_executable(settle_pair_guard_probe ../settle_pair_guard_probe.cpp)
add_executable(settle_triangle_precision_probe
               ../settle_triangle_precision_probe.cpp)
add_executable(settle_triangle_precision_strict_probe
               ../settle_triangle_precision_probe.cpp)
add_executable(matrix_constraint_precision_probe
               ../matrix_constraint_precision_probe.cpp)
target_sources(matrix_constraint_precision_probe
               PRIVATE "${SPONGE_SOURCE_DIR}/SPONGE/constrain/constrain.cpp")
add_executable(settle_triangle_double_precision_probe
               ../settle_triangle_precision_probe.cpp)
target_compile_definitions(settle_triangle_double_precision_probe
                           PRIVATE SPONGE_CONSTRAINT_DOUBLE_PROBE)
add_executable(pme_virial_probe ../pme_virial_probe.cpp)

# Including the PME implementation also exposes FFT initialization references.
# Resolve them explicitly: MSVC need not discard those references before
# linking.
if(TARGET MKL::MKL)
  target_link_libraries(pme_virial_probe PRIVATE MKL::MKL)
else()
  find_library(
    PROBE_FFTW_LIBRARY
    NAMES fftw3f libfftw3f fftw3f-3 libfftw3f-3
    HINTS "$ENV{CONDA_PREFIX}"
    PATH_SUFFIXES lib Library/lib REQUIRED)
  target_link_libraries(pme_virial_probe PRIVATE "${PROBE_FFTW_LIBRARY}")
endif()

foreach(
  target
  constraint_velocity_projection_probe
  settle_pair_guard_probe
  settle_triangle_precision_probe
  settle_triangle_precision_strict_probe
  pme_virial_probe
  matrix_constraint_precision_probe
  settle_triangle_double_precision_probe)
  target_sources(${target} PRIVATE controller_stubs.cpp
                                   "${SPONGE_SOURCE_DIR}/SPONGE/common.cpp")
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

# Reassociation can hide absolute-coordinate cancellation in an inlined CPU
# probe. Also exercise ordinary IEEE arithmetic rather than relying on it.
if(MSVC)
  target_compile_options(settle_triangle_precision_strict_probe
                         PRIVATE /fp:precise)
else()
  target_compile_options(settle_triangle_precision_strict_probe
                         PRIVATE -fno-fast-math -ffp-contract=off)
endif()
