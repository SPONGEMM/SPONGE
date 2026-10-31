cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(SPONGE LANGUAGES C CXX)

include(GNUInstallDirs)

# Set the root directory
get_filename_component(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR} ABSOLUTE)

# Load Basic Compile Functions and Variables
set(COMMON_FILE "${PROJECT_ROOT_DIR}/cmake/utils/common.cmake")
include("${COMMON_FILE}")

# Parse version from pixi.toml
set(VERSION_FILE "${PROJECT_ROOT_DIR}/cmake/utils/version.cmake")
include("${VERSION_FILE}")

# Set Parallel Backend
set(PARALLEL_FILE "${PROJECT_ROOT_DIR}/cmake/utils/parallel.cmake")
include("${PARALLEL_FILE}")

# Load MPI
set(MPI_FILE "${PROJECT_ROOT_DIR}/cmake/utils/mpi.cmake")
include("${MPI_FILE}")

option(SPONGE_ENABLE_MLIP "Bundle the native local-MP deployment entry" OFF)
if(SPONGE_ENABLE_MLIP)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
     OR NOT PARALLEL_BACKEND STREQUAL "cuda"
     OR NOT MPI)
    message(
      FATAL_ERROR
        "SPONGE_ENABLE_MLIP requires Linux, -DPARALLEL=cuda and -DMPI=ON; advanced plugin builds remain independent"
    )
  endif()
endif()

# Load Targets
set(TARGETS_FILE "${PROJECT_ROOT_DIR}/cmake/utils/targets.cmake")
include("${TARGETS_FILE}")

if(SPONGE_ENABLE_MLIP)
  set(PRIPS_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/SPONGE/prips")
  set(PRIPS_BUNDLED ON)
  set(PRIPS_BUILD_PACKAGE_DIR "${CMAKE_BINARY_DIR}/mlip/prips")
  add_subdirectory(plugins/prips)
  add_dependencies(SPONGE _prips)
  file(RELATIVE_PATH SPONGE_MLIP_INSTALL_RELATIVE
       "${CMAKE_INSTALL_FULL_BINDIR}"
       "${CMAKE_INSTALL_FULL_LIBDIR}/SPONGE/prips/_prips.so")
  target_compile_definitions(
    SPONGE
    PRIVATE SPONGE_ENABLE_MLIP=1
            SPONGE_MLIP_INSTALL_RELATIVE="${SPONGE_MLIP_INSTALL_RELATIVE}"
            SPONGE_MLIP_BUILD_EXECUTABLE="$<TARGET_FILE:SPONGE>"
            SPONGE_MLIP_BUILD_BRIDGE="$<TARGET_FILE:_prips>")
endif()

# Suppress Unnecessary Compile Warnings
set(WARNING_FILE "${PROJECT_ROOT_DIR}/cmake/utils/warning.cmake")
include("${WARNING_FILE}")

option(SPONGE_BUILD_TESTS "Build SPONGE unit tests" OFF)
if(SPONGE_BUILD_TESTS)
  target_compile_definitions(SPONGE PRIVATE SPONGE_H5_TEST_FAULT_INJECTION=1)
  enable_testing()
  add_subdirectory(tests)
endif()

install(
  DIRECTORY "${PROJECT_ROOT_DIR}/schemas/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/SPONGE/schemas"
  FILES_MATCHING
  PATTERN "*.json"
  PATTERN "*.toml")
