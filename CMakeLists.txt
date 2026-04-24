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

# Load Targets
set(TARGETS_FILE "${PROJECT_ROOT_DIR}/cmake/utils/targets.cmake")
include("${TARGETS_FILE}")

# Suppress Unnecessary Compile Warnings
set(WARNING_FILE "${PROJECT_ROOT_DIR}/cmake/utils/warning.cmake")
include("${WARNING_FILE}")
