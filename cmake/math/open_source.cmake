find_package(OpenBLAS REQUIRED)

include(FindPackageHandleStandardArgs)

find_path(
  LAPACKE_INCLUDE_DIR
  NAMES "lapacke.h"
  HINTS "$ENV{CONDA_PREFIX}"
  PATH_SUFFIXES "include")
find_library(
  LAPACKE_LIBRARY
  NAMES "lapacke" "liblapacke"
  HINTS "$ENV{CONDA_PREFIX}"
  PATH_SUFFIXES "lib")
find_package_handle_standard_args(
  LAPACKE DEFAULT_MSG LAPACKE_INCLUDE_DIR LAPACKE_LIBRARY)

find_path(
  FFTW3_INCLUDE_DIR
  NAMES "fftw3.h"
  HINTS "$ENV{CONDA_PREFIX}"
  PATH_SUFFIXES "include")
find_library(
  FFTW3F_LIBRARY
  NAMES "fftw3f" "libfftw3f"
  HINTS "$ENV{CONDA_PREFIX}"
  PATH_SUFFIXES "lib")
find_package_handle_standard_args(
  FFTW3F DEFAULT_MSG FFTW3_INCLUDE_DIR FFTW3F_LIBRARY)

add_definitions(-DUSE_OPENBLAS)

if(TARGET OpenBLAS::OpenBLAS)
  target_link_libraries(common_libraries INTERFACE OpenBLAS::OpenBLAS)
elseif(DEFINED OpenBLAS_LIBRARIES)
  target_link_libraries(common_libraries INTERFACE ${OpenBLAS_LIBRARIES})
elseif(DEFINED OpenBLAS_LIB)
  target_link_libraries(common_libraries INTERFACE ${OpenBLAS_LIB})
else()
  message(FATAL_ERROR "OpenBLAS library target was not found")
endif()

if(DEFINED OpenBLAS_INCLUDE_DIRS)
  target_include_directories(common_libraries INTERFACE ${OpenBLAS_INCLUDE_DIRS})
elseif(DEFINED OpenBLAS_INCLUDE_DIR)
  target_include_directories(common_libraries INTERFACE ${OpenBLAS_INCLUDE_DIR})
endif()

target_include_directories(common_libraries INTERFACE ${LAPACKE_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR})
target_link_libraries(common_libraries INTERFACE ${LAPACKE_LIBRARY}
                                                 ${FFTW3F_LIBRARY})
