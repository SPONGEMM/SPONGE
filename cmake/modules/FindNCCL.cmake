include(FindPackageHandleStandardArgs)

find_path(
  NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${NCCL_ROOT} ENV NCCL_ROOT
  PATH_SUFFIXES include)

find_library(
  NCCL_LIBRARY
  NAMES nccl
  HINTS ${NCCL_ROOT} ENV NCCL_ROOT
  PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(NCCL REQUIRED_VARS NCCL_LIBRARY
                                                     NCCL_INCLUDE_DIR)

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
