file(GLOB_RECURSE AMD_COMGR_CONFIG "$ENV{ROCM_PATH}/*/amd_comgr-config.cmake")

if(AMD_COMGR_CONFIG)
  get_filename_component(amd_comgr_DIR "${AMD_COMGR_CONFIG}" DIRECTORY)
endif()

find_path(
  HSA_HEADER
  NAMES hsa/hsa.h
  PATHS $ENV{ROCM_PATH} $ENV{ROCM_PATH}/.hyhal
  PATH_SUFFIXES include)

find_package(amd_comgr CONFIG REQUIRED PATHS ${amd_comgr_DIR})
