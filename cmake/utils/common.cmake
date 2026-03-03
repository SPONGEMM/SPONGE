include(CheckCXXSourceRuns)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -w")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /W0")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL
                                          "MinSizeRel")
  # Some toolchains inject "-g -G" into CMAKE_CUDA_FLAGS by default. Keep
  # release CUDA builds free of device-debug flags.
  string(REGEX REPLACE "(^| )-G( |$)" " " CMAKE_CUDA_FLAGS
                       "${CMAKE_CUDA_FLAGS}")
  string(REGEX REPLACE "(^| )-g( |$)" " " CMAKE_CUDA_FLAGS
                       "${CMAKE_CUDA_FLAGS}")
  string(REGEX REPLACE " +" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
endif()

add_library(common_libraries INTERFACE)

set(CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake/modules")

if(UNIX)
  target_link_libraries(common_libraries INTERFACE ${CMAKE_DL_LIBS})
endif()

function(SearchOptions SEARCH_PATH OUTPUT_VAR)
  file(GLOB OPTION_FILES ${SEARCH_PATH})
  set(OPTION_LIST "")
  foreach(OPTION_FILE ${OPTION_FILES})
    get_filename_component(OPTION_CODE ${OPTION_FILE} NAME_WE)
    list(APPEND OPTION_LIST ${OPTION_CODE})
  endforeach()
  set(${OUTPUT_VAR}
      "${OPTION_LIST}"
      PARENT_SCOPE)
endfunction()

find_package(OpenMP REQUIRED)
if(PARALLEL STREQUAL "cuda")
  if(MSVC)
    # nvcc must receive OpenMP host flags via -Xcompiler on Windows.
    target_compile_options(
      common_libraries
      INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>")
  elseif(OpenMP_CXX_FLAGS)
    string(REPLACE " " "," OPENMP_CUDA_HOST_FLAGS "${OpenMP_CXX_FLAGS}")
    target_compile_options(
      common_libraries
      INTERFACE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OPENMP_CUDA_HOST_FLAGS}>")
  endif()
  if(OpenMP_CXX_LIBRARIES)
    target_link_libraries(common_libraries INTERFACE ${OpenMP_CXX_LIBRARIES})
  endif()
else()
  target_link_libraries(common_libraries INTERFACE OpenMP::OpenMP_CXX)
  if(MSVC)
    target_compile_options(common_libraries INTERFACE /openmp:llvm)
  endif()
endif()
