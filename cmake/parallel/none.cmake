set(CPP_DIALECT "CXX")

# MSVC miscompiles shared Boundary access in optimized OpenMP CPU kernels. CUDA
# uses its own backend and retains MSVC as the NVCC host compiler.
if(WIN32
   AND NOT
       (CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM"
        AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
        AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  message(
    FATAL_ERROR
      "Windows CPU builds require Intel oneAPI icx-cl. Configure a fresh build directory "
      "with -DCMAKE_C_COMPILER=icx-cl -DCMAKE_CXX_COMPILER=icx-cl, "
      "or use pixi run -e dev-cpu configure.")
endif()

add_definitions(-DUSE_CPU)
find_package(ZLIB REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(Clang CONFIG QUIET)
target_include_directories(common_libraries INTERFACE ${LLVM_INCLUDE_DIRS})
if(WIN32)
  llvm_map_components_to_libnames(
    SPONGE_LLVM_LIBS
    support
    core
    executionengine
    native
    nativecodegen
    orcjit
    runtimedyld
    targetparser)
  target_link_libraries(common_libraries INTERFACE ${SPONGE_LLVM_LIBS})
  target_link_libraries(common_libraries INTERFACE clangFrontendTool)
else()
  target_link_libraries(common_libraries INTERFACE LLVM clang-cpp)
endif()

if(ON_ARM)
  message(STATUS "Use Open Source Math Libraries")
  include("${PROJECT_ROOT_DIR}/cmake/math/open_source.cmake")
else()
  message(STATUS "Use MKL as Math Library")
  include("${PROJECT_ROOT_DIR}/cmake/math/mkl.cmake")
endif()
