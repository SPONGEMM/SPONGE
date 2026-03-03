enable_language(CUDA)
set(CPP_DIALECT "CUDA")
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_ARCH
    "native"
    CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

include("${PROJECT_ROOT_DIR}/cmake/math/cuda.cmake")

add_definitions(-DUSE_GPU)
add_definitions(-DUSE_CUDA)

# Link CUDA Driver API
target_link_libraries(common_libraries INTERFACE CUDA::cuda_driver)
