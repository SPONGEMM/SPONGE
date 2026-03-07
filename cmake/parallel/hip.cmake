enable_language(HIP)
set(CPP_DIALECT "HIP")
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

include("${PROJECT_ROOT_DIR}/cmake/math/hip.cmake")

add_definitions(-DUSE_GPU)
add_definitions(-DUSE_HIP)
