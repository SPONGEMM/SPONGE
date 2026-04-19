enable_language(HIP)
set(CPP_DIALECT "HIP")
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

include("${PROJECT_ROOT_DIR}/cmake/math/hip.cmake")

add_definitions(-DUSE_GPU)
add_definitions(-DUSE_HIP)

# ROCm 6.4 clang crashes in the AMDGPU backend at -O3 on these generated ERI
# kernels. Keep the workaround scoped to HIP and to the affected files.
set_source_files_properties(
  ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/sp/sp_kernels.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/md/md_kernels.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/Rys/rys_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-O1")
