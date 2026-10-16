if(NOT MPI OR NOT PARALLEL_BACKEND STREQUAL "cuda")
  return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# The plugin is a standalone Python extension; build it in this test tree so
# CTest exercises the exact PRIPS source from the SPONGE checkout.
add_subdirectory(${PROJECT_ROOT_DIR}/plugins/prips
                 ${CMAKE_CURRENT_BINARY_DIR}/prips_plugin)

set(LOCAL_MP_TEST_GPU_COUNT
    2
    CACHE STRING "GPUs available to local-MP CTest fixtures")

set(LOCAL_MP_CASES
    test_local_mp_single_gpu_parity test_local_mp_two_pp_rank_parity
    test_local_mp_skin_rebuild test_local_mp_md_halo_refresh)

foreach(case IN LISTS LOCAL_MP_CASES)
  if(case STREQUAL "test_local_mp_single_gpu_parity"
     OR case STREQUAL "test_local_mp_skin_rebuild")
    set(ranks 1)
  else()
    set(ranks 2)
  endif()
  if(LOCAL_MP_TEST_GPU_COUNT LESS ranks)
    continue()
  endif()
  add_test(
    NAME ${case}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_local_mp_mpi.py
      --case ${case} --sponge $<TARGET_FILE:SPONGE> --plugin
      $<TARGET_FILE:_prips> --mpiexec ${MPIEXEC_EXECUTABLE}
      --numproc-flag=${MPIEXEC_NUMPROC_FLAG} --work-root
      ${CMAKE_CURRENT_BINARY_DIR}/runs)
  set_tests_properties(
    ${case}
    PROPERTIES LABELS
               "plugin;local-mp;gpu;gpu${ranks}"
               PROCESSORS
               ${ranks}
               TIMEOUT
               240
               RESOURCE_LOCK
               local_mp_gpu)
endforeach()

# Model-independent qualification of the common native edge-output service.
foreach(case IN ITEMS two_pp empty_rank)
  set(ranks 2)
  if(LOCAL_MP_TEST_GPU_COUNT LESS ranks)
    continue()
  endif()
  add_test(
    NAME test_edge_output_${case}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_edge_output_mpi.py
      --case ${case} --sponge $<TARGET_FILE:SPONGE> --plugin
      $<TARGET_FILE:_prips> --mpiexec ${MPIEXEC_EXECUTABLE}
      --numproc-flag=${MPIEXEC_NUMPROC_FLAG} --work-root
      ${CMAKE_CURRENT_BINARY_DIR}/runs)
  set_tests_properties(
    test_edge_output_${case}
    PROPERTIES LABELS
               "plugin;local-mp;edge-output;gpu;gpu${ranks}"
               PROCESSORS
               ${ranks}
               TIMEOUT
               240
               RESOURCE_LOCK
               local_mp_gpu)
endforeach()

foreach(ranks IN ITEMS 1 2)
  if(LOCAL_MP_TEST_GPU_COUNT LESS ranks)
    continue()
  endif()
  add_test(
    NAME test_restart_state_${ranks}
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_restart_state_mpi.py --sponge
      $<TARGET_FILE:SPONGE> --plugin $<TARGET_FILE:_prips> --mpiexec
      ${MPIEXEC_EXECUTABLE} --ranks ${ranks} --work-root
      ${CMAKE_CURRENT_BINARY_DIR}/restart-${ranks})
  set_tests_properties(
    test_restart_state_${ranks}
    PROPERTIES LABELS
               "plugin;local-mp;restart;gpu;gpu${ranks}"
               PROCESSORS
               ${ranks}
               TIMEOUT
               300
               RESOURCE_LOCK
               local_mp_gpu)
endforeach()
