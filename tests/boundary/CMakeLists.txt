add_executable(test_boundary_policy test_boundary_policy.cpp)
target_include_directories(test_boundary_policy
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_boundary_policy.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME test_boundary_policy COMMAND test_boundary_policy)
set_tests_properties(test_boundary_policy PROPERTIES LABELS "boundary;contract")

if(NOT PARALLEL_BACKEND STREQUAL "cuda")
  add_executable(test_boundary_openmp test_boundary_openmp.cpp)
  target_include_directories(test_boundary_openmp
                             PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
  target_link_libraries(test_boundary_openmp PRIVATE OpenMP::OpenMP_CXX)
  foreach(threads IN ITEMS 1 4)
    add_test(NAME test_boundary_openmp_${threads} COMMAND test_boundary_openmp)
    set_tests_properties(
      test_boundary_openmp_${threads}
      PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${threads}" LABELS
                 "boundary;regression")
  endforeach()
endif()

add_test(
  NAME test_boundary_source_contract
  COMMAND ${CMAKE_COMMAND} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/test_boundary_source_contract.cmake)
set_tests_properties(test_boundary_source_contract
                     PROPERTIES LABELS "boundary;contract")
