add_executable(test_boundary_policy test_boundary_policy.cpp)
target_include_directories(test_boundary_policy
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_boundary_policy.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME test_boundary_policy COMMAND test_boundary_policy)
set_tests_properties(test_boundary_policy PROPERTIES LABELS "boundary;contract")

add_test(
  NAME test_boundary_source_contract
  COMMAND ${CMAKE_COMMAND} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/test_boundary_source_contract.cmake)
set_tests_properties(test_boundary_source_contract
                     PROPERTIES LABELS "boundary;contract")
