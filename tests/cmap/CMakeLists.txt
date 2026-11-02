add_executable(test_cmap_grid_index test_cmap_grid_index.cpp)
target_include_directories(test_cmap_grid_index
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
if(PARALLEL_BACKEND STREQUAL "cuda")
  set_source_files_properties(test_cmap_grid_index.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME test_cmap_grid_index COMMAND test_cmap_grid_index)
set_tests_properties(test_cmap_grid_index PROPERTIES LABELS "cmap;contract")
