add_executable(test_rerun_cells test_rerun_cells.cpp)
target_link_libraries(test_rerun_cells PRIVATE sponge_h5_bundle_test_common)
add_test(NAME test_rerun_cells
         COMMAND test_rerun_cells $<TARGET_FILE:SPONGE>)
set_tests_properties(test_rerun_cells PROPERTIES LABELS "rerun;runtime;cell"
                                               SKIP_RETURN_CODE 77 TIMEOUT 180)
