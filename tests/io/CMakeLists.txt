if(SPONGE_WITH_XTC)
  add_executable(
    test_xtc_trajectory_writer
    test_xtc_trajectory_writer.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/io/xtc_trajectory_writer.cpp)
  target_include_directories(test_xtc_trajectory_writer
                             PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
  target_link_libraries(test_xtc_trajectory_writer PRIVATE sponge_xdrfile)
  add_test(NAME test_xtc_trajectory_writer
           COMMAND test_xtc_trajectory_writer
                   ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(test_xtc_trajectory_writer
                       PROPERTIES LABELS "io;xtc;contract")
endif()
