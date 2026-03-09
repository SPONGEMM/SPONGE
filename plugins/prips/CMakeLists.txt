cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME _prips)
project(${PROJECT_NAME} LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(PRIPS_INSTALL_DIR prips)

set(PRIPS_SOURCES src/main.cpp)

add_library(${PROJECT_NAME} SHARED ${PRIPS_SOURCES})
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME ${PROJECT_NAME}
             PREFIX ""
             SUFFIX ".so")
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE PYTHON_LIB_PATH="${Python3_LIBRARIES}")

file(
  GENERATE
  OUTPUT ${CMAKE_BINARY_DIR}/pylib.txt
  CONTENT "${Python3_LIBRARIES}")
install(
  FILES ${CMAKE_BINARY_DIR}/pylib.txt
  EXCLUDE_FROM_ALL
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})

install(
  TARGETS ${PROJECT_NAME}
  EXCLUDE_FROM_ALL
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})
