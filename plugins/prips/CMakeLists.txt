cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME _prips)
project(${PROJECT_NAME} LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Parse version from the packaged Python module so standalone builds do not
# depend on monorepo-only files.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/prips/__init__.py" _PRIPS_INIT)
string(REGEX MATCH "__version__ = \"([^\"]+)\"" _ "${_PRIPS_INIT}")
set(PRIPS_VERSION "${CMAKE_MATCH_1}")

if(NOT DEFINED PRIPS_INSTALL_DIR)
  set(PRIPS_INSTALL_DIR prips)
endif()
set(PRIPS_INSTALL_EXCLUDE EXCLUDE_FROM_ALL)
if(PRIPS_BUNDLED)
  set(PRIPS_INSTALL_EXCLUDE)
endif()

set(PRIPS_SOURCES src/main.cpp)

add_library(${PROJECT_NAME} SHARED ${PRIPS_SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE PRIPS_VERSION="${PRIPS_VERSION}")
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME ${PROJECT_NAME}
             PREFIX ""
             SUFFIX ".so")
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python ${CMAKE_DL_LIBS})
if(PRIPS_BUNDLED)
  target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE PRIPS_BUNDLED_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  # Preserve the embedding library chosen at configure time: a system Python
  # with the same SONAME may use a different package environment.
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                   TRUE)
  set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                   "${PRIPS_BUILD_PACKAGE_DIR}")
  file(
    GLOB_RECURSE PRIPS_PYTHON_FILES CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/prips"
    "${CMAKE_CURRENT_SOURCE_DIR}/prips/*.py")
  foreach(PRIPS_PYTHON_FILE IN LISTS PRIPS_PYTHON_FILES)
    configure_file("prips/${PRIPS_PYTHON_FILE}"
                   "${PRIPS_BUILD_PACKAGE_DIR}/${PRIPS_PYTHON_FILE}" COPYONLY)
  endforeach()
endif()

file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pylib.txt
  CONTENT "${Python3_LIBRARIES}")
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/pylib.txt ${PRIPS_INSTALL_EXCLUDE}
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES
        ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-DLPACK ${PRIPS_INSTALL_EXCLUDE}
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})

install(
  TARGETS ${PROJECT_NAME} ${PRIPS_INSTALL_EXCLUDE}
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/prips/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/prips/__main__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/prips/structure_input.py
        ${CMAKE_CURRENT_SOURCE_DIR}/prips/Sponge.pyi
        ${PRIPS_INSTALL_EXCLUDE}
  COMPONENT python_modules
  DESTINATION ${PRIPS_INSTALL_DIR})
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/prips/local_mp/
  DESTINATION ${PRIPS_INSTALL_DIR}/local_mp
  COMPONENT python_modules
  FILES_MATCHING
  PATTERN "*.py")
