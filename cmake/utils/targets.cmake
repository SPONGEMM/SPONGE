message(STATUS "Load build targets from ${TARGETS_FILE}")

set(TARGETS
    "SPONGE"
    CACHE STRING "targets for SPONGE" FORCE)

message(STATUS "-- Build targets: ${TARGETS}")

searchoptions("${PROJECT_ROOT_DIR}/cmake/targets/*.cmake" TARGETS_OPTIONS)

foreach(CURRENT_TARGET ${TARGETS})
  set(SOURCES)
  set(TARGET_SOURCE_LANGUAGE "${CPP_DIALECT}")
  if(CURRENT_TARGET IN_LIST TARGETS_OPTIONS)
    include("${PROJECT_ROOT_DIR}/cmake/targets/${CURRENT_TARGET}.cmake")
    if(SOURCES)
      set_source_files_properties(${SOURCES}
                                  PROPERTIES LANGUAGE ${TARGET_SOURCE_LANGUAGE})
    endif()
    target_link_libraries(${CURRENT_TARGET} PUBLIC common_libraries)
  else()
    message(FATAL_ERROR "Unrecognized build target option: ${CURRENT_TARGET}")
  endif()
endforeach()

message(
  STATUS "------------------------------------------------------------------")
