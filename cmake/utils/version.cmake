# Parse version from pixi.toml (single source of truth)
file(READ "${PROJECT_ROOT_DIR}/pixi.toml" _PIXI_TOML)
string(REGEX MATCH "version = \"([^\"]+)\"" _ "${_PIXI_TOML}")
set(SPONGE_VERSION "${CMAKE_MATCH_1}")

if(NOT SPONGE_VERSION)
  message(FATAL_ERROR "Failed to parse version from pixi.toml")
endif()

message(STATUS "SPONGE version: ${SPONGE_VERSION}")

target_compile_definitions(common_libraries
                           INTERFACE SPONGE_VERSION_STR="${SPONGE_VERSION}")
