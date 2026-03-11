include("${PROJECT_ROOT_DIR}/cmake/parallel/none.cmake")

if(MSVC)
  message(
    WARNING "NEON backend selected, but no explicit MSVC ARM NEON flag is set")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM)")
  target_compile_options(common_libraries INTERFACE -mfpu=neon)
else()
  target_compile_options(common_libraries INTERFACE -march=armv8-a+simd)
endif()
target_compile_definitions(common_libraries INTERFACE USE_NEON)
