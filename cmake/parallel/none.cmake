set(CPP_DIALECT "CXX")

add_definitions(-DUSE_CPU)

if(ON_ARM)
  message(STATUS "Use Open Source Math Libraries")
  include("${PROJECT_ROOT_DIR}/cmake/math/open_source.cmake")
else()
  message(STATUS "Use MKL as Math Library")
  include("${PROJECT_ROOT_DIR}/cmake/math/mkl.cmake")
endif()
