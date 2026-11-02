# Prevent old wrappers and a second box owner from returning unnoticed.
set(source_root "${CMAKE_CURRENT_LIST_DIR}/../../SPONGE")
file(GLOB_RECURSE sources "${source_root}/*.h" "${source_root}/*.hpp"
     "${source_root}/*.cpp")
foreach(source IN LISTS sources)
  file(READ "${source}" contents)
  if(contents
     MATCHES
     "BoundarySnapshot|PeriodicCell|Make_Periodic_Cell|Make_Periodic_Boundary|Make_Open_Boundary|Get_Boundary_Snapshot|Get_Periodic_Cell"
  )
    message(FATAL_ERROR "Legacy boundary interface in ${source}")
  endif()
endforeach()
file(READ "${source_root}/MD_core/pbc.h" owner)
if(NOT owner MATCHES "Boundary boundary;" OR owner MATCHES
                                             "LTMatrix3 (cell|rcell);")
  message(FATAL_ERROR "MD core must own one Boundary, not duplicate cell/rcell")
endif()

# Both compilation paths must implement exactly the same lightweight API.
file(READ "${source_root}/utils/boundary.h" host)
file(READ "${source_root}/third_party/jit/jit_boundary.h" jit)
string(FIND "${host}" "enum class BoundaryPolicy" start)
string(FIND "${host}" "static_assert" end)
math(EXPR length "${end} - ${start}")
string(SUBSTRING "${host}" ${start} ${length} host)
string(REPLACE "std::uint8_t" "unsigned char" host "${host}")
string(FIND "${jit}" "enum class BoundaryPolicy" start)
string(FIND "${jit}" ")JIT" end)
math(EXPR length "${end} - ${start}")
string(SUBSTRING "${jit}" ${start} ${length} jit)
string(REGEX REPLACE "[ \r\n\t]" "" host "${host}")
string(REGEX REPLACE "[ \r\n\t]" "" jit "${jit}")
if(NOT host STREQUAL jit)
  message(FATAL_ERROR "Host and JIT Boundary implementations differ")
endif()
