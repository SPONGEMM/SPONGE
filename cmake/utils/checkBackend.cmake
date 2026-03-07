function(CheckCuda)
  message(STATUS "Performing Test HAVE_CUDA")
  execute_process(
    COMMAND nvcc -V
    OUTPUT_VARIABLE NVCC_OUTPUT
    ERROR_VARIABLE NVCC_ERROR
    RESULT_VARIABLE NVCC_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NVCC_RESULT EQUAL 0)
    if(NVCC_OUTPUT MATCHES "Cuda compilation tools, release ([0-9]+\\.[0-9]+)")
      set(CUDA_VERSION
          "${CMAKE_MATCH_1}"
          PARENT_SCOPE)
      set(PARALLEL_BACKEND
          "cuda"
          PARENT_SCOPE)
      message(STATUS "Performing Test HAVE_CUDA - Success")
    else()
      message(STATUS "Performing Test HAVE_CUDA - Failed")
    endif()
  else()
    message(STATUS "Performing Test HAVE_CUDA - Failed")
  endif()
endfunction()

function(CheckHIP)
  message(STATUS "Performing Test HAVE_HIP")
  find_program(HIPCC_EXECUTABLE hipcc)
  find_program(HIPCONFIG_EXECUTABLE hipconfig)
  if(NOT HIPCC_EXECUTABLE OR NOT HIPCONFIG_EXECUTABLE)
    message(STATUS "Performing Test HAVE_HIP - Failed")
    return()
  endif()

  execute_process(
    COMMAND "${HIPCONFIG_EXECUTABLE}" --platform
    OUTPUT_VARIABLE HIP_PLATFORM_OUTPUT
    ERROR_VARIABLE HIP_PLATFORM_ERROR
    RESULT_VARIABLE HIP_PLATFORM_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(STRIP "${HIP_PLATFORM_OUTPUT}" HIP_PLATFORM_OUTPUT)

  if(HIP_PLATFORM_RESULT EQUAL 0 AND HIP_PLATFORM_OUTPUT STREQUAL "amd")
    set(PARALLEL_BACKEND
        "hip"
        PARENT_SCOPE)
    message(STATUS "Performing Test HAVE_HIP - Success")
  else()
    message(STATUS "Performing Test HAVE_HIP - Failed")
  endif()
endfunction()

function(CheckAVX512)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
  endif()

  check_cxx_source_runs(
    "
        #include <immintrin.h>
        int main() {
            __m512i a = _mm512_set1_epi32(1);
            __m512i b = _mm512_set1_epi32(2);
            __m512i c = _mm512_add_epi32(a, b);
            alignas(64) int32_t result[16];
            _mm512_store_si512(result, c);
            return result[0] == 3 ? 0 : 1;
        }
    "
    HAVE_AVX512)

  if(HAVE_AVX512)
    set(PARALLEL_BACKEND
        "avx512"
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckAVX2)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
  endif()
  check_cxx_source_runs(
    "
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_set1_epi32(1);
            __m256i b = _mm256_set1_epi32(2);
            __m256i c = _mm256_add_epi32(a, b);
            return _mm256_extract_epi32(c, 0) == 3 ? 0 : 1;
        }
    "
    HAVE_AVX2)
  if(HAVE_AVX2)
    set(PARALLEL_BACKEND
        "avx2"
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckAVX)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx")
  endif()
  check_cxx_source_runs(
    "
        #include <immintrin.h>
        #include <math.h>
        int main() {
            __m256 a = _mm256_set1_ps(1.0f);
            __m256 b = _mm256_set1_ps(2.0f);
            __m256 c = _mm256_add_ps(a, b);
            return fabsf(_mm256_cvtss_f32(c) - 3.0f) < 1e-6 ? 0 : 1;
        }
    "
    HAVE_AVX)
  if(HAVE_AVX)
    set(PARALLEL_BACKEND
        "avx"
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckSSE42)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:SSE4.2")
  else()
    set(CMAKE_REQUIRED_FLAGS "-msse4.2")
  endif()
  check_cxx_source_runs(
    "
        #include <immintrin.h>
        int main() {
            __m128i a = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
            __m128i b = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            int result = _mm_cmpestri(a, 16, b, 16, _SIDD_CMP_EQUAL_ANY);
            return (result == 0) ? 0 : 1;
        }
    "
    HAVE_SSE42)
  if(HAVE_SSE42)
    set(PARALLEL_BACKEND
        "sse42"
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckSVE2)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:SVE")
  else()
    set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve2")
  endif()
  check_cxx_source_runs(
    "
        #include <arm_sve.h>
        int main() {
            svint32_t a = svdup_s32(1);
            svint32_t b = svdup_s32(2);
            svint32_t c = svadd_s32_x(svptrue_b32(), a, b);
            int32_t result[svcntw()];
            svst1_s32(svptrue_b32(), result, c);
            return (result[0] == 3) ? 0 : 1;
        }
    "
    HAVE_SVE2)
  if(HAVE_SVE2)
    set(PARALLEL_BACKEND
        "sve2"
        PARENT_SCOPE)
    set(ON_ARM
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckSVE)
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:SVE")
  else()
    set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve")
  endif()
  check_cxx_source_runs(
    "
        #include <arm_sve.h>
        int main() {
            svint32_t a = svdup_s32(1);
            svint32_t b = svdup_s32(2);
            svint32_t c = svadd_s32_x(svptrue_b32(), a, b);
            int32_t result[svcntw()];
            svst1_s32(svptrue_b32(), result, c);
            return (result[0] == 3) ? 0 : 1;
        }
    "
    HAVE_SVE)
  if(HAVE_SVE)
    set(PARALLEL_BACKEND
        "sve"
        PARENT_SCOPE)
    set(ON_ARM
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()

function(CheckNeon)
  set(CMAKE_REQUIRED_FLAGS "")
  check_cxx_source_runs(
    "
        #include <arm_neon.h>
        int main() {
            int32x2_t a = vdup_n_s32(1);  // [1, 1]
            int32x2_t b = vdup_n_s32(2);  // [2, 2]
            int32x2_t result = vadd_s32(a, b);  // [3, 3]
            return vget_lane_s32(result, 0) == 3 ? 0 : 1;
    }
    "
    HAVE_NEON)
  if(HAVE_NEON)
    set(PARALLEL_BACKEND
        "neon"
        PARENT_SCOPE)
    set(ON_ARM
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()

set(PARALLEL_BACKEND_GPU_CHECK_FUNCTIONS CheckCuda CheckHIP)

set(PARALLEL_BACKEND_CPU_CHECK_FUNCTIONS
    CheckAVX512
    CheckAVX2
    CheckAVX
    CheckSSE42
    CheckSVE2
    CheckSVE
    CheckNeon)
