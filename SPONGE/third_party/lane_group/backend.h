#ifndef SPONGE_LANE_GROUP_BACKEND_H
#define SPONGE_LANE_GROUP_BACKEND_H

#ifdef USE_CUDA
#include "cuda.h"
#elif defined(USE_HIP)
#include "hip.h"
#elif defined(USE_SVE2) || defined(USE_SVE)
#include "sve.h"
#elif defined(USE_AVX512)
#include "avx512.h"
#elif defined(USE_AVX2)
#include "avx2.h"
#elif defined(USE_AVX)
#include "avx.h"
#elif defined(USE_SSE42)
#include "sse42.h"
#elif defined(USE_NEON)
#include "neon.h"
#else
#include "scalar.h"
#endif

#endif  // SPONGE_LANE_GROUP_BACKEND_H
