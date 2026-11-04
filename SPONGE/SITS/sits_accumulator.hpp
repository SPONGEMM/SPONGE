#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPONGE_SITS_HD __host__ __device__
#else
#define SPONGE_SITS_HD
#endif

namespace SitsAccumulator
{
SPONGE_SITS_HD inline float Log_Add(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

SPONGE_SITS_HD inline void Accumulate(float& log_pk, float gf, float gfsum,
                                      bool reset)
{
    const float log_probability = gf - gfsum;
    if (reset)
    {
        // Every record has unit total probability. Do not read the previous
        // accumulator here: it may be uninitialized or belong to an old block.
        log_pk = log_probability;
    }
    else
    {
        log_pk = Log_Add(log_pk, log_probability);
    }
}
}  // namespace SitsAccumulator

#undef SPONGE_SITS_HD
