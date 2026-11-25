#include <cmath>
#include <iostream>

#include "PM_force/grid_coordinate.h"

namespace
{
struct Sample
{
    float coordinate;
    int size;
    int expected_index;
    float expected_fraction;
};

// Non-power-of-two and different axis sizes also cover rounding after scaling.
const Sample samples[] = {
    {-5e-7f, 32, 0, 0},      {-5e-7f, 64, 0, 0},      {-5e-7f, 96, 0, 0},
    {0, 64, 0, 0},           {64, 64, 0, 0},          {-64, 64, 0, 0},
    {5e-7f, 64, 0, 5e-7f},   {16.25f, 64, 16, 0.25f}, {-0.25f, 64, 63, 0.75f},
    {63.75f, 64, 63, 0.75f}, {80.25f, 64, 16, 0.25f}, {-80.25f, 64, 47, 0.75f},
};
constexpr int count = sizeof(samples) / sizeof(samples[0]);

__host__ __device__ PME_GRID_COORDINATE Evaluate(Sample sample)
{
    float fraction = sample.coordinate / sample.size;
    fraction -= floorf(fraction);
    return PME_Get_Grid_Coordinate(fraction, sample.size);
}

bool Check(const char* backend, const PME_GRID_COORDINATE* results)
{
    bool ok = true;
    for (int i = 0; i < count; ++i)
    {
        const auto result = results[i];
        const auto sample = samples[i];
        if (result.index != sample.expected_index ||
            !std::isfinite(result.fraction) || result.fraction < 0 ||
            result.fraction >= 1 ||
            std::fabs(result.fraction - sample.expected_fraction) > 1e-6f)
        {
            std::cerr << backend << " sample " << i << ": (" << result.index
                      << ", " << result.fraction << ") expected ("
                      << sample.expected_index << ", "
                      << sample.expected_fraction << ")\n";
            ok = false;
        }
    }
    return ok;
}
}  // namespace

#ifdef USE_CUDA
__global__ void Evaluate_Device(const Sample* input,
                                PME_GRID_COORDINATE* output)
{
    int i = threadIdx.x;
    if (i < count) output[i] = Evaluate(input[i]);
}
#endif

int main()
{
    PME_GRID_COORDINATE results[count];
    for (int i = 0; i < count; ++i) results[i] = Evaluate(samples[i]);
    bool ok = Check("host", results);
#ifdef USE_CUDA
    Sample* input = nullptr;
    PME_GRID_COORDINATE* output = nullptr;
    if (cudaMalloc((void**)&input, sizeof(samples)) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&output, sizeof(results)) != cudaSuccess)
    {
        cudaFree(input);
        return 1;
    }
    bool copied = cudaMemcpy(input, samples, sizeof(samples),
                             cudaMemcpyHostToDevice) == cudaSuccess;
    if (copied)
    {
        Evaluate_Device<<<1, 32>>>(input, output);
        copied = cudaGetLastError() == cudaSuccess &&
                 cudaMemcpy(results, output, sizeof(results),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    ok = copied && Check("CUDA", results) && ok;
    ok = (cudaFree(input) == cudaSuccess) && ok;
    ok = (cudaFree(output) == cudaSuccess) && ok;
#endif
    return ok ? 0 : 1;
}
