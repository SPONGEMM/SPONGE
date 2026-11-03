#pragma once

#include <cfloat>
#include <limits>
#include <memory>
#include <type_traits>

#include "../common.h"

enum class ConstraintPrecision
{
    Single,
    Double
};

template <class Real>
__host__ __device__ constexpr Real ConstraintEpsilon()
{
    return std::is_same<Real, float>::value ? Real(FLT_EPSILON)
                                            : Real(DBL_EPSILON);
}

// Constraint arithmetic is independent of the float MD state. In double mode
// all iteration buffers stay double until the complete solve is written back.
template <class Real>
struct ConstraintVector
{
    Real x, y, z;
    __host__ __device__ ConstraintVector() : x(0), y(0), z(0) {}
    __host__ __device__ ConstraintVector(Real x, Real y, Real z)
        : x(x), y(y), z(z)
    {
    }
    friend __host__ __device__ ConstraintVector operator+(ConstraintVector a,
                                                          ConstraintVector b)
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    friend __host__ __device__ ConstraintVector operator-(ConstraintVector a,
                                                          ConstraintVector b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    friend __host__ __device__ ConstraintVector operator*(Real s,
                                                          ConstraintVector a)
    {
        return {s * a.x, s * a.y, s * a.z};
    }
    friend __host__ __device__ Real operator*(ConstraintVector a,
                                              ConstraintVector b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    friend __host__ __device__ ConstraintVector operator^(ConstraintVector a,
                                                          ConstraintVector b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
    }
};

template <class Real>
struct ConstraintTensor
{
    Real a11, a21, a22, a31, a32, a33;
    __host__ __device__ ConstraintTensor(Real a = 0, Real b = 0, Real c = 0,
                                         Real d = 0, Real e = 0, Real f = 0)
        : a11(a), a21(b), a22(c), a31(d), a32(e), a33(f)
    {
    }
    friend __host__ __device__ ConstraintTensor operator+(ConstraintTensor a,
                                                          ConstraintTensor b)
    {
        return {a.a11 + b.a11, a.a21 + b.a21, a.a22 + b.a22,
                a.a31 + b.a31, a.a32 + b.a32, a.a33 + b.a33};
    }
    friend __host__ __device__ ConstraintTensor operator-(ConstraintTensor a,
                                                          ConstraintTensor b)
    {
        return {a.a11 - b.a11, a.a21 - b.a21, a.a22 - b.a22,
                a.a31 - b.a31, a.a32 - b.a32, a.a33 - b.a33};
    }
    friend __host__ __device__ ConstraintTensor operator*(Real s,
                                                          ConstraintTensor a)
    {
        return {s * a.a11, s * a.a21, s * a.a22,
                s * a.a31, s * a.a32, s * a.a33};
    }
    friend __device__ void atomicAdd(ConstraintTensor* a, ConstraintTensor b)
    {
        atomicAdd(&a->a11, b.a11);
        atomicAdd(&a->a21, b.a21);
        atomicAdd(&a->a22, b.a22);
        atomicAdd(&a->a31, b.a31);
        atomicAdd(&a->a32, b.a32);
        atomicAdd(&a->a33, b.a33);
    }
};

__host__ __device__ inline float ConstraintSqrt(float x) { return sqrtf(x); }
__host__ __device__ inline double ConstraintSqrt(double x) { return sqrt(x); }
template <class Real>
__host__ __device__ inline Real ConstraintReciprocalNorm(Real x, Real y, Real z)
{
    return Real(1) / ConstraintSqrt(x * x + y * y + z * z);
}

template <class Real>
__host__ __device__ inline ConstraintVector<Real> Get_Displacement(
    ConstraintVector<Real> a, ConstraintVector<Real> b,
    const Boundary& boundary)
{
    auto d = a - b;
    if (boundary.policy == BoundaryPolicy::Open) return d;
    // Keep the existing cell and periodic-image convention, evaluate it in
    // Real.
    const auto& r = boundary.rcell;
    const Real ix = floor(d.x * Real(r.a11) + d.y * Real(r.a21) +
                          d.z * Real(r.a31) + Real(.5));
    const Real iy = floor(d.y * Real(r.a22) + d.z * Real(r.a32) + Real(.5));
    const Real iz = floor(d.z * Real(r.a33) + Real(.5));
    const auto& c = boundary.cell;
    return {d.x - ix * Real(c.a11) - iy * Real(c.a21) - iz * Real(c.a31),
            d.y - iy * Real(c.a22) - iz * Real(c.a32), d.z - iz * Real(c.a33)};
}

template <class Real>
__host__ __device__ inline ConstraintTensor<Real> Get_Virial_From_Force_Dis(
    const ConstraintVector<Real>& a, const ConstraintVector<Real>& b)
{
    return {a.x * b.x,
            a.x * b.y + a.y * b.x,
            a.y * b.y,
            a.x * b.z + a.z * b.x,
            a.y * b.z + a.z * b.y,
            a.z * b.z};
}

template <class Real>
__device__ inline void Warp_Sum_To(ConstraintTensor<Real>* y,
                                   ConstraintTensor<Real>& x, int width)
{
#ifdef GPU_ARCH_NAME
    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        x.a11 += deviceShflDown(FULL_MASK, x.a11, offset, width);
        x.a21 += deviceShflDown(FULL_MASK, x.a21, offset, width);
        x.a22 += deviceShflDown(FULL_MASK, x.a22, offset, width);
        x.a31 += deviceShflDown(FULL_MASK, x.a31, offset, width);
        x.a32 += deviceShflDown(FULL_MASK, x.a32, offset, width);
        x.a33 += deviceShflDown(FULL_MASK, x.a33, offset, width);
    }
    if (threadIdx.x == 0)
#endif
        atomicAdd(y, x);
}

template <class To, class From>
static __global__ void Constraint_Copy(int n, To* to, const From* from)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
#else
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
#endif
    {
        if constexpr (std::is_arithmetic<To>::value)
            to[i] = To(from[i]);
        else
            to[i] = To(from[i].x, from[i].y, from[i].z);
    }
}

template <class Real>
static __global__ void Constraint_Write_Stress(
    LTMatrix3* output, const ConstraintTensor<Real>* input)
{
#ifdef USE_GPU
    if (blockIdx.x || threadIdx.x) return;
#endif
    output->a11 += float(input->a11);
    output->a21 += float(input->a21);
    output->a22 += float(input->a22);
    output->a31 += float(input->a31);
    output->a32 += float(input->a32);
    output->a33 += float(input->a33);
}

template <class Real>
struct ConstraintWorkspace
{
    ConstraintVector<Real>*q = nullptr, *v = nullptr;
    Real *mass = nullptr, *inverse = nullptr;
    ConstraintTensor<Real>* stress = nullptr;
    int capacity = 0;
    ConstraintWorkspace() = default;
    ConstraintWorkspace(const ConstraintWorkspace&) = delete;
    ConstraintWorkspace& operator=(const ConstraintWorkspace&) = delete;
    ~ConstraintWorkspace() { Clear(); }
    void Clear()
    {
        if (q) deviceFree(q);
        if (v) deviceFree(v);
        if (mass) deviceFree(mass);
        if (inverse) deviceFree(inverse);
        if (stress) deviceFree(stress);
        q = v = nullptr;
        mass = inverse = nullptr;
        stress = nullptr;
        capacity = 0;
    }
    void Reserve(int n)
    {
        if (n <= capacity) return;
        Clear();
        capacity = n;
        Device_Malloc_Safely((void**)&q, n * sizeof(*q));
        Device_Malloc_Safely((void**)&v, n * sizeof(*v));
        Device_Malloc_Safely((void**)&mass, n * sizeof(*mass));
        Device_Malloc_Safely((void**)&inverse, n * sizeof(*inverse));
        Device_Malloc_Safely((void**)&stress, sizeof(*stress));
    }
    template <class To, class From>
    void Copy(int n, To* to, const From* from)
    {
        if (n <= 0) return;
        // Parentheses shield the template comma from the launch macro.
        Launch_Device_Kernel((Constraint_Copy<To, From>), (n + 127) / 128, 128,
                             0, 0, n, to, from);
    }
    void Coordinates(int n, const VECTOR* crd)
    {
        Reserve(n);
        Copy(n, q, crd);
    }
    void Begin(int n, const VECTOR* crd, const VECTOR* vel,
               const float* m = nullptr, const float* inv = nullptr)
    {
        Coordinates(n, crd);
        Copy(n, v, vel);
        if (m) Copy(n, mass, m);
        if (inv) Copy(n, inverse, inv);
        if (stress) deviceMemset(stress, 0, sizeof(*stress));
    }
    void End(int n, VECTOR* crd, VECTOR* vel, LTMatrix3* output = nullptr,
             bool coordinates = true)
    {
        if (coordinates) Copy(n, crd, q);
        Copy(n, vel, v);
        if (output)
            Launch_Device_Kernel(Constraint_Write_Stress<Real>, 1, 1, 0, 0,
                                 output, stress);
    }
};

// One selected backend owns its precision-specific caches. Selection happens
// at initialization and survives update-group remapping.
template <template <class> class Solver>
struct ConstraintDispatch
{
    std::unique_ptr<Solver<float>> single;
    std::unique_ptr<Solver<double>> precise;
    ConstraintWorkspace<float> single_state;
    ConstraintWorkspace<double> double_state;
    void Select(ConstraintPrecision precision)
    {
        single.reset();
        precise.reset();
        single_state.Clear();
        double_state.Clear();
        if (precision == ConstraintPrecision::Double)
            precise = std::make_unique<Solver<double>>();
        else
            single = std::make_unique<Solver<float>>();
    }
    template <class F>
    void Visit(F f)
    {
        if (precise)
            f(*precise, double_state);
        else if (single)
            f(*single, single_state);
    }
};
