#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "virtual_atoms/virtual_atom_graph.h"
#include "virtual_atoms/virtual_atom_math.h"

namespace
{
bool Near(float actual, float expected, float tolerance = 1.0e-5f)
{
    return std::fabs(actual - expected) <= tolerance;
}

bool Check_Vector(const char* label, VECTOR actual, VECTOR expected,
                  float tolerance = 1.0e-5f)
{
    if (Near(actual.x, expected.x, tolerance) &&
        Near(actual.y, expected.y, tolerance) &&
        Near(actual.z, expected.z, tolerance))
        return true;
    std::cerr << label << " mismatch: actual=(" << actual.x << ", " << actual.y
              << ", " << actual.z << "), expected=(" << expected.x << ", "
              << expected.y << ", " << expected.z << ")\n";
    return false;
}

Xponge::VirtualAtomRecord Record(int type, int target,
                                 std::initializer_list<int> sources,
                                 std::initializer_list<float> parameters)
{
    Xponge::VirtualAtomRecord record;
    record.type = type;
    record.virtual_atom = target;
    record.from.assign(sources);
    record.parameter.assign(parameters);
    return record;
}

bool Check_Graph()
{
    bool ok = true;
    VirtualAtomGraph graph;
    std::string error;
    const std::vector<Xponge::VirtualAtomRecord> shuffled = {
        Record(2, 5, {4, 2, 3}, {0.2f, 0.3f}),
        Record(1, 4, {0, 1}, {0.25f}),
    };
    ok &= Build_Virtual_Atom_Graph(shuffled, 6, &graph, &error);
    ok &= graph.max_level == 2 && graph.atom_levels[4] == 1 &&
          graph.atom_levels[5] == 2;
    ok &= graph.record_order == std::vector<std::size_t>({1, 0});
    if (!ok) std::cerr << "shuffled dependency graph failed: " << error << "\n";

    const auto expect_failure =
        [&](const char* label, std::vector<Xponge::VirtualAtomRecord> input,
            const char* expected)
    {
        VirtualAtomGraph invalid_graph;
        std::string invalid_error;
        if (Build_Virtual_Atom_Graph(input, 6, &invalid_graph,
                                     &invalid_error) ||
            invalid_error.find(expected) == std::string::npos)
        {
            std::cerr << label << " did not fail as expected: " << invalid_error
                      << "\n";
            return false;
        }
        return true;
    };
    ok &= expect_failure("self dependency", {Record(1, 4, {4, 0}, {0.5f})},
                         "depends on itself");
    ok &= expect_failure(
        "cycle", {Record(1, 4, {5, 0}, {0.5f}), Record(1, 5, {4, 1}, {0.5f})},
        "cycle");
    ok &= expect_failure(
        "duplicate target",
        {Record(0, 4, {0}, {1.0f}), Record(1, 4, {0, 1}, {0.5f})},
        "more than once");
    ok &= expect_failure("bad source", {Record(0, 4, {6}, {1.0f})}, "outside");
    ok &= expect_failure("bad target", {Record(0, -1, {0}, {1.0f})}, "outside");
    ok &= expect_failure("bad arity", {Record(2, 4, {0, 1}, {0.5f})}, "arity");
    ok &= expect_failure(
        "non-finite",
        {Record(0, 4, {0}, {std::numeric_limits<float>::infinity()})},
        "non-finite");
    return ok;
}

template <typename PositionFunction>
bool Check_Finite_Difference(const char* label, PositionFunction position,
                             std::vector<VECTOR> coordinates,
                             const std::vector<VECTOR>& analytic_forces,
                             VECTOR force_v, float tolerance = 3.0e-3f)
{
    constexpr float step = 1.0e-3f;
    bool ok = true;
    for (std::size_t atom = 0; atom < coordinates.size(); ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            std::vector<VECTOR> plus = coordinates;
            std::vector<VECTOR> minus = coordinates;
            (&plus[atom].x)[axis] += step;
            (&minus[atom].x)[axis] -= step;
            const float energy_plus = -(position(plus) * force_v);
            const float energy_minus = -(position(minus) * force_v);
            const float numerical_force =
                -(energy_plus - energy_minus) / (2.0f * step);
            const float analytic = (&analytic_forces[atom].x)[axis];
            if (!Near(numerical_force, analytic, tolerance))
            {
                std::cerr << label << " finite difference mismatch at atom "
                          << atom << " axis " << axis
                          << ": numerical=" << numerical_force
                          << ", analytic=" << analytic << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

bool Check_Math()
{
    bool ok = true;
    const LTMatrix3 cell(10.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f);
    const LTMatrix3 rcell(0.1f, 0.0f, 0.1f, 0.0f, 0.0f, 0.1f);
    const Boundary open = Boundary{BoundaryPolicy::Open, cell, rcell};
    const Boundary periodic = Boundary{BoundaryPolicy::Periodic, cell, rcell};
    const VECTOR force_v(1.2f, -0.7f, 0.4f);

    ok &= Check_Vector("type 0 compatibility position",
                       Virtual_Atom_Type_0_Position({1.0f, 2.0f, 3.0f}, 4.0f),
                       {1.0f, 2.0f, 5.0f});
    ok &= Check_Vector("type 0 source force",
                       Virtual_Atom_Type_0_Source_Force(force_v),
                       {force_v.x, force_v.y, -force_v.z});
    ok &= Check_Finite_Difference(
        "type 0", [](const std::vector<VECTOR>& r)
        { return Virtual_Atom_Type_0_Position(r[0], 4.0f); },
        {{1.0f, 2.0f, 3.0f}}, {Virtual_Atom_Type_0_Source_Force(force_v)},
        force_v);

    const VECTOR pbc_r1(9.5f, 0.0f, 0.0f);
    const VECTOR pbc_r2(0.5f, 0.0f, 0.0f);
    ok &= Check_Vector(
        "type 1 periodic position",
        Virtual_Atom_Type_1_Position(pbc_r1, pbc_r2, 0.25f, periodic),
        {9.75f, 0.0f, 0.0f});
    ok &=
        Check_Vector("type 1 open position",
                     Virtual_Atom_Type_1_Position(pbc_r1, pbc_r2, 0.25f, open),
                     {7.25f, 0.0f, 0.0f});
    VECTOR f1, f2, f3;
    Virtual_Atom_Type_1_Source_Forces(force_v, 0.25f, &f1, &f2);
    ok &= Check_Vector("type 1 source 1 force", f1, 0.75f * force_v);
    ok &= Check_Vector("type 1 source 2 force", f2, 0.25f * force_v);
    ok &= Check_Finite_Difference(
        "type 1", [&](const std::vector<VECTOR>& r)
        { return Virtual_Atom_Type_1_Position(r[0], r[1], 0.25f, open); },
        {{0.2f, -0.1f, 0.3f}, {1.1f, 0.7f, -0.4f}}, {f1, f2}, force_v);
    ok &= Check_Finite_Difference(
        "type 1 periodic", [&](const std::vector<VECTOR>& r)
        { return Virtual_Atom_Type_1_Position(r[0], r[1], 0.25f, periodic); },
        {{9.5f, 0.2f, 0.3f}, {0.5f, 0.7f, -0.4f}}, {f1, f2}, force_v);

    Virtual_Atom_Type_2_Source_Forces(force_v, 0.2f, 0.3f, &f1, &f2, &f3);
    ok &= Check_Vector("type 2 force sum", f1 + f2 + f3, force_v);
    ok &= Check_Finite_Difference(
        "type 2",
        [&](const std::vector<VECTOR>& r)
        {
            return Virtual_Atom_Type_2_Position(r[0], r[1], r[2], 0.2f, 0.3f,
                                                open);
        },
        {{0.2f, -0.1f, 0.3f}, {1.1f, 0.7f, -0.4f}, {-0.6f, 0.8f, 1.2f}},
        {f1, f2, f3}, force_v);
    ok &= Check_Finite_Difference(
        "type 2 periodic",
        [&](const std::vector<VECTOR>& r)
        {
            return Virtual_Atom_Type_2_Position(r[0], r[1], r[2], 0.2f, 0.3f,
                                                periodic);
        },
        {{9.5f, -0.1f, 0.3f}, {0.5f, 0.7f, -0.4f}, {8.6f, 0.8f, 1.2f}},
        {f1, f2, f3}, force_v);

    const std::vector<VECTOR> type3_coordinates = {
        {0.2f, -0.1f, 0.3f}, {1.1f, 0.7f, -0.4f}, {-0.6f, 0.8f, 1.2f}};
    VECTOR type3_position;
    ok &= Virtual_Atom_Type_3_Position(
        type3_coordinates[0], type3_coordinates[1], type3_coordinates[2], 1.4f,
        0.35f, open, &type3_position);
    ok &= Virtual_Atom_Type_3_Source_Forces(
        type3_coordinates[0], type3_coordinates[1], type3_coordinates[2],
        force_v, 1.4f, 0.35f, open, &f1, &f2, &f3);
    ok &= Check_Vector("type 3 force sum", f1 + f2 + f3, force_v, 2.0e-5f);
    ok &= Check_Finite_Difference(
        "type 3",
        [&](const std::vector<VECTOR>& r)
        {
            VECTOR result;
            Virtual_Atom_Type_3_Position(r[0], r[1], r[2], 1.4f, 0.35f, open,
                                         &result);
            return result;
        },
        type3_coordinates, {f1, f2, f3}, force_v, 5.0e-3f);

    const std::vector<VECTOR> type3_periodic_coordinates = {
        {9.5f, -0.1f, 0.3f}, {0.5f, 0.7f, -0.4f}, {1.2f, 1.5f, 1.2f}};
    ok &= Virtual_Atom_Type_3_Source_Forces(
        type3_periodic_coordinates[0], type3_periodic_coordinates[1],
        type3_periodic_coordinates[2], force_v, 1.4f, 0.35f, periodic, &f1, &f2,
        &f3);
    ok &= Check_Finite_Difference(
        "type 3 periodic",
        [&](const std::vector<VECTOR>& r)
        {
            VECTOR result;
            Virtual_Atom_Type_3_Position(r[0], r[1], r[2], 1.4f, 0.35f,
                                         periodic, &result);
            return result;
        },
        type3_periodic_coordinates, {f1, f2, f3}, force_v, 5.0e-3f);

    VECTOR zero_position;
    ok &= Virtual_Atom_Type_3_Position({1, 2, 3}, {1, 2, 3}, {1, 2, 3}, 0.0f,
                                       0.5f, open, &zero_position);
    ok &= Check_Vector("type 3 zero-d position", zero_position, {1, 2, 3});
    ok &= Virtual_Atom_Type_3_Source_Forces({1, 2, 3}, {1, 2, 3}, {1, 2, 3},
                                            force_v, 0.0f, 0.5f, open, &f1, &f2,
                                            &f3);
    ok &= Check_Vector("type 3 zero-d source 1", f1, force_v);
    ok &= Check_Vector("type 3 zero-d source 2", f2, VECTOR(0.0f));
    ok &= Check_Vector("type 3 zero-d source 3", f3, VECTOR(0.0f));
    ok &= !Virtual_Atom_Type_3_Position({0, 0, 0}, {1, 0, 0}, {2, 0, 0}, 1.0f,
                                        -1.0f, open, &zero_position);
    return ok;
}

static __global__ void Device_Math_Smoke(VECTOR* output, int* status)
{
    const LTMatrix3 cell(10.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f);
    const LTMatrix3 rcell(0.1f, 0.0f, 0.1f, 0.0f, 0.0f, 0.1f);
    const auto boundary = Boundary{BoundaryPolicy::Periodic, cell, rcell};
    output[0] = Virtual_Atom_Type_1_Position({9.5f, 0, 0}, {0.5f, 0, 0}, 0.25f,
                                             boundary);
    output[1] = Virtual_Atom_Type_2_Position(
        {9.5f, 0, 0}, {0.5f, 0, 0}, {9.5f, 2, 0}, 0.25f, 0.5f, boundary);
    status[0] = Virtual_Atom_Type_3_Position({0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                             2.0f, 0.5f, boundary, &output[2])
                    ? 1
                    : 0;
}

static __global__ void Device_Shared_Source_Accumulation(int count,
                                                         VECTOR* output)
{
#ifdef USE_GPU
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count)
#else
#pragma omp parallel for
    for (int i = 0; i < count; ++i)
#endif
    {
        VECTOR force_1, force_2, force_3;
        Virtual_Atom_Type_2_Source_Forces({1.0f, -2.0f, 3.0f}, 0.25f, 0.25f,
                                          &force_1, &force_2, &force_3);
        Virtual_Atom_Add_Source_Force(&output[0], force_1);
        Virtual_Atom_Add_Source_Force(&output[1], force_2);
        Virtual_Atom_Add_Source_Force(&output[2], force_3);
    }
}

bool Check_Device_Smoke()
{
    VECTOR output[3];
    int status = 0;
#if defined(USE_CUDA) || defined(USE_HIP)
    VECTOR* d_output = nullptr;
    int* d_status = nullptr;
    if (deviceMalloc((void**)&d_output, sizeof(output)) !=
            DEVICE_MALLOC_SUCCESS ||
        deviceMalloc((void**)&d_status, sizeof(status)) !=
            DEVICE_MALLOC_SUCCESS)
    {
        std::cerr << "device allocation failed\n";
        return false;
    }
    Launch_Device_Kernel(Device_Math_Smoke, 1, 1, 0, nullptr, d_output,
                         d_status);
    if (hostDeviceSynchronize() != DEVICE_MALLOC_SUCCESS)
    {
        std::cerr << "device kernel synchronization failed\n";
        return false;
    }
    deviceMemcpy(output, d_output, sizeof(output), deviceMemcpyDeviceToHost);
    deviceMemcpy(&status, d_status, sizeof(status), deviceMemcpyDeviceToHost);
    deviceFree(d_output);
    deviceFree(d_status);
#else
    Device_Math_Smoke(output, &status);
#endif
    bool ok = status == 1;
    ok &= Check_Vector("device type 1", output[0], {9.75f, 0, 0});
    ok &= Check_Vector("device type 2", output[1], {9.75f, 1, 0});
    ok &= Check_Vector("device type 3", output[2],
                       {1.7888544f, 0.8944272f, 0.0f}, 2.0e-5f);
    return ok;
}

bool Check_Shared_Source_Accumulation()
{
    constexpr int count = 4096;
    VECTOR output[3] = {VECTOR(0.0f), VECTOR(0.0f), VECTOR(0.0f)};
#if defined(USE_CUDA) || defined(USE_HIP)
    VECTOR* d_output = nullptr;
    if (deviceMalloc((void**)&d_output, sizeof(output)) !=
        DEVICE_MALLOC_SUCCESS)
    {
        std::cerr << "device allocation failed\n";
        return false;
    }
    deviceMemset(d_output, 0, sizeof(output));
    Launch_Device_Kernel(Device_Shared_Source_Accumulation, (count + 255) / 256,
                         256, 0, nullptr, count, d_output);
    if (hostDeviceSynchronize() != DEVICE_MALLOC_SUCCESS)
    {
        std::cerr << "shared-source kernel synchronization failed\n";
        deviceFree(d_output);
        return false;
    }
    deviceMemcpy(output, d_output, sizeof(output), deviceMemcpyDeviceToHost);
    deviceFree(d_output);
#else
    Device_Shared_Source_Accumulation(count, output);
#endif
    const VECTOR total_force(static_cast<float>(count),
                             -2.0f * static_cast<float>(count),
                             3.0f * static_cast<float>(count));
    bool ok = Check_Vector("shared source 1", output[0], 0.5f * total_force);
    ok &= Check_Vector("shared source 2", output[1], 0.25f * total_force);
    ok &= Check_Vector("shared source 3", output[2], 0.25f * total_force);
    return ok;
}
}  // namespace

int main()
{
    const bool ok = Check_Graph() && Check_Math() && Check_Device_Smoke() &&
                    Check_Shared_Source_Accumulation();
    return ok ? 0 : 1;
}
