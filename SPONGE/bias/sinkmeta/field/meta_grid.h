#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../../../common.h"

struct MetaGrid
{
    int ndim = 0;
    int total_size = 0;
    std::vector<int> num_points;
    std::vector<float> lower, upper, spacing, inv_spacing;
    std::vector<bool> is_periodic;

    std::vector<float> potential;
    std::vector<float> force;
    std::vector<float> normal_lse;
    std::vector<float> normal_force;

    float* d_potential = nullptr;
    float* d_force = nullptr;
    float* d_normal_lse = nullptr;
    float* d_normal_force = nullptr;

    void Initial(const std::vector<int>& npts, const std::vector<float>& lo,
                 const std::vector<float>& up, const std::vector<bool>& periodic)
    {
        ndim = static_cast<int>(npts.size());
        num_points = npts;
        is_periodic = periodic;
        lower.resize(ndim);
        upper.resize(ndim);
        spacing.resize(ndim);
        inv_spacing.resize(ndim);
        for (int d = 0; d < ndim; ++d)
        {
            spacing[d] = (up[d] - lo[d]) / npts[d];
            inv_spacing[d] = 1.0f / spacing[d];
            if (!periodic[d])
            {
                num_points[d] += 1;
                lower[d] = lo[d] - spacing[d] * 0.5f;
                upper[d] = up[d] + spacing[d] * 0.5f;
            }
            else
            {
                lower[d] = lo[d];
                upper[d] = up[d];
            }
        }
        total_size = 1;
        for (int d = 0; d < ndim; ++d)
        {
            total_size *= num_points[d];
        }
    }

    void Alloc_Device()
    {
        if (!potential.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_potential,
                                          potential.data(),
                                          sizeof(float) * potential.size());
        if (!force.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_force,
                                          force.data(),
                                          sizeof(float) * force.size());
        if (!normal_lse.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_normal_lse,
                                          normal_lse.data(),
                                          sizeof(float) * normal_lse.size());
        if (!normal_force.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_normal_force,
                                          normal_force.data(),
                                          sizeof(float) * normal_force.size());
    }

    void Sync_To_Device()
    {
        if (d_potential && !potential.empty())
            deviceMemcpy(d_potential, potential.data(),
                         sizeof(float) * potential.size(),
                         deviceMemcpyHostToDevice);
        if (d_force && !force.empty())
            deviceMemcpy(d_force, force.data(),
                         sizeof(float) * force.size(),
                         deviceMemcpyHostToDevice);
        if (d_normal_lse && !normal_lse.empty())
            deviceMemcpy(d_normal_lse, normal_lse.data(),
                         sizeof(float) * normal_lse.size(),
                         deviceMemcpyHostToDevice);
        if (d_normal_force && !normal_force.empty())
            deviceMemcpy(d_normal_force, normal_force.data(),
                         sizeof(float) * normal_force.size(),
                         deviceMemcpyHostToDevice);
    }

    void Sync_To_Host()
    {
        if (d_potential && !potential.empty())
            deviceMemcpy(potential.data(), d_potential,
                         sizeof(float) * potential.size(),
                         deviceMemcpyDeviceToHost);
        if (d_force && !force.empty())
            deviceMemcpy(force.data(), d_force,
                         sizeof(float) * force.size(),
                         deviceMemcpyDeviceToHost);
        if (d_normal_lse && !normal_lse.empty())
            deviceMemcpy(normal_lse.data(), d_normal_lse,
                         sizeof(float) * normal_lse.size(),
                         deviceMemcpyDeviceToHost);
        if (d_normal_force && !normal_force.empty())
            deviceMemcpy(normal_force.data(), d_normal_force,
                         sizeof(float) * normal_force.size(),
                         deviceMemcpyDeviceToHost);
    }

    int Get_Flat_Index(const std::vector<float>& values) const
    {
        int idx = 0;
        int fac = 1;
        for (int d = 0; d < ndim; ++d)
        {
            int i = static_cast<int>(
                std::floor((values[d] - lower[d]) * inv_spacing[d]));
            if (is_periodic[d])
            {
                i = ((i % num_points[d]) + num_points[d]) % num_points[d];
            }
            else
            {
                i = std::max(0, std::min(i, num_points[d] - 1));
            }
            idx += i * fac;
            fac *= num_points[d];
        }
        return idx;
    }

    std::vector<float> Get_Coordinates(int flat_index) const
    {
        std::vector<float> coords(ndim);
        for (int d = 0; d < ndim; ++d)
        {
            int i = flat_index % num_points[d];
            flat_index /= num_points[d];
            coords[d] = lower[d] + (i + 0.5f) * spacing[d];
        }
        return coords;
    }

    int size() const { return total_size; }
    int Get_Dimension() const { return ndim; }
};

struct MetaScatter
{
    int ndim = 0;
    int num_points = 0;
    std::vector<std::vector<float>> coordinates;
    std::vector<float> periods;

    std::vector<float> potential;
    std::vector<float> force;
    std::vector<float> rotate_v;
    std::vector<float> rotate_matrix;

    float* d_potential = nullptr;
    float* d_force = nullptr;

    void Initial(const std::vector<int>& npts, const std::vector<float>& period,
                 const std::vector<std::vector<float>>& coor)
    {
        ndim = static_cast<int>(npts.size());
        num_points = static_cast<int>(coor.size());
        coordinates = coor;
        periods = period;
    }

    void Alloc_Device()
    {
        if (!potential.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_potential,
                                          potential.data(),
                                          sizeof(float) * potential.size());
        if (!force.empty())
            Device_Malloc_And_Copy_Safely((void**)&d_force,
                                          force.data(),
                                          sizeof(float) * force.size());
    }

    void Sync_To_Device()
    {
        if (d_potential && !potential.empty())
            deviceMemcpy(d_potential, potential.data(),
                         sizeof(float) * potential.size(),
                         deviceMemcpyHostToDevice);
        if (d_force && !force.empty())
            deviceMemcpy(d_force, force.data(),
                         sizeof(float) * force.size(),
                         deviceMemcpyHostToDevice);
    }

    void Sync_To_Host()
    {
        if (d_potential && !potential.empty())
            deviceMemcpy(potential.data(), d_potential,
                         sizeof(float) * potential.size(),
                         deviceMemcpyDeviceToHost);
        if (d_force && !force.empty())
            deviceMemcpy(force.data(), d_force,
                         sizeof(float) * force.size(),
                         deviceMemcpyDeviceToHost);
    }

    int Get_Index(const std::vector<float>& values) const
    {
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = 0;
        for (int i = 0; i < num_points; ++i)
        {
            float dist = 0;
            for (int d = 0; d < ndim; ++d)
            {
                float diff = values[d] - coordinates[i][d];
                if (periods[d] > 0)
                {
                    diff -= std::round(diff / periods[d]) * periods[d];
                }
                dist += diff * diff;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                min_idx = i;
            }
        }
        return min_idx;
    }

    std::vector<int> Get_Neighbor(const std::vector<float>& values,
                                  const float* cutoff) const
    {
        std::vector<int> neighbors;
        for (int i = 0; i < num_points; ++i)
        {
            bool within = true;
            for (int d = 0; d < ndim; ++d)
            {
                float diff = values[d] - coordinates[i][d];
                if (periods[d] > 0)
                {
                    diff -= std::round(diff / periods[d]) * periods[d];
                }
                if (std::fabs(diff) > cutoff[d])
                {
                    within = false;
                    break;
                }
            }
            if (within)
            {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    const std::vector<float>& Get_Coordinate(int index) const
    {
        return coordinates[index];
    }

    int size() const { return num_points; }
    int Get_Dimension() const { return ndim; }
};
