#include "sinkmeta.h"

static float Evaluate_Gaussian_Switch(const float rij, const float center,
                                      const float inv_w, const float period,
                                      float& df)
{
    float distant = rij - center;
    if (period != 0.0f)
    {
        distant -= roundf(distant / period) * period;
    }
    const float dx = distant * inv_w;
    const float f = expf(-dx * dx / 2.0f);
    df = -dx * inv_w * f;
    return f;
}

void MetaGrid::Initial(const std::vector<int>& npts,
                       const std::vector<float>& lo,
                       const std::vector<float>& up,
                       const std::vector<bool>& periodic)
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
        lower[d] = lo[d];
        upper[d] = up[d];
    }
    total_size = 1;
    for (int d = 0; d < ndim; ++d)
    {
        total_size *= num_points[d];
    }
}

void MetaGrid::Alloc_Device()
{
    if (!potential.empty() && d_potential == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_potential, potential.data(),
                                      sizeof(float) * potential.size());
    else if (!potential.empty())
        deviceMemcpy(d_potential, potential.data(),
                     sizeof(float) * potential.size(),
                     deviceMemcpyHostToDevice);
    if (!force.empty() && d_force == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_force, force.data(),
                                      sizeof(float) * force.size());
    else if (!force.empty())
        deviceMemcpy(d_force, force.data(), sizeof(float) * force.size(),
                     deviceMemcpyHostToDevice);
    if (!normal_lse.empty() && d_normal_lse == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_normal_lse, normal_lse.data(),
                                      sizeof(float) * normal_lse.size());
    else if (!normal_lse.empty())
        deviceMemcpy(d_normal_lse, normal_lse.data(),
                     sizeof(float) * normal_lse.size(),
                     deviceMemcpyHostToDevice);
    if (!normal_force.empty() && d_normal_force == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_normal_force,
                                      normal_force.data(),
                                      sizeof(float) * normal_force.size());
    else if (!normal_force.empty())
        deviceMemcpy(d_normal_force, normal_force.data(),
                     sizeof(float) * normal_force.size(),
                     deviceMemcpyHostToDevice);
    if (ndim > 0)
    {
        if (d_num_points == NULL)
            Device_Malloc_And_Copy_Safely(
                (void**)&d_num_points, num_points.data(), sizeof(int) * ndim);
        if (d_lower == NULL)
            Device_Malloc_And_Copy_Safely((void**)&d_lower, lower.data(),
                                          sizeof(float) * ndim);
        if (d_spacing == NULL)
            Device_Malloc_And_Copy_Safely((void**)&d_spacing, spacing.data(),
                                          sizeof(float) * ndim);
    }
}

void MetaGrid::Sync_To_Device()
{
    if (d_potential && !potential.empty())
        deviceMemcpy(d_potential, potential.data(),
                     sizeof(float) * potential.size(),
                     deviceMemcpyHostToDevice);
    if (d_force && !force.empty())
        deviceMemcpy(d_force, force.data(), sizeof(float) * force.size(),
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

void MetaGrid::Sync_To_Host()
{
    if (d_potential && !potential.empty())
        deviceMemcpy(potential.data(), d_potential,
                     sizeof(float) * potential.size(),
                     deviceMemcpyDeviceToHost);
    if (d_force && !force.empty())
        deviceMemcpy(force.data(), d_force, sizeof(float) * force.size(),
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

int MetaGrid::Get_Flat_Index(const std::vector<float>& values) const
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

std::vector<float> MetaGrid::Get_Coordinates(int flat_index) const
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

void MetaScatter::Initial(const std::vector<int>& npts,
                          const std::vector<float>& period,
                          const std::vector<std::vector<float>>& coor)
{
    ndim = static_cast<int>(npts.size());
    num_points = static_cast<int>(coor.size());
    coordinates = coor;
    periods = period;
    coordinates_flat.resize(num_points * ndim);
    for (int index = 0; index < num_points; ++index)
    {
        for (int d = 0; d < ndim; ++d)
        {
            coordinates_flat[index * ndim + d] = coordinates[index][d];
        }
    }
}

void MetaScatter::Alloc_Device()
{
    if (!coordinates_flat.empty() && d_coordinates == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_coordinates,
                                      coordinates_flat.data(),
                                      sizeof(float) * coordinates_flat.size());
    else if (!coordinates_flat.empty())
        deviceMemcpy(d_coordinates, coordinates_flat.data(),
                     sizeof(float) * coordinates_flat.size(),
                     deviceMemcpyHostToDevice);
    if (!periods.empty() && d_periods == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_periods, periods.data(),
                                      sizeof(float) * periods.size());
    else if (!periods.empty())
        deviceMemcpy(d_periods, periods.data(), sizeof(float) * periods.size(),
                     deviceMemcpyHostToDevice);
    if (!potential.empty() && d_potential == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_potential, potential.data(),
                                      sizeof(float) * potential.size());
    else if (!potential.empty())
        deviceMemcpy(d_potential, potential.data(),
                     sizeof(float) * potential.size(),
                     deviceMemcpyHostToDevice);
    if (!force.empty() && d_force == NULL)
        Device_Malloc_And_Copy_Safely((void**)&d_force, force.data(),
                                      sizeof(float) * force.size());
    else if (!force.empty())
        deviceMemcpy(d_force, force.data(), sizeof(float) * force.size(),
                     deviceMemcpyHostToDevice);
}

void MetaScatter::Sync_To_Device()
{
    if (d_potential && !potential.empty())
        deviceMemcpy(d_potential, potential.data(),
                     sizeof(float) * potential.size(),
                     deviceMemcpyHostToDevice);
    if (d_force && !force.empty())
        deviceMemcpy(d_force, force.data(), sizeof(float) * force.size(),
                     deviceMemcpyHostToDevice);
}

void MetaScatter::Sync_To_Host()
{
    if (d_potential && !potential.empty())
        deviceMemcpy(potential.data(), d_potential,
                     sizeof(float) * potential.size(),
                     deviceMemcpyDeviceToHost);
    if (d_force && !force.empty())
        deviceMemcpy(force.data(), d_force, sizeof(float) * force.size(),
                     deviceMemcpyDeviceToHost);
}

int MetaScatter::Get_Index(const std::vector<float>& values) const
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

std::vector<int> MetaScatter::Get_Neighbor(const std::vector<float>& values,
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

const std::vector<float>& MetaScatter::Get_Coordinate(int index) const
{
    return coordinates[index];
}

static std::vector<float> normalize(const std::vector<float>& v)
{
    float norm = 0.;
    for (auto vi : v)
    {
        norm += vi * vi;
    }
    if (norm == 0.0)
    {
        throw std::runtime_error("Zero-length vector cannot be normalized.");
    }
    std::vector<float> new_v;
    for (int i = 0; i < v.size(); ++i)
    {
        new_v.push_back(v[i] / sqrt(norm));
    }
    return new_v;
}

static std::vector<float> cross_product(const std::vector<float>& a,
                                        const std::vector<float>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

static float determinant(const std::vector<std::vector<float>>& matrix)
{
    int n = matrix.size();
    if (n == 1)
    {
        return matrix[0][0];
    }

    float det = 0;
    for (int i = 0; i < n; ++i)
    {
        std::vector<std::vector<float>> submatrix(n - 1,
                                                  std::vector<float>(n - 1));

        for (int j = 1; j < n; ++j)
        {
            int subCol = 0;
            for (int k = 0; k < n; ++k)
            {
                if (k == i) continue;
                submatrix[j - 1][subCol] = matrix[j][k];
                subCol++;
            }
        }

        float subDet = determinant(submatrix);
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * subDet;
    }

    return det;
}

META::Axis META::Rotate_Vector(const Axis& tang_vector)
{
    std::vector<float> normal_vector;
    int reference_axis = 0;
    if (fabs(tang_vector[reference_axis]) > 0.99)
    {
        ++reference_axis;
    }
    for (int i = 0; i < ndim; ++i)
    {
        if (i == reference_axis)
        {
            normal_vector.push_back(1.);
        }
        else
        {
            normal_vector.push_back(0.);
        }
    }
    Axis jb;
    float i_min = tang_vector[reference_axis];
    float e1 = sqrtf(1 - i_min * i_min);
    float e2 = -i_min / e1;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == reference_axis)
        {
            jb.push_back(e1);
        }
        else
        {
            jb.push_back(tang_vector[i] * e2);
        }
    }
    if (ndim == 2)
    {
        std::vector<std::vector<float>> determinant_v =
            std::vector<Axis>{tang_vector, jb};
        float sign = determinant(determinant_v);
        return Axis{jb[0] * sign, jb[1] * sign};
    }
    return jb;
}

void META::Cartesian_To_Path(const Axis& Cartesian_values, Axis& Path_values)
{
    double cumulative_s = 0.0;
    Axis tang_vector(ndim, 0.);
    int index = mscatter->Get_Index(Cartesian_values);
    const Axis& values = (index < scatter_size - 1)
                             ? mscatter->Get_Coordinate(index)
                             : mscatter->Get_Coordinate(index - 1);
    const Axis& neighbor = (index < scatter_size - 1)
                               ? mscatter->Get_Coordinate(index + 1)
                               : mscatter->Get_Coordinate(index);
    Tang_Vector(tang_vector, values, neighbor);
    double projected_last =
        Project_To_Path(tang_vector, neighbor, Cartesian_values);
    Path_values.push_back(cumulative_s + projected_last);
    Axis normal_vector = Rotate_Vector(tang_vector);
    Path_values.push_back(
        Project_To_Path(normal_vector, values, Cartesian_values));
    if (ndim == 3)
    {
        Axis binormal_vector =
            normalize(cross_product(tang_vector, normal_vector));
        Path_values.push_back(
            Project_To_Path(binormal_vector, values, Cartesian_values));
    }
}

float META::Project_To_Path(const Gdata& tang_vector, const Axis& values,
                            const Axis& Cartesian)
{
    float projected_s = 0.;
    for (int i = 0; i < ndim; ++i)
    {
        projected_s += (Cartesian[i] - values[i]) * tang_vector[i];
    }
    return projected_s;
}

double META::Tang_Vector(Gdata& tang_vector, const Axis& values,
                         const Axis& neighbor)
{
    double square = 0;
    for (int i = 0; i < ndim; ++i)
    {
        double distance = neighbor[i] - values[i];
        tang_vector[i] = distance;
        square += distance * distance;
    }
    double segment_s = sqrt(square);
    for (int i = 0; i < ndim; ++i)
    {
        tang_vector[i] /= segment_s;
    }
    return segment_s;
}

void META::Set_Grid(CONTROLLER* controller)  //
{
    std::vector<int> ngrid;
    std::vector<float> lower, upper, periodic;
    std::vector<bool> isperiodic;
    border_upper.resize(ndim);
    border_lower.resize(ndim);
    est_values_.resize(ndim);
    est_sum_force_.resize(ndim);
    Device_Malloc_Safely((void**)&d_hill_centers, sizeof(float) * ndim);
    Device_Malloc_Safely((void**)&d_hill_inv_w, sizeof(float) * ndim);
    Device_Malloc_Safely((void**)&d_hill_periods, sizeof(float) * ndim);
    Device_Malloc_And_Copy_Safely((void**)&d_cutoff, cutoff,
                                  sizeof(float) * ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
        ngrid.push_back(n_grids[i]);
        lower.push_back(cv_mins[i]);
        upper.push_back(cv_maxs[i]);
        periodic.push_back(cv_periods[i]);
        isperiodic.push_back(cv_periods[i] > 0 ? true : false);
    }
    mgrid = new MetaGrid();
    mgrid->Initial(ngrid, lower, upper, isperiodic);
    mgrid->normal_force.assign(mgrid->total_size * ndim, 0.0f);
    mgrid->normal_lse.assign(mgrid->total_size, 0.0f);
    mgrid->potential.assign(mgrid->total_size, 0.0f);
    if (usegrid)
    {
        mgrid->force.assign(mgrid->total_size * ndim, 0.0f);
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        mgrid->normal_lse.assign(mgrid->total_size, log(normalization));
        mscatter = NULL;
        Sum_Hills(history_freq);
        mgrid->Alloc_Device();
    }
    else if (use_scatter)
    {
        if (mask > 0)
        {
            mgrid->force.assign(mgrid->total_size * ndim, 0.0f);
        }
        std::vector<int> nscatter;
        int oldsize = 1;
        for (size_t i = 0; i < ndim; ++i)
        {
            nscatter.push_back(n_grids[i]);
            oldsize *= n_grids[i];
        }
        max_index = floor(scatter_size / 2);
        if (oldsize < scatter_size)
        {
            mscatter = NULL;
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                           "Meta::SetGrid()\n");
            return;
        }
        std::vector<std::vector<float>> coor;
        for (size_t j = 0; j < scatter_size; ++j)
        {
            std::vector<float> p;
            for (size_t i = 0; i < ndim; ++i)
            {
                p.push_back(tcoor[i][j]);
            }
            coor.push_back(p);
        }
        mscatter = new MetaScatter();
        mscatter->Initial(nscatter, periodic, coor);
        mscatter->force.assign(scatter_size * ndim, 0.0f);
        mscatter->potential.assign(scatter_size, 0.0f);
        Edge_Effect(1, scatter_size);
        Sum_Hills(history_freq);
        mgrid->Alloc_Device();
        mscatter->Alloc_Device();
    }
    else
    {
        controller->printf("Warning! No grid version is very slow\n");
        mscatter = NULL;
    }
    if (mgrid != NULL)
    {
    }
}
void META::Estimate(const Axis& values, const bool need_potential,
                    const bool need_force)
{
    potential_local = 0;
    potential_backup = 0;

    float shift = potential_max + dip * CONSTANT_kB * temperature;
    if (do_negative)
    {
        if (grw)
        {
            shift = (welltemp_factor + dip) * CONSTANT_kB * temperature;
        }
        new_max = Normalization(values, shift, true);
    }
    float force_max = 0.0;
    float normalforce_sum = 0.0;
    for (size_t i = 0; i < ndim; ++i)
    {
        est_sum_force_[i] = 0.0f;
    }
    int nf_idx = mgrid->Get_Flat_Index(values);
    for (size_t i = 0; i < ndim; ++i)
    {
        Dpotential_local[i] = 0.0;
        force_max += fabs(mgrid->normal_force[nf_idx * ndim + i]);
    }
    if (force_max > max_force && need_force && mask)
    {
        exit_tag += 1.0;
    }
    if (use_scatter)
    {
        if (subhill)
        {
            Hill hill = Hill(values, sigmas, periods, 1.0);
            std::vector<int> indices;
            if (do_cutoff)
            {
                indices = mscatter->Get_Neighbor(values, cutoff);
            }
            else
            {
                indices = std::vector<int>(scatter_size);
                std::iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                const Axis& neighbor = mscatter->Get_Coordinate(index);
                const Gdata& tder = hill.Calc_Hill(neighbor);
                normalforce_sum += hill.potential;
                float factor =
                    (mask > 0)
                        ? mgrid->potential[mgrid->Get_Flat_Index(neighbor)]
                        : mscatter->potential[index];
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        est_sum_force_[i] += tder[i];
                        Dpotential_local[i] -= (factor)*tder[i];
                    }
                }
                potential_backup += factor * hill.potential;
            }
        }
        else
        {
            int sidx = mscatter->Get_Index(values);
            potential_backup =
                (mask > 0) ? mgrid->potential[mgrid->Get_Flat_Index(values)]
                           : mscatter->potential[sidx];
            potential_local = potential_backup - Calc_V_Shift(values);
            if (need_force)
            {
                int fidx = (mask > 0) ? mgrid->Get_Flat_Index(values) : sidx;
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        (mask > 0) ? mgrid->force[fidx * ndim + i]
                                   : mscatter->force[fidx * ndim + i];
                }
            }
        }
    }
    else if (usegrid)
    {
        if (subhill)
        {
            Hill hill = Hill(values, sigmas, periods, 1.0);
            Axis vminus(ndim), vplus(ndim);
            for (size_t i = 0; i < ndim; ++i)
            {
                float lower = values[i] - cutoff[i];
                float upper = values[i] + cutoff[i] + 0.000001;
                if (periods[i] > 0)
                {
                    vminus[i] = lower;
                    vplus[i] = upper;
                }
                else
                {
                    vminus[i] = std::fmax(lower, cv_mins[i]);
                    vplus[i] = std::fmin(upper, cv_maxs[i]);
                }
            }
            Axis loop_flag = vminus;
            int index = 0;
            while (index >= 0)
            {
                const Gdata& tder = hill.Calc_Hill(loop_flag);
                float factor =
                    mgrid->potential[mgrid->Get_Flat_Index(loop_flag)];
                potential_backup += factor * hill.potential;
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        Dpotential_local[i] -= (factor - new_max) * tder[i];
                    }
                }
                index = ndim - 1;
                while (index >= 0)
                {
                    loop_flag[index] += cv_deltas[index];
                    if (loop_flag[index] > vplus[index])
                    {
                        loop_flag[index] = vminus[index];
                        --index;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            int gidx = mgrid->Get_Flat_Index(values);
            potential_backup = mgrid->potential[gidx];
            if (need_force)
            {
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] += mgrid->force[gidx * ndim + i];
                }
            }
        }
        if (do_borderwall)
        {
            for (size_t i = 0; i < ndim; ++i)
            {
                border_upper[i] = cv_maxs[i] - cutoff[i];
                border_lower[i] = cv_mins[i] + cutoff[i];
            }
        }
    }
    if (need_potential)
    {
        potential_local = potential_backup - Calc_V_Shift(values);
    }
    if (need_force)
    {
        if (subhill)
        {
            float f0 = new_max * mgrid->normal_force[nf_idx * ndim + 0];
            if (convmeta)
            {
                new_max = shift * expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                                      mscatter->Get_Coordinate(max_index))]);
            }
            else
            {
                new_max = shift / normalforce_sum;
            }
            float f1 = new_max * est_sum_force_[0];
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * est_sum_force_[i];
            }
        }
        else
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] +=
                    new_max * mgrid->normal_force[nf_idx * ndim + i];
            }
        }
    }
    return;
}

static void Write_CV_Header(FILE* temp_file, int ndim, const CV_LIST& cvs)
{
    for (int i = 0; i < ndim; ++i)
    {
        const char* cv_name = NULL;
        if (i < static_cast<int>(cvs.size()) && cvs[i] != NULL &&
            cvs[i]->module_name[0] != '\0')
        {
            cv_name = cvs[i]->module_name;
        }
        if (cv_name != NULL)
        {
            fprintf(temp_file, "%s\t", cv_name);
        }
        else
        {
            fprintf(temp_file, "cv%d\t", i + 1);
        }
    }
}
void META::Write_Potential(void)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_potential_file_name, "w");
        if (subhill || (!usegrid && !use_scatter))
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_local\tpotential_backup");
            if (!kde)
            {
                fprintf(temp_file, "\tpotential_raw");
            }
            fprintf(temp_file, "\n");
            std::vector<float> loop_flag(ndim, 0);
            std::vector<float> loop_floor(ndim, 0);
            for (int i = 0; i < ndim; ++i)
            {
                loop_floor[i] = cv_mins[i] + 0.5 * cv_deltas[i];
                loop_flag[i] = loop_floor[i];
            }
            int i = 0;
            while (i >= 0)
            {
                Estimate(loop_flag, true, false);  // get potential
                std::ostringstream ss;
                for (const float& v : loop_flag)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f", ss.str().c_str(),
                        potential_local, potential_backup);
                if (!kde)
                {
                    if (mgrid != NULL)
                    {
                        fprintf(
                            temp_file, "\t%f",
                            mgrid->potential[mgrid->Get_Flat_Index(loop_flag)]);
                    }
                    else if (mscatter != NULL)
                    {
                        fprintf(
                            temp_file, "\t%f",
                            mscatter
                                ->potential[mscatter->Get_Index(loop_flag)]);
                    }
                }
                fprintf(temp_file, "\n");
                //  iterate over any dimensions
                i = ndim - 1;
                while (i >= 0)
                {
                    loop_flag[i] += cv_deltas[i];
                    if (loop_flag[i] > cv_maxs[i])
                    {
                        loop_flag[i] = loop_floor[i];
                        --i;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else if (mgrid != NULL)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\tvshift\n");
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                std::ostringstream ss;
                const Axis coor = mgrid->Get_Coordinates(idx);
                float vshift = Calc_V_Shift(coor);
                for (const float& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                        mgrid->potential[idx], mgrid->potential[idx] - vshift,
                        vshift);
            }
        }
        // In case of pure scattering point!
        else if (mscatter != NULL)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\n");
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                std::ostringstream ss;
                const Axis& coor = mscatter->Get_Coordinate(iter);
                float vshift = Calc_V_Shift(coor);
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\n", ss.str().c_str(),
                        mscatter->potential[iter],
                        mscatter->potential[iter] - vshift);
            }
        }
        fclose(temp_file);
    }
}
void META::Write_Directly(void)
{
    if (!is_initialized || !(use_scatter || usegrid))
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_directly_file_name, "w");
        std::string meta_type;
        if (do_negative)
        {
            std::string pm = std::to_string(potential_max);
            meta_type += "sink(kcal): " + pm;
        }
        if (mask)
        {
            meta_type += " mask ";
        }
        if (subhill)
        {
            meta_type += " subhill ";
        }
        else
        {
            meta_type += " d_force";
        }

        fprintf(temp_file, "%dD-Meta X %s\n", ndim, meta_type.c_str());
        for (int i = 0; i < ndim; ++i)
        {
            fprintf(temp_file, "%f\t%f\t%f\n", cv_mins[i], cv_maxs[i],
                    cv_deltas[i]);
        }
        int gridsize = 1;
        for (int i = 0; i < ndim; ++i)
        {
            int num_grid = round((cv_maxs[i] - cv_mins[i]) / cv_deltas[i]);
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        if (mscatter != NULL)
        {
            fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                std::ostringstream ss;
                const Axis& coor = mscatter->Get_Coordinate(iter);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                if (subhill)
                {
                    fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                            potential_local, potential_backup,
                            mscatter->potential[iter]);
                }
                else
                {
                    float result;
                    result = potential_local;
                    fprintf(temp_file, "%s%f\t", ss.str().c_str(), result);
                    float* data = &mscatter->force[iter * ndim];
                    for (int i = 0; i < ndim; ++i)
                    {
                        fprintf(temp_file, "%f\t", data[i]);
                    }
                    fprintf(temp_file, "%f\n", mscatter->potential[iter]);
                }
            }
        }
        else if (mgrid != NULL)
        {
            fprintf(temp_file, "%d\n", mgrid->total_size);
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                std::ostringstream ss;
                std::vector<float> coor = mgrid->Get_Coordinates(idx);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t", ss.str().c_str(), potential_local);
                float* data = &mgrid->force[idx * ndim];
                for (int i = 0; i < ndim; ++i)
                {
                    fprintf(temp_file, "%f\t", data[i]);
                }
                fprintf(temp_file, "%f\n", mgrid->potential[idx]);
            }
        }
        fclose(temp_file);
    }
}
void META::Read_Potential(CONTROLLER* controller)
{
    FILE* temp_file = NULL;
    Open_File_Safely(&temp_file, read_potential_file_name, "r");
    char temp_char[256];
    int scanf_ret = 0;
    char* get_val = fgets(temp_char, 256, temp_file);  // title line
    Malloc_Safely((void**)&cv_mins, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_maxs, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_deltas, sizeof(float) * ndim);
    Malloc_Safely((void**)&n_grids, sizeof(float) * ndim);
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%f %f %f\n", &cv_mins[i], &cv_maxs[i],
                           &cv_deltas[i]);
        if (scanf_ret != 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
        controller->printf(
            "    CV_minimal = %f\n    CV_maximum = %f\n    dCV = %f\n",
            cv_mins[i], cv_maxs[i], cv_deltas[i]);
    }
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%d", &n_grids[i]);
        if (scanf_ret != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
    }
    scanf_ret = fscanf(temp_file, "%d\n", &scatter_size);
    // Scatter points coordinate
    for (int i = 0; i < ndim; ++i)
    {
        float* ttoorr;
        Malloc_Safely((void**)&ttoorr, sizeof(float) * scatter_size);
        tcoor.push_back(ttoorr);
    }
    std::vector<float> potential_from_file;
    std::vector<Gdata> force_from_file;
    for (int j = 0; j < scatter_size; ++j)
    {
        char* grid_val = NULL;
        do
        {
            grid_val = fgets(temp_char, 256, temp_file);
        } while (grid_val != NULL && std::string(temp_char).find_first_not_of(
                                         " \t\r\n") == std::string::npos);
        if (grid_val == NULL)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
        std::string raw_line(temp_char);
        size_t first = raw_line.find_first_not_of(" \t\r\n");
        size_t last = raw_line.find_last_not_of(" \t\r\n");
        std::vector<std::string> words;
        if (first != std::string::npos)
        {
            words = string_split(raw_line.substr(first, last - first + 1),
                                 " \t\r\n");
        }
        int nwords = words.size();
        Gdata force(ndim, 0.);
        if (nwords < ndim)
        {
            controller->printf("size %d not match %d\n", nwords, ndim);
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file \n");
        }
        else if (nwords < ndim + 2)
        {
            potential_from_file.push_back(0.);
        }
        else if (subhill && nwords >= ndim + 2)
        {
            potential_from_file.push_back(std::stof(words[nwords - 1]));
        }
        else if (nwords == 2 * ndim + 2)
        {
            potential_from_file.push_back(
                std::stof(words[2 * ndim + 1]));  // raw hill before sink
            if (!subhill)
            {
                for (int i = 0; i < ndim; ++i)
                {
                    force[i] = std::stof(words[1 + ndim + i]);
                }
            }
        }
        for (int i = 0; i < ndim; ++i)
        {
            tcoor[i][j] = std::stof(words[i]);  // coordinate!
        }
        force_from_file.push_back(force);
    }
    fclose(temp_file);
    Set_Grid(controller);
    auto max_it = std::max_element(potential_from_file.begin(),
                                   potential_from_file.end());
    potential_max = *max_it;
    if (usegrid)
    {
        mgrid->potential = potential_from_file;  // potential
        // calculate derivative force dpotential
        if (!subhill)
        {
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                for (int d = 0; d < ndim; ++d)
                {
                    mgrid->force[idx * ndim + d] = force_from_file[idx][d];
                }
            }
        }
    }
    else if (use_scatter)
    {
        mscatter->potential = potential_from_file;
        if (convmeta)
        {
            max_index = std::distance(potential_from_file.begin(), max_it);
        }
        if (!subhill)
        {
            mscatter->force.resize(scatter_size * ndim);
            for (int idx = 0; idx < scatter_size; ++idx)
            {
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->force[idx * ndim + d] = force_from_file[idx][d];
                }
            }
        }
        if (mask)
        {
            for (int index = 0; index < mscatter->num_points; ++index)
            {
                const Axis& coor = mscatter->Get_Coordinate(index);
                int gidx = mgrid->Get_Flat_Index(coor);
                mgrid->potential[gidx] = potential_from_file[index];

                for (int d = 0; d < ndim; ++d)
                {
                    mgrid->force[gidx * ndim + d] = force_from_file[index][d];
                }
            }
        }
    }
    if (mgrid != NULL) mgrid->Sync_To_Device();
    if (mscatter != NULL) mscatter->Sync_To_Device();
}

void META::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        controller->Step_Print(this->module_name, potential_local);
        controller->Step_Print("rbias", rbias);
        controller->Step_Print("rct", rct);
        return;
    }
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        MPI_Send(&potential_local, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rbias, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&rct, 1, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
    if (CONTROLLER::MPI_rank == 0)
    {
        MPI_Recv(&potential_local, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rbias, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rct, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        controller->Step_Print(this->module_name, potential_local);
        controller->Step_Print("rbias", rbias);
        controller->Step_Print("rct", rct);
    }
#endif
}

float exp_added(float a, const float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}
using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

std::string GetTime(TimePoint& local_time)
{
    local_time = std::chrono::system_clock::now();
    time_t now_time = std::chrono::system_clock::to_time_t(local_time);
    std::string time_str(asctime(localtime(&now_time)));
    size_t line_end = time_str.find('\n');
    if (line_end == std::string::npos)
    {
        return time_str;
    }
    return time_str.substr(0, line_end);
}

std::string GetDuration(const TimePoint& late_time, const TimePoint& early_time,
                        float& duration)
{
    // Some constants.
    const auto elapsed = late_time - early_time;
    const size_t milliseconds = static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    const size_t seconds = milliseconds / 1000000L;
    const size_t microseconds = milliseconds % 1000000L;
    const size_t Second2Day = 86400L;
    const size_t Second2Hour = 3600L;
    const size_t Second2Minute = 60L;
    const size_t day = seconds / Second2Day;
    const size_t hour = seconds % Second2Day / Second2Hour;
    const size_t min = seconds % Second2Day % Second2Hour / Second2Minute;
    const size_t second = seconds % Second2Day % Second2Hour % Second2Minute;
    // Calculate duration in second.
    const int BufferSize = 2048;
    char buffer[BufferSize];
    sprintf(buffer,
            "%lu days %lu hours %lu minutes %lu seconds %.1f milliseconds",
            static_cast<unsigned long>(day), static_cast<unsigned long>(hour),
            static_cast<unsigned long>(min), static_cast<unsigned long>(second),
            microseconds * 0.001);
    duration = milliseconds * 0.000001;  // From millisecond to second.
    return std::string(buffer);
}

static __global__ void Update_Edge_Effect_Grid(
    int total_size, int ndim, int scatter_size, const int* num_points,
    const float* lower, const float* spacing, const float* sigmas,
    const float* periods, const float* scatter_coordinates, int do_negative,
    float* normal_lse, float* normal_force)
{
    const int EDGE_EFFECT_MAX_DIM = 8;
    SIMPLE_DEVICE_FOR(gidx, total_size)
    {
        float values[EDGE_EFFECT_MAX_DIM];
        float nf[EDGE_EFFECT_MAX_DIM];
        int flat_index = gidx;
        for (int d = 0; d < ndim; ++d)
        {
            int i = flat_index % num_points[d];
            flat_index /= num_points[d];
            values[d] = lower[d] + (i + 0.5f) * spacing[d];
            nf[d] = 0.0f;
        }

        float max_log = -1.0e30f;
        float sum_exp = 0.0f;
        for (int index = 0; index < scatter_size; ++index)
        {
            float diffs[EDGE_EFFECT_MAX_DIM];
            float pregauss = 0.0f;
            float hill_potential = 1.0f;
            for (int d = 0; d < ndim; ++d)
            {
                float diff = values[d] - scatter_coordinates[index * ndim + d];
                if (periods[d] != 0.0f)
                {
                    diff -= roundf(diff / periods[d]) * periods[d];
                }
                diffs[d] = diff;
                float scaled = diff * sigmas[d];
                float gauss = expf(-0.5f * scaled * scaled);
                pregauss -= 0.5f * scaled * scaled;
                hill_potential *= gauss;
            }
            if (do_negative)
            {
                for (int d = 0; d < ndim; ++d)
                {
                    nf[d] += -diffs[d] * sigmas[d] * sigmas[d] * hill_potential;
                }
            }

            if (pregauss > max_log)
            {
                sum_exp = (sum_exp == 0.0f)
                              ? 1.0f
                              : sum_exp * expf(max_log - pregauss) + 1.0f;
                max_log = pregauss;
            }
            else
            {
                sum_exp += expf(pregauss - max_log);
            }
        }

        normal_lse[gidx] = max_log + logf(sum_exp);
        if (do_negative)
        {
            for (int d = 0; d < ndim; ++d)
            {
                normal_force[gidx * ndim + d] = nf[d];
            }
        }
    }
}

static __global__ void Update_Grid_With_Hill(
    int total_size, int ndim, const int* num_points, const float* lower,
    const float* spacing, const float* hill_centers, const float* hill_inv_w,
    const float* hill_periods, float factor, int update_force, float* potential,
    float* force)
{
    SIMPLE_DEVICE_FOR(idx, total_size)
    {
        float dx[8], df[8];
        int flat = idx;
        for (int d = 0; d < ndim; ++d)
        {
            int i = flat % num_points[d];
            flat /= num_points[d];
            float coord = lower[d] + (i + 0.5f) * spacing[d];
            float diff = coord - hill_centers[d];
            if (hill_periods[d] > 0.0f)
            {
                diff -= roundf(diff / hill_periods[d]) * hill_periods[d];
            }
            float x = diff * hill_inv_w[d];
            dx[d] = expf(-0.5f * x * x);
            df[d] = -x * hill_inv_w[d] * dx[d];
        }

        float pot = 1.0f;
        for (int d = 0; d < ndim; ++d) pot *= dx[d];
        potential[idx] += factor * pot;

        if (update_force)
        {
            for (int d = 0; d < ndim; ++d)
            {
                float tder = 1.0f;
                for (int j = 0; j < ndim; ++j) tder *= (j == d) ? df[j] : dx[j];
                force[idx * ndim + d] += factor * tder;
            }
        }
    }
}

static __global__ void Update_Scatter_With_Hill(
    int num_points, int ndim, const float* coordinates, const float* periods,
    const float* hill_centers, const float* hill_inv_w, float factor,
    int update_force, int use_cutoff, const float* cutoff, float* potential,
    float* force)
{
    SIMPLE_DEVICE_FOR(index, num_points)
    {
        float dx[8], df[8];
        const float* coord = coordinates + index * ndim;
        bool within = true;
        for (int d = 0; d < ndim; ++d)
        {
            float diff = coord[d] - hill_centers[d];
            if (periods[d] > 0.0f)
            {
                diff -= roundf(diff / periods[d]) * periods[d];
            }
            if (use_cutoff && fabsf(diff) > cutoff[d])
            {
                within = false;
                break;
            }
            float x = diff * hill_inv_w[d];
            dx[d] = expf(-0.5f * x * x);
            df[d] = -x * hill_inv_w[d] * dx[d];
        }
        if (within)
        {
            float hill_potential = 1.0f;
            for (int d = 0; d < ndim; ++d)
            {
                hill_potential *= dx[d];
            }
            potential[index] += factor * hill_potential;

            if (update_force && force != NULL)
            {
                float* data = force + index * ndim;
                for (int i = 0; i < ndim; ++i)
                {
                    float tder = 1.0f;
                    for (int j = 0; j < ndim; ++j)
                    {
                        tder *= (j == i) ? df[j] : dx[j];
                    }
                    data[i] += factor * tder;
                }
            }
        }
    }
}

float PartitionFunction(const float factor, float& i_max,
                        const std::vector<float>& values)
{
    if (values.empty() || factor < 0.0000001)
    {
        return 0.0;
    }
    i_max = *std::max_element(values.begin(), values.end());
    float maxVal = factor * i_max;
    float sum = 0.0f;
    for (const auto& x : values)
    {
        sum += std::exp(factor * x - maxVal);
    }
    return maxVal + logf(sum);
}
float logSumExp(const std::vector<float>& values)
{
    if (values.empty()) return -std::numeric_limits<float>::infinity();
    float maxVal = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (const auto& v : values)
    {
        sum += std::exp(v - maxVal);
    }
    return maxVal + logf(sum);
}

void hilllog(const std::string fn, const std::vector<float>& hillcenter,
             const std::vector<float>& hillheight)
{
    if (!fn.empty())
    {
        std::ofstream hillsout;
        hillsout.open(fn.c_str(), std::fstream::app);
        hillsout.precision(8);
        for (auto& gauss : hillcenter)
        {
            hillsout << gauss << "\t";
        }
        for (auto& hh : hillheight)
        {
            hillsout << hh << "\t";
        }
        hillsout << std::endl;
        hillsout.close();
    }
}
bool META::Read_Edge_File(const char* file_name, std::vector<float>& potential)
{
    FILE* temp_file = NULL;
    int grid_size = 0;
    bool readsuccess = true;
    int total = mgrid->total_size;
    std::vector<Gdata> force_from_file;
    controller->printf("Reading %d grid of edge effect\n", total);
    temp_file = fopen(file_name, "r+");
    if (temp_file == NULL)
    {
        return false;
    }
    if (temp_file != NULL)
    {
        fseek(temp_file, 0, SEEK_END);

        if (ftell(temp_file) == 0)
        {
            controller->printf("Edge file %s is empty\n", file_name);
            fclose(temp_file);
            return false;
        }
        else
        {
            Open_File_Safely(&temp_file, file_name, "r");
            char temp_char[256] = " ";  // empty but not nullptr
            int scanf_ret = 0;
            char* grid_val = temp_char;
            while (grid_val != NULL)
            {
                grid_val = fgets(temp_char, 256, temp_file);  // grid line
                if (grid_val == NULL) break;
                std::string raw_line(temp_char);
                size_t first = raw_line.find_first_not_of(" \t\r\n");
                size_t last = raw_line.find_last_not_of(" \t\r\n");
                std::vector<std::string> words;
                if (first != std::string::npos)
                {
                    words = string_split(
                        raw_line.substr(first, last - first + 1), " \t\r\n");
                }
                int nwords = words.size();
                Gdata force(ndim, 0.);
                if (nwords < ndim + 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                                   "META::Read_Edge_File",
                                                   "Edge file format mismatch");
                }
                potential.push_back(
                    logf(std::stof(words[ndim])));  /// log sum exp!!!!!
                if (nwords == 1 + ndim * 2)
                {
                    for (int i = 0; i < ndim; ++i)
                    {
                        force[i] = std::stof(words[1 + ndim + i]);
                    }
                }
                else
                {
                    controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                                   "META::Read_Edge_File",
                                                   "Edge file format mismatch");
                }
                ++grid_size;
                force_from_file.push_back(force);
            }
        }
        fclose(temp_file);
    }
    if (grid_size != total)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "META::Read_Edge_File",
                                       "Edge file line count mismatch");
    }
    mgrid->normal_lse = potential;
    sum_max = *std::max_element(potential.begin(), potential.end());
    if (scatter_size < total && do_negative)
    {
        for (int idx = 0; idx < mgrid->total_size; ++idx)
        {
            for (int d = 0; d < ndim; ++d)
            {
                mgrid->normal_force[idx * ndim + d] = force_from_file[idx][d];
            }
        }
    }
    return readsuccess;
}
// Load hills from output file.
int META::Load_Hills(const std::string& fn)
{
    std::ifstream hillsin(fn.c_str(), std::ios::in);
    if (!hillsin.is_open())
    {
        controller->printf("Warning, No record of hills\n");
        return 0;
    }
    const std::string file_content((std::istreambuf_iterator<char>(hillsin)),
                                   std::istreambuf_iterator<char>());
    hillsin.close();

    const int& cvsize = ndim;
    std::istringstream iss(file_content);
    std::string tstr;
    std::vector<std::string> words;
    int num_hills = 0;
    while (std::getline(iss, tstr, '\n'))
    {
        Axis values;
        size_t first = tstr.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue;
        }
        size_t last = tstr.find_last_not_of(" \t\r\n");
        words = string_split(tstr.substr(first, last - first + 1), " \t\r\n");
        if (words.size() < cvsize + 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "META::Load_Hills",
                                           "Hills file format is invalid");
        }
        for (int i = 0; i < cvsize; ++i)
        {
            float center = std::stof(words[i]);
            values.push_back(center);
        }
        float theight = std::stof(words[cvsize]);
        if (do_negative || use_scatter)
        {
            float p_max = std::stof(words[cvsize + 1]);
            int p_id = std::stoi(words[cvsize + 2]);
            if (p_id < scatter_size)
            {
                float Phi_s =
                    expf(mgrid->normal_lse[mgrid->Get_Flat_Index(values)]);
                float vshift = (p_max + dip * CONSTANT_kB * temperature) *
                               expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                                   mscatter->Get_Coordinate(p_id))]);
                vsink.push_back(Phi_s * vshift);
            }
            else
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "META::Load_Hills",
                    "Sink projection index is invalid");
            }
        }
        Hill newhill = Hill(values, sigmas, periods, theight);
        hills.push_back(newhill);
        ++num_hills;
    }
    return num_hills;
}
float META::Calc_Hill(const Axis& values, const int i)
{
    float potential = 0;
    for (int j = 0; j < i; ++j)
    {
        Hill& hill = hills[j];
        const Gdata& tder = hill.Calc_Hill(values);
        potential += hill.potential * hill.height;
    }
    return potential;
}
float META::Sum_Hills(int history_freq)
{
    if (history_freq == 0)
    {
        return 0.;
    }
    TimePoint start_time, end_time;
    float duration;
    GetTime(start_time);
    int nhills = Load_Hills("myhill.log");
    FILE* temp_file = NULL;
    controller->printf("Load hills file successfully, now calculate RCT...\n");
    Open_File_Safely(&temp_file, "history.log", "w");
    // first loop: history
    float old_potential;
    minus_beta_f_plus_v =
        1. / (welltemp_factor - 1.) / CONSTANT_kB / temperature;  /// 300K
    minus_beta_f = welltemp_factor * minus_beta_f_plus_v;
    float total_gputime = 0.;
    for (int hill_index = 0; hill_index < nhills; ++hill_index)
    {
        Hill& hill = hills[hill_index];
        Axis values;
        for (int dim = 0; dim < ndim; ++dim)
        {
            values.push_back(hill.centers_[dim]);
        }
        old_potential = Calc_Hill(values, hill_index);
        if (history_freq != 0 && (hill_index % history_freq == 0))
        {
            mgrid->potential.assign(mgrid->total_size, 0.0f);
            TimePoint tstart, tend;
            float gputime;
            GetTime(tstart);
            if (use_scatter)
            {
                for (int iter = 0; iter < scatter_size; ++iter)
                {
                    mscatter->potential[iter] =
                        Calc_Hill(mscatter->Get_Coordinate(iter), hill_index);
                }
                potential_max = 0.0f;
                float Z_0 = PartitionFunction(minus_beta_f, potential_max,
                                              mscatter->potential);
                float Z_V = PartitionFunction(
                    minus_beta_f_plus_v, potential_max, mscatter->potential);
                rct = CONSTANT_kB * temperature * (Z_0 - Z_V);
            }
            else  // use grid
            {
                for (int idx = 0; idx < mgrid->total_size; ++idx)
                {
                    mgrid->potential[idx] =
                        Calc_Hill(mgrid->Get_Coordinates(idx), hill_index);
                }
                potential_max = 0.0f;
                float Z_0 = PartitionFunction(minus_beta_f, potential_max,
                                              mgrid->potential);
                float Z_V = PartitionFunction(minus_beta_f_plus_v,
                                              potential_max, mgrid->potential);
                rct = CONSTANT_kB * temperature * (Z_0 - Z_V);
            }
            GetTime(tend);
            GetDuration(tend, tstart, gputime);
            total_gputime += gputime;
            float rbias = old_potential - rct;
            fprintf(temp_file, "%f\t%f\t%f\t%f\n", old_potential, rbias, rct,
                    vsink[hill_index]);
        }
    }
    fclose(temp_file);
    GetTime(end_time);
    GetDuration(end_time, start_time, duration);
    int hours = floor(duration / 3600);
    float nohour = duration - 3600 * hours;
    int mins = floor(nohour / 60);
    float seconds = nohour - 60 * mins;
    controller->printf(
        "The RBIAS & RCT calculation cost %f of %f seconds: %d hour %d min %f "
        "second\n",
        total_gputime / duration, duration, hours, mins, seconds);
    return old_potential;
}
void META::Edge_Effect(const int dim, const int scatter_size)
{
    std::vector<float> potential_from_file;
    const char* file_name = edge_file_name;

    int total = mgrid->total_size;
    if (scatter_size == total)
    {
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi * 2);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        mgrid->normal_lse.assign(mgrid->total_size, log(normalization));
    }
    bool readsuccess = Read_Edge_File(file_name, potential_from_file);
    if (has_edge_file_input && !readsuccess)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "META::Edge_Effect",
                                       "Failed to read edge_in_file");
    }
    if (!readsuccess)
    {
        int it_progress = 0;
        controller->printf("Calculation the %d grid of edge effect\n", total);
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, file_name, "w+");
        bool can_use_device_edge = (ndim <= 8);
        Axis esigmas;
        float adjust_factor = 1.0;
        for (int i = 0; i < ndim; ++i)
        {
            esigmas.push_back(sigmas[i] * adjust_factor);
        }
        if (can_use_device_edge)
        {
            mgrid->normal_lse.assign(mgrid->total_size, 0.0f);
            mgrid->normal_force.assign(mgrid->total_size * ndim, 0.0f);
            mgrid->Alloc_Device();
            mscatter->Alloc_Device();
            deviceMemcpy(d_hill_inv_w, esigmas.data(), sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            Launch_Device_Kernel(Update_Edge_Effect_Grid, total, 32, 0, NULL,
                                 total, ndim, scatter_size, mgrid->d_num_points,
                                 mgrid->d_lower, mgrid->d_spacing, d_hill_inv_w,
                                 mscatter->d_periods, mscatter->d_coordinates,
                                 static_cast<int>(do_negative),
                                 mgrid->d_normal_lse, mgrid->d_normal_force);
            mgrid->Sync_To_Host();
            for (int gidx = 0; gidx < mgrid->total_size; ++gidx)
            {
                const Axis values = mgrid->Get_Coordinates(gidx);
                float logsumhills = mgrid->normal_lse[gidx];
                sum_max = fmaxf(logsumhills, sum_max);
                for (auto& v : values)
                {
                    fprintf(temp_file, "%f\t", v);
                }
                fprintf(temp_file, "%f\t", expf(logsumhills));
                if (do_negative)
                {
                    float* nf_data = &mgrid->normal_force[gidx * ndim];
                    for (int i = 0; i < ndim; ++i)
                    {
                        fprintf(temp_file, "%f\t", nf_data[i]);
                    }
                }
                fprintf(temp_file, "\n");
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorNotImplemented, "META::Edge_Effect",
                "Edge_Effect only supports ndim <= 8");
        }
        fclose(temp_file);
    }
    if (dim == 1)
    {
        Pick_Scatter("lnbias.dat");
    }
}

void META::Pick_Scatter(const std::string fn)
{
    std::ofstream hillsout;
    hillsout.open(fn.c_str(), std::fstream::out);
    hillsout.precision(8);
    for (int index = 0; index < scatter_size; ++index)
    {
        const Axis& neighbor = mscatter->Get_Coordinate(index);
        float lnbias = mgrid->normal_lse[mgrid->Get_Flat_Index(neighbor)];
        hillsout << index << "\t" << lnbias << "\t" << exp(lnbias) << std::endl;
    }
    hillsout.close();
}
float META::Normalization(const Axis& values, float factor, bool do_normalise)
{
    if (do_normalise)
    {
        if (usegrid)
        {
            return factor * expf(-mgrid->normal_lse[0]);
        }
        if (convmeta)
        {
            return factor * expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                                mscatter->Get_Coordinate(max_index))]);
        }
        else
        {
            return factor *
                   expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(values)]);
        }
    }
    else
    {
        return factor;
    }
}
void META::Get_Height(const Axis& values)
{
    Estimate(values, true, false);
    height = height_0;
    if (temperature < 0.00001 || welltemp_factor > 60000)
    {
        return;  // avoid /0 = nan
    }
    if (is_welltemp == 1)
    {
        // height_welltemp
        height = height_0 * expf(-potential_backup / (welltemp_factor - 1) /
                                 CONSTANT_kB / temperature);
    }
}

float META::Calc_V_Shift(const Axis& values)
{
    if (!do_negative)
    {
        return 0.;
    }
    int nidx = mgrid->Get_Flat_Index(values);
    if (convmeta)
    {
        return new_max * expf(mgrid->normal_lse[nidx]);
    }
    else  // GRW
    {
        return new_max * (mgrid->normal_lse[nidx] - sum_max) *
               expf(mgrid->normal_lse[nidx]);
    }
}
void META::Get_Reweighting_Bias(float temp)
{
    if (temperature < 0.00001)
    {
        return;  // avoid /0 = nan
    }
    float beta = 1.0 / CONSTANT_kB / temperature;
    minus_beta_f_plus_v = beta / (welltemp_factor - 1.);
    minus_beta_f = welltemp_factor * minus_beta_f_plus_v;
    bias = potential_local;
    rbias = potential_backup;
    float Z_0_sink = 0.;
    float Z_V_sink = 0.;
    if (mscatter != NULL)
    {
        for (int iter = 0; iter < scatter_size; ++iter)
        {
            const Axis& coor = mscatter->Get_Coordinate(iter);
            Estimate(coor, true, false);
            Z_0_sink = exp_added(Z_0_sink, minus_beta_f * potential_backup);
            Z_V_sink =
                exp_added(Z_V_sink, minus_beta_f_plus_v * potential_backup +
                                        beta * Calc_V_Shift(coor));
            if (potential_backup > potential_max)
            {
                max_index = iter;
                potential_max = potential_backup;
            }
        }
    }
    else if (mgrid != NULL)
    {
        if (subhill)
        {
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                Estimate(mgrid->Get_Coordinates(idx), true, false);
                potential_max = std::max(potential_max, potential_backup);
            }
        }
        else
        {
            potential_max = *std::max_element(mgrid->potential.begin(),
                                              mgrid->potential.end());
        }
    }
    rct = CONSTANT_kB * temperature * (Z_0_sink - Z_V_sink);
    rbias -= rct + temp;
}

void META::Add_Potential(float temp, int steps)
{
    if (!is_initialized)
    {
        return;
    }
    if (potential_update_interval <= 0)
    {
        return;
    }
    if (steps % potential_update_interval == 0)
    {
        Axis values;
        for (int i = 0; i < cvs.size(); ++i)
        {
            values.push_back(cvs[i]->value);
        }
        Get_Height(values);
        float vshift = Calc_V_Shift(values);
        Get_Reweighting_Bias(vshift);
        Hill hill = Hill(values, sigmas, periods, height);
        hills.push_back(hill);
        Axis hillinfo;
        hillinfo.push_back(height);
        if (do_negative)
        {
            hillinfo.push_back(potential_max);
            hillinfo.push_back(max_index);
        }
        if (mscatter != NULL)
        {
            hillinfo.push_back(mscatter->Get_Index(values));
        }
        hilllog("myhill.log", values, hillinfo);
        exit_tag = 0.0;
        if (!kde && subhill)
        {
            const Gdata& tder = hill.Calc_Hill(values);
            if (mgrid != NULL)
            {
                mgrid->potential[mgrid->Get_Flat_Index(values)] +=
                    height * hill.potential;
            }
            else if (mscatter != NULL)
            {
                int sidx = mscatter->Get_Index(values);
                mscatter->potential[sidx] += height * hill.potential;
                deviceMemcpy(mscatter->d_potential + sidx,
                             &mscatter->potential[sidx], sizeof(float),
                             deviceMemcpyHostToDevice);
            }
            return;
        }
        float factor = Normalization(values, height,
                                     kde);  // height with normalized factor
        if (use_scatter)
        {
            float h_centers[8], h_inv_w[8];
            for (int d = 0; d < ndim; ++d)
            {
                h_centers[d] = hill.centers_[d];
                h_inv_w[d] = hill.inv_w_[d];
            }
            deviceMemcpy(d_hill_centers, h_centers, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_hill_inv_w, h_inv_w, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            int update_force = (!mscatter->force.empty()) ? 1 : 0;
            Launch_Device_Kernel(
                Update_Scatter_With_Hill, (scatter_size + 255) / 256, 256, 0,
                NULL, scatter_size, ndim, mscatter->d_coordinates,
                mscatter->d_periods, d_hill_centers, d_hill_inv_w, factor,
                update_force, do_cutoff ? 1 : 0, d_cutoff,
                mscatter->d_potential, mscatter->d_force);
            mscatter->Sync_To_Host();
        }
        // Update grid potential and force with hill on device
        if (mgrid != NULL)
        {
            float h_centers[8], h_inv_w[8], h_periods[8];
            for (int d = 0; d < ndim; ++d)
            {
                h_centers[d] = hill.centers_[d];
                h_inv_w[d] = hill.inv_w_[d];
                h_periods[d] = periods[d];
            }
            deviceMemcpy(d_hill_centers, h_centers, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_hill_inv_w, h_inv_w, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_hill_periods, h_periods, sizeof(float) * ndim,
                         deviceMemcpyHostToDevice);
            int update_force = (!subhill && !mgrid->force.empty()) ? 1 : 0;
            Launch_Device_Kernel(
                Update_Grid_With_Hill, (mgrid->total_size + 255) / 256, 256, 0,
                NULL, mgrid->total_size, ndim, mgrid->d_num_points,
                mgrid->d_lower, mgrid->d_spacing, d_hill_centers, d_hill_inv_w,
                d_hill_periods, factor, update_force, mgrid->d_potential,
                mgrid->d_force);
            mgrid->Sync_To_Host();
        }
    }
}

void META::Initial(CONTROLLER* controller,
                   COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                   char* module_name)
{
    this->controller = controller;
    if (module_name == NULL)
    {
        strcpy(this->module_name, "meta");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!cv_controller->Command_Exist(this->module_name, "CV"))
    {
        controller->printf("META IS NOT INITIALIZED\n\n");
        return;
    }
    else
    {
        std::vector<std::string> cv_str =
            cv_controller->Ask_For_String_Parameter(this->module_name, "CV",
                                                    ndim);
        std::string cvv =
            std::accumulate(cv_str.begin(), cv_str.end(), std::string(""));
        controller->printf("%s contains %d dimension META\n", cvv.c_str(),
                           ndim);
    }
    if (cv_controller->Command_Exist(this->module_name, "dip"))
    {
        dip = cv_controller->Ask_For_Float_Parameter(this->module_name, "dip",
                                                     1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "welltemp_factor"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "welltemp_factor");
        welltemp_factor = temp_value[0];
        free(temp_value);
        if (welltemp_factor > 1)
        {
            is_welltemp = 1;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                "welltemp_factor must be greater than 1");
        }
    }
    cvs = cv_controller->Ask_For_CV(this->module_name, -1);
    if (cv_controller->Command_Exist(this->module_name, "Ndim"))
    {
        ndim = *cv_controller->Ask_For_Int_Parameter(this->module_name, "Ndim");
        if (ndim != cvs.size())
        {
            controller->printf("%d D-META IS NOT CONSISTANT CV size %d\n\n",
                               ndim, cvs.size());
            return;
        }
    }
    else
    {
        ndim = cvs.size();
    }
    controller->printf("START INITIALIZING %dD-META:\n", ndim);
    Malloc_Safely((void**)&Dpotential_local, sizeof(float) * ndim);
    sprintf(read_potential_file_name, "Meta_Potential.txt");
    sprintf(write_potential_file_name, "Meta_Potential.txt");
    if (controller->Command_Exist("default_in_file_prefix"))
    {
        sprintf(read_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_in_file_prefix"));
    }
    else
    {
        sprintf(read_potential_file_name, "Meta_Potential.txt");
    }
    if (controller->Command_Exist("default_out_file_prefix"))
    {
        sprintf(write_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_out_file_prefix"));
    }
    else
    {
        sprintf(write_potential_file_name, "Meta_Potential.txt");
    }
    sprintf(write_directly_file_name, "Meta_directly.txt");
    sprintf(edge_file_name, "sumhill.log");
    has_edge_file_input = false;
    if (cv_controller->Command_Exist(this->module_name, "edge_in_file"))
    {
        has_edge_file_input = true;
        strcpy(edge_file_name, cv_controller
                                   ->Ask_For_String_Parameter(this->module_name,
                                                              "edge_in_file")[0]
                                   .c_str());
    }
    if (cv_controller->Command_Exist(this->module_name, "subhill"))
    {
        subhill = true;
        controller->printf("    reading subhill for meta: 1\n");
    }
    if (cv_controller->Command_Exist(this->module_name, "kde"))
    {
        int kde_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "kde", 1)[0];
        if (kde_dim)
        {
            kde = true;
            subhill = true;
            controller->printf("    reading kde's subhill for meta: %d\n",
                               kde_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "mask"))
    {
        mask = cv_controller->Ask_For_Int_Parameter(this->module_name, "mask",
                                                    1)[0];
        if (mask)
        {
            controller->printf("    reading mask dimension meta: %d\n", mask);
            if (cv_controller->Command_Exist(this->module_name, "max_force"))
            {
                max_force = cv_controller->Ask_For_Float_Parameter(
                    this->module_name, "max_force", 1)[0];
            }
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sink"))
    {
        int sub_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "sink", 1)[0];
        if (sub_dim > 0)
        {
            do_negative = true;
            controller->printf(
                "    reading sink/submarine dimension for meta: %d\n", sub_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sumhill_freq"))
    {
        history_freq = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "sumhill_freq", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "convmeta"))
    {
        do_negative = true;
        convmeta = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                        "convmeta", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "grw"))
    {
        do_negative = true;
        grw = cv_controller->Ask_For_Int_Parameter(this->module_name, "grw",
                                                   1)[0];
    }
    cv_periods = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_period", cvs.size(), 1, false);
    cv_sigmas = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                       "CV_sigma", cvs.size());
    cutoff = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_sigma", cvs.size(), 1, false, 0., -3);
    if (cv_controller->Command_Exist(this->module_name, "cutoff"))
    {
        do_cutoff = true;
        cutoff = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                        "cutoff", cvs.size());
    }
    for (int i = 0; i < cvs.size(); i++)
    {
        if (cv_sigmas[i] <= 0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                "CV_sigma should always be greater than 0");
        }
        if (!do_cutoff)
        {
            cutoff[i] = 3 * cv_sigmas[i];
        }
        if (kde)
        {
            cv_sigmas[i] = 1.414f / cv_sigmas[i];
        }
        else
        {
            cv_sigmas[i] = 1.0 / cv_sigmas[i];
        }
    }
    for (int i = 0; i < ndim; i++)
    {
        sigmas.push_back(cv_sigmas[i]);
        periods.push_back(cv_periods[i]);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_in_file"))
    {
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else if (cv_controller->Command_Exist(this->module_name, "scatter_in_file"))
    {
        usegrid = false;
        use_scatter = true;
        controller->printf("    Use %d scatter point for CV!\n", scatter_size);
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "scatter_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else
    {
        if (cv_controller->Command_Exist(this->module_name, "scatter"))
        {
            scatter_size = *(cv_controller->Ask_For_Int_Parameter(
                this->module_name, "scatter", 1));
            if (scatter_size > 0)
            {
                usegrid = false;
                use_scatter = true;
                controller->printf("    Use %d scatter point for CV!\n",
                                   scatter_size);
                for (int i = 0; i < cvs.size(); i++)
                {
                    tcoor.push_back(cv_controller->Ask_For_Float_Parameter(
                        cvs[i]->module_name, "CV_point", scatter_size, 1,
                        false));
                }
            }
            else
            {
                controller->printf("    Not using scatter point for CV\n");
                use_scatter = false;
            }
        }

        cv_mins = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_minimal", cvs.size());
        cv_maxs = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_maximum", cvs.size());
        n_grids = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                       "CV_grid", cvs.size());
        for (int i = 0; i < cvs.size(); ++i)
        {
            if (cv_maxs[i] <= cv_mins[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_maximum should always be greater than CV_minimal");
            }
            if (n_grids[i] <= 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_grid should always be greater than 1");
            }
            cv_deltas.push_back((cv_maxs[i] - cv_mins[i]) / n_grids[i]);
        }
        Set_Grid(controller);
    }
    height_0 = 1.0;
    if (cv_controller->Command_Exist(this->module_name, "height"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "height");
        height_0 = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "wall_height"))
    {
        do_borderwall = true;
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "wall_height");
        border_potential_height = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_out_file"))
    {
        strcpy(write_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_out_file")[0]
                   .c_str());
    }
    bool has_potential_update_interval = false;
    if (cv_controller->Command_Exist(this->module_name,
                                     "potential_update_interval"))
    {
        int* temp_value = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "potential_update_interval");
        potential_update_interval = temp_value[0];
        has_potential_update_interval = true;
        free(temp_value);
    }
    if (controller->Command_Exist("write_information_interval"))
    {
        write_information_interval =
            atoi(controller->Command("write_information_interval"));
    }
    else
    {
        write_information_interval = 1000;
    }
    if (write_information_interval <= 0)
    {
        write_information_interval = 1000;
    }
    if (!has_potential_update_interval)
    {
        controller->printf(
            "    Potential update interval is set to "
            "write_information_interval by default\n");
        potential_update_interval = write_information_interval;
    }
    if (potential_update_interval <= 0)
    {
        potential_update_interval = 1000;
    }
    controller->Step_Print_Initial("meta", "%f");
    controller->Step_Print_Initial("rbias", "%f");
    controller->Step_Print_Initial("rct", "%f");
    controller->printf("    potential output file: %s\n",
                       write_potential_file_name);
    controller->printf("    edge effect file: %s\n", edge_file_name);
    is_initialized = 1;
    controller->printf("END INITIALIZING META\n\n");
}

META::Hill::Hill(const Axis& centers, const Axis& inv_w, const Axis& period,
                 const float& theight)
    : centers_(centers), inv_w_(inv_w), periods_(period), height(theight)
{
    const int n = static_cast<int>(centers_.size());
    dx_.resize(n);
    df_.resize(n);
    tder_.resize(n);
}

const META::Gdata& META::Hill::Calc_Hill(const Axis& values)
{
    const int n = static_cast<int>(values.size());
    for (int i = 0; i < n; ++i)
    {
        dx_[i] = Evaluate_Gaussian_Switch(values[i], centers_[i], inv_w_[i],
                                          periods_[i], df_[i]);
    }
    potential = 1.0;
    for (int i = 0; i < n; ++i)
    {
        tder_[i] = 1.0;
        potential *= dx_[i];
        for (int j = 0; j < n; ++j)
        {
            if (j != i)
            {
                tder_[i] *= dx_[j];
            }
            else
            {
                tder_[i] *= df_[j];
            }
        }
    }
    return tder_;
}

static __global__ void Add_Frc(const int atom_numbers, VECTOR* frc,
                               VECTOR* cv_grad, float dheight_dcv)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static __global__ void Add_Potential_Kernel(float* d_potential,
                                            const float to_add)
{
    d_potential[0] += to_add;
}

static __global__ void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                                  const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}

void META::Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                             int need_potential,
                                             int need_pressure,
                                             float* d_potential,
                                             LTMatrix3* d_virial)
{
    if (!is_initialized)
    {
        return;
    }
    Potential_And_Derivative(need_potential);
    if (do_borderwall)
    {
        Border_Derivative(border_upper.data(), border_lower.data(), cutoff,
                          Dpotential_local);
    }

    for (int i = 0; i < cvs.size(); ++i)
    {
        Launch_Device_Kernel(Add_Frc, (atom_numbers + 31 / 32), 32, 0, NULL,
                             atom_numbers, frc, cvs[i]->crd_grads,
                             Dpotential_local[i]);
        if (need_pressure)
        {
            Launch_Device_Kernel(Add_Virial, 1, 1, 0, NULL, d_virial,
                                 Dpotential_local[i], cvs[i]->virial);
        }
    }
    if (need_potential)
    {
        Launch_Device_Kernel(Add_Potential_Kernel, 1, 1, 0, NULL, d_potential,
                             potential_local);
    }
}

void META::Potential_And_Derivative(const int need_potential)
{
    if (!is_initialized)
    {
        return;
    }
    for (int i = 0; i < cvs.size(); ++i)
    {
        est_values_[i] = cvs[i]->value;
        Dpotential_local[i] = 0.f;
    }
    Estimate(est_values_, need_potential, true);
}

void META::Border_Derivative(float* border_upper, float* border_lower,
                             float* cutoff, float* Dpotential_local)
{
    for (int i = 0; i < cvs.size(); ++i)
    {
        float h_cv = cvs[i]->value;
        if (h_cv - border_lower[i] < cutoff[i])
        {
            float distance = border_lower[i] - h_cv;
            if (periods[i] > 0)
            {
                distance -= roundf(distance / cv_periods[i]) * cv_periods[i];
            }
            Dpotential_local[i] =
                Dpotential_local[i] - border_potential_height * expf(distance);
        }
        else if (border_upper[i] - h_cv < cutoff[i])
        {
            float distance = h_cv - border_upper[i];
            if (periods[i] > 0)
            {
                distance -= roundf(distance / cv_periods[i]) * cv_periods[i];
            }
            Dpotential_local[i] =
                Dpotential_local[i] + border_potential_height * expf(distance);
        }
    }
}

void META::Do_Metadynamics(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                           LTMatrix3 rcell, int step, int need_potential,
                           int need_pressure, VECTOR* frc, float* d_potential,
                           LTMatrix3* d_virial, float sys_temp)
{
    if (this->is_initialized)
    {
        int need = CV_NEED_GPU_VALUE | CV_NEED_CRD_GRADS;
        if (need_pressure)
        {
            need |= CV_NEED_VIRIAL;
        }

        for (int i = 0; i < cvs.size(); i = i + 1)
        {
            this->cvs[i]->Compute(atom_numbers, crd, cell, rcell, need, step);
        }
        temperature = sys_temp;
        Meta_Force_With_Energy_And_Virial(atom_numbers, frc, need_potential,
                                          need_pressure, d_potential, d_virial);
        Add_Potential(sys_temp, step);
    }
}
