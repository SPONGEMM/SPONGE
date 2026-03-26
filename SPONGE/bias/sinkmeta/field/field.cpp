#include "../meta.h"
void META::Setgrid(CONTROLLER* controller)  //
{
    std::vector<int> ngrid;
    std::vector<float> lower, upper, periodic;
    std::vector<bool> isperiodic;
    border_upper.resize(ndim);
    border_lower.resize(ndim);
    est_values_.resize(ndim);
    est_sum_force_.resize(ndim);
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
        mscatter = nullptr;
        Sumhills(history_freq);
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
            printf("Error, scatter size %d larger than grid %d!\n",
                   scatter_size, oldsize);
            mscatter = nullptr;
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
        if (catheter)
        {
            mscatter->rotate_v.assign(scatter_size * ndim, 0.0f);
            for (size_t index = 0; index < scatter_size - 1; ++index)
            {
                const Axis& values = mscatter->Get_Coordinate(index);
                const Axis& neighbor = mscatter->Get_Coordinate(index + 1);
                Gdata tang(ndim, 0.0f);
                double temp_s = TangVector(tang, values, neighbor);
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->rotate_v[index * ndim + d] = tang[d];
                }
            }
            {
                Gdata tang(ndim, 0.0f);
                double temp_sp =
                    TangVector(tang,
                               mscatter->Get_Coordinate(scatter_size - 2),
                               mscatter->Get_Coordinate(scatter_size - 1));
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->rotate_v[(scatter_size - 1) * ndim + d] = tang[d];
                }
            }

            mscatter->rotate_matrix.assign(scatter_size * ndim * ndim, 0.0f);
            for (size_t index = 0; index < scatter_size - 1; ++index)
            {
                const Axis& values = mscatter->Get_Coordinate(index);
                const Axis& neighbor = mscatter->Get_Coordinate(index + 1);
                Axis tang_vector(ndim, 0.);
                double segment_s = TangVector(tang_vector, values, neighbor);
                int base = index * ndim * ndim;
                int pos = 0;
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->rotate_matrix[base + pos++] = tang_vector[d];
                }
                Axis normal_vector = RotateVector(tang_vector, false);
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->rotate_matrix[base + pos++] = normal_vector[d];
                }
                if (ndim == 3)
                {
                    Axis binormal_vector =
                        normalize(crossProduct(tang_vector, normal_vector));
                    for (int d = 0; d < ndim; ++d)
                    {
                        mscatter->rotate_matrix[base + pos++] = binormal_vector[d];
                    }
                }
            }
            int rm_stride = ndim * ndim;
            for (int j = 0; j < rm_stride; ++j)
            {
                mscatter->rotate_matrix[(scatter_size - 1) * rm_stride + j] =
                    mscatter->rotate_matrix[(scatter_size - 2) * rm_stride + j];
            }
        }
        EdgeEffect(1, scatter_size);
        Sumhills(history_freq);
        mgrid->Alloc_Device();
        mscatter->Alloc_Device();
    }
    else
    {
        printf("Warning! No grid version is very slow\n");
        mscatter = nullptr;
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
    if (force_max > maxforce && need_force && mask)
    {
        exit_tag += 1.0;
    }
    if (use_scatter)
    {
        if (subhill)
        {
            Hill hill = Hill(values, sigmas, periods, 1.0);
            vector<int> indices;
            if (do_cutoff)
            {
                indices = mscatter->Get_Neighbor(values, cutoff);
            }
            else
            {
                indices = vector<int>(scatter_size);
                iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                const Axis& neighbor = mscatter->Get_Coordinate(index);
                const Gdata& tder = hill.CalcHill(neighbor);
                normalforce_sum += hill.potential;
                float factor = (mask > 0)
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
            potential_backup = (mask > 0)
                                    ? mgrid->potential[mgrid->Get_Flat_Index(values)]
                                    : mscatter->potential[sidx];
            potential_local = potential_backup - CalcVshift(values);
            if (need_force)
            {
                int fidx = (mask > 0) ? mgrid->Get_Flat_Index(values) : sidx;
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        (mask > 0)
                            ? mgrid->force[fidx * ndim + i]
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
                const Gdata& tder = hill.CalcHill(loop_flag);
                float factor = mgrid->potential[mgrid->Get_Flat_Index(loop_flag)];
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
        potential_local = potential_backup - CalcVshift(values);
    }
    if (need_force)
    {
        if (subhill)
        {
            float f0 = new_max * mgrid->normal_force[nf_idx * ndim + 0];
            if (convmeta)
            {
                new_max =
                    shift *
                    expf(-mgrid->normal_lse[mgrid->Get_Flat_Index(
                        mscatter->Get_Coordinate(max_index))]);
            }
            else
            {
                new_max = shift / normalforce_sum;
            }
            float f1 = new_max * est_sum_force_[0];
            if (fabs(f0 - f1) > shift)
            {
                printf("The shift, kde & histogram:%f: %f vs %f\n", shift, f1,
                       f0);
            }
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * est_sum_force_[i];
            }
        }
        else
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * mgrid->normal_force[nf_idx * ndim + i];
            }
        }
    }
    return;
}
