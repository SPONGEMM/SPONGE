#include "../meta.h"
void META::Setgrid(CONTROLLER* controller)  //
{
    std::vector<int> ngrid;
    std::vector<float> lower, upper, periodic;
    std::vector<bool> isperiodic;
    border_upper.resize(ndim);
    border_lower.resize(ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
        ngrid.push_back(n_grids[i]);
        lower.push_back(cv_mins[i]);
        upper.push_back(cv_maxs[i]);
        periodic.push_back(cv_periods[i]);
        isperiodic.push_back(cv_periods[i] > 0 ? true : false);
    }
    normal_force = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
    // normal_factor = new Grid<float>(ngrid, lower, upper, isperiodic);
    normal_lse = new Grid<float>(ngrid, lower, upper, isperiodic);
    normal_force->data_ = vector<Gdata>(normal_force->size(), Gdata(ndim, 0.0));
    // Write Potential need potential_grid!!!!!!!!!!
    potential_grid = new Grid<float>(ngrid, lower, upper, isperiodic);
    potential_grid->data_ = vector<float>(potential_grid->size(), 0.0);
    if (usegrid)
    {
        grid = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
        grid->data_ =
            vector<Gdata>(grid->size(), Gdata(grid->GetDimension(), 0.0));
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        // normal_factor->data_ = vector<float>(normal_factor->size(),
        // normalization);
        normal_lse->data_ =
            vector<float>(normal_lse->size(), log(normalization));
        scatter = nullptr;
        potential_scatter = nullptr;
        // EdgeEffect(ndim, normal_lse->size());
        Sumhills(history_freq);
    }
    else if (use_scatter)
    {
        if (mask > 0)
        {
            grid = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
            grid->data_ =
                vector<Gdata>(grid->size(), Gdata(grid->GetDimension(), 0.0));
        }
        else
        {
            grid = nullptr;
            potential_scatter = nullptr;
        }
        std::vector<int> nscatter;
        int oldsize = 1;  // cvs[0]->point->size();
        for (size_t i = 0; i < ndim; ++i)
        {
            nscatter.push_back(n_grids[i]);
            oldsize *= n_grids[i];
        }
        max_index = floor(scatter_size / 2);  /// initial at the middle point!
        if (oldsize < scatter_size)
        {
            printf("Error, scatter size %d larger than grid %d!\n",
                   scatter_size, oldsize);
            grid = nullptr;
            potential_scatter = nullptr;
            scatter = nullptr;
            potential_scatter = nullptr;
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                           "Meta::SetGrid()\n");
            return;
        }
        std::vector<std::vector<float>> coor;  //(oldsize);
        for (size_t j = 0; j < scatter_size; ++j)
        {
            std::vector<float> p;
            for (size_t i = 0; i < ndim; ++i)
            {
                // printf("Coordinate of (%d,%d) is %f\n",i,j,pp);
                p.push_back(tcoor[i][j]);
            }
            coor.push_back(p);
        }
        scatter = new Scatter<Gdata>(nscatter, periodic, coor);
        potential_scatter = new Scatter<float>(nscatter, periodic, coor);
        scatter->data_ =
            vector<Gdata>(scatter_size, Gdata(scatter->GetDimension(), 0.0));
        potential_scatter->data_ =
            vector<float>(potential_scatter->size(), 0.0);
        if (catheter)
        {
            // Method 3 use s and v only
            rotate_v = new Scatter<Gdata>(nscatter, periodic, coor);
            rotate_v->data_ = vector<Gdata>(scatter_size, Gdata(ndim, 0.0));
            for (size_t index = 0; index < scatter_size - 1;
                 ++index)  // : indices)
            {
                Axis values = rotate_v->GetCoordinate(index);
                Axis neighbor = rotate_v->GetCoordinate(index + 1);
                // Gdata data: Tangent Vector normalized as unit vector;
                Gdata& data = rotate_v->data()[index];
                double temp_s = TangVector(data, values, neighbor);
            }
            double temp_sp =
                TangVector(rotate_v->data()[scatter_size - 1],
                           rotate_v->GetCoordinate(scatter_size - 2),
                           rotate_v->GetCoordinate(scatter_size - 1));

            // Method 1 need R matrix
            rotate_matrix = new Scatter<Gdata>(nscatter, periodic, coor);
            rotate_matrix->data_ =
                vector<Gdata>(scatter_size, Gdata(ndim * ndim, 0.0));
            // Rotate matrix is special orthogonal matrix: R^{-1}=R^T
            for (size_t index = 0; index < scatter_size - 1;
                 ++index)  // : indices)
            {
                Gdata data;
                Axis values = rotate_matrix->GetCoordinate(index);
                Axis neighbor = rotate_matrix->GetCoordinate(index + 1);
                Axis tang_vector(ndim, 0.);  // TANGENTIAL
                double segment_s = TangVector(tang_vector, values, neighbor);
                for (auto t : tang_vector)
                {
                    data.push_back(t);
                }
                Axis normal_vector = RotateVector(tang_vector, false);
                for (auto n : normal_vector)
                {
                    data.push_back(n);
                }
                if (ndim == 3)
                {
                    Axis binormal_vector =
                        normalize(crossProduct(tang_vector, normal_vector));
                    for (auto b : binormal_vector)
                    {
                        data.push_back(b);
                    }
                }
                /*
                if (!CheckOrthogonal(data, ndim))
                {
                    controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                "METAD::SetGrid", "Rotate matrix is no an orthogonal matrix");
                }*/
                rotate_matrix->data()[index] = data;
            }
            rotate_matrix->data()[scatter_size - 1] =
                rotate_matrix->data()[scatter_size - 2];
        }
        // calculate normal_factor and print
        EdgeEffect(1, scatter_size);
        Sumhills(history_freq);
    }
    else
    {
        printf("Warning! No grid version is very slow\n");
        grid = nullptr;
        potential_scatter = nullptr;
        scatter = nullptr;
        potential_scatter = nullptr;
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
    float force_max = 0.0;  // Add the edge's force
    float normalforce_sum = 0.0;
    Gdata sum_force(ndim, 0.);
    for (size_t i = 0; i < ndim; ++i)
    {
        Dpotential_local[i] = 0.0;
        force_max += fabs(normal_force->at(values)[i]);
    }
    if (force_max > maxforce && need_force && mask)
    {
        exit_tag += 1.0;
    }
    // Axis aaaaa =
    // potential_grid->GetCoordinates(potential_grid->GetIndices(values));
    Hill hill = Hill(values, sigmas, periods, 1.0);
    if (use_scatter)
    {
        if (subhill)
        {
            vector<Gdata> derivative;
            vector<int> indices;
            if (do_cutoff)
            {
                indices = potential_scatter->GetNeighbor(values, cutoff);
            }
            else
            {
                indices = vector<int>(scatter_size);
                iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                Axis neighbor = potential_scatter->GetCoordinate(index);
                Gdata tder = hill.CalcHill(neighbor);
                normalforce_sum += hill.potential;
                float factor = (mask > 0) ? potential_grid->at(neighbor)
                                          : potential_scatter->data()[index];
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        sum_force[i] += tder[i];
                        Dpotential_local[i] -= (factor)*tder[i];
                    }
                }
                // ratio =
                // normal_lse->at(potential_scatter->GetCoordinate(potential_scatter->GetIndex(neighbor)))-normal_lse->at(neighbor);
                // potential_local += (factor- new_max) * hill.potential;
                potential_backup += factor * hill.potential;
            }
            /*
            if (convmeta)
            {
                potential_local = potential_backup - shift *
            expf(normal_lse->at(scatter->GetCoordinate(values))-normal_lse->at(scatter->GetCoordinate(max_index)));
            /// sum_max;
            }
            else
            {
                potential_local = potential_backup - shift *
            (normal_lse->at(scatter->GetCoordinate(values))-  sum_max);
            }*/
        }
        else
        {
            potential_backup = (mask > 0) ? potential_grid->at(values)
                                          : potential_scatter->at(values);
            potential_local = potential_backup - CalcVshift(values);
            if (need_force)
            {
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        (mask > 0)
                            ? grid->at(values)[i]
                            : scatter->at(values)
                                  [i];  // -
                                        // potential_local*normal_force->at(values)[i];
                }
            }
        }
        /*if (do_borderwall)
        {
            vector<float> coordinate = potential_scatter->GetCoordinate(values);
            for (size_t i = 0; i < ndim; ++i)
            {
                border_upper[i] = coordinate[i] + 0.2 / cv_sigmas[i];
                border_lower[i] = coordinate[i] - 0.2 / cv_sigmas[i];
            }
        }*/
    }
    else if (usegrid)
    {
        if (subhill)
        {
            Axis vminus, vplus;
            for (size_t i = 0; i < ndim; ++i)
            {
                float lower = values[i] - cutoff[i];
                float upper = values[i] + cutoff[i] + 0.000001;
                if (periods[i] > 0)
                {
                    vminus.push_back(
                        lower);  // - round(lower / periods[i]) * periods[i]);
                    vplus.push_back(
                        upper);  // - round(lower / periods[i]) * periods[i]);
                }
                else
                {
                    vminus.push_back(std::fmax(lower, cv_mins[i]));
                    vplus.push_back(std::fmin(upper, cv_maxs[i]));
                }
            }
            Axis loop_flag = vminus;
            int index = 0;
            while (index >= 0)
            {
                //++sum_count;
                Gdata tder = hill.CalcHill(loop_flag);
                float factor = potential_grid->at(loop_flag);
                potential_backup += factor * hill.potential;
                // potential_local += (factor-new_max) * hill.potential;
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        Dpotential_local[i] -= (factor - new_max) * tder[i];
                    }
                }
                // another dimension!
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
            potential_backup = potential_grid->at(values);
            if (need_force)
            {
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        grid->at(values)
                            [i];  // -
                                  // potential_local*normal_force->at(values)[i];;
                }
            }
        }
        if (do_borderwall)
        {
            for (size_t i = 0; i < ndim; ++i)
            {
                border_upper[i] = cv_maxs[i] - cutoff[i];  // 1.0/cv_sigmas[i] ;
                border_lower[i] = cv_mins[i] + cutoff[i];  // 1.0/cv_sigmas[i] ;
            }
        }
    }
    if (need_potential)
    {
        potential_local = potential_backup - CalcVshift(values);
    }
    if (need_force)  // && !subhill)
    {
        if (subhill)
        {
            float f0 = new_max * normal_force->at(values)[0];
            if (convmeta)
            {
                new_max =
                    shift *
                    expf(-normal_lse->at(scatter->GetCoordinate(max_index)));
            }
            else
            {
                new_max = shift / normalforce_sum;
            }
            float f1 = new_max * sum_force[0];
            if (fabs(f0 - f1) > shift)
            {
                printf("The shift, kde & histogram:%f: %f vs %f\n", shift, f1,
                       f0);
            }
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * sum_force[i];
            }
        }
        else
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * normal_force->at(values)[i];
            }
        }
    }
    /* original meta without grid!!!!
    for (auto &hill_ : hills)
    {
        Gdata tder = hill_.CalcHill(values);
        if (need_force)
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += hill_.height * tder[i];
            }
        }
        if (need_potential)
        {
            potential_local += hill_.potential * hill_.height;
        }
    }
    potential_backup = potential_local;
    */
    return;
}
