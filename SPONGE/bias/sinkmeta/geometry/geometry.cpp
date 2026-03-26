#include "../meta.h"

std::vector<float> normalize(const std::vector<float>& v)
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
    vector<float> new_v;
    for (int i = 0; i < v.size(); ++i)
    {
        new_v.push_back(v[i] / sqrt(norm));
    }
    return new_v;
}

std::vector<float> crossProduct(const std::vector<float>& a,
                                const std::vector<float>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

float determinant(const std::vector<std::vector<float>>& matrix)
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

META::Axis META::RotateVector(const Axis& tang_vector, bool do_debug)
{
    if (do_debug)
    {
        printf("\nRotate Matrix:\n(%f", tang_vector[0]);
        for (int i = 1; i < ndim; ++i)
        {
            printf(" %f", tang_vector[i]);
        }
        printf(")\n");
    }
    vector<float> normal_vector;
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
    if (do_debug) printf("(");
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
        if (do_debug)
        {
            float jbi = jb[i];
            printf(" %f", jbi);
        }
    }
    if (ndim == 2)
    {
        vector<vector<float>> determinant_v = vector<Axis>{tang_vector, jb};
        float sign = determinant(determinant_v);
        return Axis{jb[0] * sign, jb[1] * sign};
    }
    return jb;
}

void META::Cartesian2Path(const Axis& Cartesian_values, Axis& Path_values)
{
    double cumulative_s = 0.0;
    bool do_debug = false;
    Axis values, neighbor;
    Axis tang_vector(ndim, 0.);
    int index = scatter->GetIndex(Cartesian_values);
    if (index < scatter_size - 1)
    {
        values = scatter->GetCoordinate(index);
        neighbor = scatter->GetCoordinate(index + 1);
    }
    else
    {
        values = scatter->GetCoordinate(index - 1);
        neighbor = scatter->GetCoordinate(index);
    }
    TangVector(tang_vector, values, neighbor);
    double projected_last =
        ProjectToPath(tang_vector, neighbor, Cartesian_values);
    Path_values.push_back(cumulative_s + projected_last);
    Axis normal_vector = RotateVector(tang_vector, do_debug);
    Path_values.push_back(
        ProjectToPath(normal_vector, values, Cartesian_values));
    if (ndim == 3)
    {
        Axis binormal_vector =
            normalize(crossProduct(tang_vector, normal_vector));
        Path_values.push_back(
            ProjectToPath(binormal_vector, values, Cartesian_values));
    }
}

float META::ProjectToPath(const Gdata& tang_vector, const Axis& values,
                          const Axis& Cartesian)
{
    float projected_s = 0.;
    for (int i = 0; i < ndim; ++i)
    {
        projected_s += (Cartesian[i] - values[i]) * tang_vector[i];
    }
    return projected_s;
}

double META::TangVector(Gdata& tang_vector, const Axis& values,
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
