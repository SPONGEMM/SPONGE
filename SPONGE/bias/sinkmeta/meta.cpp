#include "meta.h"

META::Hill::Hill(const Axis& centers, const Axis& inv_w, const Axis& period,
                 const float& theight)
    : height(theight)
{
    for (int i = 0; i < centers.size(); ++i)
    {
        gsf.push_back(GaussianSF(centers[i], inv_w[i], period[i]));
    }
}

META::Gdata META::Hill::CalcHill(const Axis& values)
{
    const size_t& n = values.size();
    Axis dx(n, 0.0), df(n, 1.0);
    for (size_t i = 0; i < n; ++i)
    {
        GaussianSF g = gsf[i];
        dx[i] = g.Evaluate(values[i], df[i]);
    }

    Gdata tder(n, 1.0);
    potential = 1.0;
    for (size_t i = 0; i < n; ++i)
    {
        potential *= dx[i];
        for (size_t j = 0; j < n; ++j)
        {
            if (j != i)
            {
                tder[i] *= dx[j];
            }
            else
            {
                tder[i] *= df[j];
            }
        }
    }
    return tder;
}

#ifdef USE_GPU
static __global__ void Add_Frc(const int atom_numbers, VECTOR* frc,
                               VECTOR* cv_grad, float dheight_dcv)
{
    for (int i = blockIdx.x + blockDim.x * threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static __global__ void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static __global__ void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                                  const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#else
static void Add_Frc(const int atom_numbers, VECTOR* frc, VECTOR* cv_grad,
                    float dheight_dcv)
{
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                       const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#endif

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
    Potential_and_derivative(need_potential);
    if (do_borderwall)
    {
        Border_derivative(border_upper.data(), border_lower.data(), cutoff,
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
        Launch_Device_Kernel(Add_Potential, 1, 1, 0, NULL, d_potential,
                             potential_local);
    }
}

void META::Potential_and_derivative(const int need_potential)
{
    if (!is_initialized)
    {
        return;
    }
    Axis values;
    for (int i = 0; i < cvs.size(); ++i)
    {
        values.push_back(cvs[i]->value);
        Dpotential_local[i] = 0.f;
    }
    Estimate(values, need_potential, true);
}

void META::Border_derivative(float* border_upper, float* border_lower,
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
        AddPotential(sys_temp, step);
    }
}
