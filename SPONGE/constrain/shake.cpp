#include "shake.h"

#include "velocity_projection.h"

static __global__ void Constrain_Force_Cycle(
    const int constrain_pair_numbers, const VECTOR* crd, Boundary boundary,
    const CONSTRAIN_PAIR* constrain_pair, const VECTOR* pair_dr,
    VECTOR* test_frc)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr0 = pair_dr[pair_i];
        VECTOR dr = Get_Displacement(crd[cp.atom_i_serial],
                                     crd[cp.atom_j_serial], boundary);
        float r_1 = rnorm3df(dr.x, dr.y, dr.z);
        float frc_abs = 0.5 * (dr * dr - cp.constant_r * cp.constant_r) /
                        (dr * dr0) * cp.constrain_k;
        VECTOR frc_lin = frc_abs * dr0;

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}

static __global__ void Refresh_Coordinate(const int atom_numbers,
                                          const VECTOR* crd, VECTOR* test_crd,
                                          const VECTOR* test_frc,
                                          const float* mass_inverse,
                                          const float x_factor)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        float mass_lin = mass_inverse[atom_i];
        // mass_lin为mass的倒数，frc_lin已经乘以dt^2
        test_crd[atom_i] = crd_lin + x_factor * mass_lin * frc_lin;
    }
}

static __global__ void Last_Crd_To_dr(const int constrain_pair_numbers,
                                      const VECTOR* atom_crd, Boundary boundary,
                                      const CONSTRAIN_PAIR* constrain_pair,
                                      VECTOR* pair_dr)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        pair_dr[pair_i] = Get_Displacement(
            atom_crd[cp.atom_i_serial], atom_crd[cp.atom_j_serial], boundary);
    }
}

static __global__ void Refresh_Crd_Vel(
    const int atom_numbers, const float dt_inverse, const float dt, VECTOR* crd,
    VECTOR* vel, const VECTOR* test_frc, const float* mass_inverse,
    const float exp_gamma, const float half_exp_gamma_plus_half)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        VECTOR vel_lin = vel[atom_i];
        float mass_lin = mass_inverse[atom_i];

        frc_lin.x = frc_lin.x * mass_lin;
        frc_lin.y = frc_lin.y * mass_lin;
        frc_lin.z =
            frc_lin.z * mass_lin;  // mass实际为mass的倒数，frc_lin已经乘以dt^2

        crd_lin.x = crd_lin.x + half_exp_gamma_plus_half * frc_lin.x;
        crd_lin.y = crd_lin.y + half_exp_gamma_plus_half * frc_lin.y;
        crd_lin.z = crd_lin.z + half_exp_gamma_plus_half * frc_lin.z;

        vel_lin.x = (vel_lin.x + exp_gamma * frc_lin.x * dt_inverse);
        vel_lin.y = (vel_lin.y + exp_gamma * frc_lin.y * dt_inverse);
        vel_lin.z = (vel_lin.z + exp_gamma * frc_lin.z * dt_inverse);

        crd[atom_i] = crd_lin;
        vel[atom_i] = vel_lin;
    }
}

void SHAKE::Initial_SHAKE(CONTROLLER* controller, CONSTRAIN* constrain,
                          const char* module_name)
{
    // 从传入的参数复制基本信息
    this->constrain = constrain;
    if (module_name == NULL)
    {
        strcpy(this->module_name, "SHAKE");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (constrain->constrain_pair_numbers > 0)
    {
        controller[0].printf("START INITIALIZING SHAKE:\n");
        iteration_numbers = 25;
        if (controller[0].Command_Exist(this->module_name, "iteration_numbers"))
        {
            controller->Check_Float(this->module_name, "iteration_numbers",
                                    "SHAKE::Initial_SHAKE");
            int scanf_ret = sscanf(
                controller[0].Command(this->module_name, "iteration_numbers"),
                "%d", &iteration_numbers);
        }
        controller[0].printf("    constrain iteration step is %d\n",
                             iteration_numbers);

        step_length = 1.0f;
        if (controller[0].Command_Exist(this->module_name, "step_length"))
        {
            controller->Check_Float(this->module_name, "step_length",
                                    "SHAKE::Initial_SHAKE");
            int scanf_ret =
                sscanf(controller[0].Command(this->module_name, "step_length"),
                       "%f", &step_length);
        }
        controller[0].printf("    constrain step length is %.2f\n",
                             step_length);

        Device_Malloc_Safely((void**)&constrain_frc,
                             sizeof(VECTOR) * constrain->atom_numbers);
        Device_Malloc_Safely((void**)&test_crd,
                             sizeof(VECTOR) * constrain->atom_numbers);
        Device_Malloc_Safely(
            (void**)&last_pair_dr,
            sizeof(VECTOR) * constrain->constrain_pair_numbers);
        Device_Malloc_Safely(
            (void**)&d_pair_virial,
            sizeof(LTMatrix3) * constrain->constrain_pair_numbers);
        Device_Malloc_Safely((void**)&d_virial, sizeof(LTMatrix3));

        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller[0].printf("    structure last modify date is %d\n",
                                 last_modify_date);
        }
        controller[0].printf("END INITIALIZING SHAKE\n\n");
        is_initialized = 1;
    }
    else
    {
        controller[0].printf("SHAKE IS NOT INITIALIZED\n\n");
    }
}

void SHAKE::Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary)
{
    if (is_initialized)
    {
        // 获得分子模拟迭代中上一步的距离信息
        Launch_Device_Kernel(
            Last_Crd_To_dr,
            (constrain->num_pair_local + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, constrain->num_pair_local,
            crd, boundary, constrain->constrain_pair_local, last_pair_dr);
    }
}

static __device__ __host__ __forceinline__ bool
compute_velocity_constraint_correction_shake(
    const int atom_i, const int atom_j, const VECTOR* crd, Boundary boundary,
    const float* mass_inverse, const VECTOR* vel, VECTOR* correction_i,
    VECTOR* correction_j, const float relative_tolerance,
    bool* constraint_violated)
{
    if (constraint_violated != NULL) *constraint_violated = false;
    float mass_i_inverse = mass_inverse[atom_i];
    float mass_j_inverse = mass_inverse[atom_j];
    if (mass_i_inverse == 0.0f && mass_j_inverse == 0.0f) return false;

    VECTOR dr = Get_Displacement(crd[atom_i], crd[atom_j], boundary);
    float dr2 = dr * dr;
    if (dr2 < 1e-12f)
    {
        if (constraint_violated != NULL) *constraint_violated = true;
        return false;
    }

    const VECTOR velocity_i = vel[atom_i];
    const VECTOR velocity_j = vel[atom_j];
    VECTOR v_diff = velocity_i - velocity_j;
    float denom = (mass_i_inverse + mass_j_inverse) * dr2;
    if (denom < 1e-20f)
    {
        if (constraint_violated != NULL) *constraint_violated = true;
        return false;
    }

    const float projection = dr * v_diff;
    if (constraint_violated != NULL)
    {
        const float tolerance = Velocity_Constraint_Residual_Tolerance(
            dr2, velocity_i, velocity_j, v_diff, relative_tolerance);
        *constraint_violated = fabsf(projection) > tolerance;
    }
    float lambda = projection / denom;
    correction_i[0] = (-mass_i_inverse * lambda) * dr;
    correction_j[0] = (mass_j_inverse * lambda) * dr;
    return true;
}

static __global__ void project_velocity_to_shake_pairs(
    const int pair_numbers, const CONSTRAIN_PAIR* pairs, const VECTOR* crd,
    Boundary boundary, const float* mass_inverse, const VECTOR* vel,
    VECTOR* delta_vel, const float relative_tolerance, int* violation)
{
#ifdef USE_GPU
    int pair_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_i < pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < pair_numbers; pair_i++)
#endif
    {
        CONSTRAIN_PAIR cp = pairs[pair_i];
        VECTOR correction_i, correction_j;
        bool constraint_violated = false;
        if (compute_velocity_constraint_correction_shake(
                cp.atom_i_serial, cp.atom_j_serial, crd, boundary, mass_inverse,
                vel, &correction_i, &correction_j, relative_tolerance,
                violation != NULL ? &constraint_violated : NULL))
        {
            atomicAdd(&delta_vel[cp.atom_i_serial].x, correction_i.x);
            atomicAdd(&delta_vel[cp.atom_i_serial].y, correction_i.y);
            atomicAdd(&delta_vel[cp.atom_i_serial].z, correction_i.z);
            atomicAdd(&delta_vel[cp.atom_j_serial].x, correction_j.x);
            atomicAdd(&delta_vel[cp.atom_j_serial].y, correction_j.y);
            atomicAdd(&delta_vel[cp.atom_j_serial].z, correction_j.z);
        }
        if (constraint_violated && violation != NULL) atomicExch(violation, 1);
    }
}

static __global__ void apply_shake_velocity_correction(
    const int local_atom_numbers, VECTOR* vel, VECTOR* crd,
    const VECTOR* delta_vel, const float velocity_factor,
    const float coordinate_factor)
{
#ifdef USE_GPU
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i < local_atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < local_atom_numbers; atom_i++)
#endif
    {
        VECTOR delta = velocity_factor * delta_vel[atom_i];
        vel[atom_i] = vel[atom_i] + delta;
        if (coordinate_factor != 0.0f)
        {
            crd[atom_i] = crd[atom_i] + coordinate_factor * delta;
        }
    }
}

bool SHAKE::Project_Velocity_To_Constraint_Manifold(VECTOR* vel, VECTOR* crd,
                                                    const float* mass_inverse,
                                                    Boundary boundary,
                                                    int local_atom_numbers,
                                                    bool update_coordinates)
{
    if (!is_initialized || local_atom_numbers <= 0 ||
        constrain->num_pair_local <= 0)
        return true;

    constexpr int legacy_projection_iterations = 8;
    constexpr int maximum_velocity_only_iterations = 512;
    constexpr float relative_tolerance = 1.0e-5f;
    const int projection_iterations = update_coordinates
                                          ? legacy_projection_iterations
                                          : maximum_velocity_only_iterations;
    // A constraint row overlaps at most 2(d - 1) other rows, so the normalized
    // pair-overlap matrix has lambda_max <= 2d - 1.  Damping by 1/d therefore
    // keeps the parallel Richardson/Jacobi update stable.  The degree was
    // measured before SETTLE pairs were removed, so it is conservative here.
    const float velocity_factor =
        update_coordinates
            ? 1.0f
            : 1.0f / static_cast<float>(
                         std::max(1, constrain->maximum_constraint_degree));
    int* d_violation = NULL;
    if (!update_coordinates &&
        !Device_Malloc_Safely((void**)&d_violation, sizeof(int)))
        return false;
    bool converged = update_coordinates;
    for (int iter = 0; iter < projection_iterations; ++iter)
    {
        if (!update_coordinates) deviceMemset(d_violation, 0, sizeof(int));
        deviceMemset(constrain_frc, 0, sizeof(VECTOR) * local_atom_numbers);
        Launch_Device_Kernel(
            project_velocity_to_shake_pairs,
            (constrain->num_pair_local + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, constrain->num_pair_local,
            constrain->constrain_pair_local, crd, boundary, mass_inverse, vel,
            constrain_frc, relative_tolerance, d_violation);
        if (!update_coordinates)
        {
            int violation = 0;
            deviceMemcpy(&violation, d_violation, sizeof(int),
                         deviceMemcpyDeviceToHost);
            if (violation == 0)
            {
                converged = true;
                break;
            }
        }
        Launch_Device_Kernel(
            apply_shake_velocity_correction,
            (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, vel,
            crd, constrain_frc, velocity_factor,
            update_coordinates ? 0.5f * constrain->dt : 0.0f);
    }
    if (d_violation != NULL) deviceFree(d_violation);
    return converged;
}

static __global__ void Constrain_Force_Cycle_With_Virial(
    const int constrain_pair_numbers, const VECTOR* crd, Boundary boundary,
    const CONSTRAIN_PAIR* constrain_pair, const VECTOR* pair_dr,
    VECTOR* test_frc, LTMatrix3* d_pair_virial)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr0 = pair_dr[pair_i];
        VECTOR dr = Get_Displacement(crd[cp.atom_i_serial],
                                     crd[cp.atom_j_serial], boundary);
        float r_1 = rnorm3df(dr.x, dr.y, dr.z);
        float frc_abs = 0.5 * (dr * dr - cp.constant_r * cp.constant_r) /
                        (dr * dr0) * cp.constrain_k;
        VECTOR frc_lin = frc_abs * dr0;
        d_pair_virial[pair_i] =
            d_pair_virial[pair_i] - Get_Virial_From_Force_Dis(frc_lin, dr0);

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}

static __global__ void Sum_Virial_Tensor_To_Stress(
    const int constrain_pair_numbers, LTMatrix3* stress,
    const LTMatrix3* pair_virial, const float factor)
{
    LTMatrix3 virial_sum = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int i = blockDim.x * blockDim.y * blockIdx.x + blockDim.y * threadIdx.x +
            threadIdx.y;
    if (i < constrain_pair_numbers)
    {
        virial_sum = virial_sum + pair_virial[i];
    }
#else
    float v11 = 0.0f, v21 = 0.0f, v22 = 0.0f;
    float v31 = 0.0f, v32 = 0.0f, v33 = 0.0f;
#pragma omp parallel for reduction(+ : v11, v21, v22, v31, v32, v33)
    for (int i = 0; i < constrain_pair_numbers; i++)
    {
        v11 += pair_virial[i].a11;
        v21 += pair_virial[i].a21;
        v22 += pair_virial[i].a22;
        v31 += pair_virial[i].a31;
        v32 += pair_virial[i].a32;
        v33 += pair_virial[i].a33;
    }
    virial_sum = {v11, v21, v22, v31, v32, v33};
#endif
    virial_sum = factor * virial_sum;
    Warp_Sum_To(stress, virial_sum, warpSize);
}

void SHAKE::Constrain(int atom_numbers, VECTOR* crd, VECTOR* vel,
                      const float* mass_inverse, const float* d_mass,
                      Boundary boundary, int need_pressure, LTMatrix3* d_stress)
{
    if (is_initialized)
    {
        // 清空约束力和维里
        deviceMemset(constrain_frc, 0, sizeof(VECTOR) * atom_numbers);
        if (need_pressure > 0)
        {
            deviceMemset(d_pair_virial, 0,
                         sizeof(LTMatrix3) * constrain->num_pair_local);
            deviceMemset(d_virial, 0, sizeof(LTMatrix3));
        }
        for (int i = 0; i < iteration_numbers; i = i + 1)
        {
            Launch_Device_Kernel(
                Refresh_Coordinate,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd,
                test_crd, constrain_frc, mass_inverse, constrain->x_factor);

            if (need_pressure > 0)
            {
                Launch_Device_Kernel(Constrain_Force_Cycle_With_Virial,
                                     (constrain->num_pair_local +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     constrain->num_pair_local, test_crd,
                                     boundary, constrain->constrain_pair_local,
                                     last_pair_dr, constrain_frc,
                                     d_pair_virial);
            }
            else
            {
                Launch_Device_Kernel(Constrain_Force_Cycle,
                                     (constrain->num_pair_local +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     constrain->num_pair_local, test_crd,
                                     boundary, constrain->constrain_pair_local,
                                     last_pair_dr, constrain_frc);
            }
        }

        if (need_pressure > 0)
        {
            dim3 blockSize = {
                CONTROLLER::device_warp,
                CONTROLLER::device_max_thread / CONTROLLER::device_warp};
            Launch_Device_Kernel(Sum_Virial_Tensor_To_Stress,
                                 (constrain->num_pair_local +
                                  CONTROLLER::device_max_thread - 1) /
                                     CONTROLLER::device_max_thread,
                                 blockSize, 0, NULL, constrain->num_pair_local,
                                 d_stress, d_pair_virial,
                                 1 / constrain->dt / constrain->dt *
                                     boundary.rcell.a11 * boundary.rcell.a22 *
                                     boundary.rcell.a33);
        }

        Launch_Device_Kernel(
            Refresh_Crd_Vel,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            constrain->dt_inverse, constrain->dt, crd, vel, constrain_frc,
            mass_inverse, constrain->v_factor, constrain->x_factor);
    }
}
