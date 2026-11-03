#include "shake.h"

#include "velocity_projection.h"

template <class Real>
static __global__ void Constrain_Force_Cycle(
    const int constrain_pair_numbers, const ConstraintVector<Real>* crd,
    Boundary boundary, const CONSTRAIN_PAIR* constrain_pair,
    const ConstraintVector<Real>* pair_dr, ConstraintVector<Real>* test_frc)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        ConstraintPair<Real> cp = constrain_pair[pair_i];
        ConstraintVector<Real> dr0 = pair_dr[pair_i];
        ConstraintVector<Real> dr = Get_Displacement(
            crd[cp.atom_i_serial], crd[cp.atom_j_serial], boundary);
        Real frc_abs = 0.5 * (dr * dr - cp.constant_r * cp.constant_r) /
                       (dr * dr0) * cp.constrain_k;
        ConstraintVector<Real> frc_lin = frc_abs * dr0;

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}

template <class Real>
static __global__ void Refresh_Coordinate(
    const int atom_numbers, const ConstraintVector<Real>* crd,
    ConstraintVector<Real>* test_crd, const ConstraintVector<Real>* test_frc,
    const Real* mass_inverse, const Real x_factor)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        ConstraintVector<Real> crd_lin = crd[atom_i];
        ConstraintVector<Real> frc_lin = test_frc[atom_i];
        Real mass_lin = mass_inverse[atom_i];
        // mass_lin为mass的倒数，frc_lin已经乘以dt^2
        test_crd[atom_i] = crd_lin + x_factor * mass_lin * frc_lin;
    }
}

template <class Real>
static __global__ void Last_Crd_To_dr(const int constrain_pair_numbers,
                                      const ConstraintVector<Real>* atom_crd,
                                      Boundary boundary,
                                      const CONSTRAIN_PAIR* constrain_pair,
                                      ConstraintVector<Real>* pair_dr)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        ConstraintPair<Real> cp = constrain_pair[pair_i];
        pair_dr[pair_i] = Get_Displacement(
            atom_crd[cp.atom_i_serial], atom_crd[cp.atom_j_serial], boundary);
    }
}

template <class Real>
static __global__ void Refresh_Crd_Vel(
    const int atom_numbers, const Real dt_inverse, const Real dt,
    ConstraintVector<Real>* crd, ConstraintVector<Real>* vel,
    const ConstraintVector<Real>* test_frc, const Real* mass_inverse,
    const Real exp_gamma, const Real half_exp_gamma_plus_half)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        ConstraintVector<Real> crd_lin = crd[atom_i];
        ConstraintVector<Real> frc_lin = test_frc[atom_i];
        ConstraintVector<Real> vel_lin = vel[atom_i];
        Real mass_lin = mass_inverse[atom_i];

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

template <class Real>
void SHAKE_SOLVER<Real>::Initial_SHAKE(CONTROLLER* controller,
                                       CONSTRAIN* constrain,
                                       const char* module_name)
{
    // 从传入的参数复制基本信息
    this->controller = controller;
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
                                    "SHAKE_SOLVER<Real>::Initial_SHAKE");
            int scanf_ret = sscanf(
                controller[0].Command(this->module_name, "iteration_numbers"),
                "%d", &iteration_numbers);
        }
        controller[0].printf("    constrain iteration step is %d\n",
                             iteration_numbers);

        if (controller->Command_Exist(this->module_name, "early_stop"))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "SHAKE_SOLVER<Real>::Initial_SHAKE",
                "early_stop is no longer an input option; stopping is "
                "automatic, use tolerance");
        if (controller->Command_Exist(this->module_name, "tolerance"))
        {
            controller->Check_Float(this->module_name, "tolerance",
                                    "SHAKE_SOLVER<Real>::Initial_SHAKE");
            tolerance =
                atof(controller->Command(this->module_name, "tolerance"));
        }
        if (!(tolerance > 0 && tolerance < 1))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "SHAKE_SOLVER<Real>::Initial_SHAKE",
                "tolerance must be between zero and one");
        controller->printf("    small-group relative tolerance %.3g\n",
                           tolerance);

        step_length = Real(1.0);
        if (controller[0].Command_Exist(this->module_name, "step_length"))
        {
            controller->Check_Float(this->module_name, "step_length",
                                    "SHAKE_SOLVER<Real>::Initial_SHAKE");
            step_length = Real(
                atof(controller[0].Command(this->module_name, "step_length")));
        }
        controller[0].printf("    constrain step length is %.2f\n",
                             step_length);

        owns_buffers = true;
        Device_Malloc_Safely(
            (void**)&constrain_frc,
            sizeof(ConstraintVector<Real>) * constrain->atom_numbers);
        Device_Malloc_Safely((void**)&test_crd, sizeof(ConstraintVector<Real>) *
                                                    constrain->atom_numbers);
        Device_Malloc_Safely(
            (void**)&last_pair_dr,
            sizeof(ConstraintVector<Real>) * constrain->constrain_pair_numbers);
        Device_Malloc_Safely(
            (void**)&d_pair_virial,
            sizeof(ConstraintTensor<Real>) * constrain->constrain_pair_numbers);
        Device_Malloc_Safely((void**)&d_virial, sizeof(ConstraintTensor<Real>));

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

template <class Real>
void SHAKE_SOLVER<Real>::Remember_Last_Coordinates(
    const ConstraintVector<Real>* crd, Boundary boundary)
{
    if (is_initialized)
    {
        // 获得分子模拟迭代中上一步的距离信息
        Launch_Device_Kernel(
            Last_Crd_To_dr<Real>,
            (constrain->num_pair_local + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, constrain->num_pair_local,
            crd, boundary, constrain->constrain_pair_local, last_pair_dr);
    }
}

template <class Real>
static __device__ __host__ __forceinline__ bool
compute_velocity_constraint_correction_shake(
    const int atom_i, const int atom_j, const ConstraintVector<Real>* crd,
    Boundary boundary, const Real* mass_inverse,
    const ConstraintVector<Real>* vel, ConstraintVector<Real>* correction_i,
    ConstraintVector<Real>* correction_j, const Real relative_tolerance,
    bool* constraint_violated)
{
    if (constraint_violated != NULL) *constraint_violated = false;
    Real mass_i_inverse = mass_inverse[atom_i];
    Real mass_j_inverse = mass_inverse[atom_j];
    if (mass_i_inverse == Real(0.0) && mass_j_inverse == Real(0.0))
        return false;

    ConstraintVector<Real> dr =
        Get_Displacement(crd[atom_i], crd[atom_j], boundary);
    Real dr2 = dr * dr;
    if (dr2 < Real(1e-12))
    {
        if (constraint_violated != NULL) *constraint_violated = true;
        return false;
    }

    const ConstraintVector<Real> velocity_i = vel[atom_i];
    const ConstraintVector<Real> velocity_j = vel[atom_j];
    ConstraintVector<Real> v_diff = velocity_i - velocity_j;
    Real denom = (mass_i_inverse + mass_j_inverse) * dr2;
    if (denom < Real(1e-20))
    {
        if (constraint_violated != NULL) *constraint_violated = true;
        return false;
    }

    const Real projection = dr * v_diff;
    if (constraint_violated != NULL)
    {
        const Real tolerance = Velocity_Constraint_Residual_Tolerance(
            dr2, velocity_i, velocity_j, v_diff, relative_tolerance);
        *constraint_violated = fabs(projection) > tolerance;
    }
    Real lambda = projection / denom;
    correction_i[0] = (-mass_i_inverse * lambda) * dr;
    correction_j[0] = (mass_j_inverse * lambda) * dr;
    return true;
}

template <class Real>
static __global__ void project_velocity_to_shake_pairs(
    const int pair_numbers, const CONSTRAIN_PAIR* pairs,
    const ConstraintVector<Real>* crd, Boundary boundary,
    const Real* mass_inverse, const ConstraintVector<Real>* vel,
    ConstraintVector<Real>* delta_vel, const Real relative_tolerance,
    int* violation)
{
#ifdef USE_GPU
    int pair_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_i < pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < pair_numbers; pair_i++)
#endif
    {
        ConstraintPair<Real> cp = pairs[pair_i];
        ConstraintVector<Real> correction_i, correction_j;
        bool constraint_violated = false;
        if (compute_velocity_constraint_correction_shake<Real>(
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

template <class Real>
static __global__ void apply_shake_velocity_correction(
    const int local_atom_numbers, ConstraintVector<Real>* vel,
    ConstraintVector<Real>* crd, const ConstraintVector<Real>* delta_vel,
    const Real velocity_factor, const Real coordinate_factor)
{
#ifdef USE_GPU
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i < local_atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < local_atom_numbers; atom_i++)
#endif
    {
        ConstraintVector<Real> delta = velocity_factor * delta_vel[atom_i];
        vel[atom_i] = vel[atom_i] + delta;
        if (coordinate_factor != Real(0.0))
        {
            crd[atom_i] = crd[atom_i] + coordinate_factor * delta;
        }
    }
}

template <class Real>
bool SHAKE_SOLVER<Real>::Project_Velocity_To_Constraint_Manifold(
    ConstraintVector<Real>* vel, ConstraintVector<Real>* crd,
    const Real* mass_inverse, Boundary boundary, int local_atom_numbers,
    bool update_coordinates)
{
    if (!is_initialized || local_atom_numbers <= 0 ||
        constrain->num_pair_local <= 0)
        return true;

    constexpr int legacy_projection_iterations = 8;
    constexpr int maximum_velocity_only_iterations = 512;
    constexpr Real relative_tolerance = Real(1.0e-5);
    const int projection_iterations = update_coordinates
                                          ? legacy_projection_iterations
                                          : maximum_velocity_only_iterations;
    // A constraint row overlaps at most 2(d - 1) other rows, so the normalized
    // pair-overlap matrix has lambda_max <= 2d - 1.  Damping by 1/d therefore
    // keeps the parallel Richardson/Jacobi update stable.  The degree was
    // measured before SETTLE pairs were removed, so it is conservative here.
    const Real velocity_factor =
        update_coordinates
            ? Real(1.0)
            : Real(1.0) / static_cast<Real>(std::max(
                              1, constrain->maximum_constraint_degree));
    int* d_violation = NULL;
    if (!update_coordinates &&
        !Device_Malloc_Safely((void**)&d_violation, sizeof(int)))
        return false;
    bool converged = update_coordinates;
    for (int iter = 0; iter < projection_iterations; ++iter)
    {
        if (!update_coordinates) deviceMemset(d_violation, 0, sizeof(int));
        deviceMemset(constrain_frc, 0,
                     sizeof(ConstraintVector<Real>) * local_atom_numbers);
        Launch_Device_Kernel(
            project_velocity_to_shake_pairs<Real>,
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
            apply_shake_velocity_correction<Real>,
            (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, vel,
            crd, constrain_frc, velocity_factor,
            update_coordinates ? Real(0.5) * constrain->dt : Real(0.0));
    }
    if (d_violation != NULL) deviceFree(d_violation);
    return converged;
}

template <class Real>
static __global__ void Constrain_Force_Cycle_With_Virial(
    const int constrain_pair_numbers, const ConstraintVector<Real>* crd,
    Boundary boundary, const CONSTRAIN_PAIR* constrain_pair,
    const ConstraintVector<Real>* pair_dr, ConstraintVector<Real>* test_frc,
    ConstraintTensor<Real>* d_pair_virial)
{
#ifdef USE_GPU
    int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
#else
#pragma omp parallel for
    for (int pair_i = 0; pair_i < constrain_pair_numbers; pair_i++)
#endif
    {
        ConstraintPair<Real> cp = constrain_pair[pair_i];
        ConstraintVector<Real> dr0 = pair_dr[pair_i];
        ConstraintVector<Real> dr = Get_Displacement(
            crd[cp.atom_i_serial], crd[cp.atom_j_serial], boundary);
        Real r_1 = ConstraintReciprocalNorm(dr.x, dr.y, dr.z);
        Real frc_abs = 0.5 * (dr * dr - cp.constant_r * cp.constant_r) /
                       (dr * dr0) * cp.constrain_k;
        ConstraintVector<Real> frc_lin = frc_abs * dr0;
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

template <class Real>
static __global__ void Sum_Virial_Tensor_To_Stress(
    const int constrain_pair_numbers, ConstraintTensor<Real>* stress,
    const ConstraintTensor<Real>* pair_virial, const Real factor)
{
    ConstraintTensor<Real> virial_sum = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int i = blockDim.x * blockDim.y * blockIdx.x + blockDim.y * threadIdx.x +
            threadIdx.y;
    if (i < constrain_pair_numbers)
    {
        virial_sum = virial_sum + pair_virial[i];
    }
#else
    Real v11 = Real(0.0), v21 = Real(0.0), v22 = Real(0.0);
    Real v31 = Real(0.0), v32 = Real(0.0), v33 = Real(0.0);
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

template <class Real>
void SHAKE_SOLVER<Real>::Get_Local(int local_atoms)
{
    small_groups.Clear();
    if (!is_initialized || !use_small_groups) return;
    std::vector<CONSTRAIN_PAIR> pairs(constrain->num_pair_local);
    deviceMemcpy(pairs.data(), constrain->constrain_pair_local,
                 pairs.size() * sizeof(CONSTRAIN_PAIR),
                 deviceMemcpyDeviceToHost);
    small_groups.Build(controller, pairs, local_atoms);
}

// Keep the original simultaneous (Jacobi) force updates, but keep each small
// component's iteration state local instead of launching two kernels per round.
template <class Real>
static __global__ void Shake_Small_Groups(
    int count, const SMALL_CONSTRAINT_GROUP* groups,
    const CONSTRAIN_PAIR* pairs, const ConstraintVector<Real>* old,
    ConstraintVector<Real>* crd, ConstraintVector<Real>* vel, const Real* inv,
    Boundary boundary, int iterations, Real xfactor, Real velocity_factor,
    bool pressure, Real stress_factor, ConstraintTensor<Real>* stress,
    Real tolerance)
{
#ifdef USE_GPU
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
#else
#pragma omp parallel for
    for (int index = 0; index < count; ++index)
#endif
    {
        const auto g = groups[index];
        ConstraintVector<Real> base[4] = {}, force[4] = {}, trial[4] = {},
                               dr0[3] = {};
        Real inverse[4] = {}, target2[3] = {}, target[3] = {};
        Real weight[3] = {}, lambda[3] = {};
#pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            if (a < g.atom_count)
            {
                base[a] = crd[g.atoms[a]];
                inverse[a] = inv[g.atoms[a]];
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            if (j < g.pair_count)
            {
                ConstraintPair<Real> p = pairs[g.pairs[j]];
                dr0[j] = old[g.pairs[j]];
                target2[j] = p.constant_r * p.constant_r;
                target[j] = p.constant_r;
                weight[j] = p.constrain_k;
            }
        }
        bool stopped = false;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
#pragma unroll
            for (int a = 0; a < 4; ++a)
                trial[a] = base[a] + (xfactor * inverse[a]) * force[a];
            bool converged = true;
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (j < g.pair_count)
                {
                    ConstraintVector<Real> d = Get_Displacement(
                        Small_Group_Vector(trial, g.a[j]),
                        Small_Group_Vector(trial, g.b[j]), boundary);
                    converged = converged && Constraint_Within_Tolerance(
                                                 d, target[j], tolerance);
                }
            if (converged)
            {
                stopped = true;
                break;
            }
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                if (j < g.pair_count)
                {
                    ConstraintVector<Real> dr = Get_Displacement(
                        Small_Group_Vector(trial, g.a[j]),
                        Small_Group_Vector(trial, g.b[j]), boundary);
                    Real amount = 0.5 * (dr * dr - target2[j]) / (dr * dr0[j]) *
                                  weight[j];
                    ConstraintVector<Real> f = amount * dr0[j];
#pragma unroll
                    for (int a = 0; a < 4; ++a)
                    {
                        if (a == g.a[j]) force[a] = force[a] - f;
                        if (a == g.b[j]) force[a] = force[a] + f;
                    }
                    if (pressure) lambda[j] += amount;
                }
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            if (a < g.atom_count)
            {
                ConstraintVector<Real> correction = inverse[a] * force[a];
                crd[g.atoms[a]] =
                    stopped ? trial[a] : base[a] + xfactor * correction;
                vel[g.atoms[a]] =
                    vel[g.atoms[a]] + velocity_factor * correction;
            }
        }
        if (pressure)
        {
            ConstraintTensor<Real> value = {0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (j < g.pair_count)
                    value =
                        value - (stress_factor * lambda[j]) *
                                    Get_Virial_From_Force_Dis(dr0[j], dr0[j]);
            atomicAdd(stress, value);
        }
    }
}

template <class Real>
void SHAKE_SOLVER<Real>::Constrain(int atom_numbers,
                                   ConstraintVector<Real>* crd,
                                   ConstraintVector<Real>* vel,
                                   const Real* mass_inverse, const Real* d_mass,
                                   Boundary boundary, int need_pressure,
                                   ConstraintTensor<Real>* d_stress)
{
    if (is_initialized)
    {
        if (use_small_groups && small_groups.count)
        {
            Launch_Device_Kernel(
                Shake_Small_Groups<Real>, (small_groups.count + 127) / 128, 128,
                0, 0, small_groups.count, small_groups.data,
                constrain->constrain_pair_local, last_pair_dr, crd, vel,
                mass_inverse, boundary, iteration_numbers, constrain->x_factor,
                constrain->v_factor * constrain->dt_inverse, need_pressure > 0,
                constrain->dt_inverse * constrain->dt_inverse *
                    boundary.rcell.a11 * boundary.rcell.a22 *
                    boundary.rcell.a33,
                d_stress, tolerance);
            return;
        }
        // 清空约束力和维里
        deviceMemset(constrain_frc, 0,
                     sizeof(ConstraintVector<Real>) * atom_numbers);
        if (need_pressure > 0)
        {
            deviceMemset(
                d_pair_virial, 0,
                sizeof(ConstraintTensor<Real>) * constrain->num_pair_local);
            deviceMemset(d_virial, 0, sizeof(ConstraintTensor<Real>));
        }
        for (int i = 0; i < iteration_numbers; i = i + 1)
        {
            Launch_Device_Kernel(
                Refresh_Coordinate<Real>,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd,
                test_crd, constrain_frc, mass_inverse, constrain->x_factor);

            if (need_pressure > 0)
            {
                Launch_Device_Kernel(Constrain_Force_Cycle_With_Virial<Real>,
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
                Launch_Device_Kernel(Constrain_Force_Cycle<Real>,
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
            Launch_Device_Kernel(Sum_Virial_Tensor_To_Stress<Real>,
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
            Refresh_Crd_Vel<Real>,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            constrain->dt_inverse, constrain->dt, crd, vel, constrain_frc,
            mass_inverse, constrain->v_factor, constrain->x_factor);
    }
}

template struct SHAKE_SOLVER<float>;
template struct SHAKE_SOLVER<double>;

void SHAKE::Initial_SHAKE(CONTROLLER* controller, CONSTRAIN* constrain,
                          const char* name)
{
    backend.Select(constrain->precision);
    backend.Visit([&](auto& s, auto&)
                  { s.Initial_SHAKE(controller, constrain, name); });
}

void SHAKE::Get_Local(int n)
{
    local_atoms = n;
    backend.Visit([&](auto& s, auto&) { s.Get_Local(n); });
}

void SHAKE::Constrain(int n, VECTOR* crd, VECTOR* vel, const float* inv,
                      const float* mass, Boundary boundary, int pressure,
                      LTMatrix3* stress)
{
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !n) return;
            work.Begin(n, crd, vel, mass, inv);
            s.Constrain(n, work.q, work.v, work.inverse, work.mass, boundary,
                        pressure, work.stress);
            work.End(n, crd, vel, pressure ? stress : nullptr);
        });
}

bool SHAKE::Project_Velocity_To_Constraint_Manifold(VECTOR* vel, VECTOR* crd,
                                                    const float* inv,
                                                    Boundary boundary, int n,
                                                    bool update_coordinates)
{
    bool result = true;
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !n) return;
            work.Begin(n, crd, vel, nullptr, inv);
            result = s.Project_Velocity_To_Constraint_Manifold(
                work.v, work.q, work.inverse, boundary, n, update_coordinates);
            work.End(n, crd, vel, nullptr, update_coordinates);
        });
    return result;
}

void SHAKE::Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary)
{
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !local_atoms) return;
            work.Coordinates(local_atoms, crd);
            s.Remember_Last_Coordinates(work.q, boundary);
        });
}
