#include "edip.h"

#include "../utils/h5md/topology_manybody_h5_materializer.hpp"

void EDIP_INFORMATION::Initial(CONTROLLER* controller, const char* module_name,
                               bool* need_full_nl_flag)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "EDIP");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!controller->Command_Exist(this->module_name, "in_file") &&
        controller->Command_Exist("input_h5_topology_path"))
    {
        SpongeH5MD::TopologyManybodyH5Materializer reader;
        if (!reader.Open(controller->Command("input_h5_topology_path")))
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "EDIP_INFORMATION::Initial",
                                           reader.Last_Error().c_str());
        }
        if (reader.Has_EDIP())
        {
            SpongeH5MD::NativeEDIPDefinition definition;
            if (!reader.Read_EDIP(&definition))
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "EDIP_INFORMATION::Initial",
                                               reader.Last_Error().c_str());
            }

            controller->printf(
                "START INITIALIZING EDIP FORCE FROM NATIVE H5\n");
            atom_numbers = static_cast<int>(definition.atom_type.size());
            atom_type_numbers = definition.atom_type_count;
            pair_type_numbers = atom_type_numbers * atom_type_numbers;
            triple_type_numbers =
                atom_type_numbers * atom_type_numbers * atom_type_numbers;
            Malloc_Safely((void**)&h_energy_atom,
                          sizeof(float) * (atom_numbers + 1));
            Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                          sizeof(float) * (atom_numbers + 1));
            d_energy_atom = d_energy_sum + 1;
            Device_Malloc_Safely((void**)&z, sizeof(float) * atom_numbers * 2);
            dE_dz = z + atom_numbers;
            const std::size_t parameter_count =
                static_cast<std::size_t>(pair_type_numbers) * 8 +
                static_cast<std::size_t>(triple_type_numbers) * 9;
            Malloc_Safely((void**)&h_parameters,
                          sizeof(float) * parameter_count);
            Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
            std::copy(definition.pair_parameters.begin(),
                      definition.pair_parameters.end(), h_parameters);
            std::copy(definition.triple_parameters.begin(),
                      definition.triple_parameters.end(),
                      h_parameters + pair_type_numbers * 8);
            std::copy(definition.atom_type.begin(), definition.atom_type.end(),
                      h_atom_type);
            Device_Malloc_And_Copy_Safely((void**)&d_parameters, h_parameters,
                                          sizeof(float) * parameter_count);
            Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                          sizeof(int) * atom_numbers);
            if (need_full_nl_flag != NULL)
            {
                *need_full_nl_flag = true;
                controller->printf(
                    "EDIP requires full neighbor list for three-body "
                    "calculations.\n");
            }
            is_initialized = true;
            if (!is_controller_printf_initialized)
            {
                controller->Step_Print_Initial(this->module_name, "%14.7e");
                is_controller_printf_initialized = true;
            }
            controller->printf("END INITIALIZING EDIP FORCE\n\n");
            return;
        }
    }
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("EDIP FORCE IS NOT INITIALIZED\n\n");
        return;
    }
    controller->printf("START INITIALIZING EDIP FORCE\n");
    FILE* fp;
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d %d\n", &atom_numbers, &atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe number of atoms and edip types can not be found\n");
    }
    pair_type_numbers = atom_type_numbers * atom_type_numbers;
    triple_type_numbers =
        atom_type_numbers * atom_type_numbers * atom_type_numbers;

    Malloc_Safely((void**)&h_energy_atom, sizeof(float) * (atom_numbers + 1));

    Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                  sizeof(float) * (atom_numbers + 1));
    d_energy_atom = d_energy_sum + 1;
    Device_Malloc_Safely((void**)&z, sizeof(float) * atom_numbers * 2);
    dE_dz = z + atom_numbers;
    Malloc_Safely(
        (void**)&h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 9));
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    char temp[CHAR_LENGTH_MAX];
    std::map<int, bool> unrecorded;
    int type_a, type_b, type_c;
    float a, c, alpha, A, B, rho, beta, sigma, eta, gamma, lambda, Q0, mu, u1,
        u2, u3, u4;
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe first comment line can not be found\n");
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %f %f %f %f %f %f %f %f\n", &type_a, &type_b,
                   &alpha, &c, &a, &A, &B, &rho, &beta, &sigma) != 10)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers + type_b;
        if (index >= pair_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * index + 0] = alpha;
        h_parameters[8 * index + 1] = c;
        h_parameters[8 * index + 2] = a;
        h_parameters[8 * index + 3] = A;
        h_parameters[8 * index + 4] = B;
        h_parameters[8 * index + 5] = rho;
        h_parameters[8 * index + 6] = beta;
        h_parameters[8 * index + 7] = sigma;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
    }
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe second comment line can not be found\n");
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f %f %f %f %f %f %f %f\n", &type_a,
                   &type_b, &type_c, &eta, &gamma, &lambda, &Q0, &mu, &u1, &u2,
                   &u3, &u4) != 12)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers * atom_type_numbers +
                    type_b * atom_type_numbers + type_c;
        if (index >= triple_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * pair_type_numbers + 9 * index + 0] = eta;
        h_parameters[8 * pair_type_numbers + 9 * index + 1] = gamma;
        h_parameters[8 * pair_type_numbers + 9 * index + 2] = lambda;
        h_parameters[8 * pair_type_numbers + 9 * index + 3] = Q0;
        h_parameters[8 * pair_type_numbers + 9 * index + 4] = mu;
        h_parameters[8 * pair_type_numbers + 9 * index + 5] = u1;
        h_parameters[8 * pair_type_numbers + 9 * index + 6] = u2;
        h_parameters[8 * pair_type_numbers + 9 * index + 7] = u3;
        h_parameters[8 * pair_type_numbers + 9 * index + 8] = u4;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
    }
    Device_Malloc_And_Copy_Safely(
        (void**)&d_parameters, h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 9));
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe third comment line can not be found\n");
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", h_atom_type + i) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome atom types can not be found\n");
        }
        if (h_atom_type[i] >= atom_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome atom types are not right\n");
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    // 设置需要全连接近邻表
    if (need_full_nl_flag != NULL)
    {
        *need_full_nl_flag = true;
        controller->printf(
            "EDIP requires full neighbor list for three-body calculations.\n");
    }

    is_initialized = true;
    if (!is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%14.7e");
        is_controller_printf_initialized = true;
    }
    controller->printf("END INITIALIZING EDIP FORCE\n\n");
}

template <bool need_energy, bool need_virial>
static __global__
    __launch_bounds__(1024) void EDIP_Force_With_Full_Neighbor_CUDA(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, float* z, float* dE_dz, const ATOM_GROUP* nl,
        float* atom_energy, LTMatrix3* atom_virial, int* atom_types,
        float* parameters, const int atom_type_numbers,
        const int pair_type_numbers, float* this_energy)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j, atom_k;
    if (atom_i < atom_numbers)
#else
    int atom_j, atom_k;
#pragma omp parallel for firstprivate(atom_j, atom_k)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j, type_k;
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR ri = crd[atom_i], rj, rk, drij, drik, drjk;
        float rij, rik, rjk;
        float A, B, sigma, a1, a2, a3, rho, beta, eta, gamma, lambda, Q0, mu,
            u1, u2, u3, u4;
        int pair_index_1, pair_index_2, triple_index;
        float local_energy = 0, zi = z[atom_i], zj, zk, dE_dzi = 0, dE_dzj,
              dE_dzk;
        VECTOR i_force = {0.0f, 0.0f, 0.0f}, j_force, k_force;
        VECTOR temp1_force, temp2_force;
        LTMatrix3 local_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            j_force = {0.0f, 0.0f, 0.0f};
            dE_dzj = 0;
            atom_j = nl_i.atom_serial[j];
            zj = z[atom_j];
            type_j = atom_types[atom_j];
            rj = crd[atom_j];
            drij = Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            rij = norm3df(drij.x, drij.y, drij.z);
            pair_index_1 = type_i * atom_type_numbers + type_j;
            a1 = parameters[8 * pair_index_1 + 2];
            A = parameters[8 * pair_index_1 + 3];
            B = parameters[8 * pair_index_1 + 4];
            rho = parameters[8 * pair_index_1 + 5];
            beta = parameters[8 * pair_index_1 + 6];
            sigma = parameters[8 * pair_index_1 + 7];

            bool should_compute_twobody = atom_j > atom_i;

            if (rij < a1 && should_compute_twobody)
            {
                SADfloat<3> twobody1(rij, 0);
                SADfloat<3> twobody2(zi, 1);
                SADfloat<3> twobody3(zj, 2);
                twobody2 = A * (2 * powf(B / twobody1, rho) -
                                expf(-beta * twobody2 * twobody2) -
                                expf(-beta * twobody3 * twobody3));
                twobody3 = twobody2 * expf(sigma / (twobody1 - a1));
                local_energy += twobody3.val;
                temp1_force = twobody3.dval[0] / rij * drij;
                i_force = i_force - temp1_force;
                j_force = j_force + temp1_force;
                dE_dzi += twobody3.dval[1];
                dE_dzj += twobody3.dval[2];
                if (need_virial)
                    local_virial = local_virial -
                                   Get_Virial_From_Force_Dis(temp1_force, drij);
            }

            for (int k = j + 1; k < nl_i.atom_numbers; k += 1)
            {
                k_force = {0.0f, 0.0f, 0.0f};
                atom_k = nl_i.atom_serial[k];
                type_k = atom_types[atom_k];
                rk = crd[atom_k];
                drik = Get_Displacement<BoundaryPolicy::Periodic>(ri, rk,
                                                                  boundary);
                rik = norm3df(drik.x, drik.y, drik.z);
                pair_index_2 = type_i * atom_type_numbers + type_k;
                a2 = parameters[8 * pair_index_2 + 2];
                if (rik < a2 && rij < a1)
                {
                    triple_index = pair_index_1 * atom_type_numbers + type_k;
                    eta = parameters[8 * pair_type_numbers + 9 * triple_index];
                    gamma = parameters[8 * pair_type_numbers +
                                       9 * triple_index + 1];
                    lambda = parameters[8 * pair_type_numbers +
                                        9 * triple_index + 2];
                    Q0 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    3];
                    mu = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    4];
                    u1 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    5];
                    u2 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    6];
                    u3 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    7];
                    u4 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    8];
                    SADvector<7> threebody1(drij, 0, 1, 2);
                    SADvector<7> threebody2(drik, 3, 4, 5);
                    SADfloat<7> threebody3(zi, 6);
                    SADfloat<7> r1 = sqrtf(threebody1 * threebody1);
                    SADfloat<7> r2 = sqrtf(threebody2 * threebody2);
                    SADfloat<7> E = threebody1 * threebody2 / r1 / r2;
                    E = u1 +
                        u2 * (u3 * expf(-u4 * threebody3) -
                              expf(-2.0f * u4 * threebody3)) +
                        E;
                    E = E * E;
                    SADfloat<7> Q = Q0 * expf(-mu * threebody3);
                    E = lambda * (1 - expf(-Q * E) + eta * Q * E);
                    E = E * expf(gamma / (r1 - a1) + gamma / (r2 - a2));
                    temp1_force = {E.dval[0], E.dval[1], E.dval[2]};
                    temp2_force = {E.dval[3], E.dval[4], E.dval[5]};
                    k_force = k_force + temp2_force;
                    j_force = j_force + temp1_force;
                    i_force = i_force - temp1_force - temp2_force;
                    dE_dzi += E.dval[6];
                    local_energy += E.val;
                    if (need_virial)
                    {
                        local_virial =
                            local_virial -
                            Get_Virial_From_Force_Dis(temp1_force, drij) -
                            Get_Virial_From_Force_Dis(temp2_force, drik);
                    }
                }
                atomicAdd(frc + atom_k, k_force);
            }
            atomicAdd(frc + atom_j, j_force);
            atomicAdd(dE_dz + atom_j, dE_dzj);
        }
        Warp_Sum_To(frc + atom_i, i_force, warpSize);
        Warp_Sum_To(dE_dz + atom_i, dE_dzi, warpSize);
        if (need_energy)
        {
            atomicAdd(atom_energy + atom_i, local_energy);
            atomicAdd(this_energy + atom_i, local_energy);
        }
        if (need_virial)
            Warp_Sum_To(atom_virial + atom_i, local_virial, warpSize);
    }
}

static __global__ void Get_Z(const int atom_numbers, const VECTOR* crd,
                             const Boundary boundary, const ATOM_GROUP* nl,
                             const float* parameters,
                             const int atom_type_numbers, const int* atom_types,
                             float* z)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j;
    if (atom_i < atom_numbers)
#else
    int atom_j;
#pragma omp parallel for firstprivate(atom_j)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j;
        ATOM_GROUP nl_i = nl[atom_i];
        float local_z = 0, a, c, alpha, r;
        VECTOR ri = crd[atom_i], dr;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            atom_j = nl_i.atom_serial[j];
            type_j = atom_types[atom_j];
            type_j += atom_type_numbers * type_i;
            alpha = parameters[8 * type_j];
            c = parameters[8 * type_j + 1];
            a = parameters[8 * type_j + 2];
            dr = Get_Displacement<BoundaryPolicy::Periodic>(crd[atom_j], ri,
                                                            boundary);
            r = norm3df(dr.x, dr.y, dr.z);
            if (r < c)
            {
                local_z += 1.0f;
            }
            else if (r < a)
            {
                r = (r - c) / (a - c);
                r = expf(alpha / (1.0f - powf(r, -3.0f)));
                local_z += r;
            }
        }
        Warp_Sum_To(z + atom_i, local_z, warpSize);
    }
}

template <bool need_virial>
static __global__ __launch_bounds__(1024) void Redistribute_Z_to_Atoms(
    const int atom_numbers, const VECTOR* crd, const Boundary boundary,
    const ATOM_GROUP* nl, const float* parameters, const int atom_type_numbers,
    const int* atom_types, const float* dE_dz, VECTOR* frc,
    LTMatrix3* atom_virial)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j;
    if (atom_i < atom_numbers)
#else
    int atom_j;
#pragma omp parallel for firstprivate(atom_j)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j;
        ATOM_GROUP nl_i = nl[atom_i];
        float dE_dzi = dE_dz[atom_i], dE_dzj;
        float a, c, alpha;
        SADfloat<1> r(0, 0), z;
        VECTOR ri = crd[atom_i], dr;
        VECTOR local_frc = {0.0f, 0.0f, 0.0f}, f;
        LTMatrix3 local_virial(0.0f);
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            atom_j = nl_i.atom_serial[j];
            type_j = atom_types[atom_j];
            dE_dzj = dE_dz[atom_j] + dE_dzi;

            type_j += atom_type_numbers * type_i;
            alpha = parameters[8 * type_j];
            c = parameters[8 * type_j + 1];
            a = parameters[8 * type_j + 2];
            dr = Get_Displacement<BoundaryPolicy::Periodic>(ri, crd[atom_j],
                                                            boundary);
            r.val = norm3df(dr.x, dr.y, dr.z);
            if (r < a && r > c)
            {
                z = (r - c) / (a - c);
                z = expf(alpha / (1.0f - powf(z, -3.0f)));
                f = dE_dzj * z.dval[0] / r.val * dr;
                local_frc = local_frc - f;
                if (need_virial && atom_j > atom_i)
                    local_virial =
                        local_virial - Get_Virial_From_Force_Dis(f, dr);
            }
        }
        Warp_Sum_To(frc + atom_i, local_frc, warpSize);
        if (need_virial)
            Warp_Sum_To(atom_virial + atom_i, local_virial, warpSize);
    }
}

void EDIP_INFORMATION::EDIP_Force_With_Atom_Energy_And_Virial_Full_NL(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* fnl_d_nl,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial)
{
    if (!is_initialized || fnl_d_nl == NULL) return;

    if (need_atom_energy)
        deviceMemset(d_energy_sum, 0, sizeof(float) * (this->atom_numbers + 1));

    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;

    auto f1 = EDIP_Force_With_Full_Neighbor_CUDA<false, false>;
    auto f2 = Redistribute_Z_to_Atoms<false>;
    if (!need_atom_energy && !need_virial)
    {
        f1 = EDIP_Force_With_Full_Neighbor_CUDA<false, false>;
        f2 = Redistribute_Z_to_Atoms<false>;
    }
    else if (!need_atom_energy && need_virial)
    {
        f1 = EDIP_Force_With_Full_Neighbor_CUDA<false, true>;
        f2 = Redistribute_Z_to_Atoms<true>;
    }
    else if (need_atom_energy && !need_virial)
    {
        f1 = EDIP_Force_With_Full_Neighbor_CUDA<true, false>;
        f2 = Redistribute_Z_to_Atoms<false>;
    }
    else
    {
        f1 = EDIP_Force_With_Full_Neighbor_CUDA<true, true>;
        f2 = Redistribute_Z_to_Atoms<true>;
    }

    deviceMemset(this->z, 0, sizeof(float) * atom_numbers * 2);
    Launch_Device_Kernel(Get_Z, gridSize, blockSize, 0, NULL, atom_numbers, crd,
                         boundary, fnl_d_nl, this->d_parameters,
                         this->atom_type_numbers, this->d_atom_type, this->z);
    Launch_Device_Kernel(f1, gridSize, blockSize, 0, NULL, atom_numbers, crd,
                         frc, boundary, z, dE_dz, fnl_d_nl, atom_energy,
                         atom_virial, this->d_atom_type, this->d_parameters,
                         this->atom_type_numbers, this->pair_type_numbers,
                         this->d_energy_atom);
    Launch_Device_Kernel(f2, gridSize, blockSize, 0, NULL, atom_numbers, crd,
                         boundary, fnl_d_nl, this->d_parameters,
                         this->atom_type_numbers, this->d_atom_type,
                         this->dE_dz, frc, atom_virial);
}

void EDIP_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_energy_atom, d_energy_sum, atom_numbers);
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
