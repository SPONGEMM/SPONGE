#include "RMSD.h"

REGISTER_CV_STRUCTURE(CV_RMSD, "rmsd", 0);

static __global__ void Get_Center_of_Atoms(int atom_numbers, int* atoms,
                                           VECTOR* crd, VECTOR* points)
{
#ifdef USE_GPU
    __shared__ float crd_sum_x[1024];
    __shared__ float crd_sum_y[1024];
    __shared__ float crd_sum_z[1024];
    crd_sum_x[threadIdx.x] = 0.0f;
    crd_sum_y[threadIdx.x] = 0.0f;
    crd_sum_z[threadIdx.x] = 0.0f;
    VECTOR tempc;
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
    {
        tempc = crd[atoms[i]];
        points[i] = tempc;
        crd_sum_x[threadIdx.x] += tempc.x;
        crd_sum_y[threadIdx.x] += tempc.y;
        crd_sum_z[threadIdx.x] += tempc.z;
    }
    __syncthreads();
    for (int i = 512; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            crd_sum_x[threadIdx.x] += crd_sum_x[i + threadIdx.x];
            crd_sum_y[threadIdx.x] += crd_sum_y[i + threadIdx.x];
            crd_sum_z[threadIdx.x] += crd_sum_z[i + threadIdx.x];
        }
        __syncthreads();
    }
    tempc = {crd_sum_x[0] / atom_numbers, crd_sum_y[0] / atom_numbers,
             crd_sum_z[0] / atom_numbers};
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
    {
        points[i] = points[i] - tempc;
    }
#else
    // 使用标量累加器替代对结构体成员的 OpenMP reduction
    float crd_sum_x = 0.0f, crd_sum_y = 0.0f, crd_sum_z = 0.0f;

// 并行计算所有原子坐标并求和
#pragma omp parallel for reduction(+ : crd_sum_x, crd_sum_y, crd_sum_z)
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR tempc = crd[atoms[i]];
        points[i] = tempc;
        crd_sum_x += tempc.x;
        crd_sum_y += tempc.y;
        crd_sum_z += tempc.z;
    }

    // 计算中心点
    VECTOR center = {crd_sum_x / atom_numbers, crd_sum_y / atom_numbers,
                     crd_sum_z / atom_numbers};

// 减去中心点
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        points[i].x -= center.x;
        points[i].y -= center.y;
        points[i].z -= center.z;
    }
#endif
}

static __global__ void Get_Coordinate_Covariance(const int atom_numbers,
                                                 const VECTOR* A,
                                                 const VECTOR* B,
                                                 float* covariance)
{
#ifdef USE_GPU
    int atom_i = blockIdx.x * blockDim.x * blockDim.y +
                 threadIdx.x * blockDim.y + threadIdx.y;
    float local_covariance11 = 0;
    float local_covariance12 = 0;
    float local_covariance13 = 0;
    float local_covariance21 = 0;
    float local_covariance22 = 0;
    float local_covariance23 = 0;
    float local_covariance31 = 0;
    float local_covariance32 = 0;
    float local_covariance33 = 0;
    VECTOR a;
    VECTOR b;
    if (atom_i < atom_numbers)
    {
        a = A[atom_i];
        b = B[atom_i];
        local_covariance11 += a.x * b.x;
        local_covariance12 += a.x * b.y;
        local_covariance13 += a.x * b.z;
        local_covariance21 += a.y * b.x;
        local_covariance22 += a.y * b.y;
        local_covariance23 += a.y * b.z;
        local_covariance31 += a.z * b.x;
        local_covariance32 += a.z * b.y;
        local_covariance33 += a.z * b.z;
    }
    Warp_Sum_To(covariance, local_covariance11, warpSize);
    Warp_Sum_To(covariance + 1, local_covariance12, warpSize);
    Warp_Sum_To(covariance + 2, local_covariance13, warpSize);
    Warp_Sum_To(covariance + 3, local_covariance21, warpSize);
    Warp_Sum_To(covariance + 4, local_covariance22, warpSize);
    Warp_Sum_To(covariance + 5, local_covariance23, warpSize);
    Warp_Sum_To(covariance + 6, local_covariance31, warpSize);
    Warp_Sum_To(covariance + 7, local_covariance32, warpSize);
    Warp_Sum_To(covariance + 8, local_covariance33, warpSize);
#else
    // 初始化协方差矩阵
    for (int i = 0; i < 9; i++)
    {
        covariance[i] = 0.0f;
    }

    float cov00 = 0.0f, cov01 = 0.0f, cov02 = 0.0f;
    float cov10 = 0.0f, cov11 = 0.0f, cov12 = 0.0f;
    float cov20 = 0.0f, cov21 = 0.0f, cov22 = 0.0f;

#pragma omp parallel for reduction(+ : cov00, cov01, cov02, cov10, cov11, \
                                       cov12, cov20, cov21, cov22)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        VECTOR a = A[atom_i];
        VECTOR b = B[atom_i];

        cov00 += a.x * b.x;
        cov01 += a.x * b.y;
        cov02 += a.x * b.z;
        cov10 += a.y * b.x;
        cov11 += a.y * b.y;
        cov12 += a.y * b.z;
        cov20 += a.z * b.x;
        cov21 += a.z * b.y;
        cov22 += a.z * b.z;
    }

    covariance[0] = cov00;
    covariance[1] = cov01;
    covariance[2] = cov02;
    covariance[3] = cov10;
    covariance[4] = cov11;
    covariance[5] = cov12;
    covariance[6] = cov20;
    covariance[7] = cov21;
    covariance[8] = cov22;
#endif
}

static __global__ void Get_Rotated_Reference(const int atom_numbers,
                                             const VECTOR* reference,
                                             const float* R,
                                             VECTOR* rotated_reference)
{
#ifdef USE_GPU
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR inp_crd = reference[atom_i];
        VECTOR out_crd;
        out_crd.x = inp_crd.x * R[0] + inp_crd.y * R[3] + inp_crd.z * R[6];
        out_crd.y = inp_crd.x * R[1] + inp_crd.y * R[4] + inp_crd.z * R[7];
        out_crd.z = inp_crd.x * R[2] + inp_crd.y * R[5] + inp_crd.z * R[8];
        rotated_reference[atom_i] = out_crd;
    }
}

static __global__ void Get_Rotation_Matrix(float* m, float* R)
{
    SVD::Mat3x3 M(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    SVD::SVDSet result = SVD::svd(M);
    result.S.m_00 = 1;
    result.S.m_11 = 1;
    result.V = result.V.transpose();
    SVD::Mat3x3 r = result.U * result.V;
    result.S.m_22 = copysign(1.0f, r.det());
    r = result.U * result.S * result.V;
    R[0] = r.m_00;
    R[1] = r.m_01;
    R[2] = r.m_02;
    R[3] = r.m_10;
    R[4] = r.m_11;
    R[5] = r.m_12;
    R[6] = r.m_20;
    R[7] = r.m_21;
    R[8] = r.m_22;
}

static __global__ void get_diff_and_rmsd(int atom_numbers, const VECTOR* points,
                                         const VECTOR* rotated_reference,
                                         float* d_value, int* atom,
                                         VECTOR* crd_grads, LTMatrix3* virial,
                                         const LTMatrix3 cell,
                                         const LTMatrix3 rcell)
{
#ifdef USE_GPU
    __shared__ float rmsd[1024];
    rmsd[threadIdx.x] = 0;
    virial[0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    __syncthreads();
    VECTOR diff;
    float rmsd0;
    int atom_i;
    LTMatrix3 local_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < atom_numbers; i += 1024)
    {
        diff = points[i] - rotated_reference[i];
        rmsd[threadIdx.x] +=
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }
    __syncthreads();
    for (int i = 512; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            rmsd[threadIdx.x] += rmsd[i + threadIdx.x];
        }
        __syncthreads();
    }
    rmsd0 = sqrtf(rmsd[0] / atom_numbers);
    d_value[0] = rmsd0;
    rmsd0 *= atom_numbers;
    for (int i = threadIdx.x; i < atom_numbers; i += 1024)
    {
        atom_i = atom[i];
        diff = points[i] - rotated_reference[i];
        diff.x /= rmsd0;
        diff.y /= rmsd0;
        diff.z /= rmsd0;
        crd_grads[atom_i] = diff;
        local_virial =
            local_virial + Get_Virial_From_Force_Dis(diff, points[i]);
    }
    atomicAdd(virial, local_virial);
#else
    // 使用标量累加器替代对结构体成员的 OpenMP reduction
    float rmsd_sum = 0.0f;
    float local_a11 = 0.0f, local_a21 = 0.0f, local_a22 = 0.0f,
          local_a31 = 0.0f, local_a32 = 0.0f, local_a33 = 0.0f;

// 并行计算RMSD
#pragma omp parallel for reduction(+ : rmsd_sum)
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR diff = {points[i].x - rotated_reference[i].x,
                       points[i].y - rotated_reference[i].y,
                       points[i].z - rotated_reference[i].z};
        rmsd_sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

    float rmsd0 = sqrtf(rmsd_sum / atom_numbers);
    d_value[0] = rmsd0;

    // 计算梯度
    float rmsd_factor = rmsd0 * atom_numbers;

#pragma omp parallel for reduction(+ : local_a11, local_a21, local_a22, \
                                       local_a31, local_a32, local_a33)
    for (int i = 0; i < atom_numbers; i++)
    {
        int atom_i = atom[i];
        VECTOR diff = {points[i].x - rotated_reference[i].x,
                       points[i].y - rotated_reference[i].y,
                       points[i].z - rotated_reference[i].z};

        if (rmsd_factor > 0.0f)
        {
            diff.x /= rmsd_factor;
            diff.y /= rmsd_factor;
            diff.z /= rmsd_factor;
        }

        crd_grads[atom_i] = diff;

        // 计算virial贡献并累加到标量累加器
        LTMatrix3 virial_contrib = Get_Virial_From_Force_Dis(diff, points[i]);
        local_a11 += virial_contrib.a11;
        local_a21 += virial_contrib.a21;
        local_a22 += virial_contrib.a22;
        local_a31 += virial_contrib.a31;
        local_a32 += virial_contrib.a32;
        local_a33 += virial_contrib.a33;
    }

    // 将标量累加器写回 virial 结构
    LTMatrix3 local_virial = {local_a11, local_a21, local_a22,
                              local_a31, local_a32, local_a33};
    virial[0] = local_virial;
#endif
}

void CV_RMSD::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
                      const char* module_name)
{
    std::vector<int> cpu_atom =
        manager->Ask_For_Indefinite_Length_Int_Parameter(module_name, "atom");
    if (cpu_atom.size() == 0)
    {
        std::string error_reason = "Reason:\n\tatoms are required for the CV ";
        error_reason += module_name;
        error_reason += " (atom or atom_in_file)\n";
        manager->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                    "CV_RMSD::Initial", error_reason.c_str());
    }
    this->atom_numbers = cpu_atom.size();
    manager->printf("        atom_numbers is %d\n", this->atom_numbers);
    std::vector<float> cpu_reference =
        manager->Ask_For_Indefinite_Length_Float_Parameter(module_name,
                                                           "coordinate");
    if (cpu_reference.size() == 0)
    {
        std::string error_reason =
            "Reason:\n\tcoordinates are required for the CV ";
        error_reason += module_name;
        error_reason += " (coordinate or coordinate_in_file)\n";
        manager->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                    "CV_RMSD::Initial", error_reason.c_str());
    }
    if (3 * this->atom_numbers != cpu_reference.size())
    {
        std::string error_reason = "Reason:\n\tthe number of coordinates (";
        error_reason += std::to_string(cpu_reference.size());
        error_reason += ") != 3 * the number of atoms (";
        error_reason += std::to_string(this->atom_numbers);
        error_reason += ") for the CV ";
        error_reason += module_name;
        error_reason += "\n";
        manager->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                    "CV_RMSD::Initial", error_reason.c_str());
    }
    Device_Malloc_Safely((void**)&atom, sizeof(int) * this->atom_numbers);
    deviceMemcpy(atom, cpu_atom.data(), sizeof(int) * this->atom_numbers,
                 deviceMemcpyHostToDevice);
    Device_Malloc_Safely((void**)&points, sizeof(VECTOR) * this->atom_numbers);
    Device_Malloc_Safely((void**)&covariance_matrix, sizeof(float) * 9);
    Device_Malloc_Safely((void**)&rotated_ref,
                         sizeof(VECTOR) * this->atom_numbers);
    Device_Malloc_Safely((void**)&R, sizeof(float) * 9);
    VECTOR center = {0, 0, 0};
    for (int i = 0; i < 3 * this->atom_numbers; i += 3)
    {
        center.x += cpu_reference[i];
        center.y += cpu_reference[i + 1];
        center.z += cpu_reference[i + 2];
    }
    center = 1.0f / this->atom_numbers * center;
    for (int i = 0; i < 3 * this->atom_numbers; i += 3)
    {
        cpu_reference[i] -= center.x;
        cpu_reference[i + 1] -= center.y;
        cpu_reference[i + 2] -= center.z;
    }
    Device_Malloc_Safely((void**)&references,
                         sizeof(VECTOR) * this->atom_numbers);
    deviceMemcpy(references, cpu_reference.data(),
                 sizeof(VECTOR) * this->atom_numbers, deviceMemcpyHostToDevice);
    rotated_comparing = 1;
    if (manager->Command_Exist(module_name, "rotate"))
    {
        rotated_comparing =
            manager->Get_Bool(module_name, "rotate", "CV_RMSD::Initial");
    }
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_RMSD::Compute(int atom_numbers, VECTOR* crd, const LTMatrix3 cell,
                      const LTMatrix3 rcell, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need)
    {
        Launch_Device_Kernel(Get_Center_of_Atoms, 1, 1024, 0,
                             this->device_stream, this->atom_numbers,
                             this->atom, crd, this->points);
        if (rotated_comparing)
        {
            deviceMemset(this->covariance_matrix, 0, sizeof(float) * 9);
            dim3 blockSize = {
                CONTROLLER::device_warp,
                CONTROLLER::device_max_thread / CONTROLLER::device_warp};
            dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
            Launch_Device_Kernel(Get_Coordinate_Covariance, gridSize, blockSize,
                                 0, this->device_stream, this->atom_numbers,
                                 this->references, this->points,
                                 this->covariance_matrix);
            Launch_Device_Kernel(Get_Rotation_Matrix, 1, 1, 0,
                                 this->device_stream, this->covariance_matrix,
                                 this->R);
            Launch_Device_Kernel(Get_Rotated_Reference,
                                 (atom_numbers + 1023) / 1024, 1024, 0,
                                 this->device_stream, this->atom_numbers,
                                 this->references, this->R, this->rotated_ref);
            Launch_Device_Kernel(get_diff_and_rmsd, 1, 1024, 0,
                                 this->device_stream, this->atom_numbers,
                                 this->points, this->rotated_ref, d_value, atom,
                                 crd_grads, virial, cell, rcell);
        }
        else
        {
            Launch_Device_Kernel(get_diff_and_rmsd, 1, 1024, 0,
                                 this->device_stream, this->atom_numbers,
                                 this->points, this->references, d_value, atom,
                                 crd_grads, virial, cell, rcell);
        }
        deviceMemcpyAsync(&value, d_value, sizeof(float),
                          deviceMemcpyDeviceToHost, this->device_stream);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
