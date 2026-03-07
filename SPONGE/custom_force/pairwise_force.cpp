#include "pairwise_force.h"

static __global__ void pairwise_force_scatter_types(
    const int total_numbers, const int* atom_local,
    const int* global_pairwise_types, int* local_pairwise_types)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_numbers)
#else
#pragma omp parallel for
    for (int idx = 0; idx < total_numbers; idx++)
#endif
    {
        int atom = atom_local[idx];
        local_pairwise_types[idx] = global_pairwise_types[atom];
    }
}

void PAIRWISE_FORCE::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "pairwise_force");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("START INITIALIZING PAIRWISE FORCE:\n");
        this->Read_Configuration(controller);
        this->JIT_Compile(controller);
        this->Real_Initial(controller);
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->force_name.c_str(), "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING PAIRWISE FORCE\n\n");
    }
    else
    {
        controller->printf("PAIRWISE FORCE IS NOT INITIALIZED\n\n");
    }
}

void PAIRWISE_FORCE::Read_Configuration(CONTROLLER* controller)
{
    Configuration_Reader cfg;
    cfg.Open(controller->Command(this->module_name, "in_file"));
    cfg.Close();
    if (!cfg.error_reason.empty())
    {
        cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Initial",
                                       cfg.error_reason.c_str());
    }
    if (cfg.sections.size() > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            "Reason:\n\tOnly one pairwise force can be used\n");
    }
    force_name = cfg.sections[0];
    if (!cfg.Key_Exist(force_name, "potential"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            string_format("Reason:\n\tThe potential of the pairwise force "
                          "%FORCE% is required ([[ potential ]])\n",
                          {{"FORCE", force_name}})
                .c_str());
    }
    source_code = cfg.Get_Value(force_name, "potential");
    if (!cfg.Key_Exist(force_name, "parameters"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            string_format("Reason:\n\tThe parameters of the pairwise force "
                          "%FORCE% are required ([[ parameter ]])\n",
                          {{"FORCE", force_name}})
                .c_str());
    }
    std::string parameter_strings = cfg.Get_Value(force_name, "parameters");
    std::vector<std::string> parameter_and_types =
        string_split(parameter_strings, ",");
    for (std::string s : parameter_and_types)
    {
        std::vector<std::string> parameter_and_type =
            string_split(string_strip(s), " ");
        if (parameter_and_type[0] != "int" && parameter_and_type[0] != "float")
        {
            controller->Throw_SPONGE_Error(
                spongeErrorTypeErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tOnly 'int' or 'float' parameter is acceptable\n");
        }
        this->parameter_type.push_back(parameter_and_type[0]);
        this->parameter_name.push_back(parameter_and_type[1]);
    }
    n_ij_parameter = 0;
    for (auto s : this->parameter_name)
    {
        if (s.rfind("_ij") == s.length() - 3)
        {
            n_ij_parameter -= 1;
        }
        else if (n_ij_parameter < 0)
        {
            n_ij_parameter *= -1;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise parameters should be placed in front of "
                "atomic parameters");
        }
    }
    n_ij_parameter = abs(n_ij_parameter);
    with_ele = true;
    if (cfg.Key_Exist(force_name, "with_ele"))
    {
        std::string with_ele_choice = cfg.Get_Value(force_name, "with_ele");
        if (!is_str_equal(with_ele_choice.c_str(), "true") &&
            !is_str_equal(with_ele_choice.c_str(), "false") &&
            !is_str_int(with_ele_choice.c_str()))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise [[ with_ele ]] should be 'true', 'false' "
                "or integers (0 for 'false' and others for 'true')");
        }
        if (is_str_equal(with_ele_choice.c_str(), "true") ||
            (is_str_int(with_ele_choice.c_str()) &&
             atoi(with_ele_choice.c_str())))
        {
            with_ele = true;
        }
        else
        {
            with_ele = false;
        }
    }
    if (with_ele)
    {
        ele_code = "E_ele = charge_i * charge_j * erfc(beta * r_ij) / r_ij;";
        if (cfg.Key_Exist(force_name, "electrostatic_potential"))
        {
            ele_code = cfg.Get_Value(force_name, "electrostatic_potential");
        }
    }
    else
    {
        ele_code.clear();
    }
    for (auto s : cfg.value_unused)
    {
        std::string error_reason = string_format(
            "Reason:\n\t[[ %s% ]] should not be one of the keys of the "
            "pairwise force input file",
            {{"s", s.second}});
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Read_Configuration",
                                       error_reason.c_str());
    }
}

void PAIRWISE_FORCE::JIT_Compile(CONTROLLER* controller)
{
    if (with_ele)
        controller->printf(
            "        %s will be calculated with direct part of the "
            "electrostatic potential\n",
            this->force_name.c_str());
    else
        controller->printf(
            "        %s will not be calculated with direct part of the "
            "electrostatic potential\n",
            this->force_name.c_str());
    std::string full_source_code = R"JIT(#if defined(__CUDACC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_CUDA
#define USE_CUDA
#endif
#elif defined(__HIPCC__) || defined(__HIPCC_RTC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_HIP
#define USE_HIP
#endif
#endif
#include "common.h"
#ifndef USE_GPU
__forceinline__ float atomicAdd(float* x, float y)
{
    float x0;
#ifdef _WIN32
#pragma omp critical(sponge_jit_atomic_add_float)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}
__forceinline__ int atomicAdd(int* x, int y)
{
    int x0;
#ifdef _WIN32
#pragma omp critical(sponge_jit_atomic_add_int)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}
#endif
__device__ __forceinline__ int Get_Pairwise_Type(int a, int b)
{
    int y = (b - a);
    int x = y >> 31;
    y = (y ^ x) - x;
    x = b + a;
    int z = (x + y) >> 1;
    x = (x - y) >> 1;
    return (z * (z + 1) >> 1) + x;
}
__device__ __forceinline__ float norm3df(float a, float b, float c)
{
    return sqrtf(a * a + b * b + c * c);
}

extern "C" __global__ void pairwise_force_energy_and_virial(%PARM_ARGS%,
    const float* charge, const float pme_beta, ATOM_GROUP* nl, const int* pairwise_types,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    VECTOR* frc, float* atom_energy, LTMatrix3* atom_virial, float* pme_atom_energy,
    float* listed_item_energy, const int local_atom_numbers, int need_atom_energy,
    int need_virial, int atom_numbers)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
#ifdef USE_GPU
        if (atom_i >= local_atom_numbers) return;
#else
        if (atom_i >= local_atom_numbers) continue;
#endif
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR r1 = crd[atom_i];
        int pairwise_type_i = pairwise_types[atom_i];
        VECTOR frc_record = { 0.0f, 0.0f, 0.0f };
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
        float energy_total = 0.0f;
        float energy_coulomb = 0.0f;
        float charge_i = 0;
        float charge_j = 0;
        if (pme_atom_energy != NULL)
        {
            charge_i = charge[atom_i];
        }
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR vector_dr = Get_Periodic_Displacement(crd[atom_j], r1, cell, rcell);
            float float_dr_ij = norm3df(vector_dr.x, vector_dr.y, vector_dr.z);
            if (float_dr_ij < cutoff)
            {
                int atom_pairwise_type = Get_Pairwise_Type(pairwise_type_i, pairwise_types[atom_j]);
                %PARM_DEC%
                SADfloat<1> r_ij(float_dr_ij, 0);
                SADfloat<1> E, E_ele;
                %SOURCE_CODE%
                energy_total += ij_factor * E.val;
                if (pme_atom_energy != NULL)
                {
                    charge_j = charge[atom_j];
                    %COULOMB_CODE%
                    energy_coulomb += ij_factor * E_ele.val;
                }
                float frc_abs = E.dval[0] / float_dr_ij;
                if (pme_atom_energy != NULL)
                {
                    frc_abs += E_ele.dval[0] / float_dr_ij;
                }
                VECTOR frc_temp = frc_abs * vector_dr;
                if (frc != NULL)
                {
                    frc_record = frc_record + frc_temp;
                    if (atom_j < local_atom_numbers)
                        atomicAdd(frc + atom_j, -frc_temp);
                }
                if (need_virial && atom_virial != NULL)
                {
                    virial_record = virial_record -
                        ij_factor * Get_Virial_From_Force_Dis(frc_temp, vector_dr);
                }
            }
        }
        if (frc != NULL)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (pme_atom_energy != NULL && (need_atom_energy || need_virial))
        {
            float energy_coulomb_sum = energy_coulomb;
            Warp_Sum_To(pme_atom_energy + atom_i, energy_coulomb_sum, warpSize);
        }
        if (listed_item_energy != NULL)
        {
            float listed_energy_sum = energy_total;
            Warp_Sum_To(listed_item_energy + atom_i, listed_energy_sum, warpSize);
        }
        if (need_atom_energy && atom_energy != NULL)
        {
            float atom_energy_sum = energy_total;
            Warp_Sum_To(atom_energy + atom_i, atom_energy_sum, warpSize);
        }
        if (need_virial && atom_virial != NULL)
        {
            Warp_Sum_To(&(atom_virial + atom_i)->a11, virial_record.a11, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a21, virial_record.a21, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a22, virial_record.a22, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a31, virial_record.a31, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a32, virial_record.a32, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a33, virial_record.a33, warpSize);
        }
    }
}
)JIT";
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ",
                                        {parameter_type, parameter_name});
    std::string PARM_DEC = string_join(
        "                const %0% %1% = %1%_list[atom_pairwise_type];", "\n",
        {parameter_type, parameter_name});
    full_source_code =
        string_format(full_source_code, {{"PARM_ARGS", PARM_ARGS},
                                         {"PARM_DEC", PARM_DEC},
                                         {"SOURCE_CODE", source_code},
                                         {"COULOMB_CODE", ele_code}});
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "PAIRWISE_FORCE::JIT_Compile",
                                       force_function.error_reason.c_str());
    }
}

void PAIRWISE_FORCE::Real_Initial(CONTROLLER* controller)
{
    FILE* fp;
    if (!controller->Command_Exist(this->force_name.c_str(), "in_file"))
    {
        std::string error_reason = "Reason:\n\tlisted force '" +
                                   this->force_name + "' is defined, but " +
                                   this->force_name +
                                   "_in_file is not provided\n";
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                       "PAIRWISE_FORCE::Initial",
                                       error_reason.c_str());
    }
    controller->printf("    Initializing %s\n", this->force_name.c_str());
    Open_File_Safely(
        &fp, controller->Command(this->force_name.c_str(), "in_file"), "r");
    if (fscanf(fp, "%d %d", &atom_numbers, &type_numbers) != 2)
    {
        std::string error_reason =
            "Reason:\n\tFail to read the number of atoms and/or types of the "
            "pairwise force '" +
            this->force_name + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Initial",
                                       error_reason.c_str());
    }
    int total_type_pairwise_numbers = type_numbers * (type_numbers + 1) / 2;
    Malloc_Safely((void**)&cpu_parameters,
                  sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters,
                  sizeof(void*) * parameter_name.size());
    launch_args = std::vector<void*>(parameter_name.size() + 17);
    Malloc_Safely((void**)&cpu_pairwise_types, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&item_energy, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&sum_energy, sizeof(float));
    Device_Malloc_Safely((void**)&gpu_pairwise_types_local,
                         sizeof(int) * atom_numbers);
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(float) * total_type_pairwise_numbers);
        }
        launch_args[j] = gpu_parameters + j;
    }
    for (int j = 0; j < n_ij_parameter; j++)
    {
        for (int i = 0; i < total_type_pairwise_numbers; i++)
        {
            int scanf_ret = 0;
            if (parameter_type[j] == "int")
            {
                scanf_ret = fscanf(fp, "%d", ((int*)cpu_parameters[j]) + i);
            }
            else
            {
                scanf_ret = fscanf(fp, "%f", ((float*)cpu_parameters[j]) + i);
            }
            if (scanf_ret == 0)
            {
                std::string error_reason =
                    "Reason:\n\tFail to read the parameters of the pairwise "
                    "force '" +
                    this->force_name + "'\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "PAIRWISE_FORCE::Initial",
                                               error_reason.c_str());
            }
        }
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", cpu_pairwise_types + i) != 1)
        {
            std::string error_reason =
                "Reason:\n\tFail to read the types of the pairwise force '" +
                this->force_name + "'\n";
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "PAIRWISE_FORCE::Initial",
                                           error_reason.c_str());
        }
    }
    fclose(fp);
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Device_Malloc_And_Copy_Safely(
                (void**)gpu_parameters + j, cpu_parameters[j],
                sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Device_Malloc_And_Copy_Safely(
                (void**)gpu_parameters + j, cpu_parameters[j],
                sizeof(float) * total_type_pairwise_numbers);
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&gpu_pairwise_types,
                                  cpu_pairwise_types,
                                  sizeof(int) * atom_numbers);
    deviceMemcpy(gpu_pairwise_types_local, gpu_pairwise_types,
                 sizeof(int) * atom_numbers, deviceMemcpyDeviceToDevice);
    local_atom_numbers = atom_numbers;
    total_local_numbers = atom_numbers;
    this->is_initialized = 1;
}

void PAIRWISE_FORCE::Compute_Force(ATOM_GROUP* nl, const VECTOR* crd,
                                   LTMatrix3 cell, LTMatrix3 rcell,
                                   float cutoff, float pme_beta, float* charge,
                                   VECTOR* frc, int need_energy,
                                   float* atom_energy, int need_virial,
                                   LTMatrix3* atom_virial,
                                   float* pme_direct_atom_energy)
{
    if (!this->is_initialized || total_local_numbers <= 0) return;

    float* listed_item_energy = need_energy ? this->item_energy : NULL;
    if (listed_item_energy != NULL)
    {
        deviceMemset(this->item_energy, 0, sizeof(float) * local_atom_numbers);
    }
    float* NULLPTR = NULL;
    LTMatrix3* NULL_VIRIAL = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types_local;
    launch_args[parameter_name.size() + 4] = &crd;
    launch_args[parameter_name.size() + 5] = &cell;
    launch_args[parameter_name.size() + 6] = &rcell;
    launch_args[parameter_name.size() + 7] = &cutoff;
    launch_args[parameter_name.size() + 8] = &frc;
    launch_args[parameter_name.size() + 9] =
        need_energy ? &atom_energy : &NULLPTR;
    launch_args[parameter_name.size() + 10] =
        need_virial ? &atom_virial : &NULL_VIRIAL;
    float* pme_ptr = NULLPTR;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }
    launch_args[parameter_name.size() + 11] = &pme_ptr;
    if (listed_item_energy != NULL)
    {
        launch_args[parameter_name.size() + 12] = &listed_item_energy;
    }
    else
    {
        launch_args[parameter_name.size() + 12] = &NULLPTR;
    }
    int local_atom_numbers_flag = local_atom_numbers;
    int need_atom_energy_flag = need_energy ? 1 : 0;
    int need_virial_flag = need_virial ? 1 : 0;
    int total_numbers_flag = total_local_numbers;
    launch_args[parameter_name.size() + 13] = &local_atom_numbers_flag;
    launch_args[parameter_name.size() + 14] = &need_atom_energy_flag;
    launch_args[parameter_name.size() + 15] = &need_virial_flag;
    launch_args[parameter_name.size() + 16] = &total_numbers_flag;

    force_function({(total_local_numbers + 31u) / 32u, 1, 1}, {32, 32, 1}, 0, 0,
                   launch_args);

    if (need_energy)
    {
        Sum_Of_List(item_energy, sum_energy, local_atom_numbers);
        deviceMemcpy(&last_energy, sum_energy, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        last_energy = 0.0f;
    }
}

float PAIRWISE_FORCE::Get_Energy(ATOM_GROUP* nl, const VECTOR* crd,
                                 LTMatrix3 cell, LTMatrix3 rcell, float cutoff,
                                 float pme_beta, float* charge,
                                 float* pme_direct_atom_energy)
{
    if (!this->is_initialized || total_local_numbers <= 0) return 0;

    deviceMemset(this->item_energy, 0, sizeof(float) * local_atom_numbers);
    float* NULLPTR = NULL;
    LTMatrix3* NULL_VIRIAL = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types_local;
    launch_args[parameter_name.size() + 4] = &crd;
    launch_args[parameter_name.size() + 5] = &cell;
    launch_args[parameter_name.size() + 6] = &rcell;
    launch_args[parameter_name.size() + 7] = &cutoff;
    launch_args[parameter_name.size() + 8] = &NULLPTR;
    launch_args[parameter_name.size() + 9] = &item_energy;
    launch_args[parameter_name.size() + 10] = &NULL_VIRIAL;
    float* pme_ptr = NULLPTR;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }
    launch_args[parameter_name.size() + 11] = &pme_ptr;
    launch_args[parameter_name.size() + 12] = &item_energy;
    int local_atom_numbers_flag = local_atom_numbers;
    int need_atom_energy_flag = 1;
    int need_virial_flag = 0;
    int total_numbers_flag = total_local_numbers;
    launch_args[parameter_name.size() + 13] = &local_atom_numbers_flag;
    launch_args[parameter_name.size() + 14] = &need_atom_energy_flag;
    launch_args[parameter_name.size() + 15] = &need_virial_flag;
    launch_args[parameter_name.size() + 16] = &total_numbers_flag;
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (total_local_numbers + blockSize.y - 1) / blockSize.y;
    force_function(gridSize, blockSize, 0, 0, launch_args);
    Sum_Of_List(item_energy, sum_energy, local_atom_numbers);
    float h_energy = NAN;
    deviceMemcpy(&h_energy, sum_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    return h_energy;
}

void PAIRWISE_FORCE::Get_Local(int* atom_local, int local_atom_numbers,
                               int ghost_numbers, char* atom_local_label,
                               int* atom_local_id)
{
    (void)atom_local_label;
    (void)atom_local_id;
    if (!is_initialized) return;
    int total = local_atom_numbers + ghost_numbers;
    if (total <= 0) return;
    this->local_atom_numbers = local_atom_numbers;
    this->total_local_numbers = total;
    Launch_Device_Kernel(pairwise_force_scatter_types, (total + 255) / 256, 256,
                         0, NULL, total, atom_local, gpu_pairwise_types,
                         gpu_pairwise_types_local);
}

void PAIRWISE_FORCE::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || !is_controller_printf_initialized) return;
    if (CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size) return;
    h_energy = last_energy;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_energy, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print(this->force_name.c_str(), &h_energy, true);
}
