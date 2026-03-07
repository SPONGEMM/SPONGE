#include "listed_forces.h"

static constexpr int LISTED_FORCE_MAX_ATOMS = 6;

static __global__ void listed_force_get_local_device(
    int item_numbers, int parameter_numbers, const int* parameter_is_int,
    const int* parameter_is_atom, void** parameter_ptrs,
    void** parameter_ptrs_local, const char* atom_local_label,
    const int* atom_local_id, int* local_item_numbers)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    local_item_numbers[0] = 0;
    for (int item_i = 0; item_i < item_numbers; item_i++)
    {
        bool has_local_atom = false;
        bool missing_atom = false;
        int local_atom_ids[LISTED_FORCE_MAX_ATOMS];
        int atom_idx = 0;
        for (int param_i = 0; param_i < parameter_numbers; param_i++)
        {
            if (!parameter_is_atom[param_i]) continue;
            int global_atom =
                reinterpret_cast<int*>(parameter_ptrs[param_i])[item_i];
            int mapped_id = atom_local_id[global_atom];
            if (mapped_id < 0)
            {
                missing_atom = true;
                break;
            }
            local_atom_ids[atom_idx++] = mapped_id;
            if (atom_local_label[global_atom] == 1)
            {
                has_local_atom = true;
            }
        }
        if (!has_local_atom || missing_atom) continue;
        int write_idx = local_item_numbers[0];
        local_item_numbers[0] += 1;
        int local_atom_iter = 0;
        for (int param_i = 0; param_i < parameter_numbers; param_i++)
        {
            if (parameter_is_int[param_i])
            {
                int value =
                    reinterpret_cast<int*>(parameter_ptrs[param_i])[item_i];
                if (parameter_is_atom[param_i])
                {
                    value = local_atom_ids[local_atom_iter++];
                }
                reinterpret_cast<int*>(
                    parameter_ptrs_local[param_i])[write_idx] = value;
            }
            else
            {
                reinterpret_cast<float*>(
                    parameter_ptrs_local[param_i])[write_idx] =
                    reinterpret_cast<float*>(parameter_ptrs[param_i])[item_i];
            }
        }
    }
}

static LISTED_FORCE* Read_One_Force(CONTROLLER* controller, std::string section,
                                    Configuration_Reader* cfg)
{
    LISTED_FORCE* force = new LISTED_FORCE;
    strcpy(force->module_name, section.c_str());
    controller->printf("    reading the listed force named %s\n",
                       force->module_name);
    if (!cfg->Key_Exist(section, "potential"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Read_One_Force (listed_forces.cu)",
            string_format("Reason:\n\tThe potential of the listed force "
                          "%FORCE% is required ([[ potential ]])\n",
                          {{"FORCE", section}})
                .c_str());
    }
    force->source_code = cfg->Get_Value(section, "potential");
    if (!cfg->Key_Exist(section, "parameters"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Read_One_Force (listed_forces.cu)",
            string_format("Reason:\n\tThe parameters of the listed force "
                          "%FORCE% is required ([[ potential ]])\n",
                          {{"FORCE", section}})
                .c_str());
    }
    force->Initialize_Parameters(controller,
                                 cfg->Get_Value(section, "parameters"));
    if (cfg->Key_Exist(section, "connected_atoms"))
    {
        controller->printf("        parsing connected atoms of %s\n",
                           force->module_name);
        force->connected_atoms = cfg->Get_Value(section, "connected_atoms");
        if (force->connected_atoms.size() != 2 ||
            force->connected_atoms[0] == force->connected_atoms[1])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "Read_One_Force (listed_forces.cu)",
                "Reason:\n\tConnected atoms should be 2 different char");
        }
    }
    if (cfg->Key_Exist(section, "constrain_distance"))
    {
        controller->printf("        parsing constrain distance of %s\n",
                           force->module_name);
        force->constrain_distance =
            cfg->Get_Value(section, "constrain_distance");
    }
    force->Compile(controller);
    controller->printf("    end reading the listed force named %s\n",
                       force->module_name);
    return force;
}

void LISTED_FORCES::Initial(CONTROLLER* controller, CONECT* connectivity,
                            PAIR_DISTANCE* con_dis, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "listed_forces");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("START INITIALIZING LISTED FORCES:\n");
        Configuration_Reader cfg;
        cfg.Open(controller->Command(this->module_name, "in_file"));
        cfg.Close();
        if (!cfg.error_reason.empty())
        {
            cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "LISTED_FORCES::Initial",
                                           cfg.error_reason.c_str());
        }
        for (std::string section : cfg.sections)
        {
            forces.push_back(Read_One_Force(controller, section, &cfg));
        }
        for (auto s : cfg.value_unused)
        {
            std::string error_reason = string_format(
                "Reason:\n\t[[ %s% ]] should not be one of the keys of the "
                "listed force [[[ %a% ]]]",
                {{"s", s.second}, {"a", s.first}});
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "LISTED_FORCES::Initial",
                                           error_reason.c_str());
        }
    }
    if (forces.size() != 0)
    {
        is_initialized = 1;
        for (auto force : forces)
        {
            force->Initial(controller, connectivity, con_dis);
        }
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        for (auto force : forces)
        {
            controller->Step_Print_Initial(force->module_name, "%.2f");
        }
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING LISTED FORCES\n\n");
    }
    else
    {
        controller->printf("LISTED FORCES IS NOT INITIALIZED\n\n");
    }
}

void LISTED_FORCES::Compute_Force(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                                  LTMatrix3 rcell, VECTOR* frc, int need_energy,
                                  float* atom_energy, int need_pressure,
                                  LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            force->Compute_Force(atom_numbers, crd, cell, rcell, frc,
                                 need_energy, atom_energy, need_pressure,
                                 atom_virial);
        }
    }
}

void LISTED_FORCES::Get_Local(int* atom_local, int local_atom_numbers,
                              int ghost_numbers, char* atom_local_label,
                              int* atom_local_id)
{
    if (!is_initialized) return;
    for (auto force : forces)
    {
        force->Get_Local(atom_local, local_atom_numbers, ghost_numbers,
                         atom_local_label, atom_local_id);
    }
}

void LISTED_FORCES::Step_Print(CONTROLLER* controller)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            force->Step_Print(controller);
        }
    }
}

void LISTED_FORCE::Initialize_Parameters(CONTROLLER* controller,
                                         std::string parameter_string)
{
    atom_labels.clear();
    parameter_is_atom.clear();
    parameter_is_int.clear();
    std::vector<std::string> parameters_with_type =
        string_split(string_strip(parameter_string), ",");
    for (std::string parameter_with_type : parameters_with_type)
    {
        std::vector<std::string> parameter_and_type =
            string_split(string_strip(parameter_with_type), " ");
        if (parameter_and_type[0] != "int" && parameter_and_type[0] != "float")
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "LISTED_FORCE::Initialize_Parameters",
                "Reason:\n\tOnly 'int' or 'float' parameter is acceptable\n");
        }
        this->parameter_type.push_back(parameter_and_type[0]);
        this->parameter_name.push_back(parameter_and_type[1]);
    }
    parameter_is_atom.assign(parameter_name.size(), 0);
    parameter_is_int.assign(parameter_name.size(), 0);
    for (int i = 0; i < parameter_type.size(); i++)
    {
        if (parameter_type[i] == "int") parameter_is_int[i] = 1;
    }
    for (int i = 0; i < parameter_type.size(); i++)
    {
        size_t pos = parameter_name[i].rfind("atom_", 0);
        if (pos == 0 && parameter_name[i].size() == 6 &&
            parameter_type[i] == "int")
        {
            atom_labels.push_back(std::string(1, parameter_name[i][5]));
            parameter_is_atom[i] = 1;
        }
    }
    if (atom_labels.size() > 6 || atom_labels.size() < 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "LISTED_FORCE::Initialize_Parameters",
                                       "Reason:\n\tthe supported numer of "
                                       "atoms in the listed force is 1 to 6\n");
    }
    else
    {
        std::string print_hint;
        for (int i = 0; i < atom_labels.size(); i++)
        {
            print_hint += atom_labels[i];
            if (i != atom_labels.size() - 1)
            {
                print_hint += ", ";
            }
        }
        controller->printf(
            "            %d labels of atoms (%s) found in the parameters\n",
            atom_labels.size(), print_hint.c_str());
    }
}

void LISTED_FORCE::Compile(CONTROLLER* controller)
{
    std::set<std::string> needed_bond;
    std::vector<StringVector> needed_bonds(2);
    std::set<std::string> needed_angle;
    std::vector<StringVector> needed_angles(3);
    std::set<std::string> needed_dihedral;
    std::vector<StringVector> needed_dihedrals(4);
    std::string bond_pair;
    std::string angle_pair;
    std::string dihedral_pair;
    std::string temp_pair;
    for (int i = 0; i < atom_labels.size(); i++)
    {
        for (int j = 0; j < atom_labels.size(); j++)
        {
            if (i == j) continue;
            bond_pair.clear();
            bond_pair += atom_labels[i];
            bond_pair += atom_labels[j];
            if (source_code.find("r_" + bond_pair) != source_code.npos)
            {
                needed_bond.insert(bond_pair);
            }
            for (int k = 0; k < atom_labels.size(); k++)
            {
                if (bond_pair.find(atom_labels[k]) != source_code.npos)
                    continue;
                angle_pair = bond_pair + atom_labels[k];
                if (source_code.find("theta_" + angle_pair) != source_code.npos)
                {
                    needed_angle.insert(angle_pair);
                    needed_bond.insert(bond_pair);
                    temp_pair.clear();
                    temp_pair += atom_labels[k];
                    temp_pair += atom_labels[j];
                    needed_bond.insert(temp_pair);
                }
                for (int l = 0; l < atom_labels.size(); l++)
                {
                    if (angle_pair.find(atom_labels[l]) != source_code.npos)
                        continue;
                    dihedral_pair = angle_pair + atom_labels[l];
                    if (source_code.find("phi_" + dihedral_pair) !=
                        source_code.npos)
                    {
                        needed_dihedral.insert(dihedral_pair);
                        needed_bond.insert(bond_pair);
                        temp_pair.clear();
                        temp_pair += atom_labels[j];
                        temp_pair += atom_labels[k];
                        needed_bond.insert(temp_pair);
                        temp_pair.clear();
                        temp_pair += atom_labels[k];
                        temp_pair += atom_labels[l];
                        needed_bond.insert(temp_pair);
                    }
                }
            }
        }
    }
    for (auto s : needed_bond)
    {
        needed_bonds[0].push_back(std::string(1, s[0]));
        needed_bonds[1].push_back(std::string(1, s[1]));
    }
    for (auto s : needed_angle)
    {
        needed_angles[0].push_back(std::string(1, s[0]));
        needed_angles[1].push_back(std::string(1, s[1]));
        needed_angles[2].push_back(std::string(1, s[2]));
    }
    for (auto s : needed_dihedral)
    {
        needed_dihedrals[0].push_back(std::string(1, s[0]));
        needed_dihedrals[1].push_back(std::string(1, s[1]));
        needed_dihedrals[2].push_back(std::string(1, s[2]));
        needed_dihedrals[3].push_back(std::string(1, s[3]));
    }
    controller->printf(
        "            %d distance(s), %d angle(s), %d dihedral(s) needed for "
        "%s\n",
        needed_bond.size(), needed_angle.size(), needed_dihedral.size(),
        this->module_name);
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
extern "C" __global__ __launch_bounds__(1024) void listed_force_energy_and_virial(%PARM_ARGS%,
VECTOR* crd, VECTOR box_length, VECTOR *frc, float *atom_ene, LTMatrix3 *atom_virial, float *listed_item_energy, const int local_atom_numbers, int need_atom_energy, int need_virial, int only_energy, int listed_force_item_numbers)
{
#ifdef USE_GPU
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < listed_force_item_numbers)
#else
#pragma omp parallel for
    for (int tid = 0; tid < listed_force_item_numbers; tid++)
#endif
    {
        %TEMP_DEC%
        %PARM_DEC%
        %CRD_DEC%
        %BOND_DEC%
        %ANGLE_DEC%
        %DIHEDRAL_DEC%
        %SOURCE%
        if (!only_energy)
        {
            %FORCE_DEC%
            %OUTPUT%
        }
        else
        {
            atomicAdd(atom_ene + tid, E.val);
        }
    }
}
)JIT";
    std::string sadv = std::to_string(atom_labels.size() * 3 + 3);
    std::string sadf = string_format("SADfloat<%N%>", {{"N", sadv}});
    sadv = string_format("SADvector<%N%>", {{"N", sadv}});
    std::string endl = "\n        ";
    std::string TEMP_DEC = sadf + " E;\n";
    if (!needed_dihedral.empty())
    {
        TEMP_DEC += sadv + " temp1, temp2;\n";
    }
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ",
                                        {parameter_type, parameter_name});
    std::string PARM_DEC = string_join("const %0% %1% = %1%_list[tid];", endl,
                                       {parameter_type, parameter_name});
    std::string CRD_DEC =
        sadv + " box_length_with_grads(box_length, 0, 1, 2);" + endl;
    CRD_DEC +=
        string_join(string_format("%sadv% r_%0%(crd[atom_%0%], 3 * %INDEX% + "
                                  "3, 3 * %INDEX% + 4, 3 * %INDEX% + 5);",
                                  {{"sadv", sadv}}),
                    endl, {atom_labels});
    std::string BOND_DEC = string_join(
        string_format(
            R"JIT(%sadv% dr_%0%%1% = Get_Periodic_Displacement(r_%0%, r_%1%, box_length_with_grads);
        %sadf% r_%0%%1% = sqrtf(dr_%0%%1% * dr_%0%%1%);)JIT",
            {{"sadv", sadv}, {"sadf", sadf}}),
        endl, needed_bonds);
    std::string ANGLE_DEC = string_join(
        string_format(
            R"JIT(%sadf% %theta% = 1.0f / (%r1% * %r1%) / (%r2% * %r2%);
        %theta% = sqrtf(%theta%) * (%r1% * %r2%);
        if (%theta% > 0.999999f) %theta% = 0.999999f;
        else if (%theta% < -0.999999f) %theta% = -0.999999f;
        %theta% = acosf(%theta%);)JIT",
            {{"sadf", sadf},
             {"theta", "theta_%0%%1%%2%"},
             {"r1", "dr_%0%%1%"},
             {"r2", "dr_%2%%1%"}}),
        endl, needed_angles);
    std::string DIHEDRAL_DEC =
        string_join(string_format(R"JIT(temp1 = %r1% ^ %r2%;
        temp2 = %r2% ^ %r3%;
        %sadf% %phi% = temp1 * temp2 / sqrtf((temp1 * temp1) * (temp2 * temp2));
        if (%phi% > 0.999999f) %phi% = 0.999999f;
        else if (%phi% < -0.999999f) %phi% = -0.999999f;
        %phi% = acosf(%phi%);
        if (temp1 * %r3% < 0.0f) %phi% = -%phi%;)JIT",
                                  {{"sadf", sadf},
                                   {"r1", "dr_%0%%1%"},
                                   {"r2", "dr_%1%%2%"},
                                   {"r3", "dr_%2%%3%"},
                                   {"phi", "phi_%0%%1%%2%%3%"}}),
                    endl, needed_dihedrals);
    std::string FORCE_DEC = string_join(
        "VECTOR force_%0% = {-E.dval[3 * %INDEX% + 3], -E.dval[3 * %INDEX% + "
        "4], -E.dval[3 * %INDEX% + 5]};",
        endl, {atom_labels});
    std::string FORCE_OUTPUT = string_join(
        R"JIT(
        if (atom_%0% < local_atom_numbers)
        {
            atomicAdd(&frc[atom_%0%].x, force_%0%.x);
            atomicAdd(&frc[atom_%0%].y, force_%0%.y);
            atomicAdd(&frc[atom_%0%].z, force_%0%.z);
            if (need_virial && atom_virial != NULL)
            {
                atomicAdd(atom_virial + atom_%0%,
                          Get_Virial_From_Force_Dis(force_%0%, crd[atom_%0%]));
            }
        })JIT",
        endl, {atom_labels});
    std::string OUTPUT = string_format(
        "if (need_atom_energy && atom_%0% < local_atom_numbers)\n        {\n   "
        "         atomicAdd(atom_ene + atom_%0%, E.val);\n        }\n        "
        "if (listed_item_energy != NULL)\n        {\n            "
        "listed_item_energy[tid] = E.val;\n        }\n        %FORCE_OUTPUT%",
        {{"0", atom_labels[0]}, {"FORCE_OUTPUT", FORCE_OUTPUT}});
    full_source_code =
        string_format(full_source_code, {{"PARM_ARGS", PARM_ARGS},
                                         {"TEMP_DEC", TEMP_DEC},
                                         {"PARM_DEC", PARM_DEC},
                                         {"CRD_DEC", CRD_DEC},
                                         {"BOND_DEC", BOND_DEC},
                                         {"ANGLE_DEC", ANGLE_DEC},
                                         {"DIHEDRAL_DEC", DIHEDRAL_DEC},
                                         {"SOURCE", source_code},
                                         {"FORCE_DEC", FORCE_DEC},
                                         {"OUTPUT", OUTPUT}});
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        force_function.error_reason += "\nSource:\n" + full_source_code;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "LISTED_FORCE::Compile",
                                       force_function.error_reason.c_str());
    }
}

void LISTED_FORCE::Initial(CONTROLLER* controller, CONECT* connectivity,
                           PAIR_DISTANCE* con_dis)
{
    FILE* fp;
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        std::string error_reason = std::string("Reason:\n\tlisted force '") +
                                   this->module_name + "' is defined, but " +
                                   this->module_name +
                                   "_in_file is not provided\n";
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                       "LISTED_FORCE::Initial",
                                       error_reason.c_str());
    }
    controller->printf("    Initializing %s\n", this->module_name);
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d", &item_numbers) != 1)
    {
        std::string error_reason = std::string(
                                       "Reason:\n\tFail to read the number of "
                                       "items of the listed force '") +
                                   this->module_name + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "LISTED_FORCE::Initial",
                                       error_reason.c_str());
    }
    Malloc_Safely((void**)&cpu_parameters,
                  sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters,
                  sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters_local,
                  sizeof(void*) * parameter_name.size());
    launch_args = std::vector<void*>(parameter_name.size() + 11);
    Device_Malloc_Safely((void**)&item_energy, sizeof(float) * item_numbers);
    Device_Malloc_Safely((void**)&sum_energy, sizeof(float));
    Device_Malloc_Safely((void**)&d_gpu_parameters,
                         sizeof(void*) * parameter_name.size());
    Device_Malloc_Safely((void**)&d_gpu_parameters_local,
                         sizeof(void*) * parameter_name.size());
    Device_Malloc_Safely((void**)&d_parameter_is_atom,
                         sizeof(int) * parameter_name.size());
    Device_Malloc_Safely((void**)&d_parameter_is_int,
                         sizeof(int) * parameter_name.size());
    Device_Malloc_Safely((void**)&d_local_item_numbers, sizeof(int));
    for (int j = 0; j < parameter_name.size(); j++)
    {
        if (parameter_type[j] == "int")
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(int) * item_numbers);
            Device_Malloc_Safely((void**)&gpu_parameters_local[j],
                                 sizeof(int) * item_numbers);
        }
        else
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(float) * item_numbers);
            Device_Malloc_Safely((void**)&gpu_parameters_local[j],
                                 sizeof(float) * item_numbers);
        }
        launch_args[j] = gpu_parameters + j;
    }
    for (int i = 0; i < item_numbers; i++)
    {
        for (int j = 0; j < parameter_name.size(); j++)
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
                    std::string(
                        "Reason:\n\tFail to read the parameters of the listed "
                        "force '") +
                    this->module_name + "'\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "LISTED_FORCE::Initial",
                                               error_reason.c_str());
            }
        }
    }
    fclose(fp);
    int connected_a = -1, connected_b = -1, constran_id = -1;
    for (int j = 0; j < parameter_name.size(); j++)
    {
        if (parameter_type[j] == "int")
        {
            Device_Malloc_And_Copy_Safely((void**)gpu_parameters + j,
                                          cpu_parameters[j],
                                          sizeof(int) * item_numbers);
        }
        else
        {
            Device_Malloc_And_Copy_Safely((void**)gpu_parameters + j,
                                          cpu_parameters[j],
                                          sizeof(float) * item_numbers);
        }
        if (connected_atoms.size() > 0)
        {
            std::string atom_("atom_");
            if (atom_ + connected_atoms[0] == parameter_name[j])
            {
                connected_a = j;
            }
            else if (atom_ + connected_atoms[1] == parameter_name[j])
            {
                connected_b = j;
            }
        }
        if (this->constrain_distance.size() > 0)
        {
            if (this->constrain_distance == parameter_name[j])
            {
                constran_id = j;
            }
        }
    }
    deviceMemcpy(d_gpu_parameters, gpu_parameters,
                 sizeof(void*) * parameter_name.size(),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_gpu_parameters_local, gpu_parameters_local,
                 sizeof(void*) * parameter_name.size(),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_parameter_is_atom, parameter_is_atom.data(),
                 sizeof(int) * parameter_is_atom.size(),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_parameter_is_int, parameter_is_int.data(),
                 sizeof(int) * parameter_is_int.size(),
                 deviceMemcpyHostToDevice);
    deviceMemset(d_local_item_numbers, 0, sizeof(int));
    if (connected_atoms.size() > 0)
    {
        if (connected_a >= 0 && connected_b >= 0)
        {
            for (int i = 0; i < item_numbers; i++)
            {
                int atom_a = ((int*)cpu_parameters[connected_a])[i];
                int atom_b = ((int*)cpu_parameters[connected_b])[i];
                connectivity[0][atom_a].insert(atom_b);
                connectivity[0][atom_b].insert(atom_a);
                if (constran_id >= 0)
                {
                    if (atom_a < atom_b)
                    {
                        con_dis[0][std::pair<int, int>(atom_a, atom_b)] =
                            ((float*)cpu_parameters[constran_id])[i];
                    }
                    else
                    {
                        con_dis[0][std::pair<int, int>(atom_b, atom_a)] =
                            ((float*)cpu_parameters[constran_id])[i];
                    }
                }
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "LISTED_FORCE::Initial",
                "Reason:\n\tthe name of the connected atoms is not right\n");
        }
    }
    if (this->constrain_distance.size() > 0 && constran_id < 0)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "LISTED_FORCE::Initial",
            "Reason:\n\tthe name of the constrain distance is not right\n");
    }
    local_item_numbers = 0;
    local_atom_numbers = 0;
    use_domain_decomposition = 0;
}

void LISTED_FORCE::Get_Local(int* atom_local, int local_atom_numbers,
                             int ghost_numbers, char* atom_local_label,
                             int* atom_local_id)
{
    if (!item_numbers) return;
    (void)atom_local;
    (void)ghost_numbers;
    use_domain_decomposition = 1;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(listed_force_get_local_device, 1, 1, 0, NULL,
                         item_numbers, parameter_name.size(),
                         d_parameter_is_int, d_parameter_is_atom,
                         d_gpu_parameters, d_gpu_parameters_local,
                         atom_local_label, atom_local_id, d_local_item_numbers);
    deviceMemcpy(&local_item_numbers, d_local_item_numbers, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void LISTED_FORCE::Step_Print(CONTROLLER* controller)
{
    if (item_numbers == 0) return;
    if (CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size) return;
    h_energy = last_energy;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_energy, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print(this->module_name, &h_energy, true);
}

void LISTED_FORCE::Compute_Force(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                                 LTMatrix3 rcell, VECTOR* frc, int need_energy,
                                 float* atom_energy, int need_pressure,
                                 LTMatrix3* atom_virial)
{
    last_atom_numbers = atom_numbers;
    int interaction_numbers =
        use_domain_decomposition ? local_item_numbers : item_numbers;
    if (interaction_numbers <= 0)
    {
        last_energy = 0.0f;
        return;
    }
    VECTOR box_length = {cell.a11, cell.a22, cell.a33};
    (void)rcell;
    void** parameter_ptr_array =
        use_domain_decomposition ? gpu_parameters_local : gpu_parameters;
    for (int j = 0; j < parameter_name.size(); j++)
    {
        launch_args[j] = parameter_ptr_array + j;
    }
    int local_atom_bound =
        use_domain_decomposition ? local_atom_numbers : last_atom_numbers;
    int ONLY_ENERGY = 0;
    int need_atom_energy = need_energy ? 1 : 0;
    int need_virial = need_pressure ? 1 : 0;
    float* listed_item_energy = need_energy ? item_energy : NULL;
    if (listed_item_energy != NULL)
    {
        deviceMemset(listed_item_energy, 0,
                     sizeof(float) * interaction_numbers);
    }
    launch_args[parameter_name.size()] = &crd;
    launch_args[parameter_name.size() + 1] = &box_length;
    launch_args[parameter_name.size() + 2] = &frc;
    launch_args[parameter_name.size() + 3] = &atom_energy;
    launch_args[parameter_name.size() + 4] = &atom_virial;
    launch_args[parameter_name.size() + 5] = &listed_item_energy;
    launch_args[parameter_name.size() + 6] = &local_atom_bound;
    launch_args[parameter_name.size() + 7] = &need_atom_energy;
    launch_args[parameter_name.size() + 8] = &need_virial;
    launch_args[parameter_name.size() + 9] = &ONLY_ENERGY;
    launch_args[parameter_name.size() + 10] = &interaction_numbers;
    force_function({(interaction_numbers + 1023u) / 1024u, 1u, 1u},
                   {1024u, 1u, 1u}, NULL, 0, launch_args);
    if (need_energy)
    {
        Sum_Of_List(item_energy, sum_energy, interaction_numbers);
        deviceMemcpy(&last_energy, sum_energy, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        last_energy = 0.0f;
    }
}

float LISTED_FORCE::Get_Energy(VECTOR* crd, VECTOR box_length)
{
    int interaction_numbers =
        use_domain_decomposition ? local_item_numbers : item_numbers;
    if (interaction_numbers <= 0) return 0.0f;
    deviceMemset(this->item_energy, 0, sizeof(float) * item_numbers);
    int TRUE_ = 1;
    VECTOR* NULL_VECTOR = NULL;
    LTMatrix3* NULL_TENSOR = NULL;
    float* NULL_FLOAT = NULL;
    int ZERO = 0;
    void** parameter_ptr_array =
        use_domain_decomposition ? gpu_parameters_local : gpu_parameters;
    for (int j = 0; j < parameter_name.size(); j++)
    {
        launch_args[j] = parameter_ptr_array + j;
    }
    int local_atom_bound =
        use_domain_decomposition ? local_atom_numbers : last_atom_numbers;
    launch_args[parameter_name.size()] = &crd;
    launch_args[parameter_name.size() + 1] = &box_length;
    launch_args[parameter_name.size() + 2] = &NULL_VECTOR;
    launch_args[parameter_name.size() + 3] = &item_energy;
    launch_args[parameter_name.size() + 4] = &NULL_TENSOR;
    launch_args[parameter_name.size() + 5] = &NULL_FLOAT;
    launch_args[parameter_name.size() + 6] = &local_atom_bound;
    launch_args[parameter_name.size() + 7] = &ZERO;
    launch_args[parameter_name.size() + 8] = &ZERO;
    launch_args[parameter_name.size() + 9] = &TRUE_;
    launch_args[parameter_name.size() + 10] = &interaction_numbers;

    force_function({(interaction_numbers + 1023u) / 1024u, 1u, 1u},
                   {1024u, 1u, 1u}, NULL, 0, launch_args);
    Sum_Of_List(item_energy, sum_energy, interaction_numbers);
    float h_energy = NAN;
    deviceMemcpy(&h_energy, sum_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    return h_energy;
}
