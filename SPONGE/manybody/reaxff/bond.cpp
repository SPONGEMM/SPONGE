#include "bond.h"

#include "bond_order.h"  // for find_bond_index

static const int De_s = 0;
static const int De_p = 1;
static const int De_PP = 2;
static const int p_be1 = 3;
static const int p_be2 = 4;
static const int PARAM_STRIDE = 5;

template <int N>
__device__ __forceinline__ SADfloat<N> reax_bond_energy_sad(SADfloat<N> BO_s,
                                                            SADfloat<N> BO_pi,
                                                            SADfloat<N> BO_pi2,
                                                            const float* param)
{
    float De_s_val = param[De_s];
    float De_p_val = param[De_p];
    float De_PP_val = param[De_PP];
    float p_be1_val = param[p_be1];
    float p_be2_val = param[p_be2];

    SADfloat<N> pow_BOs_be2 = powf(BO_s, p_be2_val);
    SADfloat<N> exp_be12 = expf(p_be1_val * (1.0f - pow_BOs_be2));

    SADfloat<N> ebond =
        -De_s_val * BO_s * exp_be12 - De_p_val * BO_pi - De_PP_val * BO_pi2;
    return ebond;
}

static __global__ void REAXFF_Bond_Force_CUDA(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
    int* atom_types, float* params, int ntypes, float* bo_s, float* bo_pi,
    float* bo_pi2, float* d_dE_dBO_s, float* d_dE_dBO_pi, float* d_dE_dBO_pi2,
    float* atom_energy, LTMatrix3* atom_virial, float* d_energy_sum,
    const int* bond_count, const int* bond_offset, const int* bond_nbr,
    const int* bond_idx_arr)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        VECTOR ri = crd[i];
        ATOM_GROUP nl_i = nl[i];
        float en_i = 0;

        for (int jj = 0; jj < nl_i.atom_numbers; jj++)
        {
            int j = nl_i.atom_serial[jj];

            if (j <= i) continue;

            int b = find_bond_index(i, j, bond_count, bond_offset, bond_nbr,
                                    bond_idx_arr);
            if (b < 0) continue;

            int param_idx = type_i * ntypes + atom_types[j];
            const float* param = params + param_idx * PARAM_STRIDE;

            float BO_s_ij = bo_s[b];
            float BO_pi_ij = bo_pi[b];
            float BO_pi2_ij = bo_pi2[b];

            if (BO_s_ij + BO_pi_ij + BO_pi2_ij < 1e-10f) continue;

            SADfloat<3> BO_s_sad(BO_s_ij, 0);
            SADfloat<3> BO_pi_sad(BO_pi_ij, 1);
            SADfloat<3> BO_pi2_sad(BO_pi2_ij, 2);

            SADfloat<3> energy_sad =
                reax_bond_energy_sad(BO_s_sad, BO_pi_sad, BO_pi2_sad, param);

            atomicAdd(&d_dE_dBO_s[b], energy_sad.dval[0]);
            atomicAdd(&d_dE_dBO_pi[b], energy_sad.dval[1]);
            atomicAdd(&d_dE_dBO_pi2[b], energy_sad.dval[2]);

            if (atom_energy)
            {
                en_i += energy_sad.val;
                atomicAdd(d_energy_sum, energy_sad.val);
            }
        }

        if (atom_energy)
        {
            atom_energy[i] += en_i;
        }
    }
}

void REAXFF_BOND::Initial(CONTROLLER* controller, int atom_numbers,
                          const char* module_name, bool* need_full_nl_flag)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;
    if (!controller->Command_Exist(module_name, "in_file")) return;

    controller->printf("START INITIALIZING REAXFF BOND FORCE\n");
    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_BOND IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_BOND::Initial", error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp_p, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "general parameter count line");
    int n_gen_params = 0;
    if (sscanf(line, "%d", &n_gen_params) != 1 || n_gen_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }
    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
    }

    read_line_or_throw(fp_p, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1 || n_atom_types <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 1");
        char element_name[16];
        if (sscanf(line, "%15s", element_name) != 1)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(element_name)] = i;
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 2");
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 4");
    }

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }

    Malloc_Safely((void**)&h_twobody_params,
                  sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);
    memset(h_twobody_params, 0,
           sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);
    read_line_or_throw(fp_p, parameter_in_file, "bond parameter header line");

    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 1");
        int t1, t2;
        float De_s_val, De_p_val, De_pp_val, p_be1_val, p_bo5_val, v13cor_val,
            p_bo6_val, p_ovun1_val;
        if (sscanf(line, "%d %d %f %f %f %f %f %f %f %f", &t1, &t2, &De_s_val,
                   &De_p_val, &De_pp_val, &p_be1_val, &p_bo5_val, &v13cor_val,
                   &p_bo6_val, &p_ovun1_val) != 10)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse bond parameter block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        int idx1 = t1 - 1;
        int idx2 = t2 - 1;
        if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
            idx2 >= n_atom_types)
        {
            char reason[512];
            sprintf(reason,
                    "bond type index out of range at bond parameter index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 2");
        float p_be2_val, p_bo3_val, p_bo4_val, unused1, p_bo1_val, p_bo2_val,
            ovc_val;
        if (sscanf(line, "%f %f %f %f %f %f %f", &p_be2_val, &p_bo3_val,
                   &p_bo4_val, &unused1, &p_bo1_val, &p_bo2_val, &ovc_val) != 7)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse bond parameter block line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        int pair_idx1 = (idx1 * n_atom_types + idx2) * PARAM_STRIDE;
        int pair_idx2 = (idx2 * n_atom_types + idx1) * PARAM_STRIDE;

        h_twobody_params[pair_idx1 + De_s] =
            h_twobody_params[pair_idx2 + De_s] = De_s_val;
        h_twobody_params[pair_idx1 + De_p] =
            h_twobody_params[pair_idx2 + De_p] = De_p_val;
        h_twobody_params[pair_idx1 + De_PP] =
            h_twobody_params[pair_idx2 + De_PP] = De_pp_val;
        h_twobody_params[pair_idx1 + p_be1] =
            h_twobody_params[pair_idx2 + p_be1] = p_be1_val;
        h_twobody_params[pair_idx1 + p_be2] =
            h_twobody_params[pair_idx2 + p_be2] = p_be2_val;
    }
    fclose(fp_p);

    Device_Malloc_And_Copy_Safely(
        (void**)&d_twobody_params, h_twobody_params,
        sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);

    FILE* fp_t;
    Open_File_Safely(&fp_t, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp_t, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }

    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        char type_name[16];
        read_line_or_throw(fp_t, type_in_file, "atom type entry line");
        if (sscanf(line, "%15s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        std::string type_str(type_name);
        auto iter = type_map.find(type_str);
        if (iter != type_map.end())
        {
            h_atom_type[i] = iter->second;
        }
        else
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s",
                    type_name, parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
    }
    fclose(fp_t);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_atom, sizeof(float) * atom_numbers);
    deviceMemset(d_energy_sum, 0, sizeof(float));
    deviceMemset(d_energy_atom, 0, sizeof(float) * atom_numbers);

    if (need_full_nl_flag != NULL) *need_full_nl_flag = true;

    is_initialized = true;
    controller->Step_Print_Initial("REAXFF_BOND", "%14.7e");
    controller->printf("END INITIALIZING REAXFF BOND FORCE\n\n");
}

void REAXFF_BOND::REAXFF_Bond_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial)
{
    if (!is_initialized) return;

    if (need_atom_energy)
    {
        deviceMemset(d_energy_sum, 0, sizeof(float));
        if (atom_energy)
            deviceMemset(d_energy_atom, 0, sizeof(float) * atom_numbers);
    }

    dim3 blockSize(128);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);

    Launch_Device_Kernel(REAXFF_Bond_Force_CUDA, gridSize, blockSize, 0, NULL,
                         atom_numbers, crd, frc, cell, rcell, nl, d_atom_type,
                         d_twobody_params, atom_type_numbers, d_bo_s, d_bo_pi,
                         d_bo_pi2, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2,
                         atom_energy, atom_virial, d_energy_sum, d_bond_count,
                         d_bond_offset, d_bond_nbr, d_bond_idx);
}

void REAXFF_BOND::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_BOND", h_energy_sum, true);
}
