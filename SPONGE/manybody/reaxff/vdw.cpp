#include "vdw.h"

static const int p_rvdw = 0;
static const int p_epsilon = 1;
static const int p_alpha = 2;
static const int p_gamma_w = 3;
static const int PARAM_STRIDE = 8;

template <int N>
__device__ __forceinline__ SADfloat<N> reax_vdw_energy_sad(SADfloat<N> r,
                                                           const float* param,
                                                           float cutoff,
                                                           float p_vdw1)
{
    float rvdw = param[p_rvdw];
    float epsilon = param[p_epsilon];
    float alpha = param[p_alpha];
    float gamma_w = param[p_gamma_w];

    if (r.val > cutoff) return SADfloat<N>(0.0f);

    SADfloat<N> x = r / SADfloat<N>(cutoff);
    SADfloat<N> x4 = x * x * x * x;
    SADfloat<N> x5 = x4 * x;
    SADfloat<N> x6 = x5 * x;
    SADfloat<N> x7 = x6 * x;

    SADfloat<N> tap = SADfloat<N>(1.0f) - SADfloat<N>(35.0f) * x4 +
                      SADfloat<N>(84.0f) * x5 - SADfloat<N>(70.0f) * x6 +
                      SADfloat<N>(20.0f) * x7;

    float inv_gamma = 1.0f / gamma_w;
    SADfloat<N> inv_gamma_p = powf(SADfloat<N>(inv_gamma), SADfloat<N>(p_vdw1));
    SADfloat<N> r_p = powf(r, SADfloat<N>(p_vdw1));
    SADfloat<N> shielded_r =
        powf(r_p + inv_gamma_p, SADfloat<N>(1.0f / p_vdw1));

    SADfloat<N> exp_term = alpha * (1.0f - shielded_r / rvdw);
    SADfloat<N> term1 = expf(exp_term);
    SADfloat<N> term2 = -2.0f * expf(0.5f * exp_term);

    return tap * epsilon * (term1 + term2);
}

static __global__ void REAXFF_VDW_Force_CUDA(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* nl, int* atom_types,
    float* params, int ntypes, float cutoff, float p_vdw1, float* atom_energy,
    LTMatrix3* atom_virial, float* d_energy_sum)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        VECTOR ri = crd[i];
        ATOM_GROUP nl_i = nl[i];
        VECTOR fi = {0, 0, 0};
        LTMatrix3 vi = {0, 0, 0, 0, 0, 0};
        float en_i = 0;

        for (int jj = 0; jj < nl_i.atom_numbers; jj++)
        {
            int j = nl_i.atom_serial[jj];
            int type_j = atom_types[j];

            if (j <= i) continue;

            VECTOR rj = crd[j];
            VECTOR drij =
                Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            float rij = norm3df(drij.x, drij.y, drij.z);

            if (rij >= cutoff) continue;

            int param_idx = (type_i * ntypes + type_j) * PARAM_STRIDE;
            const float* param = params + param_idx;

            SADfloat<1> rij_sad(rij, 0);
            SADfloat<1> energy_sad =
                reax_vdw_energy_sad(rij_sad, param, cutoff, p_vdw1);

            float force_mag = -energy_sad.dval[0] / rij;
            VECTOR fij = {force_mag * drij.x, force_mag * drij.y,
                          force_mag * drij.z};

            fi.x += fij.x;
            fi.y += fij.y;
            fi.z += fij.z;

            atomicAdd(&frc[j].x, -fij.x);
            atomicAdd(&frc[j].y, -fij.y);
            atomicAdd(&frc[j].z, -fij.z);

            if (atom_virial)
            {
                vi = vi + Get_Virial_From_Force_Dis(fij, drij);
            }

            if (atom_energy)
            {
                en_i += energy_sad.val;
                atomicAdd(d_energy_sum, energy_sad.val);
            }
        }

        atomicAdd(&frc[i].x, fi.x);
        atomicAdd(&frc[i].y, fi.y);
        atomicAdd(&frc[i].z, fi.z);

        if (atom_energy)
        {
            atom_energy[i] += en_i;
        }
        if (atom_virial)
        {
            atomicAdd(atom_virial + i, vi);
        }
    }
}

void REAXFF_VDW::Initial(CONTROLLER* controller, int atom_numbers,
                         const char* module_name, bool* need_full_nl_flag)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;
    if (!controller->Command_Exist(module_name, "in_file")) return;

    controller->printf("START INITIALIZING REAXFF VDW FORCE\n");
    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_VDW IS NOT INITIALIZED (missing input files)\n\n");
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
                                       "REAXFF_VDW::Initial", error_msg);
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

    std::vector<float> gen_params;
    gen_params.reserve(n_gen_params);
    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
        float val;
        if (sscanf(line, "%f", &val) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse general parameter at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        gen_params.push_back(val);
    }
    if (gen_params.size() <= 28)
    {
        throw_bad_format(parameter_in_file,
                         "missing general parameter p_vdw1 at index 29");
    }
    this->p_vdw1 = gen_params[28];

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
    std::vector<float> rvdw(n_atom_types);
    std::vector<float> epsilon(n_atom_types);
    std::vector<float> alpha(n_atom_types);
    std::vector<float> gamma_w(n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 1");
        char element_name[16];
        float ro_sigma, valency, mass, rvdw_val, epsilon_val, gamma_val, ro_pi,
            valency_e;
        if (sscanf(line, "%15s %f %f %f %f %f %f %f %f", element_name,
                   &ro_sigma, &valency, &mass, &rvdw_val, &epsilon_val,
                   &gamma_val, &ro_pi, &valency_e) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(element_name)] = i;

        rvdw[i] = rvdw_val;
        epsilon[i] = epsilon_val;

        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 2");
        float alpha_val, gamma_w_val;
        if (sscanf(line, "%f %f", &alpha_val, &gamma_w_val) != 2)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        alpha[i] = alpha_val;
        gamma_w[i] = gamma_w_val;

        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 4");
    }

    Malloc_Safely((void**)&h_twobody_params,
                  sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);
    memset(h_twobody_params, 0,
           sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);

    for (int i = 0; i < n_atom_types; i++)
    {
        for (int j = 0; j < n_atom_types; j++)
        {
            int idx = (i * n_atom_types + j) * PARAM_STRIDE;
            float rvdw_ij = 2.0f * sqrtf(rvdw[i] * rvdw[j]);
            float epsilon_ij = sqrtf(epsilon[i] * epsilon[j]);
            float alpha_ij = sqrtf(alpha[i] * alpha[j]);
            float gamma_w_ij = sqrtf(gamma_w[i] * gamma_w[j]);

            h_twobody_params[idx + p_rvdw] = rvdw_ij;
            h_twobody_params[idx + p_epsilon] = epsilon_ij;
            h_twobody_params[idx + p_alpha] = alpha_ij;
            h_twobody_params[idx + p_gamma_w] = gamma_w_ij;
        }
    }

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    read_line_or_throw(fp_p, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 1");
        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 2");
    }

    if (fgets(line, 1024, fp_p) != NULL)
    {
        int n_off = 0;
        if (sscanf(line, "%d", &n_off) != 1 || n_off < 0)
        {
            throw_bad_format(parameter_in_file,
                             "failed to parse number of off-diagonal terms");
        }
        for (int off = 0; off < n_off; off++)
        {
            read_line_or_throw(fp_p, parameter_in_file,
                               "off-diagonal parameter entry");
            int t1, t2;
            float dij, rvdw_od, alfa_od, ro_sigma_od, ro_pi_od, ro_pipi_od;
            int read_cnt = sscanf(line, "%d %d %f %f %f %f %f %f", &t1, &t2,
                                  &dij, &rvdw_od, &alfa_od, &ro_sigma_od,
                                  &ro_pi_od, &ro_pipi_od);

            if (read_cnt < 8)
            {
                char reason[512];
                sprintf(
                    reason,
                    "failed to parse off-diagonal parameter entry at index %d",
                    off + 1);
                throw_bad_format(parameter_in_file, reason);
            }
            int idx1 = t1 - 1;
            int idx2 = t2 - 1;
            if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
                idx2 >= n_atom_types)
            {
                char reason[512];
                sprintf(reason,
                        "off-diagonal atom type index out of range at index %d",
                        off + 1);
                throw_bad_format(parameter_in_file, reason);
            }

            int pair_idx1 = (idx1 * n_atom_types + idx2) * PARAM_STRIDE;
            int pair_idx2 = (idx2 * n_atom_types + idx1) * PARAM_STRIDE;

            if (dij > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_epsilon] =
                    h_twobody_params[pair_idx2 + p_epsilon] = dij;
            }
            if (rvdw_od > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_rvdw] =
                    h_twobody_params[pair_idx2 + p_rvdw] = 2.0f * rvdw_od;
            }
            if (alfa_od > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_alpha] =
                    h_twobody_params[pair_idx2 + p_alpha] = alfa_od;
            }
        }
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
    controller->Step_Print_Initial("REAXFF_VDW", "%14.7e");
    controller->printf("END INITIALIZING REAXFF VDW FORCE\n\n");
}

void REAXFF_VDW::REAXFF_VDW_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* nl, const float cutoff,
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

    Launch_Device_Kernel(REAXFF_VDW_Force_CUDA, gridSize, blockSize, 0, NULL,
                         atom_numbers, crd, frc, boundary, nl, d_atom_type,
                         d_twobody_params, atom_type_numbers, cutoff,
                         this->p_vdw1, atom_energy,
                         need_virial ? atom_virial : NULL, d_energy_sum);
}

void REAXFF_VDW::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_VDW", h_energy_sum, true);
}
