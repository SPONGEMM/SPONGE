#include "eam.h"

#include <algorithm>
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct NativeEAMDefinition
{
    int atom_type_count = 0;
    int nrho = 0;
    int nr = 0;
    float drho = 0.0f;
    float dr = 0.0f;
    float cut = 0.0f;
    std::vector<float> embed;
    std::vector<float> electron_density;
    std::vector<float> pair_potential;
    std::vector<int> atom_type;
};

std::vector<float> Read_EAM_Float_Array(
    HighFive::File* file, const std::string& path,
    const std::vector<std::size_t>& expected_dimensions)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions != expected_dimensions)
    {
        throw std::runtime_error(path + " has invalid dimensions");
    }
    std::size_t count = 1;
    for (const std::size_t dimension : dimensions) count *= dimension;
    std::vector<float> values(count);
    if (H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, values.data()) < 0)
    {
        throw std::runtime_error("failed to read " + path);
    }
    return values;
}

bool Read_H5_EAM_Input(CONTROLLER* controller, const char* module_name,
                       int atom_numbers, NativeEAMDefinition* definition)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* root = "/manybody/eam";
    if (strcmp(module_name, "EAM") != 0 ||
        controller->Command_Exist(module_name, "in_file") ||
        !controller->Command_Exist(input_key))
    {
        return false;
    }
    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(root)) return false;
        NativeEAMDefinition result;
        file.getDataSet(std::string(root) + "/atom_type_count")
            .read(result.atom_type_count);
        file.getDataSet(std::string(root) + "/nrho").read(result.nrho);
        file.getDataSet(std::string(root) + "/nr").read(result.nr);
        file.getDataSet(std::string(root) + "/drho").read(result.drho);
        file.getDataSet(std::string(root) + "/dr").read(result.dr);
        file.getDataSet(std::string(root) + "/cut").read(result.cut);
        if (result.atom_type_count <= 0 || result.nrho < 2 || result.nr < 2 ||
            result.drho <= 0.0f || result.dr <= 0.0f || result.cut <= 0.0f)
        {
            throw std::runtime_error(
                "/manybody/eam table dimensions and spacings must be "
                "positive");
        }
        const std::size_t type_count =
            static_cast<std::size_t>(result.atom_type_count);
        result.embed = Read_EAM_Float_Array(
            &file, std::string(root) + "/embed/value",
            {type_count, static_cast<std::size_t>(result.nrho)});
        result.electron_density = Read_EAM_Float_Array(
            &file, std::string(root) + "/electron_density/value",
            {type_count, static_cast<std::size_t>(result.nr)});
        result.pair_potential = Read_EAM_Float_Array(
            &file, std::string(root) + "/pair_potential/value",
            {type_count, type_count, static_cast<std::size_t>(result.nr)});
        if (file.exist(std::string(root) + "/atom_type"))
        {
            HighFive::DataSet dataset =
                file.getDataSet(std::string(root) + "/atom_type");
            const auto dimensions = dataset.getSpace().getDimensions();
            if (dimensions.size() != 1 ||
                dimensions[0] != static_cast<std::size_t>(atom_numbers))
            {
                throw std::runtime_error(
                    "/manybody/eam/atom_type must match runtime atom count");
            }
            dataset.read(result.atom_type);
        }
        else if (result.atom_type_count == 1)
        {
            result.atom_type.assign(static_cast<std::size_t>(atom_numbers), 0);
        }
        else
        {
            throw std::runtime_error(
                "multi-type /manybody/eam requires /manybody/eam/atom_type");
        }
        for (const int type : result.atom_type)
        {
            if (type < 0 || type >= result.atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/eam/atom_type contains an out-of-range type");
            }
        }
        *definition = std::move(result);
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed EAM input from ") +
            root + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Read_H5_EAM_Input", message.c_str());
    }
    return false;
}
}  // namespace

template <int N>
static __device__ __forceinline__ SADfloat<N> EAM_Interpolate(float* table,
                                                              int n,
                                                              float delta,
                                                              SADfloat<N> x)
{
    SADfloat<N> index = x / delta;
    int i = (int)index.val;

    if (i < 0) i = 0;
    if (i >= n - 1) i = n - 2;

    int idx0 = (i - 1 < 0) ? 0 : i - 1;
    int idx1 = i;
    int idx2 = i + 1;
    int idx3 = (i + 2 >= n) ? n - 1 : i + 2;

    float p0 = table[idx0];
    float p1 = table[idx1];
    float p2 = table[idx2];
    float p3 = table[idx3];

    SADfloat<N> t = index - (float)i;
    SADfloat<N> t2 = t * t;
    SADfloat<N> t3 = t2 * t;

    return 0.5f * ((2.0f * p1) + (-p0 + p2) * t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

static __global__ void EAM_Calculate_Rho_CUDA(
    const int atom_numbers, const VECTOR* crd, const Boundary boundary,
    const ATOM_GROUP* nl, int* atom_types, float* rho_table, int ntypes, int nr,
    float dr, float cut, float* d_rho)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        float local_rho = 0;
        VECTOR ri = crd[atom_i];
        ATOM_GROUP nl_i = nl[atom_i];

        for (int j = 0; j < nl_i.atom_numbers; j++)
        {
            int atom_j = nl_i.atom_serial[j];
            int type_j = atom_types[atom_j];

            VECTOR rj = crd[atom_j];
            VECTOR drij =
                Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            float rij = norm3df(drij.x, drij.y, drij.z);
            if (rij < cut)
            {
                // Pointer to rho table for type_j
                float* current_rho_table = rho_table + type_j * nr;

                float index_f = rij / dr;
                int k = (int)index_f;
                if (k < 0) k = 0;
                if (k >= nr - 1) k = nr - 2;

                int idx0 = (k - 1 < 0) ? 0 : k - 1;
                int idx1 = k;
                int idx2 = k + 1;
                int idx3 = (k + 2 >= nr) ? nr - 1 : k + 2;

                float p0 = current_rho_table[idx0];
                float p1 = current_rho_table[idx1];
                float p2 = current_rho_table[idx2];
                float p3 = current_rho_table[idx3];

                float t = index_f - k;
                float t2 = t * t;
                float t3 = t2 * t;

                float rho_val =
                    0.5f * ((2.0f * p1) + (-p0 + p2) * t +
                            (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                            (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);

                local_rho += rho_val;
            }
        }
        d_rho[atom_i] = local_rho;
    }
}

template <bool need_energy>
static __global__ void EAM_Calculate_DF_Rho_CUDA(
    const int atom_numbers, float* embed_table, int* atom_types, int nrho,
    float drho, float* d_rho, float* d_df_drho, float* atom_energy,
    float* d_energy_sum)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        int type_i = atom_types[atom_i];
        float* current_embed_table = embed_table + type_i * nrho;

        SADfloat<1> rho_i(d_rho[atom_i], 0);
        SADfloat<1> F_i =
            EAM_Interpolate(current_embed_table, nrho, drho, rho_i);

        d_df_drho[atom_i] = F_i.dval[0];
        if (need_energy)
        {
            atom_energy[atom_i] += F_i.val;
            atomicAdd(d_energy_sum, F_i.val);
        }
    }
}

template <bool need_energy, bool need_virial>
static __global__ void EAM_Calculate_Force_CUDA(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* nl, int* atom_types,
    float* rho_table, float* phi_table, int ntypes, int nr, float dr, float cut,
    float* d_rho, float* d_df_drho, float* atom_energy, LTMatrix3* atom_virial,
    float* d_energy_sum)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR ri = crd[atom_i];
        VECTOR fi = {0, 0, 0};
        LTMatrix3 vi = {0, 0, 0, 0, 0, 0};
        ATOM_GROUP nl_i = nl[atom_i];
        float dfi = d_df_drho[atom_i];
        int type_i = atom_types[atom_i];

        for (int j = 0; j < nl_i.atom_numbers; j++)
        {
            int atom_j = nl_i.atom_serial[j];
            int type_j = atom_types[atom_j];

            VECTOR rj = crd[atom_j];
            VECTOR drij =
                Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            float rij_val = norm3df(drij.x, drij.y, drij.z);
            if (rij_val < cut)
            {
                SADfloat<1> r(rij_val, 0);

                float* current_phi_table =
                    phi_table + (type_i * ntypes + type_j) * nr;
                SADfloat<1> phi = EAM_Interpolate(current_phi_table, nr, dr, r);

                float* rho_table_j = rho_table + type_j * nr;
                SADfloat<1> rho_at_j = EAM_Interpolate(rho_table_j, nr, dr, r);

                float* rho_table_i = rho_table + type_i * nr;
                SADfloat<1> rho_at_i = EAM_Interpolate(rho_table_i, nr, dr, r);
                float dfj = d_df_drho[atom_j];
                SADfloat<1> E_force_term =
                    phi + dfi * rho_at_j + dfj * rho_at_i;

                VECTOR fij = -E_force_term.dval[0] / rij_val * drij;
                fi.x += fij.x;
                fi.y += fij.y;
                fi.z += fij.z;

                if (need_energy)
                {
                    atom_energy[atom_i] += 0.5f * phi.val;
                    atomicAdd(d_energy_sum, 0.5f * phi.val);
                }
                if (need_virial)
                {
                    vi = vi + 0.5f * Get_Virial_From_Force_Dis(fij, drij);
                }
            }
        }
        frc[atom_i].x += fi.x;
        frc[atom_i].y += fi.y;
        frc[atom_i].z += fi.z;
        if (need_virial) atom_virial[atom_i] = atom_virial[atom_i] + vi;
    }
}

void EAM_INFORMATION::EAM_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* nl, const int need_atom_energy,
    float* atom_energy, const int need_virial, LTMatrix3* atom_virial)
{
    if (!is_initialized) return;

    int threads = 256;
    int blocks = (atom_numbers + threads - 1) / threads;

    float* rho_table = this->d_electron_density;
    int nr_local = this->nr;
    float dr_local = this->dr;
    float cut_local = this->cut;
    int* atom_type_local = this->d_atom_type;
    float* d_rho_local = this->d_rho;
    int ntypes = this->atom_type_numbers;

    Launch_Device_Kernel(EAM_Calculate_Rho_CUDA, blocks, threads, 0, NULL,
                         atom_numbers, crd, boundary, nl, atom_type_local,
                         rho_table, ntypes, nr_local, dr_local, cut_local,
                         d_rho_local);

    if (need_atom_energy) deviceMemset(d_energy_sum, 0, sizeof(float));

    float* embed_table = this->d_embed;
    int nrho_local = this->nrho;
    float drho_local = this->drho;
    float* phi_table = this->d_pair_potential;
    float* d_df_drho_local = this->d_df_drho;
    float* d_energy_sum_local = this->d_energy_sum;

    auto df_rho_kernel = EAM_Calculate_DF_Rho_CUDA<false>;
    if (need_atom_energy) df_rho_kernel = EAM_Calculate_DF_Rho_CUDA<true>;
    Launch_Device_Kernel(df_rho_kernel, blocks, threads, 0, NULL, atom_numbers,
                         embed_table, atom_type_local, nrho_local, drho_local,
                         d_rho_local, d_df_drho_local, atom_energy,
                         d_energy_sum_local);

    auto force_kernel = EAM_Calculate_Force_CUDA<false, false>;
    if (need_atom_energy && need_virial)
    {
        force_kernel = EAM_Calculate_Force_CUDA<true, true>;
    }
    else if (need_atom_energy)
    {
        force_kernel = EAM_Calculate_Force_CUDA<true, false>;
    }
    else if (need_virial)
    {
        force_kernel = EAM_Calculate_Force_CUDA<false, true>;
    }
    Launch_Device_Kernel(force_kernel, blocks, threads, 0, NULL, atom_numbers,
                         crd, frc, boundary, nl, atom_type_local, rho_table,
                         phi_table, ntypes, nr_local, dr_local, cut_local,
                         d_rho_local, d_df_drho_local, atom_energy, atom_virial,
                         d_energy_sum_local);
}

void EAM_INFORMATION::Read_Funcfl(FILE* fp, CONTROLLER* controller)
{
    this->atom_type_numbers = 1;
    char line[CHAR_LENGTH_MAX];

    int atomic_number;
    float mass, lattice_constant;
    char lattice_type[CHAR_LENGTH_MAX];
    if (fscanf(fp, "%d %f %f %s\n", &atomic_number, &mass, &lattice_constant,
               lattice_type) != 4)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Funcfl",
                                       "Failed to read header info");
    }

    if (fscanf(fp, "%d %f %d %f %f\n", &nrho, &drho, &nr, &dr, &cut) != 5)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Funcfl",
                                       "Failed to read table parameters");
    }

    controller->printf(
        "    EAM (Funcfl) params: nrho=%d, drho=%f, nr=%d, dr=%f, cut=%f\n",
        nrho, drho, nr, dr, cut);

    Malloc_Safely((void**)&h_embed, sizeof(float) * nrho * atom_type_numbers);
    Malloc_Safely((void**)&h_electron_density,
                  sizeof(float) * nr * atom_type_numbers);
    Malloc_Safely((void**)&h_pair_potential,
                  sizeof(float) * nr * atom_type_numbers * atom_type_numbers);

    float* temp_Z;
    Malloc_Safely((void**)&temp_Z, sizeof(float) * nr);

    for (int i = 0; i < nrho; i++)
    {
        if (fscanf(fp, "%f", h_embed + i) != 1) break;
        h_embed[i] *= CONSTANT_EV_TO_KCAL_MOL;
    }

    for (int i = 0; i < nr; i++)
        if (fscanf(fp, "%f", temp_Z + i) != 1) break;

    for (int i = 0; i < nr; i++)
        if (fscanf(fp, "%f", h_electron_density + i) != 1) break;

    for (int i = 0; i < nr; i++)
    {
        float r = i * dr;
        if (i == 0) r = 1e-8f;
        float z = temp_Z[i];
        h_pair_potential[i] = (z * z / r) *
                              CONSTANT_HARTREE_BOHR_TO_EV_ANGSTROM *
                              CONSTANT_EV_TO_KCAL_MOL;
    }

    free(temp_Z);
}

void EAM_INFORMATION::Read_Setfl(FILE* fp, CONTROLLER* controller)
{
    char line[CHAR_LENGTH_MAX];
    fgets(line, CHAR_LENGTH_MAX, fp);
    fgets(line, CHAR_LENGTH_MAX, fp);
    fgets(line, CHAR_LENGTH_MAX, fp);
    if (fscanf(fp, "%d", &atom_type_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Setfl",
                                       "Failed to read atom_type_numbers");
    }
    fgets(line, CHAR_LENGTH_MAX, fp);
    if (fscanf(fp, "%d %f %d %f %f\n", &nrho, &drho, &nr, &dr, &cut) != 5)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Setfl",
                                       "Failed to read table parameters");
    }
    controller->printf(
        "    EAM (Setfl) params: ntypes=%d, nrho=%d, drho=%f, nr=%d, dr=%f, "
        "cut=%f\n",
        atom_type_numbers, nrho, drho, nr, dr, cut);
    Malloc_Safely((void**)&h_embed, sizeof(float) * nrho * atom_type_numbers);
    Malloc_Safely((void**)&h_electron_density,
                  sizeof(float) * nr * atom_type_numbers);
    Malloc_Safely((void**)&h_pair_potential,
                  sizeof(float) * nr * atom_type_numbers * atom_type_numbers);

    float* all_Z;
    Malloc_Safely((void**)&all_Z, sizeof(float) * nr * atom_type_numbers);

    for (int t = 0; t < atom_type_numbers; t++)
    {
        int atomic_number;
        float mass, lattice_constant;
        char lattice_type[CHAR_LENGTH_MAX];
        if (fscanf(fp, "%d %f %f %s\n", &atomic_number, &mass,
                   &lattice_constant, lattice_type) != 4)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "EAM_INFORMATION::Read_Setfl",
                                           "Failed to read element header");
        }
        float* this_embed = h_embed + t * nrho;
        for (int i = 0; i < nrho; i++)
        {
            if (fscanf(fp, "%f", this_embed + i) != 1) break;
            this_embed[i] *= CONSTANT_EV_TO_KCAL_MOL;
        }
        float* this_rho = h_electron_density + t * nr;
        for (int i = 0; i < nr; i++)
        {
            if (fscanf(fp, "%f", this_rho + i) != 1) break;
        }
    }
    for (int i = 0; i < atom_type_numbers; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float val;
            float* phi_ij = h_pair_potential + (i * atom_type_numbers + j) * nr;
            float* phi_ji = h_pair_potential + (j * atom_type_numbers + i) * nr;

            for (int k = 0; k < nr; k++)
            {
                if (fscanf(fp, "%f", &val) != 1) break;
                float r = k * dr;
                if (k == 0) r = 1e-8f;
                float phi = (val / r) * CONSTANT_EV_TO_KCAL_MOL;
                phi_ij[k] = phi;
                if (i != j) phi_ji[k] = phi;
            }
        }
    }

    free(all_Z);
}

void EAM_INFORMATION::Initial(CONTROLLER* controller, const int atom_numbers,
                              const char* module_name, bool* need_full_nl_flag)
{
    if (module_name == NULL)
        strcpy(this->module_name, "EAM");
    else
        strcpy(this->module_name, module_name);

    NativeEAMDefinition native_definition;
    const bool has_native_definition = Read_H5_EAM_Input(
        controller, this->module_name, atom_numbers, &native_definition);
    if (!has_native_definition &&
        !controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("%s FORCE IS NOT INITIALIZED\n\n",
                           this->module_name);
        return;
    }

    if (has_native_definition)
    {
        controller->printf("START INITIALIZING EAM FORCE FROM NATIVE H5\n");
        atom_type_numbers = native_definition.atom_type_count;
        nrho = native_definition.nrho;
        nr = native_definition.nr;
        drho = native_definition.drho;
        dr = native_definition.dr;
        cut = native_definition.cut;
        Malloc_Safely((void**)&h_embed,
                      sizeof(float) * native_definition.embed.size());
        Malloc_Safely(
            (void**)&h_electron_density,
            sizeof(float) * native_definition.electron_density.size());
        Malloc_Safely((void**)&h_pair_potential,
                      sizeof(float) * native_definition.pair_potential.size());
        std::copy(native_definition.embed.begin(),
                  native_definition.embed.end(), h_embed);
        std::copy(native_definition.electron_density.begin(),
                  native_definition.electron_density.end(), h_electron_density);
        std::copy(native_definition.pair_potential.begin(),
                  native_definition.pair_potential.end(), h_pair_potential);
    }
    else
    {
        controller->printf("START INITIALIZING EAM FORCE\n");
        FILE* fp;
        Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                         "r");
        char line[CHAR_LENGTH_MAX];
        fgets(line, CHAR_LENGTH_MAX, fp);
        long pos = ftell(fp);
        char line2[CHAR_LENGTH_MAX];
        fgets(line2, CHAR_LENGTH_MAX, fp);
        int temp_int;
        float temp_float;
        int items = sscanf(line2, "%d %f", &temp_int, &temp_float);
        fseek(fp, pos, SEEK_SET);

        if (items == 2)
        {
            controller->printf(
                "    Detected DYNAMO funcfl format (Single Element).\n");
            Read_Funcfl(fp, controller);
        }
        else
        {
            controller->printf("    Detected DYNAMO setfl format (Alloy).\n");
            fseek(fp, 0, SEEK_SET);
            Read_Setfl(fp, controller);
        }
        fclose(fp);
    }

    this->atom_numbers = atom_numbers;
    Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));

    int num_types = atom_type_numbers;
    Device_Malloc_And_Copy_Safely((void**)&d_embed, h_embed,
                                  sizeof(float) * nrho * num_types);
    Device_Malloc_And_Copy_Safely((void**)&d_electron_density,
                                  h_electron_density,
                                  sizeof(float) * nr * num_types);
    Device_Malloc_And_Copy_Safely((void**)&d_pair_potential, h_pair_potential,
                                  sizeof(float) * nr * num_types * num_types);

    Device_Malloc_Safely((void**)&d_rho, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_df_drho, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);

    if (controller->Command_Exist(this->module_name, "atom_type_in_file"))
    {
        FILE* fp_type;
        Open_File_Safely(
            &fp_type,
            controller->Command(this->module_name, "atom_type_in_file"), "r");
        for (int i = 0; i < atom_numbers; i++)
        {
            int type_val;
            if (fscanf(fp_type, "%d", &type_val) != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "EAM_INFORMATION::Initial",
                                               "Failed to read atom types");
            }
            h_atom_type[i] = type_val;
        }
        fclose(fp_type);
    }
    else if (has_native_definition)
    {
        std::copy(native_definition.atom_type.begin(),
                  native_definition.atom_type.end(), h_atom_type);
    }
    else
    {
        for (int i = 0; i < atom_numbers; i++) h_atom_type[i] = 0;
    }

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    if (need_full_nl_flag != NULL)
    {
        *need_full_nl_flag = true;
        controller->printf("    %s requires full neighbor list.\n",
                           this->module_name);
    }

    is_initialized = true;
    if (!is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = true;
    }
    controller->printf("END INITIALIZING EAM FORCE\n\n");
}

void EAM_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
