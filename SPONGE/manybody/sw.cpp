#include "sw.h"

#include <algorithm>
#include <highfive/highfive.hpp>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
template <typename T>
hid_t Native_H5_Type();

template <>
hid_t Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
hid_t Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}

template <typename T>
std::vector<T> Read_H5_Array(HighFive::File* file,
                             const std::string& dataset_path,
                             std::size_t expected_rank,
                             std::size_t expected_columns,
                             const std::string& label)
{
    HighFive::DataSet dataset = file->getDataSet(dataset_path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != expected_rank ||
        (expected_rank == 2 && dimensions[1] != expected_columns))
    {
        std::ostringstream message;
        message << label << " dataset " << dataset_path << " must have rank "
                << expected_rank;
        if (expected_rank == 2)
        {
            message << " and " << expected_columns << " columns";
        }
        throw std::runtime_error(message.str());
    }
    std::size_t value_count = 1;
    for (const std::size_t dimension : dimensions)
    {
        value_count *= dimension;
    }
    std::vector<T> values(value_count);
    if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                H5P_DEFAULT, values.data()) < 0)
    {
        throw std::runtime_error("failed to read " + label + " dataset " +
                                 dataset_path);
    }
    return values;
}

struct NativeSWDefinition
{
    int atom_type_count = 0;
    std::vector<int> atom_type;
    std::vector<float> pair_parameters;
    std::vector<float> triple_parameters;
};

bool Read_H5_SW_Input(CONTROLLER* controller, const char* module_name,
                      NativeSWDefinition* definition)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* sw_root = "/manybody/sw";
    if (strcmp(module_name, "SW") != 0 ||
        controller->Command_Exist(module_name, "in_file") ||
        !controller->Command_Exist(input_key))
    {
        return false;
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(sw_root))
        {
            return false;
        }

        int atom_type_count = 0;
        file.getDataSet(std::string(sw_root) + "/atom_type_count")
            .read(atom_type_count);
        if (atom_type_count <= 0)
        {
            throw std::runtime_error(
                "/manybody/sw/atom_type_count must be positive");
        }
        const auto atom_type = Read_H5_Array<int>(
            &file, std::string(sw_root) + "/atom_type", 1, 0, "SW atom type");
        const auto pair_type = Read_H5_Array<int>(
            &file, std::string(sw_root) + "/pair/type", 2, 2, "SW pair type");
        const auto pair_parameter = Read_H5_Array<float>(
            &file, std::string(sw_root) + "/pair/parameters", 2, 8,
            "SW pair parameters");
        const auto triple_type =
            Read_H5_Array<int>(&file, std::string(sw_root) + "/triple/type", 2,
                               3, "SW triple type");
        const auto triple_parameter = Read_H5_Array<float>(
            &file, std::string(sw_root) + "/triple/parameters", 2, 3,
            "SW triple parameters");
        if (atom_type.empty())
        {
            throw std::runtime_error(
                "/manybody/sw/atom_type must not be empty");
        }
        for (const int type : atom_type)
        {
            if (type < 0 || type >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/atom_type contains an out-of-range type");
            }
        }

        const std::size_t pair_row_count = pair_type.size() / 2;
        if (pair_parameter.size() / 8 != pair_row_count)
        {
            throw std::runtime_error(
                "/manybody/sw pair type/parameter row count mismatch");
        }
        const std::size_t full_pair_count =
            static_cast<std::size_t>(atom_type_count) * atom_type_count;
        const std::size_t triangular_pair_count =
            static_cast<std::size_t>(atom_type_count) * (atom_type_count + 1) /
            2;
        if (pair_row_count != full_pair_count &&
            pair_row_count != triangular_pair_count)
        {
            throw std::runtime_error(
                "/manybody/sw pair row count must be atom_type_count^2 or "
                "triangular");
        }
        std::map<std::pair<int, int>, std::vector<float>> pair_rows;
        for (std::size_t row = 0; row < pair_row_count; ++row)
        {
            const int a = pair_type[2 * row];
            const int b = pair_type[2 * row + 1];
            if (a < 0 || a >= atom_type_count || b < 0 || b >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/pair/type contains an out-of-range type");
            }
            if (pair_rows.count({a, b}) != 0)
            {
                throw std::runtime_error(
                    "/manybody/sw/pair/type contains a duplicate row");
            }
            const std::vector<float> parameters(
                pair_parameter.begin() + 8 * row,
                pair_parameter.begin() + 8 * row + 8);
            pair_rows[{a, b}] = parameters;
            if (pair_row_count == triangular_pair_count)
            {
                pair_rows[{b, a}] = parameters;
            }
        }

        const std::size_t triple_row_count = triple_type.size() / 3;
        const std::size_t full_triple_count =
            full_pair_count * static_cast<std::size_t>(atom_type_count);
        if (triple_parameter.size() / 3 != triple_row_count ||
            triple_row_count != full_triple_count)
        {
            throw std::runtime_error(
                "/manybody/sw triple payload must contain atom_type_count^3 "
                "matching rows");
        }
        std::vector<float> canonical_triples(full_triple_count * 3);
        std::vector<bool> triple_recorded(full_triple_count, false);
        for (std::size_t row = 0; row < triple_row_count; ++row)
        {
            const int a = triple_type[3 * row];
            const int b = triple_type[3 * row + 1];
            const int c = triple_type[3 * row + 2];
            if (a < 0 || a >= atom_type_count || b < 0 ||
                b >= atom_type_count || c < 0 || c >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/triple/type contains an out-of-range type");
            }
            const std::size_t index =
                static_cast<std::size_t>(a) * atom_type_count *
                    atom_type_count +
                static_cast<std::size_t>(b) * atom_type_count + c;
            if (triple_recorded[index])
            {
                throw std::runtime_error(
                    "/manybody/sw/triple/type contains a duplicate row");
            }
            triple_recorded[index] = true;
            for (std::size_t parameter = 0; parameter < 3; ++parameter)
            {
                canonical_triples[3 * index + parameter] =
                    triple_parameter[3 * row + parameter];
            }
        }

        NativeSWDefinition result;
        result.atom_type_count = atom_type_count;
        result.atom_type = atom_type;
        result.pair_parameters.resize(full_pair_count * 8);
        for (int a = 0; a < atom_type_count; ++a)
        {
            for (int b = 0; b < atom_type_count; ++b)
            {
                const auto found = pair_rows.find({a, b});
                if (found == pair_rows.end())
                {
                    throw std::runtime_error(
                        "missing /manybody/sw pair parameters for type " +
                        std::to_string(a) + "," + std::to_string(b));
                }
                const std::size_t index =
                    static_cast<std::size_t>(a) * atom_type_count + b;
                std::copy(found->second.begin(), found->second.end(),
                          result.pair_parameters.begin() + 8 * index);
            }
        }
        if (std::find(triple_recorded.begin(), triple_recorded.end(), false) !=
            triple_recorded.end())
        {
            throw std::runtime_error(
                "/manybody/sw triple payload does not cover all type triples");
        }
        result.triple_parameters = std::move(canonical_triples);
        *definition = std::move(result);
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed SW input from ") +
            sw_root + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Read_H5_SW_Input", message.c_str());
    }
    return false;
}
}  // namespace

void STILLINGER_WEBER_INFORMATION::Initial(CONTROLLER* controller,
                                           const char* module_name,
                                           bool* need_full_nl_flag)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "SW");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    NativeSWDefinition native_definition;
    const bool has_native_definition =
        Read_H5_SW_Input(controller, this->module_name, &native_definition);
    if (has_native_definition)
    {
        controller->printf(
            "START INITIALIZING STILLINGER WEBER FORCE FROM NATIVE H5\n");
        atom_numbers = static_cast<int>(native_definition.atom_type.size());
        atom_type_numbers = native_definition.atom_type_count;
        pair_type_numbers = atom_type_numbers * atom_type_numbers;
        triple_type_numbers =
            atom_type_numbers * atom_type_numbers * atom_type_numbers;
        Malloc_Safely((void**)&h_energy_atom, sizeof(float) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                      sizeof(float) * (atom_numbers + 1));
        d_energy_atom = d_energy_sum + 1;
        const std::size_t parameter_count =
            static_cast<std::size_t>(pair_type_numbers) * 8 +
            static_cast<std::size_t>(triple_type_numbers) * 3;
        Malloc_Safely((void**)&h_parameters, sizeof(float) * parameter_count);
        Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
        std::copy(native_definition.pair_parameters.begin(),
                  native_definition.pair_parameters.end(), h_parameters);
        std::copy(native_definition.triple_parameters.begin(),
                  native_definition.triple_parameters.end(),
                  h_parameters + pair_type_numbers * 8);
        std::copy(native_definition.atom_type.begin(),
                  native_definition.atom_type.end(), h_atom_type);
        Device_Malloc_And_Copy_Safely((void**)&d_parameters, h_parameters,
                                      sizeof(float) * parameter_count);
        Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                      sizeof(int) * atom_numbers);
        if (need_full_nl_flag != NULL)
        {
            *need_full_nl_flag = true;
            controller->printf(
                "SW requires full neighbor list for three-body "
                "calculations.\n");
        }
        is_initialized = true;
        if (!is_controller_printf_initialized)
        {
            controller->Step_Print_Initial(this->module_name, "%.2f");
            is_controller_printf_initialized = true;
        }
        controller->printf("END INITIALIZING STILLINGER WEBER FORCE\n\n");
        return;
    }
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("STILLINGER WEBER FORCE IS NOT INITIALIZED\n\n");
        return;
    }
    controller->printf("START INITIALIZING STILLINGER WEBER FORCE\n");
    FILE* fp;
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d %d\n", &atom_numbers, &atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe number of atoms and SW types can not be found\n");
    }
    pair_type_numbers = atom_type_numbers * atom_type_numbers;
    triple_type_numbers =
        atom_type_numbers * atom_type_numbers * atom_type_numbers;
    Malloc_Safely((void**)&h_energy_atom, sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                  sizeof(float) * (atom_numbers + 1));
    d_energy_atom = d_energy_sum + 1;
    Malloc_Safely(
        (void**)&h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 3));
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    char temp[CHAR_LENGTH_MAX];
    std::map<int, bool> unrecorded;
    int type_a, type_b, type_c;
    float A, B, sigma, p, q, a, gamma;
    float lambda, epsilon, b;
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        printf("'%s'\n", temp);
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe first comment line can not be found\n");
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %f %f %f %f %f %f %f %f\n", &type_a, &type_b, &A,
                   &B, &epsilon, &p, &q, &a, &gamma, &sigma) != 10)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers + type_b;
        if (index >= pair_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * index + 0] = A;
        h_parameters[8 * index + 1] = B;
        h_parameters[8 * index + 2] = epsilon;
        h_parameters[8 * index + 3] = p;
        h_parameters[8 * index + 4] = q;
        h_parameters[8 * index + 5] = a;
        h_parameters[8 * index + 6] = gamma;
        h_parameters[8 * index + 7] = sigma;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
    }
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe second comment line can not be found\n");
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f %f\n", &type_a, &type_b, &type_c,
                   &lambda, &epsilon, &b) != 6)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers * atom_type_numbers +
                    type_b * atom_type_numbers + type_c;
        if (index >= triple_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * pair_type_numbers + 3 * index + 0] = lambda;
        h_parameters[8 * pair_type_numbers + 3 * index + 1] = epsilon;
        h_parameters[8 * pair_type_numbers + 3 * index + 2] = b;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
    }
    Device_Malloc_And_Copy_Safely(
        (void**)&d_parameters, h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 3));
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe third comment line can not be found\n");
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", h_atom_type + i) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome atom types can not be found\n");
        }
        if (h_atom_type[i] >= atom_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
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
            "SW requires full neighbor list for three-body calculations.\n");
    }

    is_initialized = true;
    if (!is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = true;
    }
    controller->printf("END INITIALIZING STILLINGER WEBER FORCE\n\n");
}

template <bool need_energy, bool need_virial>
static __global__ __launch_bounds__(1024) void SW_Force_With_Full_Neighbor_CUDA(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const Boundary boundary, const ATOM_GROUP* nl, float* atom_energy,
    LTMatrix3* atom_virial, int* atom_types, float* parameters,
    const int atom_type_numbers, const int pair_type_numbers,
    float* this_energy)
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
        float A, B, sigma1, sigma2, sigma3, p, q, a1, a2, a3, gamma1, gamma2,
            gamma3, lambda, epsilon, b;
        int pair_index_1, pair_index_2, triple_index;
        float local_energy = 0;
        VECTOR i_force = {0.0f, 0.0f, 0.0f}, j_force, k_force;
        VECTOR temp1_force, temp2_force;
        LTMatrix3 local_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        SADfloat<1> twobody;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            j_force = {0.0f, 0.0f, 0.0f};
            atom_j = nl_i.atom_serial[j];

            bool should_compute_twobody = atom_j > atom_i;

            type_j = atom_types[atom_j];
            rj = crd[atom_j];
            drij = Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            rij = norm3df(drij.x, drij.y, drij.z);
            pair_index_1 = type_i * atom_type_numbers + type_j;
            A = parameters[8 * pair_index_1];
            B = parameters[8 * pair_index_1 + 1];
            epsilon = parameters[8 * pair_index_1 + 2];
            p = parameters[8 * pair_index_1 + 3];
            q = parameters[8 * pair_index_1 + 4];
            a1 = parameters[8 * pair_index_1 + 5];
            gamma1 = parameters[8 * pair_index_1 + 6];
            sigma1 = parameters[8 * pair_index_1 + 7];

            if (should_compute_twobody && rij < a1 * sigma1)
            {
                twobody.val = rij;
                twobody.dval[0] = 1;
                twobody = twobody / sigma1;
                twobody = A * epsilon *
                          (B * powf(twobody, -p) - powf(twobody, -q)) *
                          expf(1.0f / (twobody - a1));
                local_energy += twobody.val;
                temp1_force = twobody.dval[0] / rij * drij;
                i_force = i_force - temp1_force;
                j_force = j_force + temp1_force;
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
                a2 = parameters[8 * pair_index_2 + 5];
                sigma2 = parameters[8 * pair_index_2 + 7];

                if (rij < a1 * sigma1 && rik < a2 * sigma2)
                {
                    gamma2 = parameters[8 * pair_index_2 + 6];
                    triple_index =
                        type_i * atom_type_numbers * atom_type_numbers +
                        type_j * atom_type_numbers + type_k;
                    lambda =
                        parameters[8 * pair_type_numbers + 3 * triple_index];
                    epsilon = parameters[8 * pair_type_numbers +
                                         3 * triple_index + 1];
                    b = parameters[8 * pair_type_numbers + 3 * triple_index +
                                   2];

                    SADvector<6> threebody1(drij, 0, 1, 2);
                    SADvector<6> threebody2(drik, 3, 4, 5);
                    SADfloat<6> r1 = sqrtf(threebody1 * threebody1);
                    SADfloat<6> r2 = sqrtf(threebody2 * threebody2);
                    SADfloat<6> E = threebody1 * threebody2 / r1 / r2 - b;
                    r1 = r1 / sigma1;
                    r2 = r2 / sigma2;
                    E = lambda * epsilon * E * E * expf(gamma1 / (r1 - a1)) *
                        expf(gamma2 / (r2 - a2));
                    temp1_force = {E.dval[0], E.dval[1], E.dval[2]};
                    temp2_force = {E.dval[3], E.dval[4], E.dval[5]};
                    k_force = k_force + temp2_force;
                    j_force = j_force + temp1_force;
                    i_force = i_force - temp1_force - temp2_force;
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
        }
        Warp_Sum_To(frc + atom_i, i_force, warpSize);
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, local_energy, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(this_energy + atom_i, local_energy);
        }
        if (need_virial)
            Warp_Sum_To(atom_virial + atom_i, local_virial, warpSize);
    }
}

void STILLINGER_WEBER_INFORMATION::SW_Force_With_Atom_Energy_And_Virial_Full_NL(
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

    auto f = SW_Force_With_Full_Neighbor_CUDA<false, false>;

    if (!need_atom_energy && !need_virial)
    {
        f = SW_Force_With_Full_Neighbor_CUDA<false, false>;
    }
    else if (!need_atom_energy && need_virial)
    {
        f = SW_Force_With_Full_Neighbor_CUDA<false, true>;
    }
    else if (need_atom_energy && !need_virial)
    {
        f = SW_Force_With_Full_Neighbor_CUDA<true, false>;
    }
    else
    {
        f = SW_Force_With_Full_Neighbor_CUDA<true, true>;
    }

    Launch_Device_Kernel(f, gridSize, blockSize, 0, NULL, atom_numbers, crd,
                         frc, boundary, fnl_d_nl, atom_energy, atom_virial,
                         this->d_atom_type, this->d_parameters,
                         this->atom_type_numbers, this->pair_type_numbers,
                         this->d_energy_atom);
}

void STILLINGER_WEBER_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_energy_atom, d_energy_sum, atom_numbers);
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
