#pragma once

#include "third_party/toml/toml_decode.hpp"
static __global__ void MD_Atom_Ek(const int atom_numbers, float* ek,
                                  const VECTOR* atom_vel,
                                  const float* atom_mass)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR v = atom_vel[atom_i];
        ek[atom_i] = 0.5 * v * v * atom_mass[atom_i];
    }
}
static __global__ void Get_Stress_Kinetic_Contribution(
    const int atom_numbers, const VECTOR* vel, const float* mass,
    LTMatrix3* stress, const float volume_inverse)
{
    LTMatrix3 stress0 = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int tid = threadIdx.y + threadIdx.x * blockDim.y +
              blockDim.y * blockDim.x * blockIdx.x;
    for (int atom = tid; atom < atom_numbers;
         atom += blockDim.y * blockDim.x * gridDim.x)
    {
        VECTOR vel0 = vel[atom];
        stress0 = stress0 + volume_inverse * mass[atom] *
                                Get_Virial_From_Force_Dis(vel0, vel0);
    }
#else
    float s11 = 0.0f, s21 = 0.0f, s22 = 0.0f;
    float s31 = 0.0f, s32 = 0.0f, s33 = 0.0f;
#pragma omp parallel for reduction(+ : s11, s21, s22, s31, s32, s33)
    for (int atom = 0; atom < atom_numbers; atom += 1)
    {
        VECTOR vel0 = vel[atom];
        LTMatrix3 contrib =
            volume_inverse * mass[atom] * Get_Virial_From_Force_Dis(vel0, vel0);
        s11 += contrib.a11;
        s21 += contrib.a21;
        s22 += contrib.a22;
        s31 += contrib.a31;
        s32 += contrib.a32;
        s33 += contrib.a33;
    }
    stress0 = {s11, s21, s22, s31, s32, s33};
#endif
    Warp_Sum_To(stress, stress0, warpSize);
}

static __global__ void Get_Stress_Potential_Contribution(
    const int atom_numbers, const LTMatrix3* atom_virial, LTMatrix3* virial)
{
    LTMatrix3 virial0 = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int tid = threadIdx.y + threadIdx.x * blockDim.y +
              blockDim.y * blockDim.x * blockIdx.x;
    for (int atom = tid; atom < atom_numbers;
         atom += blockDim.y * blockDim.x * gridDim.x)
    {
        virial0 = virial0 + atom_virial[atom];
    }
#else
    float v11 = 0.0f, v21 = 0.0f, v22 = 0.0f;
    float v31 = 0.0f, v32 = 0.0f, v33 = 0.0f;
#pragma omp parallel for reduction(+ : v11, v21, v22, v31, v32, v33)
    for (int atom = 0; atom < atom_numbers; atom += 1)
    {
        v11 += atom_virial[atom].a11;
        v21 += atom_virial[atom].a21;
        v22 += atom_virial[atom].a22;
        v31 += atom_virial[atom].a31;
        v32 += atom_virial[atom].a32;
        v33 += atom_virial[atom].a33;
    }
    virial0 = {v11, v21, v22, v31, v32, v33};
#endif
    Warp_Sum_To(virial, virial0, warpSize);
}

static __global__ void Get_Stress_From_virial(const float volume_inverse,
                                              const LTMatrix3* virial,
                                              LTMatrix3* stress)
{
    stress[0] = stress[0] + volume_inverse * virial[0];
}

static __global__ void Add_Sum_List(int n, float* atom_virial,
                                    float* sum_virial)
{
    float temp = 0;
#ifdef GPU_ARCH_NAME
    for (int i = blockIdx.x * blockDim.x * blockDim.y +
                 threadIdx.x * blockDim.y + threadIdx.y;
         i < n; i = i + blockDim.x * blockDim.y * gridDim.x)
#else
#pragma omp parallel for reduction(+ : temp)
    for (int i = 0; i < n; i = i + 1)
#endif
    {
        temp = temp + atom_virial[i];
    }
    Warp_Sum_To(sum_virial, temp, warpSize);
}

struct TomlSchedulePoint
{
    int step = 0;
    float value = 0.0f;
};

struct TomlScheduleConfig
{
    std::optional<std::string> mode;
    std::optional<std::vector<TomlSchedulePoint>> steps;
};

struct TomlSystemScheduleInputs
{
    std::optional<std::string> target_temperature_schedule_mode;
    std::optional<std::vector<TomlSchedulePoint>>
        target_temperature_schedule_steps;
    std::optional<std::string> target_temperature_schedule_file;
    std::optional<std::string> target_pressure_schedule_mode;
    std::optional<std::vector<TomlSchedulePoint>>
        target_pressure_schedule_steps;
    std::optional<std::string> target_pressure_schedule_file;
};

namespace sponge::toml_decode
{

}  // namespace sponge::toml_decode

SPONGE_TOML_DECODE_REFLECT(TomlSchedulePoint,
                           SPONGE_TOML_DECODE_MEMBER(TomlSchedulePoint, step),
                           SPONGE_TOML_DECODE_MEMBER(TomlSchedulePoint, value))
SPONGE_TOML_DECODE_REFLECT(TomlScheduleConfig,
                           SPONGE_TOML_DECODE_MEMBER(TomlScheduleConfig, mode),
                           SPONGE_TOML_DECODE_MEMBER(TomlScheduleConfig, steps))
SPONGE_TOML_DECODE_REFLECT(
    TomlSystemScheduleInputs,
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_temperature_schedule_mode),
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_temperature_schedule_steps),
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_temperature_schedule_file),
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_pressure_schedule_mode),
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_pressure_schedule_steps),
    SPONGE_TOML_DECODE_MEMBER(TomlSystemScheduleInputs,
                              target_pressure_schedule_file))

namespace
{
constexpr const char* kSysScheduleErrorBy =
    "MD_INFORMATION::system_information::Initial";

void Throw_Schedule_Error(CONTROLLER* controller, const char* key,
                          const std::string& reason)
{
    std::string error_reason = "Reason:\n\tinvalid '";
    error_reason += key;
    error_reason += "': ";
    error_reason += reason;
    error_reason += "\n";
    controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                   kSysScheduleErrorBy, error_reason.c_str());
}

bool Parse_Toml_Schedule_Mode(const std::string& value, int* mode_out)
{
    if (mode_out == nullptr) return false;
    std::string mode = string_strip(value);
    if (mode == "step")
    {
        *mode_out = MD_INFORMATION::system_information::TARGET_SCHEDULE::STEP;
        return true;
    }
    if (mode == "linear")
    {
        *mode_out = MD_INFORMATION::system_information::TARGET_SCHEDULE::LINEAR;
        return true;
    }
    return false;
}

std::optional<TomlSystemScheduleInputs> Parse_System_Schedule_Inputs(
    CONTROLLER* controller, std::string* error_message)
{
    error_message->clear();
    if (!controller->mdin_is_toml || controller->mdin_toml_content.empty())
    {
        return std::nullopt;
    }
    try
    {
        return ::sponge::toml_decode::parse_string<TomlSystemScheduleInputs>(
            controller->mdin_toml_content, controller->mdin_toml_source_path);
    }
    catch (const std::exception& err)
    {
        *error_message = err.what();
        return std::nullopt;
    }
}

std::optional<std::string> Resolve_Schedule_File_Path(CONTROLLER* controller,
                                                      const char* file_key,
                                                      bool is_pressure)
{
    if (controller->Command_Exist(file_key))
    {
        return std::string(controller->Command(file_key));
    }
    if (!controller->Command_Exist("default_in_file_prefix"))
    {
        return std::nullopt;
    }

    const std::string prefix = controller->Command("default_in_file_prefix");
    const std::vector<std::string> candidates = {
        prefix + "." + (is_pressure ? "pres.spg.toml" : "temp.spg.toml"),
        prefix + "_" + (is_pressure ? "pres.spg.toml" : "temp.spg.toml")};
    for (const auto& candidate : candidates)
    {
        FILE* fp = fopen(candidate.c_str(), "r");
        if (fp != NULL)
        {
            fclose(fp);
            return candidate;
        }
    }
    return std::nullopt;
}

bool Load_Schedule_Values_From_TomlPoints(
    const std::vector<TomlSchedulePoint>& points,
    MD_INFORMATION::system_information::TARGET_SCHEDULE* out)
{
    out->steps.clear();
    out->values.clear();
    out->steps.reserve(points.size());
    out->values.reserve(points.size());
    for (const auto& point : points)
    {
        out->steps.push_back(point.step);
        out->values.push_back(point.value);
    }
    return !out->steps.empty();
}

float Evaluate_Target_Schedule(
    const MD_INFORMATION::system_information::TARGET_SCHEDULE& schedule,
    int current_step, float fallback_value)
{
    if (!schedule.enabled || schedule.steps.empty()) return fallback_value;
    if (current_step <= schedule.steps.front()) return schedule.values.front();
    if (current_step >= schedule.steps.back()) return schedule.values.back();

    const int upper_idx =
        (int)(std::upper_bound(schedule.steps.begin(), schedule.steps.end(),
                               current_step) -
              schedule.steps.begin());
    const int left_idx = upper_idx - 1;
    if (schedule.mode ==
        MD_INFORMATION::system_information::TARGET_SCHEDULE::STEP)
    {
        return schedule.values[left_idx];
    }
    const int left_step = schedule.steps[left_idx];
    const int right_step = schedule.steps[upper_idx];
    const float left_value = schedule.values[left_idx];
    const float right_value = schedule.values[upper_idx];
    const float ratio =
        (float)(current_step - left_step) / (float)(right_step - left_step);
    return left_value + (right_value - left_value) * ratio;
}

void Load_Target_Schedule(
    CONTROLLER* controller, const char* schedule_name, const char* mode_key,
    const char* steps_key, const char* file_key, bool is_pressure,
    MD_INFORMATION::system_information::TARGET_SCHEDULE* out)
{
    const bool has_mode = controller->Command_Exist(mode_key);
    const bool explicit_file = controller->Command_Exist(file_key);
    const bool command_has_steps = controller->Command_Exist(steps_key);

    std::string parse_error;
    const auto parsed_inputs =
        Parse_System_Schedule_Inputs(controller, &parse_error);
    if (!parse_error.empty())
    {
        Throw_Schedule_Error(controller, schedule_name,
                             "failed to decode mdin TOML: " + parse_error);
    }

    std::optional<std::vector<TomlSchedulePoint>> inline_steps;
    if (parsed_inputs.has_value())
    {
        inline_steps = is_pressure
                           ? parsed_inputs->target_pressure_schedule_steps
                           : parsed_inputs->target_temperature_schedule_steps;
    }
    else if (command_has_steps)
    {
        Throw_Schedule_Error(controller, steps_key,
                             "inline schedule steps require TOML mdin input");
    }

    const auto schedule_file_path =
        Resolve_Schedule_File_Path(controller, file_key, is_pressure);
    const bool has_file = schedule_file_path.has_value();

    out->enabled = false;
    out->mode = MD_INFORMATION::system_information::TARGET_SCHEDULE::STEP;
    out->steps.clear();
    out->values.clear();
    if (!has_mode && !inline_steps.has_value() && !has_file) return;

    if (has_file && inline_steps.has_value())
    {
        Throw_Schedule_Error(
            controller, schedule_name,
            "cannot use inline schedule and schedule file at the same time");
    }

    if (has_mode)
    {
        if (controller->Command_Choice(mode_key, "step"))
        {
            out->mode =
                MD_INFORMATION::system_information::TARGET_SCHEDULE::STEP;
        }
        else if (controller->Command_Choice(mode_key, "linear"))
        {
            out->mode =
                MD_INFORMATION::system_information::TARGET_SCHEDULE::LINEAR;
        }
        else
        {
            Throw_Schedule_Error(controller, mode_key,
                                 "mode must be 'step' or 'linear'");
        }
    }

    if (has_file)
    {
        try
        {
            const auto file_config =
                ::sponge::toml_decode::parse_file<TomlScheduleConfig>(
                    *schedule_file_path);
            if (!file_config.steps.has_value() ||
                !Load_Schedule_Values_From_TomlPoints(*file_config.steps, out))
            {
                Throw_Schedule_Error(controller, file_key,
                                     "schedule TOML must provide non-empty "
                                     "'steps'");
            }
            if (!has_mode && file_config.mode.has_value() &&
                !Parse_Toml_Schedule_Mode(*file_config.mode, &out->mode))
            {
                Throw_Schedule_Error(controller, file_key,
                                     "schedule file mode must be 'step' or "
                                     "'linear'");
            }
        }
        catch (const std::exception& err)
        {
            Throw_Schedule_Error(
                controller, file_key,
                "invalid TOML schedule file: " + std::string(err.what()));
        }
    }
    else if (inline_steps.has_value())
    {
        if (!Load_Schedule_Values_From_TomlPoints(*inline_steps, out))
        {
            Throw_Schedule_Error(controller, steps_key,
                                 "must contain at least one {step, value} "
                                 "object");
        }
    }
    else
    {
        Throw_Schedule_Error(controller, schedule_name,
                             "schedule exists but no points are provided");
    }

    if (out->steps.size() != out->values.size())
    {
        Throw_Schedule_Error(controller, schedule_name,
                             "steps and values size mismatch");
    }
    if (out->steps.empty())
    {
        Throw_Schedule_Error(controller, schedule_name,
                             "at least one point is required");
    }
    for (size_t i = 1; i < out->steps.size(); i++)
    {
        if (out->steps[i] <= out->steps[i - 1])
        {
            Throw_Schedule_Error(controller, schedule_name,
                                 "steps must be strictly increasing");
        }
    }
    if (!is_pressure)
    {
        for (float value : out->values)
        {
            if (!(value > 0.0f))
            {
                Throw_Schedule_Error(controller, schedule_name,
                                     "temperature values must be > 0");
            }
        }
    }
    else
    {
        for (float& value : out->values)
        {
            value *= CONSTANT_PRES_CONVERTION_INVERSE;
        }
    }
    out->enabled = true;
}
}  // namespace

double MD_INFORMATION::system_information::Get_Current_Time(bool plus_one_step)
{
    current_time = start_time + (double)dt_in_ps * (steps + plus_one_step);
    return current_time;
}

float MD_INFORMATION::system_information::Get_Volume()
{
    LTMatrix3 cell = md_info->pbc.cell;
    volume = cell.a11 * cell.a22 * cell.a33;
    return volume;
}

float MD_INFORMATION::system_information::Get_Density()
{
    density = total_mass * 1e24f / 6.023e23f / Get_Volume();
    return density;
}

float MD_INFORMATION::system_information::Get_Total_Atom_Ek(int is_download)
{
    int gridSize = (md_info->atom_numbers + CONTROLLER::device_max_thread - 1) /
                   CONTROLLER::device_max_thread;
    Launch_Device_Kernel(MD_Atom_Ek, gridSize, CONTROLLER::device_max_thread, 0,
                         NULL, md_info->atom_numbers, md_info->d_atom_ek,
                         md_info->vel, md_info->d_mass);
    Sum_Of_List(md_info->d_atom_ek, d_sum_of_atom_ek, md_info->atom_numbers);
    SPONGE_MPI_WRAPPER::Device_Sum(d_sum_of_atom_ek, 1, CONTROLLER::d_pp_comm);
    if (is_download)
    {
        deviceMemcpy(&h_sum_of_atom_ek, d_sum_of_atom_ek, sizeof(float),
                     deviceMemcpyDeviceToHost);
        return h_sum_of_atom_ek;
    }
    else
    {
        return 0;
    }
}

float MD_INFORMATION::system_information::Get_Atom_Temperature()
{
    h_temperature = Get_Total_Atom_Ek() * 2. / CONSTANT_kB / freedom;
    return h_temperature;
}

void MD_INFORMATION::system_information::Update_Targets_By_Schedule(
    int current_step)
{
    if (target_temperature_schedule.enabled)
    {
        target_temperature = Evaluate_Target_Schedule(
            target_temperature_schedule, current_step, target_temperature);
    }
    if (target_pressure_schedule.enabled)
    {
        target_pressure = Evaluate_Target_Schedule(
            target_pressure_schedule, current_step, target_pressure);
    }
}

void MD_INFORMATION::system_information::Get_Potential_to_stress(
    CONTROLLER* controller, int atom_numbers, LTMatrix3* d_atom_virial_tensor)
{
    float volume_inverse = 1.0f / Get_Volume();
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};

    // 计算势能贡献
    Launch_Device_Kernel(
        Get_Stress_Potential_Contribution,
        (atom_numbers + 4 * CONTROLLER::device_max_thread - 1) / 4 /
            CONTROLLER::device_max_thread,
        blockSize, 0, NULL, atom_numbers, d_atom_virial_tensor,
        d_virial_tensor);

    Launch_Device_Kernel(Get_Stress_From_virial, 1, 1, 0, NULL, volume_inverse,
                         d_virial_tensor, d_stress);
}

void MD_INFORMATION::system_information::Get_Kinetic_to_stress(
    CONTROLLER* controller, int atom_numbers, VECTOR* vel, float* atom_mass)
{
    float volume_inverse = 1.0f / Get_Volume();
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    // 计算动能贡献
    Launch_Device_Kernel(
        Get_Stress_Kinetic_Contribution,
        (atom_numbers + 4 * CONTROLLER::device_max_thread - 1) / 4 /
            CONTROLLER::device_max_thread,
        blockSize, 0, NULL, atom_numbers, vel, atom_mass, d_stress,
        volume_inverse);
}

float MD_INFORMATION::system_information::Get_Potential(int is_download)
{
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    Launch_Device_Kernel(Add_Sum_List, CONTROLLER::device_optimized_block,
                         blockSize, 0, NULL, md_info->atom_numbers,
                         md_info->d_atom_energy, d_potential);
    SPONGE_MPI_WRAPPER::Device_Sum(d_potential, 1,
                                   CONTROLLER::D_MPI_COMM_WORLD);
    if (is_download)
    {
        deviceMemcpy(&h_potential, d_potential, sizeof(float),
                     deviceMemcpyDeviceToHost);
        return h_potential;
    }
    else
    {
        return 0;
    }
}

void MD_INFORMATION::system_information::Initial(CONTROLLER* controller,
                                                 MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    steps = 0;
    if (md_info->mode != md_info->RERUN)
    {
        step_limit = 1000;
        if (controller[0].Command_Exist("step_limit"))
        {
            controller->Check_Int(
                "step_limit", "MD_INFORMATION::system_information::Initial");
            step_limit = atoi(controller[0].Command("step_limit"));
        }

        target_temperature = 300.0f;
        if (md_info->mode >= md_info->NVT &&
            controller[0].Command_Exist("target_temperature"))
        {
            controller->Check_Float(
                "target_temperature",
                "MD_INFORMATION::system_information::Initial");
            target_temperature =
                atof(controller[0].Command("target_temperature"));
        }

        target_pressure = 1;
        if (md_info->mode == md_info->NPT &&
            controller[0].Command_Exist("target_pressure"))
        {
            controller->Check_Float(
                "target_pressure",
                "MD_INFORMATION::system_information::Initial");
            target_pressure = atof(controller[0].Command("target_pressure"));
        }
        target_pressure *= CONSTANT_PRES_CONVERTION_INVERSE;
        Load_Target_Schedule(controller, "target_temperature_schedule",
                             "target_temperature_schedule_mode",
                             "target_temperature_schedule_steps",
                             "target_temperature_schedule_file", false,
                             &target_temperature_schedule);
        Load_Target_Schedule(
            controller, "target_pressure_schedule",
            "target_pressure_schedule_mode", "target_pressure_schedule_steps",
            "target_pressure_schedule_file", true, &target_pressure_schedule);
    }
    else
    {
        step_limit = INT_MAX - 1;
        if (controller[0].Command_Exist("frame_limit"))
        {
            controller->Check_Int(
                "frame_limit", "MD_INFORMATION::system_information::Initial");
            step_limit = atoi(controller[0].Command("frame_limit"));
        }
        else if (controller[0].Command_Exist("rerun_frame_limit"))
        {
            controller->Check_Int(
                "rerun_frame_limit",
                "MD_INFORMATION::system_information::Initial");
            step_limit = atoi(controller[0].Command("rerun_frame_limit"));
        }
    }
    for (int i = 0; i < md_info->atom_numbers; i++)
    {
        std::set<int> temp;
        connectivity[i] = temp;
    }
    if (md_info->mode != md_info->RERUN)
    {
        controller->Step_Print_Initial("step", "%d");
        controller->Step_Print_Initial("time", "%.3lf");
        controller->Step_Print_Initial("temperature", "%.2f");
    }
    else
    {
        controller->Step_Print_Initial("frame", "%d");
        controller->Step_Print_Initial("temperature", "%.2f");
    }
    Device_Malloc_And_Copy_Safely((void**)&this->d_virial_tensor,
                                  &this->h_virial_tensor, sizeof(LTMatrix3));
    Device_Malloc_And_Copy_Safely((void**)&this->d_pressure, &this->h_pressure,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&this->d_stress, &this->h_stress,
                                  sizeof(LTMatrix3));
    Device_Malloc_And_Copy_Safely((void**)&this->d_temperature,
                                  &this->h_temperature, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&this->d_potential,
                                  &this->h_potential, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&this->d_sum_of_atom_ek,
                                  &this->h_sum_of_atom_ek, sizeof(float));
}
