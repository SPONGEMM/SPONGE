#include "initial_velocity.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../Domain_decomposition/Domain_decomposition.h"
#include "../constrain/settle.h"
#include "../constrain/shake.h"
#include "MD_core.h"

namespace
{
constexpr const char* kModuleName = "initial_velocity";
constexpr const char* kErrorBy = "INITIAL_VELOCITY_INFORMATION::Initial";
constexpr const char* kFinalizeErrorBy =
    "INITIAL_VELOCITY_INFORMATION::Finalize";

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool Float_Bits_Are_Finite(const void* address)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(float), "unexpected float size");
    std::memcpy(&bits, address, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool Double_Bits_Are_Finite(const void* address)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(double), "unexpected double size");
    std::memcpy(&bits, address, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

void Throw_Input_Error(CONTROLLER* controller, const char* reason)
{
    controller->Throw_Formatted_SPONGE_Error(
        spongeErrorValueErrorCommand, kErrorBy,
        "Reason:\n\tinvalid [initial_velocity] configuration: %s\n", reason);
}

bool Parse_Seed(const char* token, std::uint64_t* value)
{
    if (token == nullptr || token[0] == '\0' || value == nullptr) return false;
    const std::uint64_t maximum =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t parsed = 0;
    for (const char* cursor = token; *cursor != '\0'; ++cursor)
    {
        if (*cursor < '0' || *cursor > '9') return false;
        const std::uint64_t digit = static_cast<std::uint64_t>(*cursor - '0');
        if (parsed > (maximum - digit) / 10) return false;
        parsed = parsed * 10 + digit;
    }
    *value = parsed;
    return true;
}

void Generate_Maxwell_Direction(CONTROLLER* controller, MD_INFORMATION* md_info,
                                std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double thermal_energy =
        static_cast<double>(CONSTANT_kB) *
        static_cast<double>(md_info->sys.target_temperature);
    double total_mass = 0.0;
    double momentum_x = 0.0;
    double momentum_y = 0.0;
    double momentum_z = 0.0;

    for (int atom = 0; atom < md_info->atom_numbers; ++atom)
    {
        const double mass = static_cast<double>(md_info->h_mass[atom]);
        if (!Float_Bits_Are_Finite(&md_info->h_mass[atom]) || mass < 0.0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, kErrorBy,
                "Reason:\n\tatom %d has non-finite or negative mass %.9g "
                "during Maxwell initialization\n",
                atom, mass);
        }
        if (mass == 0.0)
        {
            md_info->velocity[atom] = {0.0f, 0.0f, 0.0f};
            continue;
        }
        const double sigma = std::sqrt(thermal_energy / mass);
        VECTOR velocity = {static_cast<float>(sigma * normal(generator)),
                           static_cast<float>(sigma * normal(generator)),
                           static_cast<float>(sigma * normal(generator))};
        if (!Float_Bits_Are_Finite(&velocity.x) ||
            !Float_Bits_Are_Finite(&velocity.y) ||
            !Float_Bits_Are_Finite(&velocity.z))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorSimulationBreakDown, kErrorBy,
                "Reason:\n\tthe generated initial velocity is outside the "
                "finite single-precision range\n");
        }
        md_info->velocity[atom] = velocity;
        total_mass += mass;
        momentum_x += mass * static_cast<double>(velocity.x);
        momentum_y += mass * static_cast<double>(velocity.y);
        momentum_z += mass * static_cast<double>(velocity.z);
    }

    if (!Double_Bits_Are_Finite(&total_mass) || !(total_mass > 0.0))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, kErrorBy,
            "Reason:\n\tMaxwell initialization requires at least one atom "
            "with positive finite mass\n");
    }
    const double center_velocity_x = momentum_x / total_mass;
    const double center_velocity_y = momentum_y / total_mass;
    const double center_velocity_z = momentum_z / total_mass;
    for (int atom = 0; atom < md_info->atom_numbers; ++atom)
    {
        if (md_info->h_mass[atom] == 0.0f) continue;
        VECTOR& velocity = md_info->velocity[atom];
        velocity.x = static_cast<float>(static_cast<double>(velocity.x) -
                                        center_velocity_x);
        velocity.y = static_cast<float>(static_cast<double>(velocity.y) -
                                        center_velocity_y);
        velocity.z = static_cast<float>(static_cast<double>(velocity.z) -
                                        center_velocity_z);
        if (!Float_Bits_Are_Finite(&velocity.x) ||
            !Float_Bits_Are_Finite(&velocity.y) ||
            !Float_Bits_Are_Finite(&velocity.z))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorSimulationBreakDown, kErrorBy,
                "Reason:\n\tCOM removal produced a non-finite initial "
                "velocity\n");
        }
    }
}

double Local_Kinetic_Energy(CONTROLLER* controller,
                            const DOMAIN_INFORMATION* dd)
{
    std::vector<VECTOR> velocity(static_cast<std::size_t>(dd->atom_numbers));
    std::vector<float> mass(static_cast<std::size_t>(dd->atom_numbers));
    if (dd->atom_numbers > 0)
    {
        deviceMemcpy(
            velocity.data(), dd->vel,
            sizeof(VECTOR) * static_cast<std::size_t>(dd->atom_numbers),
            deviceMemcpyDeviceToHost);
        deviceMemcpy(mass.data(), dd->d_mass,
                     sizeof(float) * static_cast<std::size_t>(dd->atom_numbers),
                     deviceMemcpyDeviceToHost);
    }

    double kinetic_energy = 0.0;
    for (int atom = 0; atom < dd->atom_numbers; ++atom)
    {
        const VECTOR& v = velocity[atom];
        const double atom_mass = static_cast<double>(mass[atom]);
        const double velocity_squared =
            static_cast<double>(v.x) * static_cast<double>(v.x) +
            static_cast<double>(v.y) * static_cast<double>(v.y) +
            static_cast<double>(v.z) * static_cast<double>(v.z);
        const double contribution = 0.5 * atom_mass * velocity_squared;
        if (!Double_Bits_Are_Finite(&contribution) || contribution < 0.0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorSimulationBreakDown, kFinalizeErrorBy,
                "Reason:\n\tthe projected initial velocity has non-finite "
                "or negative kinetic energy\n");
        }
        kinetic_energy += contribution;
    }
    if (!Double_Bits_Are_Finite(&kinetic_energy) || kinetic_energy < 0.0)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe local initial kinetic-energy sum is non-finite "
            "or negative\n");
    }
    return kinetic_energy;
}

double Global_Kinetic_Energy(CONTROLLER* controller,
                             const DOMAIN_INFORMATION* dd)
{
    const double local = Local_Kinetic_Energy(controller, dd);
    double global = local;
#ifdef USE_MPI
    const int status = MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                                     CONTROLLER::pp_comm);
    if (status != MPI_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe PP kinetic-energy reduction failed\n");
    }
#endif
    return global;
}
}  // namespace

void INITIAL_VELOCITY_INFORMATION::Initial(CONTROLLER* controller,
                                           MD_INFORMATION* md_info)
{
    const bool has_mode = controller->Command_Exist(kModuleName, "mode");
    const bool has_seed = controller->Command_Exist(kModuleName, "seed");
    if (!has_mode && !has_seed) return;
    if (!has_mode)
    {
        Throw_Input_Error(controller, "'mode' is required");
    }
    if (!has_seed)
    {
        Throw_Input_Error(controller, "'seed' is required for mode=maxwell");
    }
    if (!controller->Command_Choice(kModuleName, "mode", "maxwell"))
    {
        Throw_Input_Error(controller, "'mode' must be \"maxwell\"");
    }
    if (!Parse_Seed(controller->Command(kModuleName, "seed"), &seed))
    {
        Throw_Input_Error(controller,
                          "'seed' must be an integer in [0, INT64_MAX]");
    }
    if (md_info->mode == md_info->MINIMIZATION ||
        md_info->mode == md_info->RERUN)
    {
        Throw_Input_Error(
            controller,
            "mode=maxwell is not defined for minimization or rerun mode");
    }
    if (md_info->atom_numbers <= 0 || md_info->velocity == nullptr ||
        md_info->vel == nullptr || md_info->h_mass == nullptr)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kErrorBy,
            "Reason:\n\tthe atom velocity or mass storage is unavailable\n");
    }
    if (md_info->sys.freedom <= 0)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand, kErrorBy,
            "Reason:\n\tMaxwell initialization requires positive final "
            "degrees of freedom; got %d\n",
            md_info->sys.freedom);
    }
    if (!Float_Bits_Are_Finite(&md_info->sys.target_temperature) ||
        !(md_info->sys.target_temperature > 0.0f))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, kErrorBy,
            "Reason:\n\tMaxwell initialization requires a finite positive "
            "step-zero target temperature\n");
    }

    if (CONTROLLER::MPI_rank == SPONGE_MPI_ROOT)
    {
        Generate_Maxwell_Direction(controller, md_info, seed);
    }
#ifdef USE_MPI
    const std::size_t byte_count =
        sizeof(VECTOR) * static_cast<std::size_t>(md_info->atom_numbers);
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kErrorBy,
            "Reason:\n\tthe initial velocity field is too large for one MPI "
            "broadcast\n");
    }
    const int status =
        MPI_Bcast(md_info->velocity, static_cast<int>(byte_count), MPI_BYTE,
                  SPONGE_MPI_ROOT, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kErrorBy,
            "Reason:\n\tthe initial velocity broadcast failed\n");
    }
#endif
    deviceMemcpy(
        md_info->vel, md_info->velocity,
        sizeof(VECTOR) * static_cast<std::size_t>(md_info->atom_numbers),
        deviceMemcpyHostToDevice);
    is_initialized = true;
    controller->printf(
        "INITIAL VELOCITY: mode=maxwell, seed %llu, target %.9g K, final "
        "freedom %d\n",
        static_cast<unsigned long long>(seed),
        static_cast<double>(md_info->sys.target_temperature),
        md_info->sys.freedom);
}

void INITIAL_VELOCITY_INFORMATION::Finalize(CONTROLLER* controller,
                                            MD_INFORMATION* md_info,
                                            DOMAIN_INFORMATION* dd,
                                            SETTLE* settle, SHAKE* shake)
{
    if (!is_initialized || is_finalized) return;
    if (CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
    {
        is_finalized = true;
        return;
    }

    if (!settle->Project_Velocity_To_Constraint_Manifold(
            dd->vel, dd->crd, dd->d_mass_inverse, md_info->pbc.cell,
            md_info->pbc.rcell, false))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe initial SETTLE velocity projection did not "
            "converge\n");
    }
    if (!shake->Project_Velocity_To_Constraint_Manifold(
            dd->vel, dd->crd, dd->d_mass_inverse, md_info->pbc.cell,
            md_info->pbc.rcell, dd->atom_numbers, false))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe initial SHAKE velocity projection did not "
            "converge\n");
    }

    const double kinetic_energy = Global_Kinetic_Energy(controller, dd);
    if (!Double_Bits_Are_Finite(&kinetic_energy) || !(kinetic_energy > 0.0))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tconstraint projection left no finite positive "
            "kinetic energy to rescale\n");
    }
    const double projected_temperature =
        2.0 * kinetic_energy /
        (static_cast<double>(md_info->sys.freedom) *
         static_cast<double>(CONSTANT_kB));
    if (!Double_Bits_Are_Finite(&projected_temperature) ||
        !(projected_temperature > 0.0))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe projected initial temperature is not finite and "
            "positive\n");
    }
    const double target_kinetic_energy =
        0.5 * static_cast<double>(md_info->sys.freedom) *
        static_cast<double>(CONSTANT_kB) *
        static_cast<double>(md_info->sys.target_temperature);
    const double scale_double =
        std::sqrt(target_kinetic_energy / kinetic_energy);
    const float scale = static_cast<float>(scale_double);
    if (!Double_Bits_Are_Finite(&target_kinetic_energy) ||
        !(target_kinetic_energy > 0.0) ||
        !Double_Bits_Are_Finite(&scale_double) || !(scale_double > 0.0) ||
        !Float_Bits_Are_Finite(&scale) || !(scale > 0.0f))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe target initial kinetic energy cannot be "
            "represented by a finite velocity scale\n");
    }
    Scale_List(reinterpret_cast<float*>(dd->vel), scale, 3 * dd->atom_numbers);

    const double final_kinetic_energy = Global_Kinetic_Energy(controller, dd);
    const double final_temperature =
        2.0 * final_kinetic_energy /
        (static_cast<double>(md_info->sys.freedom) *
         static_cast<double>(CONSTANT_kB));
    const float stored_temperature = static_cast<float>(final_temperature);
    if (!Double_Bits_Are_Finite(&final_temperature) ||
        !Float_Bits_Are_Finite(&stored_temperature) ||
        !(stored_temperature > 0.0f))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tthe finalized initial temperature is not finite and "
            "positive\n");
    }
    const double relative_temperature_error =
        std::abs(final_temperature -
                 static_cast<double>(md_info->sys.target_temperature)) /
        static_cast<double>(md_info->sys.target_temperature);
    if (!Double_Bits_Are_Finite(&relative_temperature_error) ||
        relative_temperature_error > 1.0e-5)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorSimulationBreakDown, kFinalizeErrorBy,
            "Reason:\n\tfinal initial temperature %.9g K differs from target "
            "%.9g K after float velocity scaling\n",
            final_temperature,
            static_cast<double>(md_info->sys.target_temperature));
    }
    dd->h_ek_total = static_cast<float>(final_kinetic_energy);
    dd->temperature = stored_temperature;
    md_info->sys.h_temperature = stored_temperature;
    is_finalized = true;
    controller->printf("INITIAL VELOCITY: projected %.9g K, final %.9g K\n",
                       projected_temperature,
                       static_cast<double>(stored_temperature));
}
