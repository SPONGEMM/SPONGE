#pragma once
#include <array>
#include <string>
#include <vector>

#include "../MD_core/dynamics_types.h"

namespace SpongeDynamics
{

// Public units: ps, pN, nm/ps^2, V/nm and pN nm. Conversion to the
// engine's units happens once, at the runtime boundary.
enum class DriveKind
{
    Force,
    Acceleration,
    ElectricField,
    GroupLoad
};
enum class ScheduleKind
{
    Constant,
    Ramp,
    Sine,
    Table
};

struct Schedule
{
    ScheduleKind kind = ScheduleKind::Constant;
    double start = 0.0;
    double duration = 0.0;
    double period = 0.0;
    double phase = 0.0;
    std::vector<std::array<double, 2>> points;
    double Scale(double time_ps) const;
};

struct SpatialProfile
{
    bool enabled = false;
    int axis = 2, mode = 1;
};
struct Drive
{
    DriveKind kind = DriveKind::Force;
    std::vector<int> atoms;  // Empty only when the selection was omitted: all.
    Vec3 value{0, 0, 0};
    Vec3 torque{0, 0, 0};
    Schedule schedule;
    SpatialProfile profile;
};

// Other top-level MD settings are left to their owning modules. Inside
// [[external_force]] every key is checked, including nested schedule tables.
std::vector<Drive> Parse_Drives(const std::string& toml,
                                const std::string& source);
void Validate_Selections(const std::vector<Drive>& drives, int atom_count);
}  // namespace SpongeDynamics
