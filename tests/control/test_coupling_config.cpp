#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "external_force/drive_config.h"
#include "third_party/toml/toml.h"

void Check(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
void Reject(const std::string& input, const std::string& detail)
{
    try
    {
        SpongeDynamics::Parse_Drives(input, "test.toml");
    }
    catch (const std::exception& error)
    {
        Check(std::string(error.what()).find(detail) != std::string::npos,
              error.what());
        return;
    }
    std::cerr << "invalid configuration was accepted:\n" << input << '\n';
    std::exit(1);
}
int main()
{
    using namespace SpongeDynamics;
    Check(Parse_Drives("mode='nve'", "test.toml").empty(),
          "absent coupling changes input");
    const std::string input = R"(
[[external_force]]
type='acceleration'
acceleration=[1,2,3]
atoms=[5,0]
schedule={type='ramp',start=2,duration=4}
[[external_force]]
type='electric_field'
field=[0,0,0.1]
schedule={type='sine',period=4,phase=0}
[[external_force]]
type='group_load'
atoms=[0,1,2]
torque=[0,0,1]
schedule={type='table',points=[[0,0],[2,1],[4,-1]]}
)";
    auto drives = Parse_Drives(input, "test.toml");
    Check(drives.size() == 3 && drives[0].atoms[0] == 5,
          "lost driver or global selection");
    Validate_Selections(drives, 6);
    try
    {
        Validate_Selections(drives, 5);
        Check(false, "out of range selection accepted");
    }
    catch (const std::runtime_error&)
    {
    }
    Check(drives[0].schedule.Scale(1) == 0 &&
              drives[0].schedule.Scale(4) == 0.5 &&
              drives[0].schedule.Scale(10) == 1,
          "ramp endpoints or absolute time incorrect");
    Check(std::abs(drives[1].schedule.Scale(1) - 1) < 1e-14,
          "sine period units incorrect");
    Check(drives[2].schedule.Scale(-1) == 0 &&
              drives[2].schedule.Scale(1) == 0.5 &&
              drives[2].schedule.Scale(3) == 0 &&
              drives[2].schedule.Scale(10) == -1,
          "table interpolation/clamping incorrect");
    // Repeated force evaluation must leave the protocol unchanged.
    drives[2].schedule.Scale(10);
    Check(drives[2].schedule.Scale(1) == 0.5, "schedule has hidden state");
    std::map<std::string, std::string> commands;
    std::string error;
    Check(sponge::toml_wrap::ParseAndFlatten(input, "test.toml", &commands,
                                             &error),
          error.c_str());
    auto roundtrip =
        Parse_Drives("external_force=" + commands.at("external_force"), "flat.toml");
    Check(roundtrip.size() == 3 && roundtrip[2].schedule.Scale(3) == 0,
          "structured controller serialization lost data");
    Reject("[external_force]\ntype='force'", "array-of-tables");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2]", "three numbers");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2,nan]", "finite");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2,3]\nforec=[1,2,3]",
           "unknown parameter");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2,3]\natoms=[]", "nonempty");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2,3]\natoms=[1,1]",
           "duplicate");
    Reject("[[external_force]]\ntype='force'\nforce=[1,2,3]\natoms=[1.5]", "integer");
    Reject("[[external_force]]\ntype='group_load'\ntorque=[0,0,1]", "selection");
    Reject(
        "[[external_force]]\ntype='force'\nforce=[0,0,1]\nschedule={type='ramp',"
        "duration=0}",
        "positive");
    Reject(
        "[[external_force]]\ntype='force'\nforce=[0,0,1]\nschedule={type='constant',"
        "period=1}",
        "unknown parameter");
    Reject(
        "[[external_force]]\ntype='force'\nforce=[0,0,1]\nschedule={type='table',"
        "points=[[1,0],[1,1]]}",
        "strictly increase");
    return 0;
}
