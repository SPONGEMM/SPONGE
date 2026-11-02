#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "utils/h5md/input_plan.hpp"

class FakeController
{
   public:
    bool Command_Exist(const char* key) const
    {
        return key != nullptr && commands_.count(key) != 0;
    }

    const char* Command(const char* key) const
    {
        const auto iter = commands_.find(key == nullptr ? "" : key);
        if (iter == commands_.end())
        {
            return "";
        }
        return iter->second.c_str();
    }

    void Set(const char* key, const char* value) { commands_[key] = value; }

   private:
    std::map<std::string, std::string> commands_;
};

namespace
{
void Expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

void Expect_Error_Contains(const SpongeH5InputPlan::ResolvedInputPlan& plan,
                           const char* needle)
{
    Expect(!plan.valid, "Expected input plan to be invalid");
    if (plan.error_message.find(needle) == std::string::npos)
    {
        std::cerr << "Expected error to contain '" << needle << "', got '"
                  << plan.error_message << "'\n";
        std::exit(1);
    }
}

FakeController Standard_H5_Controller()
{
    FakeController controller;
    controller.Set("mode", "npt");
    controller.Set("input_h5_topology_path",
                   "topologies/protein.topology.spgt.h5");
    controller.Set("input_h5_protocol_path",
                   "protocols/metadyn.protocol.spgp.h5");
    controller.Set("input_h5_restart_path", "runs/prod_0007.restart.spgr.h5");
    return controller;
}

void Expect_External_Format_Rejected(const char* key, const char* format)
{
    FakeController controller = Standard_H5_Controller();
    controller.Set(key, "external.input");
    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    Expect_Error_Contains(plan, "support only SPONGE native inputs");
    Expect_Error_Contains(plan, format);
    Expect_Error_Contains(plan, key);
}
}  // namespace

int main()
{
    {
        FakeController controller;
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid, "Legacy empty input plan should be valid");
        Expect(!plan.any_h5_input_enabled,
               "Legacy empty input plan should not enable H5 input");
        Expect(plan.legacy_input_allowed,
               "Legacy empty input plan should allow legacy input");
    }

    {
        FakeController controller = Standard_H5_Controller();
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid, "Standard H5 input plan should be valid");
        Expect(plan.any_h5_input_enabled,
               "Standard H5 input plan should enable H5 input");
        Expect(!plan.legacy_input_allowed,
               "Standard H5 input plan should disable legacy fallback");
        Expect(plan.topology.has_recommended_suffix,
               "Topology suffix should be recognized");
        Expect(plan.protocol.has_recommended_suffix,
               "Protocol suffix should be recognized");
        Expect(plan.restart.binding.has_recommended_suffix,
               "Restart suffix should be recognized");
        Expect(plan.restart.load_policy ==
                   SpongeH5InputContract::RestartLoadPolicy::structural,
               "Restart load should default to structural");
    }

    {
        Expect_External_Format_Rejected("gromacs_top", "GROMACS");
        Expect_External_Format_Rejected("gromacs_gro", "GROMACS");
        Expect_External_Format_Rejected("gromacs_include_dir", "GROMACS");
        Expect_External_Format_Rejected("gromacs_define", "GROMACS");
        Expect_External_Format_Rejected("amber_parm7", "AMBER");
        Expect_External_Format_Rejected("amber_rst7", "AMBER");
    }

    {
        FakeController controller;
        controller.Set("gromacs_top", "system.top");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid,
               "GROMACS input without H5 bindings should remain valid");
    }

    {
        FakeController controller;
        controller.Set("amber_parm7", "system.parm7");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid,
               "AMBER input without H5 bindings should remain valid");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("input_h5_restart_load", "protocol");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid, "Protocol restart load should be valid");
        Expect(plan.restart.load_policy ==
                   SpongeH5InputContract::RestartLoadPolicy::protocol,
               "Restart load should parse protocol");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("input_h5_restart_load", "fresh");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "invalid input_h5_restart_load");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("input_h5_restart_load", "custom");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "custom is reserved");
    }

    {
        FakeController controller;
        controller.Set("input_h5_topology_path",
                       "topologies/protein.topology.spgt.h5");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "input_h5_protocol_path");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("mode", "rerun");
        controller.Set("input_h5_trajectory_path", "runs/prod.spg.h5md");
        controller.Set("input_h5_trajectory_particle_stream", "solute");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid, "H5 rerun trajectory input should be valid");
        Expect(plan.trajectory.binding.enabled,
               "H5 rerun trajectory binding should be enabled");
        Expect(plan.trajectory.binding.has_recommended_suffix,
               "Trajectory suffix should be recognized");
        Expect(plan.trajectory.particle_stream == "solute",
               "Trajectory particle stream should be parsed");
    }

    {
        FakeController controller;
        controller.Set("mode", "rerun");
        controller.Set("input_h5_topology_path",
                       "topologies/protein.topology.spgt.h5");
        controller.Set("input_h5_protocol_path",
                       "protocols/metadyn.protocol.spgp.h5");
        controller.Set("input_h5_trajectory_path", "runs/prod.spg.h5md");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect(plan.valid,
               "H5 rerun trajectory input should not require restart binding");
        Expect(!plan.restart.binding.enabled,
               "H5 rerun trajectory-only plan should leave restart disabled");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("input_h5_trajectory_path", "runs/prod.spg.h5md");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "mode = rerun");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("mode", "rerun");
        controller.Set("input_h5_trajectory_path", "runs/prod.spg.h5md");
        controller.Set("crd", "legacy.crd");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "crd/box/vel");
    }

    {
        FakeController controller = Standard_H5_Controller();
        controller.Set("coordinate_in_file", "legacy_coordinate.txt");
        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        Expect_Error_Contains(plan, "coordinate/velocity");
    }

    return 0;
}
