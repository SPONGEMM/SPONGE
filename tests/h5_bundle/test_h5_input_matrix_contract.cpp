#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "h5_input_matrix_fixture.hpp"
#include "utils/h5md/h5_legacy_sidecar.hpp"
#include "utils/h5md/input_plan.hpp"

using SpongeH5InputMatrix::Core_Structural_Path;

namespace
{
std::filesystem::path Core_Legacy_Input()
{
    return Core_Structural_Path() / "legacy_input";
}

std::filesystem::path Core_Bundled_Input()
{
    return Core_Structural_Path() / "bundled_input" / "bundle";
}

std::filesystem::path Core_Bundled_Input_With_Sidecar()
{
    return Core_Structural_Path() / "bundled_input_with_legacy_sidecar" /
           "bundle";
}

std::filesystem::path Full_Contract_Bundled_Input_With_Sidecar()
{
    return SpongeH5InputMatrix::Full_Contract_Rerun_Path() /
           "bundled_input_with_legacy_sidecar" / "bundle";
}

std::filesystem::path Full_Contract_Legacy_Input()
{
    return SpongeH5InputMatrix::Full_Contract_Rerun_Path() / "legacy_input";
}

void Require_Contains(const std::string& text, const std::string& needle)
{
    REQUIRE_TRUE(text.find(needle) != std::string::npos);
}

std::set<std::string> Sidecar_Key_Set(
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars)
{
    std::set<std::string> keys;
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(keys.insert(sidecar.key).second);
    }
    return keys;
}

std::string Read_Binary_File(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    REQUIRE_TRUE(in.good());
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

bool Has_Key(const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars,
             const std::string& key)
{
    return std::any_of(sidecars.begin(), sidecars.end(),
                       [&](const SpongeH5MD::LegacySidecarBinding& sidecar)
                       { return sidecar.key == key; });
}

std::string Path_For_Key(
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars,
    const std::string& key)
{
    for (const auto& sidecar : sidecars)
    {
        if (sidecar.key == key)
        {
            return sidecar.path;
        }
    }
    throw TestFailure("missing sidecar key: " + key);
}

struct ScopedCurrentPath
{
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : previous(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath() { std::filesystem::current_path(previous); }

    std::filesystem::path previous;
};

void Require_All_Sidecar_Paths_Resolved(
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars)
{
    REQUIRE_TRUE(!sidecars.empty());
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(!sidecar.key.empty());
        REQUIRE_TRUE(std::filesystem::path(sidecar.path).is_absolute());
        REQUIRE_TRUE(std::filesystem::exists(sidecar.path));
    }
}

void Require_All_Sidecar_Paths_In_Bundle_Tree(
    const std::filesystem::path& bundle_dir,
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars)
{
    const auto sidecar_root =
        std::filesystem::absolute(bundle_dir / "legacy_sidecars")
            .lexically_normal();
    REQUIRE_TRUE(std::filesystem::is_directory(sidecar_root));

    for (const auto& sidecar : sidecars)
    {
        const auto sidecar_path =
            std::filesystem::absolute(std::filesystem::path(sidecar.path))
                .lexically_normal();
        const auto relative = sidecar_path.lexically_relative(sidecar_root);
        std::vector<std::filesystem::path> components;
        for (const auto& component : relative)
        {
            components.push_back(component);
        }

        REQUIRE_EQ(components.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(components[0].string(), sidecar.key);
        REQUIRE_TRUE(!components[1].empty());
        REQUIRE_TRUE(sidecar_path.parent_path().filename() == sidecar.key);
        REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
    }
}

void Require_All_Sidecar_Keys_Are_Allowed(
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars,
    const std::set<std::string>& allowed_keys)
{
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(
            SpongeH5MD::Command_Key_Allowed(allowed_keys, sidecar.key));
    }
}

void Set_Core_H5_Bindings(CONTROLLER* controller,
                          const std::filesystem::path& bundle_dir)
{
    controller->Set("mode", "nve");
    controller->Set("input_h5_topology_path",
                    (bundle_dir / "topology.spgt.h5").string());
    controller->Set("input_h5_protocol_path",
                    (bundle_dir / "protocol.spgp.h5").string());
    controller->Set("input_h5_restart_path",
                    (bundle_dir / "restart.spgr.h5").string());
    controller->Set("input_h5_restart_load", "full");
}

void Set_Full_Rerun_H5_Bindings(CONTROLLER* controller,
                                const std::filesystem::path& bundle_dir)
{
    controller->Set("mode", "rerun");
    controller->Set("input_h5_topology_path",
                    (bundle_dir / "topology.spgt.h5").string());
    controller->Set("input_h5_protocol_path",
                    (bundle_dir / "protocol.spgp.h5").string());
    controller->Set("input_h5_restart_path",
                    (bundle_dir / "restart.spgr.h5").string());
    controller->Set("input_h5_restart_load", "full");
    controller->Set("input_h5_trajectory_path",
                    (bundle_dir / "trajectory.spg.h5md").string());
    controller->Set("input_h5_trajectory_particle_stream", "all");
}

std::vector<SpongeH5MD::LegacySidecarBinding> Read_Sidecars(
    const std::filesystem::path& path)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(path.string(),
                                                          &sidecars, &error));
    return sidecars;
}

void Require_H5_Legacy_Sidecar_Table(const std::filesystem::path& path)
{
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::legacy_sidecar_keys));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::legacy_sidecar_paths));
}

void Test_Legacy_Input_Allows_Legacy_Path()
{
    const auto legacy = Core_Legacy_Input();
    SpongeH5InputMatrix::Require_Path_Exists(legacy / "mdin.spg.toml");

    CONTROLLER controller;
    controller.Set("mode", "nve");
    controller.Set("coordinate_in_file", (legacy / "coordinate.txt").string());
    controller.Set("velocity_in_file", (legacy / "velocity.txt").string());
    controller.Set("mass_in_file", (legacy / "mass.txt").string());

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(!plan.any_h5_input_enabled);
    REQUIRE_TRUE(plan.legacy_input_allowed);
}

void Test_Empty_H5_Input_Paths_Do_Not_Disable_Legacy_Fallback()
{
    REQUIRE_TRUE(!SpongeH5InputPlan::Has_H5_Input_Binding<CONTROLLER>(nullptr));

    CONTROLLER controller;
    controller.Set("mode", "nve");
    controller.Set(SpongeH5InputContract::kTopologyPathKey, "");
    controller.Set(SpongeH5InputContract::kProtocolPathKey, "");
    controller.Set(SpongeH5InputContract::kRestartPathKey, "");
    controller.Set(SpongeH5InputContract::kTrajectoryPathKey, "");
    controller.Set(SpongeH5InputContract::kTrajectoryParticleStreamKey,
                   "solute");

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(!plan.any_h5_input_enabled);
    REQUIRE_TRUE(plan.legacy_input_allowed);
    REQUIRE_TRUE(!plan.topology.enabled);
    REQUIRE_TRUE(!plan.protocol.enabled);
    REQUIRE_TRUE(!plan.restart.binding.enabled);
    REQUIRE_TRUE(!plan.trajectory.binding.enabled);
    REQUIRE_EQ(plan.trajectory.particle_stream, std::string("solute"));
    REQUIRE_TRUE(!SpongeH5InputPlan::Has_H5_Input_Binding(&controller));

    controller.Set(SpongeH5InputContract::kTopologyPathKey, "system.spgt.h5");
    REQUIRE_TRUE(SpongeH5InputPlan::Has_H5_Input_Binding(&controller));
}

void Test_H5_Input_Path_Suffix_Flags_Are_Non_Fatal()
{
    CONTROLLER controller;
    controller.Set("mode", "rerun");
    controller.Set(SpongeH5InputContract::kTopologyPathKey, "system.h5");
    controller.Set(SpongeH5InputContract::kProtocolPathKey, "protocol.h5");
    controller.Set(SpongeH5InputContract::kRestartPathKey, "restart.h5");
    controller.Set(SpongeH5InputContract::kTrajectoryPathKey, "trajectory.h5");

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_input_enabled);
    REQUIRE_TRUE(!plan.legacy_input_allowed);
    REQUIRE_TRUE(plan.topology.enabled);
    REQUIRE_TRUE(plan.protocol.enabled);
    REQUIRE_TRUE(plan.restart.binding.enabled);
    REQUIRE_TRUE(plan.trajectory.binding.enabled);
    REQUIRE_TRUE(!plan.topology.has_recommended_suffix);
    REQUIRE_TRUE(!plan.protocol.has_recommended_suffix);
    REQUIRE_TRUE(!plan.restart.binding.has_recommended_suffix);
    REQUIRE_TRUE(!plan.trajectory.binding.has_recommended_suffix);
}

void Test_Pure_Bundled_Input_Has_No_Legacy_Sidecars()
{
    const auto bundle = Core_Bundled_Input();
    SpongeH5InputMatrix::Require_Path_Exists(bundle / "mdin.bundled.spg.toml");

    CONTROLLER controller;
    Set_Core_H5_Bindings(&controller, bundle);

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_input_enabled);
    REQUIRE_TRUE(!plan.legacy_input_allowed);
    REQUIRE_TRUE(plan.topology.enabled);
    REQUIRE_TRUE(plan.protocol.enabled);
    REQUIRE_TRUE(plan.restart.binding.enabled);
    REQUIRE_TRUE(plan.restart.load_policy ==
                 SpongeH5InputContract::RestartLoadPolicy::full);

    const auto topology_sidecars = Read_Sidecars(bundle / "topology.spgt.h5");
    const auto protocol_sidecars = Read_Sidecars(bundle / "protocol.spgp.h5");
    const auto restart_sidecars = Read_Sidecars(bundle / "restart.spgr.h5");
    REQUIRE_TRUE(topology_sidecars.empty());
    REQUIRE_TRUE(protocol_sidecars.empty());
    REQUIRE_TRUE(restart_sidecars.empty());

    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &controller, (bundle / "topology.spgt.h5").string(),
        SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology", &error));
    REQUIRE_TRUE(!controller.Command_Exist("mass_in_file"));
}

void Test_Bundled_Input_With_Sidecar_Injects_Allowed_Keys()
{
    const auto bundle = Core_Bundled_Input_With_Sidecar();
    const auto topology_path = bundle / "topology.spgt.h5";
    const auto protocol_path = bundle / "protocol.spgp.h5";
    const auto restart_path = bundle / "restart.spgr.h5";

    Require_H5_Legacy_Sidecar_Table(topology_path);
    Require_H5_Legacy_Sidecar_Table(protocol_path);
    Require_H5_Legacy_Sidecar_Table(restart_path);

    const auto topology_sidecars = Read_Sidecars(topology_path);
    REQUIRE_TRUE(Has_Key(topology_sidecars, "mass_in_file"));
    REQUIRE_TRUE(Has_Key(topology_sidecars, "charge_in_file"));
    Require_All_Sidecar_Paths_Resolved(topology_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, topology_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        topology_sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys());
    REQUIRE_EQ(Path_For_Key(topology_sidecars, "mass_in_file"),
               std::filesystem::absolute(bundle / "legacy_sidecars" /
                                         "mass_in_file" / "mass.txt")
                   .lexically_normal()
                   .string());

    const auto protocol_sidecars = Read_Sidecars(protocol_path);
    REQUIRE_TRUE(Has_Key(protocol_sidecars, "cv_in_file"));
    REQUIRE_TRUE(Has_Key(protocol_sidecars, "restrain_in_file"));
    Require_All_Sidecar_Paths_Resolved(protocol_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, protocol_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        protocol_sidecars, SpongeH5MD::H5_Protocol_Sidecar_Command_Keys());

    const auto restart_sidecars = Read_Sidecars(restart_path);
    REQUIRE_TRUE(Has_Key(restart_sidecars, "SITS_nk_in_file"));
    REQUIRE_TRUE(Has_Key(restart_sidecars, "cv_in_file"));
    REQUIRE_TRUE(Has_Key(restart_sidecars, "restrain_in_file"));
    REQUIRE_TRUE(Has_Key(restart_sidecars, "SITS_in_file"));
    Require_All_Sidecar_Paths_Resolved(restart_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, restart_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        restart_sidecars, SpongeH5MD::H5_Protocol_Sidecar_Command_Keys());

    CONTROLLER topology_controller;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &topology_controller, topology_path.string(),
        SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology", &error));
    REQUIRE_TRUE(topology_controller.Command_Exist("mass_in_file"));
    REQUIRE_TRUE(topology_controller.Command_Exist("charge_in_file"));
    REQUIRE_EQ(topology_controller.Check_Value("mass_in_file"), 0);

    CONTROLLER protocol_controller;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &protocol_controller, protocol_path.string(),
        SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(), "protocol", &error));
    REQUIRE_TRUE(protocol_controller.Command_Exist("cv_in_file"));
    REQUIRE_TRUE(protocol_controller.Command_Exist("restrain_in_file"));
    REQUIRE_EQ(protocol_controller.Check_Value("cv_in_file"), 0);

    CONTROLLER restart_controller;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &restart_controller, restart_path.string(),
        SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(), "restart", &error));
    REQUIRE_TRUE(restart_controller.Command_Exist("SITS_nk_in_file"));
    REQUIRE_EQ(restart_controller.Check_Value("SITS_nk_in_file"), 0);
}

void Test_Legacy_Sidecar_Override_Conflict()
{
    const auto bundle = Core_Bundled_Input_With_Sidecar();
    SpongeH5InputMatrix::Require_Path_Exists(bundle /
                                             "mdin.override_conflict.spg.toml");
    const auto topology_path = bundle / "topology.spgt.h5";
    const auto sidecars = Read_Sidecars(topology_path);
    const std::string h5_mass_path = Path_For_Key(sidecars, "mass_in_file");

    CONTROLLER same_path_controller;
    same_path_controller.Set("mass_in_file", h5_mass_path);
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &same_path_controller, topology_path.string(),
        SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology", &error));
    REQUIRE_EQ(std::string(same_path_controller.Command("mass_in_file")),
               h5_mass_path);

    CONTROLLER relative_same_path_controller;
    relative_same_path_controller.Set(
        "charge_in_file", "legacy_sidecars/charge_in_file/charge.txt");
    {
        const ScopedCurrentPath cwd(bundle);
        REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
            &relative_same_path_controller, topology_path.string(),
            SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology",
            &error));
    }
    REQUIRE_EQ(
        std::string(relative_same_path_controller.Command("charge_in_file")),
        "legacy_sidecars/charge_in_file/charge.txt");

    CONTROLLER conflict_controller;
    conflict_controller.Set("mass_in_file",
                            (bundle / "override_mass.txt").string());
    REQUIRE_TRUE(!SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &conflict_controller, topology_path.string(),
        SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology", &error));
    Require_Contains(error, "conflicts with existing command mass_in_file");
}

void Test_Full_Contract_Sidecar_Key_Sets_Are_Injectable()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();
    const auto topology_path = bundle / "topology.spgt.h5";
    const auto protocol_path = bundle / "protocol.spgp.h5";
    const auto restart_path = bundle / "restart.spgr.h5";

    const auto topology_sidecars = Read_Sidecars(topology_path);
    const std::set<std::string> expected_topology_keys = {
        "pairwise_force_in_file",
        "listed_forces_in_file",
        "mass_in_file",
        "charge_in_file",
        "exclude_in_file",
        "bond_in_file",
        "angle_in_file",
        "dihedral_in_file",
        "LJ_in_file",
        "nb14_extra_in_file",
        "urey_bradley_in_file",
        "cmap_in_file",
        "gb_in_file",
        "virtual_atom_in_file",
        "LJ_soft_core_in_file",
        "subsys_division_in_file",
        "EAM_in_file",
        "EAM_atom_type_in_file",
        "SW_in_file",
        "EDIP_in_file",
        "TERSOFF_in_file",
        "REAXFF_in_file",
        "REAXFF_type_in_file"};
    REQUIRE_TRUE(Sidecar_Key_Set(topology_sidecars) == expected_topology_keys);
    Require_All_Sidecar_Paths_Resolved(topology_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, topology_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        topology_sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys());

    const auto protocol_sidecars = Read_Sidecars(protocol_path);
    const std::set<std::string> expected_protocol_keys = {
        "cv_in_file",        "constrain_in_file",
        "restrain_in_file",  "soft_walls_in_file",
        "SITS_in_file",      "SITS_atom_in_file",
        "restrain_atom_id",  "restrain_weight_in_file",
        "meta_edge_in_file", "restrain_cv_in_file",
        "steer_cv_in_file"};
    REQUIRE_TRUE(Sidecar_Key_Set(protocol_sidecars) == expected_protocol_keys);
    Require_All_Sidecar_Paths_Resolved(protocol_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, protocol_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        protocol_sidecars, SpongeH5MD::H5_Protocol_Sidecar_Command_Keys());

    const auto restart_sidecars = Read_Sidecars(restart_path);
    const std::set<std::string> expected_restart_keys = {
        "SITS_nk_in_file",
        "restrain_coordinate_in_file",
        "meta_potential_in_file",
        "meta_scatter_in_file",
        "nose_hoover_chain_restart_input",
        "hills_in_file",
        "cv_in_file",
        "constrain_in_file",
        "restrain_in_file",
        "pairwise_force_in_file",
        "listed_forces_in_file",
        "soft_walls_in_file",
        "SITS_in_file",
        "SITS_atom_in_file",
        "restrain_atom_id",
        "restrain_weight_in_file",
        "meta_edge_in_file",
        "restrain_cv_in_file",
        "steer_cv_in_file"};
    REQUIRE_TRUE(Sidecar_Key_Set(restart_sidecars) == expected_restart_keys);
    Require_All_Sidecar_Paths_Resolved(restart_sidecars);
    Require_All_Sidecar_Paths_In_Bundle_Tree(bundle, restart_sidecars);
    Require_All_Sidecar_Keys_Are_Allowed(
        restart_sidecars, SpongeH5MD::H5_Protocol_Sidecar_Command_Keys());

    CONTROLLER topology_controller;
    CONTROLLER protocol_controller;
    CONTROLLER restart_controller;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &topology_controller, topology_path.string(),
        SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "full topology",
        &error));
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &protocol_controller, protocol_path.string(),
        SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(), "full protocol",
        &error));
    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
        &restart_controller, restart_path.string(),
        SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(), "full restart",
        &error));

    for (const auto& key : expected_topology_keys)
    {
        REQUIRE_TRUE(topology_controller.Command_Exist(key.c_str()));
        REQUIRE_EQ(topology_controller.Check_Value(key), 0);
    }
    for (const auto& key : expected_protocol_keys)
    {
        REQUIRE_TRUE(protocol_controller.Command_Exist(key.c_str()));
        REQUIRE_EQ(protocol_controller.Check_Value(key), 0);
    }
    for (const auto& key : expected_restart_keys)
    {
        REQUIRE_TRUE(restart_controller.Command_Exist(key.c_str()));
        REQUIRE_EQ(restart_controller.Check_Value(key), 0);
    }
}

void Test_Full_Contract_Sidecars_Match_Legacy_Source_Files()
{
    const auto fixture_root = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto legacy_root = fixture_root / "legacy_input";
    const auto bundle =
        fixture_root / "bundled_input_with_legacy_sidecar" / "bundle";
    const auto expected_file_by_key = std::map<std::string, std::string>{
        {"pairwise_force_in_file", "pairwise_force.txt"},
        {"listed_forces_in_file", "listed_forces.txt"},
        {"mass_in_file", "mass.txt"},
        {"charge_in_file", "charge.txt"},
        {"residue_in_file", "residue.txt"},
        {"exclude_in_file", "exclude.txt"},
        {"bond_in_file", "bond.txt"},
        {"angle_in_file", "angle.txt"},
        {"dihedral_in_file", "dihedral.txt"},
        {"LJ_in_file", "lj.txt"},
        {"nb14_extra_in_file", "nb14_extra.txt"},
        {"urey_bradley_in_file", "urey_bradley.txt"},
        {"cmap_in_file", "cmap.txt"},
        {"gb_in_file", "gb.txt"},
        {"virtual_atom_in_file", "virtual_atom.txt"},
        {"LJ_soft_core_in_file", "lj_soft_core.txt"},
        {"subsys_division_in_file", "subsys_division.txt"},
        {"EAM_in_file", "eam.txt"},
        {"EAM_atom_type_in_file", "eam_atom_type.txt"},
        {"SW_in_file", "sw.txt"},
        {"EDIP_in_file", "edip.txt"},
        {"TERSOFF_in_file", "tersoff.txt"},
        {"REAXFF_in_file", "reaxff.txt"},
        {"REAXFF_type_in_file", "reaxff_type.txt"},
        {"cv_in_file", "cv.txt"},
        {"constrain_in_file", "constrain.txt"},
        {"restrain_in_file", "restrain.txt"},
        {"soft_walls_in_file", "soft_walls.txt"},
        {"SITS_in_file", "sits.txt"},
        {"SITS_atom_in_file", "sits_atom.txt"},
        {"SITS_nk_in_file", "sits_nk.txt"},
        {"restrain_atom_id", "restrain_atom_id.txt"},
        {"restrain_coordinate_in_file", "restrain_coordinate.txt"},
        {"restrain_weight_in_file", "restrain_weight.txt"},
        {"meta_edge_in_file", "meta_edge.txt"},
        {"meta_potential_in_file", "meta_potential.txt"},
        {"meta_scatter_in_file", "meta_scatter.txt"},
        {"restrain_cv_in_file", "restrain_cv.txt"},
        {"steer_cv_in_file", "steer_cv.txt"},
        {"nose_hoover_chain_restart_input", "nhc_restart.txt"},
        {"hills_in_file", "hills.txt"},
    };

    std::vector<SpongeH5MD::LegacySidecarBinding> all_sidecars;
    for (const auto& h5_name :
         {"topology.spgt.h5", "protocol.spgp.h5", "restart.spgr.h5"})
    {
        const auto sidecars = Read_Sidecars(bundle / h5_name);
        all_sidecars.insert(all_sidecars.end(), sidecars.begin(),
                            sidecars.end());
    }

    REQUIRE_TRUE(!all_sidecars.empty());
    for (const auto& sidecar : all_sidecars)
    {
        const auto expected_iter = expected_file_by_key.find(sidecar.key);
        REQUIRE_TRUE(expected_iter != expected_file_by_key.end());
        const std::filesystem::path sidecar_path(sidecar.path);
        REQUIRE_TRUE(sidecar_path.parent_path().filename() == sidecar.key);
        REQUIRE_TRUE(sidecar_path.filename() == expected_iter->second);

        const auto legacy_source = legacy_root / expected_iter->second;
        SpongeH5InputMatrix::Require_Path_Exists(legacy_source);
        REQUIRE_EQ(Read_Binary_File(sidecar_path),
                   Read_Binary_File(legacy_source));
    }
}

void Test_H5_Restart_Rejects_Legacy_Restart_Inputs()
{
    const auto bundle = Core_Bundled_Input();
    const auto legacy = Core_Legacy_Input();

    struct LegacyRestartInput
    {
        const char* key;
        std::filesystem::path path;
    };
    const std::vector<LegacyRestartInput> legacy_restart_inputs = {
        {"coordinate_in_file", legacy / "coordinate.txt"},
        {"velocity_in_file", legacy / "velocity.txt"},
        {"rst7", legacy / "coordinate.txt"},
    };

    for (const auto& legacy_restart_input : legacy_restart_inputs)
    {
        CONTROLLER controller;
        Set_Core_H5_Bindings(&controller, bundle);
        controller.Set(legacy_restart_input.key,
                       legacy_restart_input.path.string());

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        Require_Contains(plan.error_message, "mutually exclusive");
        Require_Contains(plan.error_message,
                         "coordinate/velocity restart inputs");
    }
}

void Test_Full_Contract_Rerun_H5_Trajectory_Input_Resolves()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();
    SpongeH5InputMatrix::Require_Path_Exists(bundle / "trajectory.spg.h5md");

    CONTROLLER controller;
    Set_Full_Rerun_H5_Bindings(&controller, bundle);

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_input_enabled);
    REQUIRE_TRUE(!plan.legacy_input_allowed);
    REQUIRE_TRUE(plan.topology.enabled);
    REQUIRE_TRUE(plan.protocol.enabled);
    REQUIRE_TRUE(plan.restart.binding.enabled);
    REQUIRE_TRUE(plan.trajectory.binding.enabled);
    REQUIRE_TRUE(plan.topology.has_recommended_suffix);
    REQUIRE_TRUE(plan.protocol.has_recommended_suffix);
    REQUIRE_TRUE(plan.restart.binding.has_recommended_suffix);
    REQUIRE_TRUE(plan.trajectory.binding.has_recommended_suffix);
    REQUIRE_TRUE(plan.restart.load_policy ==
                 SpongeH5InputContract::RestartLoadPolicy::full);
    REQUIRE_EQ(plan.trajectory.particle_stream, std::string("all"));
}

void Test_H5_Trajectory_Rejects_Legacy_Rerun_Inputs()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();
    const auto legacy = Full_Contract_Legacy_Input();

    struct LegacyRerunInput
    {
        const char* key;
        std::filesystem::path path;
    };
    const std::vector<LegacyRerunInput> legacy_rerun_inputs = {
        {"crd", legacy / "traj.dat"},
        {"box", legacy / "traj_box.dat"},
        {"vel", legacy / "traj_vel.dat"},
    };

    for (const auto& legacy_rerun_input : legacy_rerun_inputs)
    {
        CONTROLLER controller;
        Set_Full_Rerun_H5_Bindings(&controller, bundle);
        controller.Set(legacy_rerun_input.key,
                       legacy_rerun_input.path.string());

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        Require_Contains(plan.error_message, "mutually exclusive");
        Require_Contains(plan.error_message, "crd/box/vel");
    }
}

void Test_H5_Trajectory_Is_Rerun_Only()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();

    CONTROLLER controller;
    controller.Set("mode", "nve");
    controller.Set("input_h5_topology_path",
                   (bundle / "topology.spgt.h5").string());
    controller.Set("input_h5_protocol_path",
                   (bundle / "protocol.spgp.h5").string());
    controller.Set("input_h5_restart_path",
                   (bundle / "restart.spgr.h5").string());
    controller.Set("input_h5_trajectory_path",
                   (bundle / "trajectory.spg.h5md").string());

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(!plan.valid);
    Require_Contains(plan.error_message,
                     "input_h5_trajectory_path is currently only valid");
    Require_Contains(plan.error_message, "mode = rerun");
}

void Test_Rerun_H5_Trajectory_Input_Does_Not_Require_Restart_Binding()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();

    CONTROLLER controller;
    controller.Set("mode", "rerun");
    controller.Set("input_h5_topology_path",
                   (bundle / "topology.spgt.h5").string());
    controller.Set("input_h5_protocol_path",
                   (bundle / "protocol.spgp.h5").string());
    controller.Set("input_h5_trajectory_path",
                   (bundle / "trajectory.spg.h5md").string());
    controller.Set("input_h5_trajectory_particle_stream", "all");

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_input_enabled);
    REQUIRE_TRUE(!plan.restart.binding.enabled);
    REQUIRE_TRUE(plan.trajectory.binding.enabled);
    REQUIRE_EQ(plan.trajectory.particle_stream, std::string("all"));
}

void Test_Rerun_H5_Trajectory_Default_Stream_And_Mode_Case()
{
    const auto bundle = Full_Contract_Bundled_Input_With_Sidecar();

    CONTROLLER controller;
    controller.Set("mode", "RERUN");
    controller.Set("input_h5_topology_path",
                   (bundle / "topology.spgt.h5").string());
    controller.Set("input_h5_protocol_path",
                   (bundle / "protocol.spgp.h5").string());
    controller.Set("input_h5_trajectory_path",
                   (bundle / "trajectory.spg.h5md").string());

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_input_enabled);
    REQUIRE_TRUE(!plan.legacy_input_allowed);
    REQUIRE_TRUE(!plan.restart.binding.enabled);
    REQUIRE_TRUE(plan.trajectory.binding.enabled);
    REQUIRE_EQ(
        plan.trajectory.particle_stream,
        std::string(SpongeH5InputContract::kDefaultTrajectoryParticleStream));
}

void Test_Normal_H5_Input_Requires_Restart_Binding()
{
    const auto bundle = Core_Bundled_Input();

    CONTROLLER controller;
    controller.Set("mode", "nve");
    controller.Set("input_h5_topology_path",
                   (bundle / "topology.spgt.h5").string());
    controller.Set("input_h5_protocol_path",
                   (bundle / "protocol.spgp.h5").string());

    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    REQUIRE_TRUE(!plan.valid);
    Require_Contains(plan.error_message, "missing required H5 input binding");
    Require_Contains(plan.error_message, "input_h5_restart_path");
}

void Test_H5_Input_Requires_Topology_And_Protocol_Bindings()
{
    const auto bundle = Core_Bundled_Input();

    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_protocol_path",
                       (bundle / "protocol.spgp.h5").string());
        controller.Set("input_h5_restart_path",
                       (bundle / "restart.spgr.h5").string());

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.any_h5_input_enabled);
        REQUIRE_TRUE(!plan.legacy_input_allowed);
        Require_Contains(plan.error_message,
                         "missing required H5 input binding");
        Require_Contains(plan.error_message, "input_h5_topology_path");
    }

    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_topology_path",
                       (bundle / "topology.spgt.h5").string());
        controller.Set("input_h5_restart_path",
                       (bundle / "restart.spgr.h5").string());

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.any_h5_input_enabled);
        REQUIRE_TRUE(!plan.legacy_input_allowed);
        Require_Contains(plan.error_message,
                         "missing required H5 input binding");
        Require_Contains(plan.error_message, "input_h5_protocol_path");
    }
}

void Test_H5_Restart_Load_Policy_Validation()
{
    const auto bundle = Core_Bundled_Input();

    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_topology_path",
                       (bundle / "topology.spgt.h5").string());
        controller.Set("input_h5_protocol_path",
                       (bundle / "protocol.spgp.h5").string());
        controller.Set("input_h5_restart_path",
                       (bundle / "restart.spgr.h5").string());

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.restart.load_policy ==
                     SpongeH5InputContract::RestartLoadPolicy::structural);
    }

    struct ValidPolicyCase
    {
        const char* text;
        SpongeH5InputContract::RestartLoadPolicy expected;
    };
    const std::vector<ValidPolicyCase> valid_policies = {
        {"structural", SpongeH5InputContract::RestartLoadPolicy::structural},
        {"dynamic", SpongeH5InputContract::RestartLoadPolicy::dynamic},
        {"protocol", SpongeH5InputContract::RestartLoadPolicy::protocol},
        {"full", SpongeH5InputContract::RestartLoadPolicy::full},
    };

    for (const auto& valid_policy : valid_policies)
    {
        CONTROLLER controller;
        Set_Core_H5_Bindings(&controller, bundle);
        controller.Set("input_h5_restart_load", valid_policy.text);

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.restart.load_policy == valid_policy.expected);
    }

    {
        CONTROLLER controller;
        Set_Core_H5_Bindings(&controller, bundle);
        controller.Set("input_h5_restart_load", "everything");

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        Require_Contains(plan.error_message,
                         "invalid input_h5_restart_load value");
        Require_Contains(plan.error_message, "everything");
    }

    {
        CONTROLLER controller;
        Set_Core_H5_Bindings(&controller, bundle);
        controller.Set("input_h5_restart_load", "custom");

        const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
        REQUIRE_TRUE(!plan.valid);
        Require_Contains(plan.error_message,
                         "input_h5_restart_load = custom is reserved");
    }
}
}  // namespace

int main()
{
    try
    {
        Test_Legacy_Input_Allows_Legacy_Path();
        Test_Empty_H5_Input_Paths_Do_Not_Disable_Legacy_Fallback();
        Test_H5_Input_Path_Suffix_Flags_Are_Non_Fatal();
        Test_Pure_Bundled_Input_Has_No_Legacy_Sidecars();
        Test_Bundled_Input_With_Sidecar_Injects_Allowed_Keys();
        Test_Legacy_Sidecar_Override_Conflict();
        Test_Full_Contract_Sidecar_Key_Sets_Are_Injectable();
        Test_Full_Contract_Sidecars_Match_Legacy_Source_Files();
        Test_H5_Restart_Rejects_Legacy_Restart_Inputs();
        Test_Full_Contract_Rerun_H5_Trajectory_Input_Resolves();
        Test_H5_Trajectory_Rejects_Legacy_Rerun_Inputs();
        Test_H5_Trajectory_Is_Rerun_Only();
        Test_Rerun_H5_Trajectory_Input_Does_Not_Require_Restart_Binding();
        Test_Rerun_H5_Trajectory_Default_Stream_And_Mode_Case();
        Test_Normal_H5_Input_Requires_Restart_Binding();
        Test_H5_Input_Requires_Topology_And_Protocol_Bindings();
        Test_H5_Restart_Load_Policy_Validation();
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << '\n';
        return 1;
    }
    return 0;
}
