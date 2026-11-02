#pragma once

#include <filesystem>
#include <sstream>
#include <string>

#include "h5_bundle_test_common.hpp"

namespace SpongeH5InputMatrix
{
struct FixtureOutputPaths
{
    std::filesystem::path output_dir;
    std::filesystem::path mdout;
    std::filesystem::path mdinfo;
    std::filesystem::path h5_restart;
    std::filesystem::path h5_trajectory;
    std::filesystem::path h5_observable;
};

struct FixtureCasePaths
{
    std::filesystem::path root;
    std::filesystem::path mdin;
    std::filesystem::path topology_h5;
    std::filesystem::path protocol_h5;
    std::filesystem::path restart_h5;
    std::filesystem::path trajectory_h5;
    FixtureOutputPaths normal_output;
    FixtureOutputPaths rerun_output;
};

inline std::filesystem::path Fixture_Root()
{
#ifdef SPONGE_H5_INPUT_MATRIX_FIXTURE_ROOT
    return std::filesystem::path(SPONGE_H5_INPUT_MATRIX_FIXTURE_ROOT);
#else
    return std::filesystem::current_path() / "tests" / "h5_bundle" /
           "fixtures" / "input_matrix";
#endif
}

inline void Require_Path_Exists(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        std::ostringstream message;
        message << "missing H5 input matrix fixture path: " << path;
        throw TestFailure(message.str());
    }
}

inline std::filesystem::path Group_Path(const std::string& group)
{
    const auto path = Fixture_Root() / group;
    Require_Path_Exists(path);
    return path;
}

inline std::filesystem::path Core_Structural_Path()
{
    return Group_Path("core_structural");
}

inline std::filesystem::path Full_Contract_Rerun_Path()
{
    return Group_Path("full_contract_rerun");
}

inline FixtureOutputPaths Normal_Output_Paths(const std::filesystem::path& root)
{
    FixtureOutputPaths paths;
    paths.output_dir = root / "output";
    paths.mdout = root / "mdout.txt";
    paths.mdinfo = root / "mdinfo.txt";
    paths.h5_restart = paths.output_dir / "restart_out.spgr.h5";
    paths.h5_trajectory = paths.output_dir / "traj_out.spg.h5md";
    paths.h5_observable = paths.output_dir / "observable_out.obs.spg.h5md";
    return paths;
}

inline FixtureOutputPaths Rerun_Output_Paths(const std::filesystem::path& root)
{
    FixtureOutputPaths paths;
    paths.output_dir = root / "output";
    paths.mdout = root / "mdout.txt";
    paths.mdinfo = root / "mdinfo.txt";
    paths.h5_restart = std::filesystem::path();
    paths.h5_trajectory = paths.output_dir / "rerun_traj_out.spg.h5md";
    paths.h5_observable =
        paths.output_dir / "rerun_observable_out.obs.spg.h5md";
    return paths;
}

inline FixtureCasePaths Describe_Case(const std::filesystem::path& root,
                                      const std::string& mdin_name)
{
    Require_Path_Exists(root);
    FixtureCasePaths paths;
    paths.root = root;
    paths.mdin = root / mdin_name;
    paths.topology_h5 = root / "topology.spgt.h5";
    paths.protocol_h5 = root / "protocol.spgp.h5";
    paths.restart_h5 = root / "restart.spgr.h5";
    paths.trajectory_h5 = root / "trajectory.spg.h5md";
    Require_Path_Exists(paths.mdin);
    Require_Path_Exists(paths.topology_h5);
    Require_Path_Exists(paths.protocol_h5);
    Require_Path_Exists(paths.restart_h5);
    Require_Path_Exists(paths.trajectory_h5);
    paths.normal_output = Normal_Output_Paths(root);
    paths.rerun_output = Rerun_Output_Paths(root);
    return paths;
}

inline std::filesystem::path Copy_Case_To_Temp(
    const std::filesystem::path& source, const std::string& name)
{
    Require_Path_Exists(source);
    const auto destination = SpongeH5Test::Unique_Temp_Path(name);
    std::filesystem::create_directories(destination.parent_path());
    std::filesystem::copy(source, destination,
                          std::filesystem::copy_options::recursive);
    return destination;
}

inline FixtureCasePaths Copy_Case_To_Temp(const std::filesystem::path& source,
                                          const std::string& name,
                                          const std::string& mdin_name)
{
    return Describe_Case(Copy_Case_To_Temp(source, name), mdin_name);
}
}  // namespace SpongeH5InputMatrix
