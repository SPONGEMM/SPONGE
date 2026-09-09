#include <array>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"

namespace
{
void Ensure_Group(HighFive::File& file, const std::string& group_path)
{
    if (group_path.empty() || group_path == "/")
    {
        return;
    }
    std::string current;
    std::size_t begin = 1;
    while (begin < group_path.size())
    {
        const std::size_t slash = group_path.find('/', begin);
        const std::string component = group_path.substr(
            begin,
            slash == std::string::npos ? std::string::npos : slash - begin);
        current += "/" + component;
        if (!file.exist(current))
        {
            file.createGroup(current);
        }
        if (slash == std::string::npos)
        {
            break;
        }
        begin = slash + 1;
    }
}

void Ensure_Parent_Group(HighFive::File& file, const std::string& dataset_path)
{
    const std::size_t slash = dataset_path.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
    {
        return;
    }
    Ensure_Group(file, dataset_path.substr(0, slash));
}

template <typename T>
void Write_Scalar(HighFive::File& file, const std::string& path, const T& value)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<T>(path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

void Write_String_Vector(HighFive::File& file, const std::string& path,
                         const std::vector<std::string>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset = file.createDataSet<std::string>(
        path, HighFive::DataSpace({values.size()}));
    dataset.write(values);
}

void Write_Metadata_Files(const std::filesystem::path& topology_path,
                          const std::filesystem::path& protocol_path,
                          const std::filesystem::path& source_dir)
{
    const auto mass = std::filesystem::absolute(source_dir / "mass.txt");
    const auto charge = std::filesystem::absolute(source_dir / "charge.txt");
    {
        HighFive::File file(topology_path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/schema/version", std::string("1"));
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(2));
        Write_Scalar(file, "/topology/atom_order_hash",
                     std::string("h2_atoms"));
        Write_Scalar(file, "/topology/topology_hash", std::string("h2_top"));
        Write_Scalar(file, "/topology/forcefield_hash", std::string("h2_ff"));
        Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_keys,
                            {"mass_in_file", "charge_in_file"});
        Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_paths,
                            {mass.string(), charge.string()});
    }
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/schema/version", std::string("1"));
        Write_Scalar(file, "/protocol/topology_compatibility/topology_hash",
                     std::string("h2_top"));
        Write_Scalar(file, "/identity/content_hash",
                     std::string("h2_protocol"));
    }
}

SpongeH5OutputPlan::ResolvedOutputPlan Restart_Plan(
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
}

SpongeH5OutputPlan::ResolvedOutputPlan Trajectory_Plan(
    const std::filesystem::path& trajectory_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = trajectory_path.string();
    plan.trajectory.vds = false;
    return plan;
}

void Write_Restart(const std::filesystem::path& restart_path)
{
    const std::vector<float> position = {
        0.0f, 0.0f, -0.37f, 0.0f, 0.0f, 0.37f,
    };
    const std::vector<float> velocity(position.size(), 0.0f);
    const std::array<float, 9> box = {
        40.0f, 0.0f, 0.0f, 0.0f, 40.0f, 0.0f, 0.0f, 0.0f, 40.0f,
    };

    SpongeH5MD::HighFiveBackend backend;
    SpongeH5MD::RestartH5Writer writer(&backend);
    if (!writer.Open(Restart_Plan(restart_path), "1") ||
        !writer.Define_Structural_State(2, true) ||
        !writer.Write_Structural_State(0, 0.0, position.data(), box.data(),
                                       velocity.data()) ||
        !writer.Finalize() || !writer.Close())
    {
        throw std::runtime_error("failed to write restart fixture: " +
                                 writer.Last_Error());
    }
}

void Write_Trajectory(const std::filesystem::path& trajectory_path)
{
    const std::array<float, 9> box = {
        40.0f, 0.0f, 0.0f, 0.0f, 40.0f, 0.0f, 0.0f, 0.0f, 40.0f,
    };
    const std::vector<float> position0 = {
        0.0f, 0.0f, -0.37f, 0.0f, 0.0f, 0.37f,
    };
    const std::vector<float> position1 = {
        0.0f, 0.0f, -0.36f, 0.0f, 0.0f, 0.36f,
    };

    SpongeH5MD::HighFiveBackend backend;
    SpongeH5MD::TrajectoryH5Writer writer(&backend);
    if (!writer.Open_Single_File(Trajectory_Plan(trajectory_path), "1") ||
        !writer.Define_Particle_Datasets(2, false, false) ||
        !writer.Append_Particle_Frame(0, 0.0, position0.data(), box.data()) ||
        !writer.Append_Particle_Frame(1, 0.0, position1.data(), box.data()) ||
        !writer.Finalize() || !writer.Close())
    {
        throw std::runtime_error("failed to write trajectory fixture: " +
                                 writer.Last_Error());
    }
}

void Write_Text(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path.c_str());
    out << text;
    if (!out.good())
    {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void Write_Source_Sidecars(const std::filesystem::path& source_dir)
{
    std::filesystem::create_directories(source_dir);
    Write_Text(source_dir / "mass.txt", "2\n1.008\n1.008\n");
    Write_Text(source_dir / "charge.txt", "2\n0.0\n0.0\n");
}

void Write_Mdin_Files(const std::filesystem::path& output_dir,
                      const std::filesystem::path& source_dir)
{
    const auto mass = std::filesystem::absolute(source_dir / "mass.txt");
    const auto charge = std::filesystem::absolute(source_dir / "charge.txt");

    Write_Text(output_dir / "h5_structural.mdin.toml",
               "md_name = \"h2 h5 structural smoke\"\n"
               "mode = \"nve\"\n"
               "dt = 0\n"
               "step_limit = 0\n"
               "print_zeroth_frame = 1\n"
               "mass_in_file = \"" +
                   mass.string() +
                   "\"\n"
                   "charge_in_file = \"" +
                   charge.string() +
                   "\"\n"
                   "thermostat = \"middle_langevin\"\n"
                   "target_temperature = 300\n"
                   "mdinfo = \"mdinfo.txt\"\n"
                   "mdout = \"mdout.txt\"\n"
                   "write_mdout_interval = 1\n"
                   "write_trajectory_interval = 0\n"
                   "write_restart_file_interval = 1\n"
                   "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
                   "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
                   "input_h5_restart_path = \"restart/restart.spgr.h5\"\n"
                   "input_h5_restart_load = \"structural\"\n"
                   "output_h5_restart_path = \"output/restart_out.spgr.h5\"\n");

    Write_Text(output_dir / "h5_structural_sidecar_materialized.mdin.toml",
               "md_name = \"h2 h5 topology sidecar materialization smoke\"\n"
               "mode = \"nve\"\n"
               "dt = 0\n"
               "step_limit = 0\n"
               "print_zeroth_frame = 1\n"
               "thermostat = \"middle_langevin\"\n"
               "target_temperature = 300\n"
               "mdinfo = \"sidecar_materialized_mdinfo.txt\"\n"
               "mdout = \"sidecar_materialized_mdout.txt\"\n"
               "write_mdout_interval = 1\n"
               "write_trajectory_interval = 0\n"
               "write_restart_file_interval = 1\n"
               "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
               "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
               "input_h5_restart_path = \"restart/restart.spgr.h5\"\n"
               "input_h5_restart_load = \"structural\"\n");

    Write_Text(output_dir / "h5_rerun.mdin.toml",
               "md_name = \"h2 h5 rerun smoke\"\n"
               "mode = \"rerun\"\n"
               "step_limit = 0\n"
               "rerun_start = 0\n"
               "rerun_strip = 0\n"
               "rerun_frame_limit = 1\n"
               "rerun_need_box_update = 0\n"
               "print_zeroth_frame = 1\n"
               "mass_in_file = \"" +
                   mass.string() +
                   "\"\n"
                   "charge_in_file = \"" +
                   charge.string() +
                   "\"\n"
                   "mdinfo = \"rerun_mdinfo.txt\"\n"
                   "mdout = \"rerun_mdout.txt\"\n"
                   "write_mdout_interval = 1\n"
                   "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
                   "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
                   "input_h5_trajectory_path = \"trajectory/prod.spg.h5md\"\n"
                   "input_h5_trajectory_particle_stream = \"all\"\n");

    Write_Text(output_dir / "h5_rerun_eof.mdin.toml",
               "md_name = \"h2 h5 rerun eof smoke\"\n"
               "mode = \"rerun\"\n"
               "step_limit = 0\n"
               "rerun_start = 0\n"
               "rerun_strip = 0\n"
               "rerun_need_box_update = 0\n"
               "print_zeroth_frame = 1\n"
               "mass_in_file = \"" +
                   mass.string() +
                   "\"\n"
                   "charge_in_file = \"" +
                   charge.string() +
                   "\"\n"
                   "mdinfo = \"rerun_eof_mdinfo.txt\"\n"
                   "mdout = \"rerun_eof_mdout.txt\"\n"
                   "write_mdout_interval = 1\n"
                   "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
                   "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
                   "input_h5_trajectory_path = \"trajectory/prod.spg.h5md\"\n"
                   "input_h5_trajectory_particle_stream = \"all\"\n");

    Write_Text(output_dir / "h5_structural_npt.mdin.toml",
               "md_name = \"h2 h5 structural npt smoke\"\n"
               "mode = \"npt\"\n"
               "dt = 0\n"
               "step_limit = 0\n"
               "print_zeroth_frame = 1\n"
               "mass_in_file = \"" +
                   mass.string() +
                   "\"\n"
                   "charge_in_file = \"" +
                   charge.string() +
                   "\"\n"
                   "thermostat = \"middle_langevin\"\n"
                   "target_temperature = 300\n"
                   "barostat = \"monte_carlo_barostat\"\n"
                   "target_pressure = 1\n"
                   "monte_carlo_barostat_update_interval = 100\n"
                   "mdinfo = \"npt_mdinfo.txt\"\n"
                   "mdout = \"npt_mdout.txt\"\n"
                   "write_mdout_interval = 1\n"
                   "write_trajectory_interval = 0\n"
                   "write_restart_file_interval = 1\n"
                   "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
                   "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
                   "input_h5_restart_path = \"restart/restart.spgr.h5\"\n"
                   "input_h5_restart_load = \"structural\"\n");

    Write_Text(output_dir / "h5_structural_restart_output.mdin.toml",
               "md_name = \"h2 h5 restart output smoke\"\n"
               "mode = \"nve\"\n"
               "dt = 0\n"
               "step_limit = 1\n"
               "print_zeroth_frame = 1\n"
               "mass_in_file = \"" +
                   mass.string() +
                   "\"\n"
                   "charge_in_file = \"" +
                   charge.string() +
                   "\"\n"
                   "thermostat = \"middle_langevin\"\n"
                   "target_temperature = 300\n"
                   "mdinfo = \"restart_output_mdinfo.txt\"\n"
                   "mdout = \"restart_output_mdout.txt\"\n"
                   "write_mdout_interval = 1\n"
                   "write_trajectory_interval = 0\n"
                   "write_restart_file_interval = 1\n"
                   "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
                   "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
                   "input_h5_restart_path = \"restart/restart.spgr.h5\"\n"
                   "input_h5_restart_load = \"structural\"\n"
                   "output_h5_restart_path = \"output/restart_out.spgr.h5\"\n");

    Write_Text(
        output_dir / "h5_structural_trajectory_output.mdin.toml",
        "md_name = \"h2 h5 trajectory output smoke\"\n"
        "mode = \"nve\"\n"
        "dt = 0\n"
        "step_limit = 1\n"
        "print_zeroth_frame = 1\n"
        "mass_in_file = \"" +
            mass.string() +
            "\"\n"
            "charge_in_file = \"" +
            charge.string() +
            "\"\n"
            "thermostat = \"middle_langevin\"\n"
            "target_temperature = 300\n"
            "mdinfo = \"trajectory_output_mdinfo.txt\"\n"
            "mdout = \"trajectory_output_mdout.txt\"\n"
            "crd = \"output/legacy_crd.dat\"\n"
            "box = \"output/legacy_box.dat\"\n"
            "write_mdout_interval = 1\n"
            "write_trajectory_interval = 1\n"
            "write_restart_file_interval = 0\n"
            "input_h5_topology_path = \"topology/system.spgt.h5\"\n"
            "input_h5_protocol_path = \"protocol/protocol.spgp.h5\"\n"
            "input_h5_restart_path = \"restart/restart.spgr.h5\"\n"
            "input_h5_restart_load = \"structural\"\n"
            "output_h5_trajectory_path = \"output/traj_out.spg.h5md\"\n"
            "output_h5_observable_path = \"output/obs_out.obs.spg.h5md\"\n");
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr
            << "usage: h5_input_smoke_fixture <output_dir> <source_dir>\n";
        return 2;
    }
    const std::filesystem::path output_dir(argv[1]);
    const std::filesystem::path source_dir(argv[2]);
    std::filesystem::create_directories(output_dir / "topology");
    std::filesystem::create_directories(output_dir / "protocol");
    std::filesystem::create_directories(output_dir / "restart");
    std::filesystem::create_directories(output_dir / "trajectory");
    std::filesystem::create_directories(output_dir / "output");

    try
    {
        Write_Source_Sidecars(source_dir);
        Write_Metadata_Files(output_dir / "topology/system.spgt.h5",
                             output_dir / "protocol/protocol.spgp.h5",
                             source_dir);
        Write_Restart(output_dir / "restart/restart.spgr.h5");
        Write_Trajectory(output_dir / "trajectory/prod.spg.h5md");
        Write_Mdin_Files(output_dir, source_dir);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}
