#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include "../../SPONGE/utils/h5md/topology_native_h5_reader.hpp"

int main()
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("sponge-ordinary-h5-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    for (int scenario = 0; scenario < 13; ++scenario)
    {
        const auto path = directory / (std::to_string(scenario) + ".h5");
        const std::string root = "/forcefield/fep_core_ordinary";
        {
            HighFive::File file(path.string(), HighFive::File::Overwrite);
            file.createGroup("/topology");
            file.createDataSet("/topology/atom_count", 3);
            file.createGroup("/forcefield");
            file.createGroup("/forcefield/nb14");
            file.createDataSet("/forcefield/nb14/atoms", std::vector<std::vector<int>>{{0, 2}});
            file.createDataSet("/forcefield/nb14/params", std::vector<std::vector<float>>{{24, 18, .5}});
            if (scenario != 0)
            {
                file.createGroup(root);
                std::vector<std::vector<int>> atoms{{0, 1}};
                if (scenario == 7) atoms[0][1] = 3;
                if (scenario == 8) atoms[0][1] = 0;
                if (scenario != 3)
                {
                    if (scenario == 12)
                        file.createDataSet(root + "/atoms", std::vector<std::vector<float>>{{0, 1}});
                    else file.createDataSet(root + "/atoms", atoms);
                }
                std::vector<std::vector<float>> params{{2, 3, .4}};
                if (scenario == 5) params[0][0] = -1;
                if (scenario == 6) params[0][2] = 1.1;
                if (scenario == 9) params[0][1] = std::numeric_limits<float>::quiet_NaN();
                if (scenario == 10) params.push_back(params[0]);
                if (scenario == 11) params[0].pop_back();
                if (scenario != 4) file.createDataSet(root + "/params", params);
                if (scenario == 2) file.createDataSet(root + "/softcore_delta", std::vector<float>{0});
            }
        }
        SpongeH5MD::TopologyNativeH5Reader reader;
        SpongeH5MD::NativeTopologyCoreState state;
        if (!reader.Open(path.string())) throw std::runtime_error(reader.Last_Error());
        const bool valid = reader.Read_Core_State(&state);
        if (valid != (scenario < 2))
            throw std::runtime_error("ordinary H5 scenario " + std::to_string(scenario) + ": " + reader.Last_Error());
        if (valid && (state.nb14.A != std::vector<float>{24} ||
                      state.nb14.B != std::vector<float>{18}))
            throw std::runtime_error("ordinary input changed native14 parameters");
        if (scenario == 1 && (!state.has_fep_core_ordinary ||
            state.fep_core_ordinary.A != std::vector<float>{24} ||
            state.fep_core_ordinary.B != std::vector<float>{18} ||
            state.fep_core_ordinary.cf_scale_factor != std::vector<float>{.4f}))
            throw std::runtime_error("ordinary physical coefficient conversion mismatch");
        if (scenario == 0 && state.has_fep_core_ordinary)
            throw std::runtime_error("legacy topology enabled ordinary restoration");
    }
    std::filesystem::remove_all(directory);
    std::cout << "13 ordinary restoration H5 cases passed\n";
}
