#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include "../../SPONGE/utils/h5md/topology_native_h5_reader.hpp"

static void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("sponge-independent-input-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    int checks = 0;
    for (int scenario = 0; scenario < 14; ++scenario)
    {
        const auto path = directory / (std::to_string(scenario) + ".h5");
        const std::string root = "/forcefield/lj_soft_core";
        {
            HighFive::File file(path.string(), HighFive::File::Overwrite);
            file.createGroup("/topology");
            file.createDataSet("/topology/atom_count", 3);
            file.createGroup("/forcefield");
            file.createGroup(root);
            if (scenario != 13)
            {
                const std::vector<int> types{0, 0, 0};
                for (const std::string suffix : {"/atom_type_A", "/atom_type_B"})
                    file.createDataSet(root + suffix, types);
                for (const std::string suffix : {"/pair_AA", "/pair_AB", "/pair_BA", "/pair_BB"})
                    file.createDataSet(root + suffix, std::vector<float>{1.f});
            }
            if (scenario != 0)
            {
                const auto sub = root + "/independent_sterics";
                file.createGroup(sub);
                std::vector<int> branches{0, 1, 2};
                if (scenario == 3) branches = {0, 1};
                if (scenario == 4) branches[2] = 3;
                if (scenario != 5)
                {
                    if (scenario == 6)
                        file.createDataSet(sub + "/atom_branch", std::vector<float>{0, 1, 2});
                    else file.createDataSet(sub + "/atom_branch", branches);
                }
                std::vector<float> weights{0.8f, 0.9f};
                if (scenario == 2) weights = {0, 1};
                if (scenario == 7) weights[0] = std::numeric_limits<float>::quiet_NaN();
                if (scenario == 8) weights[1] = std::numeric_limits<float>::infinity();
                if (scenario == 9) weights[0] = -0.1f;
                if (scenario == 10) weights[1] = 1.1f;
                if (scenario == 11) weights = {1};
                if (scenario != 12) file.createDataSet(sub + "/weights", weights);
            }
        }
        SpongeH5MD::TopologyNativeH5Reader reader;
        require(reader.Open(path.string()), reader.Last_Error());
        SpongeH5MD::NativeTopologyCoreState state;
        bool valid = reader.Read_Core_State(&state);
        require(valid == (scenario < 3), "H5 scenario " + std::to_string(scenario) + ": " + reader.Last_Error());
        if (scenario == 0) require(!state.lj_soft_core.independent_sterics.enabled(), "legacy path changed");
        if (scenario == 1)
        {
            const auto& input = state.lj_soft_core.independent_sterics;
            require(input.atom_branch == std::vector<int>{0, 1, 2}, "branch membership changed");
            require(input.weight_a == 0.8f && input.weight_b == 0.9f, "weights changed");
        }
        ++checks;
    }
    for (const std::string text : {
        "3 .8 .9\n0 0\n1 1\n2 2\n", "3 nan .9\n0 0\n1 1\n2 2\n",
        "3 .8 .9\n0 0\n2 1\n1 2\n", "3 .8 .9\n0 0\n1 1\n2 -1\n",
        "2 .8 .9\n0 0\n1 1\n", "3 .8 .9\n0 0\n1 1\n2 2\nextra\n"})
    {
        FILE* file = std::tmpfile();
        require(file != nullptr, "tmpfile failed");
        std::fputs(text.c_str(), file);
        std::rewind(file);
        bool valid = true;
        try { auto input = Read_Independent_Sterics_Raw(file, 3); }
        catch (const std::exception&) { valid = false; }
        std::fclose(file);
        require(valid == (checks == 14), "RAW input validity mismatch");
        ++checks;
    }
    std::filesystem::remove_all(directory);
    std::cout << checks << " independent sterics H5/RAW input cases passed\n";
}
