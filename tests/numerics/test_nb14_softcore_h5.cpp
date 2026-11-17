#include <filesystem>
#include <limits>
#include <iostream>
#include <stdexcept>
#include "../../SPONGE/utils/h5md/topology_native_h5_reader.hpp"
#include "../../SPONGE/xponge/load/native/nb14_extra_h5.hpp"

void require(bool ok, const std::string& message) {if (!ok) throw std::runtime_error(message);}
int main() {
    const auto dir=std::filesystem::temp_directory_path()/"sponge-nb14-softcore-h5";
    std::filesystem::create_directories(dir);
    int checks=0;
    for (const std::string root : {"/forcefield/nb14", "/forcefield/nb14_extra"})
    for (int scenario=0; scenario<8; ++scenario) {
        auto path=dir/("case"+std::to_string(checks)+".h5");
        {
            HighFive::File file(path.string(),HighFive::File::Overwrite);
            file.createGroup("/topology");file.createDataSet<int>("/topology/atom_count",HighFive::DataSpace::From(2)).write(2);
            file.createGroup("/forcefield");file.createGroup(root);
            std::vector<std::vector<int>> atoms={{0,1}};
            std::vector<std::vector<float>> params={{12.,6.,0.5}};
            if (scenario!=7) {
                file.createDataSet<int>(root+"/atoms",HighFive::DataSpace::From(atoms)).write(atoms);
                file.createDataSet<float>(root+"/params",HighFive::DataSpace::From(params)).write(params);
            }
            if (scenario!=0) {
                std::vector<float> delta={scenario==1 ? 0.f : 123.f};
                if (scenario==3) delta={-1.f};
                if (scenario==4) delta={std::numeric_limits<float>::quiet_NaN()};
                if (scenario==5) delta={std::numeric_limits<float>::infinity()};
                if (scenario==6) delta={1.f,2.f};
                file.createDataSet<float>(root+"/softcore_delta",HighFive::DataSpace::From(delta)).write(delta);
            }
        }
        bool ok;
        std::vector<float> delta;
        if (root=="/forcefield/nb14") {
            SpongeH5MD::TopologyNativeH5Reader reader;
            require(reader.Open(path.string()),reader.Last_Error());
            SpongeH5MD::NativeTopologyCoreState state;
            ok=reader.Read_Core_State(&state); delta=state.nb14.softcore_delta;
        } else {
            SpongeH5MD::NativeNB14ExtraH5Reader reader;
            require(reader.Open(path.string()),reader.Last_Error());
            SpongeH5MD::NativeNB14ExtraState state;
            ok=reader.Read(2,&state); delta=state.softcore_delta;
        }
        require(ok==(scenario<3),"wrong validity for "+root+" scenario "+std::to_string(scenario));
        if(scenario==0) require(delta.empty(),"legacy input should retain old path");
        if(scenario==1) require(delta==std::vector<float>{0.f},"explicit zero must be retained");
        if(scenario==2) require(delta==std::vector<float>{123.f},"offset changed on load");
        std::filesystem::remove(path);++checks;
    }
    std::cout<<checks<<" H5 softcore cases passed\n";
}
