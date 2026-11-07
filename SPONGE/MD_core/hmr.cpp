#include "hmr.h"

#include <locale>
#include <set>
#include <sstream>

#include "../xponge/xponge.h"
#include "MD_core.h"

void MD_INFORMATION::Apply_HMR(CONTROLLER* controller)
{
    try
    {
        const std::set<std::string> allowed{"hmr_enabled", "hmr_ratio_xh1",
                                            "hmr_ratio_other"};
        bool present = false;
        for (const auto& command : controller->commands)
            if (command.first.compare(0, 4, "hmr_") == 0)
            {
                present = true;
                if (!allowed.count(command.first))
                    throw std::runtime_error("unknown HMR parameter: " +
                                             command.first);
            }
        if (!present) return;
        if (!controller->Command_Exist("hmr_enabled"))
            throw std::runtime_error("hmr.enabled is required");
        hmr_enabled = controller->Get_Bool("hmr_enabled", "HMR");
        if (!hmr_enabled)
        {
            if (controller->Command_Exist("hmr_ratio_xh1") ||
                controller->Command_Exist("hmr_ratio_other"))
                throw std::runtime_error("HMR ratios require hmr.enabled=true");
            return;
        }
        if (mode != NVE && mode != NVT && mode != NPT)
            throw std::runtime_error("HMR supports NVE, NVT and NPT only");
        auto ratio = [&](const char* key, double fallback)
        {
            if (!controller->Command_Exist(key)) return fallback;
            std::istringstream in(controller->Command(key));
            in.imbue(std::locale::classic());
            double value;
            if (!(in >> value))
                throw std::runtime_error(std::string(key) + " must be numeric");
            in >> std::ws;
            if (!in.eof())
                throw std::runtime_error(std::string(key) +
                                         " has trailing data");
            return value;
        };
        // Force-field family cannot be inferred from an input file format.
        if (!controller->Command_Exist("hmr_ratio_other"))
            throw std::runtime_error(
                "hmr.ratio_other is required (e.g. AMBER 2.5, CHARMM 3.0)");
        const double xh1 = ratio("hmr_ratio_xh1", 2.0);
        const double other = ratio("hmr_ratio_other", 2.5);
        const auto& ff = Xponge::system.classical_force_field;
        std::vector<std::pair<int, int>> bonds;
        for (std::size_t i = 0; i < ff.bonds.atom_a.size(); ++i)
            bonds.emplace_back(ff.bonds.atom_a.at(i), ff.bonds.atom_b.at(i));
        std::vector<int> virtual_atoms;
        for (const auto& site : Xponge::system.virtual_atoms.records)
            virtual_atoms.push_back(site.virtual_atom);
        const auto result = SpongeHMR::Repartition(
            Xponge::system.atoms.mass, bonds, virtual_atoms, xh1, other);
        std::copy(result.mass.begin(), result.mass.end(), h_mass);
        controller->printf(
            "HMR: XH1 ratio %.8g, other ratio %.8g; %d hydrogens "
            "repartitioned, %d waters excluded\n",
            xh1, other, result.hydrogens, result.waters);
    }
    catch (const std::exception& error)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "MD_INFORMATION::Apply_HMR",
                                       error.what());
    }
}
