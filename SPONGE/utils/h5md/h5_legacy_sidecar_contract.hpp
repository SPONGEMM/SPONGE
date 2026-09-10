#pragma once

#include <set>
#include <string>

namespace SpongeH5MD
{
struct LegacySidecarBinding
{
    std::string key;
    std::string path;
};

inline const std::set<std::string>& H5_Topology_Sidecar_Command_Keys()
{
    static const std::set<std::string> keys = {
        "pairwise_force_in_file",
        "listed_forces_in_file",
        "mass_in_file",
        "charge_in_file",
        "residue_in_file",
        "exclude_in_file",
        "bond_in_file",
        "angle_in_file",
        "dihedral_in_file",
        "improper_in_file",
        "LJ_in_file",
        "nb14_in_file",
        "nb14_extra_in_file",
        "urey_bradley_in_file",
        "cmap_in_file",
        "LJ_soft_core_in_file",
        "subsys_division_in_file",
        "gb_in_file",
        "virtual_atom_in_file",
        "virtual_atoms_in_file",
        "EAM_in_file",
        "EAM_atom_type_in_file",
        "SW_in_file",
        "EDIP_in_file",
        "TERSOFF_in_file",
        "REAXFF_in_file",
        "REAXFF_type_in_file",
    };
    return keys;
}

inline const std::set<std::string>& H5_Protocol_Sidecar_Command_Keys()
{
    static const std::set<std::string> keys = {
        "cv_in_file",
        "constrain_in_file",
        "restrain_in_file",
        "pairwise_force_in_file",
        "listed_forces_in_file",
        "soft_walls_in_file",
        "SITS_in_file",
        "SITS_atom_in_file",
        "SITS_nk_in_file",
        "restrain_atom_id",
        "restrain_coordinate_in_file",
        "restrain_weight_in_file",
        "meta_edge_in_file",
        "meta_potential_in_file",
        "meta_scatter_in_file",
        "restrain_cv_in_file",
        "steer_cv_in_file",
        "nose_hoover_chain_restart_input",
        "hills_in_file",
    };
    return keys;
}

inline bool Command_Key_Allowed(const std::set<std::string>& allowed_keys,
                                const std::string& key)
{
    return allowed_keys.count(key) != 0;
}
}  // namespace SpongeH5MD
