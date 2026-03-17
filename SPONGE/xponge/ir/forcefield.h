#pragma once

#include <vector>

namespace Xponge
{

struct Bonds
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<float> k;
    std::vector<float> r0;
};

struct Angles
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<float> k;
    std::vector<float> theta0;
};

struct Torsions
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<int> atom_d;
    std::vector<float> pk;
    std::vector<float> pn;
    std::vector<int> ipn;
    std::vector<float> gamc;
    std::vector<float> gams;
};

struct NB14
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> cf_scale_factor;
};

struct LennardJones
{
    std::vector<int> atom_type;
    std::vector<float> pair_A;
    std::vector<float> pair_B;
    int atom_type_numbers = 0;
};

struct CMap
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<int> atom_d;
    std::vector<int> atom_e;
    std::vector<int> cmap_type;
    std::vector<int> resolution;
    std::vector<float> grid_value;
    std::vector<float> interpolation_coeff;
    std::vector<int> type_offset;
    int unique_type_numbers = 0;
    int unique_gridpoint_numbers = 0;
};

struct UreyBradley
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<float> angle_k;
    std::vector<float> angle_theta0;
    std::vector<float> bond_k;
    std::vector<float> bond_r0;
};

struct LJSoftCore
{
    int atom_numbers = 0;
    int atom_type_numbers_A = 0;
    int atom_type_numbers_B = 0;
    std::vector<float> LJ_AA;
    std::vector<float> LJ_AB;
    std::vector<float> LJ_BA;
    std::vector<float> LJ_BB;
    std::vector<int> atom_LJ_type_A;
    std::vector<int> atom_LJ_type_B;
    std::vector<int> subsystem_division;
};

struct GeneralizedBorn
{
    std::vector<float> radius;
    std::vector<float> scale_factor;
};

struct VirtualAtomRecord
{
    int type = -1;
    int virtual_atom = -1;
    std::vector<int> from;
    std::vector<float> parameter;
};

struct VirtualAtoms
{
    std::vector<VirtualAtomRecord> records;
};

struct ClassicalForceField
{
    Bonds bonds;
    Angles angles;
    Torsions dihedrals;
    Torsions impropers;
    NB14 nb14;
    LennardJones lj;
    CMap cmap;
    UreyBradley urey_bradley;
    LJSoftCore lj_soft_core;
};

}  // namespace Xponge
