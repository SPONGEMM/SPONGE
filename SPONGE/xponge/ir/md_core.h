#pragma once

#include <vector>

namespace Xponge
{

struct Atoms
{
    std::vector<float> mass;
    std::vector<float> charge;
    std::vector<float> coordinate;
    std::vector<float> velocity;
};

struct Box
{
    std::vector<float> box_length;
    std::vector<float> box_angle;
};

struct Residues
{
    std::vector<int> atom_numbers;
};

struct Exclusions
{
    std::vector<std::vector<int>> excluded_atoms;
};

}  // namespace Xponge
