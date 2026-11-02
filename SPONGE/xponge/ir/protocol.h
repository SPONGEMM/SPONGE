#pragma once

#include <string>
#include <vector>

namespace Xponge
{

struct PositionalRestraint
{
    bool present = false;
    std::string name = "default";
    std::vector<int> atom_indices;
    std::vector<float> weight;
    std::vector<float> reference_coordinates;
    bool has_single_weight_default = false;
    float single_weight_default = 0.0f;
    bool has_refcoord_scaling_default = false;
    std::string refcoord_scaling_default;
    bool has_calc_virial_default = false;
    bool calc_virial_default = true;
};

}  // namespace Xponge
