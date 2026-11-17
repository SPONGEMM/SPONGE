#pragma once

#include <highfive/highfive.hpp>
#include "../../Lennard_Jones_force/independent_sterics_input.hpp"

namespace SpongeH5MD
{
inline IndependentStericsInput Read_Independent_Sterics_H5(
    const HighFive::File& file, std::size_t atom_count)
{
    const std::string root = "/forcefield/lj_soft_core/independent_sterics";
    if (!file.exist(root)) return {};
    if (!file.exist(root + "/atom_branch") || !file.exist(root + "/weights"))
        throw std::runtime_error(root + " requires atom_branch and weights");
    const auto branches = file.getDataSet(root + "/atom_branch");
    const auto weights = file.getDataSet(root + "/weights");
    if (branches.getSpace().getDimensions() != std::vector<std::size_t>{atom_count} ||
        weights.getSpace().getDimensions() != std::vector<std::size_t>{2})
        throw std::runtime_error(root + " requires shapes [atom_count] and [2]");
    if (branches.getDataType().getClass() != HighFive::DataTypeClass::Integer)
        throw std::runtime_error(root + "/atom_branch must have integer dtype");
    IndependentStericsInput input;
    std::vector<float> weight;
    branches.read(input.atom_branch);
    weights.read(weight);
    input.weight_a = weight[0];
    input.weight_b = weight[1];
    input.Validate(atom_count);
    return input;
}
}  // namespace SpongeH5MD
