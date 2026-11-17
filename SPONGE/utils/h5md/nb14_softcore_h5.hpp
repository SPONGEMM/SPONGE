#pragma once
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../float_classification.hpp"

namespace SpongeH5MD
{
inline std::vector<float> Read_NB14_Softcore_Delta(const HighFive::File& file,
                                                   const std::string& root,
                                                   std::size_t count)
{
    const std::string path = root + "/softcore_delta";
    if (!file.exist(path)) return {};
    const auto dataset = file.getDataSet(path);
    if (dataset.getSpace().getDimensions() != std::vector<std::size_t>{count})
        throw std::runtime_error(path + " must have shape [nb14 pair count]");
    std::vector<float> values;
    dataset.read(values);
    for (float value : values)
        if (!SpongeFloat::Is_Finite(value) || value < 0.0f)
            throw std::runtime_error(
                path + " must contain finite nonnegative values in length^6");
    return values;
}
}  // namespace SpongeH5MD
