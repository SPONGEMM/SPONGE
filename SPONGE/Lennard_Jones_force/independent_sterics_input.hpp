#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include "../utils/float_classification.hpp"

// Explicit membership is required even for atoms with zero LJ parameters.
// This immutable input describes one Hamiltonian, not a dynamic TI schedule.
struct IndependentStericsInput
{
    std::vector<int> atom_branch;
    float weight_a = 0;
    float weight_b = 0;

    bool enabled() const { return !atom_branch.empty(); }

    void Validate(std::size_t atom_count) const
    {
        if (atom_count == 0 || atom_branch.size() != atom_count)
            throw std::runtime_error("independent sterics membership must match atom count");
        for (int branch : atom_branch)
            if (branch < 0 || branch > 2)
                throw std::runtime_error("independent sterics membership must be 0, 1 or 2");
        for (float weight : {weight_a, weight_b})
            if (!SpongeFloat::Is_Finite(weight) || weight < 0 || weight > 1)
                throw std::runtime_error("independent sterics weights must be finite and in [0,1]");
    }
};

inline IndependentStericsInput Read_Independent_Sterics_Raw(
    FILE* file, std::size_t atom_count)
{
    IndependentStericsInput input;
    int count = 0;
    if (std::fscanf(file, "%d %f %f", &count, &input.weight_a, &input.weight_b) != 3 ||
        count < 1 || static_cast<std::size_t>(count) != atom_count)
        throw std::runtime_error("independent sterics header must be: atom_count weight_A weight_B");
    input.atom_branch.resize(atom_count);
    for (int i = 0; i < count; ++i)
    {
        int index = -1;
        if (std::fscanf(file, "%d %d", &index, &input.atom_branch[i]) != 2 || index != i)
            throw std::runtime_error("independent sterics rows must be ordered: atom_index branch");
    }
    char trailing;
    if (std::fscanf(file, " %c", &trailing) == 1)
        throw std::runtime_error("independent sterics file has trailing data");
    input.Validate(atom_count);
    return input;
}
