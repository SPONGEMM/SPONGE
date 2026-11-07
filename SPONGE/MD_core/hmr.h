#pragma once

#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils/float_classification.hpp"

namespace SpongeHMR
{
struct Result
{
    std::vector<float> mass;
    int hydrogens = 0, waters = 0;
};

// Ordinary protium only. Isotopic, pre-repartitioned and coarse-grained
// topologies require explicit atom identities and are outside this interface.
inline bool Hydrogen(float mass) { return mass >= 0.9f && mass <= 1.1f; }

inline Result Repartition(const std::vector<float>& original,
                          const std::vector<std::pair<int, int>>& bonds,
                          const std::vector<int>& virtual_atoms,
                          double ratio_xh1, double ratio_other)
{
    if (!SpongeFloat::Is_Finite(ratio_xh1) || ratio_xh1 < 1 ||
        !SpongeFloat::Is_Finite(ratio_other) || ratio_other < 1)
        throw std::runtime_error("HMR ratios must be finite and >= 1");
    const int n = static_cast<int>(original.size());
    std::vector<bool> is_virtual(n, false), water(n, false);
    for (int i : virtual_atoms)
    {
        if (i < 0 || i >= n)
            throw std::runtime_error("HMR virtual atom index out of range");
        is_virtual[i] = true;
    }
    for (int i = 0; i < n; ++i)
        if (!SpongeFloat::Is_Finite(original[i]) || original[i] < 0 ||
            (original[i] == 0 && !is_virtual[i]))
            throw std::runtime_error("HMR requires positive real-atom masses");
    std::vector<std::vector<int>> adjacent(n);
    for (const auto& edge : bonds)
    {
        const int a = edge.first, b = edge.second;
        if (a < 0 || b < 0 || a >= n || b >= n || a == b)
            throw std::runtime_error("HMR invalid bond indices");
        if (is_virtual[a] || is_virtual[b]) continue;
        adjacent[a].push_back(b);
        adjacent[b].push_back(a);
    }
    for (auto& neighbors : adjacent)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
    }
    Result result{original};
    // Isolated OHH real-atom component, including an optional SETTLE H-H edge.
    // Registered virtual sites do not participate in chemical connectivity.
    for (int o = 0; o < n; ++o)
    {
        if (is_virtual[o] || original[o] < 15.5f || original[o] > 16.5f ||
            adjacent[o].size() != 2)
            continue;
        const int a = adjacent[o][0], b = adjacent[o][1];
        if (!Hydrogen(original[a]) || !Hydrogen(original[b])) continue;
        bool isolated = true;
        for (int h : {a, b})
            for (int neighbor : adjacent[h])
                if (neighbor != o && neighbor != a && neighbor != b)
                    isolated = false;
        if (isolated)
        {
            water[o] = water[a] = water[b] = true;
            ++result.waters;
        }
    }
    std::vector<int> donor(n, -1), count(n, 0);
    for (int h = 0; h < n; ++h)
    {
        if (is_virtual[h] || !Hydrogen(original[h]) || water[h]) continue;
        // Non-water H must have exactly one covalent heavy-atom neighbor.
        if (adjacent[h].size() != 1 || original[adjacent[h][0]] <= 4.0f)
            throw std::runtime_error(
                "HMR hydrogen must have one heavy-atom bond: atom " +
                std::to_string(h));
        donor[h] = adjacent[h][0];
        ++count[donor[h]];
    }
    std::vector<double> removed(n, 0);
    for (int h = 0; h < n; ++h)
    {
        if (donor[h] < 0) continue;
        const double ratio = count[donor[h]] == 1 ? ratio_xh1 : ratio_other;
        result.mass[h] = static_cast<float>(original[h] * ratio);
        if (!SpongeFloat::Is_Finite(result.mass[h]))
            throw std::runtime_error("HMR hydrogen mass overflow");
        removed[donor[h]] += double(result.mass[h]) - original[h];
        ++result.hydrogens;
    }
    for (int i = 0; i < n; ++i)
        if (count[i])
        {
            result.mass[i] = static_cast<float>(original[i] - removed[i]);
            if (!SpongeFloat::Is_Finite(result.mass[i]) || result.mass[i] <= 0)
                throw std::runtime_error("HMR exhausts heavy-atom mass: atom " +
                                         std::to_string(i));
        }
    return result;
}

// Canonical byte order and IEEE float bits make this independent of locale
// and host endianness. This is a compatibility checksum, not a security hash.
inline std::string Mass_Signature(const float* mass, int count)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < count; ++i)
    {
        const auto bits = SpongeFloat::Bits(mass[i]);
        for (int shift = 0; shift < 32; shift += 8)
        {
            hash ^= (bits >> shift) & 255;
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "mass-fnv1a64-v1:" << count << ':' << std::hex << hash;
    return out.str();
}
}  // namespace SpongeHMR
