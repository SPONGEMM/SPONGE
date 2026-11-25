#pragma once

#include <random>
#include <utility>
#include <vector>

#include "protocol.h"

namespace sponge_exchange
{
struct Attempt
{
    std::size_t left;
    std::size_t right;
    double log_acceptance;
    double draw;
    bool accepted;
};

using Pair = std::pair<std::size_t, std::size_t>;
std::vector<Pair> Pair_Labels(std::size_t count, std::size_t round);
double Log_Acceptance(const Report& left, const Report& right);
std::vector<Attempt> Decide(const std::vector<Report>& reports,
                            std::size_t round, std::mt19937_64& random,
                            bool enabled);
}  // namespace sponge_exchange
