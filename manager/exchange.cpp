#include "exchange.h"

#include <cmath>

namespace sponge_exchange
{
std::vector<Pair> Pair_Labels(std::size_t count, std::size_t round)
{
    std::vector<Pair> pairs;
    for (std::size_t left = count == 2 ? 0 : round % 2; left + 1 < count;
         left += 2)
        pairs.emplace_back(left, left + 1);
    return pairs;
}

double Log_Acceptance(const Report& left, const Report& right)
{
    return (left.current - left.partner) + (right.current - right.partner);
}

std::vector<Attempt> Decide(const std::vector<Report>& reports,
                            std::size_t round, std::mt19937_64& random,
                            bool enabled)
{
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Attempt> attempts;
    for (const auto& [left, right] : Pair_Labels(reports.size(), round))
    {
        const double log_acceptance =
            Log_Acceptance(reports[left], reports[right]);
        if (!std::isfinite(log_acceptance))
            throw std::runtime_error("non-finite replica exchange acceptance");
        const double draw = uniform(random);
        const bool accepted = enabled && std::log(draw) < log_acceptance;
        attempts.push_back({left, right, log_acceptance, draw, accepted});
    }
    return attempts;
}
}  // namespace sponge_exchange
