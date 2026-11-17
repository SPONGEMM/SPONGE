#pragma once
#if defined(__linux__)
#include <functional>
#include <vector>

#include "../../manager/protocol.h"

struct MD_INFORMATION;
struct DOMAIN_INFORMATION;
struct CONSTRAIN;
struct SITS_INFORMATION;

namespace sponge_exchange
{
struct Window
{
    Hello configuration;
    std::function<double()> energy;
    std::function<void(std::vector<float>&)> capture;
    std::function<void(const std::vector<float>&, double)> install;
};

Window Make_Window(MD_INFORMATION& md, DOMAIN_INFORMATION& domain,
                   const CONSTRAIN& constraints, const SITS_INFORMATION& sits,
                   std::function<void()> update_box,
                   std::function<void()> refresh_force);
}  // namespace sponge_exchange
#endif
