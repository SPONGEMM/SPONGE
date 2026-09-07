#pragma once

#include <cstdint>

struct CONTROLLER;
struct DOMAIN_INFORMATION;
struct MD_INFORMATION;
struct SETTLE;
struct SHAKE;

// Optional generation of a new initial velocity field.  Initial() runs after
// the step-zero target schedule and final freedom calculation; Finalize() runs
// after domain decomposition has built the local constraint maps.
struct INITIAL_VELOCITY_INFORMATION
{
    bool is_initialized = false;
    bool is_finalized = false;
    std::uint64_t seed = 0;

    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    void Finalize(CONTROLLER* controller, MD_INFORMATION* md_info,
                  DOMAIN_INFORMATION* dd, SETTLE* settle, SHAKE* shake);
};
