#pragma once

static inline float QC_Effective_Shell_Screen_Tol(const float base_tol,
                                                  const int iter)
{
    // Adaptive screening: looser early, floor at 1e-8 for float precision
    if (iter <= 0) return std::max(base_tol, 1.0e-7f);
    return std::max(base_tol, 1.0e-8f);
}

static inline float QC_Effective_Prim_Screen_Tol(const float base_tol,
                                                 const int iter)
{
    if (iter <= 0) return std::max(base_tol, 1.0e-7f);
    return std::max(base_tol, 1.0e-8f);
}
