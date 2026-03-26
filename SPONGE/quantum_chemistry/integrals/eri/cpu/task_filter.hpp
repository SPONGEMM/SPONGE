#pragma once

static inline float QC_Effective_Shell_Screen_Tol(const float base_tol,
                                                  const int iter)
{
    if (iter <= 0) return std::max(base_tol, 1.0e-7f);
    if (iter == 1) return std::max(base_tol, 1.0e-8f);
    return base_tol;
}

static inline float QC_Effective_Prim_Screen_Tol(const float base_tol,
                                                 const int iter)
{
    if (iter <= 0) return std::max(base_tol, 1.0e-7f);
    if (iter == 1) return std::max(base_tol, 1.0e-8f);
    return base_tol;
}
