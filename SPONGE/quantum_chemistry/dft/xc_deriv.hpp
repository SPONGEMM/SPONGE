#pragma once

// ===================== XC 泛函解析导数 =====================
// 替代有限差分，提供精确的 v_ρ 和 v_σ
// 所有函数签名: (输入) → (exc, vrho, vsigma)
// ε_xc 是 energy per volume (不是 per electron)
// v_ρ = ∂ε_xc/∂ρ, v_σ = ∂ε_xc/∂σ
// ============================================================

#include <cmath>

// ===================== Slater Exchange =====================
// ε_x = -C_x · ρ^{4/3},  v_ρ = -(4/3)·C_x·ρ^{1/3}
static inline __host__ __device__ void QC_VXC_Slater(
    double rho, double& exc, double& vrho)
{
    exc = vrho = 0.0;
    if (rho <= 1e-18) return;
    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    const double rho13 = cbrt(rho);
    exc = -Cx * rho * rho13;
    vrho = -(4.0 / 3.0) * Cx * rho13;
}

// ===================== VWN5 Correlation =====================
// ε_c(rs) via Padé form; v_ρ = ε_c - (rs/3)·dε_c/drs
static inline __host__ __device__ double QC_VWN5_Eps_And_Deps(
    double rs, double& deps_drs)
{
    if (rs <= 0.0) { deps_drs = 0.0; return 0.0; }
    const double s = sqrt(rs);
    const double x0 = -0.10498;
    const double A = 0.0621814 / 2.0;  // A = p1/2
    const double b = 3.72744;
    const double c = 12.9352;

    const double X = rs + b * s + c;
    const double X0 = x0 * x0 + b * x0 + c;
    const double Q = sqrt(4.0 * c - b * b);

    const double eps = A * (log(rs / X) + 2.0 * b / Q * atan(Q / (2.0 * s + b))
                            - (b * x0 / X0) * (log((s - x0) * (s - x0) / X)
                            + 2.0 * (b + 2.0 * x0) / Q * atan(Q / (2.0 * s + b))));

    // dε/ds where s = sqrt(rs)
    const double dX_ds = 2.0 * s + b;
    const double t1 = 2.0 / s;  // d(ln rs)/ds = 2/s
    const double t2 = -dX_ds / X;  // d(ln X)/ds
    const double t3 = -2.0 * Q / (Q * Q + (2.0 * s + b) * (2.0 * s + b)) * 2.0;
    // d(atan(Q/(2s+b)))/ds = -2Q/((2s+b)^2+Q^2)

    const double t_log_y = 2.0 / (s - x0);  // d(ln(s-x0)^2)/ds
    const double t_log_X = dX_ds / X;

    const double deps_ds = A * (t1 + t2 + 2.0 * b / Q * t3
                            - (b * x0 / X0) * (t_log_y - t_log_X
                            + 2.0 * (b + 2.0 * x0) / Q * t3));

    // dε/drs = dε/ds · ds/drs = dε/ds · 1/(2·sqrt(rs))
    deps_drs = deps_ds / (2.0 * s);
    return eps;
}

static inline __host__ __device__ void QC_VXC_VWN5(
    double rho, double& exc, double& vrho)
{
    exc = vrho = 0.0;
    if (rho <= 1e-18) return;
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    double deps_drs;
    const double eps = QC_VWN5_Eps_And_Deps(rs, deps_drs);
    exc = rho * eps;
    // v_ρ = ∂(ρ·ε)/∂ρ = ε + ρ·dε/dρ = ε + ρ·(dε/drs)·(drs/dρ)
    // drs/dρ = -(1/3)·rs/ρ
    vrho = eps - (rs / 3.0) * deps_drs;
}

// ===================== PBE Exchange =====================
// ε_x = -C_x · ρ^{4/3} · F_x(s),  s = |∇ρ|/(2·k_F·ρ)
// F_x = 1 + κ - κ/(1 + μ·s²/κ)
static inline __host__ __device__ void QC_VXC_PBE_X(
    double rho, double sigma, double& exc, double& vrho, double& vsigma)
{
    exc = vrho = vsigma = 0.0;
    if (rho <= 1e-18) return;

    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;

    const double rho13 = cbrt(rho);
    const double rho43 = rho * rho13;
    const double kf = cbrt(3.0 * CONSTANT_Pi * CONSTANT_Pi * rho);
    const double denom = 2.0 * kf * rho;
    const double s2 = sigma / fmax(1e-30, denom * denom);

    const double p = mu * s2 / kappa;
    const double fx = 1.0 + kappa - kappa / (1.0 + p);
    const double dfx_dp = kappa / ((1.0 + p) * (1.0 + p));
    const double dfx_ds2 = dfx_dp * mu / kappa;

    exc = -Cx * rho43 * fx;

    // v_ρ = ∂ε_xc/∂ρ
    // ε_xc = -Cx·ρ^{4/3}·Fx(s²)
    // s² = σ / (2kf·ρ)²,  kf = (3π²ρ)^{1/3}
    // ds²/dρ = -s² · (8/3)/ρ  (since denom² = 4kf²ρ² ∝ ρ^{8/3})
    const double ds2_drho = -s2 * (8.0 / 3.0) / rho;
    vrho = -Cx * (4.0 / 3.0) * rho13 * fx + (-Cx * rho43) * dfx_ds2 * ds2_drho;

    // v_σ = ∂ε_xc/∂σ = -Cx·ρ^{4/3}·(∂Fx/∂s²)·(∂s²/∂σ)
    // ds²/dσ = 1/denom²
    vsigma = -Cx * rho43 * dfx_ds2 / fmax(1e-30, denom * denom);
}

// ===================== PW92 Correlation (unpolarized) =====================
static inline __host__ __device__ double QC_PW92_Eopt_And_Deriv(
    double sqrt_rs, const double t[6], double& deps_drs)
{
    const double rs = sqrt_rs * sqrt_rs;
    const double s = sqrt_rs;
    const double poly = s * (t[2] + s * (t[3] + s * (t[4] + t[5] * s)));
    const double log_arg = 1.0 + 0.5 / (t[0] * poly);
    const double pref = -2.0 * t[0] * (1.0 + t[1] * rs);
    const double eps = pref * log(log_arg);

    // Derivative: dε/drs
    const double dpoly_ds = t[2] + s * (2.0 * t[3] + s * (3.0 * t[4] + 4.0 * t[5] * s));
    const double dlog_arg_ds = -0.5 * dpoly_ds / (t[0] * poly * poly);
    const double dpref_drs = -2.0 * t[0] * t[1];
    const double ds_drs = 0.5 / s;

    deps_drs = dpref_drs * log(log_arg) + pref * (dlog_arg_ds * ds_drs) / log_arg;
    return eps;
}

static inline __host__ __device__ void QC_VXC_PW92_Unpol(
    double rho, double& exc, double& vrho)
{
    exc = vrho = 0.0;
    if (rho <= 1e-18) return;
    static const double p[6] = {0.03109070, 0.21370, 7.59570,
                                3.5876,     1.63820, 0.49294};
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    double deps_drs;
    const double eps = QC_PW92_Eopt_And_Deriv(sqrt(rs), p, deps_drs);
    exc = rho * eps;
    vrho = eps - (rs / 3.0) * deps_drs;
}

// ===================== PBE Correlation =====================
static inline __host__ __device__ void QC_VXC_PBE_C(
    double rho, double sigma, double& exc, double& vrho, double& vsigma)
{
    exc = vrho = vsigma = 0.0;
    if (rho <= 1e-18) return;

    const double gamma = (1.0 - log(2.0)) / (CONSTANT_Pi * CONSTANT_Pi);
    const double beta = 0.06672455060314922;
    const double bg = beta / gamma;

    // PW92 base
    static const double p[6] = {0.03109070, 0.21370, 7.59570,
                                3.5876,     1.63820, 0.49294};
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    double deps_pw_drs;
    const double eps_pw = QC_PW92_Eopt_And_Deriv(sqrt(rs), p, deps_pw_drs);

    const double A_denom = expm1(-eps_pw / gamma);
    const double A = bg / A_denom;

    // t² from density gradient
    const double d2c = pow((1.0 / 12.0) * pow(3.0, 5.0 / 6.0) *
                           pow(CONSTANT_Pi, 1.0 / 6.0), 2.0);
    const double rho73 = pow(rho, 7.0 / 3.0);
    const double t2 = d2c * fmax(0.0, sigma) / rho73;

    const double At2 = A * t2;
    const double num = 1.0 + At2;
    const double den = 1.0 + At2 + At2 * At2;
    const double H = gamma * log(1.0 + bg * t2 * num / den);

    exc = rho * (eps_pw + H);

    // ∂H/∂t² (holding A constant)
    const double g = bg * t2 * num / den;
    const double dnum_dt2 = A;
    const double dden_dt2 = A + 2.0 * A * At2;
    const double dg_dt2 = bg * (num / den + t2 * (dnum_dt2 * den - num * dden_dt2) / (den * den));
    const double dH_dt2 = gamma * dg_dt2 / (1.0 + g);

    // ∂t²/∂σ = d2c / ρ^{7/3}
    vsigma = rho * dH_dt2 * d2c / rho73;

    // ∂(ρ·(eps_pw + H))/∂ρ
    // = eps_pw + H + ρ·∂eps_pw/∂ρ + ρ·∂H/∂ρ
    // ∂H/∂ρ has contributions from: ∂t²/∂ρ and ∂A/∂ρ (through eps_pw)
    const double dt2_drho = -(7.0 / 3.0) * t2 / rho;
    const double dH_from_t2 = dH_dt2 * dt2_drho;

    // ∂A/∂ρ = ∂A/∂eps_pw · ∂eps_pw/∂ρ
    const double dA_deps = bg * exp(-eps_pw / gamma) / (gamma * A_denom * A_denom);
    const double deps_pw_drho = deps_pw_drs * (-(rs / (3.0 * rho)));

    // ∂H/∂A
    const double dnum_dA = t2;
    const double dden_dA = t2 + 2.0 * t2 * At2;
    const double dg_dA = bg * t2 * (dnum_dA * den - num * dden_dA) / (den * den);
    const double dH_dA = gamma * dg_dA / (1.0 + g);
    const double dH_from_A = dH_dA * dA_deps * deps_pw_drho;

    vrho = eps_pw + H + rho * (deps_pw_drho + dH_from_t2 + dH_from_A);
}

// ===================== RKS Dispatch =====================
static inline __host__ __device__ void QC_VXC_Analytical_RKS(
    QC_METHOD method, double rho, double sigma,
    double& exc, double& vrho, double& vsigma)
{
    exc = vrho = vsigma = 0.0;
    if (rho <= 1e-18) return;

    double e1, v1, e2, v2, vs1 = 0, vs2 = 0;
    switch (method)
    {
        case QC_METHOD::LDA:
            QC_VXC_Slater(rho, e1, v1);
            QC_VXC_VWN5(rho, e2, v2);
            exc = e1 + e2;
            vrho = v1 + v2;
            vsigma = 0.0;
            break;
        case QC_METHOD::PBE:
            QC_VXC_PBE_X(rho, sigma, e1, v1, vs1);
            QC_VXC_PBE_C(rho, sigma, e2, v2, vs2);
            exc = e1 + e2;
            vrho = v1 + v2;
            vsigma = vs1 + vs2;
            break;
        // TODO: B88, LYP, BLYP, B3LYP, PBE0
        default:
        {
            // Fallback to FD for unsupported functionals
            rho = fmax(rho, 1e-14);
            sigma = fmax(sigma, 0.0);
            exc = QC_Local_Exc_Density(method, rho, sigma);
            const double dr = fmax(1e-12, 1e-4 * rho);
            const double ds = fmax(1e-14, 1e-4 * (sigma + 1e-12));
            vrho = (QC_Local_Exc_Density(method, rho + dr, sigma) -
                    QC_Local_Exc_Density(method, fmax(1e-14, rho - dr), sigma)) /
                   (rho + dr - fmax(1e-14, rho - dr));
            vsigma = (QC_Local_Exc_Density(method, rho, sigma + ds) -
                      QC_Local_Exc_Density(method, rho, fmax(0.0, sigma - ds))) /
                     (sigma + ds - fmax(0.0, sigma - ds));
            break;
        }
    }
}
