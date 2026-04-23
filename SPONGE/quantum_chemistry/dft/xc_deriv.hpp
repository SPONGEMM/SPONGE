#pragma once

// XC 泛函解析导数
// 替代有限差分，提供精确的 v_ρ 和 v_σ
// 所有函数签名: (输入) → (exc, vrho, vsigma)
// ε_xc 是 energy per volume (不是 per electron)
// v_ρ = ∂ε_xc/∂ρ, v_σ = ∂ε_xc/∂σ

#include <cmath>

// Slater Exchange
// ε_x = -C_x · ρ^{4/3},  v_ρ = -(4/3)·C_x·ρ^{1/3}
static inline __host__ __device__ void QC_VXC_Slater(double rho, double& exc,
                                                     double& vrho)
{
    exc = vrho = 0.0;
    if (rho <= 1e-18) return;
    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    const double rho13 = cbrt(rho);
    exc = -Cx * rho * rho13;
    vrho = -(4.0 / 3.0) * Cx * rho13;
}

// VWN5 Correlation
// ε_c(rs) via Padé form; v_ρ = ε_c - (rs/3)·dε_c/drs
static inline __host__ __device__ double QC_VWN5_Eps_And_Deps(double rs,
                                                              double& deps_drs)
{
    if (rs <= 0.0)
    {
        deps_drs = 0.0;
        return 0.0;
    }
    const double s = sqrt(rs);
    const double x0 = -0.10498;
    const double A = 0.0621814 / 2.0;  // A = p1/2
    const double b = 3.72744;
    const double c = 12.9352;

    const double X = rs + b * s + c;
    const double X0 = x0 * x0 + b * x0 + c;
    const double Q = sqrt(4.0 * c - b * b);

    const double eps =
        A *
        (log(rs / X) + 2.0 * b / Q * atan(Q / (2.0 * s + b)) -
         (b * x0 / X0) * (log((s - x0) * (s - x0) / X) +
                          2.0 * (b + 2.0 * x0) / Q * atan(Q / (2.0 * s + b))));

    // dε/ds where s = sqrt(rs)
    const double dX_ds = 2.0 * s + b;
    const double t1 = 2.0 / s;     // d(ln rs)/ds = 2/s
    const double t2 = -dX_ds / X;  // d(ln X)/ds
    const double t3 = -2.0 * Q / (Q * Q + (2.0 * s + b) * (2.0 * s + b)) * 2.0;
    // d(atan(Q/(2s+b)))/ds = -2Q/((2s+b)^2+Q^2)

    const double t_log_y = 2.0 / (s - x0);  // d(ln(s-x0)^2)/ds
    const double t_log_X = dX_ds / X;

    const double deps_ds =
        A *
        (t1 + t2 + 2.0 * b / Q * t3 -
         (b * x0 / X0) * (t_log_y - t_log_X + 2.0 * (b + 2.0 * x0) / Q * t3));

    // dε/drs = dε/ds · ds/drs = dε/ds · 1/(2·sqrt(rs))
    deps_drs = deps_ds / (2.0 * s);
    return eps;
}

static inline __host__ __device__ void QC_VXC_VWN5(double rho, double& exc,
                                                   double& vrho)
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

// PBE Exchange
// ε_x = -C_x · ρ^{4/3} · F_x(s),  s = |∇ρ|/(2·k_F·ρ)
// F_x = 1 + κ - κ/(1 + μ·s²/κ)
static inline __host__ __device__ void QC_VXC_PBE_X(double rho, double sigma,
                                                    double& exc, double& vrho,
                                                    double& vsigma)
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

// PW92 Correlation (unpolarized)
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
    const double dpoly_ds =
        t[2] + s * (2.0 * t[3] + s * (3.0 * t[4] + 4.0 * t[5] * s));
    const double dlog_arg_ds = -0.5 * dpoly_ds / (t[0] * poly * poly);
    const double dpref_drs = -2.0 * t[0] * t[1];
    const double ds_drs = 0.5 / s;

    deps_drs =
        dpref_drs * log(log_arg) + pref * (dlog_arg_ds * ds_drs) / log_arg;
    return eps;
}

static inline __host__ __device__ void QC_VXC_PW92_Unpol(double rho,
                                                         double& exc,
                                                         double& vrho)
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

// PBE Correlation
static inline __host__ __device__ void QC_VXC_PBE_C(double rho, double sigma,
                                                    double& exc, double& vrho,
                                                    double& vsigma)
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
    const double d2c = pow(
        (1.0 / 12.0) * pow(3.0, 5.0 / 6.0) * pow(CONSTANT_Pi, 1.0 / 6.0), 2.0);
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
    const double dg_dt2 =
        bg * (num / den + t2 * (dnum_dt2 * den - num * dden_dt2) / (den * den));
    const double dH_dt2 = gamma * dg_dt2 / (1.0 + g);

    // ∂t²/∂σ = d2c / ρ^{7/3}
    vsigma = rho * dH_dt2 * d2c / rho73;

    // ∂(ρ·(eps_pw + H))/∂ρ
    // = eps_pw + H + ρ·∂eps_pw/∂ρ + ρ·∂H/∂ρ
    // ∂H/∂ρ has contributions from: ∂t²/∂ρ and ∂A/∂ρ (through eps_pw)
    const double dt2_drho = -(7.0 / 3.0) * t2 / rho;
    const double dH_from_t2 = dH_dt2 * dt2_drho;

    // ∂A/∂ρ = ∂A/∂eps_pw · ∂eps_pw/∂ρ
    const double dA_deps =
        bg * exp(-eps_pw / gamma) / (gamma * A_denom * A_denom);
    const double deps_pw_drho = deps_pw_drs * (-(rs / (3.0 * rho)));

    // ∂H/∂A
    const double dnum_dA = t2;
    const double dden_dA = t2 + 2.0 * t2 * At2;
    const double dg_dA =
        bg * t2 * (dnum_dA * den - num * dden_dA) / (den * den);
    const double dH_dA = gamma * dg_dA / (1.0 + g);
    const double dH_from_A = dH_dA * dA_deps * deps_pw_drho;

    vrho = eps_pw + H + rho * (deps_pw_drho + dH_from_t2 + dH_from_A);
}

// RKS Dispatch
static inline __host__ __device__ void QC_VXC_Analytical_RKS(
    QC_METHOD method, double rho, double sigma, double& exc, double& vrho,
    double& vsigma)
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
        default:
        {
            // Fallback to FD for unsupported functionals
            rho = fmax(rho, 1e-14);
            sigma = fmax(sigma, 0.0);
            exc = QC_Local_Exc_Density(method, rho, sigma);
            const double dr = fmax(1e-12, 1e-4 * rho);
            const double ds = fmax(1e-14, 1e-4 * (sigma + 1e-12));
            vrho =
                (QC_Local_Exc_Density(method, rho + dr, sigma) -
                 QC_Local_Exc_Density(method, fmax(1e-14, rho - dr), sigma)) /
                (rho + dr - fmax(1e-14, rho - dr));
            vsigma =
                (QC_Local_Exc_Density(method, rho, sigma + ds) -
                 QC_Local_Exc_Density(method, rho, fmax(0.0, sigma - ds))) /
                (sigma + ds - fmax(0.0, sigma - ds));
            break;
        }
    }
}

// PBE Correlation (spin-resolved)
// F = ρ·(ε_lsda(ρ,ζ) + H(ρ,ζ,σ))
static inline __host__ __device__ void QC_VXC_PBE_C_Spin(
    double rho_a, double rho_b, double sigma_aa, double sigma_ab,
    double sigma_bb, double& energy, double& vrho_a, double& vrho_b,
    double& vsigma_aa, double& vsigma_ab, double& vsigma_bb)
{
    const double rho = rho_a + rho_b;
    energy = vrho_a = vrho_b = vsigma_aa = vsigma_ab = vsigma_bb = 0.0;
    if (rho <= 1e-18) return;
    const double zeta = (rho_a - rho_b) / rho;
    const double z = fmax(-1.0 + 1e-12, fmin(1.0 - 1e-12, zeta));
    const double sigma = fmax(0.0, sigma_aa + 2.0 * sigma_ab + sigma_bb);

    // PW92 spin interpolation: eps_lsda and derivatives
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double sqrs = sqrt(rs);

    static const double p0[6] = {0.03109070, 0.21370, 7.59570,
                                 3.5876,     1.63820, 0.49294};
    static const double p1[6] = {0.01554535, 0.20548, 14.11890,
                                 6.1977,     3.36620, 0.62517};
    static const double pa[6] = {0.01688690, 0.11125, 10.35700,
                                 3.6231,     0.88026, 0.49671};

    auto pw92_gd = [&](const double t[6], double& val, double& dval_drho)
    {
        const double s = sqrs;
        const double poly = s * (t[2] + s * (t[3] + s * (t[4] + t[5] * s)));
        const double Q = t[0] * poly;
        const double la = 1.0 + 0.5 / Q;
        const double pf = -2.0 * t[0] * (1.0 + t[1] * rs);
        val = pf * log(la);
        const double dp_ds =
            t[2] + s * (2.0 * t[3] + s * (3.0 * t[4] + 4.0 * t[5] * s));
        const double dQ_ds = t[0] * dp_ds;
        const double dpf_ds = -4.0 * t[0] * t[1] * s;
        const double dG_ds =
            dpf_ds * log(la) + pf * (-dQ_ds / (2.0 * Q * Q)) / la;
        dval_drho = dG_ds * (-s / (6.0 * rho));
    };

    double ec0, dec0;
    pw92_gd(p0, ec0, dec0);
    double ec1, dec1;
    pw92_gd(p1, ec1, dec1);
    double eca, deca;
    pw92_gd(pa, eca, deca);
    static constexpr double fz20 = 1.70992093416136561756;
    const double ec2 = eca / fz20, dec2 = deca / fz20;

    const double opz = 1.0 + z, omz = 1.0 - z;
    const double opz13 = cbrt(opz), omz13 = cbrt(omz);
    const double opz43 = opz * opz13, omz43 = omz * omz13;
    const double fzd = pow(2.0, 4.0 / 3.0) - 2.0;
    const double fz = (opz43 + omz43 - 2.0) / fzd;
    const double fzp = (4.0 / 3.0) * (opz13 - omz13) / fzd;
    const double z2 = z * z, z3 = z2 * z, z4 = z2 * z2;

    const double eps_lsda = ec0 + fz * (z4 * (ec1 - ec0) - (1.0 - z4) * ec2);
    const double deps_drho =
        dec0 + fz * (z4 * (dec1 - dec0) - (1.0 - z4) * dec2);
    const double deps_dz = fzp * (z4 * (ec1 - ec0) - (1.0 - z4) * ec2) +
                           fz * 4.0 * z3 * (ec1 - ec0 + ec2);

    // phi
    const double opz23 = opz13 * opz13, omz23 = omz13 * omz13;
    const double phi = 0.5 * (opz23 + omz23);
    const double dphi_dz =
        (1.0 / 3.0) * (1.0 / fmax(1e-20, opz13) - 1.0 / fmax(1e-20, omz13));

    // PBE H
    const double gamma = (1.0 - log(2.0)) / (CONSTANT_Pi * CONSTANT_Pi);
    const double beta = 0.06672455060314922;
    const double bg = beta / gamma;
    const double ph3 = phi * phi * phi;
    const double w = -eps_lsda / fmax(1e-16, gamma * ph3);
    const double ew = exp(w);
    const double em1 = expm1(w);
    const double A = bg / fmax(1e-30, em1);

    const double kf = cbrt(3.0 * CONSTANT_Pi * CONSTANT_Pi * rho);
    const double ks = sqrt(fmax(1e-20, 4.0 * kf / CONSTANT_Pi));
    const double dt = 2.0 * phi * ks * rho;
    const double dt2v = fmax(1e-40, dt * dt);
    const double t2 = sigma / dt2v;
    const double At2 = A * t2;
    const double Di = 1.0 + At2 + At2 * At2;
    const double fr = bg * t2 * (1.0 + At2) / Di;
    const double H = gamma * ph3 * log(1.0 + fr);
    energy = rho * (eps_lsda + H);

    // A derivatives
    const double Af = A + A * A / bg;
    const double dA_deps = Af / (gamma * ph3);
    const double dA_dphi =
        -3.0 * eps_lsda * Af / (gamma * phi * phi * phi * phi);

    // H derivatives
    const double i1f = 1.0 / (1.0 + fr);
    const double Di2 = Di * Di;
    const double dH_dt2 = gamma * ph3 * bg * (1.0 + 2.0 * A * t2) / Di2 * i1f;
    const double t4 = t2 * t2;
    const double dfr_dA = -bg * A * t4 * (2.0 * t2 + A * t4) / Di2;
    const double dH_dA = gamma * ph3 * dfr_dA * i1f;
    const double dH_dp_dir = (phi > 1e-20) ? 3.0 * H / phi : 0.0;

    // Assemble dH/drho, dH/dz
    const double dt2_drho = -7.0 * t2 / (3.0 * rho);
    const double dH_drho = dH_dt2 * dt2_drho + dH_dA * dA_deps * deps_drho;
    const double dH_dz =
        dphi_dz * (dH_dp_dir - 2.0 * t2 * dH_dt2 / fmax(1e-20, phi) +
                   dH_dA * dA_dphi) +
        dH_dA * dA_deps * deps_dz;

    const double dz_dra = 2.0 * rho_b / (rho * rho);
    const double dz_drb = -2.0 * rho_a / (rho * rho);

    vrho_a = (eps_lsda + H) + rho * (deps_drho + dH_drho) +
             rho * (deps_dz + dH_dz) * dz_dra;
    vrho_b = (eps_lsda + H) + rho * (deps_drho + dH_drho) +
             rho * (deps_dz + dH_dz) * dz_drb;

    const double vs_common = rho * dH_dt2 / dt2v;
    vsigma_aa = vs_common;
    vsigma_ab = 2.0 * vs_common;
    vsigma_bb = vs_common;
}

// UKS 解析导数
// Exchange: spin-separable, ε_x = ½ε_x(2ρα) + ½ε_x(2ρβ)
//   v_ρα = dε_x(2ρα)/d(2ρα) · 2 · ½ = dε_x(2ρα)/d(2ρα)
//   v_σαα = dε_x(2ρα)/d(4σαα) · 4 · ½ = 2·dε_x(2ρα)/d(4σαα)

// UKS Slater exchange
static inline __host__ __device__ void QC_VXC_Slater_Spin(double ra, double rb,
                                                          double& exc,
                                                          double& vra,
                                                          double& vrb)
{
    double ea, va, eb, vb;
    QC_VXC_Slater(2.0 * ra, ea, va);
    QC_VXC_Slater(2.0 * rb, eb, vb);
    exc = 0.5 * ea + 0.5 * eb;
    vra = va;  // dε(2ρα)/d(ρα) = dε(2ρα)/d(2ρα) · 2 · (1/2 from half) = va
    vrb = vb;
}

// UKS PBE exchange (spin-separable)
static inline __host__ __device__ void QC_VXC_PBE_X_Spin(
    double ra, double rb, double saa, double sbb, double& exc, double& vra,
    double& vrb, double& vsaa, double& vsbb)
{
    double ea, va, vsa, eb, vb, vsb;
    QC_VXC_PBE_X(2.0 * ra, 4.0 * fmax(0.0, saa), ea, va, vsa);
    QC_VXC_PBE_X(2.0 * rb, 4.0 * fmax(0.0, sbb), eb, vb, vsb);
    exc = 0.5 * ea + 0.5 * eb;
    vra = va;
    vrb = vb;
    vsaa = 2.0 * vsa;  // chain rule: d(½ε(2ρ,4σ))/dσ = ½·dε/d(4σ)·4 = 2·vsa
    vsbb = 2.0 * vsb;
}

// UKS PBE correlation (使用 spin-resolved PW92 + PBE H)
// 这是最复杂的部分。暂时对 PBE correlation 用 FD
// (exchange 用解析导数已经解决了大部分精度问题)

// UKS dispatch
static inline __host__ __device__ void QC_VXC_Analytical_UKS(
    QC_METHOD method, double ra, double rb, double saa, double sab, double sbb,
    double& exc, double& vra, double& vrb, double& vsaa, double& vsab,
    double& vsbb)
{
    exc = vra = vrb = vsaa = vsab = vsbb = 0.0;
    const double rho = ra + rb;
    if (rho <= 1e-18) return;

    switch (method)
    {
        case QC_METHOD::LDA:
        {
            // Exchange: Slater spin-scaled
            double ex, vxa, vxb;
            QC_VXC_Slater_Spin(ra, rb, ex, vxa, vxb);
            // Correlation: VWN5 spin — 用 FD（解析版太复杂）
            double ec, vca, vcb;
            ra = fmax(ra, 1e-14);
            rb = fmax(rb, 1e-14);
            ec = QC_Ec_VWN5_Spin(ra, rb);
            const double dra = fmax(1e-12, 1e-6 * ra);
            const double drb = fmax(1e-12, 1e-6 * rb);
            vca = (QC_Ec_VWN5_Spin(ra + dra, rb) -
                   QC_Ec_VWN5_Spin(fmax(1e-14, ra - dra), rb)) /
                  (ra + dra - fmax(1e-14, ra - dra));
            vcb = (QC_Ec_VWN5_Spin(ra, rb + drb) -
                   QC_Ec_VWN5_Spin(ra, fmax(1e-14, rb - drb))) /
                  (rb + drb - fmax(1e-14, rb - drb));
            exc = ex + ec;
            vra = vxa + vca;
            vrb = vxb + vcb;
            vsaa = vsab = vsbb = 0.0;
            break;
        }
        case QC_METHOD::PBE:
        {
            // Exchange: PBE spin-scaled (解析)
            double ex, vxa, vxb, vxsaa, vxsbb;
            QC_VXC_PBE_X_Spin(ra, rb, saa, sbb, ex, vxa, vxb, vxsaa, vxsbb);
            // Correlation: PBE spin (解析)
            double ec, vca, vcb, vcsaa, vcsab, vcsbb;
            QC_VXC_PBE_C_Spin(ra, rb, saa, sab, sbb, ec, vca, vcb, vcsaa, vcsab,
                              vcsbb);
            exc = ex + ec;
            vra = vxa + vca;
            vrb = vxb + vcb;
            vsaa = vxsaa + vcsaa;
            vsab = vcsab;
            vsbb = vxsbb + vcsbb;
            break;
        }
        case QC_METHOD::PBE0:
        {
            double ex, vxa, vxb, vxsaa, vxsbb;
            QC_VXC_PBE_X_Spin(ra, rb, saa, sbb, ex, vxa, vxb, vxsaa, vxsbb);
            double ec, vca, vcb, vcsaa, vcsab, vcsbb;
            QC_VXC_PBE_C_Spin(ra, rb, saa, sab, sbb, ec, vca, vcb, vcsaa, vcsab,
                              vcsbb);
            exc = 0.75 * ex + ec;
            vra = 0.75 * vxa + vca;
            vrb = 0.75 * vxb + vcb;
            vsaa = 0.75 * vxsaa + vcsaa;
            vsab = vcsab;
            vsbb = 0.75 * vxsbb + vcsbb;
            break;
        }
        default:
        {
            // Full FD fallback (B3LYP, BLYP etc.)
            // 使用更小的步长
            ra = fmax(ra, 1e-14);
            rb = fmax(rb, 1e-14);
            saa = fmax(saa, 0.0);
            sbb = fmax(sbb, 0.0);
            exc = QC_Local_Exc_Density_UKS(method, ra, rb, saa, sab, sbb);
            const double h = 1e-7;
            const double dra = h * fmax(1., ra), drb = h * fmax(1., rb);
            const double dsaa = h * fmax(1., saa),
                         dsab = h * fmax(1., fabs(sab) + 1e-10),
                         dsbb = h * fmax(1., sbb);
            vra =
                (QC_Local_Exc_Density_UKS(method, ra + dra, rb, saa, sab, sbb) -
                 QC_Local_Exc_Density_UKS(method, fmax(1e-14, ra - dra), rb,
                                          saa, sab, sbb)) /
                (ra + dra - fmax(1e-14, ra - dra));
            vrb =
                (QC_Local_Exc_Density_UKS(method, ra, rb + drb, saa, sab, sbb) -
                 QC_Local_Exc_Density_UKS(method, ra, fmax(1e-14, rb - drb),
                                          saa, sab, sbb)) /
                (rb + drb - fmax(1e-14, rb - drb));
            vsaa = (QC_Local_Exc_Density_UKS(method, ra, rb, saa + dsaa, sab,
                                             sbb) -
                    QC_Local_Exc_Density_UKS(method, ra, rb,
                                             fmax(0., saa - dsaa), sab, sbb)) /
                   (saa + dsaa - fmax(0., saa - dsaa));
            vsab = (QC_Local_Exc_Density_UKS(method, ra, rb, saa, sab + dsab,
                                             sbb) -
                    QC_Local_Exc_Density_UKS(method, ra, rb, saa, sab - dsab,
                                             sbb)) /
                   (2. * dsab);
            vsbb = (QC_Local_Exc_Density_UKS(method, ra, rb, saa, sab,
                                             sbb + dsbb) -
                    QC_Local_Exc_Density_UKS(method, ra, rb, saa, sab,
                                             fmax(0., sbb - dsbb))) /
                   (sbb + dsbb - fmax(0., sbb - dsbb));
            break;
        }
    }
}
