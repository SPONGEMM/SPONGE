#pragma once

#include "dft.hpp"

static inline __host__ __device__ double QC_Exc_Slater(double rho)
{
    if (rho <= 1e-18) return 0.0;
    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    return -Cx * pow(rho, 4.0 / 3.0);
}

static inline __host__ __device__ double QC_Exc_B88(double rho, double sigma)
{
    if (rho <= 1e-18) return 0.0;
    const double grad = sqrt(fmax(0.0, sigma));
    const double rho43 = pow(rho, 4.0 / 3.0);
    const double x = grad / fmax(1e-20, rho43);
    const double beta = 0.0042;
    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    // Spin-unpolarized B88 is built from spin-scaling over rho/2 channels.
    const double c2 = cbrt(2.0);
    const double x_sigma = c2 * x;
    const double corr = beta * c2 * rho43 * x * x /
                        (1.0 + 6.0 * beta * x_sigma * asinh(x_sigma));
    return -Cx * rho43 - corr;
}

static inline __host__ __device__ double QC_Eps_VWN5(double rho)
{
    if (rho <= 1e-18) return 0.0;
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double s = sqrt(rs);
    const double p0 = -0.10498;
    const double p1 = 0.0621814;
    const double p2 = 3.72744;
    const double p3 = 12.9352;
    const double den = (p0 * p0 + p0 * p2 + p3);
    const double sq = sqrt(4.0 * p3 - p2 * p2);
    const double A = p0 * p2 / den - 1.0;
    const double B = 2.0 * A + 2.0;
    const double C = 2.0 * p2 * (1.0 / sq - p0 / (den * sq / (p2 + 2.0 * p0)));
    const double x = s * s + p2 * s + p3;
    const double y = s - p0;
    const double z = sq / (2.0 * s + p2);
    return 0.5 * p1 * (2.0 * log(s) + A * log(x) - B * log(y) + C * atan(z));
}

static inline __host__ __device__ double QC_Ec_VWN5(double rho)
{
    return rho * QC_Eps_VWN5(rho);
}

static inline __host__ __device__ double QC_PW92_Eopt(double sqrt_rs,
                                                      const double t[6])
{
    const double rs = sqrt_rs * sqrt_rs;
    const double poly =
        sqrt_rs * (t[2] + sqrt_rs * (t[3] + sqrt_rs * (t[4] + t[5] * sqrt_rs)));
    const double log_arg = 1.0 + 0.5 / (t[0] * poly);
    const double pref = -2.0 * t[0] * (1.0 + t[1] * rs);
    return pref * log(log_arg);
}

static inline __host__ __device__ double QC_Eps_PW92_Unpol(double rho)
{
    if (rho <= 1e-18) return 0.0;
    static const double p[6] = {0.03109070, 0.21370, 7.59570,
                                3.5876,     1.63820, 0.49294};
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double sqrt_rs = sqrt(rs);
    return QC_PW92_Eopt(sqrt_rs, p);
}

static inline __host__ __device__ double QC_Exc_PBE(double rho, double sigma)
{
    if (rho <= 1e-18) return 0.0;
    const double Cx = 0.75 * cbrt(3.0 / CONSTANT_Pi);
    const double grad = sqrt(fmax(0.0, sigma));
    const double kf = cbrt(3.0 * CONSTANT_Pi * CONSTANT_Pi * rho);
    const double s = grad / fmax(1e-20, 2.0 * kf * rho);
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;
    const double fx = 1.0 + kappa - kappa / (1.0 + mu * s * s / kappa);
    return -Cx * pow(rho, 4.0 / 3.0) * fx;
}

static inline __host__ __device__ double QC_Ec_PBE(double rho, double sigma)
{
    if (rho <= 1e-18) return 0.0;
    const double eps_pw92 = QC_Eps_PW92_Unpol(rho);
    const double gamma = (1.0 - log(2.0)) / (CONSTANT_Pi * CONSTANT_Pi);
    const double beta = 0.06672455060314922;
    const double beta_gamma = beta / gamma;
    const double A = beta_gamma / expm1(-eps_pw92 / gamma);
    const double d2_const = pow(
        (1.0 / 12.0) * pow(3.0, 5.0 / 6.0) * pow(CONSTANT_Pi, 1.0 / 6.0), 2.0);
    const double d2 = d2_const * fmax(0.0, sigma) / pow(rho, 7.0 / 3.0);
    const double d2A = d2 * A;
    const double H = gamma * log(1.0 + beta_gamma * d2 * (1.0 + d2A) /
                                           (1.0 + d2A * (1.0 + d2A)));
    return rho * (eps_pw92 + H);
}

static inline __host__ __device__ double QC_Ec_LYP(double rho, double sigma)
{
    if (rho <= 1e-18) return 0.0;
    const double a = 0.04918;
    const double b = 0.132;
    const double c = 0.2533;
    const double d = 0.349;

    const double rho13 = cbrt(rho);
    const double inv_rho13 = 1.0 / rho13;
    const double den = 1.0 + d * inv_rho13;
    const double den_inv = 1.0 / den;
    const double exp_term = exp(-c * inv_rho13);
    const double t16 = (d * den_inv + c) * inv_rho13;
    const double rho83_inv = pow(rho, -8.0 / 3.0);
    const double grad_coeff = (3.0 + 7.0 * t16) / 72.0;
    const double const_term =
        0.3 * pow(3.0, 2.0 / 3.0) * pow(CONSTANT_Pi, 4.0 / 3.0);
    const double t62 = fmax(0.0, sigma) * rho83_inv * grad_coeff - const_term;
    const double eps = a * (b * exp_term * den_inv * t62 - den_inv);
    return rho * eps;
}

static inline __host__ __device__ double QC_Local_Exc_Density(QC_METHOD method,
                                                              double rho,
                                                              double sigma)
{
    switch (method)
    {
        case QC_METHOD::LDA:
            return QC_Exc_Slater(rho) + QC_Ec_VWN5(rho);
        case QC_METHOD::PBE:
            return QC_Exc_PBE(rho, sigma) + QC_Ec_PBE(rho, sigma);
        case QC_METHOD::BLYP:
            return QC_Exc_B88(rho, sigma) + QC_Ec_LYP(rho, sigma);
        case QC_METHOD::PBE0:
            return 0.75 * QC_Exc_PBE(rho, sigma) + QC_Ec_PBE(rho, sigma);
        case QC_METHOD::B3LYP:
            return 0.08 * QC_Exc_Slater(rho) + 0.72 * QC_Exc_B88(rho, sigma) +
                   0.81 * QC_Ec_LYP(rho, sigma) + 0.19 * QC_Ec_VWN5(rho);
        default:
            return 0.0;
    }
}

static inline __host__ __device__ void QC_Local_Vrho_Vsigma_FD(
    QC_METHOD method, double rho, double sigma, double& e, double& vrho,
    double& vsigma)
{
    rho = fmax(rho, 1e-14);
    sigma = fmax(sigma, 0.0);
    e = QC_Local_Exc_Density(method, rho, sigma);
    // Keep finite-difference perturbations local to the current density region.
    const double dr = fmax(1e-12, 1e-4 * rho);
    const double ds = fmax(1e-14, 1e-4 * (sigma + 1e-12));
    const double rp = rho + dr;
    const double rm = fmax(1e-14, rho - dr);
    const double sp = sigma + ds;
    const double sm = fmax(0.0, sigma - ds);
    const double erp = QC_Local_Exc_Density(method, rp, sigma);
    const double erm = QC_Local_Exc_Density(method, rm, sigma);
    const double esp = QC_Local_Exc_Density(method, rho, sp);
    const double esm = QC_Local_Exc_Density(method, rho, sm);
    vrho = (erp - erm) / (rp - rm);
    vsigma = (esp - esm) / fmax(1e-16, (sp - sm));
}

static inline __host__ __device__ double QC_Clamp_Zeta(double z)
{
    return fmax(-1.0 + 1e-12, fmin(1.0 - 1e-12, z));
}

static inline __host__ __device__ double QC_PW92_Fzeta(double z)
{
    const double zc = QC_Clamp_Zeta(z);
    const double denom = pow(2.0, 4.0 / 3.0) - 2.0;
    const double up = pow(1.0 + zc, 4.0 / 3.0);
    const double dn = pow(1.0 - zc, 4.0 / 3.0);
    return (up + dn - 2.0) / denom;
}

static inline __host__ __device__ double QC_Eps_PW92_Pol(double rho)
{
    if (rho <= 1e-18) return 0.0;
    static const double p[6] = {0.01554535, 0.20548, 14.11890,
                                6.1977,     3.36620, 0.62517};
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    return QC_PW92_Eopt(sqrt(rs), p);
}

static inline __host__ __device__ double QC_Eps_PW92_Alpha(double rho)
{
    if (rho <= 1e-18) return 0.0;
    static const double p[6] = {0.01688690, 0.11125, 10.35700,
                                3.6231,     0.88026, 0.49671};
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    return QC_PW92_Eopt(sqrt(rs), p);
}

static inline __host__ __device__ double QC_Eps_PW92_Spin(double rho,
                                                          double zeta)
{
    if (rho <= 1e-18) return 0.0;
    const double z = QC_Clamp_Zeta(zeta);
    const double fz = QC_PW92_Fzeta(z);
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double ec0 = QC_Eps_PW92_Unpol(rho);
    const double ec1 = QC_Eps_PW92_Pol(rho);
    // PW92 spin interpolation includes the spin-stiffness correction term.
    static constexpr double pw92_fz20 = 1.70992093416136561756;
    const double ec2 = QC_Eps_PW92_Alpha(rho) / pw92_fz20;
    return ec0 + fz * (z4 * (ec1 - ec0) - (1.0 - z4) * ec2);
}

static inline __host__ __device__ double QC_Exc_B88_Spin(double rho_a,
                                                         double rho_b,
                                                         double sigma_aa,
                                                         double sigma_bb)
{
    return 0.5 * QC_Exc_B88(2.0 * rho_a, 4.0 * fmax(0.0, sigma_aa)) +
           0.5 * QC_Exc_B88(2.0 * rho_b, 4.0 * fmax(0.0, sigma_bb));
}

static inline __host__ __device__ double QC_Exc_PBE_Spin(double rho_a,
                                                         double rho_b,
                                                         double sigma_aa,
                                                         double sigma_bb)
{
    return 0.5 * QC_Exc_PBE(2.0 * rho_a, 4.0 * fmax(0.0, sigma_aa)) +
           0.5 * QC_Exc_PBE(2.0 * rho_b, 4.0 * fmax(0.0, sigma_bb));
}

static inline __host__ __device__ double QC_Eps_VWN5_Pol(double rho)
{
    if (rho <= 1e-18) return 0.0;
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double s = sqrt(rs);
    const double den = rs + 7.06042 * s + 18.0578;
    return 0.01554535 * log(rs / den) +
           0.052491393169780936218 *
               atan(4.7309269095601128300 / (2.0 * s + 7.06042)) +
           0.0022478670955426118383 *
               log(((s + 0.32500) * (s + 0.32500)) / den);
}

static inline __host__ __device__ double QC_Eps_VWN5_Alpha(double rho)
{
    if (rho <= 1e-18) return 0.0;
    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double s = sqrt(rs);
    const double den = rs + 1.13107 * s + 13.0045;
    const double inv_pi2 = 1.0 / (CONSTANT_Pi * CONSTANT_Pi);
    return inv_pi2 * (log(rs / den) +
                      0.31770800474394146400 *
                          atan(7.1231089178181179908 / (2.0 * s + 1.13107)) +
                      0.00041403379428206274608 *
                          log(((s + 0.0047584) * (s + 0.0047584)) / den));
}

static inline __host__ __device__ double QC_Ec_VWN5_Spin(double rho_a,
                                                         double rho_b)
{
    const double rho = rho_a + rho_b;
    if (rho <= 1e-18) return 0.0;
    const double z = QC_Clamp_Zeta((rho_a - rho_b) / rho);
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double eps0 = QC_Eps_VWN5(rho);
    const double eps1 = QC_Eps_VWN5_Pol(rho);
    const double eps_alpha = QC_Eps_VWN5_Alpha(rho);
    const double fz_num =
        pow(1.0 + z, 4.0 / 3.0) + pow(1.0 - z, 4.0 / 3.0) - 2.0;
    const double cbrt2 = cbrt(2.0);
    const double eps = eps0 - eps_alpha * fz_num * (1.0 - z4) * (3.0 / 16.0) +
                       (eps1 - eps0) * fz_num * z4 / (2.0 * (cbrt2 - 1.0));
    return rho * eps;
}

static inline __host__ __device__ double QC_Ec_PBE_Spin(double rho_a,
                                                        double rho_b,
                                                        double sigma_aa,
                                                        double sigma_ab,
                                                        double sigma_bb)
{
    const double rho = rho_a + rho_b;
    if (rho <= 1e-18) return 0.0;
    const double zeta = (rho_a - rho_b) / rho;
    const double z = QC_Clamp_Zeta(zeta);
    const double sigma = fmax(0.0, sigma_aa + 2.0 * sigma_ab + sigma_bb);

    const double eps_lsda = QC_Eps_PW92_Spin(rho, z);
    const double phi =
        0.5 * (pow(1.0 + z, 2.0 / 3.0) + pow(1.0 - z, 2.0 / 3.0));
    const double gamma = (1.0 - log(2.0)) / (CONSTANT_Pi * CONSTANT_Pi);
    const double beta = 0.06672455060314922;
    const double beta_gamma = beta / gamma;
    const double ph3 = phi * phi * phi;
    const double A = beta_gamma / expm1(-eps_lsda / fmax(1e-16, gamma * ph3));

    const double kf = cbrt(3.0 * CONSTANT_Pi * CONSTANT_Pi * rho);
    const double ks = sqrt(fmax(1e-20, 4.0 * kf / CONSTANT_Pi));
    const double t = sqrt(sigma) / fmax(1e-20, 2.0 * phi * ks * rho);
    const double t2 = t * t;
    const double At2 = A * t2;
    const double H =
        gamma * ph3 *
        log(1.0 + beta_gamma * t2 * (1.0 + At2) / (1.0 + At2 + At2 * At2));
    return rho * (eps_lsda + H);
}

static inline __host__ __device__ double QC_Ec_LYP_Spin(double rho_a,
                                                        double rho_b,
                                                        double sigma_aa,
                                                        double sigma_ab,
                                                        double sigma_bb)
{
    const double rho = rho_a + rho_b;
    if (rho <= 1e-18) return 0.0;
    const double rho_a_safe = fmax(rho_a, 1e-20);
    const double rho_b_safe = fmax(rho_b, 1e-20);
    const double z = QC_Clamp_Zeta((rho_a - rho_b) / rho);
    const double z2 = z * z;
    const double one_minus_z2 = 1.0 - z2;

    const double sigma_t = fmax(0.0, sigma_aa + 2.0 * sigma_ab + sigma_bb);
    const double sigma_a = fmax(0.0, sigma_aa);
    const double sigma_b = fmax(0.0, sigma_bb);

    const double rho43 = pow(rho, 4.0 / 3.0);
    const double rho_a43 = pow(rho_a_safe, 4.0 / 3.0);
    const double rho_b43 = pow(rho_b_safe, 4.0 / 3.0);
    const double xt = sqrt(sigma_t) / fmax(1e-30, rho43);
    const double xs_a = sqrt(sigma_a) / fmax(1e-30, rho_a43);
    const double xs_b = sqrt(sigma_b) / fmax(1e-30, rho_b43);

    const double A = 0.04918;
    const double B = 0.132;
    const double c = 0.2533;
    const double d = 0.349;

    const double rs = cbrt(3.0 / (4.0 * CONSTANT_Pi * rho));
    const double cbrt2 = cbrt(2.0);
    const double cbrt3 = cbrt(3.0);
    const double cbrt4 = cbrt(4.0);
    const double pref = cbrt3 * cbrt3 * cbrt4 * cbrt(CONSTANT_Pi);

    const double den = 1.0 + d * rs * pref / 3.0;
    const double den_inv = 1.0 / den;
    const double damp = one_minus_z2 * den_inv;
    const double exp_term = exp(-c * rs * pref / 3.0);
    const double t32 = (d * den_inv + c) * rs * pref;
    const double t34 = 47.0 - (7.0 / 3.0) * t32;
    const double t37 = one_minus_z2 * t34 / 72.0 - 2.0 / 3.0;

    const double a = 1.0 + z;
    const double b = 1.0 - z;
    const double a2 = a * a;
    const double b2 = b * b;
    const double a83 = pow(a, 8.0 / 3.0);
    const double b83 = pow(b, 8.0 / 3.0);
    const double ssum = a83 + b83;

    const double xs_a2 = xs_a * xs_a;
    const double xs_b2 = xs_b * xs_b;
    const double t60 = 2.5 - t32 / 54.0;
    const double t62 = xs_a2 * a83;
    const double t64 = xs_b2 * b83;
    const double t66 = t60 * (t62 + t64);
    const double t70 = t32 / 3.0 - 11.0;
    const double t77 = xs_a2 * a83 * a + xs_b2 * b83 * b;
    const double t78 = t70 * t77;
    const double t83 = a2 * xs_b2;
    const double t86 = b2 * xs_a2;
    const double xt2 = xt * xt;
    const double gfac = pow(3.0 * CONSTANT_Pi * CONSTANT_Pi, 2.0 / 3.0);
    const double t92 = -xt2 * t37 - (3.0 / 20.0) * gfac * one_minus_z2 * ssum +
                       cbrt2 * one_minus_z2 * t66 / 32.0 +
                       cbrt2 * one_minus_z2 * t78 / 576.0 -
                       cbrt2 *
                           ((2.0 / 3.0) * t62 + (2.0 / 3.0) * t64 -
                            0.25 * t83 * b83 - 0.25 * t86 * a83) /
                           8.0;
    const double eps = A * (B * exp_term * den_inv * t92 - damp);
    return rho * eps;
}

static inline __host__ __device__ double QC_Local_Exc_Density_UKS(
    QC_METHOD method, double rho_a, double rho_b, double sigma_aa,
    double sigma_ab, double sigma_bb)
{
    const double ex_slater_spin =
        0.5 * QC_Exc_Slater(2.0 * rho_a) + 0.5 * QC_Exc_Slater(2.0 * rho_b);
    switch (method)
    {
        case QC_METHOD::LDA:
            return ex_slater_spin + QC_Ec_VWN5_Spin(rho_a, rho_b);
        case QC_METHOD::PBE:
            return QC_Exc_PBE_Spin(rho_a, rho_b, sigma_aa, sigma_bb) +
                   QC_Ec_PBE_Spin(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb);
        case QC_METHOD::BLYP:
            return QC_Exc_B88_Spin(rho_a, rho_b, sigma_aa, sigma_bb) +
                   QC_Ec_LYP_Spin(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb);
        case QC_METHOD::PBE0:
            return 0.75 * QC_Exc_PBE_Spin(rho_a, rho_b, sigma_aa, sigma_bb) +
                   QC_Ec_PBE_Spin(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb);
        case QC_METHOD::B3LYP:
            return 0.08 * ex_slater_spin +
                   0.72 * QC_Exc_B88_Spin(rho_a, rho_b, sigma_aa, sigma_bb) +
                   0.81 * QC_Ec_LYP_Spin(rho_a, rho_b, sigma_aa, sigma_ab,
                                         sigma_bb) +
                   0.19 * QC_Ec_VWN5_Spin(rho_a, rho_b);
        default:
            return 0.0;
    }
}

static inline __host__ __device__ void QC_Local_UKS_Derivs_FD(
    QC_METHOD method, double rho_a, double rho_b, double sigma_aa,
    double sigma_ab, double sigma_bb, double& e, double& v_rho_a,
    double& v_rho_b, double& v_sigma_aa, double& v_sigma_ab, double& v_sigma_bb)
{
    rho_a = fmax(rho_a, 1e-14);
    rho_b = fmax(rho_b, 1e-14);
    sigma_aa = fmax(sigma_aa, 0.0);
    sigma_bb = fmax(sigma_bb, 0.0);
    e = QC_Local_Exc_Density_UKS(method, rho_a, rho_b, sigma_aa, sigma_ab,
                                 sigma_bb);

    const double dra = fmax(1e-12, 1e-4 * rho_a);
    const double drb = fmax(1e-12, 1e-4 * rho_b);
    const double dsaa = fmax(1e-14, 1e-4 * (sigma_aa + 1e-12));
    const double dsab = fmax(1e-14, 1e-4 * (fabs(sigma_ab) + 1e-12));
    const double dsbb = fmax(1e-14, 1e-4 * (sigma_bb + 1e-12));

    const double rap = rho_a + dra;
    const double ram = fmax(1e-14, rho_a - dra);
    const double rbp = rho_b + drb;
    const double rbm = fmax(1e-14, rho_b - drb);
    const double saap = sigma_aa + dsaa;
    const double saam = fmax(0.0, sigma_aa - dsaa);
    const double sabp = sigma_ab + dsab;
    const double sabm = sigma_ab - dsab;
    const double sbbp = sigma_bb + dsbb;
    const double sbbm = fmax(0.0, sigma_bb - dsbb);

    const double e_rap = QC_Local_Exc_Density_UKS(method, rap, rho_b, sigma_aa,
                                                  sigma_ab, sigma_bb);
    const double e_ram = QC_Local_Exc_Density_UKS(method, ram, rho_b, sigma_aa,
                                                  sigma_ab, sigma_bb);
    const double e_rbp = QC_Local_Exc_Density_UKS(method, rho_a, rbp, sigma_aa,
                                                  sigma_ab, sigma_bb);
    const double e_rbm = QC_Local_Exc_Density_UKS(method, rho_a, rbm, sigma_aa,
                                                  sigma_ab, sigma_bb);
    const double e_saap = QC_Local_Exc_Density_UKS(method, rho_a, rho_b, saap,
                                                   sigma_ab, sigma_bb);
    const double e_saam = QC_Local_Exc_Density_UKS(method, rho_a, rho_b, saam,
                                                   sigma_ab, sigma_bb);
    const double e_sabp = QC_Local_Exc_Density_UKS(method, rho_a, rho_b,
                                                   sigma_aa, sabp, sigma_bb);
    const double e_sabm = QC_Local_Exc_Density_UKS(method, rho_a, rho_b,
                                                   sigma_aa, sabm, sigma_bb);
    const double e_sbbp = QC_Local_Exc_Density_UKS(method, rho_a, rho_b,
                                                   sigma_aa, sigma_ab, sbbp);
    const double e_sbbm = QC_Local_Exc_Density_UKS(method, rho_a, rho_b,
                                                   sigma_aa, sigma_ab, sbbm);

    v_rho_a = (e_rap - e_ram) / (rap - ram);
    v_rho_b = (e_rbp - e_rbm) / (rbp - rbm);
    v_sigma_aa = (e_saap - e_saam) / fmax(1e-16, saap - saam);
    v_sigma_ab = (e_sabp - e_sabm) / fmax(1e-16, sabp - sabm);
    v_sigma_bb = (e_sbbp - e_sbbm) / fmax(1e-16, sbbp - sbbm);
}

#include "xc_deriv.hpp"
