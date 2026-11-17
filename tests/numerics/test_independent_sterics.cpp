#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../SPONGE/Lennard_Jones_force/independent_sterics.h"

static void close(double actual, double expected, double tolerance)
{
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
        throw std::runtime_error("independent sterics numerical mismatch");
}

int main()
{
    const double sigma_a = 3.2, sigma_b = 2.8;
    const double c6a = 0.6 * std::pow(sigma_a, 6);
    const double c6b = 0.8 * std::pow(sigma_b, 6);
    const double c12a = c6a * std::pow(sigma_a, 6);
    const double c12b = c6b * std::pow(sigma_b, 6);
    const double mix = 0.37, alpha = 0.5, power = 2, beta = 0.27;
    int checks = 0;
    for (int i : {0, 1, 2})
        for (int j : {0, 1, 2})
            for (double wa : {0., 0.2, 0.8, 1.})
                for (double wb : {0., 0.3, 0.9, 1.})
                    for (double r : {0.1, 1., 2.7, 6.})
                    {
                        // Electrostatics deliberately nonzero even for unequal
                        // steric weights: this isolates pair mathematics;
                        // runtime validates charge scheduling.
                        const double q = -0.4;
                        auto potential = [&](double x)
                        {
                            if ((i == 1 && j == 2) || (i == 2 && j == 1))
                                return 0.;
                            auto lj = [&](double c12, double c6, double w,
                                          double sigma)
                            {
                                if (w == 0) return 0.;
                                double denominator =
                                    std::pow(x, 6) +
                                    alpha * std::pow(1 - w, power) *
                                        std::pow(sigma, 6);
                                return w * (c12 / (denominator * denominator) -
                                            c6 / denominator);
                            };
                            double energy;
                            if (i == 1 || j == 1)
                                energy = lj(c12a, c6a, wa, sigma_a);
                            else if (i == 2 || j == 2)
                                energy = lj(c12b, c6b, wb, sigma_b);
                            else
                                energy = (1 - mix) * lj(c12a, c6a, 1, sigma_a) +
                                         mix * lj(c12b, c6b, 1, sigma_b);
                            return energy + q * std::erfc(beta * x) / x;
                        };
                        auto value = Evaluate_Independent_Sterics(
                            r * r, i, j, 12 * c12a, 6 * c6a, 12 * c12b, 6 * c6b,
                            mix, wa, wb, alpha, power, std::pow(3., 6), 0., q,
                            beta);
                        close(value.lj_energy + value.direct_energy,
                              potential(r), 1.e-12);
                        double h = r * 1.e-5;
                        double derivative =
                            (potential(r + h) - potential(r - h)) / (2 * h);
                        close(value.force_over_r * r, derivative, 2.e-6);
                        close(-value.force_over_r * r * r, -r * derivative,
                              2.e-6);
                        double tail = 0;
                        if (i == 0 && j == 0)
                            tail = (1 - mix) * c6a + mix * c6b;
                        else if (i != 2 && j != 2)
                            tail = wa * c6a;
                        else if (i != 1 && j != 1)
                            tail = wb * c6b;
                        close(
                            Independent_Sterics_C6(i, j, c6a, c6b, mix, wa, wb),
                            tail, 1.e-12);
                        ++checks;
                    }
    // Zero-LJ atoms still retain their explicitly assigned branch and charge.
    auto charged_zero_lj =
        Evaluate_Independent_Sterics(4., 1, 0, 0., 0., 0., 0., mix, 1., 0.,
                                     alpha, power, 729., 0., 0.8, beta);
    close(charged_zero_lj.lj_energy, 0., 0.);
    close(charged_zero_lj.direct_energy, 0.4 * std::erfc(2 * beta), 1.e-12);
    for (double w : {0., 1.e-8, 0.1, 0.9})
    {
        auto overlap = Evaluate_Independent_Sterics(
            0., 2, 0, 12 * c12a, 6 * c6a, 12 * c12b, 6 * c6b, mix, 1., w, alpha,
            power, 729., 0., 0., beta);
        if (!std::isfinite(overlap.lj_energy))
            throw std::runtime_error("overlap energy");
        close(overlap.force_over_r, 0., 0.);
    }
    auto zero =
        Evaluate_Independent_Sterics(0., 0, 0, 0., 0., 0., 0., mix, 1., 1.,
                                     alpha, power, 729., 0., 0., beta);
    close(zero.lj_energy + zero.direct_energy + zero.force_over_r, 0., 0.);
    std::cout << checks
              << " pair states, tail coefficients and overlap guards passed\n";
}
