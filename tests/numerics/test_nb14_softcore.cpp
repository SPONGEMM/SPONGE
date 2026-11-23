#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../SPONGE/nb14/softcore.h"

void close(double a, double b, double tol)
{
    if (!std::isfinite(a) || !std::isfinite(b) ||
        std::abs(a - b) > tol * (1 + std::abs(b)))
        throw std::runtime_error("nb14 softcore numerical mismatch");
}
int main()
{
    int checks = 0;
    const double sigma = 3.2, epsilon = 0.15;
    const double c6 = 4 * epsilon * std::pow(sigma, 6),
                 c12 = c6 * std::pow(sigma, 6);
    for (double w : {0., 1.e-6, 0.1, 0.5, 0.9, 1.})
        for (double r : {0., 1.e-4, 0.2, 1., 2., 3.2, 6.})
        {
            if (w == 1. && r == 0.)
                continue;  // physical hard endpoint is singular
            const double delta = 0.5 * std::pow(1 - w, 2) * std::pow(sigma, 6);
            double e, f;
            NB14_Softcore_LJ(r * r, 12 * w * c12, 6 * w * c6, delta, &e, &f);
            const double s = std::pow(r, 6) + delta;
            close(e, w * (c12 / (s * s) - c6 / s), 1.e-12);
            if (w == 0.)
            {
                close(e, 0., 0.);
                close(f, 0., 0.);
            }
            if (r > 0.001)
            {
                const double h = r * 1.e-5;
                auto potential = [&](double x)
                {
                    double t = std::pow(x, 6) + delta;
                    return w * (c12 / (t * t) - c6 / t);
                };
                close(f * r, (potential(r + h) - potential(r - h)) / (2 * h),
                      2.e-6);
                close(-f * r * r,
                      -r * (potential(r + h) - potential(r - h)) / (2 * h),
                      2.e-6);
            }
            ++checks;
        }
    double e, f;
    NB14_Softcore_LJ(0., 0., 0., 0., &e, &f);
    close(e, 0., 0.);
    close(f, 0., 0.);
    std::cout << checks << " pair states plus zero/zero endpoint passed\n";
}
