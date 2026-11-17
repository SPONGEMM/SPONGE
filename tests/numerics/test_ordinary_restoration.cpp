#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../../SPONGE/nb14/ordinary.h"

static void close(double actual, double expected, double tolerance)
{
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual-expected) > tolerance*(1+std::abs(expected)))
        throw std::runtime_error("ordinary restoration mismatch");
}

int main()
{
    int count = 0;
    const double c6 = 0.6*std::pow(3.2, 6), c12 = c6*std::pow(3.2, 6);
    for (double w : {0., 0.3, 1.})
    for (double q : {0., -0.7, 1.2})
    for (double r : {2.7, 7., 9., 14.})
    {
        auto potential = [&](double x) {
            if (x < 8) return w*(c12/std::pow(x,12)-c6/std::pow(x,6)+q/x);
            // Negative of the excluded reciprocal correction.
            return w*q*std::erf(0.3*x)/x;
        };
        double lj, cf, force;
        NB14_Ordinary_Restoration(r*r, 12*w*c12, 6*w*c6, w*q, 8., 0.3, &lj, &cf, &force);
        close(lj+cf, potential(r), 1.e-12);
        const double h = r*1.e-5;
        const double derivative = (potential(r+h)-potential(r-h))/(2*h);
        close(force*r, derivative, 2.e-6);
        close(-force*r*r, -derivative*r, 2.e-6);
        ++count;
    }
    double lj, cf, force;
    NB14_Ordinary_Restoration(0., 0., 0., 0., 8., 0.3, &lj, &cf, &force);
    close(lj+cf+force, 0., 0.);
    std::cout << count << " ordinary restoration E/F/virial states and zero guard passed\n";
}
