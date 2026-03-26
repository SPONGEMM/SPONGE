/**
 * @file
 * @author
 * - Zhijun Pan
 */

#include "switch_function.h"

#include <cmath>
using namespace std;

SwitchFunction::~SwitchFunction(void) {}

GaussianSF::GaussianSF(const float& td0, const float& tinv_r0,
                       const float& tperiod)
    : d0(td0), inv_r0(tinv_r0), period(tperiod)
{
}
GaussianSF::GaussianSF(const float& td0, const float& tinv_r0)
    : d0(td0), inv_r0(tinv_r0), period(0.0)
{
}
GaussianSF::GaussianSF(void) : d0(0.0), inv_r0(0.625), period(0.0) {}

float GaussianSF::Evaluate(const float& rij, float& df) const
{
    float distant = rij - d0;
    if (period != 0.0)
    {
        distant -= roundf(distant / period) * period;
    }
    const float& dx = distant * inv_r0;
    const float& f = exp(-dx * dx / 2.);
    const float& pre = -dx * inv_r0;

    df = pre * f;
    return f;
}

const float& GaussianSF::GetCenter(void) const { return d0; }
const float& GaussianSF::GetWidth(void) const { return inv_r0; }
RationalSF::RationalSF(const float& td0, const float& tinv_r0, const int& tn,
                       const int& tm)
    : d0(td0), inv_r0(tinv_r0), n(tn), m(tm)
{
}
RationalSF::RationalSF(const float& td0, const float& tinv_r0)
    : d0(td0), inv_r0(tinv_r0)
{
    n = 16;
    m = 32;
}
RationalSF::RationalSF(void)
{
    d0 = 0.0;
    inv_r0 = 0.625;
    n = 16;
    m = 32;
}

float RationalSF::Evaluate(const float& rij, float& df) const
{
    const float& dx = (rij - d0) * inv_r0;
    const float& xn = pow(dx, n);
    const float& xm = pow(dx, m);
    const float& f = (1. - xn) / (1. - xm);

    df = f / (d0 - rij) * (n * xn / (1. - xn) + m * xm / (xm - 1.));
    return f;
}

SmapSF::SmapSF(const float& td0, const float& tinv_r0, const float& ta,
               const float& tb, const float& tc, const float& td)
    : d0(td0), inv_r0(tinv_r0), a(ta), b(tb), c(tc), d(td)
{
}
SmapSF::SmapSF(void) : d0(0.0), inv_r0(0.625), a(1.0), b(1.0), c(1.0), d(-1.0)
{
}

float SmapSF::Evaluate(const float& rij, float& df) const
{
    const float& dx = (rij - d0) * inv_r0;
    const float& sx = c * pow(dx, a);
    const float& f = pow(1.0 + sx, d);
    df = -b * sx / dx * f / (1.0 + sx);
    return f;
}

QSF::QSF(const float& tbeta, const float& lambda, const float& ref)
    : beta(tbeta), r0(lambda * ref)
{
}
QSF::QSF(void) : beta(5.0), r0(1.8) {}

float QSF::Evaluate(const float& rij, float& df) const
{
    const float& dx = beta * (rij - r0);
    const float& xexp = exp(dx);
    const float& f = 1.0 / (1. + xexp);
    df = beta * f * (f - 1.0);

    return f;
}

CosinusSF::CosinusSF(const float& td0, const float& tinv_r0)
    : d0(td0), inv_r0(tinv_r0)
{
}
CosinusSF::CosinusSF(void) : d0(0.0), inv_r0(0.625) {}

float CosinusSF::Evaluate(const float& rij, float& df) const
{
    const float& PI = 3.14159265358979323846;
    const float& dx = (rij - d0) * inv_r0;
    float f;
    df = 0.0;
    if (dx <= 0.0)
    {
        f = 1.;
    }
    else if (dx <= 1.0)
    {
        const float& tmpcos = cos(dx * PI);
        const float& tmpsin = sin(dx * PI);
        f = 0.5 * (tmpcos + 1.0);
        df = -0.5 * PI * tmpsin * inv_r0;
    }
    else
    {
        f = 0.0;
    }

    return f;
}
