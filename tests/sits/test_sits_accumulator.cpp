#include <cmath>
#include <iostream>
#include <limits>

#include "../../SPONGE/SITS/sits_accumulator.hpp"
#include "../../SPONGE/utils/float_classification.hpp"

bool Check(float actual, double expected, const char* label)
{
    if (!SpongeFloat::Is_Finite(actual) ||
        std::fabs(static_cast<double>(actual) - expected) > 2e-4)
    {
        std::cerr << label << ": got " << actual << ", expected " << expected
                  << "\n";
        return false;
    }
    return true;
}

int main()
{
    bool ok = true;
    // Nonuniform samples, independent per-record offsets, and several reset
    // boundaries. The oracle sums normalized probabilities in linear space.
    const float samples[][3] = {{2, 3, 5}, {7, 2, 1}, {1, 8, 1}, {4, 4, 2}};
    for (int interval : {1, 2, 4})
    {
        for (float shift : {-1000.0f, 0.0f, 1000.0f})
        {
            float accum[3] = {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
            double expected[3] = {0, 0, 0};
            for (int record = 0; record < 12; ++record)
            {
                const bool reset = record % interval == 0;
                const auto& sample = samples[record % 4];
                const float offset = shift + 3.0f * record;
                float gf[3];
                double sum = 0;
                for (int i = 0; i < 3; ++i)
                {
                    gf[i] = std::log(sample[i]) + offset;
                    sum += sample[i];
                    if (reset) expected[i] = 0;
                }
                const float gfsum = SitsAccumulator::Log_Add(
                    SitsAccumulator::Log_Add(gf[0], gf[1]), gf[2]);
                double total = 0;
                for (int i = 0; i < 3; ++i)
                {
                    SitsAccumulator::Accumulate(accum[i], gf[i], gfsum, reset);
                    expected[i] += sample[i] / sum;
                    ok &= Check(accum[i], std::log(expected[i]), "log_pk");
                    total += std::exp(static_cast<double>(accum[i]));
                }
                ok &= Check(static_cast<float>(total), record % interval + 1,
                            "total probability mass");
            }
        }
    }
    // A real uninitialized destination must not be evaluated on reset.
    float fresh;
    SitsAccumulator::Accumulate(fresh, std::log(2.0f), std::log(5.0f), true);
    ok &= Check(fresh, std::log(0.4), "first record");
    return ok ? 0 : 1;
}
