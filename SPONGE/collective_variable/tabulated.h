// Tabulated CV
#pragma once

#include "collective_variable.h"

struct CV_TABULATED : public COLLECTIVE_VARIABLE_PROTOTYPE
{
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    float* parameters;
    float cv_min;
    float cv_max;
    float delta;
    void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
                 const char* module_name);
    void Compute(int atom_numbers, VECTOR* crd, const Boundary boundary,
                 int need, int step);
};
