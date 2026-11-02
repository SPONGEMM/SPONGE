// CV COMBINATION
#pragma once

#include "../third_party/jit/jit.hpp"
#include "collective_variable.h"

struct CV_COMBINE : public COLLECTIVE_VARIABLE_PROTOTYPE
{
    CV_LIST cv_lists;
    JIT_Function first_step;
    JIT_Function second_step;
    float** d_cv_values;
    VECTOR** cv_crd_grads;
    LTMatrix3** cv_virials;
    float** df_dcv;
    void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
                 const char* module_name);
    void Compute(int atom_numbers, VECTOR* crd, const Boundary boundary,
                 int need, int step);
};
