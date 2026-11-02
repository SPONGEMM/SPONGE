// CV RMSD
#pragma once
#include "collective_variable.h"

struct CV_RMSD : public COLLECTIVE_VARIABLE_PROTOTYPE
{
    int rotated_comparing = 1;
    int atom_numbers = 0;
    int* atom = NULL;
    VECTOR* references = NULL;
    VECTOR* points = NULL;

    void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
                 const char* module_name);
    void Compute(int atom_numbers, VECTOR* crd, const Boundary boundary,
                 int need, int step);

    // private:
    float* covariance_matrix = NULL;
    VECTOR* rotated_ref = NULL;
    float* R = NULL;
    float* U = NULL;
    float* SU = NULL;
    float* VT = NULL;
};
