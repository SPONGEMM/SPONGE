#pragma once
#include "../collective_variable/collective_variable.h"

// E = \sum weight * (CV - reference) ^ 2
struct RESTRAIN_CV
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int CV_numbers;
    CV_LIST cv_list;
    float* weight;
    float* reference;
    float* period;
    int* start_step;
    int* max_step;
    int* reduce_step;
    int* stop_step;
    float *h_ene, *d_ene;
    void Initial(CONTROLLER* controller,
                 COLLECTIVE_VARIABLE_CONTROLLER* manager);
    void Restraint(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                   LTMatrix3 rcell, int step, float* d_ene, LTMatrix3* d_virial,
                   VECTOR* frc, int need_potential, int need_pressure);
    void Step_Print(CONTROLLER* controller);
};
