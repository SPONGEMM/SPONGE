#ifndef REAXFF_VDW_H
#define REAXFF_VDW_H

#include "../../common.h"
#include "../../control.h"

struct REAXFF_VDW
{
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;
    float p_vdw1 = 0.0f;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // ReaxFF general parameters
    float* h_general_params = NULL;
    float* d_general_params = NULL;

    // Two-body parameters for VDW
    float* h_twobody_params = NULL;  // indexed by [type_i * ntypes + type_j]
    float* d_twobody_params = NULL;

    float* h_energy_atom = NULL;
    float h_energy_sum = 0;
    float* d_energy_atom = NULL;
    float* d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name = NULL,
                 bool* need_full_nl_flag = NULL);

    void REAXFF_VDW_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* nl, const float cutoff,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
