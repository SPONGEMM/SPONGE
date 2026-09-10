#ifndef STILLINGER_WEBER_FORCE_H
#define STILLINGER_WEBER_FORCE_H
#include "../common.h"
#include "../control.h"

// Stillinger and Weber, Phys Rev B, 31, 5262 (1985).
struct STILLINGER_WEBER_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;
    int pair_type_numbers = 0;
    int triple_type_numbers = 0;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    float *h_parameters, *d_parameters;

    float* h_energy_atom = NULL;
    float h_energy_sum = 0;
    float* d_energy_atom = NULL;
    float* d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, const char* module_name = NULL,
                 bool* need_full_nl_flag = NULL);

    void SW_Force_With_Atom_Energy_And_Virial_Full_NL(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* fnl_d_nl,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};
#endif
