#ifndef REAXFF_BOND_H
#define REAXFF_BOND_H

#include "../../common.h"
#include "../../control.h"

struct REAXFF_BOND
{
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // ReaxFF general parameters
    float* h_general_params = NULL;
    float* d_general_params = NULL;

    // Two-body parameters for bonds
    float* h_twobody_params = NULL;  // indexed by [type_i * ntypes + type_j]
    float* d_twobody_params = NULL;

    // Bond order data (sparse per-bond arrays from bond_order module)
    float* d_bo_s = NULL;
    float* d_bo_pi = NULL;
    float* d_bo_pi2 = NULL;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    // CSR bond lookup (set from bond_order module)
    int* d_bond_count = NULL;
    int* d_bond_offset = NULL;
    int* d_bond_nbr = NULL;
    int* d_bond_idx = NULL;

    float* h_energy_atom = NULL;
    float h_energy_sum = 0;
    float* d_energy_atom = NULL;
    float* d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name = NULL,
                 bool* need_full_nl_flag = NULL);

    void REAXFF_Bond_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
