#ifndef REAXFF_HYDROGEN_BOND_H
#define REAXFF_HYDROGEN_BOND_H

#include "../../MD_core/MD_core.h"
#include "../../common.h"
#include "../../control.h"
#include "bond_order.h"

struct REAXFF_HB_Entry
{
    float r0_hb;
    float p_hb1;
    float p_hb2;
    float p_hb3;
};

struct REAXFF_HB_Info
{
    int start_idx;
    int entry_count;
};

struct REAXFF_HYDROGEN_BOND
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Global parameters
    float p_hb1 = 0.0f;
    float p_hb2 = 0.0f;
    float p_hb3 = 0.0f;

    // Per-atom-type parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;
    int* h_is_hydrogen = NULL;
    int* d_is_hydrogen = NULL;

    // HB parameters [acc][don][hyd]
    REAXFF_HB_Info* h_hb_info = NULL;
    REAXFF_HB_Info* d_hb_info = NULL;
    REAXFF_HB_Entry* h_hb_entries = NULL;
    REAXFF_HB_Entry* d_hb_entries = NULL;

    // Energy variables
    float* d_energy_hb_sum = NULL;
    float h_energy_hb = 0.0f;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name);
    void Calculate_HB_Energy_And_Force(
        int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* nl,
        REAXFF_BOND_ORDER* bo_module, const int need_atom_energy,
        float* atom_energy, const int need_virial, LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
