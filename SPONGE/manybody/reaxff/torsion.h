#ifndef REAXFF_TORSION_H
#define REAXFF_TORSION_H

#include "../../MD_core/MD_core.h"
#include "../../common.h"
#include "../../control.h"
#include "bond_order.h"

struct REAXFF_TORSION_Entry
{
    float p_tor1;
    float V1, V2, V3;
    float p_tor2;
    float p_cot1;
};

struct REAXFF_TORSION_Info
{
    int start_idx;
    int entry_count;
};

struct REAXFF_TORSION
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Global parameters
    float p_tor2 = 0.0f;
    float p_tor3 = 0.0f;
    float p_tor4 = 0.0f;
    float p_cot1 = 0.0f;
    float p_cot2 = 0.0f;
    float thb_cut = 0.0f;

    // Per-atom-type parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // 4-body parameters
    REAXFF_TORSION_Info* h_torsion_info = NULL;  // [t1][t2][t3][t4]
    REAXFF_TORSION_Info* d_torsion_info = NULL;
    REAXFF_TORSION_Entry* h_torsion_entries = NULL;
    REAXFF_TORSION_Entry* d_torsion_entries = NULL;

    // Energy variables
    float* d_energy_tor_sum = NULL;
    float h_energy_tor = 0.0f;
    float* d_energy_cot_sum = NULL;
    float h_energy_cot = 0.0f;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;
    float* d_CdDelta = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name);
    void Calculate_Torsion_Energy_And_Force(
        int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* nl,
        REAXFF_BOND_ORDER* bo_module, const float* Delta_boc,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
