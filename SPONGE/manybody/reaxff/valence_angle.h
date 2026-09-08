#ifndef REAXFF_VALENCE_ANGLE_H
#define REAXFF_VALENCE_ANGLE_H

#include "../../MD_core/MD_core.h"
#include "../../common.h"
#include "../../control.h"
#include "bond_order.h"

struct REAXFF_THBP_Entry
{
    float theta_00;
    float p_val1, p_val2, p_val4, p_val7;
    float p_pen1;
    float p_coa1;
};

struct REAXFF_THBP_Info
{
    int start_idx;
    int entry_count;
};

struct REAXFF_VALENCE_ANGLE_PARAMS
{
    float p_val6, p_val8, p_val9, p_val10;
    float p_pen2, p_pen3, p_pen4;
    float p_coa2, p_coa3, p_coa4;
    float thb_cut, thb_cutsq;
};

struct REAXFF_VALENCE_ANGLE
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Global parameters
    REAXFF_VALENCE_ANGLE_PARAMS params;
    float p_ovun3 = 0.0f;
    float p_ovun4 = 0.0f;
    float p_ovun6 = 0.0f;
    float p_ovun7 = 0.0f;
    float p_ovun8 = 0.0f;

    // Per-atom-type parameters
    float* h_p_val3 = NULL;
    float* h_p_val5 = NULL;
    float* d_p_val3 = NULL;
    float* d_p_val5 = NULL;
    float* h_mass = NULL;
    float* d_mass = NULL;
    float* h_valency_boc = NULL;
    float* d_valency_boc = NULL;

    // 3-body parameters
    REAXFF_THBP_Info* h_thbp_info = NULL;  // Dense matrix [t1][t2][t3]
    REAXFF_THBP_Info* d_thbp_info = NULL;
    REAXFF_THBP_Entry* h_thbp_entries = NULL;  // Flat list of all entries
    REAXFF_THBP_Entry* d_thbp_entries = NULL;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // Energy variables
    float* d_energy_ang_sum = NULL;
    float h_energy_ang = 0.0f;
    float* d_energy_pen_sum = NULL;
    float h_energy_pen = 0.0f;
    float* d_energy_coa_sum = NULL;
    float h_energy_coa = 0.0f;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;
    float* d_CdDelta = NULL;

    float* d_dbo_s_dr = NULL;
    float* d_dbo_pi_dr = NULL;
    float* d_dbo_pi2_dr = NULL;
    float* d_dbo_s_dDelta_i = NULL;
    float* d_dbo_pi_dDelta_i = NULL;
    float* d_dbo_pi2_dDelta_i = NULL;
    float* d_dbo_s_dDelta_j = NULL;
    float* d_dbo_pi_dDelta_j = NULL;
    float* d_dbo_pi2_dDelta_j = NULL;
    float* d_dbo_raw_total_dr = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name);
    void Calculate_Valence_Angle_Energy_And_Force(
        int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* nl,
        REAXFF_BOND_ORDER* bo_module, const float* Delta,
        const float* Delta_boc, const float* Delta_val, const float* nlp,
        const float* vlpex, const float* dDelta_lp, float* CdDelta,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
