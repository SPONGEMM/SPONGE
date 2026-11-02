#ifndef REAXFF_OVER_UNDER_H
#define REAXFF_OVER_UNDER_H

#include "../../MD_core/MD_core.h"
#include "../../common.h"
#include "../../control.h"
#include "bond_order.h"

struct REAXFF_OVER_UNDER
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Parameters from ffield
    float p_lp1 = 0.0f;
    float p_lp3 = 0.0f;
    float p_ovun3 = 0.0f;
    float p_ovun4 = 0.0f;
    float p_ovun6 = 0.0f;
    float p_ovun7 = 0.0f;
    float p_ovun8 = 0.0f;

    // Per-atom-type parameters (Host)
    float* h_p_lp2 = NULL;
    float* h_p_ovun2 = NULL;
    float* h_p_ovun5 = NULL;
    float* h_valency = NULL;
    float* h_valency_e = NULL;
    float* h_valency_val = NULL;
    float* h_valency_boc = NULL;
    float* h_mass = NULL;

    // Per-atom-type parameters (Device)
    float* d_p_lp2 = NULL;
    float* d_p_ovun2 = NULL;
    float* d_p_ovun5 = NULL;
    float* d_valency = NULL;
    float* d_valency_e = NULL;
    float* d_valency_val = NULL;
    float* d_valency_boc = NULL;
    float* d_mass = NULL;

    // Per-pair parameters (Host)
    float* h_p_ovun1 = NULL;
    float* h_De_s = NULL;

    // Per-pair parameters (Device)
    float* d_p_ovun1 = NULL;
    float* d_De_s = NULL;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // Intermediate variables (Device)
    float* d_Delta = NULL;
    float* d_Delta_boc = NULL;
    float* d_Delta_val = NULL;
    float* d_Delta_lp = NULL;
    float* d_nlp = NULL;
    float* d_vlpex = NULL;
    float* d_Delta_lp_temp = NULL;
    float* d_dDelta_lp = NULL;
    float* d_CdDelta = NULL;  // Coefficient for d(TotalBO)/dx

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

    // Energy variables
    float* d_energy_sum = NULL;
    float h_energy_sum = 0.0f;
    float* d_energy_elp_sum = NULL;
    float h_energy_lp = 0.0f;
    float* d_energy_ovun_sum = NULL;
    float h_energy_ovun = 0.0f;
    float* d_energy_atom = NULL;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    // Initialization
    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name);

    // Calculation
    void Calculate_Over_Under_Energy_And_Force(
        int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary,
        REAXFF_BOND_ORDER* bo_module,  // Access to bond order data
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
    void Step_Print_ELP(CONTROLLER* controller);
};

#endif
