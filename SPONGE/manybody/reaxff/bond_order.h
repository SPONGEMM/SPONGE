#ifndef REAXFF_BOND_ORDER_H
#define REAXFF_BOND_ORDER_H

#include "../../common.h"
#include "../../control.h"

// Device helper for sparse bond index lookup via CSR structure.
// Given an atom and its neighbor, returns the canonical bond index,
// or -1 if no bond exists between them.
static __device__ __forceinline__ int find_bond_index(int atom, int neighbor,
                                                      const int* bond_count,
                                                      const int* bond_offset,
                                                      const int* bond_nbr,
                                                      const int* bond_idx)
{
    int start = bond_offset[atom];
    int end = start + bond_count[atom];
    for (int k = start; k < end; k++)
    {
        if (bond_nbr[k] == neighbor) return bond_idx[k];
    }
    return -1;
}

struct REAXFF_BOND_ORDER
{
    int is_initialized = 0;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // General parameters from ffield
    float gp_boc1 = 0.0f;
    float gp_boc2 = 0.0f;
    float gp_bo_cut = 0.001f;

    // Per-atom-type boc parameters (for computing per-pair p_boc3/4/5)
    float* h_b_o_131 = NULL;  // p_boc4 per atom type
    float* h_b_o_132 = NULL;  // p_boc3 per atom type
    float* h_b_o_133 = NULL;  // p_boc5 per atom type

    // Parameters from ffield
    float* h_ro_sigma = NULL;
    float* h_ro_pi = NULL;
    float* h_ro_pi2 = NULL;
    float* h_bo_1 = NULL;
    float* h_bo_2 = NULL;
    float* h_bo_3 = NULL;
    float* h_bo_4 = NULL;
    float* h_bo_5 = NULL;
    float* h_bo_6 = NULL;
    // Pair-specific bond-order radii (r_s, r_p, r_pp). Initialized to averages
    // and overridden by off-diagonal entries if present in ffield.
    float* h_r_s = NULL;
    float* h_r_p = NULL;
    float* h_r_pp = NULL;

    float* d_ro_sigma = NULL;
    float* d_ro_pi = NULL;
    float* d_ro_pi2 = NULL;
    float* d_bo_1 = NULL;
    float* d_bo_2 = NULL;
    float* d_bo_3 = NULL;
    float* d_bo_4 = NULL;
    float* d_bo_5 = NULL;
    float* d_bo_6 = NULL;
    float* d_r_s = NULL;
    float* d_r_p = NULL;
    float* d_r_pp = NULL;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // 键级修正参数
    float* h_valency = NULL;      // 价态参数
    float* h_valency_val = NULL;  // 价态参数
    float* h_ovc = NULL;          // 过配位修正开关
    float* h_v13cor = NULL;       // 1-3键修正开关
    float* h_p_boc1 = NULL;       // 过配位修正参数1
    float* h_p_boc2 = NULL;       // 过配位修正参数2
    float* h_p_boc3 = NULL;       // 1-3键修正参数3
    float* h_p_boc4 = NULL;       // 1-3键修正参数4
    float* h_p_boc5 = NULL;       // 1-3键修正参数5

    float* d_valency = NULL;
    float* d_valency_val = NULL;
    float* d_ovc = NULL;
    float* d_v13cor = NULL;
    float* d_p_boc1 = NULL;
    float* d_p_boc2 = NULL;
    float* d_p_boc3 = NULL;
    float* d_p_boc4 = NULL;
    float* d_p_boc5 = NULL;

    // --- Sparse bond list ---
    int max_bonds = 0;  // Allocation size for per-bond arrays

    // CSR structure for per-atom bond neighbor lookup
    int* d_bond_count = NULL;   // [N] number of bonds per atom
    int* d_bond_offset = NULL;  // [N+1] prefix sum offsets
    int* d_bond_nbr = NULL;     // [2*max_bonds] neighbor atom per CSR entry
    int* d_bond_idx = NULL;  // [2*max_bonds] canonical bond index per CSR entry
    int* d_fill_count = NULL;  // [N] temp counter for CSR fill phase

    // Per-atom arrays
    float* d_total_bond_order = NULL;            // [N] 每个原子的总键级
    float* d_total_corrected_bond_order = NULL;  // [N]

    // Per-bond sparse arrays (indexed by bond/pair index)
    int* d_pair_i = NULL;            // [max_bonds]
    int* d_pair_j = NULL;            // [max_bonds]
    float* d_pair_distances = NULL;  // [max_bonds]
    int* d_num_pairs_ptr = NULL;     // device counter
    int h_num_pairs = 0;             // host copy

    float* d_corrected_bo_s = NULL;    // [max_bonds] 修正后的sigma键级
    float* d_corrected_bo_pi = NULL;   // [max_bonds] 修正后的pi键级
    float* d_corrected_bo_pi2 = NULL;  // [max_bonds] 修正后的pi-pi键级

    // Derivatives of total energy with respect to bond orders (accumulated from
    // all modules)
    float* d_dE_dBO_s = NULL;    // [max_bonds]
    float* d_dE_dBO_pi = NULL;   // [max_bonds]
    float* d_dE_dBO_pi2 = NULL;  // [max_bonds]

    // Derivatives of corrected bond orders w.r.t r, Delta_i, Delta_j
    float* d_dbo_s_dr = NULL;  // [max_bonds]
    float* d_dbo_pi_dr = NULL;
    float* d_dbo_pi2_dr = NULL;
    float* d_dbo_s_dDelta_i = NULL;  // [max_bonds] d/d(Delta_{pair_i})
    float* d_dbo_pi_dDelta_i = NULL;
    float* d_dbo_pi2_dDelta_i = NULL;
    float* d_dbo_s_dDelta_j = NULL;  // [max_bonds] d/d(Delta_{pair_j})
    float* d_dbo_pi_dDelta_j = NULL;
    float* d_dbo_pi2_dDelta_j = NULL;
    float* d_dbo_raw_total_dr = NULL;  // [max_bonds]
    float* d_CdDelta_prime = NULL;     // [N]

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* parameter_in_file, const char* type_in_file,
                 const float cutoff, float* cutoff_full);
    void Calculate_Bond_Order(int atom_numbers, const VECTOR* d_crd,
                              const LTMatrix3 cell, const LTMatrix3 rcell,
                              const ATOM_GROUP* fnl_d_nl, float cutoff);
    void Calculate_Corrected_Bond_Order(int atom_numbers, const VECTOR* d_crd,
                                        const LTMatrix3 cell,
                                        const LTMatrix3 rcell,
                                        const ATOM_GROUP* fnl_d_nl,
                                        float cutoff);
    void Calculate_Forces(int atom_numbers, const VECTOR* d_crd, VECTOR* d_frc,
                          const LTMatrix3 cell, const LTMatrix3 rcell,
                          float cutoff, float* d_CdDelta, int need_virial,
                          LTMatrix3* atom_virial);
    void Clear_Derivatives(int atom_numbers, float* d_CdDelta);

    // GPU kernel declarations
    void Calculate_Uncorrected_Bond_Orders_GPU(
        int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, float cutoff, const ATOM_GROUP* d_nl,
        int* d_pair_i, int* d_pair_j, float* d_distances, int* d_num_pairs_ptr);
    void Calculate_Corrected_Bond_Orders_GPU(
        int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, float cutoff, int num_pairs, int* d_pair_i,
        int* d_pair_j, float* d_distances);
    void Build_Bond_CSR(int atom_numbers, int num_bonds);
};

#endif
