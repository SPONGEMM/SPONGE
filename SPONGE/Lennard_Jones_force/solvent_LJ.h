#pragma once

#include "../MD_core/MD_core.h"
#include "../common.h"
#include "../control.h"
#include "LJ_soft_core.h"
#include "Lennard_Jones_force.h"

struct SOLVENT_LENNARD_JONES
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int solvent_numbers = 0;
    int solvent_start = 0;
    int water_points = 0;

    LENNARD_JONES_INFORMATION* lj_info;
    LJ_SOFT_CORE* lj_soft_info;
    VECTOR_LJ* soft_to_hard_crd;
    void Initial(CONTROLLER* controller, LENNARD_JONES_INFORMATION* lj,
                 LJ_SOFT_CORE* lj_soft, MD_INFORMATION* md_info,
                 bool default_enable, const char* module_name = NULL);
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int residue_numbers,
        const int* d_res_start, const VECTOR* crd, const float* charge,
        VECTOR* frc, const Boundary boundary, const ATOM_GROUP* nl,
        const float pme_beta, const int need_atom_energy, float* atom_energy,
        const int need_virial, LTMatrix3* atom_lj_virial,
        float* atom_direct_pme_energy);

    /*
        以下用于区域分解
    */
    int local_solvent_numbers = 0;
    int* d_local_solvent_numbers;
    int solvent_start_local = 0;
    int* d_solvent_start_local;
    void Get_Local(const int res_numbers, const int* d_res_len,
                   const int atom_numbers, float* local_mass);
};