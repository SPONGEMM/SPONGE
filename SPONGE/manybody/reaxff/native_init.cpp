#include "native_init.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "reaxff.h"
#include "utils/h5md/topology_manybody_h5_materializer.hpp"

namespace
{
using SpongeH5MD::NativeReaxFFDefinition;

template <typename T>
void Host_Copy(T** output, const std::vector<T>& values)
{
    if (values.empty())
    {
        *output = NULL;
        return;
    }
    Malloc_Safely((void**)output, sizeof(T) * values.size());
    std::copy(values.begin(), values.end(), *output);
}

template <typename T>
void Device_Copy(T** output, const T* values, const std::size_t count)
{
    if (count == 0)
    {
        *output = NULL;
        return;
    }
    Device_Malloc_And_Copy_Safely((void**)output, const_cast<T*>(values),
                                  sizeof(T) * count);
}

void Copy_Atom_Types(int** host, int** device,
                     const NativeReaxFFDefinition& definition)
{
    Host_Copy(host, definition.atom_type);
    Device_Copy(device, *host, definition.atom_type.size());
}

void Initialize_EEQ(REAXFF_EEQ* eeq, CONTROLLER* controller,
                    const NativeReaxFFDefinition& definition,
                    const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    eeq->atom_numbers = atom_numbers;
    eeq->atom_type_numbers = ntypes;
    controller->printf("START INITIALIZING REAXFF_EEQ (native H5)\n");
    Malloc_Safely((void**)&eeq->h_chi, sizeof(float) * ntypes);
    Malloc_Safely((void**)&eeq->h_eta, sizeof(float) * ntypes);
    Malloc_Safely((void**)&eeq->h_gamma, sizeof(float) * ntypes);
    Malloc_Safely((void**)&eeq->h_shield, sizeof(float) * ntypes * ntypes);
    for (int type = 0; type < ntypes; ++type)
    {
        const auto& value = definition.atoms[type].values;
        eeq->h_gamma[type] = value[5];
        eeq->h_chi[type] = value[13] * CONSTANT_EV_TO_KCAL_MOL;
        eeq->h_eta[type] = value[14] * CONSTANT_EV_TO_KCAL_MOL * 2.0f;
    }
    for (int i = 0; i < ntypes; ++i)
    {
        for (int j = 0; j < ntypes; ++j)
        {
            eeq->h_shield[i * ntypes + j] =
                powf(eeq->h_gamma[i] * eeq->h_gamma[j], -1.5f);
        }
    }
    Copy_Atom_Types(&eeq->h_atom_type, &eeq->d_atom_type, definition);
    Device_Copy(&eeq->d_chi, eeq->h_chi, ntypes);
    Device_Copy(&eeq->d_eta, eeq->h_eta, ntypes);
    Device_Copy(&eeq->d_gamma, eeq->h_gamma, ntypes);
    Device_Copy(&eeq->d_shield, eeq->h_shield,
                static_cast<std::size_t>(ntypes) * ntypes);

    Device_Malloc_Safely((void**)&eeq->d_b, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_r, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_p, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_Ap, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_q, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_s, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_t, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_z, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_temp_sum, sizeof(float));
    Malloc_Safely((void**)&eeq->h_h_numnbrs, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&eeq->h_h_firstnbrs, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_h_numnbrs, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_h_firstnbrs,
                         sizeof(int) * atom_numbers);
    deviceMemset(eeq->d_q, 0, sizeof(float) * atom_numbers);
    deviceMemset(eeq->d_s, 0, sizeof(float) * atom_numbers);
    deviceMemset(eeq->d_t, 0, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_rr_old, sizeof(float));
    Device_Malloc_Safely((void**)&eeq->d_rr_new, sizeof(float));
    Device_Malloc_Safely((void**)&eeq->d_pAp_buf, sizeof(float));
    Device_Malloc_Safely((void**)&eeq->d_cg_alpha, sizeof(float));
    Device_Malloc_Safely((void**)&eeq->d_cg_beta, sizeof(float));
    Device_Malloc_Safely((void**)&eeq->d_s_hist,
                         sizeof(float) * REAXFF_EEQ::HIST_SIZE * atom_numbers);
    Device_Malloc_Safely((void**)&eeq->d_t_hist,
                         sizeof(float) * REAXFF_EEQ::HIST_SIZE * atom_numbers);
    eeq->nprev = 0;
    eeq->is_initialized = 1;
    controller->Step_Print_Initial("REAXFF_EEQ", "%14.7e");
    controller->printf("END INITIALIZING REAXFF_EEQ\n\n");
}

void Initialize_Bond_Order(REAXFF_BOND_ORDER* bo, CONTROLLER* controller,
                           const NativeReaxFFDefinition& definition,
                           const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    const std::size_t npairs = static_cast<std::size_t>(ntypes) * ntypes;
    bo->atom_numbers = atom_numbers;
    bo->atom_type_numbers = ntypes;
    bo->gp_boc1 = definition.general[0];
    bo->gp_boc2 = definition.general[1];
    bo->gp_bo_cut = 0.01f * definition.general[29];
    controller->printf("START INITIALIZING REAXFF_BOND_ORDER (native H5)\n");

    const std::vector<float**> atom_arrays = {
        &bo->h_ro_sigma,    &bo->h_ro_pi,   &bo->h_ro_pi2,  &bo->h_valency,
        &bo->h_valency_val, &bo->h_b_o_131, &bo->h_b_o_132, &bo->h_b_o_133};
    for (float** pointer : atom_arrays)
    {
        Malloc_Safely((void**)pointer, sizeof(float) * ntypes);
    }
    for (int type = 0; type < ntypes; ++type)
    {
        const auto& value = definition.atoms[type].values;
        bo->h_ro_sigma[type] = value[0];
        bo->h_valency[type] = value[1];
        bo->h_ro_pi[type] = value[6];
        bo->h_ro_pi2[type] = value[16];
        bo->h_b_o_131[type] = value[19];
        bo->h_b_o_132[type] = value[20];
        bo->h_b_o_133[type] = value[21];
        bo->h_valency_val[type] = value[27];
    }

    const std::vector<float**> pair_arrays = {
        &bo->h_bo_1,   &bo->h_bo_2, &bo->h_bo_3,   &bo->h_bo_4,   &bo->h_bo_5,
        &bo->h_bo_6,   &bo->h_ovc,  &bo->h_v13cor, &bo->h_p_boc3, &bo->h_p_boc4,
        &bo->h_p_boc5, &bo->h_r_s,  &bo->h_r_p,    &bo->h_r_pp};
    for (float** pointer : pair_arrays)
    {
        Malloc_Safely((void**)pointer, sizeof(float) * npairs);
        memset(*pointer, 0, sizeof(float) * npairs);
    }
    for (int i = 0; i < ntypes; ++i)
    {
        for (int j = 0; j < ntypes; ++j)
        {
            const int index = i * ntypes + j;
            bo->h_p_boc3[index] = sqrtf(bo->h_b_o_132[i] * bo->h_b_o_132[j]);
            bo->h_p_boc4[index] = sqrtf(bo->h_b_o_131[i] * bo->h_b_o_131[j]);
            bo->h_p_boc5[index] = sqrtf(bo->h_b_o_133[i] * bo->h_b_o_133[j]);
            bo->h_r_s[index] = 0.5f * (bo->h_ro_sigma[i] + bo->h_ro_sigma[j]);
            bo->h_r_p[index] = 0.5f * (bo->h_ro_pi[i] + bo->h_ro_pi[j]);
            bo->h_r_pp[index] = 0.5f * (bo->h_ro_pi2[i] + bo->h_ro_pi2[j]);
        }
    }
    for (const auto& bond : definition.bonds)
    {
        const int a = bond.type[0];
        const int b = bond.type[1];
        for (const int index : {a * ntypes + b, b * ntypes + a})
        {
            bo->h_bo_1[index] = bond.values[12];
            bo->h_bo_2[index] = bond.values[13];
            bo->h_bo_3[index] = bond.values[9];
            bo->h_bo_4[index] = bond.values[10];
            bo->h_bo_5[index] = bond.values[4];
            bo->h_bo_6[index] = bond.values[6];
            bo->h_ovc[index] = bond.values[14];
            bo->h_v13cor[index] = bond.values[5];
        }
    }
    for (const auto& row : definition.off_diagonal)
    {
        const int a = row.type[0];
        const int b = row.type[1];
        for (const int index : {a * ntypes + b, b * ntypes + a})
        {
            if (row.values[3] > 0.0f) bo->h_r_s[index] = row.values[3];
            if (row.values[4] > 0.0f) bo->h_r_p[index] = row.values[4];
            if (row.values[5] > 0.0f) bo->h_r_pp[index] = row.values[5];
        }
    }
    Copy_Atom_Types(&bo->h_atom_type, &bo->d_atom_type, definition);
#define COPY_BO_ARRAY(name, count) \
    Device_Copy(&bo->d_##name, bo->h_##name, (count))
    COPY_BO_ARRAY(ro_sigma, ntypes);
    COPY_BO_ARRAY(ro_pi, ntypes);
    COPY_BO_ARRAY(ro_pi2, ntypes);
    COPY_BO_ARRAY(bo_1, npairs);
    COPY_BO_ARRAY(bo_2, npairs);
    COPY_BO_ARRAY(bo_3, npairs);
    COPY_BO_ARRAY(bo_4, npairs);
    COPY_BO_ARRAY(bo_5, npairs);
    COPY_BO_ARRAY(bo_6, npairs);
    COPY_BO_ARRAY(r_s, npairs);
    COPY_BO_ARRAY(r_p, npairs);
    COPY_BO_ARRAY(r_pp, npairs);
    COPY_BO_ARRAY(valency, ntypes);
    COPY_BO_ARRAY(valency_val, ntypes);
    COPY_BO_ARRAY(ovc, npairs);
    COPY_BO_ARRAY(v13cor, npairs);
    COPY_BO_ARRAY(p_boc3, npairs);
    COPY_BO_ARRAY(p_boc4, npairs);
    COPY_BO_ARRAY(p_boc5, npairs);
#undef COPY_BO_ARRAY

    bo->max_bonds = atom_numbers * 32;
    Device_Malloc_Safely((void**)&bo->d_total_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&bo->d_total_corrected_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&bo->d_CdDelta_prime,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&bo->d_bond_count, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&bo->d_bond_offset,
                         sizeof(int) * (atom_numbers + 1));
    Device_Malloc_Safely((void**)&bo->d_bond_nbr,
                         sizeof(int) * 2 * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_bond_idx,
                         sizeof(int) * 2 * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_fill_count, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&bo->d_pair_i, sizeof(int) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_pair_j, sizeof(int) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_pair_distances,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_num_pairs_ptr, sizeof(int));
    Device_Malloc_Safely((void**)&bo->d_corrected_bo_s,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_corrected_bo_pi,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_corrected_bo_pi2,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dE_dBO_s,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dE_dBO_pi,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dE_dBO_pi2,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_s_dr,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi_dr,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi2_dr,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_s_dDelta_i,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi_dDelta_i,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi2_dDelta_i,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_s_dDelta_j,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi_dDelta_j,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_pi2_dDelta_j,
                         sizeof(float) * bo->max_bonds);
    Device_Malloc_Safely((void**)&bo->d_dbo_raw_total_dr,
                         sizeof(float) * bo->max_bonds);
    bo->is_initialized = 1;
    controller->printf("  Sparse bond storage: max_bonds = %d\n",
                       bo->max_bonds);
    controller->printf("END INITIALIZING REAXFF_BOND_ORDER\n\n");
}

void Initialize_Bond(REAXFF_BOND* bond, CONTROLLER* controller,
                     const NativeReaxFFDefinition& definition,
                     const int atom_numbers, bool* need_full_nl_flag)
{
    constexpr int stride = 5;
    const int ntypes = static_cast<int>(definition.atoms.size());
    const std::size_t count =
        static_cast<std::size_t>(ntypes) * ntypes * stride;
    bond->atom_numbers = atom_numbers;
    bond->atom_type_numbers = ntypes;
    controller->printf("START INITIALIZING REAXFF BOND FORCE (native H5)\n");
    Malloc_Safely((void**)&bond->h_twobody_params, sizeof(float) * count);
    memset(bond->h_twobody_params, 0, sizeof(float) * count);
    for (const auto& row : definition.bonds)
    {
        for (const int pair : {row.type[0] * ntypes + row.type[1],
                               row.type[1] * ntypes + row.type[0]})
        {
            float* value = bond->h_twobody_params + pair * stride;
            value[0] = row.values[0];
            value[1] = row.values[1];
            value[2] = row.values[2];
            value[3] = row.values[3];
            value[4] = row.values[8];
        }
    }
    Device_Copy(&bond->d_twobody_params, bond->h_twobody_params, count);
    Copy_Atom_Types(&bond->h_atom_type, &bond->d_atom_type, definition);
    Device_Malloc_Safely((void**)&bond->d_energy_sum, sizeof(float));
    Device_Malloc_Safely((void**)&bond->d_energy_atom,
                         sizeof(float) * atom_numbers);
    deviceMemset(bond->d_energy_sum, 0, sizeof(float));
    deviceMemset(bond->d_energy_atom, 0, sizeof(float) * atom_numbers);
    if (need_full_nl_flag != NULL) *need_full_nl_flag = true;
    bond->is_initialized = 1;
    controller->Step_Print_Initial("REAXFF_BOND", "%14.7e");
    controller->printf("END INITIALIZING REAXFF BOND FORCE\n\n");
}

void Initialize_VDW(REAXFF_VDW* vdw, CONTROLLER* controller,
                    const NativeReaxFFDefinition& definition,
                    const int atom_numbers, bool* need_full_nl_flag)
{
    constexpr int stride = 8;
    const int ntypes = static_cast<int>(definition.atoms.size());
    const std::size_t count =
        static_cast<std::size_t>(ntypes) * ntypes * stride;
    vdw->atom_numbers = atom_numbers;
    vdw->atom_type_numbers = ntypes;
    vdw->p_vdw1 = definition.general[28];
    controller->printf("START INITIALIZING REAXFF VDW FORCE (native H5)\n");
    Malloc_Safely((void**)&vdw->h_twobody_params, sizeof(float) * count);
    memset(vdw->h_twobody_params, 0, sizeof(float) * count);
    for (int i = 0; i < ntypes; ++i)
    {
        for (int j = 0; j < ntypes; ++j)
        {
            float* value = vdw->h_twobody_params + (i * ntypes + j) * stride;
            value[0] = 2.0f * sqrtf(definition.atoms[i].values[3] *
                                    definition.atoms[j].values[3]);
            value[1] = sqrtf(definition.atoms[i].values[4] *
                             definition.atoms[j].values[4]);
            value[2] = sqrtf(definition.atoms[i].values[8] *
                             definition.atoms[j].values[8]);
            value[3] = sqrtf(definition.atoms[i].values[9] *
                             definition.atoms[j].values[9]);
        }
    }
    for (const auto& row : definition.off_diagonal)
    {
        for (const int pair : {row.type[0] * ntypes + row.type[1],
                               row.type[1] * ntypes + row.type[0]})
        {
            float* value = vdw->h_twobody_params + pair * stride;
            if (row.values[0] > 0.0f) value[1] = row.values[0];
            if (row.values[1] > 0.0f) value[0] = 2.0f * row.values[1];
            if (row.values[2] > 0.0f) value[2] = row.values[2];
        }
    }
    Device_Copy(&vdw->d_twobody_params, vdw->h_twobody_params, count);
    Copy_Atom_Types(&vdw->h_atom_type, &vdw->d_atom_type, definition);
    Device_Malloc_Safely((void**)&vdw->d_energy_sum, sizeof(float));
    Device_Malloc_Safely((void**)&vdw->d_energy_atom,
                         sizeof(float) * atom_numbers);
    deviceMemset(vdw->d_energy_sum, 0, sizeof(float));
    deviceMemset(vdw->d_energy_atom, 0, sizeof(float) * atom_numbers);
    if (need_full_nl_flag != NULL) *need_full_nl_flag = true;
    vdw->is_initialized = 1;
    controller->Step_Print_Initial("REAXFF_VDW", "%14.7e");
    controller->printf("END INITIALIZING REAXFF VDW FORCE\n\n");
}

void Initialize_Over_Under(REAXFF_OVER_UNDER* ovun, CONTROLLER* controller,
                           const NativeReaxFFDefinition& definition,
                           const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    const std::size_t npairs = static_cast<std::size_t>(ntypes) * ntypes;
    ovun->atom_numbers = atom_numbers;
    ovun->atom_type_numbers = ntypes;
    ovun->p_lp1 = definition.general[15];
    ovun->p_lp3 = definition.general[5];
    ovun->p_ovun3 = definition.general[32];
    ovun->p_ovun4 = definition.general[31];
    ovun->p_ovun6 = definition.general[6];
    ovun->p_ovun7 = definition.general[8];
    ovun->p_ovun8 = definition.general[9];
    controller->printf(
        "START INITIALIZING REAXFF OVER/UNDER COORD (native H5)\n");
    controller->Step_Print_Initial("REAXFF_ELP", "%14.7e");
    controller->Step_Print_Initial("REAXFF_OVUN", "%14.7e");
    const std::vector<float**> atom_arrays = {
        &ovun->h_valency,     &ovun->h_valency_e, &ovun->h_valency_boc,
        &ovun->h_valency_val, &ovun->h_mass,      &ovun->h_p_lp2,
        &ovun->h_p_ovun2,     &ovun->h_p_ovun5};
    for (float** pointer : atom_arrays)
        Malloc_Safely((void**)pointer, sizeof(float) * ntypes);
    for (int type = 0; type < ntypes; ++type)
    {
        const auto& value = definition.atoms[type].values;
        ovun->h_valency[type] = value[1];
        ovun->h_mass[type] = value[2];
        ovun->h_valency_e[type] = value[7];
        ovun->h_valency_boc[type] = value[10];
        ovun->h_p_ovun5[type] = value[11];
        ovun->h_p_lp2[type] = value[17];
        ovun->h_p_ovun2[type] = value[24];
        ovun->h_valency_val[type] = value[27];
    }
    Malloc_Safely((void**)&ovun->h_p_ovun1, sizeof(float) * npairs);
    Malloc_Safely((void**)&ovun->h_De_s, sizeof(float) * npairs);
    memset(ovun->h_p_ovun1, 0, sizeof(float) * npairs);
    memset(ovun->h_De_s, 0, sizeof(float) * npairs);
    for (const auto& row : definition.bonds)
    {
        for (const int pair : {row.type[0] * ntypes + row.type[1],
                               row.type[1] * ntypes + row.type[0]})
        {
            ovun->h_De_s[pair] = row.values[0];
            ovun->h_p_ovun1[pair] = row.values[7];
        }
    }
    Copy_Atom_Types(&ovun->h_atom_type, &ovun->d_atom_type, definition);
#define COPY_OVUN_ARRAY(name, count) \
    Device_Copy(&ovun->d_##name, ovun->h_##name, (count))
    COPY_OVUN_ARRAY(p_lp2, ntypes);
    COPY_OVUN_ARRAY(p_ovun2, ntypes);
    COPY_OVUN_ARRAY(p_ovun5, ntypes);
    COPY_OVUN_ARRAY(valency, ntypes);
    COPY_OVUN_ARRAY(valency_e, ntypes);
    COPY_OVUN_ARRAY(valency_boc, ntypes);
    COPY_OVUN_ARRAY(valency_val, ntypes);
    COPY_OVUN_ARRAY(mass, ntypes);
    COPY_OVUN_ARRAY(p_ovun1, npairs);
    COPY_OVUN_ARRAY(De_s, npairs);
#undef COPY_OVUN_ARRAY
    Device_Malloc_Safely((void**)&ovun->d_Delta, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_Delta_boc,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_Delta_val,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_Delta_lp,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_nlp, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_vlpex, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_Delta_lp_temp,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_dDelta_lp,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_CdDelta,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&ovun->d_energy_elp_sum, sizeof(float));
    Device_Malloc_Safely((void**)&ovun->d_energy_ovun_sum, sizeof(float));
    Device_Malloc_Safely((void**)&ovun->d_energy_atom,
                         sizeof(float) * atom_numbers);
    ovun->is_initialized = 1;
    controller->printf("END INITIALIZING REAXFF OVER/UNDER COORD\n\n");
}

void Initialize_Angle(REAXFF_VALENCE_ANGLE* angle, CONTROLLER* controller,
                      const NativeReaxFFDefinition& definition,
                      const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    const int n3 = ntypes * ntypes * ntypes;
    angle->atom_numbers = atom_numbers;
    angle->atom_type_numbers = ntypes;
    angle->params.p_coa2 = definition.general[2];
    angle->params.p_val6 = definition.general[14];
    angle->params.p_val9 = definition.general[16];
    angle->params.p_val10 = definition.general[17];
    angle->params.p_pen2 = definition.general[19];
    angle->params.p_pen3 = definition.general[20];
    angle->params.p_pen4 = definition.general[21];
    angle->params.p_coa4 = definition.general[30];
    angle->params.p_val8 = definition.general[33];
    angle->params.p_coa3 = definition.general[38];
    angle->params.thb_cut = 0.001f;
    if (controller->Command_Exist("REAXFF", "thb_cutoff"))
    {
        angle->params.thb_cut =
            atof(controller->Command("REAXFF", "thb_cutoff"));
    }
    angle->params.thb_cutsq = angle->params.thb_cut * angle->params.thb_cut;
    controller->printf("START INITIALIZING REAXFF VALENCE ANGLE (native H5)\n");
    controller->Step_Print_Initial("REAXFF_ANG", "%14.7e");
    controller->Step_Print_Initial("REAXFF_PEN", "%14.7e");
    controller->Step_Print_Initial("REAXFF_COA", "%14.7e");
    Malloc_Safely((void**)&angle->h_p_val3, sizeof(float) * ntypes);
    Malloc_Safely((void**)&angle->h_p_val5, sizeof(float) * ntypes);
    Malloc_Safely((void**)&angle->h_mass, sizeof(float) * ntypes);
    Malloc_Safely((void**)&angle->h_valency_boc, sizeof(float) * ntypes);
    for (int type = 0; type < ntypes; ++type)
    {
        const auto& value = definition.atoms[type].values;
        angle->h_mass[type] = value[2];
        angle->h_valency_boc[type] = value[10];
        angle->h_p_val3[type] = value[25];
        angle->h_p_val5[type] = value[28];
    }
    std::vector<REAXFF_THBP_Entry> all_entries;
    std::map<int, std::vector<int>> entries_by_triplet;
    for (const auto& row : definition.angles)
    {
        REAXFF_THBP_Entry entry{};
        entry.theta_00 = row.values[0];
        entry.p_val1 = row.values[1];
        entry.p_val2 = row.values[2];
        entry.p_coa1 = row.values[3];
        entry.p_val7 = row.values[4];
        entry.p_pen1 = row.values[5];
        entry.p_val4 = row.values[6];
        const int entry_index = static_cast<int>(all_entries.size());
        all_entries.push_back(entry);
        const int direct =
            (row.type[0] * ntypes + row.type[1]) * ntypes + row.type[2];
        entries_by_triplet[direct].push_back(entry_index);
        if (row.type[0] != row.type[2])
        {
            const int reverse =
                (row.type[2] * ntypes + row.type[1]) * ntypes + row.type[0];
            entries_by_triplet[reverse].push_back(entry_index);
        }
    }
    Malloc_Safely((void**)&angle->h_thbp_info, sizeof(REAXFF_THBP_Info) * n3);
    memset(angle->h_thbp_info, 0, sizeof(REAXFF_THBP_Info) * n3);
    std::vector<REAXFF_THBP_Entry> sorted_entries;
    for (int index = 0; index < n3; ++index)
    {
        const auto found = entries_by_triplet.find(index);
        if (found == entries_by_triplet.end()) continue;
        angle->h_thbp_info[index].start_idx =
            static_cast<int>(sorted_entries.size());
        angle->h_thbp_info[index].entry_count =
            static_cast<int>(found->second.size());
        for (const int entry_index : found->second)
            sorted_entries.push_back(all_entries[entry_index]);
    }
    Host_Copy(&angle->h_thbp_entries, sorted_entries);
    Copy_Atom_Types(&angle->h_atom_type, &angle->d_atom_type, definition);
    Device_Copy(&angle->d_p_val3, angle->h_p_val3, ntypes);
    Device_Copy(&angle->d_p_val5, angle->h_p_val5, ntypes);
    Device_Copy(&angle->d_mass, angle->h_mass, ntypes);
    Device_Copy(&angle->d_valency_boc, angle->h_valency_boc, ntypes);
    Device_Copy(&angle->d_thbp_info, angle->h_thbp_info, n3);
    Device_Copy(&angle->d_thbp_entries, angle->h_thbp_entries,
                sorted_entries.size());
    Device_Malloc_Safely((void**)&angle->d_energy_ang_sum, sizeof(float));
    Device_Malloc_Safely((void**)&angle->d_energy_pen_sum, sizeof(float));
    Device_Malloc_Safely((void**)&angle->d_energy_coa_sum, sizeof(float));
    angle->is_initialized = 1;
    controller->printf("END INITIALIZING REAXFF VALENCE ANGLE\n\n");
}

void Initialize_Torsion(REAXFF_TORSION* torsion, CONTROLLER* controller,
                        const NativeReaxFFDefinition& definition,
                        const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    const int n4 = ntypes * ntypes * ntypes * ntypes;
    torsion->atom_numbers = atom_numbers;
    torsion->atom_type_numbers = ntypes;
    torsion->p_tor2 = definition.general[23];
    torsion->p_tor3 = definition.general[24];
    torsion->p_tor4 = definition.general[25];
    torsion->p_cot2 = definition.general[27];
    controller->Step_Print_Initial("REAXFF_TOR", "%14.7e");
    controller->Step_Print_Initial("REAXFF_CONJ", "%14.7e");
    struct Pending
    {
        std::array<int, 4> type;
        REAXFF_TORSION_Entry entry;
    };
    std::vector<Pending> pending;
    for (const auto& row : definition.torsions)
    {
        Pending item{};
        item.type = row.type;
        item.entry.p_tor1 = row.values[3];
        item.entry.V1 = row.values[0];
        item.entry.V2 = row.values[1];
        item.entry.V3 = row.values[2];
        item.entry.p_tor2 = 0.0f;
        item.entry.p_cot1 = row.values[4];
        pending.push_back(item);
    }
    std::vector<int> explicit_entry(n4, 0);
    std::vector<REAXFF_TORSION_Entry> all_entries;
    std::map<int, std::vector<int>> entries_by_quartet;
    const auto quartet_index =
        [ntypes](const int a, const int b, const int c, const int d)
    { return ((a * ntypes + b) * ntypes + c) * ntypes + d; };
    for (const auto& item : pending)
    {
        if (item.type[0] < 0 || item.type[3] < 0) continue;
        const int direct = quartet_index(item.type[0], item.type[1],
                                         item.type[2], item.type[3]);
        const int reverse = quartet_index(item.type[3], item.type[2],
                                          item.type[1], item.type[0]);
        const int entry_index = static_cast<int>(all_entries.size());
        all_entries.push_back(item.entry);
        entries_by_quartet[direct] = {entry_index};
        explicit_entry[direct] = 1;
        entries_by_quartet[reverse] = {entry_index};
        explicit_entry[reverse] = 1;
    }
    for (const auto& item : pending)
    {
        if (item.type[0] != -1 || item.type[3] != -1) continue;
        const int entry_index = static_cast<int>(all_entries.size());
        all_entries.push_back(item.entry);
        for (int outer_a = 0; outer_a < ntypes; ++outer_a)
        {
            for (int outer_b = 0; outer_b < ntypes; ++outer_b)
            {
                const int direct =
                    quartet_index(outer_a, item.type[1], item.type[2], outer_b);
                const int reverse =
                    quartet_index(outer_b, item.type[2], item.type[1], outer_a);
                if (!explicit_entry[direct])
                    entries_by_quartet[direct] = {entry_index};
                if (!explicit_entry[reverse])
                    entries_by_quartet[reverse] = {entry_index};
            }
        }
    }
    Malloc_Safely((void**)&torsion->h_torsion_info,
                  sizeof(REAXFF_TORSION_Info) * n4);
    memset(torsion->h_torsion_info, 0, sizeof(REAXFF_TORSION_Info) * n4);
    std::vector<REAXFF_TORSION_Entry> sorted_entries;
    for (int index = 0; index < n4; ++index)
    {
        const auto found = entries_by_quartet.find(index);
        if (found == entries_by_quartet.end()) continue;
        torsion->h_torsion_info[index].start_idx =
            static_cast<int>(sorted_entries.size());
        torsion->h_torsion_info[index].entry_count =
            static_cast<int>(found->second.size());
        for (const int entry_index : found->second)
            sorted_entries.push_back(all_entries[entry_index]);
    }
    Host_Copy(&torsion->h_torsion_entries, sorted_entries);
    Copy_Atom_Types(&torsion->h_atom_type, &torsion->d_atom_type, definition);
    Device_Copy(&torsion->d_torsion_info, torsion->h_torsion_info, n4);
    Device_Copy(&torsion->d_torsion_entries, torsion->h_torsion_entries,
                sorted_entries.size());
    Device_Malloc_Safely((void**)&torsion->d_energy_tor_sum, sizeof(float));
    Device_Malloc_Safely((void**)&torsion->d_energy_cot_sum, sizeof(float));
    torsion->thb_cut = 0.001f;
    if (controller->Command_Exist("REAXFF", "thb_cutoff"))
        torsion->thb_cut = atof(controller->Command("REAXFF", "thb_cutoff"));
    torsion->is_initialized = 1;
}

void Initialize_Hydrogen_Bond(REAXFF_HYDROGEN_BOND* hb, CONTROLLER* controller,
                              const NativeReaxFFDefinition& definition,
                              const int atom_numbers)
{
    const int ntypes = static_cast<int>(definition.atoms.size());
    const int n3 = ntypes * ntypes * ntypes;
    hb->atom_numbers = atom_numbers;
    hb->atom_type_numbers = ntypes;
    controller->Step_Print_Initial("REAXFF_HB", "%14.7e");
    std::vector<REAXFF_HB_Entry> all_entries;
    std::map<int, std::vector<int>> entries_by_triplet;
    for (const auto& row : definition.hydrogen_bonds)
    {
        REAXFF_HB_Entry entry = {row.values[0], row.values[1], row.values[2],
                                 row.values[3]};
        const int entry_index = static_cast<int>(all_entries.size());
        all_entries.push_back(entry);
        const int index =
            (row.type[0] * ntypes + row.type[1]) * ntypes + row.type[2];
        entries_by_triplet[index].push_back(entry_index);
    }
    Malloc_Safely((void**)&hb->h_hb_info, sizeof(REAXFF_HB_Info) * n3);
    memset(hb->h_hb_info, 0, sizeof(REAXFF_HB_Info) * n3);
    std::vector<REAXFF_HB_Entry> sorted_entries;
    for (int index = 0; index < n3; ++index)
    {
        const auto found = entries_by_triplet.find(index);
        if (found == entries_by_triplet.end()) continue;
        hb->h_hb_info[index].start_idx =
            static_cast<int>(sorted_entries.size());
        hb->h_hb_info[index].entry_count =
            static_cast<int>(found->second.size());
        for (const int entry_index : found->second)
            sorted_entries.push_back(all_entries[entry_index]);
    }
    Host_Copy(&hb->h_hb_entries, sorted_entries);
    Copy_Atom_Types(&hb->h_atom_type, &hb->d_atom_type, definition);
    Malloc_Safely((void**)&hb->h_is_hydrogen, sizeof(int) * atom_numbers);
    for (int atom = 0; atom < atom_numbers; ++atom)
    {
        hb->h_is_hydrogen[atom] =
            definition.atoms[definition.atom_type[atom]].name == "H";
    }
    Device_Copy(&hb->d_is_hydrogen, hb->h_is_hydrogen, atom_numbers);
    Device_Copy(&hb->d_hb_info, hb->h_hb_info, n3);
    Device_Copy(&hb->d_hb_entries, hb->h_hb_entries, sorted_entries.size());
    Device_Malloc_Safely((void**)&hb->d_energy_hb_sum, sizeof(float));
    hb->is_initialized = 1;
}

}  // namespace

void Initial_ReaxFF_From_Native(REAXFF* reaxff, CONTROLLER* controller,
                                const NativeReaxFFDefinition& definition,
                                const int atom_numbers, const float cutoff,
                                float* cutoff_full, bool* need_full_nl_flag)
{
    (void)cutoff;
    (void)cutoff_full;
    if (static_cast<int>(definition.atom_type.size()) != atom_numbers)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Initial_ReaxFF_From_Native",
            "ReaxFF assigned atom type count does not match the system atom "
            "count");
    }
    Initialize_EEQ(&reaxff->eeq, controller, definition, atom_numbers);
    Initialize_Bond_Order(&reaxff->bond_order, controller, definition,
                          atom_numbers);
    Initialize_Bond(&reaxff->bond, controller, definition, atom_numbers,
                    need_full_nl_flag);
    Initialize_VDW(&reaxff->vdw, controller, definition, atom_numbers,
                   need_full_nl_flag);
    Initialize_Over_Under(&reaxff->ovun, controller, definition, atom_numbers);
    Initialize_Angle(&reaxff->angle, controller, definition, atom_numbers);
    Initialize_Torsion(&reaxff->torsion, controller, definition, atom_numbers);
    Initialize_Hydrogen_Bond(&reaxff->hb, controller, definition, atom_numbers);
    if (reaxff->bond.is_initialized && reaxff->vdw.is_initialized &&
        reaxff->eeq.is_initialized)
    {
        controller->Step_Print_Initial("REAXFF", "%14.7e");
    }
}
