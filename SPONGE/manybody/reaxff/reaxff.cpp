#include "reaxff.h"

void REAXFF::Initial(CONTROLLER* controller, int atom_numbers, float cutoff,
                     float* cutoff_full, bool* need_full_nl_flag)
{
    is_initialized = 0;
    if (!controller->Command_Exist("REAXFF", "in_file"))
    {
        return;
    }

    const char* parameter_in_file = controller->Command("REAXFF", "in_file");
    const char* type_in_file = controller->Command("REAXFF", "type_in_file");

    eeq.Initial(controller, atom_numbers, parameter_in_file, type_in_file);
    bond_order.Initial(controller, atom_numbers, parameter_in_file, type_in_file,
                       cutoff, cutoff_full);
    bond.Initial(controller, atom_numbers, "REAXFF", need_full_nl_flag);
    vdw.Initial(controller, atom_numbers, "REAXFF", need_full_nl_flag);
    ovun.Initial(controller, atom_numbers, "REAXFF");
    angle.Initial(controller, atom_numbers, "REAXFF");
    torsion.Initial(controller, atom_numbers, "REAXFF");
    hb.Initial(controller, atom_numbers, "REAXFF");

    Wire_Shared_State();
    is_initialized = 1;
}

void REAXFF::Wire_Shared_State()
{
    bond.d_dE_dBO_s = bond_order.d_dE_dBO_s;
    bond.d_dE_dBO_pi = bond_order.d_dE_dBO_pi;
    bond.d_dE_dBO_pi2 = bond_order.d_dE_dBO_pi2;
    bond.d_dbo_s_dr = bond_order.d_dbo_s_dr;
    bond.d_dbo_pi_dr = bond_order.d_dbo_pi_dr;
    bond.d_dbo_pi2_dr = bond_order.d_dbo_pi2_dr;
    bond.d_dbo_s_dDelta_i = bond_order.d_dbo_s_dDelta_i;
    bond.d_dbo_pi_dDelta_i = bond_order.d_dbo_pi_dDelta_i;
    bond.d_dbo_pi2_dDelta_i = bond_order.d_dbo_pi2_dDelta_i;
    bond.d_dbo_s_dDelta_j = bond_order.d_dbo_s_dDelta_j;
    bond.d_dbo_pi_dDelta_j = bond_order.d_dbo_pi_dDelta_j;
    bond.d_dbo_pi2_dDelta_j = bond_order.d_dbo_pi2_dDelta_j;
    bond.d_dbo_raw_total_dr = bond_order.d_dbo_raw_total_dr;

    ovun.d_dE_dBO_s = bond_order.d_dE_dBO_s;
    ovun.d_dE_dBO_pi = bond_order.d_dE_dBO_pi;
    ovun.d_dE_dBO_pi2 = bond_order.d_dE_dBO_pi2;
    ovun.d_dbo_s_dr = bond_order.d_dbo_s_dr;
    ovun.d_dbo_pi_dr = bond_order.d_dbo_pi_dr;
    ovun.d_dbo_pi2_dr = bond_order.d_dbo_pi2_dr;
    ovun.d_dbo_s_dDelta_i = bond_order.d_dbo_s_dDelta_i;
    ovun.d_dbo_pi_dDelta_i = bond_order.d_dbo_pi_dDelta_i;
    ovun.d_dbo_pi2_dDelta_i = bond_order.d_dbo_pi2_dDelta_i;
    ovun.d_dbo_s_dDelta_j = bond_order.d_dbo_s_dDelta_j;
    ovun.d_dbo_pi_dDelta_j = bond_order.d_dbo_pi_dDelta_j;
    ovun.d_dbo_pi2_dDelta_j = bond_order.d_dbo_pi2_dDelta_j;
    ovun.d_dbo_raw_total_dr = bond_order.d_dbo_raw_total_dr;

    bond.d_CdDelta = ovun.d_CdDelta;

    angle.d_dE_dBO_s = bond_order.d_dE_dBO_s;
    angle.d_dE_dBO_pi = bond_order.d_dE_dBO_pi;
    angle.d_dE_dBO_pi2 = bond_order.d_dE_dBO_pi2;
    angle.d_CdDelta = ovun.d_CdDelta;
    angle.d_dbo_s_dr = bond_order.d_dbo_s_dr;
    angle.d_dbo_pi_dr = bond_order.d_dbo_pi_dr;
    angle.d_dbo_pi2_dr = bond_order.d_dbo_pi2_dr;
    angle.d_dbo_s_dDelta_i = bond_order.d_dbo_s_dDelta_i;
    angle.d_dbo_pi_dDelta_i = bond_order.d_dbo_pi_dDelta_i;
    angle.d_dbo_pi2_dDelta_i = bond_order.d_dbo_pi2_dDelta_i;
    angle.d_dbo_s_dDelta_j = bond_order.d_dbo_s_dDelta_j;
    angle.d_dbo_pi_dDelta_j = bond_order.d_dbo_pi_dDelta_j;
    angle.d_dbo_pi2_dDelta_j = bond_order.d_dbo_pi2_dDelta_j;
    angle.d_dbo_raw_total_dr = bond_order.d_dbo_raw_total_dr;

    torsion.d_dE_dBO_s = bond_order.d_dE_dBO_s;
    torsion.d_dE_dBO_pi = bond_order.d_dE_dBO_pi;
    torsion.d_dE_dBO_pi2 = bond_order.d_dE_dBO_pi2;
    torsion.d_CdDelta = ovun.d_CdDelta;

    hb.d_dE_dBO_s = bond_order.d_dE_dBO_s;
    hb.d_dE_dBO_pi = bond_order.d_dE_dBO_pi;
    hb.d_dE_dBO_pi2 = bond_order.d_dE_dBO_pi2;
}

void REAXFF::Step_Print(CONTROLLER* controller, const float* d_charge)
{
    bond.Step_Print(controller);
    vdw.Step_Print(controller);
    eeq.Step_Print(controller);
    if (eeq.is_initialized)
    {
        eeq.Print_Charges(d_charge);
    }
    ovun.Step_Print_ELP(controller);
    ovun.Step_Print(controller);
    angle.Step_Print(controller);
    torsion.Step_Print(controller);
    hb.Step_Print(controller);

    if (bond.is_initialized && vdw.is_initialized && eeq.is_initialized)
    {
        const float total_reaxff =
            bond.h_energy_sum + vdw.h_energy_sum + eeq.h_energy +
            ovun.h_energy_lp + ovun.h_energy_ovun + angle.h_energy_ang +
            angle.h_energy_pen + angle.h_energy_coa + torsion.h_energy_tor +
            torsion.h_energy_cot + hb.h_energy_hb;
        controller->Step_Print("REAXFF", total_reaxff);
    }
}

void REAXFF::Calculate_Force(DOMAIN_INFORMATION* dd, MD_INFORMATION* md_info,
                             NEIGHBOR_LIST* neighbor_list)
{
    eeq.Calculate_Charges(dd->atom_numbers, md_info->d_charge, dd->crd,
                          md_info->pbc.cell, md_info->pbc.rcell,
                          neighbor_list->d_nl, md_info->nb.cutoff, dd->d_energy,
                          dd->frc, md_info->need_pressure, dd->d_virial);
    if (CONTROLLER::PP_MPI_size == 1 && dd->d_charge != md_info->d_charge)
    {
        dd->Sync_Local_Charge_From_Global(md_info->d_charge);
    }

    bond_order.Calculate_Bond_Order(dd->atom_numbers, dd->crd, md_info->pbc.cell,
                                    md_info->pbc.rcell,
                                    neighbor_list->full_neighbor_list.d_nl,
                                    md_info->nb.cutoff);

    if (bond_order.is_initialized)
    {
        bond_order.Clear_Derivatives(dd->atom_numbers, ovun.d_CdDelta);
    }

    if (bond.is_initialized && bond_order.is_initialized)
    {
        deviceMemcpy(bond.d_bo_s, bond_order.d_corrected_bo_s,
                     sizeof(float) * dd->atom_numbers * dd->atom_numbers,
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(bond.d_bo_pi, bond_order.d_corrected_bo_pi,
                     sizeof(float) * dd->atom_numbers * dd->atom_numbers,
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(bond.d_bo_pi2, bond_order.d_corrected_bo_pi2,
                     sizeof(float) * dd->atom_numbers * dd->atom_numbers,
                     deviceMemcpyDeviceToDevice);
    }

    bond.REAXFF_Bond_Force_With_Atom_Energy_And_Virial(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, neighbor_list->d_nl, md_info->need_potential,
        dd->d_energy, md_info->need_pressure, dd->d_virial);
    vdw.REAXFF_VDW_Force_With_Atom_Energy_And_Virial(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, neighbor_list->d_nl, md_info->nb.cutoff,
        md_info->need_potential, dd->d_energy, md_info->need_pressure,
        dd->d_virial);
    ovun.Calculate_Over_Under_Energy_And_Force(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, &bond_order, md_info->need_potential, dd->d_energy,
        md_info->need_pressure, dd->d_virial);
    angle.Calculate_Valence_Angle_Energy_And_Force(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, neighbor_list->full_neighbor_list.d_nl, &bond_order,
        ovun.d_Delta, ovun.d_Delta_boc, ovun.d_Delta_val, ovun.d_nlp,
        ovun.d_vlpex, ovun.d_dDelta_lp, ovun.d_CdDelta,
        md_info->need_potential, dd->d_energy, md_info->need_pressure,
        dd->d_virial);
    torsion.Calculate_Torsion_Energy_And_Force(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, neighbor_list->full_neighbor_list.d_nl, &bond_order,
        ovun.d_Delta_boc, md_info->need_potential, dd->d_energy,
        md_info->need_pressure, dd->d_virial);
    hb.Calculate_HB_Energy_And_Force(
        dd->atom_numbers, dd->crd, dd->frc, md_info->pbc.cell,
        md_info->pbc.rcell, neighbor_list->full_neighbor_list.d_nl, &bond_order,
        md_info->need_potential, dd->d_energy, md_info->need_pressure,
        dd->d_virial);

    if (bond_order.is_initialized)
    {
        bond_order.Calculate_Forces(dd->atom_numbers, dd->crd, dd->frc,
                                    md_info->pbc.cell, md_info->pbc.rcell,
                                    md_info->nb.cutoff, ovun.d_CdDelta,
                                    md_info->need_pressure, dd->d_virial);
    }
}
