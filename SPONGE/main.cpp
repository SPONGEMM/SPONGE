#include "main.h"

#include "manybody/edip.h"
#include "manybody/reaxff/bond.h"
#include "manybody/reaxff/bond_order.h"
#include "manybody/reaxff/eeq.h"
#include "manybody/reaxff/hydrogen_bond.h"
#include "manybody/reaxff/torsion.h"
#include "manybody/reaxff/vdw.h"
#include "neighbor_list/full_neighbor_list.h"

#define SUBPACKAGE_HINT \
    "SPONGE, for general-purpose molecular dynamics simulations"
#define THERMOSTAT_IS(name)                             \
    (controller.Command_Choice("thermostat", (name)) || \
     controller.Command_Choice("thermostat_mode", (name)))

CONTROLLER controller;
MD_INFORMATION md_info;
DOMAIN_INFORMATION dd;
MIDDLE_Langevin_INFORMATION middle_langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
BUSSI_THERMOSTAT_INFORMATION bussi_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
PRESSURE_BASED_BAROSTAT_INFORMATION press_baro;
MC_BAROSTAT_INFORMATION mc_baro;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
LJ_SOFT_CORE lj_soft;
SOLVENT_LENNARD_JONES solvent_lj;
Particle_Mesh pm;
ANGLE angle;
UREY_BRADLEY urey_bradley;
BOND bond;
BOND_SOFT bond_soft;
CMAP cmap;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
STEER_CV steer_cv;
RESTRAIN_CV restrain_cv;
META meta;
LISTED_FORCES listed_forces;
PAIRWISE_FORCE pairwise_force;
HARD_WALL hard_wall;
SOFT_WALLS soft_walls;
LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;
SITS_INFORMATION sits;
DIHEDRAL sits_dihedral;
NON_BOND_14 sits_nb14;
CMAP sits_cmap;
STILLINGER_WEBER_INFORMATION sw;
EDIP_INFORMATION edip;
EAM_INFORMATION eam;
TERSOFF_INFORMATION tersoff;
REAXFF_EEQ reaxff_eeq;
REAXFF_BOND_ORDER reaxff_bond_order;
REAXFF_BOND reaxff_bond;
REAXFF_VDW reaxff_vdw;
REAXFF_OVER_UNDER reaxff_ovun;
REAXFF_VALENCE_ANGLE reaxff_angle;
REAXFF_TORSION reaxff_torsion;
REAXFF_HYDROGEN_BOND reaxff_hb;
QUANTUM_CHEMISTRY qc;
SPONGE_PLUGIN plugin;

int main(int argc, char* argv[])
{
    Main_Initial(argc, argv);
    for (md_info.sys.steps = 0; md_info.sys.steps <= md_info.sys.step_limit;
         md_info.sys.steps++)
    {
        Main_Calculate_Force();
        Main_Iteration();
        Main_Print();
    }
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char* argv[])
{
#ifdef USE_CPU
    max_omp_threads = omp_get_max_threads();
#endif
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    cv_controller.Initial(&controller,
                          &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    qc.Initial(&controller, md_info.atom_numbers, md_info.crd);
    controller.Step_Print_Initial("potential", "%.2f");
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);

    if (controller.Command_Exist("REAXFF", "in_file"))
    {
        reaxff_eeq.Initial(&controller, md_info.atom_numbers,
                           controller.Command("REAXFF", "in_file"),
                           controller.Command("REAXFF", "type_in_file"));
        reaxff_bond_order.Initial(&controller, md_info.atom_numbers,
                                  controller.Command("REAXFF", "in_file"),
                                  controller.Command("REAXFF", "type_in_file"),
                                  md_info.nb.cutoff,
                                  &neighbor_list.cutoff_full);
        reaxff_bond.Initial(&controller, md_info.atom_numbers, "REAXFF",
                            &neighbor_list.is_needed_full);
        reaxff_bond.d_dE_dBO_s = reaxff_bond_order.d_dE_dBO_s;
        reaxff_bond.d_dE_dBO_pi = reaxff_bond_order.d_dE_dBO_pi;
        reaxff_bond.d_dE_dBO_pi2 = reaxff_bond_order.d_dE_dBO_pi2;
        reaxff_bond.d_dbo_s_dr = reaxff_bond_order.d_dbo_s_dr;
        reaxff_bond.d_dbo_pi_dr = reaxff_bond_order.d_dbo_pi_dr;
        reaxff_bond.d_dbo_pi2_dr = reaxff_bond_order.d_dbo_pi2_dr;
        reaxff_bond.d_dbo_s_dDelta_i = reaxff_bond_order.d_dbo_s_dDelta_i;
        reaxff_bond.d_dbo_pi_dDelta_i = reaxff_bond_order.d_dbo_pi_dDelta_i;
        reaxff_bond.d_dbo_pi2_dDelta_i = reaxff_bond_order.d_dbo_pi2_dDelta_i;
        reaxff_bond.d_dbo_s_dDelta_j = reaxff_bond_order.d_dbo_s_dDelta_j;
        reaxff_bond.d_dbo_pi_dDelta_j = reaxff_bond_order.d_dbo_pi_dDelta_j;
        reaxff_bond.d_dbo_pi2_dDelta_j = reaxff_bond_order.d_dbo_pi2_dDelta_j;
        reaxff_bond.d_dbo_raw_total_dr = reaxff_bond_order.d_dbo_raw_total_dr;

        reaxff_vdw.Initial(&controller, md_info.atom_numbers, "REAXFF",
                           &neighbor_list.is_needed_full);
        reaxff_ovun.Initial(&controller, md_info.atom_numbers, "REAXFF");
        reaxff_ovun.d_dE_dBO_s = reaxff_bond_order.d_dE_dBO_s;
        reaxff_ovun.d_dE_dBO_pi = reaxff_bond_order.d_dE_dBO_pi;
        reaxff_ovun.d_dE_dBO_pi2 = reaxff_bond_order.d_dE_dBO_pi2;
        reaxff_ovun.d_dbo_s_dr = reaxff_bond_order.d_dbo_s_dr;
        reaxff_ovun.d_dbo_pi_dr = reaxff_bond_order.d_dbo_pi_dr;
        reaxff_ovun.d_dbo_pi2_dr = reaxff_bond_order.d_dbo_pi2_dr;
        reaxff_ovun.d_dbo_s_dDelta_i = reaxff_bond_order.d_dbo_s_dDelta_i;
        reaxff_ovun.d_dbo_pi_dDelta_i = reaxff_bond_order.d_dbo_pi_dDelta_i;
        reaxff_ovun.d_dbo_pi2_dDelta_i = reaxff_bond_order.d_dbo_pi2_dDelta_i;
        reaxff_ovun.d_dbo_s_dDelta_j = reaxff_bond_order.d_dbo_s_dDelta_j;
        reaxff_ovun.d_dbo_pi_dDelta_j = reaxff_bond_order.d_dbo_pi_dDelta_j;
        reaxff_ovun.d_dbo_pi2_dDelta_j = reaxff_bond_order.d_dbo_pi2_dDelta_j;
        reaxff_ovun.d_dbo_raw_total_dr = reaxff_bond_order.d_dbo_raw_total_dr;

        reaxff_bond.d_CdDelta = reaxff_ovun.d_CdDelta;
        reaxff_angle.Initial(&controller, md_info.atom_numbers, "REAXFF");
        reaxff_angle.d_dE_dBO_s = reaxff_bond_order.d_dE_dBO_s;
        reaxff_angle.d_dE_dBO_pi = reaxff_bond_order.d_dE_dBO_pi;
        reaxff_angle.d_dE_dBO_pi2 = reaxff_bond_order.d_dE_dBO_pi2;
        reaxff_angle.d_CdDelta = reaxff_ovun.d_CdDelta;
        reaxff_angle.d_dbo_s_dr = reaxff_bond_order.d_dbo_s_dr;
        reaxff_angle.d_dbo_pi_dr = reaxff_bond_order.d_dbo_pi_dr;
        reaxff_angle.d_dbo_pi2_dr = reaxff_bond_order.d_dbo_pi2_dr;
        reaxff_angle.d_dbo_s_dDelta_i = reaxff_bond_order.d_dbo_s_dDelta_i;
        reaxff_angle.d_dbo_pi_dDelta_i = reaxff_bond_order.d_dbo_pi_dDelta_i;
        reaxff_angle.d_dbo_pi2_dDelta_i = reaxff_bond_order.d_dbo_pi2_dDelta_i;
        reaxff_angle.d_dbo_s_dDelta_j = reaxff_bond_order.d_dbo_s_dDelta_j;
        reaxff_angle.d_dbo_pi_dDelta_j = reaxff_bond_order.d_dbo_pi_dDelta_j;
        reaxff_angle.d_dbo_pi2_dDelta_j = reaxff_bond_order.d_dbo_pi2_dDelta_j;
        reaxff_angle.d_dbo_raw_total_dr = reaxff_bond_order.d_dbo_raw_total_dr;

        reaxff_torsion.Initial(&controller, md_info.atom_numbers, "REAXFF");
        reaxff_torsion.d_dE_dBO_s = reaxff_bond_order.d_dE_dBO_s;
        reaxff_torsion.d_dE_dBO_pi = reaxff_bond_order.d_dE_dBO_pi;
        reaxff_torsion.d_dE_dBO_pi2 = reaxff_bond_order.d_dE_dBO_pi2;
        reaxff_torsion.d_CdDelta = reaxff_ovun.d_CdDelta;
        reaxff_hb.Initial(&controller, md_info.atom_numbers, "REAXFF");
        reaxff_hb.d_dE_dBO_s = reaxff_bond_order.d_dE_dBO_s;
        reaxff_hb.d_dE_dBO_pi = reaxff_bond_order.d_dE_dBO_pi;
        reaxff_hb.d_dE_dBO_pi2 = reaxff_bond_order.d_dE_dBO_pi2;
    }

    //------------------------- thermostat initialization-----------------------
    if (md_info.mode >= md_info.NVT &&
        (!controller.Command_Exist("thermostat") &&
         !controller.Command_Exist("thermostat_mode")))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if (md_info.mode >= md_info.NVT &&
        (THERMOSTAT_IS("middle_langevin") || THERMOSTAT_IS("langevin")))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers,
                                md_info.sys.target_temperature, md_info.h_mass);
    }
    else if (md_info.mode >= md_info.NVT && THERMOSTAT_IS("andersen"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature,
                          md_info.atom_numbers, md_info.sys.dt_in_ps,
                          md_info.h_mass);
    }
    else if (md_info.mode >= md_info.NVT && THERMOSTAT_IS("bussi_thermostat"))
    {
        bussi_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (md_info.mode >= md_info.NVT &&
             THERMOSTAT_IS("berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (md_info.mode >= md_info.NVT && THERMOSTAT_IS("nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.atom_numbers,
                    md_info.sys.target_temperature, md_info.h_mass);
    }
    //------------------------- barostat initialization-----------------------
    if (md_info.mode == md_info.NPT && !controller.Command_Exist("barostat"))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tbarostat is required for NPT simulations\n");
    }
    if (md_info.mode == md_info.NPT &&
        (controller.Command_Choice("barostat", "andersen_barostat") ||
         (controller.Command_Choice("barostat_mode", "andersen_barostat") ||
          controller.Command_Choice("barostat", "bussi_barostat") ||
          (controller.Command_Choice("barostat_mode", "bussi_barostat"))) ||
         controller.Command_Choice("barostat", "berendsen_barostat") ||
         (controller.Command_Choice("barostat_mode", "berendsen_barostat"))))
    {
        press_baro.Initial(&controller, md_info.sys.target_pressure,
                           md_info.pbc.cell, &Main_Box_Change);
    }
    if (md_info.mode == md_info.NPT &&
        (controller.Command_Choice("barostat", "monte_carlo_barostat") ||
         controller.Command_Choice("barostat_mode", "monte_carlo_barostat")))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers,
                        md_info.sys.target_pressure, md_info.sys.box_length,
                        md_info.pbc.cell);
    }
    // -----------------force field initialization-------------------------
    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff);
        lj_soft.Initial(&controller, md_info.nb.cutoff);
        pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
                   md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
                   md_info.no_direct_interaction_virtual_atom_numbers);
        pairwise_force.Initial(&controller);
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);
        // SITS initialization
        sits.Initial(&controller, md_info.atom_numbers);
        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Initial(&controller, "sits_dihedral");
            sits_nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B,
                              lj.h_atom_LJ_type, "sits_nb14");
            sits_cmap.Initial(&controller, "sits_cmap");
        }
        sits.Check_Solvent(&controller, md_info.atom_numbers,
                           solvent_lj.solvent_numbers);
    }
    else
    {
        LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
        CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
        if (controller.Command_Exist("gb", "in_file"))
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B,
                     LJ_NOPBC.h_atom_LJ_type);
        sits.Initial(&controller, md_info.atom_numbers);
    }
    listed_forces.Initial(&controller, &md_info.sys.connectivity,
                          &md_info.sys.connected_distance);

    sw.Initial(&controller, "SW", &neighbor_list.is_needed_full);
    edip.Initial(&controller, "EDIP", &neighbor_list.is_needed_full);
    eam.Initial(&controller, md_info.atom_numbers, "EAM",
                &neighbor_list.is_needed_full);
    tersoff.Initial(&controller, md_info.atom_numbers, "TERSOFF",
                    &neighbor_list.is_needed_full);

    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    bond.Initial(&controller, &md_info.sys.connectivity,
                 &md_info.sys.connected_distance);
    bond_soft.Initial(&controller, &md_info.sys.connectivity,
                      &md_info.sys.connected_distance);
    cmap.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    hard_wall.Initial(&controller, md_info.sys.target_temperature,
                      md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    if (controller.Command_Exist("constrain_mode"))
    {
        constrain.Initial_List(&controller, md_info.sys.connected_distance,
                               md_info.h_mass);
        constrain.Initial_Constrain(&controller, md_info.atom_numbers,
                                    md_info.dt, md_info.sys.box_length,
                                    md_info.h_mass, &md_info.sys.freedom);
        settle.Initial(&controller, &constrain, md_info.h_mass);
        if (controller.Command_Choice("constrain_mode", "SHAKE"))
        {
            shake.Initial_SHAKE(&controller, &constrain);
        }
        if (md_info.mode == md_info.MINIMIZATION)
        {
            constrain.v_factor = 0.0f;
        }
        if (middle_langevin.is_initialized)
        {
            constrain.v_factor = middle_langevin.exp_gamma;
            constrain.x_factor = 0.5 * (1. + middle_langevin.exp_gamma);
        }
    }
    vatom.Initial(&controller, &cv_controller, md_info.atom_numbers,
                  md_info.no_direct_interaction_virtual_atom_numbers,
                  cv_controller.cv_vatom_name, md_info.h_mass,
                  &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.crd, md_info.pbc.cell, md_info.pbc.rcell);

    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers,
                              md_info.nb.cutoff, md_info.nb.skin,
                              md_info.pbc.cell, md_info.pbc.rcell);
    }
    steer_cv.Initial(&controller, &cv_controller);
    restrain_cv.Initial(&controller, &cv_controller);
    meta.Initial(&controller, &cv_controller);

    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();

    // Initialize UG connectivity
    md_info.ug.Initial_Edge(md_info.atom_numbers);
    constrain.update_ug_connectivity(&md_info.ug.connectivity);
    settle.update_ug_connectivity(&md_info.ug.connectivity);
    vatom.update_ug_connectivity(&md_info.ug.connectivity);
    md_info.ug.Read_Update_Group(md_info.atom_numbers);
    md_info.mol.Initial(&controller);

    // Acceleration of solvent LJ initialization by UG info
    if (md_info.pbc.pbc)
    {
        solvent_lj.Initial(&controller, &lj, &lj_soft, &md_info,
                           md_info.mode >= md_info.NVT);
    }
    Main_Process_Management();
    if (CONTROLLER::PP_MPI_size > 1)
    {
        md_info.nb.Excluded_List_Reform(md_info.atom_numbers);
    }
    pm.exclude_factor = CONTROLLER::PP_MPI_size == 1 ? 1.0f : 0.5f;

    // ---------------end process partition---------------
    deviceStreamCreate(&main_stream);
    dd.Create_Stream();
    pm.Create_Stream();

    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        dd.Send_Recv_Dom_Dec(&controller);
        dd.Find_Neighbor_Domain(&controller, &md_info);
        dd.Get_Atoms(&controller, &md_info);
        dd.Get_Ghost(&controller, &md_info);
        dd.Get_Excluded(&controller, &md_info);
        printf("rank_id=%d, atom_numbers=%d, ghost_numbers=%d\n",
               CONTROLLER::PP_MPI_rank, dd.atom_numbers, dd.ghost_numbers);

        neighbor_list.Update(
            dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
            md_info.pbc.cell, md_info.pbc.rcell, 0, neighbor_list.FORCED_UPDATE,
            md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
            md_info.nb.d_excluded_numbers);

        middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
        ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
        nhc.Get_Local(dd.atom_local, dd.atom_numbers);

        lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        lj_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        solvent_lj.Get_Local(dd.res_numbers, dd.res_len, dd.atom_numbers,
                             dd.d_mass);
        listed_forces.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers,
                                 dd.ghost_numbers, dd.atom_local_label,
                                 dd.atom_local_id);

        angle.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                        dd.atom_local_label, dd.atom_local_id);
        urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                               dd.atom_local_label, dd.atom_local_id);
        bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        bond_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
        cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        dihedral.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
        improper.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
        nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        restrain.Get_Local(dd.atom_local, dd.atom_numbers, dd.atom_local_label,
                           dd.atom_local_id);
        constrain.Get_Local(dd.atom_local_id, dd.atom_local_label,
                            dd.atom_numbers);
        settle.Get_Local(dd.atom_local_id, dd.atom_local_label,
                         dd.atom_numbers);
        vatom.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
        sits.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                    dd.ghost_numbers, dd.atom_local_label,
                                    dd.atom_local_id);
            sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
            sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        }
    }

    deviceMemset(md_info.crd, 0, sizeof(VECTOR) * md_info.atom_numbers);
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);

    if (pm.is_initialized)
        Reset_List(pm.d_direct_atom_energy, 0, md_info.atom_numbers);
    if (lj.is_initialized)
        Reset_List(lj.d_LJ_energy_atom, 0, md_info.atom_numbers);
    if (lj_soft.is_initialized)
        Reset_List(lj_soft.d_LJ_energy_atom, 0, md_info.atom_numbers);

    MPI_Barrier(MPI_COMM_WORLD);
    controller.Print_First_Line_To_Mdout();
}

void Main_Calculate_Force()
{
    bool use_reaxff_eeq = reaxff_eeq.is_initialized;
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    qc.Solve_SCF(dd.crd, md_info.sys.box_length);
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps,
                                        &md_info.need_potential);
    press_baro.Ask_For_Calculate_Pressure(md_info.sys.steps,
                                          &md_info.need_pressure);
    if (bd_thermo.is_initialized || bussi_thermo.is_initialized ||
        nhc.is_initialized)
    {
        md_info.need_kinetic = 1;
    }
    sits.Reset_Force_Energy(&md_info.need_potential);

    controller.Get_Time_Recorder("Calculate_Force")->Start();
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, false, false, true, false);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        dd.Reset_Force_and_Virial(&md_info);
        dd.Update_Ghost(&controller);
        neighbor_list.Update(
            dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
            md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
            neighbor_list.CONDITIONAL_UPDATE, md_info.nb.d_excluded_list_start,
            md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

        reaxff_eeq.Calculate_Charges(
            dd.atom_numbers, md_info.d_charge, dd.crd, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.d_nl, md_info.nb.cutoff,
            dd.d_energy, dd.frc, md_info.need_pressure, dd.d_virial);
        if (CONTROLLER::PP_MPI_size == 1 && dd.d_charge != md_info.d_charge)
        {
            dd.Sync_Local_Charge_From_Global(md_info.d_charge);
        }
        reaxff_bond_order.Calculate_Bond_Order(
            dd.atom_numbers, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
            neighbor_list.full_neighbor_list.d_nl, md_info.nb.cutoff);

        if (reaxff_bond_order.is_initialized)
        {
            reaxff_bond_order.Clear_Derivatives(dd.atom_numbers,
                                                reaxff_ovun.d_CdDelta);
        }

        if (reaxff_bond.is_initialized && reaxff_bond_order.is_initialized)
        {
            deviceMemcpy(reaxff_bond.d_bo_s, reaxff_bond_order.d_corrected_bo_s,
                         sizeof(float) * dd.atom_numbers * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
            deviceMemcpy(reaxff_bond.d_bo_pi,
                         reaxff_bond_order.d_corrected_bo_pi,
                         sizeof(float) * dd.atom_numbers * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
            deviceMemcpy(reaxff_bond.d_bo_pi2,
                         reaxff_bond_order.d_corrected_bo_pi2,
                         sizeof(float) * dd.atom_numbers * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }
        reaxff_bond.REAXFF_Bond_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.d_nl, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        reaxff_vdw.REAXFF_VDW_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.d_nl, md_info.nb.cutoff,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        reaxff_ovun.Calculate_Over_Under_Energy_And_Force(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, &reaxff_bond_order, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        reaxff_angle.Calculate_Valence_Angle_Energy_And_Force(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            &reaxff_bond_order, reaxff_ovun.d_Delta, reaxff_ovun.d_Delta_boc,
            reaxff_ovun.d_Delta_val, reaxff_ovun.d_nlp, reaxff_ovun.d_vlpex,
            reaxff_ovun.d_dDelta_lp, reaxff_ovun.d_CdDelta,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        reaxff_torsion.Calculate_Torsion_Energy_And_Force(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            &reaxff_bond_order, reaxff_ovun.d_Delta_boc, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        reaxff_hb.Calculate_HB_Energy_And_Force(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            &reaxff_bond_order, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);

        if (reaxff_bond_order.is_initialized)
        {
            reaxff_bond_order.Calculate_Forces(
                dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.nb.cutoff, reaxff_ovun.d_CdDelta,
                md_info.need_pressure, dd.d_virial);
        }

        // NOPBC START
        LJ_NOPBC.LJ_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.frc, md_info.need_potential,
            dd.d_energy, dd.d_excluded_list_start, dd.d_excluded_list,
            dd.d_excluded_numbers);
        CF_NOPBC.Coulomb_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.d_charge, dd.frc,
            md_info.need_potential, dd.d_energy, dd.d_excluded_list_start,
            dd.d_excluded_list, dd.d_excluded_numbers);
        gb.Get_Effective_Born_Radius(dd.crd);
        gb.GB_Force_With_Atom_Energy(dd.atom_numbers, dd.crd, dd.d_charge,
                                     dd.frc, dd.d_energy);
        // NOPBC END

        if (!use_reaxff_eeq)
        {
            pm.MPI_PME_Excluded_Force_With_Atom_Energy(
                dd.atom_numbers, dd.atom_local, dd.atom_local_id, dd.crd,
                md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                dd.d_excluded_list_start, dd.d_excluded_list,
                dd.d_excluded_numbers, dd.frc, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial);
        }

        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
                dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_cmap.CMAP_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, md_info.nb.cutoff, pm.beta,
                md_info.need_potential, dd.d_energy, md_info.need_pressure,
                dd.d_virial, pm.d_direct_atom_energy);
            sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj_soft, dd.frc, md_info.pbc.cell,
                md_info.pbc.rcell, neighbor_list.d_nl, md_info.nb.cutoff,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial, pm.d_direct_atom_energy);
        }
        else
        {
            lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, pm.beta, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);

            lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, pm.beta, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);
        }
        solvent_lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.res_numbers, dd.res_start, dd.crd, dd.d_charge,
            dd.frc, md_info.pbc.cell, md_info.pbc.rcell, neighbor_list.d_nl,
            pm.beta, md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, pm.d_direct_atom_energy);

        lj.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);

        lj_soft.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);
        sw.SW_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        edip.EDIP_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        eam.EAM_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        tersoff.TERSOFF_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        listed_forces.Compute_Force(dd.atom_numbers, dd.crd, md_info.pbc.cell,
                                    md_info.pbc.rcell, dd.frc,
                                    md_info.need_potential, dd.d_energy,
                                    md_info.need_pressure, dd.d_virial);
        pairwise_force.Compute_Force(
            neighbor_list.d_nl, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
            md_info.nb.cutoff, pm.beta, dd.d_charge, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, pm.d_direct_atom_energy);
        angle.Angle_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        bond.Bond_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        bond_soft.Soft_Bond_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, false);
        cmap.CMAP_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        improper.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
            dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        soft_walls.Compute_Force(dd.atom_numbers, dd.crd, dd.frc,
                                 md_info.need_potential, dd.d_energy);
        plugin.Calculate_Force();

        restrain.Restraint(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                           md_info.need_potential, dd.d_energy,
                           md_info.need_pressure, dd.d_virial, dd.frc, &md_info,
                           &dd);

        // 如果单进程，PP-PM 共享进程， PM直接调用dd.crd等
        // frc, virial, energy 直接存到原位，即不必再pme通信力等
        if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
        {
            vatom.Coordinate_Refresh_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            if (!use_reaxff_eeq)
            {
                pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                    dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                    dd.frc, md_info.need_pressure, md_info.need_potential,
                    dd.d_virial, dd.d_energy, md_info.sys.steps);
            }

            cv_controller.Compute_CV_For_Print(
                md_info.atom_numbers, dd.crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.sys.steps,
                md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);

            steer_cv.Steer(md_info.atom_numbers, dd.crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps, dd.d_energy,
                           dd.d_virial, dd.frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                md_info.atom_numbers, dd.crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.sys.steps, dd.d_energy, dd.d_virial,
                dd.frc, md_info.need_potential, md_info.need_pressure);
            meta.Do_Metadynamics(md_info.atom_numbers, dd.crd, md_info.pbc.cell,
                                 md_info.pbc.rcell, md_info.sys.steps,
                                 md_info.need_potential, md_info.need_pressure,
                                 dd.frc, dd.d_energy, dd.d_virial,
                                 md_info.sys.h_temperature);
            // CV虚原子的力重分配就地进行
            vatom.Force_Redistribute_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, dd.frc);
        }
        else
        {
            if (!use_reaxff_eeq)
            {
                pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                                   dd.atom_numbers);
            }
        }
        sits.Update_And_Enhance(
            md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure,
            dd.d_virial, dd.frc,
            1.0f / (CONSTANT_kB * md_info.sys.target_temperature));
        vatom.Force_Redistribute(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                                 dd.frc);
    }
    else
    {
        if (!use_reaxff_eeq)
        {
            pm.reset_global_force(
                md_info.no_direct_interaction_virtual_atom_numbers);
            // CV虚原子坐标更新在此处进行
            vatom.Coordinate_Refresh_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                md_info.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.d_charge, md_info.frc, md_info.need_pressure,
                md_info.need_potential, md_info.d_atom_virial_tensor,
                md_info.d_atom_energy, md_info.sys.steps);
            cv_controller.Compute_CV_For_Print(
                md_info.atom_numbers, pm.g_crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.sys.steps,
                md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);
            steer_cv.Steer(md_info.atom_numbers, pm.g_crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps,
                           md_info.d_atom_energy, md_info.d_atom_virial_tensor,
                           pm.g_frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                md_info.atom_numbers, pm.g_crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.sys.steps, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, pm.g_frc, md_info.need_potential,
                md_info.need_pressure);
            meta.Do_Metadynamics(
                md_info.atom_numbers, pm.g_crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.sys.steps, md_info.need_potential,
                md_info.need_pressure, pm.g_frc, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, md_info.sys.h_temperature);
            // CV虚原子的力重分配就地进行
            vatom.Force_Redistribute_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, pm.g_frc);
            pm.add_force_g_to_l(md_info.frc);
            pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                               dd.atom_numbers);
        }
    }
    md_info.min.Scale_Force_For_Dynamic_Dt(dd.atom_numbers, dd.d_mass_inverse,
                                           dd.frc, dd.vel, dd.acc);
    controller.Get_Time_Recorder("Calculate_Force")->Stop();
}

void Main_Iteration()
{
    controller.Get_Time_Recorder("Iteration")->Start();
    // calc ek, potential
    if (md_info.need_potential || md_info.need_pressure || md_info.need_kinetic)
    {
        dd.Get_Ek_and_Temperature(&controller, &md_info);
    }
    dd.Get_Potential(&controller, &md_info);

    if (md_info.mode != md_info.RERUN)
    {
        Main_MC_Barostat();
        //   thermostat and constraint
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            settle.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                             md_info.pbc.rcell);
            shake.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                            md_info.pbc.rcell);

            if (md_info.mode == md_info.NVE)
            {
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
            }
            else if (md_info.mode == md_info.MINIMIZATION)
            {
                md_info.min.Gradient_Descent(dd.atom_numbers, dd.crd, dd.frc,
                                             dd.vel, dd.d_mass_inverse);
                constrain.v_factor = fmaxf(FLT_MIN, md_info.min.momentum_keep);
            }
            else if (middle_langevin.is_initialized)
            {
                middle_langevin.MD_Iteration_Leap_Frog(dd.frc, dd.vel, dd.acc,
                                                       dd.crd);
                constrain.v_factor = middle_langevin.exp_gamma;
                constrain.x_factor = 0.5f * middle_langevin.exp_gamma + 0.5f;
            }
            else if (bd_thermo.is_initialized)
            {
                bd_thermo.Record_Temperature(dd.temperature,
                                             md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bd_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
            }
            else if (bussi_thermo.is_initialized)
            {
                bussi_thermo.Record_Temperature(dd.temperature,
                                                md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bussi_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
            }
            else if (ad_thermo.is_initialized)
            {
                if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
                {
                    ad_thermo.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc,
                                                     dd.acc, md_info.dt);
                    settle.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell);
                    shake.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell, dd.atom_numbers);
                    constrain.v_factor = FLT_MIN;
                    constrain.x_factor = 0.5;
                }
                else
                {
                    md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd,
                                          dd.frc, dd.d_mass_inverse,
                                          md_info.dt);
                    constrain.v_factor = 1.0;
                    constrain.x_factor = 1.0;
                }
            }
            else if (nhc.is_initialized)
            {
                nhc.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc, dd.acc,
                                           md_info.dt, dd.h_ek_total,
                                           md_info.sys.freedom);
            }

            settle.Do_SETTLE(dd.d_mass, dd.crd, md_info.pbc.cell,
                             md_info.pbc.rcell, dd.vel, md_info.need_pressure,
                             md_info.sys.d_stress);
            shake.Constrain(dd.atom_numbers, dd.crd, dd.vel, dd.d_mass_inverse,
                            dd.d_mass, md_info.pbc.cell, md_info.pbc.rcell,
                            md_info.need_pressure, md_info.sys.d_stress);
            hard_wall.Reflect(dd.atom_numbers, dd.crd, dd.vel);
        }
        // barostat
        if (md_info.need_pressure && !mc_baro.is_initialized)
        {
            md_info.Get_pressure(&controller, dd.atom_numbers, dd.vel,
                                 dd.d_mass, dd.d_virial, main_stream);
            md_info.sys.Get_Density();
            press_baro.Regulate_Pressure(
                md_info.sys.steps, md_info.sys.h_stress, md_info.pbc.cell,
                md_info.dt, md_info.sys.target_pressure,
                md_info.sys.target_temperature);
        }
    }
    else  // for rerun update
    {
        md_info.rerun.Iteration();
        if (md_info.rerun.need_box_update)
        {
            Main_Box_Change(md_info.rerun.g, 1, 0, 0);
        }
        md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }

    // vatom refresh and domain-decomposition get-local
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        vatom.Coordinate_Refresh(dd.crd, md_info.pbc.cell, md_info.pbc.rcell);
        // need_change_particles
        if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
            md_info.mode == md_info.RERUN)
        {
            if (CONTROLLER::PP_MPI_size != 1)
            {
                controller.Get_Time_Recorder("Communication")->Start();
                dd.Exchange_Particles(&controller, &md_info);
                dd.Get_Ghost(&controller, &md_info);
                controller.Get_Time_Recorder("Communication")->Stop();
                dd.Get_Excluded(&controller, &md_info);
                middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
                ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
                nhc.Get_Local(dd.atom_local, dd.atom_numbers);
                lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
                lj_soft.Get_Local(dd.atom_local, dd.atom_numbers,
                                  dd.ghost_numbers);
                solvent_lj.Get_Local(dd.res_numbers, dd.res_len,
                                     dd.atom_numbers, dd.d_mass);
                angle.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
                urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers,
                                       dd.ghost_numbers, dd.atom_local_label,
                                       dd.atom_local_id);
                listed_forces.Get_Local(dd.atom_local, dd.atom_numbers,
                                        dd.ghost_numbers, dd.atom_local_label,
                                        dd.atom_local_id);
                pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers,
                                         dd.ghost_numbers, dd.atom_local_label,
                                         dd.atom_local_id);
                bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                               dd.atom_local_label, dd.atom_local_id);
                bond_soft.Get_Local(dd.atom_local, dd.atom_numbers,
                                    dd.ghost_numbers, dd.atom_local_label,
                                    dd.atom_local_id);
                cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                               dd.atom_local_label, dd.atom_local_id);
                dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                   dd.ghost_numbers, dd.atom_local_label,
                                   dd.atom_local_id);
                improper.Get_Local(dd.atom_local, dd.atom_numbers,
                                   dd.ghost_numbers, dd.atom_local_label,
                                   dd.atom_local_id);
                nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                               dd.atom_local_label, dd.atom_local_id);
                restrain.Get_Local(dd.atom_local, dd.atom_numbers,
                                   dd.atom_local_label, dd.atom_local_id);
                constrain.Get_Local(dd.atom_local_id, dd.atom_local_label,
                                    dd.atom_numbers);
                settle.Get_Local(dd.atom_local_id, dd.atom_local_label,
                                 dd.atom_numbers);
                vatom.Get_Local(dd.atom_local_id, dd.atom_local_label,
                                dd.atom_numbers);
                sits.Get_Local(dd.atom_local, dd.atom_numbers,
                               dd.ghost_numbers);
                if (sits.is_initialized && sits.selectively_applied)
                {
                    sits_dihedral.Get_Local(
                        dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                        dd.atom_local_label, dd.atom_local_id);
                    sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers,
                                        dd.ghost_numbers, dd.atom_local_label,
                                        dd.atom_local_id);
                    sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers,
                                        dd.ghost_numbers, dd.atom_local_label,
                                        dd.atom_local_id);
                }
            }
            neighbor_list.Update(
                dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
                md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
                neighbor_list.FORCED_UPDATE, md_info.nb.d_excluded_list_start,
                md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
        }
    }

    // pme refresh atoms
    if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
        md_info.mode == md_info.RERUN)
    {
        controller.Get_Time_Recorder("Communication")->Start();
        pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge,
                     dd.atom_numbers, dd.crd, dd.d_charge, dd.atom_local, true,
                     true, true, true);
        controller.Get_Time_Recorder("Communication")->Stop();
    }
    controller.Get_Time_Recorder("Iteration")->Stop();
}

// -------  temporary test print function for domain decomposition -------

void Main_Print()
{
    if (md_info.output.Check_Mdout_Step())
    {
        md_info.Step_Print(&controller);
        if (!md_info.pbc.pbc)
        {
            // NoPBC non-bonded benergy
            CF_NOPBC.Step_Print(&controller);
            LJ_NOPBC.Step_Print(&controller);
            gb.Step_Print(&controller);
        }
        else
        {
            lj.Step_Print(&controller);
            lj_soft.Step_Print(&controller);
            pm.Step_Print(&controller);
            sits.Step_Print(&controller, 1.0f / md_info.sys.target_temperature /
                                             CONSTANT_kB);
        }
        // SITS information
        sits_dihedral.Step_Print(&controller, false);
        sits_nb14.Step_Print(&controller, false);
        sits_cmap.Step_Print(&controller, false);

        sw.Step_Print(&controller);
        eam.Step_Print(&controller);
        tersoff.Step_Print(&controller);
        reaxff_bond.Step_Print(&controller);
        reaxff_vdw.Step_Print(&controller);
        reaxff_eeq.Step_Print(&controller);
        if (reaxff_eeq.is_initialized)
        {
            reaxff_eeq.Print_Charges(md_info.d_charge);
        }
        reaxff_ovun.Step_Print_ELP(&controller);
        reaxff_ovun.Step_Print(&controller);
        reaxff_angle.Step_Print(&controller);
        reaxff_torsion.Step_Print(&controller);
        reaxff_hb.Step_Print(&controller);

        if (reaxff_bond.is_initialized && reaxff_vdw.is_initialized &&
            reaxff_eeq.is_initialized)
        {
            float total_reaxff =
                reaxff_bond.h_energy_sum + reaxff_vdw.h_energy_sum +
                reaxff_eeq.h_energy + reaxff_ovun.h_energy_lp +
                reaxff_ovun.h_energy_ovun + reaxff_angle.h_energy_ang +
                reaxff_angle.h_energy_pen + reaxff_angle.h_energy_coa +
                reaxff_torsion.h_energy_tor + reaxff_torsion.h_energy_cot +
                reaxff_hb.h_energy_hb;
            controller.Step_Print("REAXFF", total_reaxff);
        }

        // potential total is already summed in dd.d_energy by all modules
        controller.Step_Print("potential", dd.h_sum_ene_total);

        pairwise_force.Step_Print(&controller);
        angle.Step_Print(&controller);
        urey_bradley.Step_Print(&controller);
        bond.Step_Print(&controller);
        bond_soft.Step_Print(&controller);
        cmap.Step_Print(&controller);
        listed_forces.Step_Print(&controller);
        dihedral.Step_Print(&controller);
        improper.Step_Print(&controller);
        nb14.Step_Print(&controller);
        restrain.Step_Print(&controller);
        if (qc.is_initialized)
        {
            qc.Step_Print(&controller);
        }
        cv_controller.Step_Print();
        plugin.Mdout_Print();
        steer_cv.Step_Print(&controller);
        restrain_cv.Step_Print(&controller);
        meta.Step_Print(&controller);
        soft_walls.Step_Print(&controller);
        controller.Step_Print(
            "pressure", md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION);
        if (md_info.output.print_virial)
        {
            controller.Step_Print(
                "Pxx", md_info.sys.h_stress.a11 * CONSTANT_PRES_CONVERTION);
            controller.Step_Print(
                "Pyy", md_info.sys.h_stress.a22 * CONSTANT_PRES_CONVERTION);
            controller.Step_Print(
                "Pzz", md_info.sys.h_stress.a33 * CONSTANT_PRES_CONVERTION);
            controller.Step_Print("Pxy", md_info.sys.h_stress.a21 * 0.5f *
                                             CONSTANT_PRES_CONVERTION);
            controller.Step_Print("Pxz", md_info.sys.h_stress.a31 * 0.5f *
                                             CONSTANT_PRES_CONVERTION);
            controller.Step_Print("Pyz", md_info.sys.h_stress.a32 * 0.5f *
                                             CONSTANT_PRES_CONVERTION);
        }
        deviceMemcpy(&md_info.sys.h_potential, md_info.sys.d_potential,
                     sizeof(float), deviceMemcpyDeviceToHost);
        controller.Step_Print("potential", md_info.sys.h_potential);
        controller.Step_Print("density", md_info.sys.density);
        controller.Print_To_Screen_And_Mdout();
    }

    if (md_info.output.Check_Trajectory_Step())
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
        if (md_info.pbc.pbc)
        {
            md_info.mol.Molecule_Crd_Map();
            md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id, main_stream);
        }
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Vel_Traj_File();
        md_info.output.Append_Box_Traj_File();
        meta.Write_Potential();
        nhc.Save_Trajectory_File();
    }

    if (md_info.output.is_frc_traj && md_info.output.Check_Force_Step())
    {
        md_info.Frc_dd_to_Host(dd.frc, dd.atom_local_label, dd.atom_local_id,
                               main_stream);
        md_info.output.Append_Frc_Traj_File();
    }

    if (md_info.output.Check_Restart_Step())
    {
        md_info.output.Export_Restart_File();
        nhc.Save_Restart_File();
    }
}

void Main_Clear()
{
    dd.Destroy_Stream();
    pm.Destroy_Stream();
    deviceStreamDestroy(main_stream);

    controller.Final_Time_Summary(
        md_info.sys.steps, md_info.sys.speed_time_factor,
        md_info.sys.speed_unit_name.c_str(), md_info.mode);

    controller.Clear();
}

float Main_Box_Change(LTMatrix3 g, int scale_box, int scale_crd, int scale_vel)
{
    if (scale_box)
    {
        md_info.pbc.Update_Box(g);
    }
    // 放缩坐标与速度
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Scale_Positions_And_Velocities(
            g, scale_crd, scale_vel, dd.crd,
            dd.vel);  // rescale dd进程原子坐标与速度
        restrain.Update_Refcoord_Scaling(&md_info, g, md_info.dt, dd.atom_local,
                                         dd.atom_numbers, dd.atom_local_label,
                                         dd.atom_local_id);
    }

    // 大幅度放缩盒子时，重新初始化相关模块
    if (scale_box && md_info.pbc.Check_Change_Large())
    {
        Main_Box_Change_Largely();
    }

    // 更新域分解盒子
    else
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Update_Box(g, md_info.dt);
        }
        if (CONTROLLER::PM_MPI_rank < CONTROLLER::PM_MPI_size &&
            CONTROLLER::PM_MPI_rank != -1)
        {
            pm.Update_Box(md_info.pbc.cell, md_info.pbc.rcell, g, md_info.dt);
        }
    }
    return md_info.sys.Get_Volume();
}

void Main_Box_Change_Largely()
{
    controller.printf(
        "Some modules are based on the meshing methods, and it is more "
        "precise "
        "to re-initialize these modules now for a large box change.\n");

    // 先清零md_info.crd, 然后把dd.crd赋值给md_info.crd,
    // 最后规约所有进程的md_info.crd
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }
    neighbor_list.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                          md_info.nb.skin, md_info.pbc.cell, md_info.pbc.rcell);
    pm.Clear();
    pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
               md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
               md_info.no_direct_interaction_virtual_atom_numbers);
    dd.Free_Buffer();
    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        dd.Send_Recv_Dom_Dec(&controller);
        dd.Find_Neighbor_Domain(&controller, &md_info);
        dd.Get_Atoms(&controller, &md_info);
        dd.Get_Ghost(&controller, &md_info);
        dd.Get_Excluded(&controller, &md_info);
        printf("rank %d, atom_number %d, ghost_number %d\n",
               controller.MPI_rank, dd.atom_numbers, dd.ghost_numbers);

        neighbor_list.Update(
            dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
            md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
            neighbor_list.FORCED_UPDATE, md_info.nb.d_excluded_list_start,
            md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

        middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
        ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
        nhc.Get_Local(dd.atom_local, dd.atom_numbers);

        lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        lj_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        solvent_lj.Get_Local(dd.res_numbers, dd.res_len, dd.atom_numbers,
                             dd.d_mass);
        angle.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                        dd.atom_local_label, dd.atom_local_id);
        urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                               dd.atom_local_label, dd.atom_local_id);
        listed_forces.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers,
                                 dd.ghost_numbers, dd.atom_local_label,
                                 dd.atom_local_id);
        bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        bond_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
        cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        dihedral.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
        improper.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
        nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
        restrain.Get_Local(dd.atom_local, dd.atom_numbers, dd.atom_local_label,
                           dd.atom_local_id);
        constrain.Get_Local(dd.atom_local_id, dd.atom_local_label,
                            dd.atom_numbers);
        settle.Get_Local(dd.atom_local_id, dd.atom_local_label,
                         dd.atom_numbers);
        vatom.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
        sits.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                    dd.ghost_numbers, dd.atom_local_label,
                                    dd.atom_local_id);
            sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
            sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        }
    }
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);
    // pm.Get_Ghost(&controller, md_info.crd, md_info.d_charge,
    // md_info.pbc.cell, md_info.pbc.rcell); pm.Get_Local(&controller, 0,
    // md_info.sys.box_length, md_info.d_charge);   //
    // 这里设置step为零以重初始化PME相关内存
    MPI_Barrier(MPI_COMM_WORLD);
    controller.printf(
        "------------------------------------------------------------------"
        "----"
        "--------------------------------------\n");
}

void Main_Process_Management()
{
    /*
    粒子进程(Particle Process, PP)和静电进程(Particle Mesh, PM)划分
    PM 进程数 = 0：不计算Particle Mesh 倒空间部分
    PM 进程数 >= 1: 计算Particle Mesh 倒空间部分。
    2025-10-14:暂时只支持单进程PM 若总进程数=1， 则PP进程与PM进程为同一进程
    若总进程数>1， 则PP进程与PM进程分开，PM进程独享一组进程
    若多进程，最后一个进程一定包括CC进程
*/

    // ---------------start process partition---------------
    // PP 进程划分，至少要有一个PP进程
    CONTROLLER::PM_MPI_size = pm.PM_MPI_size;
    CONTROLLER::PP_MPI_size =
        (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
         CONTROLLER::CC_MPI_size) <= 0
            ? 1
            : (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
               CONTROLLER::CC_MPI_size);

    // Case 1: 单进程，不需要MPI_Comm_split
    if (CONTROLLER::MPI_size == 1)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::pm_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = 0;
        dd.pp_rank = 0;
        // 若开启PM，则PP进程与PM进程为同一进程
        if (CONTROLLER::PM_MPI_size != 0)
        {
            CONTROLLER::PM_MPI_rank = 0;
            pm.pm_rank = 0;
        }
        else
        {
            CONTROLLER::PM_MPI_rank = -1;
            pm.pm_rank = -1;
        }
    }
    // Case 2: 多进程，PM进程数=0, 初始化PP进程
    else if (CONTROLLER::PM_MPI_size == 0)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = CONTROLLER::MPI_rank;
        dd.pp_rank = CONTROLLER::PP_MPI_rank;
        pm.pm_rank = -1;
#ifdef USE_XCCL
        printf("Init XCCL for PP processes\n");
        xcclUniqueId pp_id;
        if (CONTROLLER::PP_MPI_rank == 0)
        {
            xcclGetUniqueId(&pp_id);
        }
        MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
        xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size, pp_id,
                         CONTROLLER::PP_MPI_rank);
#else
        CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
    }
    // Case 3: 多进程，PM进程数>=1, 初始化PP进程与PM进程
    else
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, CONTROLLER::MPI_rank,
                           &CONTROLLER::pp_comm);
            MPI_Comm_rank(CONTROLLER::pp_comm, &dd.pp_rank);
            CONTROLLER::PP_MPI_rank = dd.pp_rank;
#ifdef USE_XCCL
            printf("Init XCCL for PP processes\n");
            xcclUniqueId pp_id;
            if (CONTROLLER::PP_MPI_rank == 0)
            {
                xcclGetUniqueId(&pp_id);
            }
            MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
            xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size,
                             pp_id, CONTROLLER::PP_MPI_rank);
#else
            CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
        }
        else
        {
            CONTROLLER::PP_MPI_rank =
                CONTROLLER::PP_MPI_size;  // PP_MPI_rank 设置>=
                                          // PP_MPI_size，表示非PP进程
            MPI_Comm_split(MPI_COMM_WORLD, 1, CONTROLLER::MPI_rank,
                           &CONTROLLER::pm_comm);
            MPI_Comm_rank(CONTROLLER::pm_comm, &pm.pm_rank);
            CONTROLLER::PM_MPI_rank = pm.pm_rank;
#ifdef USE_XCCL
            printf("Init XCCL for PM processes\n");
            xcclUniqueId pm_id;
            if (CONTROLLER::PM_MPI_rank == 0)
            {
                xcclGetUniqueId(&pm_id);
            }
            MPI_Bcast(&pm_id, sizeof(pm_id), MPI_BYTE, 0, CONTROLLER::pm_comm);
            xcclCommInitRank(&CONTROLLER::d_pm_comm, CONTROLLER::PM_MPI_size,
                             pm_id, CONTROLLER::PM_MPI_rank);
#else
            CONTROLLER::d_pm_comm = CONTROLLER::pm_comm;
#endif
        }
    }

    controller.printf(
        "MPI process total: MPI_size=%d, PP_MPI_size=%d, PM_MPI_size=%d\n",
        CONTROLLER::MPI_size, CONTROLLER::PP_MPI_size, CONTROLLER::PM_MPI_size);
    controller.MPI_printf(
        "MPI process partition: MPI_rank=%d, PP_MPI_rank=%d, "
        "PM_MPI_rank=%d\n",
        CONTROLLER::MPI_rank, CONTROLLER::PP_MPI_rank, CONTROLLER::PM_MPI_rank);
}

void Main_MC_Barostat()
{
    if (mc_baro.is_initialized &&
        md_info.sys.steps % mc_baro.update_interval == 0)
    {
        mc_baro.energy_old = dd.h_sum_ene_total;
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            deviceMemcpy(mc_baro.frc_backup, dd.frc,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
            deviceMemcpy(mc_baro.crd_backup, dd.crd,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }
        mc_baro.Volume_Change_Attempt(md_info.sys.box_length, md_info.dt);
        Main_Box_Change(mc_baro.g, 1, 0, 0);
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Res_Crd_Map(mc_baro.g, md_info.dt);
        }

        Main_Calculate_Force();
        dd.Get_Potential(&controller, &md_info);
        mc_baro.energy_new = dd.h_sum_ene_total;
        mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV -
                             md_info.ug.ug_numbers * CONSTANT_kB *
                                 md_info.sys.target_temperature *
                                 logf(mc_baro.VDevided);
        if (mc_baro.couple_dimension != mc_baro.NO &&
            mc_baro.couple_dimension != mc_baro.XYZ)
        {
            mc_baro.extra_term -= mc_baro.surface_number *
                                  mc_baro.surface_tension * mc_baro.DeltaS;
        }
        mc_baro.accept_possibility =
            mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility =
            expf(-mc_baro.accept_possibility /
                 (CONSTANT_kB * md_info.sys.target_temperature));

        if (!mc_baro.Check_MC_Barostat_Accept())  // 如果不接受
        {
            mc_baro.g = {-mc_baro.g.a11, 0, -mc_baro.g.a22, 0, 0,
                         -mc_baro.g.a33};
            if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
            {
                deviceMemcpy(dd.frc, mc_baro.frc_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
                deviceMemcpy(dd.crd, mc_baro.crd_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
            }
            Main_Box_Change(mc_baro.g, 1, 0, 0);
        }
        mc_baro.Delta_Box_Length_Max_Update();
        dd.h_sum_ene_total = mc_baro.energy_old;  // 恢复能量值
    }
}
