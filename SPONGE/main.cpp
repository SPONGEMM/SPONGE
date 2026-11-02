#include "main.h"

#define SUBPACKAGE_HINT \
    "SPONGE, for general-purpose molecular dynamics simulations"
CONTROLLER controller;
Xponge::System Xponge::system;
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
CMAP cmap;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SHAKE shake;
MATRIX_CONSTRAINT matrix_constraint;
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
REAXFF reaxff;
SPONGE_PLUGIN plugin;

deviceStream_t main_stream;

namespace
{
SpongeIO::RuntimeState io_runtime{
    controller, Xponge::system, md_info, dd,         middle_langevin,
    ad_thermo,  bussi_thermo,   nhc,     press_baro, mc_baro,
    restrain,   cv_controller,  meta,    sits,       reaxff,
    main_stream};
SpongeIO::InputSession input_session(io_runtime);
SpongeIO::OutputSession output_session(io_runtime);
SpongeRuntime::InitialStateBuilder initial_state(
    {io_runtime, input_session, bd_thermo, constrain, settle, shake,
     matrix_constraint, vatom, pm, plugin, steer_cv, restrain_cv});

void Reject_Open_Boundary_Module(const bool is_initialized,
                                 const char* module_name)
{
    if (md_info.pbc.boundary.policy != BoundaryPolicy::Open || !is_initialized)
        return;
    std::string reason = "Reason:\n\t";
    reason += module_name;
    reason +=
        " currently requires periodic boundary conditions and a periodic "
        "neighbor-list backend\n";
    controller.Throw_SPONGE_Error(spongeErrorConflictingCommand, "Main_Initial",
                                  reason.c_str());
}

}  // namespace

int main(int argc, char* argv[])
{
    Main_Initial(argc, argv);
    initial_state.Finalize_Run_Range();
    for (; md_info.sys.steps <= md_info.sys.step_limit; md_info.sys.steps++)
    {
        Main_Sync_Dynamic_Targets_To_Controllers();
        Main_Calculate_Force();
        Main_Iteration();
        Main_Print();
    }
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char* argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    initial_state.Load_Base_State();
    controller.Step_Print_Initial("potential", "%.2f");
    controller.Step_Print_Initial("eff_pot", "%.7e");
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);

    initial_state.Initialize_Dynamics(&Main_Box_Change);
    pairwise_force.Initial(&controller);

    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff);
        lj_soft.Initial(&controller, md_info.nb.cutoff);
        pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.boundary,
                   md_info.sys.box_length, md_info.nb.cutoff,
                   md_info.no_direct_interaction_virtual_atom_numbers);
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);

        sits.Initial(&controller, md_info.atom_numbers,
                     md_info.pbc.boundary.policy);
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
        if (controller.Command_Exist("gb", "in_file") ||
            !Xponge::system.generalized_born.radius.empty())
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B,
                     LJ_NOPBC.h_atom_LJ_type);
        sits.Initial(&controller, md_info.atom_numbers,
                     md_info.pbc.boundary.policy);
    }

    bond.Initial(&controller, &md_info.sys.connectivity,
                 &md_info.sys.connected_distance);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    cmap.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    listed_forces.Initial(&controller, &md_info.sys.connectivity,
                          &md_info.sys.connected_distance);

    sw.Initial(&controller, "SW", &neighbor_list.is_needed_full);
    edip.Initial(&controller, "EDIP", &neighbor_list.is_needed_full);
    eam.Initial(&controller, md_info.atom_numbers, "EAM",
                &neighbor_list.is_needed_full);
    tersoff.Initial(&controller, md_info.atom_numbers, "TERSOFF",
                    &neighbor_list.is_needed_full);
    reaxff.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                   &neighbor_list.cutoff_full, &neighbor_list.is_needed_full);
    Reject_Open_Boundary_Module(pairwise_force.is_initialized,
                                "pairwise_force");
    Reject_Open_Boundary_Module(sw.is_initialized, "SW");
    Reject_Open_Boundary_Module(edip.is_initialized, "EDIP");
    Reject_Open_Boundary_Module(eam.is_initialized, "EAM");
    Reject_Open_Boundary_Module(tersoff.is_initialized, "TERSOFF");
    Reject_Open_Boundary_Module(reaxff.is_initialized, "REAXFF");

    if (Xponge::system.positional_restraint.present)
    {
        restrain.Initial(&controller, md_info.atom_numbers, md_info.crd,
                         Xponge::system.positional_restraint);
    }
    else
    {
        restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    }
    hard_wall.Initial(&controller, md_info.sys.target_temperature,
                      md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    initial_state.Build_Constraints_And_Velocities();

    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers,
                              md_info.nb.cutoff, md_info.nb.skin,
                              md_info.pbc.boundary);
    }
    initial_state.Restore_Protocol_State();

    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();

    initial_state.Build_Coordinate_Derivatives();
    if (md_info.pbc.pbc)
    {
        solvent_lj.Initial(&controller, &lj, &lj_soft, &md_info,
                           md_info.mode >= md_info.NVT);
    }
    initial_state.Distribute_State(&Main_Process_Management,
                                   &Main_Refresh_Local_State);

    output_session.Initial();
}

void Main_Calculate_Force()
{
    const Boundary& boundary = md_info.pbc.boundary;
    bool use_reaxff_eeq = reaxff.eeq.is_initialized;
    const int cv_atom_numbers =
        md_info.atom_numbers +
        md_info.no_direct_interaction_virtual_atom_numbers;
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps,
                                        &md_info.need_potential);
    press_baro.Ask_For_Calculate_Pressure(md_info.sys.steps,
                                          &md_info.need_pressure);
    if (press_baro.is_initialized && output_session.Observables_Due())
    {
        md_info.need_pressure = 1;
    }
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
            md_info.pbc.boundary, md_info.sys.steps,
            neighbor_list.CONDITIONAL_UPDATE, md_info.nb.d_excluded_list_start,
            md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

        reaxff.Calculate_Force(&dd, &md_info, &neighbor_list);

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

        if (!use_reaxff_eeq)
        {
            pm.MPI_PME_Excluded_Force_With_Atom_Energy(
                dd.atom_numbers, dd.atom_local, dd.atom_local_id, dd.crd,
                md_info.pbc.boundary, dd.d_charge, dd.d_excluded_list_start,
                dd.d_excluded_list, dd.d_excluded_numbers, dd.frc,
                md_info.need_potential, dd.d_energy, md_info.need_pressure,
                dd.d_virial);
        }

        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
                dd.crd, boundary, sits.pw_select.select_force[0],
                md_info.need_potential, sits.pw_select.select_atom_energy[0],
                md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
                dd.crd, dd.d_charge, boundary, sits.pw_select.select_force[0],
                md_info.need_potential, sits.pw_select.select_atom_energy[0],
                md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_cmap.CMAP_Force_With_Atom_Energy_And_Virial(
                dd.crd, boundary, sits.pw_select.select_force[0],
                md_info.need_potential, sits.pw_select.select_atom_energy[0],
                md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj, dd.frc, md_info.pbc.boundary,
                neighbor_list.d_nl, md_info.nb.cutoff, pm.beta,
                md_info.need_potential, dd.d_energy, md_info.need_pressure,
                dd.d_virial, pm.d_direct_atom_energy);
            sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj_soft, dd.frc, md_info.pbc.boundary,
                neighbor_list.d_nl, md_info.nb.cutoff, pm.beta,
                md_info.need_potential, dd.d_energy, md_info.need_pressure,
                dd.d_virial, pm.d_direct_atom_energy);
        }
        else
        {
            lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.boundary, neighbor_list.d_nl,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial, pm.d_direct_atom_energy);

            lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.boundary, neighbor_list.d_nl,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial, pm.d_direct_atom_energy);
        }
        solvent_lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.res_numbers, dd.res_start, dd.crd, dd.d_charge,
            dd.frc, md_info.pbc.boundary, neighbor_list.d_nl, pm.beta,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, pm.d_direct_atom_energy);

        lj.Long_Range_Correction(md_info.need_pressure, dd.d_virial,
                                 md_info.need_potential, dd.d_energy,
                                 md_info.pbc.boundary.cell.a11 *
                                     md_info.pbc.boundary.cell.a22 *
                                     md_info.pbc.boundary.cell.a33);

        lj_soft.Long_Range_Correction(md_info.need_pressure, dd.d_virial,
                                      md_info.need_potential, dd.d_energy,
                                      md_info.pbc.boundary.cell.a11 *
                                          md_info.pbc.boundary.cell.a22 *
                                          md_info.pbc.boundary.cell.a33);
        sw.SW_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.boundary,
            neighbor_list.full_neighbor_list.d_nl, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        edip.EDIP_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.boundary,
            neighbor_list.full_neighbor_list.d_nl, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        eam.EAM_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.boundary,
            neighbor_list.full_neighbor_list.d_nl, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        tersoff.TERSOFF_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.boundary,
            neighbor_list.full_neighbor_list.d_nl, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        listed_forces.Compute_Force(dd.atom_numbers, dd.crd, boundary, dd.frc,
                                    md_info.need_potential, dd.d_energy,
                                    md_info.need_pressure, dd.d_virial);
        pairwise_force.Compute_Force(
            neighbor_list.d_nl, dd.crd, md_info.pbc.boundary, md_info.nb.cutoff,
            pm.beta, dd.d_charge, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial, pm.d_direct_atom_energy);
        angle.Angle_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        bond.Bond_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        cmap.CMAP_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        improper.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, boundary, dd.frc, md_info.need_potential, dd.d_energy,
            md_info.need_pressure, dd.d_virial);
        nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
            dd.crd, dd.d_charge, boundary, dd.frc, md_info.need_potential,
            dd.d_energy, md_info.need_pressure, dd.d_virial);
        soft_walls.Compute_Force(dd.atom_numbers, dd.crd, dd.frc,
                                 md_info.need_potential, dd.d_energy);
        plugin.Calculate_Force();

        restrain.Restraint(dd.crd, boundary, md_info.need_potential,
                           dd.d_energy, md_info.need_pressure, dd.d_virial,
                           dd.frc, &md_info, &dd);

        if (CONTROLLER::MPI_size == 1)
        {
            vatom.Coordinate_Refresh_CV(dd.crd, boundary);
            if (!use_reaxff_eeq)
            {
                pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                    dd.crd, md_info.pbc.boundary, dd.d_charge, dd.frc,
                    md_info.need_pressure, md_info.need_potential, dd.d_virial,
                    dd.d_energy, md_info.sys.steps);
            }

            cv_controller.Compute_CV_For_Print(
                cv_atom_numbers, dd.crd, boundary, md_info.sys.steps,
                md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);

            steer_cv.Steer(cv_atom_numbers, dd.crd, boundary, md_info.sys.steps,
                           dd.d_energy, dd.d_virial, dd.frc,
                           md_info.need_potential, md_info.need_pressure);
            restrain_cv.Restraint(cv_atom_numbers, dd.crd, boundary,
                                  md_info.sys.steps, dd.d_energy, dd.d_virial,
                                  dd.frc, md_info.need_potential,
                                  md_info.need_pressure);
            meta.Do_Metadynamics(cv_atom_numbers, dd.crd, boundary,
                                 md_info.sys.steps, md_info.need_potential,
                                 md_info.need_pressure, dd.frc, dd.d_energy,
                                 dd.d_virial, md_info.sys.h_temperature);
            output_session.Write_Metadynamics_Hills();
            vatom.Force_Redistribute_CV(dd.crd, boundary, dd.frc);
        }
        else
        {
            pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                               dd.atom_numbers);
        }
        sits.Update_And_Enhance(
            md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure,
            dd.d_virial, dd.frc,
            1.0f / (CONSTANT_kB * md_info.sys.target_temperature));
        output_session.Write_Pending_Sits();
        vatom.Force_Redistribute(dd.crd, boundary, dd.frc);
    }
    else if (CONTROLLER::MPI_rank == CONTROLLER::CV_MPI_rank)
    {
        pm.reset_global_force(
            md_info.no_direct_interaction_virtual_atom_numbers);
        vatom.Coordinate_Refresh_CV(pm.g_crd, boundary);
        if (!use_reaxff_eeq)
        {
            pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                md_info.crd, md_info.pbc.boundary, md_info.d_charge,
                md_info.frc, md_info.need_pressure, md_info.need_potential,
                md_info.d_atom_virial_tensor, md_info.d_atom_energy,
                md_info.sys.steps);
        }
        cv_controller.Compute_CV_For_Print(cv_atom_numbers, pm.g_crd, boundary,
                                           md_info.sys.steps,
                                           md_info.output.write_mdout_interval,
                                           md_info.output.print_zeroth_frame);
        steer_cv.Steer(cv_atom_numbers, pm.g_crd, boundary, md_info.sys.steps,
                       md_info.d_atom_energy, md_info.d_atom_virial_tensor,
                       pm.g_frc, md_info.need_potential, md_info.need_pressure);
        restrain_cv.Restraint(cv_atom_numbers, pm.g_crd, boundary,
                              md_info.sys.steps, md_info.d_atom_energy,
                              md_info.d_atom_virial_tensor, pm.g_frc,
                              md_info.need_potential, md_info.need_pressure);
        meta.Do_Metadynamics(
            cv_atom_numbers, pm.g_crd, boundary, md_info.sys.steps,
            md_info.need_potential, md_info.need_pressure, pm.g_frc,
            md_info.d_atom_energy, md_info.d_atom_virial_tensor,
            md_info.sys.h_temperature);
        vatom.Force_Redistribute_CV(pm.g_crd, boundary, pm.g_frc);
        pm.add_force_g_to_l(md_info.frc);
        pm.Send_Recv_Force(&controller, md_info.frc, dd.frc, dd.atom_numbers);
    }
    md_info.min.Scale_Force_For_Dynamic_Dt(dd.atom_numbers, dd.d_mass_inverse,
                                           dd.frc, dd.vel, dd.acc);
    controller.Get_Time_Recorder("Calculate_Force")->Stop();
}

void Main_Refresh_Local_State(bool rebuild_dd)
{
    if (rebuild_dd)
    {
        dd.Send_Recv_Dom_Dec(&controller);
        dd.Find_Neighbor_Domain(&controller, &md_info);
        dd.Get_Atoms(&controller, &md_info);
    }
    dd.Get_Ghost(&controller, &md_info);
    dd.Get_Excluded(&controller, &md_info);

    neighbor_list.Update(
        dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
        md_info.pbc.boundary, md_info.sys.steps, neighbor_list.FORCED_UPDATE,
        md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers);

    middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
    ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
    nhc.Get_Local(dd.atom_local, dd.atom_numbers);

    lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    lj_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    solvent_lj.Get_Local(dd.res_numbers, dd.res_len, dd.atom_numbers,
                         dd.d_mass);
    listed_forces.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                             dd.atom_local_label, dd.atom_local_id);

    angle.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                    dd.atom_local_label, dd.atom_local_id);
    urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
    bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
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
    constrain.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    settle.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    shake.Get_Local(dd.atom_numbers);
    matrix_constraint.Get_Local(dd.atom_local, dd.atom_numbers);
    vatom.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    sits.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    if (sits.is_initialized && sits.selectively_applied)
    {
        sits_dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
        sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    }
}

void Main_Iteration()
{
    const Boundary& boundary = md_info.pbc.boundary;
    controller.Get_Time_Recorder("Iteration")->Start();
    if (md_info.need_potential || md_info.need_pressure || md_info.need_kinetic)
    {
        dd.Get_Ek_and_Temperature(&controller, &md_info);
    }
    dd.Get_Potential(&controller, &md_info);
    if (md_info.mode != md_info.RERUN)
    {
        Main_MC_Barostat();

        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            settle.Remember_Last_Coordinates(dd.crd, boundary);
            shake.Remember_Last_Coordinates(dd.crd, boundary);
            matrix_constraint.Remember_Last_Coordinates(dd.crd, boundary);

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
                        dd.vel, dd.crd, dd.d_mass_inverse, boundary);
                    shake.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, boundary,
                        dd.atom_numbers);
                    matrix_constraint.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, boundary,
                        dd.atom_numbers);
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

            settle.Do_SETTLE(&controller, dd.atom_local, dd.d_mass, dd.crd,
                             boundary, dd.vel, md_info.need_pressure,
                             md_info.sys.d_stress);
            shake.Constrain(dd.atom_numbers, dd.crd, dd.vel, dd.d_mass_inverse,
                            dd.d_mass, boundary, md_info.need_pressure,
                            md_info.sys.d_stress);
            matrix_constraint.Constrain(
                dd.atom_numbers, dd.crd, dd.vel, dd.d_mass_inverse, dd.d_mass,
                boundary, md_info.need_pressure, md_info.sys.d_stress);
            hard_wall.Reflect(dd.atom_numbers, dd.crd, dd.vel);
        }
        if (md_info.need_pressure && !mc_baro.is_initialized)
        {
            md_info.Get_pressure(&controller, dd.atom_numbers, dd.vel,
                                 dd.d_mass, dd.d_virial, main_stream);
            md_info.sys.Get_Density();
            press_baro.Regulate_Pressure(
                md_info.sys.steps, md_info.sys.h_stress,
                md_info.pbc.boundary.cell, md_info.dt,
                md_info.sys.target_pressure, md_info.sys.target_temperature);
        }
    }
    else
    {
        md_info.rerun.Iteration();
        if (md_info.rerun.need_box_update)
        {
            Main_Box_Change(md_info.rerun.g, 1, 0, 0);
        }
        md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        vatom.Coordinate_Refresh(dd.crd, boundary);
        if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
            md_info.mode == md_info.RERUN)
        {
            if (CONTROLLER::PP_MPI_size != 1)
            {
                controller.Get_Time_Recorder("Communication")->Start();
                dd.Exchange_Particles(&controller, &md_info);
                controller.Get_Time_Recorder("Communication")->Stop();
                Main_Refresh_Local_State(false);
            }
            else
            {
                neighbor_list.Update(
                    dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
                    md_info.pbc.boundary, md_info.sys.steps,
                    neighbor_list.FORCED_UPDATE,
                    md_info.nb.d_excluded_list_start,
                    md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
            }
        }
    }
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

void Main_Print()
{
    if (output_session.Observables_Due())
    {
        if (vatom.has_type_3)
            vatom.Throw_If_Runtime_Error(
                "coordinate refresh or force redistribution");
        md_info.Step_Print(&controller);
        if (!md_info.pbc.pbc)
        {
            CF_NOPBC.Step_Print(&controller);
            LJ_NOPBC.Step_Print(&controller);
            gb.Step_Print(&controller);
        }
        else
        {
            lj.Step_Print(&controller);
            lj_soft.Step_Print(&controller);
            pm.Step_Print(&controller);
        }
        sits.Step_Print(&controller,
                        1.0f / md_info.sys.target_temperature / CONSTANT_kB);
        sits_dihedral.Step_Print(&controller, false);
        sits_nb14.Step_Print(&controller, false);
        sits_cmap.Step_Print(&controller, false);

        sw.Step_Print(&controller);
        edip.Step_Print(&controller);
        eam.Step_Print(&controller);
        tersoff.Step_Print(&controller);
        output_session.Write_Reaxff();
        pairwise_force.Step_Print(&controller);
        angle.Step_Print(&controller);
        urey_bradley.Step_Print(&controller);
        bond.Step_Print(&controller);
        cmap.Step_Print(&controller);
        listed_forces.Step_Print(&controller);
        dihedral.Step_Print(&controller);
        improper.Step_Print(&controller);
        nb14.Step_Print(&controller);

        controller.Step_Print("potential", dd.h_sum_ene_total);

        restrain.Step_Print(&controller);
        cv_controller.Step_Print();
        plugin.Mdout_Print();
        steer_cv.Step_Print(&controller);
        restrain_cv.Step_Print(&controller);
        meta.Step_Print(&controller);
        output_session.Write_Metadynamics_Scalars();
        soft_walls.Step_Print(&controller);
        output_session.Write_Observables();
    }

    if (output_session.Trajectory_Due())
    {
        output_session.Write_Trajectory();
    }

    if (output_session.Force_Due())
    {
        output_session.Write_Force();
    }

    if (output_session.Restart_Due())
    {
        output_session.Write_Restart();
    }
    output_session.Publish();
}

void Main_Clear()
{
    output_session.Close();

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
    else  // 更新域分解盒子
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Update_Box(g, md_info.dt);
        }
        if (CONTROLLER::PM_MPI_rank < CONTROLLER::PM_MPI_size &&
            CONTROLLER::PM_MPI_rank != -1)
        {
            pm.Update_Box(md_info.pbc.boundary, g, md_info.dt);
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

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }
    neighbor_list.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                          md_info.nb.skin, md_info.pbc.boundary);
    pm.Clear();
    pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.boundary,
               md_info.sys.box_length, md_info.nb.cutoff,
               md_info.no_direct_interaction_virtual_atom_numbers);
    dd.Free_Buffer();
    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Main_Refresh_Local_State(true);
        plugin.Set_Domain_Information(&dd);
    }
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);
    MPI_Barrier(MPI_COMM_WORLD);
    controller.printf(
        "------------------------------------------------------------------"
        "----"
        "--------------------------------------\n");
}

void Main_Process_Management()
{
    CONTROLLER::PM_MPI_size = pm.PM_MPI_size;
    CONTROLLER::PP_MPI_size =
        (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
         CONTROLLER::CC_MPI_size) <= 0
            ? 1
            : (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
               CONTROLLER::CC_MPI_size);
    CONTROLLER::CV_MPI_rank =
        CONTROLLER::MPI_size == 1
            ? 0
            : (CONTROLLER::PM_MPI_size == 1 ? CONTROLLER::PP_MPI_size : -1);

    if (CONTROLLER::MPI_size > 1 && cv_controller.is_initialized &&
        CONTROLLER::CV_MPI_rank < 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "Main_Process_Management",
            "Reason:\n\tMPI CV calculations require exactly one PM process\n");
    }

    if (CONTROLLER::MPI_size == 1)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::pm_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = 0;
        dd.pp_rank = 0;
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
    else if (CONTROLLER::PM_MPI_size == 0)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = CONTROLLER::MPI_rank;
        dd.pp_rank = CONTROLLER::PP_MPI_rank;
        pm.pm_rank = -1;
#ifdef USE_XCCL
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
    else
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, CONTROLLER::MPI_rank,
                           &CONTROLLER::pp_comm);
            MPI_Comm_rank(CONTROLLER::pp_comm, &dd.pp_rank);
            CONTROLLER::PP_MPI_rank = dd.pp_rank;
#ifdef USE_XCCL
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
        "MPI process total: MPI_size=%d, PP_MPI_size=%d, PM_MPI_size=%d, "
        "CV_MPI_rank=%d\n",
        CONTROLLER::MPI_size, CONTROLLER::PP_MPI_size, CONTROLLER::PM_MPI_size,
        CONTROLLER::CV_MPI_rank);
    controller.MPI_printf(
        "MPI process partition: MPI_rank=%d, PP_MPI_rank=%d, "
        "PM_MPI_rank=%d\n",
        CONTROLLER::MPI_rank, CONTROLLER::PP_MPI_rank, CONTROLLER::PM_MPI_rank);

    if (CONTROLLER::PP_MPI_size > 1)
    {
        md_info.nb.Excluded_List_Reform(md_info.atom_numbers);
    }
    pm.exclude_factor = CONTROLLER::PP_MPI_size == 1 ? 1.0f : 0.5f;

    deviceStreamCreate(&main_stream);
    dd.Create_Stream();
    pm.Create_Stream();

    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
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
            mc_baro.g = Get_Reverse_Diagonal_Box_Change(mc_baro.g, md_info.dt);
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

void Main_Sync_Dynamic_Targets_To_Controllers()
{
    md_info.sys.Update_Targets_By_Schedule(md_info.sys.steps);
    const float target_temperature = md_info.sys.target_temperature;
    bd_thermo.Set_Target_Temperature(target_temperature);
    bussi_thermo.Set_Target_Temperature(target_temperature);
    ad_thermo.Set_Target_Temperature(target_temperature);
    middle_langevin.Set_Target_Temperature(target_temperature);
    nhc.Set_Target_Temperature(target_temperature);
}
