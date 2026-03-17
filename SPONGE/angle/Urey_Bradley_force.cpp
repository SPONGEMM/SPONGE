#include "Urey_Bradley_force.h"

#include "../xponge/load/native/urey_bradley.hpp"
#include "../xponge/xponge.h"

void UREY_BRADLEY::Initial(CONTROLLER* controller, char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "urey_bradley");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");
    const auto& urey_bradley =
        Xponge::system.classical_force_field.urey_bradley;
    Xponge::UreyBradley local_urey_bradley;
    const Xponge::UreyBradley* urey_bradley_to_use = NULL;
    const char* init_source = NULL;
    if (module_name == NULL)
    {
        urey_bradley_to_use = &urey_bradley;
        init_source = "Xponge::system";
    }
    else if (controller[0].Command_Exist(this->module_name, file_name_suffix))
    {
        Xponge::Native_Load_Urey_Bradley(&local_urey_bradley, controller,
                                         this->module_name);
        urey_bradley_to_use = &local_urey_bradley;
    }
    if (urey_bradley_to_use != NULL)
    {
        Urey_Bradley_numbers =
            static_cast<int>(urey_bradley_to_use->atom_a.size());
    }

    if (Urey_Bradley_numbers > 0)
    {
        if (module_name == NULL)
        {
            controller[0].printf("START INITIALIZING UREY BRADLEY (%s):\n",
                                 init_source);
        }
        else
        {
            controller[0].printf("START INITIALIZING UREY BRADLEY (%s_%s):\n",
                                 this->module_name, file_name_suffix);
        }
        controller[0].printf("    urey_bradley_numbers is %d\n",
                             Urey_Bradley_numbers);

        bond.bond_numbers = Urey_Bradley_numbers;
        angle.angle_numbers = Urey_Bradley_numbers;

        bond.Memory_Allocate();
        angle.Memory_Allocate();

        for (int i = 0; i < Urey_Bradley_numbers; i++)
        {
            angle.h_atom_a[i] = urey_bradley_to_use->atom_a[i];
            angle.h_atom_b[i] = urey_bradley_to_use->atom_b[i];
            angle.h_atom_c[i] = urey_bradley_to_use->atom_c[i];
            angle.h_angle_k[i] = urey_bradley_to_use->angle_k[i];
            angle.h_angle_theta0[i] = urey_bradley_to_use->angle_theta0[i];
            bond.h_atom_a[i] = urey_bradley_to_use->atom_a[i];
            bond.h_atom_b[i] = urey_bradley_to_use->atom_c[i];
            bond.h_k[i] = urey_bradley_to_use->bond_k[i];
            bond.h_r0[i] = urey_bradley_to_use->bond_r0[i];
        }

        bond.Parameter_Host_To_Device();
        angle.Parameter_Host_To_Device();

        bond.is_initialized = 1;
        angle.is_initialized = 1;

        is_initialized = 1;
    }
    else
    {
        controller[0].printf("UREY BRADLEY IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING UREY BRADLEY\n\n");
    }
}

void UREY_BRADLEY::Urey_Bradley_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        bond.Bond_Force_With_Atom_Energy_And_Virial(
            crd, cell, rcell, frc, need_atom_energy, atom_energy, need_virial,
            atom_virial);
        angle.Angle_Force_With_Atom_Energy_And_Virial(
            crd, cell, rcell, frc, need_atom_energy, atom_energy, need_virial,
            atom_virial);
    }
}

void UREY_BRADLEY::Get_Local(int* atom_local, int local_atom_numbers,
                             int ghost_numbers, char* atom_local_label,
                             int* atom_local_id)
{
    bond.Get_Local(atom_local, local_atom_numbers, ghost_numbers,
                   atom_local_label, atom_local_id);
    angle.Get_Local(atom_local, local_atom_numbers, ghost_numbers,
                    atom_local_label, atom_local_id);
}

void UREY_BRADLEY::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
        return;

    Sum_Of_List(bond.d_bond_ene, bond.d_sigma_of_bond_ene, bond.num_bond_local);
    Sum_Of_List(angle.d_angle_ene, angle.d_sigma_of_angle_ene,
                angle.num_angle_local);
    deviceMemcpy(bond.h_sigma_of_bond_ene, bond.d_sigma_of_bond_ene,
                 sizeof(float), deviceMemcpyDeviceToHost);
    deviceMemcpy(angle.h_sigma_of_angle_ene, angle.d_sigma_of_angle_ene,
                 sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, angle.h_sigma_of_angle_ene, 1, MPI_FLOAT,
                  MPI_SUM, CONTROLLER::pp_comm);
    MPI_Allreduce(MPI_IN_PLACE, bond.h_sigma_of_bond_ene, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif

    if (CONTROLLER::MPI_rank == 0)
    {
        controller->Step_Print(
            this->module_name,
            bond.h_sigma_of_bond_ene[0] + angle.h_sigma_of_angle_ene[0], true);
    }
}
