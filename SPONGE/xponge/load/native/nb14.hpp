#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_NB14(NB14* nb14, const int* atom_type,
                             const float* pair_A, const float* pair_B,
                             CONTROLLER* controller,
                             const char* module_name = "nb14")
{
    int nb14_numbers = 0;
    int extra_numbers = 0;
    FILE* fp = NULL;
    FILE* fp_extra = NULL;

    if (controller->Command_Exist(module_name, "in_file"))
    {
        if (atom_type == NULL || pair_A == NULL || pair_B == NULL)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "Xponge::Native_Load_NB14",
                "Reason:\n\t'nb14_in_file' should be provided with initialized "
                "LJ parameters\n");
        }
        Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
        if (fscanf(fp, "%d", &nb14_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_NB14",
                "Reason:\n\tthe format of nb14_in_file is not right\n");
        }
    }

    if (controller->Command_Exist(module_name, "extra_in_file"))
    {
        Open_File_Safely(
            &fp_extra, controller->Command(module_name, "extra_in_file"), "r");
        if (fscanf(fp_extra, "%d", &extra_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_NB14",
                "Reason:\n\tthe format of nb14_extra_in_file is not right\n");
        }
    }

    int total_numbers = nb14_numbers + extra_numbers;
    if (total_numbers == 0)
    {
        if (fp != NULL) fclose(fp);
        if (fp_extra != NULL) fclose(fp_extra);
        return;
    }

    nb14->atom_a.resize(total_numbers);
    nb14->atom_b.resize(total_numbers);
    nb14->A.resize(total_numbers);
    nb14->B.resize(total_numbers);
    nb14->cf_scale_factor.resize(total_numbers);

    for (int i = extra_numbers; i < total_numbers; i++)
    {
        float lj_scale_factor = 0.0f;
        if (fscanf(fp, "%d %d %f %f", &nb14->atom_a[i], &nb14->atom_b[i],
                   &lj_scale_factor, &nb14->cf_scale_factor[i]) != 4)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_NB14",
                "Reason:\n\tthe format of nb14_in_file is not right\n");
        }
        int small_type = atom_type[nb14->atom_a[i]];
        int large_type = atom_type[nb14->atom_b[i]];
        if (small_type > large_type)
        {
            int temp = small_type;
            small_type = large_type;
            large_type = temp;
        }
        int pair_type = large_type * (large_type + 1) / 2 + small_type;
        nb14->A[i] = lj_scale_factor * pair_A[pair_type];
        nb14->B[i] = lj_scale_factor * pair_B[pair_type];
    }

    for (int i = 0; i < extra_numbers; i++)
    {
        if (fscanf(fp_extra, "%d %d %f %f %f", &nb14->atom_a[i],
                   &nb14->atom_b[i], &nb14->A[i], &nb14->B[i],
                   &nb14->cf_scale_factor[i]) != 5)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_NB14",
                "Reason:\n\tthe format of nb14_extra_in_file is not right\n");
        }
        nb14->A[i] *= 12.0f;
        nb14->B[i] *= 6.0f;
    }

    if (fp != NULL) fclose(fp);
    if (fp_extra != NULL) fclose(fp_extra);
}

static void Native_Load_NB14(System* system, CONTROLLER* controller)
{
    const LennardJones& lj = system->classical_force_field.lj;
    const int* atom_type = lj.atom_type.empty() ? NULL : lj.atom_type.data();
    const float* pair_A = lj.pair_A.empty() ? NULL : lj.pair_A.data();
    const float* pair_B = lj.pair_B.empty() ? NULL : lj.pair_B.data();
    Native_Load_NB14(&system->classical_force_field.nb14, atom_type, pair_A,
                     pair_B, controller);
}

}  // namespace Xponge
