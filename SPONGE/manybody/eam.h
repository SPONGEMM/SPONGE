#ifndef EAM_FORCE_H
#define EAM_FORCE_H
#include "../common.h"
#include "../control.h"

// Daw and Baskes, Phys. Rev. B 29, 6443 (1984)
struct EAM_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    int nrho, nr;
    float drho, dr;
    float cut;

    float *h_embed = NULL, *d_embed = NULL;
    float *h_electron_density = NULL, *d_electron_density = NULL;
    float *h_pair_potential = NULL, *d_pair_potential = NULL;

    float *h_rho = NULL, *d_rho = NULL;
    float *h_df_drho = NULL, *d_df_drho = NULL;

    float *h_energy_atom = NULL, *d_energy_atom = NULL;
    float h_energy_sum = 0, *d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, const int atom_numbers,
                 const char* module_name = NULL,
                 bool* need_full_nl_flag = NULL);
    void Read_Funcfl(FILE* fp, CONTROLLER* controller);
    void Read_Setfl(FILE* fp, CONTROLLER* controller);

    void EAM_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const Boundary boundary, const ATOM_GROUP* nl,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller);
};

#endif
