#pragma once
#include "../common.h"
#include "../control.h"
#include "constrain.h"

struct CONSTRAIN_TRIANGLE
{
    int atom_A;
    int atom_B;
    int atom_C;
    float ra;
    float rb;
    float rc;
    float rd;
    float re;
};

struct SETTLE
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    CONSTRAIN* constrain;

    void Initial(CONTROLLER* controller, CONSTRAIN* constrain, float* h_mass,
                 const char* module_name = NULL);

    int triangle_numbers = 0;
    CONSTRAIN_TRIANGLE *d_triangles = NULL, *h_triangles = NULL;

    int pair_numbers = 0;
    CONSTRAIN_PAIR *d_pairs = NULL, *h_pairs = NULL;

    VECTOR* last_pair_AB = NULL;
    VECTOR* last_triangle_BA = NULL;
    VECTOR* last_triangle_CA = NULL;
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);

    LTMatrix3* virial_tensor = NULL;
    void Do_SETTLE(CONTROLLER* controller, const int* atom_local,
                   const float* d_mass, VECTOR* crd, Boundary boundary,
                   VECTOR* vel, const int need_pressure, LTMatrix3* d_stress);
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* mass_inverse, Boundary boundary,
        bool update_coordinates = true);

    int local_atom_numbers = 0;
    int num_triangle_local = 0;
    int num_pair_local = 0;
    int* d_num_triangle_local = NULL;
    int* d_num_pair_local = NULL;
    CONSTRAIN_TRIANGLE* d_triangles_local = NULL;
    CONSTRAIN_PAIR* d_pairs_local = NULL;
    VECTOR* d_delta_vel_local = NULL;

    void update_ug_connectivity(CONECT* connectivity);
    // 获得本设备中的constrain
    void Get_Local(const int* atom_local_id, const char* atom_local_label,
                   const int local_atom_numbers);
};
