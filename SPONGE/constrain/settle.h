#pragma once
#include "../common.h"
#include "../control.h"
#include "constrain.h"

template <class Real>
struct ConstraintTriangle
{
    int atom_A;
    int atom_B;
    int atom_C;
    Real ra;
    Real rb;
    Real rc;
    Real rd;
    Real re;
};

template <class Real>
struct SETTLE_SOLVER
{
    using Scalar = Real;
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    SETTLE_SOLVER() = default;
    SETTLE_SOLVER(const SETTLE_SOLVER&) = delete;
    SETTLE_SOLVER& operator=(const SETTLE_SOLVER&) = delete;
    bool owns_buffers = false;
    ~SETTLE_SOLVER()
    {
        if (!owns_buffers) return;
        Free_Host_And_Device_Pointer((void**)&h_triangles,
                                     (void**)&d_triangles);
        Free_Host_And_Device_Pointer((void**)&h_pairs, (void**)&d_pairs);
        if (last_pair_AB) deviceFree(last_pair_AB);
        if (last_triangle_BA) deviceFree(last_triangle_BA);
        if (last_triangle_CA) deviceFree(last_triangle_CA);
        if (virial_tensor) deviceFree(virial_tensor);
        if (d_num_triangle_local) deviceFree(d_num_triangle_local);
        if (d_num_pair_local) deviceFree(d_num_pair_local);
        if (d_triangles_local) deviceFree(d_triangles_local);
        if (d_pairs_local) deviceFree(d_pairs_local);
        if (d_delta_vel_local) deviceFree(d_delta_vel_local);
    }
    CONSTRAIN* constrain;

    void Initial(CONTROLLER* controller, CONSTRAIN* constrain, Real* h_mass,
                 const char* module_name = NULL);

    int triangle_numbers = 0;
    ConstraintTriangle<Real>*d_triangles = NULL, *h_triangles = NULL;

    int pair_numbers = 0;
    CONSTRAIN_PAIR *d_pairs = NULL, *h_pairs = NULL;

    ConstraintVector<Real>* last_pair_AB = NULL;
    ConstraintVector<Real>* last_triangle_BA = NULL;
    ConstraintVector<Real>* last_triangle_CA = NULL;
    void Remember_Last_Coordinates(const ConstraintVector<Real>* crd,
                                   Boundary boundary);

    ConstraintTensor<Real>* virial_tensor = NULL;
    void Do_SETTLE(CONTROLLER* controller, const int* atom_local,
                   const Real* d_mass, ConstraintVector<Real>* crd,
                   Boundary boundary, ConstraintVector<Real>* vel,
                   const int need_pressure, ConstraintTensor<Real>* d_stress);
    bool Project_Velocity_To_Constraint_Manifold(
        ConstraintVector<Real>* vel, ConstraintVector<Real>* crd,
        const Real* mass_inverse, Boundary boundary,
        bool update_coordinates = true);

    int local_atom_numbers = 0;
    int num_triangle_local = 0;
    int num_pair_local = 0;
    int* d_num_triangle_local = NULL;
    int* d_num_pair_local = NULL;
    ConstraintTriangle<Real>* d_triangles_local = NULL;
    CONSTRAIN_PAIR* d_pairs_local = NULL;
    ConstraintVector<Real>* d_delta_vel_local = NULL;

    void update_ug_connectivity(CONECT* connectivity);
    // 获得本设备中的constrain
    void Get_Local(const int* atom_local_id, const char* atom_local_label,
                   const int local_atom_numbers);
};

// Float-storage MD boundary; implementation and work arrays use selected
// precision.
struct SETTLE
{
    void Initial(CONTROLLER* controller, CONSTRAIN* constrain, float* mass,
                 const char* name = nullptr);
    void Get_Local(const int* id, const char* label, int n);
    void Do_SETTLE(CONTROLLER* controller, const int* atoms, const float* mass,
                   VECTOR* crd, Boundary boundary, VECTOR* vel, int pressure,
                   LTMatrix3* stress);
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* inv, Boundary boundary,
        bool update_coordinates = true);
    void update_ug_connectivity(CONECT* connectivity);
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);

   private:
    int local_atoms = 0;
    ConstraintDispatch<SETTLE_SOLVER> backend;
};
