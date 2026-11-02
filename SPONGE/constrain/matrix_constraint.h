#pragma once

#include "constrain.h"

// Shared storage/assembly, two distinct solvers. LINCS uses the instantaneous
// old-direction Neumann series; CCMA uses a frozen reference sparse inverse.
// SETTLE removes independent pairs/triangles before this object is initialized.
struct MATRIX_CONSTRAINT
{
    enum class Algorithm
    {
        LINCS,
        CCMA
    };
    Algorithm algorithm = Algorithm::LINCS;
    bool is_initialized = false;
    void Initial(CONTROLLER* controller, CONSTRAIN* constraints,
                 const float* host_mass, const VECTOR* device_reference,
                 Boundary boundary);
    void Get_Local(const int* device_local_to_global, int local_atom_numbers);
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);
    void Constrain(int atom_numbers, VECTOR* crd, VECTOR* vel,
                   const float* mass_inverse, const float* mass,
                   Boundary boundary, int need_pressure, LTMatrix3* stress);
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* mass_inverse, Boundary boundary,
        int local_atom_numbers, bool update_coordinates = true);
    void Clear();
    ~MATRIX_CONSTRAINT() { Clear(); }
    MATRIX_CONSTRAINT() = default;
    MATRIX_CONSTRAINT(const MATRIX_CONSTRAINT&) = delete;
    MATRIX_CONSTRAINT& operator=(const MATRIX_CONSTRAINT&) = delete;

   private:
    int order = 8;
    int corrections = 1;
    int iterations = 8;
    float tolerance = 1e-4f;
    bool use_small_groups = true;
    SMALL_CONSTRAINT_GROUPS small_groups;
    float inverse_cutoff = 0.01f;
    int inverse_radius = 3;
    int inverse_max_size = 128;

    CONSTRAIN* constrain = nullptr;
    // Build reference data from host coordinates before creating local maps.
    void Initialize_Data(CONTROLLER* controller, CONSTRAIN* constraints,
                         const float* host_mass, const VECTOR* host_reference,
                         Boundary boundary);
    CONTROLLER* controller = nullptr;
    std::map<std::pair<int, int>, int> pair_index;
    std::vector<std::vector<std::pair<int, float>>> host_matrix;
    std::vector<float> host_scale;
    int count = 0, active_count = 0;
    int *rows = nullptr, *columns = nullptr;
    int *active_atoms = nullptr, *atom_rows = nullptr, *atom_pairs = nullptr;
    float *coefficients = nullptr, *matrix = nullptr, *scale = nullptr;
    float *rhs = nullptr, *term = nullptr, *next = nullptr, *solution = nullptr;
    float* total_lambda = nullptr;
    VECTOR *directions = nullptr, *old_displacements = nullptr;
    VECTOR* total_delta = nullptr;
    int* velocity_violation = nullptr;
    template <bool is_ccma>
    void Constrain_Small_Groups(VECTOR* crd, VECTOR* vel, const float* inv,
                                Boundary boundary, int pressure,
                                LTMatrix3* stress);
    void Free_Local();
    void Solve();
};
