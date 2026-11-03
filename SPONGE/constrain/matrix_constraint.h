#pragma once

#include "constrain.h"

// Shared storage/assembly, two distinct solvers. LINCS uses the instantaneous
// old-direction Neumann series; CCMA uses a frozen reference sparse inverse.
// SETTLE removes independent pairs/triangles before this object is initialized.
enum class ConstraintMatrixAlgorithm
{
    LINCS,
    CCMA
};

template <class Real>
struct MATRIX_CONSTRAINT_SOLVER
{
    using Scalar = Real;
    using Algorithm = ConstraintMatrixAlgorithm;
    Algorithm algorithm = Algorithm::LINCS;
    bool is_initialized = false;
    void Initial(CONTROLLER* controller, CONSTRAIN* constraints,
                 const Real* host_mass,
                 const ConstraintVector<Real>* device_reference,
                 Boundary boundary);
    void Get_Local(const int* device_local_to_global, int local_atom_numbers);
    void Remember_Last_Coordinates(const ConstraintVector<Real>* crd,
                                   Boundary boundary);
    void Constrain(int atom_numbers, ConstraintVector<Real>* crd,
                   ConstraintVector<Real>* vel, const Real* mass_inverse,
                   const Real* mass, Boundary boundary, int need_pressure,
                   ConstraintTensor<Real>* stress);
    bool Project_Velocity_To_Constraint_Manifold(
        ConstraintVector<Real>* vel, ConstraintVector<Real>* crd,
        const Real* mass_inverse, Boundary boundary, int local_atom_numbers,
        bool update_coordinates = true);
    void Clear();
    ~MATRIX_CONSTRAINT_SOLVER() { Clear(); }
    MATRIX_CONSTRAINT_SOLVER() = default;
    MATRIX_CONSTRAINT_SOLVER(const MATRIX_CONSTRAINT_SOLVER&) = delete;
    MATRIX_CONSTRAINT_SOLVER& operator=(const MATRIX_CONSTRAINT_SOLVER&) =
        delete;

   private:
    int order = 8;
    int corrections = 1;
    int iterations = 8;
    Real tolerance = Real(1e-4);
    bool use_small_groups = true;
    SMALL_CONSTRAINT_GROUPS small_groups;
    Real inverse_cutoff = Real(0.01);
    int inverse_radius = 3;
    int inverse_max_size = 128;

    CONSTRAIN* constrain = nullptr;
    // Build reference data from host coordinates before creating local maps.
    void Initialize_Data(CONTROLLER* controller, CONSTRAIN* constraints,
                         const Real* host_mass,
                         const ConstraintVector<Real>* host_reference,
                         Boundary boundary);
    CONTROLLER* controller = nullptr;
    std::map<std::pair<int, int>, int> pair_index;
    std::vector<std::vector<std::pair<int, Real>>> host_matrix;
    std::vector<Real> host_scale;
    int count = 0, active_count = 0;
    int *rows = nullptr, *columns = nullptr;
    int *active_atoms = nullptr, *atom_rows = nullptr, *atom_pairs = nullptr;
    Real *coefficients = nullptr, *matrix = nullptr, *scale = nullptr;
    Real *rhs = nullptr, *term = nullptr, *next = nullptr, *solution = nullptr;
    Real* total_lambda = nullptr;
    ConstraintVector<Real>*directions = nullptr, *old_displacements = nullptr;
    ConstraintVector<Real>* total_delta = nullptr;
    int* velocity_violation = nullptr;
    template <bool is_ccma>
    void Constrain_Small_Groups(ConstraintVector<Real>* crd,
                                ConstraintVector<Real>* vel, const Real* inv,
                                Boundary boundary, int pressure,
                                ConstraintTensor<Real>* stress);
    void Free_Local();
    void Solve();
};

// Float-storage MD boundary; implementation and work arrays use selected
// precision.
struct MATRIX_CONSTRAINT
{
    using Algorithm = ConstraintMatrixAlgorithm;
    Algorithm algorithm = Algorithm::LINCS;
    void Initial(CONTROLLER* controller, CONSTRAIN* constrain,
                 const float* mass, const VECTOR* reference, Boundary boundary);
    void Get_Local(const int* id, int n);
    void Clear();
    void Constrain(int n, VECTOR* crd, VECTOR* vel, const float* inv,
                   const float* mass, Boundary boundary, int pressure,
                   LTMatrix3* stress);
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* inv, Boundary boundary, int n,
        bool update_coordinates = true);
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);

   private:
    int local_atoms = 0;
    ConstraintDispatch<MATRIX_CONSTRAINT_SOLVER> backend;
};
