#include "matrix_constraint.h"

#include "velocity_projection.h"

namespace
{
#ifdef USE_GPU
#define MATRIX_FOR(i, n)                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n)
#else
#define MATRIX_FOR(i, n) PRAGMA(omp parallel for) for (int i = 0; i < n; ++i)
#endif
constexpr int threads = 128;
int Blocks(int n) { return (n + threads - 1) / threads; }

template <class T>
void Copy_To_Device(T*& p, const std::vector<T>& x)
{
    if (p) deviceFree(p);
    p = nullptr;
    if (!x.empty())
    {
        // The existing combined allocator aliases host memory on CPU. These
        // vectors are temporary, so this module needs an owned allocation.
        Device_Malloc_Safely((void**)&p, sizeof(T) * x.size());
        deviceMemcpy(p, x.data(), sizeof(T) * x.size(),
                     deviceMemcpyHostToDevice);
    }
}
template <class T>
void Release(T*& p)
{
    if (p) deviceFree(p);
    p = nullptr;
}

// n_i points from atom b to atom a. Position correction is -M^-1 B^T lambda.
template <class Real>
__global__ void Matrix_Directions(int n, const CONSTRAIN_PAIR* pairs,
                                  const ConstraintVector<Real>* crd,
                                  Boundary boundary,
                                  ConstraintVector<Real>* unit,
                                  ConstraintVector<Real>* old)
{
    MATRIX_FOR(i, n)
    {
        const ConstraintPair<Real> p = pairs[i];
        ConstraintVector<Real> d = Get_Displacement(
            crd[p.atom_i_serial], crd[p.atom_j_serial], boundary);
        old[i] = d;
        unit[i] = (Real(1.0) / ConstraintSqrt(d * d)) * d;
    }
}
template <class Real>
__global__ void Matrix_Coupling(int n, const int* rows, const int* columns,
                                const Real* prefactor,
                                const ConstraintVector<Real>* unit,
                                Real* values)
{
    MATRIX_FOR(i, n)
    {
        for (int k = rows[i]; k < rows[i + 1]; ++k)
            values[k] = -prefactor[k] * (unit[i] * unit[columns[k]]);
    }
}
// mode 0: first LINCS projection; 1: exact transverse rotation correction;
// 2: CCMA diagonal Newton residual, with the inverse coupling frozen at setup.
template <class Real>
__global__ void Matrix_Residual(int n, const CONSTRAIN_PAIR* pairs,
                                const ConstraintVector<Real>* crd,
                                Boundary boundary,
                                const ConstraintVector<Real>* unit,
                                const Real* scale, int mode, Real* rhs)
{
    MATRIX_FOR(i, n)
    {
        const ConstraintPair<Real> p = pairs[i];
        ConstraintVector<Real> d = Get_Displacement(
            crd[p.atom_i_serial], crd[p.atom_j_serial], boundary);
        Real along = d * unit[i], target = p.constant_r, r2 = d * d;
        Real error;
        if (mode == 0)
            error = along - target;
        else if (mode == 1)
        {
            Real radicand = target * target - (r2 - along * along);
            if (!(radicand > 0) || !(along > 0))
            {
                radicand = 0;
            }
            error = along - ConstraintSqrt(radicand);
        }
        else
        {
            if (!(along > Real(1e-8)))
            {
                along = Real(1e-8);
            }
            error = (r2 - target * target) / (2 * along);
        }
        rhs[i] = scale[i] * error;
    }
}
template <class Real>
__global__ void Matrix_Product(int n, const int* rows, const int* columns,
                               const Real* matrix, const Real* input,
                               Real* output, Real* sum)
{
    MATRIX_FOR(i, n)
    {
        Real value = 0;
        for (int k = rows[i]; k < rows[i + 1]; ++k)
            value += matrix[k] * input[columns[k]];
        output[i] = value;
        if (sum) sum[i] += value;
    }
}
template <class Real>
__global__ void Matrix_Apply(int n, const int* active, const int* rows,
                             const int* incidence,
                             const ConstraintVector<Real>* unit,
                             const Real* scale, const Real* solution,
                             const Real* inv, ConstraintVector<Real>* crd,
                             ConstraintVector<Real>* total)
{
    MATRIX_FOR(i, n)
    {
        ConstraintVector<Real> delta = {0, 0, 0};
        for (int k = rows[i]; k < rows[i + 1]; ++k)
        {
            int code = incidence[k], pair = abs(code) - 1;
            Real sign = code > 0 ? -Real(1.0) : Real(1.0);
            delta = delta + (sign * scale[pair] * solution[pair]) * unit[pair];
        }
        delta = inv[active[i]] * delta;
        crd[active[i]] = crd[active[i]] + delta;
        total[i] = total[i] + delta;
    }
}
template <class Real>
__global__ void Matrix_Accumulate(int n, const Real* scale,
                                  const Real* solution, Real* total)
{
    MATRIX_FOR(i, n) { total[i] += scale[i] * solution[i]; }
}
template <class Real>
__global__ void Matrix_Finish(int n, const int* active,
                              const ConstraintVector<Real>* delta,
                              ConstraintVector<Real>* vel, Real factor)
{
    MATRIX_FOR(i, n) { vel[active[i]] = vel[active[i]] + factor * delta[i]; }
}
template <class Real>
__global__ void Matrix_Stress(int n, const ConstraintVector<Real>* old,
                              const ConstraintVector<Real>* unit,
                              const Real* lambda, Real factor,
                              ConstraintTensor<Real>* stress)
{
    MATRIX_FOR(i, n)
    {
        ConstraintTensor<Real> v =
            (-factor * lambda[i]) * Get_Virial_From_Force_Dis(unit[i], old[i]);
        atomicAdd(&stress->a11, v.a11);
        atomicAdd(&stress->a21, v.a21);
        atomicAdd(&stress->a22, v.a22);
        atomicAdd(&stress->a31, v.a31);
        atomicAdd(&stress->a32, v.a32);
        atomicAdd(&stress->a33, v.a33);
    }
}
// Each independent group owns its atoms throughout all solver passes. Keep
// the same matrix, order and residual as the general path.
// CCMA tests convergence; LINCS performs its fixed number of corrections.
template <class Real, bool is_ccma>
__global__ void Matrix_Small_Groups(
    int n, const SMALL_CONSTRAINT_GROUP* groups, const CONSTRAIN_PAIR* pairs,
    const int* rows, const int* columns, const Real* values,
    const ConstraintVector<Real>* unit, const ConstraintVector<Real>* old,
    const Real* scale, const Real* inv, ConstraintVector<Real>* crd,
    ConstraintVector<Real>* vel, Boundary boundary, int lincs, int order,
    int passes, Real velocity_factor, int pressure, Real stress_factor,
    ConstraintTensor<Real>* stress, Real tolerance)
{
    MATRIX_FOR(gid, n)
    {
        const auto g = groups[gid];
        Real coupling[3][3] = {}, lambda[3] = {};
        ConstraintVector<Real> delta_total[4] = {};
        for (int i = 0; i < g.pair_count; ++i)
            for (int k = rows[g.pairs[i]]; k < rows[g.pairs[i] + 1]; ++k)
                for (int j = 0; j < g.pair_count; ++j)
                    if (columns[k] == g.pairs[j]) coupling[i][j] = values[k];
        for (int pass = 0; pass < passes; ++pass)
        {
            Real term[3] = {}, solution[3] = {}, next[3] = {};
            bool converged = true;
            for (int i = 0; i < g.pair_count; ++i)
            {
                int pi = g.pairs[i];
                ConstraintPair<Real> p = pairs[pi];
                ConstraintVector<Real> d = Get_Displacement(
                    crd[p.atom_i_serial], crd[p.atom_j_serial], boundary);
                Real along = d * unit[pi], r2 = d * d;
                Real target = p.constant_r, error;
                if (is_ccma)
                    converged = converged && Constraint_Within_Tolerance(
                                                 d, target, tolerance);
                if (lincs && pass == 0)
                    error = along - target;
                else if (lincs)
                {
                    Real radicand = target * target - (r2 - along * along);
                    if (!(radicand > 0) || !(along > 0))
                    {
                        radicand = 0;
                    }
                    error = along - ConstraintSqrt(radicand);
                }
                else
                {
                    if (!(along > Real(1e-8)))
                    {
                        along = Real(1e-8);
                    }
                    error = (r2 - target * target) / (2 * along);
                }
                term[i] = scale[pi] * error;
                solution[i] = lincs ? term[i] : 0;
            }
            if (is_ccma && converged) break;
            for (int iteration = 0; iteration < (lincs ? order : 1);
                 ++iteration)
            {
                for (int i = 0; i < g.pair_count; ++i)
                {
                    Real value = 0;
                    for (int j = 0; j < g.pair_count; ++j)
                        value += coupling[i][j] * term[j];
                    next[i] = value;
                    solution[i] += value;
                }
                for (int i = 0; i < g.pair_count; ++i) term[i] = next[i];
            }
            for (int a = 0; a < g.atom_count; ++a)
            {
                ConstraintVector<Real> delta = {0, 0, 0};
                for (int i = 0; i < g.pair_count; ++i)
                {
                    int pi = g.pairs[i];
                    Real sign = g.a[i] == a ? -Real(1.0)
                                            : (g.b[i] == a ? Real(1.0) : 0);
                    delta = delta + (sign * scale[pi] * solution[i]) * unit[pi];
                }
                delta = inv[g.atoms[a]] * delta;
                crd[g.atoms[a]] = crd[g.atoms[a]] + delta;
                delta_total[a] = delta_total[a] + delta;
            }
            if (pressure)
                for (int i = 0; i < g.pair_count; ++i)
                    lambda[i] += scale[g.pairs[i]] * solution[i];
        }
        for (int a = 0; a < g.atom_count; ++a)
            vel[g.atoms[a]] =
                vel[g.atoms[a]] + velocity_factor * delta_total[a];
        if (pressure)
            for (int i = 0; i < g.pair_count; ++i)
                atomicAdd(stress, (-stress_factor * lambda[i]) *
                                      Get_Virial_From_Force_Dis(
                                          unit[g.pairs[i]], old[g.pairs[i]]));
    }
}
template <class Real>
__global__ void Matrix_Velocity_Residual(int n, const CONSTRAIN_PAIR* pairs,
                                         const ConstraintVector<Real>* crd,
                                         const ConstraintVector<Real>* vel,
                                         Boundary boundary,
                                         const ConstraintVector<Real>* unit,
                                         const Real* scale, Real* rhs, int* bad)
{
    MATRIX_FOR(i, n)
    {
        ConstraintPair<Real> p = pairs[i];
        ConstraintVector<Real> d = Get_Displacement(
            crd[p.atom_i_serial], crd[p.atom_j_serial], boundary);
        ConstraintVector<Real> dv = vel[p.atom_i_serial] - vel[p.atom_j_serial];
        rhs[i] = scale[i] * (unit[i] * dv);
        Real tol = Velocity_Constraint_Residual_Tolerance(
            d * d, vel[p.atom_i_serial], vel[p.atom_j_serial], dv, Real(1e-5));
        if (!(fabs(d * dv) <= tol)) atomicExch(bad, 1);
    }
}
template <class Real>
__global__ void Matrix_Shift_Coordinates(int n, const int* active,
                                         const ConstraintVector<Real>* delta,
                                         ConstraintVector<Real>* crd,
                                         Real factor)
{
    MATRIX_FOR(i, n) { crd[active[i]] = crd[active[i]] + factor * delta[i]; }
}
}  // namespace

template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Free_Local()
{
    small_groups.Clear();
    Release(rows);
    Release(columns);
    Release(coefficients);
    Release(matrix);
    Release(scale);
    Release(active_atoms);
    Release(atom_rows);
    Release(atom_pairs);
    Release(rhs);
    Release(term);
    Release(next);
    Release(solution);
    Release(total_lambda);
    Release(directions);
    Release(old_displacements);
    Release(total_delta);
    Release(velocity_violation);
    count = active_count = 0;
}
template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Clear()
{
    Free_Local();
    pair_index.clear();
    host_matrix.clear();
    host_scale.clear();
    is_initialized = false;
}

template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_Data(
    CONTROLLER* controller, CONSTRAIN* c, const Real* mass,
    const ConstraintVector<Real>* reference, Boundary boundary)
{
    Clear();
    this->controller = controller;
    constrain = c;
    if (order < 1 || order > 64 || corrections < 0 || corrections > 16 ||
        iterations < 1 || iterations > 1000 || !(tolerance > 0) ||
        !(inverse_cutoff >= 0 && inverse_cutoff < 1) || inverse_radius < 1 ||
        inverse_radius > 8 || inverse_max_size < 2 || inverse_max_size > 512)
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_Data",
            "invalid LINCS/CCMA settings");
    int n = c->constrain_pair_numbers;
    if (!n) return;
    std::vector<std::vector<std::pair<int, int>>> incidence(c->atom_numbers);
    std::vector<ConstraintVector<Real>> direction(n);
    host_scale.resize(n);
    host_matrix.resize(n);
    for (int i = 0; i < n; i++)
    {
        auto p = c->h_constrain_pair[i];
        int a = p.atom_i_serial, b = p.atom_j_serial;
        if (a < 0 || b < 0 || a >= c->atom_numbers || b >= c->atom_numbers ||
            a == b || !(mass[a] > 0) || !(mass[b] > 0) ||
            !std::isfinite(mass[a]) || !std::isfinite(mass[b]) ||
            !(p.constant_r > 0) || !std::isfinite(p.constant_r))
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_Data",
                "invalid matrix constraint pair/mass");
        if (!pair_index.emplace(std::minmax(a, b), i).second)
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_Data",
                "duplicate matrix constraint");
        incidence[a].push_back({i, 1});
        incidence[b].push_back({i, -1});
        host_scale[i] = 1 / std::sqrt(1 / mass[a] + 1 / mass[b]);
        ConstraintVector<Real> d =
            Get_Displacement(reference[a], reference[b], boundary);
        if (!(d * d > Real(1e-12)) || !std::isfinite(d * d))
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_Data",
                "coincident reference constraint atoms");
        direction[i] = (1 / std::sqrt(d * d)) * d;
    }
    // Cbar = S B M^-1 B^T S, S_ii = (1/ma + 1/mb)^(-1/2).
    std::vector<std::map<int, double>> reference_matrix(n);
    for (int i = 0; i < n; i++) reference_matrix[i][i] = 1;
    for (int atom = 0; atom < c->atom_numbers; atom++)
        for (auto i : incidence[atom])
            for (auto j : incidence[atom])
                if (i.first != j.first)
                {
                    Real factor = i.second * j.second * host_scale[i.first] *
                                  host_scale[j.first] / mass[atom];
                    host_matrix[i.first].push_back({j.first, factor});
                    reference_matrix[i.first][j.first] =
                        factor * (direction[i.first] * direction[j.first]);
                }
    if (algorithm == Algorithm::CCMA)
    {
        // Bounded local principal inverses form a sparse approximate inverse of
        // the CONSTANT reference Cbar. Small components are inverted exactly;
        // large components use graph-radius neighborhoods (no global dense
        // N^2). This is the CCMA preconditioner construction variant, not
        // per-step Newton matrix inversion. Residual iterations correct its
        // approximation.
        for (int center = 0; center < n; center++)
        {
            std::vector<int> ids{center};
            std::map<int, int> index{{center, 0}};
            size_t first = 0, last = 1;
            for (int depth = 0; depth < inverse_radius; depth++)
            {
                for (size_t k = first; k < last; k++)
                    for (auto edge : reference_matrix[ids[k]])
                    {
                        if (index.count(edge.first)) continue;
                        if (ids.size() >= size_t(inverse_max_size))
                            controller->Throw_SPONGE_Error(
                                spongeErrorValueErrorCommand,
                                "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_"
                                "Data",
                                "CCMA inverse neighborhood exceeds "
                                "inverse_max_size; reduce inverse_radius or "
                                "increase the bound");
                        index[edge.first] = ids.size();
                        ids.push_back(edge.first);
                    }
                first = last;
                last = ids.size();
            }
            const int size = ids.size();
            std::vector<double> L(size * size, 0), x(size, 0);
            for (int i = 0; i < size; i++)
                for (auto e : reference_matrix[ids[i]])
                {
                    auto it = index.find(e.first);
                    if (it != index.end()) L[i * size + it->second] = e.second;
                }
            for (int i = 0; i < size; i++)
                for (int j = 0; j <= i; j++)
                {
                    double v = L[i * size + j];
                    for (int k = 0; k < j; k++)
                        v -= L[i * size + k] * L[j * size + k];
                    if (i == j)
                    {
                        if (!(v > 1e-10))
                            controller->Throw_SPONGE_Error(
                                spongeErrorBadFileFormat,
                                "MATRIX_CONSTRAINT_SOLVER<Real>::Initialize_"
                                "Data",
                                "CCMA reference matrix is "
                                "singular/ill-conditioned");
                        L[i * size + j] = sqrt(v);
                    }
                    else
                        L[i * size + j] = v / L[j * size + j];
                }
            for (int i = 0; i < size; i++)
            {
                double v = i == 0 ? 1 : 0;
                for (int j = 0; j < i; j++) v -= L[i * size + j] * x[j];
                x[i] = v / L[i * size + i];
            }
            for (int i = size - 1; i >= 0; i--)
            {
                double v = x[i];
                for (int j = i + 1; j < size; j++) v -= L[j * size + i] * x[j];
                x[i] = v / L[i * size + i];
            }
            host_matrix[center].clear();
            for (int i = 0; i < size; i++)
                if (i == 0 || fabs(x[i]) >= inverse_cutoff)
                    host_matrix[center].push_back({ids[i], Real(x[i])});
        }
    }
    is_initialized = true;
}

template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Initial(
    CONTROLLER* controller, CONSTRAIN* c, const Real* mass,
    const ConstraintVector<Real>* device_reference, Boundary boundary)
{
    const char* name = algorithm == Algorithm::LINCS ? "LINCS" : "CCMA";
    auto read_int = [&](const char* key, int& value)
    {
        if (controller->Command_Exist(name, key))
        {
            controller->Check_Int(name, key,
                                  "MATRIX_CONSTRAINT_SOLVER<Real>::Initial");
            value = atoi(controller->Command(name, key));
        }
    };
    auto read_float = [&](const char* key, Real& value)
    {
        if (controller->Command_Exist(name, key))
        {
            controller->Check_Float(name, key,
                                    "MATRIX_CONSTRAINT_SOLVER<Real>::Initial");
            value = atof(controller->Command(name, key));
        }
    };
    if (algorithm == Algorithm::LINCS)
    {
        read_int("order", order);
        read_int("corrections", corrections);
        if (controller->Command_Exist(name, "tolerance"))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initial",
                "LINCS has no tolerance setting; use order and corrections");
    }
    else
    {
        read_int("iteration_numbers", iterations);
        read_int("inverse_radius", inverse_radius);
        read_int("inverse_max_size", inverse_max_size);
        read_float("inverse_cutoff", inverse_cutoff);
        read_float("tolerance", tolerance);
        if (controller->Command_Exist(name, "early_stop"))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initial",
                "early_stop is no longer an input option; stopping is "
                "automatic, use tolerance");
        if (!(tolerance > 0 && tolerance < 1))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Initial",
                "tolerance must be between zero and one");
    }
    std::vector<ConstraintVector<Real>> reference(c->atom_numbers);
    deviceMemcpy(reference.data(), device_reference,
                 sizeof(ConstraintVector<Real>) * reference.size(),
                 deviceMemcpyDeviceToHost);
    Initialize_Data(controller, c, mass, reference.data(), boundary);
    controller->printf("    %s constraints: %d\n", name,
                       c->constrain_pair_numbers);
    if (algorithm == Algorithm::LINCS)
        controller->printf("    LINCS order %d, corrections %d\n", order,
                           corrections);
    else
        controller->printf(
            "    CCMA small-group relative tolerance %.3g, iteration limit "
            "%d\n",
            tolerance, iterations);
}

template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local(
    const int* device_local_to_global, int local_atoms)
{
    if (!is_initialized) return;
    Free_Local();
    count = constrain->num_pair_local;
    if (!count) return;
    std::vector<int> global_atoms(local_atoms), global_pairs(count),
        local_pairs(constrain->constrain_pair_numbers, -1);
    deviceMemcpy(global_atoms.data(), device_local_to_global,
                 local_atoms * sizeof(int), deviceMemcpyDeviceToHost);
    std::vector<CONSTRAIN_PAIR> pairs(count);
    deviceMemcpy(pairs.data(), constrain->constrain_pair_local,
                 count * sizeof(CONSTRAIN_PAIR), deviceMemcpyDeviceToHost);
    std::vector<std::vector<int>> incidence(local_atoms);
    for (int i = 0; i < count; i++)
    {
        auto p = pairs[i];
        int a = p.atom_i_serial, b = p.atom_j_serial;
        if (a < 0 || b < 0 || a >= local_atoms || b >= local_atoms)
            controller->Throw_SPONGE_Error(
                spongeErrorNotImplemented,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local",
                "matrix constraint crosses update group ownership");
        if (global_atoms[a] < 0 || global_atoms[b] < 0 ||
            global_atoms[a] >= constrain->atom_numbers ||
            global_atoms[b] >= constrain->atom_numbers)
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local",
                "invalid local-to-global constraint mapping");
        auto it =
            pair_index.find(std::minmax(global_atoms[a], global_atoms[b]));
        if (it == pair_index.end())
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local",
                "unknown local constraint");
        if (local_pairs[it->second] >= 0)
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local",
                "duplicate local constraint mapping");
        global_pairs[i] = it->second;
        local_pairs[it->second] = i;
        incidence[a].push_back(i + 1);
        incidence[b].push_back(-i - 1);
    }
    std::vector<int> row{0}, col, active, arow{0}, apair;
    std::vector<Real> value, scales;
    for (int i = 0; i < count; i++)
    {
        int g = global_pairs[i];
        scales.push_back(host_scale[g]);
        for (auto e : host_matrix[g])
        {
            int j = local_pairs[e.first];
            if (j < 0)
                controller->Throw_SPONGE_Error(
                    spongeErrorNotImplemented,
                    "MATRIX_CONSTRAINT_SOLVER<Real>::Get_Local",
                    "coupled constraints split across MPI ranks");
            col.push_back(j);
            value.push_back(e.second);
        }
        row.push_back(col.size());
    }
    for (int a = 0; a < local_atoms; a++)
        if (!incidence[a].empty())
        {
            active.push_back(a);
            apair.insert(apair.end(), incidence[a].begin(), incidence[a].end());
            arow.push_back(apair.size());
        }
    if (use_small_groups) small_groups.Build(controller, pairs, local_atoms);
    active_count = active.size();
    Copy_To_Device(rows, row);
    Copy_To_Device(columns, col);
    Copy_To_Device(coefficients, value);
    Copy_To_Device(matrix, value);
    Copy_To_Device(scale, scales);
    Copy_To_Device(active_atoms, active);
    Copy_To_Device(atom_rows, arow);
    Copy_To_Device(atom_pairs, apair);
    std::vector<Real> zeros(count, 0);
    Copy_To_Device(rhs, zeros);
    Copy_To_Device(term, zeros);
    Copy_To_Device(next, zeros);
    Copy_To_Device(solution, zeros);
    Copy_To_Device(total_lambda, zeros);
    Copy_To_Device(directions, std::vector<ConstraintVector<Real>>(count));
    Copy_To_Device(old_displacements,
                   std::vector<ConstraintVector<Real>>(count));
    Copy_To_Device(total_delta,
                   std::vector<ConstraintVector<Real>>(active_count));
    Copy_To_Device(velocity_violation, std::vector<int>(1, 0));
}

template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Remember_Last_Coordinates(
    const ConstraintVector<Real>* crd, Boundary boundary)
{
    if (!count) return;
    Launch_Device_Kernel(Matrix_Directions<Real>, Blocks(count), threads, 0, 0,
                         count, constrain->constrain_pair_local, crd, boundary,
                         directions, old_displacements);
    if (algorithm == Algorithm::LINCS)
        Launch_Device_Kernel(Matrix_Coupling<Real>, Blocks(count), threads, 0,
                             0, count, rows, columns, coefficients, directions,
                             matrix);
}
template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Solve()
{
    if (algorithm == Algorithm::CCMA)
    {
        Launch_Device_Kernel(Matrix_Product<Real>, Blocks(count), threads, 0, 0,
                             count, rows, columns, matrix, rhs, solution,
                             nullptr);
    }
    else
    {
        deviceMemcpy(solution, rhs, count * sizeof(Real),
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(term, rhs, count * sizeof(Real),
                     deviceMemcpyDeviceToDevice);
        for (int i = 0; i < order; i++)
        {
            Launch_Device_Kernel(Matrix_Product<Real>, Blocks(count), threads,
                                 0, 0, count, rows, columns, matrix, term, next,
                                 solution);
            std::swap(term, next);
        }
    }
}
template <class Real>
template <bool is_ccma>
void MATRIX_CONSTRAINT_SOLVER<Real>::Constrain_Small_Groups(
    ConstraintVector<Real>* crd, ConstraintVector<Real>* vel, const Real* inv,
    Boundary boundary, int pressure, ConstraintTensor<Real>* stress)
{
    Real factor = constrain->dt_inverse * constrain->dt_inverse /
                  constrain->x_factor * boundary.rcell.a11 *
                  boundary.rcell.a22 * boundary.rcell.a33;
    Launch_Device_Kernel(
        (Matrix_Small_Groups<Real, is_ccma>), Blocks(small_groups.count),
        threads, 0, 0, small_groups.count, small_groups.data,
        constrain->constrain_pair_local, rows, columns, matrix, directions,
        old_displacements, scale, inv, crd, vel, boundary,
        algorithm == Algorithm::LINCS, order,
        algorithm == Algorithm::LINCS ? 1 + corrections : iterations,
        constrain->v_factor / constrain->x_factor * constrain->dt_inverse,
        pressure, factor, stress, tolerance);
}
template <class Real>
void MATRIX_CONSTRAINT_SOLVER<Real>::Constrain(
    int atom_numbers, ConstraintVector<Real>* crd, ConstraintVector<Real>* vel,
    const Real* inv, const Real* mass, Boundary boundary, int pressure,
    ConstraintTensor<Real>* stress)
{
    if (!count) return;
    if (use_small_groups && small_groups.count)
    {
        if (algorithm == Algorithm::CCMA)
            Constrain_Small_Groups<true>(crd, vel, inv, boundary, pressure,
                                         stress);
        else
            Constrain_Small_Groups<false>(crd, vel, inv, boundary, pressure,
                                          stress);
    }
    else
    {
        deviceMemset(total_delta, 0,
                     active_count * sizeof(ConstraintVector<Real>));
        deviceMemset(total_lambda, 0, count * sizeof(Real));
        const int passes =
            algorithm == Algorithm::LINCS ? 1 + corrections : iterations;
        for (int i = 0; i < passes; i++)
        {
            int mode = algorithm == Algorithm::CCMA ? 2 : (i == 0 ? 0 : 1);
            Launch_Device_Kernel(Matrix_Residual<Real>, Blocks(count), threads,
                                 0, 0, count, constrain->constrain_pair_local,
                                 crd, boundary, directions, scale, mode, rhs);
            Solve();
            Launch_Device_Kernel(Matrix_Apply<Real>, Blocks(active_count),
                                 threads, 0, 0, active_count, active_atoms,
                                 atom_rows, atom_pairs, directions, scale,
                                 solution, inv, crd, total_delta);
            if (pressure)
                Launch_Device_Kernel(Matrix_Accumulate<Real>, Blocks(count),
                                     threads, 0, 0, count, scale, solution,
                                     total_lambda);
        }
        Launch_Device_Kernel(
            Matrix_Finish<Real>, Blocks(active_count), threads, 0, 0,
            active_count, active_atoms, total_delta, vel,
            constrain->v_factor / constrain->x_factor * constrain->dt_inverse);
        if (pressure)
        {
            Real factor = constrain->dt_inverse * constrain->dt_inverse /
                          constrain->x_factor * boundary.rcell.a11 *
                          boundary.rcell.a22 * boundary.rcell.a33;
            Launch_Device_Kernel(Matrix_Stress<Real>, Blocks(count), threads, 0,
                                 0, count, old_displacements, directions,
                                 total_lambda, factor, stress);
        }
    }
}

template <class Real>
bool MATRIX_CONSTRAINT_SOLVER<Real>::Project_Velocity_To_Constraint_Manifold(
    ConstraintVector<Real>* vel, ConstraintVector<Real>* crd, const Real* inv,
    Boundary boundary, int local_atoms, bool update_coordinates)
{
    if (!count) return true;
    ConstraintVector<Real>*saved_directions = nullptr, *saved_old = nullptr;
    Device_Malloc_Safely((void**)&saved_directions,
                         count * sizeof(ConstraintVector<Real>));
    Device_Malloc_Safely((void**)&saved_old,
                         count * sizeof(ConstraintVector<Real>));
    deviceMemcpy(saved_directions, directions,
                 count * sizeof(ConstraintVector<Real>),
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(saved_old, old_displacements,
                 count * sizeof(ConstraintVector<Real>),
                 deviceMemcpyDeviceToDevice);
    Remember_Last_Coordinates(crd, boundary);
    deviceMemset(total_delta, 0, active_count * sizeof(ConstraintVector<Real>));
    bool converged = false;
    for (int i = 0; i <= 100; i++)
    {
        deviceMemset(velocity_violation, 0, sizeof(int));
        Launch_Device_Kernel(
            Matrix_Velocity_Residual<Real>, Blocks(count), threads, 0, 0, count,
            constrain->constrain_pair_local, crd, vel, boundary, directions,
            scale, rhs, velocity_violation);
        int bad;
        deviceMemcpy(&bad, velocity_violation, sizeof(int),
                     deviceMemcpyDeviceToHost);
        if (!bad)
        {
            converged = true;
            break;
        }
        if (i == 100) break;
        Solve();
        Launch_Device_Kernel(Matrix_Apply<Real>, Blocks(active_count), threads,
                             0, 0, active_count, active_atoms, atom_rows,
                             atom_pairs, directions, scale, solution, inv, vel,
                             total_delta);
    }
    if (update_coordinates)
        Launch_Device_Kernel(Matrix_Shift_Coordinates<Real>,
                             Blocks(active_count), threads, 0, 0, active_count,
                             active_atoms, total_delta, crd,
                             Real(0.5) * constrain->dt);
    deviceMemcpy(directions, saved_directions,
                 count * sizeof(ConstraintVector<Real>),
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(old_displacements, saved_old,
                 count * sizeof(ConstraintVector<Real>),
                 deviceMemcpyDeviceToDevice);
    deviceFree(saved_directions);
    deviceFree(saved_old);
    if (algorithm == Algorithm::LINCS)
        Launch_Device_Kernel(Matrix_Coupling<Real>, Blocks(count), threads, 0,
                             0, count, rows, columns, coefficients, directions,
                             matrix);
    return converged;
}

template struct MATRIX_CONSTRAINT_SOLVER<float>;
template struct MATRIX_CONSTRAINT_SOLVER<double>;

void MATRIX_CONSTRAINT::Initial(CONTROLLER* controller, CONSTRAIN* constrain,
                                const float* mass, const VECTOR* reference,
                                Boundary boundary)
{
    backend.Select(constrain->precision);
    backend.Visit(
        [&](auto& s, auto& work)
        {
            using Real = typename std::decay_t<decltype(s)>::Scalar;
            std::vector<Real> host_mass(mass, mass + constrain->atom_numbers);
            work.Coordinates(constrain->atom_numbers, reference);
            s.algorithm = algorithm;
            s.Initial(controller, constrain, host_mass.data(), work.q,
                      boundary);
        });
}

void MATRIX_CONSTRAINT::Get_Local(const int* id, int n)
{
    local_atoms = n;
    backend.Visit([&](auto& s, auto&) { s.Get_Local(id, n); });
}

void MATRIX_CONSTRAINT::Clear()
{
    backend.Visit(
        [&](auto& s, auto& work)
        {
            s.Clear();
            work.Clear();
        });
    local_atoms = 0;
}

void MATRIX_CONSTRAINT::Constrain(int n, VECTOR* crd, VECTOR* vel,
                                  const float* inv, const float* mass,
                                  Boundary boundary, int pressure,
                                  LTMatrix3* stress)
{
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !n) return;
            work.Begin(n, crd, vel, mass, inv);
            s.Constrain(n, work.q, work.v, work.inverse, work.mass, boundary,
                        pressure, work.stress);
            work.End(n, crd, vel, pressure ? stress : nullptr);
        });
}

bool MATRIX_CONSTRAINT::Project_Velocity_To_Constraint_Manifold(
    VECTOR* vel, VECTOR* crd, const float* inv, Boundary boundary, int n,
    bool update_coordinates)
{
    bool result = true;
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !n) return;
            work.Begin(n, crd, vel, nullptr, inv);
            result = s.Project_Velocity_To_Constraint_Manifold(
                work.v, work.q, work.inverse, boundary, n, update_coordinates);
            work.End(n, crd, vel, nullptr, update_coordinates);
        });
    return result;
}

void MATRIX_CONSTRAINT::Remember_Last_Coordinates(const VECTOR* crd,
                                                  Boundary boundary)
{
    backend.Visit(
        [&](auto& s, auto& work)
        {
            if (!s.is_initialized || !local_atoms) return;
            work.Coordinates(local_atoms, crd);
            s.Remember_Last_Coordinates(work.q, boundary);
        });
}
