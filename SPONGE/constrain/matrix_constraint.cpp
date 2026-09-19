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
__global__ void Matrix_Directions(int n, const CONSTRAIN_PAIR* pairs,
                                  const VECTOR* crd, Boundary boundary,
                                  VECTOR* unit, VECTOR* old)
{
    MATRIX_FOR(i, n)
    {
        const auto p = pairs[i];
        VECTOR d = Get_Displacement(crd[p.atom_i_serial], crd[p.atom_j_serial],
                                    boundary);
        old[i] = d;
        unit[i] = (1.0f / sqrtf(d * d)) * d;
    }
}
__global__ void Matrix_Coupling(int n, const int* rows, const int* columns,
                                const float* prefactor, const VECTOR* unit,
                                float* values)
{
    MATRIX_FOR(i, n)
    {
        for (int k = rows[i]; k < rows[i + 1]; ++k)
            values[k] = -prefactor[k] * (unit[i] * unit[columns[k]]);
    }
}
// mode 0: first LINCS projection; 1: exact transverse rotation correction;
// 2: CCMA diagonal Newton residual, with the inverse coupling frozen at setup.
__global__ void Matrix_Residual(int n, const CONSTRAIN_PAIR* pairs,
                                const VECTOR* crd, Boundary boundary,
                                const VECTOR* unit, const float* scale,
                                int mode, float* rhs)
{
    MATRIX_FOR(i, n)
    {
        const auto p = pairs[i];
        VECTOR d = Get_Displacement(crd[p.atom_i_serial], crd[p.atom_j_serial],
                                    boundary);
        float along = d * unit[i], target = p.constant_r, r2 = d * d;
        float error;
        if (mode == 0)
            error = along - target;
        else if (mode == 1)
        {
            float radicand = target * target - (r2 - along * along);
            if (!(radicand > 0) || !(along > 0))
            {
                radicand = 0;
            }
            error = along - sqrtf(radicand);
        }
        else
        {
            if (!(along > 1e-8f))
            {
                along = 1e-8f;
            }
            error = (r2 - target * target) / (2 * along);
        }
        rhs[i] = scale[i] * error;
    }
}
__global__ void Matrix_Product(int n, const int* rows, const int* columns,
                               const float* matrix, const float* input,
                               float* output, float* sum)
{
    MATRIX_FOR(i, n)
    {
        float value = 0;
        for (int k = rows[i]; k < rows[i + 1]; ++k)
            value += matrix[k] * input[columns[k]];
        output[i] = value;
        if (sum) sum[i] += value;
    }
}
__global__ void Matrix_Apply(int n, const int* active, const int* rows,
                             const int* incidence, const VECTOR* unit,
                             const float* scale, const float* solution,
                             const float* inv, VECTOR* crd, VECTOR* total)
{
    MATRIX_FOR(i, n)
    {
        VECTOR delta = {0, 0, 0};
        for (int k = rows[i]; k < rows[i + 1]; ++k)
        {
            int code = incidence[k], pair = abs(code) - 1;
            float sign = code > 0 ? -1.0f : 1.0f;
            delta = delta + (sign * scale[pair] * solution[pair]) * unit[pair];
        }
        delta = inv[active[i]] * delta;
        crd[active[i]] = crd[active[i]] + delta;
        total[i] = total[i] + delta;
    }
}
__global__ void Matrix_Accumulate(int n, const float* scale,
                                  const float* solution, float* total)
{
    MATRIX_FOR(i, n) { total[i] += scale[i] * solution[i]; }
}
__global__ void Matrix_Finish(int n, const int* active, const VECTOR* delta,
                              VECTOR* vel, float factor)
{
    MATRIX_FOR(i, n) { vel[active[i]] = vel[active[i]] + factor * delta[i]; }
}
__global__ void Matrix_Stress(int n, const VECTOR* old, const VECTOR* unit,
                              const float* lambda, float factor,
                              LTMatrix3* stress)
{
    MATRIX_FOR(i, n)
    {
        LTMatrix3 v =
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
template <bool is_ccma>
__global__ void Matrix_Small_Groups(
    int n, const SMALL_CONSTRAINT_GROUP* groups, const CONSTRAIN_PAIR* pairs,
    const int* rows, const int* columns, const float* values,
    const VECTOR* unit, const VECTOR* old, const float* scale, const float* inv,
    VECTOR* crd, VECTOR* vel, Boundary boundary, int lincs, int order,
    int passes, float velocity_factor, int pressure, float stress_factor,
    LTMatrix3* stress, float tolerance)
{
    MATRIX_FOR(gid, n)
    {
        const auto g = groups[gid];
        float coupling[3][3] = {}, lambda[3] = {};
        VECTOR delta_total[4] = {};
        for (int i = 0; i < g.pair_count; ++i)
            for (int k = rows[g.pairs[i]]; k < rows[g.pairs[i] + 1]; ++k)
                for (int j = 0; j < g.pair_count; ++j)
                    if (columns[k] == g.pairs[j]) coupling[i][j] = values[k];
        for (int pass = 0; pass < passes; ++pass)
        {
            float term[3] = {}, solution[3] = {}, next[3] = {};
            bool converged = true;
            for (int i = 0; i < g.pair_count; ++i)
            {
                int pi = g.pairs[i];
                auto p = pairs[pi];
                VECTOR d = Get_Displacement(crd[p.atom_i_serial],
                                            crd[p.atom_j_serial], boundary);
                float along = d * unit[pi], r2 = d * d;
                float target = p.constant_r, error;
                if (is_ccma)
                    converged = converged && Constraint_Within_Tolerance(
                                                 d, target, tolerance);
                if (lincs && pass == 0)
                    error = along - target;
                else if (lincs)
                {
                    float radicand = target * target - (r2 - along * along);
                    if (!(radicand > 0) || !(along > 0))
                    {
                        radicand = 0;
                    }
                    error = along - sqrtf(radicand);
                }
                else
                {
                    if (!(along > 1e-8f))
                    {
                        along = 1e-8f;
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
                    float value = 0;
                    for (int j = 0; j < g.pair_count; ++j)
                        value += coupling[i][j] * term[j];
                    next[i] = value;
                    solution[i] += value;
                }
                for (int i = 0; i < g.pair_count; ++i) term[i] = next[i];
            }
            for (int a = 0; a < g.atom_count; ++a)
            {
                VECTOR delta = {0, 0, 0};
                for (int i = 0; i < g.pair_count; ++i)
                {
                    int pi = g.pairs[i];
                    float sign = g.a[i] == a ? -1.0f : (g.b[i] == a ? 1.0f : 0);
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
__global__ void Matrix_Velocity_Residual(int n, const CONSTRAIN_PAIR* pairs,
                                         const VECTOR* crd, const VECTOR* vel,
                                         Boundary boundary, const VECTOR* unit,
                                         const float* scale, float* rhs,
                                         int* bad)
{
    MATRIX_FOR(i, n)
    {
        auto p = pairs[i];
        VECTOR d = Get_Displacement(crd[p.atom_i_serial], crd[p.atom_j_serial],
                                    boundary);
        VECTOR dv = vel[p.atom_i_serial] - vel[p.atom_j_serial];
        rhs[i] = scale[i] * (unit[i] * dv);
        float tol = Velocity_Constraint_Residual_Tolerance(
            d * d, vel[p.atom_i_serial], vel[p.atom_j_serial], dv, 1e-5f);
        if (!(fabsf(d * dv) <= tol)) atomicExch(bad, 1);
    }
}
__global__ void Matrix_Shift_Coordinates(int n, const int* active,
                                         const VECTOR* delta, VECTOR* crd,
                                         float factor)
{
    MATRIX_FOR(i, n) { crd[active[i]] = crd[active[i]] + factor * delta[i]; }
}
}  // namespace

void MATRIX_CONSTRAINT::Free_Local()
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
void MATRIX_CONSTRAINT::Clear()
{
    Free_Local();
    pair_index.clear();
    host_matrix.clear();
    host_scale.clear();
    is_initialized = false;
}

void MATRIX_CONSTRAINT::Initialize_Data(CONTROLLER* controller, CONSTRAIN* c,
                                        const float* mass,
                                        const VECTOR* reference,
                                        Boundary boundary)
{
    Clear();
    this->controller = controller;
    constrain = c;
    if (order < 1 || order > 64 || corrections < 0 || corrections > 16 ||
        iterations < 1 || iterations > 1000 || !(tolerance > 0) ||
        !(inverse_cutoff >= 0 && inverse_cutoff < 1) || inverse_radius < 1 ||
        inverse_radius > 8 || inverse_max_size < 2 || inverse_max_size > 512)
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "MATRIX_CONSTRAINT::Initialize_Data",
                                       "invalid LINCS/CCMA settings");
    int n = c->constrain_pair_numbers;
    if (!n) return;
    std::vector<std::vector<std::pair<int, int>>> incidence(c->atom_numbers);
    std::vector<VECTOR> direction(n);
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
                spongeErrorBadFileFormat, "MATRIX_CONSTRAINT::Initialize_Data",
                "invalid matrix constraint pair/mass");
        if (!pair_index.emplace(std::minmax(a, b), i).second)
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "MATRIX_CONSTRAINT::Initialize_Data",
                                           "duplicate matrix constraint");
        incidence[a].push_back({i, 1});
        incidence[b].push_back({i, -1});
        host_scale[i] = 1 / std::sqrt(1 / mass[a] + 1 / mass[b]);
        VECTOR d = Get_Displacement(reference[a], reference[b], boundary);
        if (!(d * d > 1e-12f) || !std::isfinite(d * d))
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "MATRIX_CONSTRAINT::Initialize_Data",
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
                    float factor = i.second * j.second * host_scale[i.first] *
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
                                "MATRIX_CONSTRAINT::Initialize_Data",
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
                                "MATRIX_CONSTRAINT::Initialize_Data",
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
                    host_matrix[center].push_back({ids[i], float(x[i])});
        }
    }
    is_initialized = true;
}

void MATRIX_CONSTRAINT::Initial(CONTROLLER* controller, CONSTRAIN* c,
                                const float* mass,
                                const VECTOR* device_reference,
                                Boundary boundary)
{
    const char* name = algorithm == Algorithm::LINCS ? "LINCS" : "CCMA";
    auto read_int = [&](const char* key, int& value)
    {
        if (controller->Command_Exist(name, key))
        {
            controller->Check_Int(name, key, "MATRIX_CONSTRAINT::Initial");
            value = atoi(controller->Command(name, key));
        }
    };
    auto read_float = [&](const char* key, float& value)
    {
        if (controller->Command_Exist(name, key))
        {
            controller->Check_Float(name, key, "MATRIX_CONSTRAINT::Initial");
            value = atof(controller->Command(name, key));
        }
    };
    if (algorithm == Algorithm::LINCS)
    {
        read_int("order", order);
        read_int("corrections", corrections);
        if (controller->Command_Exist(name, "tolerance"))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "MATRIX_CONSTRAINT::Initial",
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
                spongeErrorValueErrorCommand, "MATRIX_CONSTRAINT::Initial",
                "early_stop is no longer an input option; stopping is "
                "automatic, use tolerance");
        if (!(tolerance > 0 && tolerance < 1))
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "MATRIX_CONSTRAINT::Initial",
                "tolerance must be between zero and one");
    }
    std::vector<VECTOR> reference(c->atom_numbers);
    deviceMemcpy(reference.data(), device_reference,
                 sizeof(VECTOR) * reference.size(), deviceMemcpyDeviceToHost);
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

void MATRIX_CONSTRAINT::Get_Local(const int* device_local_to_global,
                                  int local_atoms)
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
                spongeErrorNotImplemented, "MATRIX_CONSTRAINT::Get_Local",
                "matrix constraint crosses update group ownership");
        if (global_atoms[a] < 0 || global_atoms[b] < 0 ||
            global_atoms[a] >= constrain->atom_numbers ||
            global_atoms[b] >= constrain->atom_numbers)
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "MATRIX_CONSTRAINT::Get_Local",
                "invalid local-to-global constraint mapping");
        auto it =
            pair_index.find(std::minmax(global_atoms[a], global_atoms[b]));
        if (it == pair_index.end())
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "MATRIX_CONSTRAINT::Get_Local",
                                           "unknown local constraint");
        if (local_pairs[it->second] >= 0)
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "MATRIX_CONSTRAINT::Get_Local",
                "duplicate local constraint mapping");
        global_pairs[i] = it->second;
        local_pairs[it->second] = i;
        incidence[a].push_back(i + 1);
        incidence[b].push_back(-i - 1);
    }
    std::vector<int> row{0}, col, active, arow{0}, apair;
    std::vector<float> value, scales;
    for (int i = 0; i < count; i++)
    {
        int g = global_pairs[i];
        scales.push_back(host_scale[g]);
        for (auto e : host_matrix[g])
        {
            int j = local_pairs[e.first];
            if (j < 0)
                controller->Throw_SPONGE_Error(
                    spongeErrorNotImplemented, "MATRIX_CONSTRAINT::Get_Local",
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
    std::vector<float> zeros(count, 0);
    Copy_To_Device(rhs, zeros);
    Copy_To_Device(term, zeros);
    Copy_To_Device(next, zeros);
    Copy_To_Device(solution, zeros);
    Copy_To_Device(total_lambda, zeros);
    Copy_To_Device(directions, std::vector<VECTOR>(count));
    Copy_To_Device(old_displacements, std::vector<VECTOR>(count));
    Copy_To_Device(total_delta, std::vector<VECTOR>(active_count));
    Copy_To_Device(velocity_violation, std::vector<int>(1, 0));
}

void MATRIX_CONSTRAINT::Remember_Last_Coordinates(const VECTOR* crd,
                                                  Boundary boundary)
{
    if (!count) return;
    Launch_Device_Kernel(Matrix_Directions, Blocks(count), threads, 0, 0, count,
                         constrain->constrain_pair_local, crd, boundary,
                         directions, old_displacements);
    if (algorithm == Algorithm::LINCS)
        Launch_Device_Kernel(Matrix_Coupling, Blocks(count), threads, 0, 0,
                             count, rows, columns, coefficients, directions,
                             matrix);
}
void MATRIX_CONSTRAINT::Solve()
{
    if (algorithm == Algorithm::CCMA)
    {
        Launch_Device_Kernel(Matrix_Product, Blocks(count), threads, 0, 0,
                             count, rows, columns, matrix, rhs, solution,
                             nullptr);
    }
    else
    {
        deviceMemcpy(solution, rhs, count * sizeof(float),
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(term, rhs, count * sizeof(float),
                     deviceMemcpyDeviceToDevice);
        for (int i = 0; i < order; i++)
        {
            Launch_Device_Kernel(Matrix_Product, Blocks(count), threads, 0, 0,
                                 count, rows, columns, matrix, term, next,
                                 solution);
            std::swap(term, next);
        }
    }
}
template <bool is_ccma>
void MATRIX_CONSTRAINT::Constrain_Small_Groups(VECTOR* crd, VECTOR* vel,
                                               const float* inv,
                                               Boundary boundary, int pressure,
                                               LTMatrix3* stress)
{
    float factor = constrain->dt_inverse * constrain->dt_inverse /
                   constrain->x_factor * boundary.rcell.a11 *
                   boundary.rcell.a22 * boundary.rcell.a33;
    Launch_Device_Kernel(
        Matrix_Small_Groups<is_ccma>, Blocks(small_groups.count), threads, 0, 0,
        small_groups.count, small_groups.data, constrain->constrain_pair_local,
        rows, columns, matrix, directions, old_displacements, scale, inv, crd,
        vel, boundary, algorithm == Algorithm::LINCS, order,
        algorithm == Algorithm::LINCS ? 1 + corrections : iterations,
        constrain->v_factor / constrain->x_factor * constrain->dt_inverse,
        pressure, factor, stress, tolerance);
}
void MATRIX_CONSTRAINT::Constrain(int atom_numbers, VECTOR* crd, VECTOR* vel,
                                  const float* inv, const float* mass,
                                  Boundary boundary, int pressure,
                                  LTMatrix3* stress)
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
        deviceMemset(total_delta, 0, active_count * sizeof(VECTOR));
        deviceMemset(total_lambda, 0, count * sizeof(float));
        const int passes =
            algorithm == Algorithm::LINCS ? 1 + corrections : iterations;
        for (int i = 0; i < passes; i++)
        {
            int mode = algorithm == Algorithm::CCMA ? 2 : (i == 0 ? 0 : 1);
            Launch_Device_Kernel(Matrix_Residual, Blocks(count), threads, 0, 0,
                                 count, constrain->constrain_pair_local, crd,
                                 boundary, directions, scale, mode, rhs);
            Solve();
            Launch_Device_Kernel(Matrix_Apply, Blocks(active_count), threads, 0,
                                 0, active_count, active_atoms, atom_rows,
                                 atom_pairs, directions, scale, solution, inv,
                                 crd, total_delta);
            if (pressure)
                Launch_Device_Kernel(Matrix_Accumulate, Blocks(count), threads,
                                     0, 0, count, scale, solution,
                                     total_lambda);
        }
        Launch_Device_Kernel(
            Matrix_Finish, Blocks(active_count), threads, 0, 0, active_count,
            active_atoms, total_delta, vel,
            constrain->v_factor / constrain->x_factor * constrain->dt_inverse);
        if (pressure)
        {
            float factor = constrain->dt_inverse * constrain->dt_inverse /
                           constrain->x_factor * boundary.rcell.a11 *
                           boundary.rcell.a22 * boundary.rcell.a33;
            Launch_Device_Kernel(Matrix_Stress, Blocks(count), threads, 0, 0,
                                 count, old_displacements, directions,
                                 total_lambda, factor, stress);
        }
    }
}

bool MATRIX_CONSTRAINT::Project_Velocity_To_Constraint_Manifold(
    VECTOR* vel, VECTOR* crd, const float* inv, Boundary boundary,
    int local_atoms, bool update_coordinates)
{
    if (!count) return true;
    VECTOR *saved_directions = nullptr, *saved_old = nullptr;
    Device_Malloc_Safely((void**)&saved_directions, count * sizeof(VECTOR));
    Device_Malloc_Safely((void**)&saved_old, count * sizeof(VECTOR));
    deviceMemcpy(saved_directions, directions, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(saved_old, old_displacements, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    Remember_Last_Coordinates(crd, boundary);
    deviceMemset(total_delta, 0, active_count * sizeof(VECTOR));
    bool converged = false;
    for (int i = 0; i <= 100; i++)
    {
        deviceMemset(velocity_violation, 0, sizeof(int));
        Launch_Device_Kernel(Matrix_Velocity_Residual, Blocks(count), threads,
                             0, 0, count, constrain->constrain_pair_local, crd,
                             vel, boundary, directions, scale, rhs,
                             velocity_violation);
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
        Launch_Device_Kernel(Matrix_Apply, Blocks(active_count), threads, 0, 0,
                             active_count, active_atoms, atom_rows, atom_pairs,
                             directions, scale, solution, inv, vel,
                             total_delta);
    }
    if (update_coordinates)
        Launch_Device_Kernel(Matrix_Shift_Coordinates, Blocks(active_count),
                             threads, 0, 0, active_count, active_atoms,
                             total_delta, crd, 0.5f * constrain->dt);
    deviceMemcpy(directions, saved_directions, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(old_displacements, saved_old, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    deviceFree(saved_directions);
    deviceFree(saved_old);
    if (algorithm == Algorithm::LINCS)
        Launch_Device_Kernel(Matrix_Coupling, Blocks(count), threads, 0, 0,
                             count, rows, columns, coefficients, directions,
                             matrix);
    return converged;
}
