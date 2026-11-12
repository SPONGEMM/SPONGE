// SPDX-License-Identifier: GPL-2.0-or-later
// Reference-only temperature compute for matched SPC/E NHC+SHAKE validation.
// Independent per-water 3x3 solves; no SPONGE integrator or projector is
// linked.
#include <array>
#include <cmath>
#include <vector>

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lammpsplugin.h"
#include "update.h"
#include "version.h"

using namespace LAMMPS_NS;
namespace
{
using V = std::array<double, 3>;
class WaterTangentTemperature : public Compute
{
    std::vector<V> retained_, thermal_;
    std::vector<double> masses_;
    std::vector<unsigned char> bias_active_;
    int removed_ = 0;

    std::array<double, 3> solve(double a[3][3], std::array<double, 3> b)
    {
        for (int k = 0; k < 3; ++k)
        {
            int pivot = k;
            for (int j = k + 1; j < 3; ++j)
                if (std::abs(a[j][k]) > std::abs(a[pivot][k])) pivot = j;
            if (std::abs(a[pivot][k]) < 1e-14)
                error->all(FLERR, "Degenerate water constraint metric");
            for (int j = 0; j < 3; ++j) std::swap(a[k][j], a[pivot][j]);
            std::swap(b[k], b[pivot]);
            const double d = a[k][k];
            for (int j = k; j < 3; ++j) a[k][j] /= d;
            b[k] /= d;
            for (int i = 0; i < 3; ++i)
                if (i != k)
                {
                    const double c = a[i][k];
                    for (int j = k; j < 3; ++j) a[i][j] -= c * a[k][j];
                    b[i] -= c * b[k];
                }
        }
        return b;
    }

    void rebuild()
    {
        if (removed_)
            error->all(
                FLERR,
                "Cannot rebuild water bias while velocities are bias-removed");
        const int n = atom->nlocal;
        if (comm->nprocs != 1 || domain->triclinic || igroup != 0 ||
            n != atom->natoms || n < 3 || n % 3 || !atom->map_style)
            error->all(FLERR,
                       "Water tangent compute requires one rank, all atoms, "
                       "orthogonal periodic water");
        if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
            error->all(FLERR,
                       "Water tangent compute requires three periodic axes");
        retained_.assign(n, V{});
        thermal_.assign(n, V{});
        masses_.resize(n);
        bias_active_.assign(n, 0);
        std::vector<V> mode(n), projected(n);
        const double lengths[3] = {domain->xprd, domain->yprd, domain->zprd};
        const double wave = 2 * std::acos(-1.0) / lengths[2];
        for (int i = 0; i < n; ++i)
        {
            masses_[i] =
                atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
            if (!(masses_[i] > 0)) error->all(FLERR, "Nonpositive water mass");
            projected[i] = {atom->v[i][0], atom->v[i][1], atom->v[i][2]};
            mode[i] = {std::cos(wave * (atom->x[i][2] - domain->boxlo[2])), 0,
                       0};
        }
        for (int first = 1; first <= n; first += 3)
        {
            int id[3] = {atom->map(first), atom->map(first + 1),
                         atom->map(first + 2)};
            for (int i : id)
                if (i < 0 || i >= n)
                    error->all(FLERR,
                               "Water IDs must be consecutive O,H,H triplets");
            if (!(masses_[id[0]] > masses_[id[1]] &&
                  masses_[id[1]] == masses_[id[2]]))
                error->all(FLERR, "Water mass/order mismatch");
            double jac[3][9] = {};
            const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
            for (int row = 0; row < 3; ++row)
            {
                const int a = pairs[row][0], b = pairs[row][1];
                V d{};
                double norm = 0;
                for (int j = 0; j < 3; ++j)
                {
                    d[j] = atom->x[id[a]][j] - atom->x[id[b]][j];
                    d[j] -= lengths[j] * std::floor(d[j] / lengths[j] + .5);
                    norm += d[j] * d[j];
                }
                if (!(norm > 0)) error->all(FLERR, "Coincident water sites");
                for (int j = 0; j < 3; ++j)
                {
                    jac[row][3 * a + j] = d[j] / std::sqrt(norm);
                    jac[row][3 * b + j] = -jac[row][3 * a + j];
                }
            }
            for (auto* values : {&projected, &mode})
            {
                double metric[3][3] = {};
                std::array<double, 3> rhs{};
                for (int a = 0; a < 3; ++a)
                    for (int k = 0; k < 9; ++k)
                    {
                        rhs[a] += jac[a][k] * (*values)[id[k / 3]][k % 3];
                        for (int b = 0; b < 3; ++b)
                            metric[a][b] +=
                                jac[a][k] * jac[b][k] / masses_[id[k / 3]];
                    }
                const auto lambda = solve(metric, rhs);
                for (int k = 0; k < 9; ++k)
                    for (int a = 0; a < 3; ++a)
                        (*values)[id[k / 3]][k % 3] -=
                            jac[a][k] * lambda[a] / masses_[id[k / 3]];
            }
        }
        double mass = 0;
        V mean{}, mode_mean{};
        for (int i = 0; i < n; ++i)
        {
            mass += masses_[i];
            for (int j = 0; j < 3; ++j)
            {
                mean[j] += masses_[i] * projected[i][j];
                mode_mean[j] += masses_[i] * mode[i][j];
            }
        }
        for (int j = 0; j < 3; ++j)
        {
            mean[j] /= mass;
            mode_mean[j] /= mass;
        }
        double numerator = 0, denominator = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j)
            {
                mode[i][j] -= mode_mean[j];
                numerator +=
                    masses_[i] * (projected[i][j] - mean[j]) * mode[i][j];
                denominator += masses_[i] * mode[i][j] * mode[i][j];
            }
        if (!(denominator > 1e-12 * mass))
            error->all(FLERR, "Degenerate tangent cosine basis");
        const double coefficient = numerator / denominator;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j)
            {
                thermal_[i][j] =
                    projected[i][j] - mean[j] - coefficient * mode[i][j];
                retained_[i][j] = atom->v[i][j] - thermal_[i][j];
            }
        dof = 2 * n - 4;  // three independent distance constraints per water
    }

   public:
    WaterTangentTemperature(LAMMPS* lmp, int narg, char** arg)
        : Compute(lmp, narg, arg)
    {
        if (narg != 3)
            error->all(FLERR, "Water tangent temperature takes no options");
        scalar_flag = vector_flag = tempflag = tempbias = 1;
        size_vector = 6;
        extscalar = 0;
        extvector = 1;
        vector = new double[6];
    }
    ~WaterTangentTemperature() override { delete[] vector; }
    void init() override { rebuild(); }
    void setup() override { rebuild(); }
    double compute_scalar() override
    {
        rebuild();
        invoked_scalar = update->ntimestep;
        double energy2 = 0;
        for (int i = 0; i < atom->nlocal; ++i)
            for (double v : thermal_[i]) energy2 += masses_[i] * v * v;
        scalar = energy2 * force->mvv2e / (dof * force->boltz);
        return scalar;
    }
    void compute_vector() override
    {
        rebuild();
        invoked_vector = update->ntimestep;
        for (int j = 0; j < 6; ++j) vector[j] = 0;
        const int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
        for (int i = 0; i < atom->nlocal; ++i)
            for (int j = 0; j < 6; ++j)
                vector[j] += masses_[i] * thermal_[i][a[j]] *
                             thermal_[i][b[j]] * force->mvv2e;
    }
    void remove_bias(int i, double* v) override
    {
        if (bias_active_[i]) error->all(FLERR, "Water bias already removed");
        bias_active_[i] = 1;
        ++removed_;
        for (int j = 0; j < 3; ++j) v[j] -= retained_[i][j];
    }
    void restore_bias(int i, double* v) override
    {
        if (!bias_active_[i])
            error->all(FLERR, "Water bias restore without remove");
        for (int j = 0; j < 3; ++j) v[j] += retained_[i][j];
        bias_active_[i] = 0;
        --removed_;
    }
    void remove_bias_all() override
    {
        for (int i = 0; i < atom->nlocal; ++i) remove_bias(i, atom->v[i]);
    }
    void restore_bias_all() override
    {
        for (int i = 0; i < atom->nlocal; ++i) restore_bias(i, atom->v[i]);
    }
};

void* create(void* lmp, int narg, char** arg)
{
    return new WaterTangentTemperature(static_cast<LAMMPS*>(lmp), narg, arg);
}
}  // namespace

extern "C" void lammpsplugin_init(void* lmp, void* handle, void* regfunc)
{
    lammpsplugin_t plugin{};
    plugin.version = LAMMPS_VERSION;
    plugin.style = "compute";
    plugin.name = "temp/water/cos/tangent";
    plugin.info = "Joint tangent COM+cosine water temperature (benchmark only)";
    plugin.author = "SPONGE validation";
    plugin.creator.v2 = create;
    plugin.handle = handle;
    reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&plugin, lmp);
}
