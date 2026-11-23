#include "state.h"
#if defined(__linux__)
#include <algorithm>
#include <tuple>

#include "../main.h"
extern deviceStream_t main_stream;

namespace sponge_exchange
{
namespace
{
void Hash_Bytes(std::uint64_t& hash, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
}
}  // namespace

Window Make_Window(MD_INFORMATION& md, DOMAIN_INFORMATION& domain,
                   const CONSTRAIN& constraints, const SITS_INFORMATION& sits,
                   std::function<void()> update_box,
                   std::function<void()> refresh_force)
{
    static_assert(sizeof(VECTOR) == 3 * sizeof(float));
    const auto values = static_cast<std::size_t>(md.atom_numbers) * 3;
    const auto bytes = values * sizeof(float);
    std::uint64_t phase_space = 14695981039346656037ULL;
    Hash_Bytes(phase_space, md.h_mass, md.atom_numbers * sizeof(float));
    Hash_Bytes(phase_space, &md.pbc.pbc, sizeof(md.pbc.pbc));
    const bool npt = md.mode == md.NPT;
    if (!npt) Hash_Bytes(phase_space, &md.sys.box_length, sizeof(VECTOR));
    Hash_Bytes(phase_space, &md.sys.box_angle, sizeof(VECTOR));
    std::vector<std::tuple<int, int, double>> pairs;
    for (int index = 0; index < constraints.constrain_pair_numbers; ++index)
    {
        const auto& pair = constraints.h_constrain_pair[index];
        const auto atoms = std::minmax(pair.atom_i_serial, pair.atom_j_serial);
        pairs.emplace_back(atoms.first, atoms.second, pair.constant_r);
    }
    std::sort(pairs.begin(), pairs.end());
    for (const auto& [first, second, distance] : pairs)
    {
        Hash_Bytes(phase_space, &first, sizeof(first));
        Hash_Bytes(phase_space, &second, sizeof(second));
        Hash_Bytes(phase_space, &distance, sizeof(distance));
    }
    const auto& atom_order = md.output.h5_output_plan.restart.atom_order_hash;
    Hash_Bytes(phase_space, atom_order.data(), atom_order.size());

    const double temperature = md.sys.target_temperature;
    const double pressure = npt ? md.sys.target_pressure : 0.0;
    const auto length = md.sys.box_length;
    Window window{};
    window.configuration = {protocol_version,    npt,
                            md.sys.steps,        md.sys.step_limit,
                            md.atom_numbers,     md.dt,
                            temperature,         pressure,
                            length.y / length.x, length.z / length.x,
                            phase_space};
    window.energy = [&md, &domain, &sits, pressure, temperature]
    {
        double potential = domain.h_sum_ene_total;
        if (sits.is_initialized && sits.bias_strength != 0)
        {
            float bias = 0;
            deviceMemcpy(&bias, sits.classic_sits.d_bias, sizeof(bias),
                         deviceMemcpyDeviceToHost);
            potential += static_cast<double>(sits.bias_strength) * bias;
        }
        const auto box = md.sys.box_length;
        const double volume = static_cast<double>(box.x) * box.y * box.z;
        return (potential + pressure * volume) /
               (static_cast<double>(CONSTANT_kB) * temperature);
    };
    window.capture = [&md, &domain, values, bytes](std::vector<float>& data)
    {
        md.Crd_Vel_dd_to_Device(domain.crd, domain.vel, domain.atom_local_label,
                                domain.atom_local_id, main_stream);
        deviceStreamSynchronize(main_stream);
        deviceMemcpy(data.data(), md.crd, bytes, deviceMemcpyDeviceToHost);
        deviceMemcpy(data.data() + values, md.vel, bytes,
                     deviceMemcpyDeviceToHost);
        const auto box = md.sys.box_length;
        data[2 * values] = box.x;
        data[2 * values + 1] = box.y;
        data[2 * values + 2] = box.z;
    };
    window.install =
        [&md, &domain, values, bytes, npt, update_box, refresh_force](
            const std::vector<float>& data, double scale)
    {
        deviceMemcpy(md.crd, data.data(), bytes, deviceMemcpyHostToDevice);
        deviceMemcpy(md.vel, data.data() + values, bytes,
                     deviceMemcpyHostToDevice);
        if (scale != 1)
            Scale_List(reinterpret_cast<float*>(md.vel),
                       static_cast<float>(scale), values);
        md.Crd_Vel_Device_to_dd(domain.crd, domain.vel, domain.atom_local_label,
                                domain.atom_local_id, main_stream);
        deviceStreamSynchronize(main_stream);
        md.output.current_crd_synchronized_step = -1;
        if (npt)
        {
            const VECTOR box{data[2 * values], data[2 * values + 1],
                             data[2 * values + 2]};
            if (!(box.x > 2 * md.nb.cutoff && box.y > 2 * md.nb.cutoff &&
                  box.z > 2 * md.nb.cutoff))
                throw std::runtime_error(
                    "incoming replica box is too small for the cutoff");
            auto& current = md.sys.box_length;
            if (box.x != current.x || box.y != current.y || box.z != current.z)
            {
                current = box;
                update_box();
            }
        }
        refresh_force();
    };
    return window;
}
}  // namespace sponge_exchange
#endif
