#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "utils/h5md/input_assembler.hpp"
#include "utils/h5md/restart_h5_reader.hpp"

struct ProbeVector
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: h5_restart_read_probe <restart.spgr.h5> "
                     "<expected_atom_count>\n";
        return 2;
    }

    SpongeH5MD::RestartH5Reader reader;
    if (!reader.Open(argv[1]))
    {
        std::cerr << reader.Last_Error() << "\n";
        return 1;
    }

    SpongeH5MD::RestartStructuralState state;
    if (!reader.Read_Structural_State(&state))
    {
        std::cerr << reader.Last_Error() << "\n";
        return 1;
    }

    const std::size_t expected_atom_count =
        static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (state.atom_count != expected_atom_count)
    {
        std::cerr << "atom_count mismatch: expected " << expected_atom_count
                  << ", got " << state.atom_count << "\n";
        return 1;
    }
    if (state.position_xyz.size() != expected_atom_count * 3)
    {
        std::cerr << "position size mismatch\n";
        return 1;
    }
    if (state.has_velocity &&
        state.velocity_xyz.size() != expected_atom_count * 3)
    {
        std::cerr << "velocity size mismatch\n";
        return 1;
    }
    if (!state.has_velocity && !state.velocity_xyz.empty())
    {
        std::cerr << "velocity payload exists without velocity metadata\n";
        return 1;
    }

    std::vector<ProbeVector> coordinates(expected_atom_count);
    std::vector<ProbeVector> velocities(expected_atom_count);
    ProbeVector box_length;
    ProbeVector box_angle;
    double start_time = 0.0;
    std::string apply_error;
    if (!SpongeH5MD::Apply_Restart_Structural_State(
            state, static_cast<int>(expected_atom_count), coordinates.data(),
            velocities.data(), &box_length, &box_angle, &start_time,
            &apply_error))
    {
        std::cerr << apply_error << "\n";
        return 1;
    }
    if (!state.has_velocity)
    {
        for (const auto& velocity : velocities)
        {
            if (velocity.x != 0.0f || velocity.y != 0.0f || velocity.z != 0.0f)
            {
                std::cerr << "missing velocity was not zero-filled\n";
                return 1;
            }
        }
    }

    std::cout << "restart structural state: atoms=" << state.atom_count
              << " step=" << state.step << " time=" << state.time
              << " velocity="
              << (state.has_velocity ? "present" : "zero-filled") << "\n";
    return 0;
}
