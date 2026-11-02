#include <cstdlib>
#include <iostream>

#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/trajectory_h5_reader.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: h5_trajectory_read_probe <trajectory.spg.h5md> "
                     "<particle_stream> <expected_atom_count>\n";
        return 2;
    }

    SpongeH5MD::TrajectoryH5Reader reader;
    if (!reader.Open(argv[1], argv[2]))
    {
        std::cerr << reader.Last_Error() << "\n";
        return 1;
    }

    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    if (!reader.Read_Metadata(&metadata))
    {
        std::cerr << reader.Last_Error() << "\n";
        return 1;
    }

    const std::size_t expected_atom_count =
        static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));
    if (metadata.atom_count != expected_atom_count)
    {
        std::cerr << "atom_count mismatch: expected " << expected_atom_count
                  << ", got " << metadata.atom_count << "\n";
        return 1;
    }
    if (metadata.frame_count == 0)
    {
        std::cerr << "trajectory has no frames\n";
        return 1;
    }

    SpongeH5MD::RestartStructuralState frame;
    if (!reader.Read_Frame(0, &frame))
    {
        std::cerr << reader.Last_Error() << "\n";
        return 1;
    }
    if (frame.atom_count != expected_atom_count ||
        frame.position_xyz.size() != expected_atom_count * 3)
    {
        std::cerr << "frame structural state size mismatch\n";
        return 1;
    }

    std::cout << "trajectory frames=" << metadata.frame_count
              << " atoms=" << metadata.atom_count << "\n";
    return 0;
}
