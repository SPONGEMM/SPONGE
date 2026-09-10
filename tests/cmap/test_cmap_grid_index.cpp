#include <iostream>

#include "cmap/cmap.h"

// This standalone test does not link control.cpp. CUDA retains the host
// controller printing path, which needs the serial rank definition.
int CONTROLLER::MPI_rank = 0;

struct GridIndexCase
{
    int resolution;
    int grid_cell;
    int expected_index;
};

int main()
{
    const GridIndexCase cases[] = {
        {2, -2, 1},   {2, -1, 0},  {2, 0, 1},    {2, 1, 0},
        {3, -2, 2},   {3, -1, 0},  {3, 0, 1},    {3, 2, 0},
        {24, -12, 0}, {24, 0, 12}, {24, 11, 23}, {24, 12, 0},
        {25, -12, 0}, {25, 0, 12}, {25, 12, 24}, {25, 13, 0},
    };

    for (const auto& test_case : cases)
    {
        const int actual =
            CMAP_Periodic_Grid_Index(test_case.grid_cell, test_case.resolution);
        if (actual != test_case.expected_index)
        {
            std::cerr << "CMAP grid index mismatch for resolution="
                      << test_case.resolution
                      << ", cell=" << test_case.grid_cell
                      << ": expected=" << test_case.expected_index
                      << ", actual=" << actual << "\n";
            return 1;
        }
        if (actual < 0 || actual >= test_case.resolution)
        {
            std::cerr << "CMAP grid index is out of range for resolution="
                      << test_case.resolution
                      << ", cell=" << test_case.grid_cell << ": " << actual
                      << "\n";
            return 1;
        }
    }
    return 0;
}
