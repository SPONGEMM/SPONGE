#pragma once

#include <cstdint>

// Private, same-build input bridge. Buffers are borrowed only for the
// synchronous sink call; neither Python objects nor STL types cross the library
// boundary.
struct SPONGE_STRUCTURE_REQUEST
{
    const char* path;
    const char* format;
    int frame;
    int has_cell;
    double cell[3];
    int has_pbc;
    int pbc[3];
};

struct SPONGE_STRUCTURE_VIEW
{
    int atom_count;
    const double* positions;
    const double* masses;
    const double* velocities;
    const double* cell;
    int type_count;
    const char* const* type_names;
    const int32_t* atom_types;
};

using SPONGE_STRUCTURE_SINK = void (*)(const SPONGE_STRUCTURE_VIEW*, void*);
using SPONGE_READ_STRUCTURE = int (*)(const SPONGE_STRUCTURE_REQUEST*,
                                      SPONGE_STRUCTURE_SINK, void*);
using SPONGE_SET_STRUCTURE_TYPES = int (*)(int, const char* const*, int,
                                           const int32_t*);
