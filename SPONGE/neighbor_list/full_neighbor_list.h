#ifndef FULL_NEIGHBOR_LIST_H
#define FULL_NEIGHBOR_LIST_H
#include "../common.h"
#include "../control.h"

struct FULL_NEIGHBOR_LIST
{
    bool is_initialized = false;
    int atom_numbers = 0;
    int max_neighbor_numbers = 0;

    ATOM_GROUP* d_nl = NULL;
    ATOM_GROUP* h_nl = NULL;
    int* d_temp = NULL;
    int* d_overflow = NULL;

    void Initial(int atom_numbers, int max_neighbor_numbers);

    void Build_From_Half(const ATOM_GROUP* half_nl, int atom_numbers);

    void Build_From_Half_With_Cutoff(const ATOM_GROUP* half_nl,
                                     int atom_numbers, const VECTOR* crd,
                                     const Boundary boundary, float cutoff);

    void Clear();
};
#endif
