#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

using StringMap = std::map<std::string, std::string>;

constexpr int SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION = 20260216;
constexpr size_t SPONGE_PRIPS_CONTROLLER_LAST_MODIFY_DATE_OFFSET = 524;
constexpr size_t SPONGE_PRIPS_MD_INFO_LAST_MODIFY_DATE_OFFSET = 4;
constexpr int CHAR_LENGTH_MAX = 512;

struct VECTOR
{
    float x;
    float y;
    float z;
};

struct ATOM_GROUP
{
    int atom_numbers = 0;
    int ghost_numbers = 0;
    int* atom_serial = NULL;
};

struct CONTROLLER
{
    int _MPI_rank;
    char _pad0[584 - sizeof(int)];
    StringMap commands;
    char _pad1[840 - 584 - sizeof(StringMap)];
    FILE* mdinfo = NULL;
};

struct MD_INFORMATION
{
    struct system_information
    {
        char _pad0[12];
        int steps = 0;
    };

    char _pad0[528];
    int atom_numbers = 0;
    char _pad1[600 - 528 - sizeof(int)];
    VECTOR* crd = NULL;
    char _pad2[624 - 600 - sizeof(VECTOR*)];
    VECTOR* frc = NULL;
    char _pad3[1960 - 624 - sizeof(VECTOR*)];
    system_information sys;
};

struct NEIGHBOR_LIST
{
    char _pad0[80];
    ATOM_GROUP* h_nl = NULL;
    char _pad1[96 - 80 - sizeof(ATOM_GROUP*)];
    int max_neighbor_numbers = 0;
};

struct DOMAIN_INFORMATION
{
    char _pad0[516];
    int atom_numbers = 0;
    int res_numbers = 0;
    int ghost_numbers = 0;
    int ghost_res_numbers = 0;
    int max_atom_numbers = 0;
    char _pad1[576 - 516 - 5 * sizeof(int)];
    int* atom_local = NULL;
    char* atom_local_label = NULL;
    int* atom_local_id = NULL;
    int* res_start = NULL;
    int* res_len = NULL;
    VECTOR* vel = NULL;
    VECTOR* crd = NULL;
    VECTOR* acc = NULL;
    VECTOR* frc = NULL;
    char _pad2[1200 - 640 - sizeof(VECTOR*)];
    int pp_rank = 0;
    char _pad3[1600 - 1200 - sizeof(int)];
    int is_initialized = 0;
};

static_assert(offsetof(CONTROLLER, commands) == 584);
static_assert(offsetof(CONTROLLER, mdinfo) == 840);
static_assert(offsetof(MD_INFORMATION, atom_numbers) == 528);
static_assert(offsetof(MD_INFORMATION, crd) == 600);
static_assert(offsetof(MD_INFORMATION, frc) == 624);
static_assert(offsetof(MD_INFORMATION, sys) == 1960);
static_assert(offsetof(MD_INFORMATION::system_information, steps) == 12);
static_assert(offsetof(NEIGHBOR_LIST, h_nl) == 80);
static_assert(offsetof(NEIGHBOR_LIST, max_neighbor_numbers) == 96);
static_assert(offsetof(DOMAIN_INFORMATION, atom_numbers) == 516);
static_assert(offsetof(DOMAIN_INFORMATION, ghost_numbers) == 524);
static_assert(offsetof(DOMAIN_INFORMATION, max_atom_numbers) == 532);
static_assert(offsetof(DOMAIN_INFORMATION, atom_local) == 576);
static_assert(offsetof(DOMAIN_INFORMATION, atom_local_label) == 584);
static_assert(offsetof(DOMAIN_INFORMATION, atom_local_id) == 592);
static_assert(offsetof(DOMAIN_INFORMATION, crd) == 624);
static_assert(offsetof(DOMAIN_INFORMATION, frc) == 640);
static_assert(offsetof(DOMAIN_INFORMATION, pp_rank) == 1200);
static_assert(offsetof(DOMAIN_INFORMATION, is_initialized) == 1600);
