/*
 * Copyright 2021-2023 Gao's lab, Peking University, CCME. All rights reserved.
 *
 * This file is part of SPONGE and is licensed under the SPONGE Licensing
 * Agreement in the repository root. The English text in LICENSE controls.
 */

#pragma once
#include "../common.h"
#include "../control.h"
#include "../third_party/jit/jit.hpp"

struct LISTED_FORCE
{
    char module_name[CHAR_LENGTH_MAX];
    std::vector<std::string> atom_labels;
    std::vector<std::string> parameter_type;
    std::vector<std::string> parameter_name;
    std::vector<int> parameter_is_atom;
    std::vector<int> parameter_is_int;
    std::string source_code;
    std::string connected_atoms;
    std::string constrain_distance;
    JIT_Function force_function;
    int item_numbers;
    void** gpu_parameters;
    void** gpu_parameters_local;
    void** cpu_parameters;
    void** d_gpu_parameters;
    void** d_gpu_parameters_local;
    int* d_parameter_is_atom;
    int* d_parameter_is_int;
    int* d_local_item_numbers;
    std::vector<void*> launch_args;
    float* item_energy;
    float* sum_energy;
    float last_energy = 0.0f;
    float h_energy = 0.0f;
    int local_item_numbers = 0;
    int local_atom_numbers = 0;
    int use_domain_decomposition = 0;
    int last_atom_numbers = 0;
    bool has_native_parameters = false;
    int native_item_count = 0;
    std::vector<float> native_parameter_values;
    void Initialize_Parameters(CONTROLLER* controller,
                               std::string parameter_string);
    void Compile(CONTROLLER* controller);
    void Initial(CONTROLLER* controller, CONECT* connectivity,
                 PAIR_DISTANCE* con_dis);
    void Compute_Force(int atom_numbers, VECTOR* crd, Boundary boundary,
                       VECTOR* frc, int need_energy, float* atom_energy,
                       int need_pressure, LTMatrix3* atom_virial);
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label, int* atom_local_id);
    float Get_Energy(VECTOR* crd, Boundary boundary);
    void Step_Print(CONTROLLER* controller);
};

struct LISTED_FORCES
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    std::vector<LISTED_FORCE*> forces;
    void Initial(CONTROLLER* controller, CONECT* connectivity,
                 PAIR_DISTANCE* con_dis, const char* module_name = NULL);
    void Compute_Force(int atom_numbers, VECTOR* crd, Boundary boundary,
                       VECTOR* frc, int need_energy, float* atom_energy,
                       int need_pressure, LTMatrix3* atom_virial);
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label, int* atom_local_id);
    void Step_Print(CONTROLLER* controller);
};
