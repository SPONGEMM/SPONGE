#pragma once
#include "../common.h"
#include "../control.h"
#include "../third_party/jit/jit.hpp"

struct PAIRWISE_FORCE
{
    char module_name[CHAR_LENGTH_MAX];
    bool is_initialized = false;
    bool is_controller_printf_initialized = false;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int type_numbers = 0;
    bool with_ele = true;

    std::string force_name;
    int n_ij_parameter = 0;
    std::vector<std::string> parameter_type;
    std::vector<std::string> parameter_name;
    std::string source_code;
    std::string ele_code;
    JIT_Function force_function;

    void** gpu_parameters;
    void** cpu_parameters;
    int* cpu_pairwise_types;
    int* gpu_pairwise_types;
    int* gpu_pairwise_types_local;
    std::vector<void*> launch_args;

    float* item_energy;
    float* sum_energy;
    float last_energy = 0.0f;
    float h_energy = 0.0f;
    int local_atom_numbers = 0;
    int total_local_numbers = 0;
    bool has_native_parameters = false;
    std::vector<float> native_parameter_values;
    std::vector<int> native_atom_types;

    void Initial(CONTROLLER* controller, const char* module_name = NULL);
    void Read_Configuration(CONTROLLER* controller);
    void JIT_Compile(CONTROLLER* controller);
    void Real_Initial(CONTROLLER* controller);
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label, int* atom_local_id);
    void Compute_Force(ATOM_GROUP* nl, const VECTOR* crd,
                       const Boundary boundary, float cutoff, float pme_beta,
                       float* charge, VECTOR* frc, int need_energy,
                       float* atom_energy, int need_virial,
                       LTMatrix3* atom_virial, float* pme_direct_atom_energy);
    float Get_Energy(ATOM_GROUP* nl, const VECTOR* crd, const Boundary boundary,
                     float cutoff, float pme_beta, float* charge,
                     float* pme_direct_atom_energy);
    void Step_Print(CONTROLLER* controller);
};
