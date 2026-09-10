#pragma once

#include "../angle/angle.h"
#include "../bond/bond.h"

struct UREY_BRADLEY
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    BOND bond;
    ANGLE angle;

    int Urey_Bradley_numbers = 0;

    void Initial(CONTROLLER* controller, char* module_name = NULL);
    void Urey_Bradley_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, Boundary boundary, VECTOR* frc, int need_atom_energy,
        float* atom_energy, int need_virial, LTMatrix3* atom_virial);

    // 区域分解
    //  局部函数：allocated模块，查询当前进程domain内需要计算的angle&bond序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配angle信息
    void Step_Print(CONTROLLER* controller);
};
