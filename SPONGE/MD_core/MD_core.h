#pragma once

#include "../common.h"
#include "../control.h"
#include "../io/restart_output_state.h"
#include "../utils/h5md/highfive_backend.hpp"
#include "../utils/h5md/input_assembler.hpp"
#include "../utils/h5md/input_plan.hpp"
#include "../utils/h5md/observable_h5_writer.hpp"
#include "../utils/h5md/output_plan.hpp"
#include "../utils/h5md/restart_h5_reader.hpp"
#include "../utils/h5md/restart_h5_writer.hpp"
#include "../utils/h5md/trajectory_h5_reader.hpp"
#include "../utils/h5md/trajectory_h5_writer.hpp"
#include "../utils/h5md/vds_trajectory_h5_writer.hpp"

// 普通分子模拟所涉及的大部分信息
struct MD_INFORMATION
{
    int is_initialized = 0;
    int last_modify_date = 20260216;

    // clang-format off
#include "min.h"
#include "mol.h"
#include "nb.h"
#include "nve.h"
#include "output.h"
#include "pbc.h"
#include "rerun.h"
#include "sys.h"
#include "ug.h"
    // clang-format on

    // sponge输入初始化
    void Initial(CONTROLLER* controller);

    char md_name[CHAR_LENGTH_MAX];
    int mode = 0;  // md的模式(-2：NPT重跑，-1: 最小化, 0: NVE, 1: NVT, 2: NPT)
    enum MD_MODE
    {
        RERUN = -2,
        MINIMIZATION = -1,
        NVE = 0,
        NVT = 1,
        NPT = 2
    };

    void Read_Mode(CONTROLLER* controller);  // 读取mode
    // 模拟步长，单位为(0.0488878 ps)==(1/20.455 ps)
    float dt = 0.001f * CONSTANT_TIME_CONVERTION;
    void Read_dt(CONTROLLER* controller);  // 读取dt

    // 模拟的总原子数目
    int atom_numbers = 0;
    // 不参与正常相互作用的虚原子数量
    int no_direct_interaction_virtual_atom_numbers = 0;

    // 每个原子的基本物理测量量，on host
    VECTOR* velocity = NULL;
    VECTOR* coordinate = NULL;
    VECTOR* acceleration = NULL;
    VECTOR* force = NULL;
    float* h_mass = NULL;
    float* h_mass_inverse = NULL;
    float* h_charge = NULL;

    // 每个原子的基本物理测量量，on device
    VECTOR* vel = NULL;
    VECTOR* crd = NULL;
    VECTOR* last_crd = NULL;
    VECTOR* acc = NULL;
    VECTOR* frc = NULL;
    float* d_mass = NULL;
    float* d_mass_inverse = NULL;
    float* d_charge = NULL;

    // 坐标读取处理
    void Read_Coordinate_And_Velocity(CONTROLLER* controller);
    // 读rst7文件获得坐标、速度（可选）、系统时间、盒子
    void Read_Rst7(const char* file_name, int irest, CONTROLLER controller);
    // 读 H5 restart structural state 获得坐标、速度（可选）、系统时间、盒子
    void Read_H5_Restart_Structural(
        const char* file_name, CONTROLLER* controller,
        SpongeH5InputContract::RestartLoadPolicy load_policy);
    // 读坐标文件获得坐标、速度（可选）、系统时间、盒子
    void Read_Coordinate_In_File(const char* file_name, CONTROLLER controller);
    // 读取质量
    void Read_Mass(CONTROLLER* controller);
    // 读取电荷
    void Read_Charge(CONTROLLER* controller);
    // 从python初始化电荷、质量、坐标、速度、系统时间和盒子
    void Atom_Information_Initial(std::map<std::string, SpongeTensor*>& args);

    // 每个原子的能量和维里相关
    int need_pressure = 0;
    int need_potential = 0;
    int need_kinetic = 0;
    float* h_atom_energy = NULL;
    LTMatrix3* h_atom_virial_tensor = NULL;
    float* d_atom_energy = NULL;
    LTMatrix3* d_atom_virial_tensor = NULL;
    float* d_atom_ek = NULL;
    // 为结构体中的数组变量分配存储空间
    void Atom_Information_Initial();
    // 计算力前将原子能量和维里和力归零（如果需要计算时）
    void MD_Reset_Atom_Energy_And_Virial_And_Force();
    // 总和所有进程的力，以及各进程各原子的压强和势能
    void Sum_Force_Pressure_And_Potential_If_Needed();

    // 计算总张力
    void Get_pressure(CONTROLLER* controller, float dd_atom_numbers,
                      VECTOR* dd_vel, float* dd_d_mass, LTMatrix3* dd_d_virial,
                      deviceStream_t stream);
    // 将坐标和速度放缩
    void Scale_Positions_And_Velocities(LTMatrix3 g, int scale_crd,
                                        int scale_vel, VECTOR* crd,
                                        VECTOR* vel);

    // 将frc拷贝到cpu上
    void MD_Information_Frc_Device_To_Host();

    // 将force拷贝到gpu上
    void MD_Information_Frc_Host_To_Device();

    // 将crd拷贝到cpu上
    void Crd_Vel_Device_To_Host(int forced = 0);

    // 将crd与vel从粒子进程拷贝到md_info上
    void Crd_Vel_dd_to_Device(VECTOR* dd_crd, VECTOR* dd_vel,
                              char* dd_atom_local_label, int* dd_atom_local_id,
                              deviceStream_t stream);
    // 假设坐标变化不大时，反向拷贝力到粒子进程
    void Crd_Vel_Device_to_dd(VECTOR* dd_crd, VECTOR* dd_vel,
                              char* dd_atom_local_label, int* dd_atom_local_id,
                              deviceStream_t stream);

    void Frc_dd_to_Host(VECTOR* dd_frc, char* dd_atom_local_label,
                        int* dd_atom_local_id, deviceStream_t stream);
    // 每步打印信息
    void Step_Print(CONTROLLER* controller);

    MINIMIZATION_iteration min;              // 最小化迭代
    residue_information res;                 // 残基信息
    molecule_information mol;                // 分子信息
    non_bond_information nb;                 // 非键信息
    NVE_iteration nve;                       // 微正则系综迭代
    trajectory_output output;                // 轨迹输出信息
    periodic_box_condition_information pbc;  // 周期性边界条件信息
    RERUN_information rerun;                 // 重跑迭代
    system_information sys;                  // 系统整体信息
    update_group_information ug;             // 更新组信息
};
