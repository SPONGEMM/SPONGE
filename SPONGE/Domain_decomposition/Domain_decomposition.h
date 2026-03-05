#pragma once
#include "../MD_core/MD_core.h"
#include "../common.h"
#include "../control.h"

#define MAX_NEIGHBOR_NUM 9  // 每个区域一个方向上最大的邻居数，用于负载均衡
#define MAX_RANK_NUM 16     // 最大的进程数

struct DOMAIN_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int tmp_frc_size = 0;

    // 存储局域粒子与残基数目
    int atom_numbers = 0;  // 区域内粒子数
    int res_numbers = 0;
    int ghost_numbers = 0;  // 区域附近Ghost粒子数目
    int ghost_res_numbers = 0;
    int max_atom_numbers = 0;  // 以下buffer中最大的粒子数目
    int max_res_numbers = 0;
    int update_interval = 100;

    // ---------------------- ON DEVICE ---------------------------------
    int* d_atom_numbers;  // 区域内粒子数
    int* d_res_numbers;
    int* d_ghost_numbers;  // 区域附近Ghost粒子数目
    int* d_ghost_res_numbers;

    // 存储局域粒子的信息，
    int* atom_local =
        NULL;  // 区域内粒子的全局id，共有atom_numbers+atom_ghosts个粒子编号
    char* atom_local_label =
        NULL;  // 某个粒子是否在当前区域，如果在就置1，不在就置0,
    int* atom_local_id =
        NULL;  // 某个粒子是否在当前区域，如果在，就存储其局域的编号（即在atom_local中的编号），如果不在就置为-1
    // 存储局域残基的信息,以及Ghost残基的信息
    int* res_start = NULL;
    int* res_len = NULL;

    // 区域内每个原子的基本物理测量量on device
    VECTOR* vel = NULL;  // 存储粒子速度
    VECTOR* crd = NULL;  // 存储粒子坐标
    // VECTOR *last_crd = NULL;//存储上次的粒子坐标
    VECTOR* acc = NULL;               // 存储粒子的加速度
    VECTOR* frc = NULL;               // 存储粒子的受力
    VECTOR* frc_buffer = NULL;        // 用于分发ghost力信息的buffer
    float* d_mass = NULL;             // 存储粒子的质量
    float* d_mass_inverse = NULL;     // 存储粒子质量的倒数
    VECTOR* d_center_of_mass = NULL;  // 存储局部残基质心
    float* d_charge = NULL;           // 存储粒子电荷

    // 动能与温度计算
    float* d_ek = NULL;        // 逐原子动能
    float* d_ek_local = NULL;  // 存储区域内粒子的总动能
    float* d_ek_total = NULL;  // 存储整体动能
    float h_ek_total = 0.0f;
    float temperature = 0.0f;  // 存储区域内粒子的温度

    // 若使用xccl，需要初始化stream流；在cpu环境下stream被指向int
    deviceStream_t temp_stream;

    float* d_energy = NULL;         // 存储逐原子能量
    float* d_sum_ene_local = NULL;  // 存储区域内粒子的总能量
    float* d_sum_ene_total = NULL;  // 存储整体能量
    float h_sum_ene_total = 0.0f;

    LTMatrix3* d_virial = NULL;  // 存储逐原子维里

    int* d_excluded_list_start = NULL;  // 局域粒子的排除表开端
    int* d_excluded_list = NULL;        // 局域粒子的排除表
    int* d_excluded_numbers = NULL;     // 局域粒子的排除表长度
    // ---------------------- END ON DEVICE ---------------------------------

    void Get_Atoms(
        CONTROLLER* controller,
        MD_INFORMATION* md_info);  // 获取区域内粒子信息，并将相关buffer初始化
    void Get_Ghost(CONTROLLER* controller,
                   MD_INFORMATION*
                       md_info);  // 获取Ghost区域粒子信息，主要是粒子编号，坐标
    void Get_Excluded(CONTROLLER* controller,
                      MD_INFORMATION* md_info);  // 获取区域内粒子的排除表
    void Update_Ghost(CONTROLLER* controller);  // 更新Ghost粒子信息，主要是坐标
    void Update_Ghost_Tensor(
        CONTROLLER* controller, float* atom_tensor,
        int dim_feature);  // 使用AI力场时，对ghost粒子进行通信
    void Sync_Local_Charge_From_Global(const float* global_charge);
    void Distribute_Ghost_Information(
        CONTROLLER* controller, VECTOR* frc_);  // 传递Ghost粒子信息，用于力计算
    void Reset_Force_and_Virial(MD_INFORMATION* md_info);
    void Free_Buffer();

    // 初始化与销毁stream
    void Create_Stream();
    void Destroy_Stream();

    // 存储区域分解信息
    VECTOR min_corner;  // x,y,z坐标最小的点
    VECTOR max_corner;  // x,y,z坐标最大的点

    /*
    通信分析
    neighbor_num 与 neighbor_dir 理论只需要存储在host上， 用于管理传播方向
    num_ghost_dir，num_ghost_dir_id， num_ghost_res_dir， num_ghost_res_dir_re
    会在device上计算，需要存储两套

    */
    int h_neighbor_num[6];                    // 6个方向上邻居的数量  host
    int h_neighbor_dir[6][MAX_NEIGHBOR_NUM];  // 不同方向上邻居的rank_id  host
    int h_num_ghost_dir[6];                   // 6个方向上传递的粒子数目 host
    int*
        h_num_ghost_dir_id;  // 6个方向上传递的粒子id，其中id的buffer的大小为6*max_atom_numbers
                             // host
    int h_num_ghost_dir_re[6];      // 6个方向上接收的Ghost粒子数目 host
    int h_num_ghost_res_dir[6];     // 6个方向上需要传递的残基的数量 host
    int h_num_ghost_res_dir_re[6];  // 6个方向上需要接收的残基的数量 host 内存

    int* d_num_ghost_dir;  // 6个方向上传递的粒子数目 device
    int*
        d_num_ghost_dir_id;  // 6个方向上传递的粒子id，buffer的大小为6*max_atom_numbers
    int* d_num_ghost_dir_re;      // 6个方向上接收的Ghost粒子数目 device
    int* d_num_ghost_res_dir;     // 6个方向上需要传递的残基的数量 device
    int* d_num_ghost_res_dir_re;  // 6个方向上需要接收的残基的数量 device

    // pp子通信域的rank_id
    int pp_rank;

    // rank 0存储全局分解信息
    INT_VECTOR dom_dec_split_num;         // host
    VECTOR min_corner_set[MAX_RANK_NUM];  // host
    VECTOR max_corner_set[MAX_RANK_NUM];  // host

    void Domain_Decomposition(CONTROLLER* controller,
                              MD_INFORMATION* md_info);  // 区域分解信息
    void Send_Recv_Dom_Dec(CONTROLLER* controller);  // 传送或接收区域分解的信息
    void Find_Neighbor_Domain(
        CONTROLLER* controller,
        MD_INFORMATION*
            md_info);  // 根据dom_dec_split_num和自己的坐标判断寻找近临区域的id。
    void Exchange_Particles(CONTROLLER* controller, MD_INFORMATION* md_info);

    void Get_Ek_and_Temperature(CONTROLLER* controller,
                                MD_INFORMATION* md_info);
    void Get_Potential(CONTROLLER* controller, MD_INFORMATION* md_info);
    void Update_Box(LTMatrix3 g, float dt);
    void Res_Crd_Map(LTMatrix3 g, float dt);
    int is_initialized = 0;
};
