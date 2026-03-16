#pragma once

struct trajectory_output
{
    MD_INFORMATION* md_info = NULL;
    int current_crd_synchronized_step = 0;
    bool print_zeroth_frame = false;
    bool print_virial = false;
    float last_density = NAN;
    int write_trajectory_interval = 1000;    // 打印轨迹内容的所隔步数
    int write_mdout_interval = 1000;         // 打印能量信息的所隔步数
    int write_restart_file_interval = 1000;  // restart文件重新创建的所隔步数
    FILE* crd_traj = NULL;
    FILE* box_traj = NULL;
    char restart_name[CHAR_LENGTH_MAX];
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    void Export_Restart_File(const char* rst7_name = NULL);
    void Append_Crd_Traj_File(FILE* fp = NULL);
    void Append_Box_Traj_File(FILE* fp = NULL);
    // 20210827用于输出速度和力
    int is_frc_traj = 0, is_vel_traj = 0;
    int restart_export_count = 0;
    int max_restart_export_count = 1;
    FILE* frc_traj = NULL;
    FILE* vel_traj = NULL;
    void Append_Frc_Traj_File(FILE* fp = NULL);
    void Append_Vel_Traj_File(FILE* fp = NULL);
    bool Check_Mdout_Step();
    bool Check_Force_Step();
    bool Check_Trajectory_Step();
    bool Check_Restart_Step();
};
