#pragma once

struct system_information
{
    struct TARGET_SCHEDULE
    {
        enum MODE
        {
            STEP = 0,
            LINEAR = 1
        };
        bool enabled = false;
        int mode = STEP;
        std::vector<int> steps;
        std::vector<float> values;
    };

    MD_INFORMATION* md_info = NULL;
    int freedom = 0;        // 体系自由度
    int steps = 0;          // 当前模拟的步数
    int step_limit = 1000;  // 需要模拟的步数
    // 时间相关信息
    double start_time = 0;    // 系统初始时间 ps
    double dt_in_ps = 0.001;  // 用于记录时间所用的dt
    double current_time = 0;  // 系统现在时间 ps
    double Get_Current_Time(bool plus_one = true);
    std::string speed_unit_name;
    float speed_time_factor = 1.0f;

    float total_mass = 0;  // 总质量 道尔顿
    VECTOR box_length;     // 模拟体系的边界大小 angstrom
    VECTOR box_angle;      // 模拟体系的角度 度

    float volume = 0;  // 体积 angstrom^3
    float Get_Volume();

    float density = 0;  // 密度 g/cm^3
    float Get_Density();

    LTMatrix3 h_virial_tensor;          // 系统总张量维里 kcal/mol
    LTMatrix3* d_virial_tensor = NULL;  // 系统总张量维里 kcal/mol

    float* d_pressure = NULL;  // 体系压强 系统单位
    float h_pressure;          // 体系压强 系统单位
    float target_pressure;     // 外界压浴压强 系统单位

    LTMatrix3* d_stress = NULL;  // 体系张力 系统单位
    LTMatrix3 h_stress;          // 体系张力 系统单位

    void Get_Potential_to_stress(CONTROLLER* controller, int atom_numbers,
                                 LTMatrix3* d_atom_virial_tensor);
    void Get_Kinetic_to_stress(CONTROLLER* controller, int atom_numbers,
                               VECTOR* vel, float* atom_mass);

    float h_potential;          // 体系势能
    float* d_potential = NULL;  // 体系势能
    float Get_Potential(int is_download = 1);

    float h_sum_of_atom_ek;          // 体系原子动能
    float* d_sum_of_atom_ek = NULL;  // 体系原子动能
    float Get_Total_Atom_Ek(int is_download = 1);

    float h_temperature;           // 体系温度 K
    float* d_temperature = NULL;   // 体系温度 K
    float target_temperature;      // 外界热浴温度 K
    float Get_Atom_Temperature();  // 自由度还有问题
    TARGET_SCHEDULE target_temperature_schedule;
    TARGET_SCHEDULE target_pressure_schedule;
    void Update_Targets_By_Schedule(int current_step);

    CONECT connectivity;               // 体系的连接性信息
    PAIR_DISTANCE connected_distance;  // 连接的原子的距离
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
};
