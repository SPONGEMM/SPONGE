#pragma once

struct residue_information
{
    int is_initialized = 0;
    MD_INFORMATION* md_info =
        NULL;  // 指向自己主结构体的指针，以方便调用主结构体的信息
    int residue_numbers = 0;  // 模拟的总残基数目

    float* h_mass = NULL;             // 残基质量
    float* h_mass_inverse = NULL;     // 残基质量的倒数
    int* h_res_start = NULL;          // 残基起始编号
    int* h_res_end = NULL;            // 残基终止编号（实际为终止编号+1）
    float* h_momentum = NULL;         // 残基动量
    VECTOR* h_center_of_mass = NULL;  // 残基质心
    float* h_sigma_of_res_ek = NULL;  // 残基平动能求和

    float* res_ek_energy = NULL;  // 残基平动能（求温度时已乘系数）

    float* sigma_of_res_ek = NULL;    // 残基平动能求和
    int* d_res_start = NULL;          // 残基起始编号
    int* d_res_end = NULL;            // 残基终止编号（实际为终止编号+1）
    float* d_mass = NULL;             // 残基质量
    float* d_mass_inverse = NULL;     // 残基质量的倒数
    float* d_momentum = NULL;         // 残基动量
    VECTOR* d_center_of_mass = NULL;  // 残基质心
    void Residue_Crd_Map(
        VECTOR scaler);  // 将坐标质心映射到盒子中，并乘上scaler

    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    void Read_AMBER_Parm7(const char* file_name, CONTROLLER controller);
    void Split_Disconnected_By_UG_Connectivity(const CONECT* connectivity);
};  // 残基信息

struct molecule_information
{
    int is_initialized = 0;
    MD_INFORMATION* md_info =
        NULL;  // 指向自己主结构体的指针，以方便调用主结构体的信息
    int molecule_numbers = 0;  // 模拟的总分子数目

    float* h_mass = NULL;             // 分子质量
    float* h_mass_inverse = NULL;     // 分子质量的倒数
    int* h_atom_start = NULL;         // 分子起始的原子编号
    int* h_atom_end = NULL;           // 分子终止的原子编号（实际为终止编号+1）
    int* h_residue_start = NULL;      // 分子起始的残基编号
    int* h_residue_end = NULL;        // 分子终止的残基编号（实际为终止编号+1）
    VECTOR* h_center_of_mass = NULL;  // 分子质心
    std::vector<int> h_periodicity;   // 分子是否是无限长的

    int* d_atom_start = NULL;         // 分子起始的原子编号
    int* d_atom_end = NULL;           // 分子终止的原子编号（实际为终止编号+1）
    int* d_residue_start = NULL;      // 分子起始的残基编号
    int* d_residue_end = NULL;        // 分子终止的残基编号（实际为终止编号+1）
    float* d_mass = NULL;             // 分子质量
    float* d_mass_inverse = NULL;     // 分子质量的倒数
    VECTOR* d_center_of_mass = NULL;  // 分子质心
    int* d_periodicity = NULL;        // 分子是否是无限长的
    bool force_whole_output = false;  // 是否强制整分子输出映射

    void Molecule_Crd_Map(
        float scaler =
            1.0f);  // 将坐标质心映射到盒子中，且如果scaler>0则乘上scaler
    void Molecule_Crd_Map(
        VECTOR scaler);  // 将坐标质心映射到盒子中，且如果scaler>0则乘上scaler

    void Initial(CONTROLLER* controller);
};
