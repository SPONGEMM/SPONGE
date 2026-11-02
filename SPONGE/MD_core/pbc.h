#pragma once

struct periodic_box_condition_information
{
    MD_INFORMATION* md_info = NULL;
    bool pbc = false;
    Boundary boundary;

    // 初始化时的盒子，用于判断是否变化合适
    LTMatrix3 cell0;
    // 盒子的质量倒数
    LTMatrix3 cell_mass_inverse;
    // 盒子的动量
    LTMatrix3 cell_momentum;

    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    void No_PBC_Check(CONTROLLER* controller);
    void PBC_Check();
    LTMatrix3 Get_Cell(VECTOR box_length, VECTOR box_angle);
    void Update_Box(LTMatrix3 g);
    bool Check_Change_Large();
};
