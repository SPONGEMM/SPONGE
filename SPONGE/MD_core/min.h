#pragma once

struct MINIMIZATION_iteration
{
    MD_INFORMATION* md_info = NULL;
    float max_move = 0.02f;
    int dynamic_dt = 1;
    int last_decrease_step = 0;
    float momentum_keep = 0;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-4f;
    float learning_rate = 3e-4f;
    void Gradient_Descent(int atom_numbers, VECTOR* crd, VECTOR* frc,
                          VECTOR* vel, const float* d_mass_inverse);
    void Scale_Force_For_Dynamic_Dt(int atom_numbers, float* d_mass_inverse,
                                    VECTOR* frc, VECTOR* vel, VECTOR* acc);
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
};
