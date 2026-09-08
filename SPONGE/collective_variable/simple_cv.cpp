#include "simple_cv.h"

REGISTER_CV_STRUCTURE(CV_POSITION, "position_x", 0);
REGISTER_CV_STRUCTURE(CV_POSITION, "position_y", 1);
REGISTER_CV_STRUCTURE(CV_POSITION, "position_z", 2);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_x", 3);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_y", 4);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_z", 5);

static __global__ void position_x_get_all(const int atom, const VECTOR* crd,
                                          const Boundary boundary, float* value,
                                          VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    value[0] = this_crd.x;
    VECTOR this_frc = {1.0f, 0.0f, 0.0f};
    crd_grads[atom] = this_frc;
    virial[0] = Get_Virial_From_Force_Dis(this_crd, this_frc);
}

static __global__ void position_y_get_all(const int atom, const VECTOR* crd,
                                          const Boundary boundary, float* value,
                                          VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    value[0] = this_crd.y;
    VECTOR this_frc = {0.0f, 1.0f, 0.0f};
    crd_grads[atom] = this_frc;
    virial[0] = Get_Virial_From_Force_Dis(this_crd, this_frc);
}

static __global__ void position_z_get_all(const int atom, const VECTOR* crd,
                                          const Boundary boundary, float* value,
                                          VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    value[0] = this_crd.z;
    VECTOR this_frc = {0.0f, 0.0f, 1.0f};
    crd_grads[atom] = this_frc;
    virial[0] = Get_Virial_From_Force_Dis(this_crd, this_frc);
}

static __global__ void scaled_position_x_get_all(
    const int atom, const VECTOR* crd, const Boundary boundary, float* value,
    VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    this_crd = this_crd * boundary.rcell;
    value[0] = this_crd.x;
    crd_grads[atom].x = boundary.rcell.a11;
}

static __global__ void scaled_position_y_get_all(
    const int atom, const VECTOR* crd, const Boundary boundary, float* value,
    VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    this_crd = this_crd * boundary.rcell;
    value[0] = this_crd.y;
    crd_grads[atom].x = boundary.rcell.a21;
    crd_grads[atom].y = boundary.rcell.a22;
}

static __global__ void scaled_position_z_get_all(
    const int atom, const VECTOR* crd, const Boundary boundary, float* value,
    VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR this_crd = crd[atom];
    this_crd = this_crd * boundary.rcell;
    value[0] = this_crd.z;
    crd_grads[atom].x = boundary.rcell.a31;
    crd_grads[atom].y = boundary.rcell.a32;
    crd_grads[atom].z = boundary.rcell.a33;
}

void CV_POSITION::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                          int atom_numbers, const char* module_name)
{
    atom = manager->Ask_For_Int_Parameter(module_name, "atom", 1, 2);
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_POSITION::Compute(int atom_numbers, VECTOR* crd,
                          const Boundary boundary, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need != CV_NEED_NONE)
    {
        if (strcmp(type_name, "position_x") == 0)
        {
            Launch_Device_Kernel(position_x_get_all, 1, 1, 0, NULL, *atom, crd,
                                 boundary, d_value, crd_grads, virial);
        }
        else if (strcmp(type_name, "position_y") == 0)
        {
            Launch_Device_Kernel(position_y_get_all, 1, 1, 0, NULL, *atom, crd,
                                 boundary, d_value, crd_grads, virial);
        }
        else if (strcmp(type_name, "position_z") == 0)
        {
            Launch_Device_Kernel(position_z_get_all, 1, 1, 0, NULL, *atom, crd,
                                 boundary, d_value, crd_grads, virial);
        }
        else if (strcmp(type_name, "scaled_position_x") == 0)
        {
            Launch_Device_Kernel(scaled_position_x_get_all, 1, 1, 0, NULL,
                                 *atom, crd, boundary, d_value, crd_grads,
                                 virial);
        }
        else if (strcmp(type_name, "scaled_position_y") == 0)
        {
            Launch_Device_Kernel(scaled_position_y_get_all, 1, 1, 0, NULL,
                                 *atom, crd, boundary, d_value, crd_grads,
                                 virial);
        }
        else if (strcmp(type_name, "scaled_position_z") == 0)
        {
            Launch_Device_Kernel(scaled_position_z_get_all, 1, 1, 0, NULL,
                                 *atom, crd, boundary, d_value, crd_grads,
                                 virial);
        }
        deviceMemcpy(&value, d_value, sizeof(float), deviceMemcpyDeviceToHost);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_x", 0);
REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_y", 1);
REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_z", 2);

static __global__ void box_length_x_get_all(const LTMatrix3 cell, float* value,
                                            LTMatrix3* virial)
{
    value[0] = cell.a11;
    virial[0].a11 = 1;
}

static __global__ void box_length_y_get_all(const LTMatrix3 cell, float* value,
                                            LTMatrix3* virial)
{
    float l = sqrtf(cell.a21 * cell.a21 + cell.a22 * cell.a22);
    value[0] = l;
    virial[0].a21 = cell.a21 / l;
    virial[0].a22 = cell.a22 / l;
}

static __global__ void box_length_z_get_all(const LTMatrix3 cell, float* value,
                                            LTMatrix3* virial)
{
    float l =
        sqrtf(cell.a31 * cell.a31 + cell.a32 * cell.a32 + cell.a33 * cell.a33);
    value[0] = l;
    virial[0].a31 = cell.a31 / l;
    virial[0].a32 = cell.a32 / l;
    virial[0].a33 = cell.a33 / l;
}

void CV_BOX_LENGTH::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                            int atom_numbers, const char* module_name)
{
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_BOX_LENGTH::Compute(int atom_numbers, VECTOR* crd,
                            const Boundary boundary, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need != CV_NEED_NONE)
    {
        if (strcmp(type_name, "box_length_x") == 0)
        {
            Launch_Device_Kernel(box_length_x_get_all, 1, 1, 0, NULL,
                                 boundary.cell, d_value, virial);
            value = boundary.cell.a11;
        }
        else if (strcmp(type_name, "box_length_y") == 0)
        {
            Launch_Device_Kernel(box_length_y_get_all, 1, 1, 0, NULL,
                                 boundary.cell, d_value, virial);
            value = sqrtf(boundary.cell.a21 * boundary.cell.a21 +
                          boundary.cell.a22 * boundary.cell.a22);
        }
        else if (strcmp(type_name, "box_length_z") == 0)
        {
            Launch_Device_Kernel(box_length_z_get_all, 1, 1, 0, NULL,
                                 boundary.cell, d_value, virial);
            value = sqrtf(boundary.cell.a31 * boundary.cell.a31 +
                          boundary.cell.a32 * boundary.cell.a32 +
                          boundary.cell.a33 * boundary.cell.a33);
        }
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_DISTANCE, "distance", 0);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_x", 1);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_y", 2);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_z", 3);

static __global__ void distance_get_all(const int atom0, const int atom1,
                                        const VECTOR* crd,
                                        const Boundary boundary, float* value,
                                        VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR dr = Get_Displacement<BoundaryPolicy::Periodic>(
        crd[atom1], crd[atom0], boundary);
    float dr_abs = norm3df(dr.x, dr.y, dr.z);
    float dr_1 = 1.0f / dr_abs;
    VECTOR drdx = dr_1 * dr;
    value[0] = dr_abs;
    crd_grads[atom1] = drdx;
    crd_grads[atom0] = -drdx;
    virial[0] = Get_Virial_From_Force_Dis(dr, drdx);
}

static __global__ void displacement_x_get_all(const int atom0, const int atom1,
                                              const VECTOR* crd,
                                              const Boundary boundary,
                                              float* value, VECTOR* crd_grads,
                                              LTMatrix3* virial)
{
    VECTOR dr = Get_Displacement<BoundaryPolicy::Periodic>(
        crd[atom1], crd[atom0], boundary);
    value[0] = dr.x;
    VECTOR drdx = {1.0f, 0.0f, 0.0f};
    crd_grads[atom1] = drdx;
    crd_grads[atom0] = -drdx;
    virial[0] = Get_Virial_From_Force_Dis(dr, drdx);
}

static __global__ void displacement_y_get_all(const int atom0, const int atom1,
                                              const VECTOR* crd,
                                              const Boundary boundary,
                                              float* value, VECTOR* crd_grads,
                                              LTMatrix3* virial)
{
    VECTOR dr = Get_Displacement<BoundaryPolicy::Periodic>(
        crd[atom1], crd[atom0], boundary);
    value[0] = dr.y;
    VECTOR drdx = {0.0f, 1.0f, 0.0f};
    crd_grads[atom1] = drdx;
    crd_grads[atom0] = -drdx;
    virial[0] = Get_Virial_From_Force_Dis(dr, drdx);
}

static __global__ void displacement_z_get_all(const int atom0, const int atom1,
                                              const VECTOR* crd,
                                              const Boundary boundary,
                                              float* value, VECTOR* crd_grads,
                                              LTMatrix3* virial)
{
    VECTOR dr = Get_Displacement<BoundaryPolicy::Periodic>(
        crd[atom1], crd[atom0], boundary);
    value[0] = dr.z;
    VECTOR drdx = {0.0f, 0.0f, 1.0f};
    crd_grads[atom1] = drdx;
    crd_grads[atom0] = -drdx;
    virial[0] = Get_Virial_From_Force_Dis(dr, drdx);
}

void CV_DISTANCE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                          int atom_numbers, const char* module_name)
{
    atom = manager->Ask_For_Int_Parameter(module_name, "atom", 2, 2);
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_DISTANCE::Compute(int atom_numbers, VECTOR* crd,
                          const Boundary boundary, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need != CV_NEED_NONE)
    {
        if (strcmp(type_name, "distance") == 0)
            Launch_Device_Kernel(distance_get_all, 1, 1, 0, NULL, atom[0],
                                 atom[1], crd, boundary, d_value, crd_grads,
                                 virial);
        else if (strcmp(type_name, "displacement_x"))
            Launch_Device_Kernel(displacement_x_get_all, 1, 1, 0, NULL, atom[0],
                                 atom[1], crd, boundary, d_value, crd_grads,
                                 virial);
        else if (strcmp(type_name, "displacement_y"))
            Launch_Device_Kernel(displacement_y_get_all, 1, 1, 0, NULL, atom[0],
                                 atom[1], crd, boundary, d_value, crd_grads,
                                 virial);
        else if (strcmp(type_name, "displacement_z"))
            Launch_Device_Kernel(displacement_z_get_all, 1, 1, 0, NULL, atom[0],
                                 atom[1], crd, boundary, d_value, crd_grads,
                                 virial);
        deviceMemcpy(&value, d_value, sizeof(float), deviceMemcpyDeviceToHost);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_ANGLE, "angle", 0);

static __global__ void angle_get_all(const int atom0, const int atom1,
                                     const int atom2, const VECTOR* crd,
                                     const Boundary boundary, float* value,
                                     VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR r1 = crd[atom1];
    VECTOR r0 =
        Get_Displacement<BoundaryPolicy::Periodic>(crd[atom0], r1, boundary);
    VECTOR r2 =
        Get_Displacement<BoundaryPolicy::Periodic>(crd[atom2], r1, boundary);
    SADvector<6> dr01(r0, 0, 1, 2);
    SADvector<6> dr21(r2, 3, 4, 5);
    SADfloat<6> temp = 1.0f / (dr01 * dr01) / (dr21 * dr21);
    temp = sqrtf(temp) * (dr01 * dr21);
    if (temp > 0.999999f)
        temp = 0.999999f;
    else if (temp < -0.999999f)
        temp = -0.999999f;
    temp = acosf(temp);
    value[0] = temp.val;
    VECTOR frc01 = {temp.dval[0], temp.dval[1], temp.dval[2]};
    VECTOR frc21 = {temp.dval[3], temp.dval[4], temp.dval[5]};
    crd_grads[atom0] = frc01;
    crd_grads[atom1] = -frc01 - frc21;
    crd_grads[atom2] = frc21;
    virial[0] = Get_Virial_From_Force_Dis(frc01, r0) +
                Get_Virial_From_Force_Dis(frc21, r2);
}

void CV_ANGLE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                       int atom_numbers, const char* module_name)
{
    atom = manager->Ask_For_Int_Parameter(module_name, "atom", 3, 2);
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_ANGLE::Compute(int atom_numbers, VECTOR* crd, const Boundary boundary,
                       int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need != CV_NEED_NONE)
    {
        Launch_Device_Kernel(angle_get_all, 1, 1, 0, NULL, atom[0], atom[1],
                             atom[2], crd, boundary, d_value, crd_grads,
                             virial);
        deviceMemcpy(&value, d_value, sizeof(float), deviceMemcpyDeviceToHost);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_DIHEDRAL, "dihedral", 0);

static __global__ void dihedral_get_all(const int atom0, const int atom1,
                                        const int atom2, const int atom3,
                                        const VECTOR* crd,
                                        const Boundary boundary, float* value,
                                        VECTOR* crd_grads, LTMatrix3* virial)
{
    VECTOR r0 = crd[atom0];
    VECTOR r1 = crd[atom1];
    VECTOR r2 = crd[atom2];
    VECTOR r3 = crd[atom3];
    r0 = Get_Displacement<BoundaryPolicy::Periodic>(r1, r0, boundary);
    r1 = Get_Displacement<BoundaryPolicy::Periodic>(r2, r1, boundary);
    r2 = Get_Displacement<BoundaryPolicy::Periodic>(r3, r2, boundary);
    SADvector<9> dr01(r0, 0, 1, 2);
    SADvector<9> dr21(r1, 3, 4, 5);
    SADvector<9> dr23(r2, 6, 7, 8);

    SADvector<9> dr_temp = dr01 ^ dr21;
    dr01 = dr21 ^ dr23;

    SADfloat<9> temp = 1.0f / (dr01 * dr01) / (dr_temp * dr_temp);
    temp = sqrtf(temp) * (dr01 * dr_temp);
    if (temp.val > 0.999999f)
        temp.val = 0.999999f;
    else if (temp.val < -0.999999f)
        temp.val = -0.999999f;
    temp = acosf(temp);
    if ((dr_temp * dr23).val < 0) temp = -temp;
    value[0] = temp.val;
    VECTOR f1 = {temp.dval[0], temp.dval[1], temp.dval[2]};
    VECTOR f2 = {temp.dval[3], temp.dval[4], temp.dval[5]};
    VECTOR f3 = {temp.dval[6], temp.dval[7], temp.dval[8]};
    crd_grads[atom0] = -f1;
    crd_grads[atom1] = f1 - f2;
    crd_grads[atom2] = f2 - f3;
    crd_grads[atom3] = f3;
    virial[0] = Get_Virial_From_Force_Dis(f1, r0) +
                Get_Virial_From_Force_Dis(f2, r1) +
                Get_Virial_From_Force_Dis(f3, r2);
}

void CV_DIHEDRAL::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                          int atom_numbers, const char* module_name)
{
    atom = manager->Ask_For_Int_Parameter(module_name, "atom", 4, 2);
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_DIHEDRAL::Compute(int atom_numbers, VECTOR* crd,
                          const Boundary boundary, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need != CV_NEED_NONE)
    {
        Launch_Device_Kernel(dihedral_get_all, 1, 1, 0, NULL, atom[0], atom[1],
                             atom[2], atom[3], crd, boundary, d_value,
                             crd_grads, virial);
        deviceMemcpy(&value, d_value, sizeof(float), deviceMemcpyDeviceToHost);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
