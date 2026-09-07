#pragma once

void MD_INFORMATION::periodic_box_condition_information::Initial(
    CONTROLLER* controller, MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    this->pbc = true;
    if (controller->Command_Exist("pbc"))
    {
        this->pbc = controller->Get_Bool(
            "pbc",
            "MD_INFORMATION::periodic_box_condition_information::Initial");
    }
    this->boundary.policy =
        this->pbc ? BoundaryPolicy::Periodic : BoundaryPolicy::Open;
    this->No_PBC_Check(controller);
    this->PBC_Check();
    this->cell0 = boundary.cell;
}

void MD_INFORMATION::periodic_box_condition_information::No_PBC_Check(
    CONTROLLER* controller)
{
    if (this->pbc) return;

    if (controller->MPI_size > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "NOPBC can not be used in Multi-Process mode");
    }

    if (md_info->nb.cutoff < 100)
    {
        controller->Warn(
            "The cutoff for NOPBC is not greater than 100 angstrom, which may "
            "be inaccurate");
    }
    if (md_info->sys.box_length.x < 900 || md_info->sys.box_length.y < 900 ||
        md_info->sys.box_length.z < 900)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "The box length of the system should always be greater than 900 "
            "angstrom for NOPBC");
    }
    if (md_info->mode == md_info->NPT)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "NPT mode can not be used for NOPBC");
    }
}

void MD_INFORMATION::periodic_box_condition_information::PBC_Check()
{
    VECTOR length = md_info->sys.box_length;
    VECTOR angle = CONSTANT_DEG_TO_RAD * md_info->sys.box_angle;
    float a = length.x;
    float b = length.y;
    float c = length.z;
    float alpha = angle.x;
    float beta = angle.y;
    float gamma = angle.z;
    float za = sqrtf(1 - cosf(alpha) * cosf(alpha) - cosf(beta) * cosf(beta) -
                     cosf(gamma) * cosf(gamma) +
                     2 * cosf(alpha) * cosf(beta) * cosf(gamma));

    boundary.cell.a11 = a;
    boundary.cell.a21 = b * cosf(gamma);
    boundary.cell.a22 = b * sinf(gamma);
    boundary.cell.a31 = c * cosf(beta);
    boundary.cell.a32 =
        c / sinf(gamma) * (cosf(alpha) - cosf(beta) * cosf(gamma));
    boundary.cell.a33 = c / sinf(gamma) * za;

    boundary.rcell.a11 = 1.0f / boundary.cell.a11;
    boundary.rcell.a22 = 1.0f / boundary.cell.a22;
    boundary.rcell.a33 = 1.0f / boundary.cell.a33;
    boundary.rcell.a21 = -boundary.rcell.a11 / tanf(gamma);
    boundary.rcell.a31 =
        (cosf(alpha) / tanf(gamma) - cosf(beta) / sinf(gamma)) / za / a;
    boundary.rcell.a32 =
        (cosf(beta) / tanf(gamma) - cosf(alpha) / sinf(gamma)) / za / b;

    boundary.cell.a21 = fabsf(boundary.cell.a21) < 1e-3 ? 0 : boundary.cell.a21;
    boundary.cell.a31 = fabsf(boundary.cell.a31) < 1e-3 ? 0 : boundary.cell.a31;
    boundary.cell.a32 = fabsf(boundary.cell.a32) < 1e-3 ? 0 : boundary.cell.a32;

    boundary.rcell.a21 =
        fabsf(boundary.rcell.a21) < 1e-3 ? 0 : boundary.rcell.a21;
    boundary.rcell.a31 =
        fabsf(boundary.rcell.a31) < 1e-3 ? 0 : boundary.rcell.a31;
    boundary.rcell.a32 =
        fabsf(boundary.rcell.a32) < 1e-3 ? 0 : boundary.rcell.a32;
}

void MD_INFORMATION::periodic_box_condition_information::Update_Box(LTMatrix3 g)
{
    boundary.cell.a11 =
        boundary.cell.a11 + md_info->dt * boundary.cell.a11 * g.a11;
    boundary.cell.a22 =
        boundary.cell.a22 + md_info->dt * boundary.cell.a22 * g.a22;
    boundary.cell.a33 =
        boundary.cell.a33 + md_info->dt * boundary.cell.a33 * g.a33;
    boundary.cell.a21 =
        boundary.cell.a21 +
        md_info->dt * (boundary.cell.a21 * g.a11 + boundary.cell.a22 * g.a21);
    boundary.cell.a31 =
        boundary.cell.a31 +
        md_info->dt * (boundary.cell.a31 * g.a11 + boundary.cell.a32 * g.a21 +
                       boundary.cell.a33 * g.a31);
    boundary.cell.a32 =
        boundary.cell.a32 +
        md_info->dt * (boundary.cell.a32 * g.a22 + boundary.cell.a33 * g.a32);
    VECTOR va = {boundary.cell.a11, 0, 0};
    VECTOR vb = {boundary.cell.a21, boundary.cell.a22, 0};
    VECTOR vc = {boundary.cell.a31, boundary.cell.a32, boundary.cell.a33};
    float a = sqrtf(va * va);
    float b = sqrtf(vb * vb);
    float c = sqrtf(vc * vc);
    float alpha = acos(va * vb / a / b);
    float beta = acos(va * vc / a / c);
    float gamma = acos(vc * vb / c / b);
    float za = sqrtf(1 - cosf(alpha) * cosf(alpha) - cosf(beta) * cosf(beta) -
                     cosf(gamma) * cosf(gamma) +
                     2 * cosf(alpha) * cosf(beta) * cosf(gamma));
    boundary.rcell.a11 = 1.0f / boundary.cell.a11;
    boundary.rcell.a22 = 1.0f / boundary.cell.a22;
    boundary.rcell.a33 = 1.0f / boundary.cell.a33;
    boundary.rcell.a21 = -boundary.rcell.a11 / tanf(gamma);
    boundary.rcell.a31 =
        (cosf(alpha) / tanf(gamma) - cosf(beta) / sinf(gamma)) / za / a;
    boundary.rcell.a32 =
        (cosf(beta) / tanf(gamma) - cosf(alpha) / sinf(gamma)) / za / b;

    md_info->sys.box_length.x = a;
    md_info->sys.box_length.y = b;
    md_info->sys.box_length.z = c;
    md_info->sys.box_angle.x = alpha * CONSTANT_RAD_TO_DEG;
    md_info->sys.box_angle.y = beta * CONSTANT_RAD_TO_DEG;
    md_info->sys.box_angle.z = gamma * CONSTANT_RAD_TO_DEG;
}

bool MD_INFORMATION::periodic_box_condition_information::Check_Change_Large()
{
    bool result = false;
    float grid_length = 0.5f * (md_info->nb.cutoff + md_info->nb.skin);
    float* cell = (float*)&this->boundary.cell;
    float* cell0 = (float*)&this->cell0;
    int i1, i0;
    float f1, f0;
    for (int i = 0; i < 6; i += 1)
    {
        i1 = cell[i] / grid_length;
        i0 = cell0[i] / grid_length;
        f1 = cell[i];
        f0 = cell0[i];
        if (fabsf(f1 - f0) > 0.5f * md_info->nb.skin && i1 != i0)
        {
            result = true;
        }
    }
    if (result)
    {
        this->cell0 = this->boundary.cell;
    }
    return result;
}

LTMatrix3 MD_INFORMATION::periodic_box_condition_information::Get_Cell(
    VECTOR box_length, VECTOR box_angle)
{
    LTMatrix3 cell;
    double a = box_length.x;
    double b = box_length.y;
    double c = box_length.z;
    double alpha = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.x;
    double beta = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.y;
    double gamma = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.z;
    double za = std::sqrt(1 - cos(alpha) * cos(alpha) - cos(beta) * cos(beta) -
                          cos(gamma) * cos(gamma) +
                          2 * cos(alpha) * cos(beta) * cos(gamma));
    cell.a11 = a;
    cell.a21 = b * cos(gamma);
    cell.a22 = b * sin(gamma);
    cell.a31 = c * cos(beta);
    cell.a32 = c / sin(gamma) * (cos(alpha) - cos(beta) * cos(gamma));
    cell.a33 = c / sin(gamma) * za;
    return cell;
}
