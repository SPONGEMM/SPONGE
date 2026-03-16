#include "MD_core.h"

#include "../xponge/xponge.h"

#define BOX_TRAJ_COMMAND "box"
#define BOX_TRAJ_DEFAULT_FILENAME "mdbox.txt"
#define TRAJ_COMMAND "crd"
#define TRAJ_DEFAULT_FILENAME "mdcrd.dat"
#define RESTART_COMMAND "rst"
#define RESTART_DEFAULT_FILENAME "restart"
// 20210827用于输出速度和力
#define FRC_TRAJ_COMMAND "frc"
#define VEL_TRAJ_COMMAND "vel"
// 20230303 用于错误输出原因
#define ATOM_NUMBERS_DISMATCH                                          \
    "Reason:\n\t'atom_numbers' (the number of atoms) is diiferent in " \
    "different input files\n"
#define ATOM_NUMBERS_MISSING                                        \
    "Reason:\n\tno 'atom_numbers' (the number of atoms) found. No " \
    "'mass_in_file' or 'amber_parm7' is provided\n"

#include "min.hpp"
#include "mol.hpp"
#include "nb.hpp"
#include "nve.hpp"
#include "output.hpp"
#include "pbc.hpp"
#include "rerun.hpp"
#include "sys.hpp"
#include "ug.hpp"

static int Xponge_Atom_Numbers()
{
    if (!Xponge::system.atoms.mass.empty())
    {
        return (int)Xponge::system.atoms.mass.size();
    }
    if (!Xponge::system.atoms.charge.empty())
    {
        return (int)Xponge::system.atoms.charge.size();
    }
    if (!Xponge::system.atoms.coordinate.empty())
    {
        return (int)Xponge::system.atoms.coordinate.size() / 3;
    }
    if (!Xponge::system.atoms.velocity.empty())
    {
        return (int)Xponge::system.atoms.velocity.size() / 3;
    }
    return 0;
}

static __global__ void Scale_Positions_Device(const int atom_numbers,
                                              const LTMatrix3 g, VECTOR* crd,
                                              float dt)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR r = crd[atom_i], r_dash;
        r_dash.x = r.x + dt * (r.x * g.a11 + r.y * g.a21 + r.z * g.a31);
        r_dash.y = r.y + dt * (r.y * g.a22 + r.z * g.a32);
        r_dash.z = r.z + dt * r.z * g.a33;
        crd[atom_i] = r_dash;
    }
}

static __global__ void Scale_Velocities_Device(const int atom_numbers,
                                               const LTMatrix3 g, VECTOR* vel,
                                               float dt)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR v = vel[atom_i], v_dash;
        v_dash.x = v.x - dt * (v.x * g.a11 + v.y * g.a21 + v.z * g.a31);
        v_dash.y = v.y - dt * (v.y * g.a22 + v.z * g.a32);
        v_dash.z = v.z - dt * v.z * g.a33;
        vel[atom_i] = v_dash;
    }
}

void MD_INFORMATION::Read_Mode(CONTROLLER* controller)
{
    if (controller->Command_Choice("mode", "nvt"))
    {
        controller->printf("    Mode set to NVT\n");
        this->sys.speed_unit_name = "ns/day";
        mode = 1;
    }
    else if (controller->Command_Choice("mode", "npt"))
    {
        controller->printf("    Mode set to NPT\n");
        this->sys.speed_unit_name = "ns/day";
        mode = 2;
    }
    else if (controller->Command_Choice("mode", "minimization") ||
             controller->Command_Choice("mode", "min"))
    {
        controller->printf("    Mode set to Energy Minimization\n");
        this->sys.speed_unit_name = "step/second";
        mode = -1;
    }
    else if (controller->Command_Choice("mode", "nve"))
    {
        controller->printf("    Mode set to NVE\n");
        this->sys.speed_unit_name = "ns/day";
        mode = 0;
    }
    else if (controller->Command_Choice("mode", "rerun"))
    {
        controller->printf("    Mode set to RERUN\n");
        this->sys.speed_unit_name = "frame/second";
        mode = -2;
    }
    else
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                       "MD_INFORMATION::Read_Mode",
                                       "Reason:\n\t wrong 'mode'\n");
    }
}

void MD_INFORMATION::Read_dt(CONTROLLER* controller)
{
    if (mode == RERUN)
    {
        return;
    }
    if (controller[0].Command_Exist("dt"))
    {
        controller->Check_Float("dt", "MD_INFORMATION::Read_dt");
        controller->printf("    dt set to %f ps\n",
                           atof(controller[0].Command("dt")));
        dt = atof(controller[0].Command("dt")) * CONSTANT_TIME_CONVERTION;
        sscanf(controller[0].Command("dt"), "%lf", &sys.dt_in_ps);
        sys.speed_time_factor = 86.4f * sys.dt_in_ps;
    }
    else
    {
        if (mode != MINIMIZATION)
        {
            dt = 0.001;
            sys.dt_in_ps = 0.001;
        }
        controller->printf("    dt set to %e ps\n", dt);
        dt *= CONSTANT_TIME_CONVERTION;
    }
    if (mode == MINIMIZATION || mode == RERUN)
    {
        sys.dt_in_ps = 0;
        sys.speed_time_factor = 1.0f;
    }
}

void MD_INFORMATION::Read_Coordinate_And_Velocity(CONTROLLER* controller)
{
    sys.start_time = 0.0;
    if (mode == RERUN)
    {
        if (atom_numbers == 0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "MD_INFORMATION::Read_Coordinate_And_Velocity",
                "Reason:\n\tFor the 'rerun' mode, the number of atoms should "
                "be provided by mass_in_file or charge_in_file\n");
        }
        Malloc_Safely(
            (void**)&coordinate,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        Device_Malloc_And_Copy_Safely(
            (void**)&crd, coordinate,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        Device_Malloc_Safely(
            (void**)&last_crd,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        deviceMemset(
            last_crd, 0,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        Malloc_Safely(
            (void**)&velocity,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        Device_Malloc_And_Copy_Safely(
            (void**)&vel, velocity,
            sizeof(VECTOR) * (this->atom_numbers +
                              no_direct_interaction_virtual_atom_numbers));
        rerun.Initial(controller, this);
        rerun.Iteration(rerun.start_frame);
        return;
    }
    if (Xponge::system.atoms.coordinate.empty())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "MD_INFORMATION::Read_Coordinate_And_Velocity",
            "Reason:\n\tno coordinate information found in Xponge::system\n");
    }
    atom_numbers = Xponge_Atom_Numbers();
    Malloc_Safely(
        (void**)&coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_Safely(
        (void**)&last_crd,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    deviceMemset(last_crd, 0,
                 sizeof(VECTOR) * (this->atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    Malloc_Safely(
        (void**)&velocity,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));

    for (int i = 0; i < atom_numbers; i++)
    {
        coordinate[i].x = Xponge::system.atoms.coordinate[3 * i];
        coordinate[i].y = Xponge::system.atoms.coordinate[3 * i + 1];
        coordinate[i].z = Xponge::system.atoms.coordinate[3 * i + 2];
    }
    if (!Xponge::system.atoms.velocity.empty())
    {
        for (int i = 0; i < atom_numbers; i++)
        {
            velocity[i].x = Xponge::system.atoms.velocity[3 * i];
            velocity[i].y = Xponge::system.atoms.velocity[3 * i + 1];
            velocity[i].z = Xponge::system.atoms.velocity[3 * i + 2];
        }
    }
    else
    {
        for (int i = 0; i < atom_numbers; i++)
        {
            velocity[i].x = 0;
            velocity[i].y = 0;
            velocity[i].z = 0;
        }
    }
    if (Xponge::system.box.box_length.size() >= 3)
    {
        sys.box_length.x = Xponge::system.box.box_length[0];
        sys.box_length.y = Xponge::system.box.box_length[1];
        sys.box_length.z = Xponge::system.box.box_length[2];
    }
    if (Xponge::system.box.box_angle.size() >= 3)
    {
        sys.box_angle.x = Xponge::system.box.box_angle[0];
        sys.box_angle.y = Xponge::system.box.box_angle[1];
        sys.box_angle.z = Xponge::system.box.box_angle[2];
    }
    sys.start_time = Xponge::system.start_time;
    Device_Malloc_And_Copy_Safely(
        (void**)&crd, coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_And_Copy_Safely(
        (void**)&vel, velocity,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
}

void MD_INFORMATION::Read_Mass(CONTROLLER* controller)
{
    if (Xponge::system.atoms.mass.empty())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "MD_INFORMATION::Read_Mass",
            "Reason:\n\tno mass information found in Xponge::system\n");
    }
    atom_numbers = Xponge_Atom_Numbers();
    Malloc_Safely((void**)&h_mass, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_mass_inverse, sizeof(float) * atom_numbers);
    sys.total_mass = 0;
    for (int i = 0; i < atom_numbers; i++)
    {
        h_mass[i] = Xponge::system.atoms.mass[i];
        if (h_mass[i] == 0)
            h_mass_inverse[i] = 0;
        else
            h_mass_inverse[i] = 1.0f / h_mass[i];
        sys.total_mass += h_mass[i];
    }
    if (atom_numbers > 0)
    {
        Device_Malloc_And_Copy_Safely((void**)&d_mass, h_mass,
                                      sizeof(float) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_mass_inverse, h_mass_inverse,
                                      sizeof(float) * atom_numbers);
    }
}

void MD_INFORMATION::Read_Charge(CONTROLLER* controller)
{
    if (Xponge::system.atoms.charge.empty())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "MD_INFORMATION::Read_Charge",
            "Reason:\n\tno charge information found in Xponge::system\n");
    }
    atom_numbers = Xponge_Atom_Numbers();
    Malloc_Safely((void**)&h_charge, sizeof(float) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        h_charge[i] = Xponge::system.atoms.charge[i];
    }
    if (atom_numbers > 0)
    {
        Device_Malloc_And_Copy_Safely((void**)&d_charge, h_charge,
                                      sizeof(float) * atom_numbers);
    }
}

// MD_INFORMATION成员函数
void MD_INFORMATION::Initial(CONTROLLER* controller)
{
    controller->printf("START INITIALIZING MD CORE:\n");
    atom_numbers = 0;  // 初始化，使得能够进行所有原子数目是否相等的判断

    strcpy(md_name, controller[0].Command("md_name"));
    Read_Mode(controller);

    Read_Mass(controller);
    Read_Charge(controller);
    Atom_Information_Initial();

    Read_Coordinate_And_Velocity(controller);

    sys.Initial(controller, this);
    nb.Initial(controller, this);

    output.Initial(controller, this);

    nve.Initial(controller, this);

    min.Initial(controller, this);

    ug.Initial_Edge(atom_numbers);

    res.Initial(controller, this);

    mol.md_info = this;

    pbc.Initial(controller, this);

    Read_dt(controller);

    is_initialized = 1;
    controller->printf("    structure last modify date is %d\n",
                       last_modify_date);
    controller->printf("END INITIALIZING MD CORE\n\n");
}

void MD_INFORMATION::Atom_Information_Initial()
{
    int all_numbers = atom_numbers + no_direct_interaction_virtual_atom_numbers;
    Malloc_Safely((void**)&force, sizeof(VECTOR) * all_numbers);
    memset(force, 0, sizeof(VECTOR) * all_numbers);
    Malloc_Safely((void**)&acceleration, sizeof(VECTOR) * all_numbers);
    memset(acceleration, 0, sizeof(VECTOR) * all_numbers);
    Malloc_Safely((void**)&h_atom_energy, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_atom_virial_tensor,
                  sizeof(LTMatrix3) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&frc, force,
                                  sizeof(VECTOR) * all_numbers);
    Device_Malloc_And_Copy_Safely((void**)&acc, acceleration,
                                  sizeof(VECTOR) * all_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_energy, h_atom_energy,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_virial_tensor,
                                  h_atom_virial_tensor,
                                  sizeof(LTMatrix3) * atom_numbers);
    Device_Malloc_Safely((void**)&d_atom_ek, sizeof(float) * atom_numbers);
    sys.freedom = 3 * atom_numbers;  // 最大自由度，后面减
}

void MD_INFORMATION::Atom_Information_Initial(
    std::map<std::string, SpongeTensor*>& args)
{
    atom_numbers = args["crd"]->shape(0);
    coordinate = (VECTOR*)args["coordinate"]->data();
    velocity = (VECTOR*)args["velocity"]->data();
    h_mass = (float*)args["h_mass"]->data();
    h_mass_inverse = (float*)args["h_mass_inverse"]->data();
    h_charge = (float*)args["h_charge"]->data();
    crd = (VECTOR*)args["crd"]->data();
    vel = (VECTOR*)args["vel"]->data();
    d_mass = (float*)args["d_mass"]->data();
    d_mass_inverse = (float*)args["d_mass_inverse"]->data();
    d_charge = (float*)args["d_charge"]->data();
    Atom_Information_Initial();
}

void MD_INFORMATION::Read_Coordinate_In_File(const char* file_name,
                                             CONTROLLER controller)
{
    FILE* fp = NULL;
    controller.printf("    Start reading coordinate_in_file:\n");
    Open_File_Safely(&fp, file_name, "r");
    char lin[CHAR_LENGTH_MAX];
    char* get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
    int atom_numbers = 0;
    int scanf_ret = sscanf(lin, "%d %lf", &atom_numbers, &sys.start_time);
    if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
    {
        controller.Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                      "MD_INFORMATION::Read_Coordinate_In_File",
                                      ATOM_NUMBERS_DISMATCH);
    }
    else if (this->atom_numbers == 0)
    {
        this->atom_numbers = atom_numbers;
    }
    if (scanf_ret == 0)
    {
        std::string error_reason =
            "Reason:\n\tthe format of the coordinate_in_file (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Coordinate_In_File",
                                      error_reason.c_str());
    }
    else if (scanf_ret == 1)
    {
        sys.start_time = 0;
    }

    controller.printf("        atom_numbers is %d\n", this->atom_numbers);
    controller.printf("        system start_time is %lf\n",
                      this->sys.start_time);
    Malloc_Safely(
        (void**)&coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_Safely(
        (void**)&last_crd,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    deviceMemset(last_crd, 0,
                 sizeof(VECTOR) * (this->atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));

    for (int i = 0; i < atom_numbers; i++)
    {
        scanf_ret = fscanf(fp, "%f %f %f", &coordinate[i].x, &coordinate[i].y,
                           &coordinate[i].z);
        if (scanf_ret != 3)
        {
            std::string error_reason =
                "Reason:\n\tthe format of the coordinate_in_file (";
            error_reason += file_name;
            error_reason += ") is not right\n";
            controller.Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "MD_INFORMATION::Read_Coordinate_In_File",
                error_reason.c_str());
        }
    }
    scanf_ret = fscanf(fp, "%f %f %f", &sys.box_length.x, &sys.box_length.y,
                       &sys.box_length.z);
    if (scanf_ret != 3)
    {
        std::string error_reason =
            "Reason:\n\tthe format of the coordinate_in_file (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Coordinate_In_File",
                                      error_reason.c_str());
    }
    scanf_ret = fscanf(fp, "%f %f %f", &sys.box_angle.x, &sys.box_angle.y,
                       &sys.box_angle.z);
    if (scanf_ret != 3)
    {
        std::string error_reason =
            "Reason:\n\tthe format of the coordinate_in_file (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Coordinate_In_File",
                                      error_reason.c_str());
    }
    controller.printf(
        "        box_length is\n            x: %f\n            y: %f\n         "
        "   z: %f\n",
        sys.box_length.x, sys.box_length.y, sys.box_length.z);
    controller.printf(
        "        box_angle is\n            alpha: %f\n            beta: %f\n   "
        "         gamma: %f\n",
        sys.box_angle.x, sys.box_angle.y, sys.box_angle.z);
    Device_Malloc_And_Copy_Safely(
        (void**)&crd, coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    controller.printf("    End reading coordinate_in_file\n\n");
    fclose(fp);
}
void MD_INFORMATION::Read_Rst7(const char* file_name, int irest,
                               CONTROLLER controller)
{
    FILE* fin = NULL;
    Open_File_Safely(&fin, file_name, "r");
    controller.printf("    Start reading AMBER rst7:\n");
    char lin[CHAR_LENGTH_MAX];
    int atom_numbers = 0;
    char* get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    int has_vel = 0;
    int scanf_ret = sscanf(lin, "%d %lf", &atom_numbers, &sys.start_time);
    if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
    {
        controller.Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                      "MD_INFORMATION::Read_Rst7",
                                      ATOM_NUMBERS_DISMATCH);
    }
    else if (this->atom_numbers == 0)
    {
        this->atom_numbers = atom_numbers;
    }
    if (scanf_ret == 0)
    {
        std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Rst7",
                                      error_reason.c_str());
    }
    else if (scanf_ret == 2)
    {
        has_vel = 1;
    }
    else
    {
        sys.start_time = 0;
    }

    Malloc_Safely(
        (void**)&coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_Safely(
        (void**)&last_crd,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    deviceMemset(last_crd, 0,
                 sizeof(VECTOR) * (this->atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    Malloc_Safely(
        (void**)&velocity,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));

    controller.printf("        atom_numbers is %d\n", this->atom_numbers);
    controller.printf("        system start time is %lf\n",
                      this->sys.start_time);

    if (has_vel == 0 || irest == 0)
    {
        controller.printf("        All velocity will be set to 0\n");
    }

    for (int i = 0; i < this->atom_numbers; i = i + 1)
    {
        scanf_ret = fscanf(fin, "%f %f %f", &this->coordinate[i].x,
                           &this->coordinate[i].y, &this->coordinate[i].z);
        if (scanf_ret != 3)
        {
            std::string error_reason =
                "Reason:\n\tthe format of the amber_rst7 (";
            error_reason += file_name;
            error_reason += ") is not right (missing the coordinate of atom ";
            error_reason += i;
            error_reason += ")\n";
            controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                          "MD_INFORMATION::Read_Rst7",
                                          error_reason.c_str());
        }
    }
    if (has_vel)
    {
        for (int i = 0; i < this->atom_numbers; i = i + 1)
        {
            scanf_ret = fscanf(fin, "%f %f %f", &this->velocity[i].x,
                               &this->velocity[i].y, &this->velocity[i].z);
            if (scanf_ret != 3)
            {
                std::string error_reason =
                    "Reason:\n\tthe format of the amber_rst7 (";
                error_reason += file_name;
                error_reason += ") is not right (missing the velocity of atom ";
                error_reason += i;
                error_reason += ")\n";
                controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                              "MD_INFORMATION::Read_Rst7",
                                              error_reason.c_str());
            }
        }
    }
    if (irest == 0 || !has_vel)
    {
        for (int i = 0; i < this->atom_numbers; i = i + 1)
        {
            this->velocity[i].x = 0.0;
            this->velocity[i].y = 0.0;
            this->velocity[i].z = 0.0;
        }
    }
    scanf_ret = fscanf(fin, "%f %f %f", &this->sys.box_length.x,
                       &this->sys.box_length.y, &this->sys.box_length.z);
    if (scanf_ret != 3)
    {
        std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
        error_reason += file_name;
        error_reason += ") is not right (missing the box information)\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Rst7",
                                      error_reason.c_str());
    }
    controller.printf("        box lengths are %f %f %f\n",
                      this->sys.box_length.x, this->sys.box_length.y,
                      this->sys.box_length.z);

    scanf_ret = fscanf(fin, "%f %f %f", &this->sys.box_angle.x,
                       &this->sys.box_angle.y, &this->sys.box_angle.z);
    if (scanf_ret != 3)
    {
        std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
        error_reason += file_name;
        error_reason += ") is not right (missing the box information)\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "MD_INFORMATION::Read_Rst7",
                                      error_reason.c_str());
    }
    controller.printf("        box angles are %f %f %f\n",
                      this->sys.box_angle.x, this->sys.box_angle.y,
                      this->sys.box_angle.z);
    Device_Malloc_And_Copy_Safely(
        (void**)&crd, coordinate,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_And_Copy_Safely(
        (void**)&vel, velocity,
        sizeof(VECTOR) *
            (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    fclose(fin);
    controller.printf("    End reading AMBER rst7\n\n");
}

void MD_INFORMATION::MD_Reset_Atom_Energy_And_Virial_And_Force()
{
    need_potential = 0;
    need_kinetic = 0;
    if ((output.print_zeroth_frame || sys.steps) &&
        sys.steps % output.write_mdout_interval == 0)
    {
        need_potential = 1;
        need_pressure = output.print_virial;
        need_kinetic = 1;
    }

    deviceMemset(d_atom_energy, 0, sizeof(float) * atom_numbers);
    // 修正：frc需要对包括虚拟原子进行清零，不论此前是否做过force_distribute
    deviceMemset(frc, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    deviceMemset(d_atom_virial_tensor, 0, sizeof(LTMatrix3) * atom_numbers);

    deviceMemset(sys.d_potential, 0, sizeof(float));
    deviceMemset(sys.d_virial_tensor, 0, sizeof(LTMatrix3));
    deviceMemset(sys.d_stress, 0, sizeof(LTMatrix3));
}

/*
void MD_INFORMATION::Sum_Force_Pressure_And_Potential_If_Needed()
{
    SPONGE_MPI_WRAPPER::Device_Sum(frc, 3 * atom_numbers,
                                   CONTROLLER::D_MPI_COMM_WORLD);

    if (need_pressure > 0)
    {
        sys.Get_Pressure(false);
    }

    if (need_potential > 0)
    {
        sys.Get_Potential(false);
    }
}
*/
void MD_INFORMATION::Scale_Positions_And_Velocities(LTMatrix3 g, int scale_crd,
                                                    int scale_vel, VECTOR* crd,
                                                    VECTOR* vel)
{
    // 处理坐标
    switch (scale_crd)
    {
        case SCALE_COORDINATES_BY_ATOM:
            Launch_Device_Kernel(
                Scale_Positions_Device,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, g, crd,
                dt);
            break;

        default:
            break;
    }
    // 处理速度
    switch (scale_vel)
    {
        case SCALE_VELOCITIES_BY_ATOM:
            Launch_Device_Kernel(
                Scale_Velocities_Device,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, g, vel,
                dt);
            break;

        default:
            break;
    }
}

void MD_INFORMATION::MD_Information_Frc_Device_To_Host()
{
    deviceMemcpy(this->force, this->frc, sizeof(VECTOR) * this->atom_numbers,
                 deviceMemcpyDeviceToHost);
}

void MD_INFORMATION::MD_Information_Frc_Host_To_Device()
{
    deviceMemcpy(this->frc, this->force, sizeof(VECTOR) * this->atom_numbers,
                 deviceMemcpyHostToDevice);
}

void MD_INFORMATION::Crd_Vel_Device_To_Host(int forced)
{
    if (output.current_crd_synchronized_step != sys.steps || forced)
    {
        output.current_crd_synchronized_step = sys.steps;
        deviceMemcpy(this->coordinate, this->crd,
                     sizeof(VECTOR) * this->atom_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(this->velocity, this->vel,
                     sizeof(VECTOR) * this->atom_numbers,
                     deviceMemcpyDeviceToHost);
    }
}

static __global__ void dd_crd_and_vel_to_global(int atom_numbers,
                                                char* atom_local_label,
                                                VECTOR* crd, VECTOR* dd_crd,
                                                VECTOR* vel, VECTOR* dd_vel,
                                                int* atom_local_id)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        if (atom_local_label[i] == 1)
        {
            crd[i] = dd_crd[atom_local_id[i]];
            vel[i] = dd_vel[atom_local_id[i]];
        }
        else
        {
            crd[i] = {0, 0, 0};
            vel[i] = {0, 0, 0};
        }
    }
}

static __global__ void global_crd_and_vel_to_dd(int atom_numbers,
                                                char* atom_local_label,
                                                VECTOR* crd, VECTOR* dd_crd,
                                                VECTOR* vel, VECTOR* dd_vel,
                                                int* atom_local_id)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        if (atom_local_label[i] == 1)
        {
            int local_id = atom_local_id[i];
            if (local_id >= 0)
            {
                dd_crd[local_id] = crd[i];
                dd_vel[local_id] = vel[i];
            }
        }
    }
}

void MD_INFORMATION::Crd_Vel_dd_to_Device(VECTOR* dd_crd, VECTOR* dd_vel,
                                          char* dd_atom_local_label,
                                          int* dd_atom_local_id,
                                          deviceStream_t stream)
{
    // Always map local ordering back to the global ordering before output;
    // even with a single PP rank, Domain_Decomposition may reorder atoms.
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        deviceMemset(vel, 0, sizeof(VECTOR) * atom_numbers);
        deviceMemset(crd, 0, sizeof(VECTOR) * atom_numbers);
        Launch_Device_Kernel(dd_crd_and_vel_to_global,
                             (atom_numbers + 255) / 256, 256, 0, stream,
                             atom_numbers, dd_atom_local_label, crd, dd_crd,
                             vel, dd_vel, dd_atom_local_id);
#ifdef USE_MPI
        if (CONTROLLER::PP_MPI_size == 1) return;
        D_MPI_Allreduce_IN_PLACE(crd, atom_numbers * 3, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, stream);
        D_MPI_Barrier(CONTROLLER::d_pp_comm, stream);
        D_MPI_Allreduce_IN_PLACE(vel, atom_numbers * 3, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, stream);
        D_MPI_Barrier(CONTROLLER::d_pp_comm, stream);
#endif
    }
}

void MD_INFORMATION::Crd_Vel_Device_to_dd(VECTOR* dd_crd, VECTOR* dd_vel,
                                          char* dd_atom_local_label,
                                          int* dd_atom_local_id,
                                          deviceStream_t stream)
{
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Launch_Device_Kernel(
            global_crd_and_vel_to_dd,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, stream, atom_numbers,
            dd_atom_local_label, crd, dd_crd, vel, dd_vel, dd_atom_local_id);
    }
}

static __global__ void dd_frc_to_global(int atom_numbers,
                                        char* atom_local_label, VECTOR* frc,
                                        VECTOR* dd_frc, int* atom_local_id)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        if (atom_local_label[i] == 1)
        {
            frc[i] = dd_frc[atom_local_id[i]];
        }
        else
        {
            frc[i] = {0, 0, 0};
        }
    }
}

void MD_INFORMATION::Frc_dd_to_Host(VECTOR* dd_frc, char* dd_atom_local_label,
                                    int* dd_atom_local_id,
                                    deviceStream_t stream)
{
    if (CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
    {
        return;
    }

    if (CONTROLLER::PP_MPI_size == 1)
    {
        deviceMemcpy(frc, dd_frc, sizeof(VECTOR) * atom_numbers,
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(this->force, this->frc,
                     sizeof(VECTOR) * this->atom_numbers,
                     deviceMemcpyDeviceToHost);
    }

    else
    {
#ifdef USE_MPI
        deviceMemset(frc, 0, sizeof(VECTOR) * atom_numbers);
        Launch_Device_Kernel(dd_frc_to_global, (atom_numbers + 255) / 256, 256,
                             0, stream, atom_numbers, dd_atom_local_label, frc,
                             dd_frc, dd_atom_local_id);
        D_MPI_Allreduce_IN_PLACE(frc, atom_numbers * 3, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, stream);
        D_MPI_Barrier(CONTROLLER::d_pp_comm, stream);
#endif
        deviceMemcpy(this->force, this->frc,
                     sizeof(VECTOR) * this->atom_numbers,
                     deviceMemcpyDeviceToHost);
    }
}

void MD_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (mode == RERUN)
    {
        controller->Step_Print("frame", this->sys.steps + 1);
        controller->Step_Print("temperature", sys.h_temperature);
    }
    else
    {
        controller->Step_Print("step", this->sys.steps);
        controller->Step_Print("time", this->sys.Get_Current_Time(false));
        controller->Step_Print("temperature", sys.h_temperature);
    }
}

void MD_INFORMATION::Get_pressure(CONTROLLER* controller, float dd_atom_numbers,
                                  VECTOR* dd_vel, float* dd_d_mass,
                                  LTMatrix3* dd_d_virial, deviceStream_t stream)
{
    if (need_pressure == 0) return;
    // PP进程下计算势能与动能贡献，在单进程情况下PME势能贡献已考虑在内
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        sys.Get_Potential_to_stress(controller, dd_atom_numbers, dd_d_virial);
        sys.Get_Kinetic_to_stress(controller, dd_atom_numbers, dd_vel,
                                  dd_d_mass);
    }
    // PME进程计算势能贡献（如果有的话）
    else
    {
        sys.Get_Potential_to_stress(controller, atom_numbers,
                                    d_atom_virial_tensor);
    }
#ifdef USE_MPI
    D_MPI_Allreduce_IN_PLACE(sys.d_stress, 6, D_MPI_FLOAT, D_MPI_SUM,
                             CONTROLLER::D_MPI_COMM_WORLD, stream);
    D_MPI_Barrier(CONTROLLER::D_MPI_COMM_WORLD, stream);
#endif
    deviceMemcpy(&sys.h_stress, sys.d_stress, sizeof(LTMatrix3),
                 deviceMemcpyDeviceToHost);
    sys.h_pressure =
        (sys.h_stress.a11 + sys.h_stress.a22 + sys.h_stress.a33) / 3;
}
