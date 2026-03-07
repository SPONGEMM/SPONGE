#include "soft_wall.h"

static SOFT_WALL* Read_One_Force(CONTROLLER* controller, std::string section,
                                 Configuration_Reader* cfg)
{
    SOFT_WALL* force = new SOFT_WALL;
    strcpy(force->module_name, section.c_str());
    controller->printf("    reading the soft wall named %s\n",
                       force->module_name);
    if (!cfg->Key_Exist(section, "potential"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Read_One_Force (listed_forces.cu)",
            string_format("Reason:\n\tThe potential of the listed force "
                          "%FORCE% is required ([[ potential ]])\n",
                          {{"FORCE", section}})
                .c_str());
    }
    force->source_code = cfg->Get_Value(section, "potential");
    force->Compile(controller);
    controller->printf("    end reading the listed force named %s\n",
                       force->module_name);
    return force;
}

void SOFT_WALLS::Initial(CONTROLLER* controller, int atom_numbers,
                         const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "soft_walls");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("START INITIALIZING SOFT WALLS:\n");
        Configuration_Reader cfg;
        cfg.Open(controller->Command(this->module_name, "in_file"));
        cfg.Close();
        if (!cfg.error_reason.empty())
        {
            cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "SOFT_WALLS::Initial",
                                           cfg.error_reason.c_str());
        }
        for (std::string section : cfg.sections)
        {
            forces.push_back(Read_One_Force(controller, section, &cfg));
        }
        for (auto s : cfg.value_unused)
        {
            std::string error_reason = string_format(
                "Reason:\n\t[[ %s% ]] should not be one of the keys of the "
                "soft wall [[[ %a% ]]]",
                {{"s", s.second}, {"a", s.first}});
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "SOFT_WALLS::Initial",
                                           error_reason.c_str());
        }
    }
    if (forces.size() != 0)
    {
        is_initialized = 1;
        for (auto force : forces)
        {
            force->Initial(atom_numbers);
        }
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        for (auto force : forces)
        {
            controller->Step_Print_Initial(force->module_name, "%.2f");
        }
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING SOFT WALLS\n\n");
    }
    else
    {
        controller->printf("SOFT WALLS ARE NOT INITIALIZED\n\n");
    }
}

void SOFT_WALLS::Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc,
                               int need_potential, float* atom_energy)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            force->Compute_Force(atom_numbers, crd, frc, need_potential,
                                 atom_energy);
        }
    }
}

void SOFT_WALLS::Step_Print(CONTROLLER* controller)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            controller->Step_Print(force->module_name, force->h_sum_energy);
        }
    }
}

void SOFT_WALL::Compile(CONTROLLER* controller)
{
    std::string full_source_code =
        string_format(R"JIT(#if defined(__CUDACC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_CUDA
#define USE_CUDA
#endif
#elif defined(__HIPCC__) || defined(__HIPCC_RTC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_HIP
#define USE_HIP
#endif
#endif
#include "common.h"
extern "C" __global__ void soft_wall_energy(int atom_numbers, VECTOR* crd, VECTOR* frc, float* atom_ene, float* item_energy, bool need_potential)
{
#ifdef USE_GPU
    int atom_i = threadIdx.x + blockIdx.x * blockDim.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR local_crd = crd[atom_i];
        SADfloat<3> x(local_crd.x, 0);
        SADfloat<3> y(local_crd.y, 1);
        SADfloat<3> z(local_crd.z, 2);
        SADfloat<3> E;
        %source_code%
        VECTOR local_frc = frc[atom_i];
        local_frc.x -= E.dval[0];
        local_frc.y -= E.dval[1];
        local_frc.z -= E.dval[2];
        frc[atom_i] = local_frc;
        if (need_potential)
        {
            atom_ene[atom_i] += E.val;
            item_energy[atom_i] = E.val;
        }
    }
}
)JIT",
                      {{"source_code", source_code}});
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "SOFT_WALL::Compile",
                                       force_function.error_reason.c_str());
    }
}

void SOFT_WALL::Initial(int atom_numbers)
{
    Device_Malloc_Safely((void**)&item_energy, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&sum_energy, sizeof(float));
}

void SOFT_WALL::Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc,
                              int need_potential, float* atom_energy)
{
    force_function({(atom_numbers + 1023u) / 1024u, 1, 1}, {1024, 1, 1}, 0, 0,
                   {&atom_numbers, &crd, &frc, &atom_energy, &item_energy,
                    &need_potential});
    if (need_potential)
    {
        Sum_Of_List(item_energy, sum_energy, atom_numbers);
#ifdef USE_MPI
        D_MPI_Allreduce_IN_PLACE(sum_energy, 1, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, 0);
#endif
        h_sum_energy = NAN;
        deviceMemcpy(&h_sum_energy, sum_energy, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
}
