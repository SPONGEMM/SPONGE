#include "combine.h"

REGISTER_CV_STRUCTURE(CV_COMBINE, "combination", 0);

void CV_COMBINE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                         int atom_numbers, const char* module_name)
{
    cv_lists = manager->Ask_For_CV(module_name, -1, 0, 2);
    void** temp_ptr;
    Malloc_Safely((void**)&temp_ptr, sizeof(void*) * cv_lists.size());
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->d_value;
    }
    Device_Malloc_Safely((void**)&d_cv_values,
                         sizeof(float*) * cv_lists.size());
    deviceMemcpy(d_cv_values, temp_ptr, sizeof(float*) * cv_lists.size(),
                 deviceMemcpyHostToDevice);
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->crd_grads;
    }
    Device_Malloc_Safely((void**)&cv_crd_grads,
                         sizeof(VECTOR*) * cv_lists.size());
    deviceMemcpy(cv_crd_grads, temp_ptr, sizeof(VECTOR*) * cv_lists.size(),
                 deviceMemcpyHostToDevice);
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->virial;
    }
    Device_Malloc_Safely((void**)&cv_virials,
                         sizeof(LTMatrix3*) * cv_lists.size());
    deviceMemcpy(cv_virials, temp_ptr, sizeof(LTMatrix3*) * cv_lists.size(),
                 deviceMemcpyHostToDevice);
    free(temp_ptr);
    Device_Malloc_Safely((void**)&df_dcv, sizeof(float) * cv_lists.size());
    if (!manager->Command_Exist(module_name, "function"))
    {
        manager->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "CV_COMBINE::Initial",
            "Reason:\n\tNeed to specify the function of the CV combination");
    }

    std::string function_code = string_replace(
        manager->Original_Command(module_name, "function"), ")(", ", ");
    std::string sadf = "SADfloat<" + std::to_string(cv_lists.size()) + ">";
    std::string source_code = R"JIT(
#include "common.h"
extern "C" __global__ void cv_combine_first_step( const float **CV_values, const LTMatrix3 **cv_box_grads, 
float* out_value, float* out_df_dcv, LTMatrix3* box_grads)
{
    %PARM_DEC%
    %sadf% %NAME% = %FUNC_CODE%;
    out_value[0] = %NAME%.val;
    LTMatrix3 local_box_grads = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < %N%; i++)
    {
        out_df_dcv[i] = %NAME%.dval[i];
        local_box_grads = local_box_grads + %NAME%.dval[i] * cv_box_grads[i][0];
    }
    box_grads[0] = local_box_grads;
}
)JIT";
    StringVector cv_names;
    for (auto cv : cv_lists)
    {
        cv_names.push_back(cv->module_name);
    }
    std::string endl = "\n    ";
    std::string PARM_DEC = string_join(
        sadf + " %0%(CV_values[%INDEX%][0], %INDEX%);", endl, {cv_names});
    source_code =
        string_format(source_code, {{"sadf", sadf},
                                    {"PARM_DEC", PARM_DEC},
                                    {"FUNC_CODE", function_code},
                                    {"NAME", module_name},
                                    {"N", std::to_string(cv_lists.size())}});

    first_step.Compile(source_code);
    if (!first_step.error_reason.empty())
    {
        first_step.error_reason = "Reason:\n" + first_step.error_reason;
        manager->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                    "CV_COMBINE::Initial (first step)",
                                    first_step.error_reason.c_str());
    }

    source_code = string_format(
        R"JIT(#if defined(__CUDACC__)
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
extern "C" __global__ void cv_combine_second_step(const int atom_numbers, const VECTOR **CV_crd_grads, const float* out_df_dcv, VECTOR* crd_grads)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR local_crd_grads = {0.0f, 0.0f, 0.0f};
        %ADD_GRADS%
        crd_grads[atom_i] = local_crd_grads;
    }
}
)JIT",
        {{"ADD_GRADS",
          string_join("local_crd_grads = local_crd_grads + out_df_dcv[%INDEX%] "
                      "* CV_crd_grads[%INDEX%][atom_i];",
                      endl, {cv_names})}});
    second_step.Compile(source_code);
    if (!second_step.error_reason.empty())
    {
        second_step.error_reason = "Reason:\n" + second_step.error_reason;
        manager->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                    "CV_COMBINE::Initial (second step)",
                                    second_step.error_reason.c_str());
    }
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_COMBINE::Compute(int atom_numbers, VECTOR* crd, const LTMatrix3 cell,
                         const LTMatrix3 rcell, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need)
    {
        for (auto cv : cv_lists)
        {
            cv->Compute(atom_numbers, crd, cell, rcell, CV_NEED_ALL, step);
        }
        for (auto cv : cv_lists)
        {
            deviceStreamSynchronize(cv->device_stream);
        }
        first_step({1, 1, 1}, {1, 1, 1}, this->device_stream, 0,
                   {&d_cv_values, &cv_virials, &d_value, &df_dcv, &virial});
        unsigned int blocks = (atom_numbers + 1023u) / 1024u;
        second_step({blocks, 1, 1}, {1024, 1, 1}, this->device_stream, 0,
                    {&atom_numbers, &cv_crd_grads, &df_dcv, &crd_grads});
        deviceMemcpyAsync(&value, d_value, sizeof(float),
                          deviceMemcpyDeviceToHost, this->device_stream);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
