#pragma once

#include "../../plugins/prips/include/sponge_plugin_api.h"
#include "../Domain_decomposition/Domain_decomposition.h"
#include "../MD_core/MD_core.h"
#include "../collective_variable/collective_variable.h"
#include "../common.h"
#include "../control.h"
#include "../neighbor_list/neighbor_list.h"

typedef std::vector<std::vector<std::string>> CVRegisterString;
typedef CVRegisterString (*CVRegisterFunction)();
typedef void (*cv_init_func)(COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*);
typedef void (*cv_compute_func)(int, UNSIGNED_INT_VECTOR*, VECTOR, VECTOR*,
                                VECTOR, int, int);
typedef std::string (*NameFunction)();
typedef std::string (*VersionCheckFunction)(int);
typedef void (*InitialStableFunction)(const SPONGE_PLUGIN_API* api);
typedef void (*InitialFunction)(MD_INFORMATION* md_info, CONTROLLER* controller,
                                NEIGHBOR_LIST* neighbor_list,
                                COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                                CV_MAP_TYPE*, CV_INSTANCE_TYPE*);
typedef void (*SetDomainInformationFunction)(DOMAIN_INFORMATION* dd);
typedef void (*SetBackendDeviceTypeFunction)(int device_type);
typedef void (*RuntimeFunction)();

struct SPONGE_PLUGIN
{
    static std::map<
        std::string,
        std::function<void(COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*)>>
        cv_init_functions;
    static std::map<std::string,
                    std::function<void(int, UNSIGNED_INT_VECTOR*, VECTOR,
                                       VECTOR*, VECTOR, int, int)>>
        cv_compute_functions;

    int plugin_numbers = 0;
    HMODULE* plugin_handles = NULL;

    int after_init_func_numbers = 0;
    RuntimeFunction* after_init_funcs = NULL;

    int force_func_numbers = 0;
    RuntimeFunction* force_funcs = NULL;

    int print_func_numbers = 0;
    RuntimeFunction* print_funcs = NULL;

    int set_domain_info_func_numbers = 0;
    SetDomainInformationFunction* set_domain_info_funcs = NULL;

    void Initial(MD_INFORMATION* md_info, CONTROLLER* controller,
                 COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                 NEIGHBOR_LIST* neighbor_list);
    void Set_Domain_Information(DOMAIN_INFORMATION* dd);
    void After_Initial();
    void Calculate_Force();
    void Mdout_Print();
};
