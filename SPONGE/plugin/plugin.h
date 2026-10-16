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
typedef void (*InitialLocalMPFunction)(const SPONGE_LOCAL_MP_API_V1* api);
typedef void (*InitialFunction)(MD_INFORMATION* md_info, CONTROLLER* controller,
                                NEIGHBOR_LIST* neighbor_list,
                                COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                                CV_MAP_TYPE*, CV_INSTANCE_TYPE*);
typedef void (*SetDomainInformationFunction)(DOMAIN_INFORMATION* dd);
typedef void (*SetBackendDeviceTypeFunction)(int device_type);
typedef void (*RuntimeFunction)();

class MLIP_NEIGHBOR_LIST;

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
    bool Validate_Local_MP_Capabilities();
    // PP-only admission before first force/feature exchange; no hot-path
    // collective after success. Main aborts the world on a false result.
    bool Ensure_Local_MP_Setup();
    bool Has_Local_MP_Provider() const;
    bool Get_Mlip_Neighbor_Parameters(double& cutoff, double& skin) const;
    uint64_t Get_Domain_Epoch() const;
    void Invalidate_Mlip_Neighbor_View();
    // Caller completes device production before publishing borrowed pointers.
    void Publish_Mlip_Neighbor_View(const MLIP_NEIGHBOR_LIST& graph,
                                    int owned_count, int ghost_count);
    // Immutable declarations are queried after all initialization callbacks.
    bool classic_neighbors_required = false;
    bool Uses_Classic_Neighbors() const { return classic_neighbors_required; }
    void After_Initial();
    void Calculate_Force();
    void Mdout_Print();

   private:
    bool local_mp_setup_checked = false;
    bool local_mp_setup_ready = false;
    std::vector<int> after_init_plugin_indices;
    std::vector<int> force_plugin_indices;
    std::vector<int> print_plugin_indices;
    std::vector<int> domain_plugin_indices;
    std::vector<SPONGE_GET_RUNTIME_ERROR_V1> checked_error_queries;
    void Check_Runtime_Error(int plugin_index);
};
