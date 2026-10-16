#include "plugin.h"

#include <stdexcept>

#include "../local_mp/local_mp_service.h"
#include "../utils/mlip_geometry.hpp"

namespace
{
MD_INFORMATION* g_plugin_md_info = NULL;
CONTROLLER* g_plugin_controller = NULL;
NEIGHBOR_LIST* g_plugin_neighbor_list = NULL;
SPONGE_PLUGIN_API g_prips_api = {};
SPONGE_LOCAL_MP_API_V1 g_local_mp_api = {};
bool g_local_mp_plugin_loaded = false;
struct MlipCapabilities
{
    bool registered = false;
    double cutoff = 0.0;
    double skin = 0.0;
    int depth = 0;
    int dtype = 0;
    uint64_t hash = 0;
};
MlipCapabilities g_mlip_capabilities;
int g_current_plugin_index = -1;
int g_mlip_registration_owner = -1;
bool g_multiple_mlip_registration_owners = false;
}  // namespace

std::map<std::string,
         std::function<void(COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*)>>
    SPONGE_PLUGIN::cv_init_functions;
std::map<std::string, std::function<void(int, UNSIGNED_INT_VECTOR*, VECTOR,
                                         VECTOR*, VECTOR, int, int)>>
    SPONGE_PLUGIN::cv_compute_functions;

static std::string DlErrorString()
{
#ifdef _WIN32
    return std::to_string(static_cast<unsigned long>(dlerror()));
#else
    const char* err = dlerror();
    return err == NULL ? std::string() : std::string(err);
#endif
}

static int PluginBackendDeviceType()
{
#ifdef USE_HIP
    return 10;  // kDLROCM
#elif defined(USE_CUDA)
    return 2;  // kDLCUDA
#else
    return 1;  // kDLCPU
#endif
}

namespace
{
const char* PluginGetCommand(const char* key)
{
    if (g_plugin_controller == NULL || !g_plugin_controller->Command_Exist(key))
    {
        return NULL;
    }
    return g_plugin_controller->Command(key);
}

void PluginLogMessage(const char* message)
{
    if (g_plugin_controller == NULL || message == NULL) return;
    g_plugin_controller->printf("%s", message);
}

int PluginGetMPIRank() { return CONTROLLER::MPI_rank; }

int PluginGetAtomNumbers()
{
    return g_plugin_md_info == NULL ? 0 : g_plugin_md_info->atom_numbers;
}

int PluginGetSteps()
{
    return g_plugin_md_info == NULL ? 0 : g_plugin_md_info->sys.steps;
}

void* PluginGetCoordinatePtr()
{
    return g_plugin_md_info == NULL ? NULL : g_plugin_md_info->crd;
}

void* PluginGetForcePtr()
{
    return g_plugin_md_info == NULL ? NULL : g_plugin_md_info->frc;
}

int PluginGetNeighborListMaxNumbers()
{
    return g_plugin_neighbor_list == NULL
               ? 0
               : g_plugin_neighbor_list->max_neighbor_numbers;
}

int PluginGetNeighborListCount(int atom_index)
{
    if (g_plugin_neighbor_list == NULL ||
        g_plugin_neighbor_list->h_nl == NULL || g_plugin_md_info == NULL ||
        atom_index < 0 || atom_index >= g_plugin_md_info->atom_numbers)
    {
        return 0;
    }
    return g_plugin_neighbor_list->h_nl[atom_index].atom_numbers;
}

void* PluginGetNeighborListIndexPtr()
{
    if (g_plugin_neighbor_list == NULL || g_plugin_neighbor_list->h_nl == NULL)
    {
        return NULL;
    }
    return g_plugin_neighbor_list->h_nl->atom_serial;
}

int PluginGetLocalAtomNumbers()
{
    return Local_MP_Service().Domain() == NULL
               ? 0
               : Local_MP_Service().Domain()->atom_numbers;
}

int PluginGetLocalGhostNumbers()
{
    return Local_MP_Service().Domain() == NULL
               ? 0
               : Local_MP_Service().Domain()->ghost_numbers;
}

int PluginGetLocalPPRank()
{
    return Local_MP_Service().Domain() == NULL
               ? 0
               : Local_MP_Service().Domain()->pp_rank;
}

int PluginRegisterMlipCapabilities(double cutoff_angstrom, double skin_angstrom,
                                   int interaction_depth, int feature_dtype,
                                   uint64_t capability_hash)
{
    if (g_current_plugin_index >= 0)
    {
        if (g_mlip_registration_owner < 0)
            g_mlip_registration_owner = g_current_plugin_index;
        else if (g_mlip_registration_owner != g_current_plugin_index)
            g_multiple_mlip_registration_owners = true;
    }
    g_mlip_capabilities = {true,          cutoff_angstrom,
                           skin_angstrom, interaction_depth,
                           feature_dtype, capability_hash};
    // Invalid declarations are rejected collectively after PP setup.
    return 0;
}

int PluginGetPPRank()
{
    return CONTROLLER::PP_MPI_rank >= 0 &&
                   CONTROLLER::PP_MPI_rank < CONTROLLER::PP_MPI_size
               ? CONTROLLER::PP_MPI_rank
               : -1;
}

int PluginGetPPSize() { return CONTROLLER::PP_MPI_size; }

int PluginGetDeviceIndex()
{
    return g_plugin_controller == NULL ? -1
                                       : g_plugin_controller->working_device;
}

uint64_t PluginGetDomainEpoch() { return Local_MP_Service().Domain_Epoch(); }

int PluginGetNeedPotential()
{
    return g_plugin_md_info == NULL ? 0 : g_plugin_md_info->need_potential;
}

int PluginGetNeedPressure()
{
    return g_plugin_md_info == NULL ? 0 : g_plugin_md_info->need_pressure;
}

int PluginGetCell(float cell_row_major[9], uint8_t* pbc_mask)
{
    if (g_plugin_md_info == NULL || cell_row_major == NULL || pbc_mask == NULL)
        return 1;
    const LTMatrix3& cell = g_plugin_md_info->pbc.boundary.cell;
    cell_row_major[0] = cell.a11;
    cell_row_major[1] = 0.0f;
    cell_row_major[2] = 0.0f;
    cell_row_major[3] = cell.a21;
    cell_row_major[4] = cell.a22;
    cell_row_major[5] = 0.0f;
    cell_row_major[6] = cell.a31;
    cell_row_major[7] = cell.a32;
    cell_row_major[8] = cell.a33;
    *pbc_mask =
        g_plugin_md_info->pbc.boundary.policy == BoundaryPolicy::Periodic ? 0x7
                                                                          : 0;
    return 0;
}

int PluginGetLocalMaxAtomNumbers()
{
    return Local_MP_Service().Domain() == NULL
               ? 0
               : Local_MP_Service().Domain()->max_atom_numbers;
}

void* PluginGetAtomLocalPtr()
{
    return Local_MP_Service().Domain() == NULL
               ? NULL
               : Local_MP_Service().Domain()->atom_local;
}

void* PluginGetAtomLocalLabelPtr()
{
    return Local_MP_Service().Domain() == NULL
               ? NULL
               : Local_MP_Service().Domain()->atom_local_label;
}

void* PluginGetAtomLocalIdPtr()
{
    return Local_MP_Service().Domain() == NULL
               ? NULL
               : Local_MP_Service().Domain()->atom_local_id;
}

void* PluginGetLocalCoordinatePtr()
{
    return Local_MP_Service().Domain() == NULL
               ? NULL
               : Local_MP_Service().Domain()->crd;
}

void* PluginGetLocalForcePtr()
{
    return Local_MP_Service().Domain() == NULL
               ? NULL
               : Local_MP_Service().Domain()->frc;
}

const SPONGE_PLUGIN_API* BuildPripsApi()
{
    g_prips_api.api_version = SPONGE_PRIPS_API_VERSION;
    g_prips_api.device_type = PluginBackendDeviceType();
    g_prips_api.get_command = PluginGetCommand;
    g_prips_api.log_message = PluginLogMessage;
    g_prips_api.get_mpi_rank = PluginGetMPIRank;
    g_prips_api.get_atom_numbers = PluginGetAtomNumbers;
    g_prips_api.get_steps = PluginGetSteps;
    g_prips_api.get_coordinate_ptr = PluginGetCoordinatePtr;
    g_prips_api.get_force_ptr = PluginGetForcePtr;
    g_prips_api.get_neighbor_list_max_numbers = PluginGetNeighborListMaxNumbers;
    g_prips_api.get_neighbor_list_count = PluginGetNeighborListCount;
    g_prips_api.get_neighbor_list_index_ptr = PluginGetNeighborListIndexPtr;
    g_prips_api.get_local_atom_numbers = PluginGetLocalAtomNumbers;
    g_prips_api.get_local_ghost_numbers = PluginGetLocalGhostNumbers;
    g_prips_api.get_local_pp_rank = PluginGetLocalPPRank;
    g_prips_api.get_local_max_atom_numbers = PluginGetLocalMaxAtomNumbers;
    g_prips_api.get_atom_local_ptr = PluginGetAtomLocalPtr;
    g_prips_api.get_atom_local_label_ptr = PluginGetAtomLocalLabelPtr;
    g_prips_api.get_atom_local_id_ptr = PluginGetAtomLocalIdPtr;
    g_prips_api.get_local_coordinate_ptr = PluginGetLocalCoordinatePtr;
    g_prips_api.get_local_force_ptr = PluginGetLocalForcePtr;
    return &g_prips_api;
}

uint64_t PluginGetMlipNeighborEpoch()
{
    return Local_MP_Service().Neighbor_Epoch();
}

int PluginGetMlipNeighborView(SPONGE_MLIP_NEIGHBOR_VIEW* view)
{
    return Local_MP_Service().Neighbor_View(view);
}

#ifdef USE_CUDA
int PluginExchangeOnStream(float* rows, int row_count, int width, int reverse,
                           uint64_t epoch, uintptr_t producer_stream)
{
    return Local_MP_Service().Exchange_On_Stream(
        rows, row_count, width, reverse, epoch, producer_stream);
}

int PluginSubmitOwnedOutputs(uint64_t domain_epoch, int owned_count,
                             const float* forces, const float* energy,
                             const float* virial9, uintptr_t producer_stream)
{
    return Local_MP_Service().Submit_Owned_Outputs(
        domain_epoch, owned_count, forces, energy, virial9, producer_stream);
}

int PluginAssembleSelectedEdgeGradients(
    uint64_t domain_epoch, uint64_t neighbor_epoch, int selected_count,
    const int64_t* selected_ids, const float* gradients_model,
    float* owned_forces_model, float* virial_model_9, int need_virial,
    double length_to_angstrom, uintptr_t producer_stream)
{
    return Local_MP_Service().Assemble_Selected_Edge_Gradients(
        domain_epoch, neighbor_epoch, selected_count, selected_ids,
        gradients_model, owned_forces_model, virial_model_9, need_virial,
        length_to_angstrom, producer_stream);
}
#endif

const SPONGE_LOCAL_MP_API_V1* BuildLocalMPApi()
{
    g_local_mp_api = {};
    g_local_mp_api.api_version = SPONGE_LOCAL_MP_API_VERSION;
    g_local_mp_api.prips_api = BuildPripsApi();
    g_local_mp_api.register_mlip_capabilities = PluginRegisterMlipCapabilities;
    g_local_mp_api.get_pp_rank = PluginGetPPRank;
    g_local_mp_api.get_pp_size = PluginGetPPSize;
    g_local_mp_api.get_device_index = PluginGetDeviceIndex;
    g_local_mp_api.get_domain_epoch = PluginGetDomainEpoch;
    g_local_mp_api.get_need_potential = PluginGetNeedPotential;
    g_local_mp_api.get_need_pressure = PluginGetNeedPressure;
    g_local_mp_api.get_cell = PluginGetCell;
    g_local_mp_api.get_mlip_neighbor_epoch = PluginGetMlipNeighborEpoch;
    g_local_mp_api.get_mlip_neighbor_view = PluginGetMlipNeighborView;
#ifdef USE_CUDA
    g_local_mp_api.exchange_on_stream = PluginExchangeOnStream;
    g_local_mp_api.assemble_selected_edge_gradients =
        PluginAssembleSelectedEdgeGradients;
    g_local_mp_api.submit_owned_outputs = PluginSubmitOwnedOutputs;
#endif
    return &g_local_mp_api;
}
}  // namespace

void SPONGE_PLUGIN::Initial(MD_INFORMATION* md_info, CONTROLLER* controller,
                            COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                            NEIGHBOR_LIST* neighbor_list)
{
    classic_neighbors_required = false;
    g_mlip_capabilities = {};
    g_current_plugin_index = -1;
    g_mlip_registration_owner = -1;
    g_multiple_mlip_registration_owners = false;
    local_mp_setup_checked = local_mp_setup_ready = false;
    after_init_plugin_indices.clear();
    force_plugin_indices.clear();
    print_plugin_indices.clear();
    domain_plugin_indices.clear();
    checked_error_queries.clear();
    g_local_mp_plugin_loaded = false;
    Local_MP_Service().Initial(md_info, controller);
    if (!controller->Command_Exist("plugin"))
    {
        return;
    }
    if (md_info->pbc.boundary.policy == BoundaryPolicy::Open)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "SPONGE_PLUGIN::Initial",
            "Reason:\n\tplugins currently require periodic boundary "
            "conditions because plugin API v2 does not expose a boundary "
            "policy\n");
    }

    classic_neighbors_required = true;
    controller->printf("START INITIALIZING SPONGE PLUGIN:\n");
    plugin_numbers = 0;
    g_plugin_md_info = md_info;
    g_plugin_controller = controller;
    g_plugin_neighbor_list = neighbor_list;

    std::string command(controller->Original_Command("plugin"));
    auto last_pos = command.find_first_not_of(" ", 0);
    auto pos = command.find_first_of(" ", last_pos);
    while (pos != std::string::npos || last_pos != std::string::npos)
    {
        plugin_numbers += 1;
        last_pos = command.find_first_not_of(" ", pos);
        pos = command.find_first_of(" ", last_pos);
    }

    controller->printf("%d plugin(s) to load\n", plugin_numbers);
    checked_error_queries.assign(plugin_numbers, nullptr);
    Malloc_Safely((void**)&plugin_handles, sizeof(HMODULE) * plugin_numbers);
    Malloc_Safely((void**)&after_init_funcs,
                  sizeof(RuntimeFunction) * plugin_numbers);
    Malloc_Safely((void**)&force_funcs,
                  sizeof(RuntimeFunction) * plugin_numbers);
    Malloc_Safely((void**)&print_funcs,
                  sizeof(RuntimeFunction) * plugin_numbers);
    Malloc_Safely((void**)&set_domain_info_funcs,
                  sizeof(SetDomainInformationFunction) * plugin_numbers);

    int count = 0;
    std::string plugin_name, plugin_version, version_check_error;
    char plugin_path[CHAR_LENGTH_MAX];
    NameFunction name_func, version_func;
    VersionCheckFunction version_check_func;
    SetBackendDeviceTypeFunction set_backend_device_type_func;
    InitialStableFunction stable_initial_func;
    InitialLocalMPFunction local_mp_initial_func;

    last_pos = command.find_first_not_of(" ", 0);
    pos = command.find_first_of(" ", last_pos);
    while (pos != std::string::npos || last_pos != std::string::npos)
    {
        int funcs_loaded = 1;
        sscanf(command.substr(last_pos, pos - last_pos).c_str(), "%s",
               plugin_path);

#ifdef _WIN32
        constexpr int dlopen_mode = 0;
#else
        constexpr int dlopen_mode = RTLD_LAZY | RTLD_GLOBAL;
#endif
        plugin_handles[count] = dlopen(plugin_path, dlopen_mode);
        if (plugin_handles[count] == NULL)
        {
            std::string error_reason = "Reason:\n\tOpen Dynamic Library from ";
            error_reason += plugin_path;
            error_reason += " failed\n";
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        name_func = (NameFunction)dlsym(plugin_handles[count], "Name");
        if (name_func == NULL)
        {
            std::string error_reason =
                "Reason:\n\tFind the name of the plugin from ";
            error_reason += plugin_path;
            error_reason += " failed\n";
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        plugin_name = name_func();
        version_func = (NameFunction)dlsym(plugin_handles[count], "Version");
        if (version_func == NULL)
        {
            std::string error_reason =
                "Reason:\n\tFind the version of the plugin from ";
            error_reason += plugin_path;
            error_reason += " (" + plugin_name + ") failed\n";
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        plugin_version = version_func();
        version_check_func =
            (VersionCheckFunction)dlsym(plugin_handles[count], "Version_Check");
        if (version_check_func == NULL)
        {
            std::string error_reason =
                "Reason:\n\tFind the version check function of the plugin "
                "from ";
            error_reason += plugin_path;
            error_reason += " (" + plugin_name + " version: " + plugin_version +
                            ") failed\n";
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        stable_initial_func = (InitialStableFunction)dlsym(
            plugin_handles[count], "Initial_Stable");
        local_mp_initial_func = (InitialLocalMPFunction)dlsym(
            plugin_handles[count], "Initial_Local_MP_V1");
        version_check_error = version_check_func(
            local_mp_initial_func != NULL ? SPONGE_LOCAL_MP_API_VERSION
            : stable_initial_func != NULL ? SPONGE_PRIPS_API_VERSION
                                          : controller->last_modify_date);
        if (!version_check_error.empty())
        {
            std::string error_reason =
                "Reason:\n\tThe version check of the plugin from ";
            error_reason += plugin_path;
            error_reason += " (" + plugin_name + " version: " + plugin_version +
                            ") failed\n" + version_check_error;
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        controller->printf(
            "Plugin %d:\n    name: %s\n    version: %s\n    path: %s\n    "
            "functions loaded: ",
            plugin_numbers, plugin_name.c_str(), plugin_version.c_str(),
            plugin_path);

        InitialFunction func =
            (InitialFunction)dlsym(plugin_handles[count], "Initial");
        if (func == NULL && stable_initial_func == NULL &&
            local_mp_initial_func == NULL)
        {
            std::string error_reason =
                "Reason:\n\tFind the initial function of the plugin from ";
            error_reason += plugin_path;
            error_reason += " (" + plugin_name + " version: " + plugin_version +
                            ") failed\n";
            error_reason += DlErrorString();
            controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                           "SPONGE_PLUGIN::Initial",
                                           error_reason.c_str());
        }

        controller->printf(" Initial");

        auto enable_checked =
            reinterpret_cast<SPONGE_ENABLE_CHECKED_RUNTIME_ERRORS_V1>(dlsym(
                plugin_handles[count], "Enable_Checked_Runtime_Errors_V1"));
        auto get_error = reinterpret_cast<SPONGE_GET_RUNTIME_ERROR_V1>(
            dlsym(plugin_handles[count], "Get_Runtime_Error_V1"));
        if ((enable_checked == nullptr) != (get_error == nullptr))
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "SPONGE_PLUGIN::Initial",
                "Plugin checked-error protocol requires both named exports.");
        if (enable_checked != nullptr)
        {
            checked_error_queries[count] = get_error;
            if (enable_checked() != 0)
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "SPONGE_PLUGIN::Initial",
                    "Plugin could not enable checked runtime errors.");
            Check_Runtime_Error(count);
        }
        g_current_plugin_index = count;

        set_backend_device_type_func = (SetBackendDeviceTypeFunction)dlsym(
            plugin_handles[count], "Set_Backend_Device_Type");
        if (set_backend_device_type_func != NULL)
        {
            funcs_loaded += 1;
            controller->printf(" Set_Backend_Device_Type");
            set_backend_device_type_func(PluginBackendDeviceType());
            Check_Runtime_Error(count);
        }

        after_init_funcs[after_init_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "After_Initial");
        if (after_init_funcs[after_init_func_numbers] != NULL)
        {
            after_init_plugin_indices.push_back(count);
            funcs_loaded += 1;
            after_init_func_numbers += 1;
            controller->printf(" After_Initial");
        }

        force_funcs[force_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "Calculate_Force");
        if (force_funcs[force_func_numbers] != NULL)
        {
            force_plugin_indices.push_back(count);
            funcs_loaded += 1;
            force_func_numbers += 1;
            controller->printf(" Calculate_Force");
        }

        print_funcs[print_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "Mdout_Print");
        if (print_funcs[print_func_numbers] != NULL)
        {
            print_plugin_indices.push_back(count);
            funcs_loaded += 1;
            print_func_numbers += 1;
            controller->printf(" Mdout_Print");
        }

        set_domain_info_funcs[set_domain_info_func_numbers] =
            (SetDomainInformationFunction)dlsym(plugin_handles[count],
                                                "Set_Domain_Information");
        if (set_domain_info_funcs[set_domain_info_func_numbers] != NULL)
        {
            domain_plugin_indices.push_back(count);
            funcs_loaded += 1;
            set_domain_info_func_numbers += 1;
            controller->printf(" Set_Domain_Information");
        }

        controller->printf(" (%d in total)\n", funcs_loaded);
        if (local_mp_initial_func != NULL)
        {
            g_local_mp_plugin_loaded = true;
            local_mp_initial_func(BuildLocalMPApi());
            Check_Runtime_Error(count);
        }
        else if (stable_initial_func != NULL)
        {
            stable_initial_func(BuildPripsApi());
            Check_Runtime_Error(count);
        }
        else
        {
            func(md_info, controller, neighbor_list, cv_controller, CV_MAP,
                 CV_INSTANCE_MAP);
            Check_Runtime_Error(count);
        }

        g_current_plugin_index = -1;
        count += 1;
        last_pos = command.find_first_not_of(" ", pos);
        pos = command.find_first_of(" ", last_pos);
    }

    controller->printf("END INITIALIZING SPONGE PLUGIN\n\n");
}

void SPONGE_PLUGIN::Set_Domain_Information(DOMAIN_INFORMATION* dd)
{
    Local_MP_Service().Set_Domain(dd);
    for (int i = 0; i < set_domain_info_func_numbers; i++)
    {
        const int owner = i < static_cast<int>(domain_plugin_indices.size())
                              ? domain_plugin_indices[i]
                              : -1;
        g_current_plugin_index = owner;
        set_domain_info_funcs[i](dd);
        Check_Runtime_Error(owner);
        g_current_plugin_index = -1;
    }
}

void SPONGE_PLUGIN::Check_Runtime_Error(int plugin_index)
{
    if (plugin_index < 0 ||
        plugin_index >= static_cast<int>(checked_error_queries.size()) ||
        checked_error_queries[plugin_index] == nullptr)
        return;
    const char* stage = nullptr;
    const char* message = nullptr;
    const int status = checked_error_queries[plugin_index](&stage, &message);
    if (status == 0) return;
    const std::string detail =
        status == 1
            ? std::string(stage ? stage : "unknown plugin stage") + ":\n" +
                  (message ? message : "missing plugin error message")
            : "Plugin returned an invalid checked-error status.";
    // Controller output is rank-zero-only; retain the failing rank's original
    // Python stage and exception before the unchanged world-abort path.
    fprintf(stderr, "Plugin runtime callback failed on rank %d:\n%s\n",
            CONTROLLER::MPI_rank, detail.c_str());
    fflush(stderr);
    g_plugin_controller->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                            "Plugin runtime callback",
                                            detail.c_str());
}

bool SPONGE_PLUGIN::Ensure_Local_MP_Setup()
{
    if (PluginGetPPRank() < 0) return true;
    if (local_mp_setup_checked) return local_mp_setup_ready;
    local_mp_setup_checked = true;
    if (!g_mlip_capabilities.registered) return local_mp_setup_ready = true;

    const int owner = g_mlip_registration_owner;
    SPONGE_PREPARE_LOCAL_MP_SETUP_V1 prepare = nullptr;
    if (!g_multiple_mlip_registration_owners && owner >= 0 &&
        owner < plugin_numbers && checked_error_queries[owner] != nullptr)
        prepare = reinterpret_cast<SPONGE_PREPARE_LOCAL_MP_SETUP_V1>(
            dlsym(plugin_handles[owner], "Prepare_Local_MP_Setup_V1"));
    int provider_ready = prepare != nullptr ? 1 : 0;
    int all_provider_ready = provider_ready;
#ifdef USE_MPI
    MPI_Allreduce(&provider_ready, &all_provider_ready, 1, MPI_INT, MPI_MIN,
                  CONTROLLER::pp_comm);
#endif
    if (!all_provider_ready) return false;

    int ready = 0;
    try
    {
        ready = prepare() == 1 ? 1 : 0;
    }
    catch (...)
    {
        ready = 0;
    }
    const char* stage = nullptr;
    const char* message = nullptr;
    const int error = checked_error_queries[owner](&stage, &message);
    if (error != 0)
    {
        ready = 0;
        // Setup failure is reduced before the world abort. Earlier callback
        // failures instead abort immediately through Check_Runtime_Error.
        fprintf(stderr, "Local-MP setup failed on rank %d [%s]: %s\n",
                CONTROLLER::MPI_rank, stage ? stage : "setup",
                message ? message : "invalid checked-error status");
    }
    int all_ready = ready;
#ifdef USE_MPI
    MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, CONTROLLER::pp_comm);
#endif
    return local_mp_setup_ready = all_ready != 0;
}

bool SPONGE_PLUGIN::Get_Mlip_Neighbor_Parameters(double& cutoff,
                                                 double& skin) const
{
    if (!g_local_mp_plugin_loaded || !g_mlip_capabilities.registered)
        return false;
    cutoff = g_mlip_capabilities.cutoff;
    skin = g_mlip_capabilities.skin;
    return true;
}

uint64_t SPONGE_PLUGIN::Get_Domain_Epoch() const
{
    return Local_MP_Service().Domain_Epoch();
}

void SPONGE_PLUGIN::Invalidate_Mlip_Neighbor_View()
{
    Local_MP_Service().Invalidate_Neighbor_View();
}

void SPONGE_PLUGIN::Publish_Mlip_Neighbor_View(const MLIP_NEIGHBOR_LIST& graph,
                                               int owned_count, int ghost_count)
{
    Local_MP_Service().Publish_Neighbor_View(graph, owned_count, ghost_count);
}

bool SPONGE_PLUGIN::Validate_Local_MP_Capabilities()
{
    if (PluginGetPPRank() < 0) return true;

    const int local_requires = g_local_mp_plugin_loaded ? 1 : 0;
    int min_requires = local_requires;
    int max_requires = local_requires;
#ifdef USE_MPI
    MPI_Allreduce(&local_requires, &min_requires, 1, MPI_INT, MPI_MIN,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&local_requires, &max_requires, 1, MPI_INT, MPI_MAX,
                  CONTROLLER::pp_comm);
#endif
    if (min_requires != max_requires) return false;
    if (max_requires == 0) return true;

    const auto& cap = g_mlip_capabilities;
    const int local_registered = cap.registered ? 1 : 0;
    int min_registered = local_registered;
    int max_registered = local_registered;
#ifdef USE_MPI
    MPI_Allreduce(&local_registered, &min_registered, 1, MPI_INT, MPI_MIN,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&local_registered, &max_registered, 1, MPI_INT, MPI_MAX,
                  CONTROLLER::pp_comm);
#endif
    if (min_registered != max_registered) return false;
    // A local-MP-capable PRIPS plugin may run an existing non-MLIP Python
    // script. Only a declared model requests a larger halo or tensor exchange.
    if (max_registered == 0) return true;

    const bool finite_values = SpongeFloat::Is_Finite(cap.cutoff) &&
                               SpongeFloat::Is_Finite(cap.skin) &&
                               SpongeFloat::Is_Finite(cap.cutoff + cap.skin);
    const int local_valid =
        finite_values && cap.cutoff > 0.0 && cap.skin >= 0.0 && cap.depth > 0 &&
                cap.dtype == SPONGE_MLIP_FEATURE_FP32 &&
                cap.cutoff + cap.skin <= std::numeric_limits<float>::max()
            ? 1
            : 0;
    int all_valid = local_valid;
#ifdef USE_MPI
    MPI_Allreduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                  CONTROLLER::pp_comm);
#endif
    if (!all_valid) return false;

    double local_reals[2] = {cap.cutoff, cap.skin};
    double min_reals[2] = {cap.cutoff, cap.skin};
    double max_reals[2] = {cap.cutoff, cap.skin};
    int local_depth = cap.depth;
    int min_depth = cap.depth;
    int max_depth = cap.depth;
    uint64_t min_hash = cap.hash;
    uint64_t max_hash = cap.hash;
#ifdef USE_MPI
    MPI_Allreduce(local_reals, min_reals, 2, MPI_DOUBLE, MPI_MIN,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(local_reals, max_reals, 2, MPI_DOUBLE, MPI_MAX,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&local_depth, &min_depth, 1, MPI_INT, MPI_MIN,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&local_depth, &max_depth, 1, MPI_INT, MPI_MAX,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&cap.hash, &min_hash, 1, MPI_UINT64_T, MPI_MIN,
                  CONTROLLER::pp_comm);
    MPI_Allreduce(&cap.hash, &max_hash, 1, MPI_UINT64_T, MPI_MAX,
                  CONTROLLER::pp_comm);
#endif
    if (min_reals[0] != max_reals[0] || min_reals[1] != max_reals[1] ||
        min_depth != max_depth || min_hash != max_hash)
        return false;

    g_plugin_controller->mlip_ghost_radius =
        MlipGeometry::Radius(cap.cutoff, cap.skin);
    return true;
}

void SPONGE_PLUGIN::After_Initial()
{
    for (int i = 0; i < after_init_func_numbers; i++)
    {
        const int owner = i < static_cast<int>(after_init_plugin_indices.size())
                              ? after_init_plugin_indices[i]
                              : -1;
        g_current_plugin_index = owner;
        after_init_funcs[i]();
        Check_Runtime_Error(owner);
        g_current_plugin_index = -1;
    }
    classic_neighbors_required = false;
    for (int i = 0; i < plugin_numbers; ++i)
    {
        using ClassicNeighborQuery = int (*)();
        auto query = reinterpret_cast<ClassicNeighborQuery>(
            dlsym(plugin_handles[i], "Uses_Classic_Neighbor_List"));
        bool required = true;
        if (query != nullptr)
        {
            try
            {
                required = query() != 0;
            }
            catch (...)
            {
                // Unknown consumers retain the classical skin constraint.
                required = true;
            }
        }
        Check_Runtime_Error(i);
        classic_neighbors_required |= required;
    }
}

bool SPONGE_PLUGIN::Has_Local_MP_Provider() const
{
    return g_mlip_capabilities.registered;
}

void SPONGE_PLUGIN::Calculate_Force()
{
    for (int i = 0; i < force_func_numbers; i++)
    {
        const int owner = i < static_cast<int>(force_plugin_indices.size())
                              ? force_plugin_indices[i]
                              : -1;
        g_current_plugin_index = owner;
        force_funcs[i]();
        Check_Runtime_Error(owner);
        g_current_plugin_index = -1;
    }
}

void SPONGE_PLUGIN::Mdout_Print()
{
    for (int i = 0; i < print_func_numbers; i++)
    {
        const int owner = i < static_cast<int>(print_plugin_indices.size())
                              ? print_plugin_indices[i]
                              : -1;
        g_current_plugin_index = owner;
        print_funcs[i]();
        Check_Runtime_Error(owner);
        g_current_plugin_index = -1;
    }
}
