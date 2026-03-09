#include "plugin.h"

namespace
{
MD_INFORMATION* g_plugin_md_info = NULL;
CONTROLLER* g_plugin_controller = NULL;
NEIGHBOR_LIST* g_plugin_neighbor_list = NULL;
DOMAIN_INFORMATION* g_plugin_domain_info = NULL;
SPONGE_PLUGIN_API g_prips_api = {};
}

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

int PluginGetMPIRank()
{
    return CONTROLLER::MPI_rank;
}

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
    return g_plugin_neighbor_list == NULL ? 0
                                          : g_plugin_neighbor_list->max_neighbor_numbers;
}

int PluginGetNeighborListCount(int atom_index)
{
    if (g_plugin_neighbor_list == NULL || g_plugin_neighbor_list->h_nl == NULL ||
        g_plugin_md_info == NULL || atom_index < 0 ||
        atom_index >= g_plugin_md_info->atom_numbers)
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
    return g_plugin_domain_info == NULL ? 0 : g_plugin_domain_info->atom_numbers;
}

int PluginGetLocalGhostNumbers()
{
    return g_plugin_domain_info == NULL ? 0 : g_plugin_domain_info->ghost_numbers;
}

int PluginGetLocalPPRank()
{
    return g_plugin_domain_info == NULL ? 0 : g_plugin_domain_info->pp_rank;
}

int PluginGetLocalMaxAtomNumbers()
{
    return g_plugin_domain_info == NULL ? 0
                                        : g_plugin_domain_info->max_atom_numbers;
}

void* PluginGetAtomLocalPtr()
{
    return g_plugin_domain_info == NULL ? NULL : g_plugin_domain_info->atom_local;
}

void* PluginGetAtomLocalLabelPtr()
{
    return g_plugin_domain_info == NULL ? NULL
                                        : g_plugin_domain_info->atom_local_label;
}

void* PluginGetAtomLocalIdPtr()
{
    return g_plugin_domain_info == NULL ? NULL : g_plugin_domain_info->atom_local_id;
}

void* PluginGetLocalCoordinatePtr()
{
    return g_plugin_domain_info == NULL ? NULL : g_plugin_domain_info->crd;
}

void* PluginGetLocalForcePtr()
{
    return g_plugin_domain_info == NULL ? NULL : g_plugin_domain_info->frc;
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
}  // namespace

void SPONGE_PLUGIN::Initial(MD_INFORMATION* md_info, CONTROLLER* controller,
                            COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                            NEIGHBOR_LIST* neighbor_list)
{
    if (!controller->Command_Exist("plugin"))
    {
        return;
    }

    controller->printf("START INITIALIZING SPONGE PLUGIN:\n");
    plugin_numbers = 0;
    g_plugin_md_info = md_info;
    g_plugin_controller = controller;
    g_plugin_neighbor_list = neighbor_list;
    g_plugin_domain_info = NULL;

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

        stable_initial_func =
            (InitialStableFunction)dlsym(plugin_handles[count], "Initial_Stable");
        version_check_error = version_check_func(stable_initial_func != NULL
                                                     ? SPONGE_PRIPS_API_VERSION
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
        if (func == NULL && stable_initial_func == NULL)
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

        set_backend_device_type_func = (SetBackendDeviceTypeFunction)dlsym(
            plugin_handles[count], "Set_Backend_Device_Type");
        if (set_backend_device_type_func != NULL)
        {
            funcs_loaded += 1;
            controller->printf(" Set_Backend_Device_Type");
            set_backend_device_type_func(PluginBackendDeviceType());
        }

        after_init_funcs[after_init_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "After_Initial");
        if (after_init_funcs[after_init_func_numbers] != NULL)
        {
            funcs_loaded += 1;
            after_init_func_numbers += 1;
            controller->printf(" After_Initial");
        }

        force_funcs[force_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "Calculate_Force");
        if (force_funcs[force_func_numbers] != NULL)
        {
            funcs_loaded += 1;
            force_func_numbers += 1;
            controller->printf(" Calculate_Force");
        }

        print_funcs[print_func_numbers] =
            (RuntimeFunction)dlsym(plugin_handles[count], "Mdout_Print");
        if (print_funcs[print_func_numbers] != NULL)
        {
            funcs_loaded += 1;
            print_func_numbers += 1;
            controller->printf(" Mdout_Print");
        }

        set_domain_info_funcs[set_domain_info_func_numbers] =
            (SetDomainInformationFunction)dlsym(plugin_handles[count],
                                                "Set_Domain_Information");
        if (set_domain_info_funcs[set_domain_info_func_numbers] != NULL)
        {
            funcs_loaded += 1;
            set_domain_info_func_numbers += 1;
            controller->printf(" Set_Domain_Information");
        }

        controller->printf(" (%d in total)\n", funcs_loaded);
        if (stable_initial_func != NULL)
        {
            stable_initial_func(BuildPripsApi());
        }
        else
        {
            func(md_info, controller, neighbor_list, cv_controller, CV_MAP,
                 CV_INSTANCE_MAP);
        }

        count += 1;
        last_pos = command.find_first_not_of(" ", pos);
        pos = command.find_first_of(" ", last_pos);
    }

    controller->printf("END INITIALIZING SPONGE PLUGIN\n\n");
}

void SPONGE_PLUGIN::Set_Domain_Information(DOMAIN_INFORMATION* dd)
{
    g_plugin_domain_info = dd;
    for (int i = 0; i < set_domain_info_func_numbers; i++)
    {
        set_domain_info_funcs[i](dd);
    }
}

void SPONGE_PLUGIN::After_Initial()
{
    for (int i = 0; i < after_init_func_numbers; i++)
    {
        after_init_funcs[i]();
    }
}

void SPONGE_PLUGIN::Calculate_Force()
{
    for (int i = 0; i < force_func_numbers; i++)
    {
        force_funcs[i]();
    }
}

void SPONGE_PLUGIN::Mdout_Print()
{
    for (int i = 0; i < print_func_numbers; i++)
    {
        print_funcs[i]();
    }
}
