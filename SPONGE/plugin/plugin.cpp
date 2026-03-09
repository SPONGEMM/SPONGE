#include "plugin.h"

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

        version_check_error = version_check_func(controller->last_modify_date);
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
        if (func == NULL)
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
        func(md_info, controller, neighbor_list, cv_controller, CV_MAP,
             CV_INSTANCE_MAP);

        count += 1;
        last_pos = command.find_first_not_of(" ", pos);
        pos = command.find_first_of(" ", last_pos);
    }

    controller->printf("END INITIALIZING SPONGE PLUGIN\n\n");
}

void SPONGE_PLUGIN::Set_Domain_Information(DOMAIN_INFORMATION* dd)
{
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
