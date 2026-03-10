#include "control.h"

#define SPONGE_VERSION "v2.0.0 2026-02-16 Spring Festive"

static const char* SPONGE_ASCII_ART = R"( ____  ____   ___  _   _  ____ _____
/ ___||  _ \ / _ \| \ | |/ ___| ____|
\___ \| |_) | | | |  \| | |  _|  _|
 ___) |  __/| |_| | |\  | |_| | |___
|____/|_|    \___/|_| \_|\____|_____|
)";

#define MDIN_DEFAULT_FILENAME "mdin.txt"
#define MDIN_TOML_DEFAULT_FILENAME "mdin.spg.toml"
#define MDOUT_DEFAULT_FILENAME "mdout.txt"
#define MDINFO_DEFAULT_FILENAME "mdinfo.txt"

#define MDIN_COMMAND "mdin"
#define MDOUT_COMMAND "mdout"
#define MDINFO_COMMAND "mdinfo"

int CONTROLLER::MPI_rank = 0;
int CONTROLLER::PP_MPI_rank = -1;
int CONTROLLER::PM_MPI_rank = -1;
int CONTROLLER::CC_MPI_rank = -1;
int CONTROLLER::MPI_size = 1;
int CONTROLLER::PP_MPI_size = 1;
int CONTROLLER::PM_MPI_size = 1;
int CONTROLLER::CC_MPI_size = 0;
unsigned int CONTROLLER::device_optimized_block = 64;
unsigned int CONTROLLER::device_warp = 32;
unsigned int CONTROLLER::device_max_thread = 1024;
MPI_Comm CONTROLLER::pp_comm;
MPI_Comm CONTROLLER::pm_comm;

D_MPI_Comm CONTROLLER::D_MPI_COMM_WORLD;
D_MPI_Comm CONTROLLER::d_pp_comm;
D_MPI_Comm CONTROLLER::d_pm_comm;

bool CONTROLLER::Command_Exist(const char* key)
{
    const char* temp = strstr(key, "in_file");
    command_check[key] = 0;
    if (temp != NULL && strcmp(temp, "in_file") == 0)
    {
        if (commands.count(key))
        {
            return true;
        }
        else if (Command_Exist("default_in_file_prefix"))
        {
            std::string buffer, buffer2;
            buffer = key;
            buffer = buffer.substr(0, strlen(key) - strlen(temp) - 1);
            buffer2 =
                Command("default_in_file_prefix") + ("_" + buffer + ".txt");
            FILE* ftemp = fopen(buffer2.c_str(), "r");
            if (ftemp != NULL)
            {
                commands[key] = buffer2;
                fclose(ftemp);
                return true;
            }
            return false;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return (bool)commands.count(key);
    }
}

bool CONTROLLER::Command_Exist(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Exist(temp);
}

bool CONTROLLER::Command_Choice(const char* key, const char* value,
                                bool case_sensitive)
{
    if (commands.count(key))
    {
        if (is_str_equal(commands[key].c_str(), value, case_sensitive))
        {
            command_check[key] = 0;
            choice_check[key] = 1;
            return true;
        }
        else
        {
            command_check[key] = 0;
            if (choice_check[key] != 1) choice_check[key] = 2;
            return false;
        }
    }
    else
    {
        choice_check[key] = 3;
        return false;
    }
}

bool CONTROLLER::Command_Choice(const char* prefix, const char* key,
                                const char* value, bool case_sensitive)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Choice(temp, value, case_sensitive);
}

const char* CONTROLLER::Command(const char* key)
{
    command_check[key] = 0;
    return commands[key].c_str();
}

const char* CONTROLLER::Command(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return commands[temp].c_str();
}

const char* CONTROLLER::Original_Command(const char* key)
{
    command_check[key] = 0;
    return original_commands[key].c_str();
}

const char* CONTROLLER::Original_Command(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return original_commands[temp].c_str();
}

static int judge_if_flag(const char* str)
{
    if (strlen(str) <= 1) return 0;
    if (str[0] != '-') return 0;
    if (str[1] >= '0' && str[1] <= '9') return 0;
    return 1;
}

void CONTROLLER::Arguments_Parse(int argc, char** argv)
{
    char temp1[CHAR_LENGTH_MAX];
    char temp2[CHAR_LENGTH_MAX];
    char temp3[CHAR_LENGTH_MAX];
    int j = 1;
    for (int i = 1; i < argc; i++)
    {
        temp1[0] = 0;
        strcat(temp1, argv[i]);
        if (judge_if_flag(temp1))
        {
            temp2[0] = ' ';
            temp2[1] = 0;
            j = 1;
            while (i + j < argc)
            {
                strcpy(temp3, argv[i + j]);
                if (!judge_if_flag(temp3))
                {
                    strcat(temp2, " ");
                    strcat(temp2, temp3);
                    j++;
                }
                else
                    break;
            }
            Set_Command(temp1 + 1, temp2);
            if (is_str_equal(temp1 + 1, "workspace", 1))
            {
                workspace_from_cli = true;
            }
        }
    }
}

void CONTROLLER::Get_Command(char* line, char* prefix)
{
    if ((prefix[0] == '#' && prefix[1] == '#') || prefix[0] == ' ' ||
        prefix[0] == '\t')
    {
        return;
    }
    char Flag[CHAR_LENGTH_MAX];
    char Value[CHAR_LENGTH_MAX];
    char* flag = strtok(line, "=");
    char* command = strtok(NULL, "=");

    if (flag == NULL || command == NULL)
    {
        return;
    }

    sscanf(flag, "%s", Flag);

    // Trim leading/trailing spaces in command/Value
    char* v_ptr = command;
    while (*v_ptr == ' ' || *v_ptr == '\t') v_ptr++;
    char* v_end = v_ptr + strlen(v_ptr) - 1;
    while (v_end > v_ptr && (*v_end == ' ' || *v_end == '\t' ||
                             *v_end == '\r' || *v_end == '\n'))
        *v_end-- = '\0';

    strcpy(Value, v_ptr);
    Set_Command(Flag, Value, 1, prefix);
}

static int read_one_line(FILE* In_File, char* line, char* ender)
{
    int line_count = 0;
    int ender_count = 0;
    signed char c;
    while ((c = getc(In_File)) != EOF)
    {
        if (line_count == 0 && (c == '\t' || c == ' '))
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            line[line_count] = c;
            line_count += 1;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
            break;
        }
    }
    while ((c = getc(In_File)) != EOF)
    {
        if (c == ' ' || c == '\t')
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            fseek(In_File, -1, SEEK_CUR);
            break;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
        }
    }
    line[line_count] = 0;
    ender[ender_count] = 0;
    if (line_count == 0 && ender_count == 0)
    {
        return EOF;
    }
    return 1;
}

static fs::path Resolve_Path_With_Base(const std::string& raw_path,
                                       const fs::path& base_dir)
{
    fs::path path(raw_path);
    if (path.empty()) return base_dir;
    if (path.is_absolute()) return path.lexically_normal();
    return fs::absolute(base_dir / path).lexically_normal();
}

void CONTROLLER::Commands_From_In_File(int argc, char** argv,
                                       const char* subpackage_hint)
{
    const fs::path startup_cwd = fs::current_path();
    fs::path mdin_dir = startup_cwd;
    fs::path resolved_mdin_path;
    bool mdin_found = false;
    const bool command_only = Command_Exist("command_only");
    MdinInputFormat mdin_format = MdinInputFormat::None;
    std::string mdin_path;
    std::string toml_content;
    FILE* In_File = NULL;
    if (command_only)
    {
        // Skip mdin discovery when only command-line inputs are requested.
        // All required parameters must be provided via CLI flags.
    }
    else if (!Command_Exist(MDIN_COMMAND))
    {
        if (fs::exists(MDIN_TOML_DEFAULT_FILENAME))
        {
            mdin_format = MdinInputFormat::Toml;
            mdin_path = MDIN_TOML_DEFAULT_FILENAME;
        }
        else
        {
            In_File = fopen(MDIN_DEFAULT_FILENAME, "r");
            if (In_File == NULL)
            {
                commands["md_name"] = "Default SPONGE MD Task Name";
            }
            else
            {
                mdin_format = MdinInputFormat::Text;
                mdin_path = MDIN_DEFAULT_FILENAME;
            }
        }
    }
    else
    {
        mdin_path = Command(MDIN_COMMAND);
        std::string ext =
            to_lower_copy(fs::path(mdin_path).extension().string());
        if (ext == ".toml")
        {
            mdin_format = MdinInputFormat::Toml;
        }
        else
        {
            mdin_format = MdinInputFormat::Text;
        }
    }

    if (mdin_format == MdinInputFormat::Toml)
    {
        resolved_mdin_path = Resolve_Path_With_Base(mdin_path, startup_cwd);
        toml_content = Read_File_To_String(resolved_mdin_path.string(), this);
        mdin_dir = resolved_mdin_path.parent_path();
        mdin_found = true;
        mdin_path = resolved_mdin_path.string();
        Load_Toml_Commands(toml_content, mdin_path, this,
                           "CONTROLLER::Commands_From_In_File");
    }
    else if (mdin_format == MdinInputFormat::Text)
    {
        if (!mdin_path.empty())
        {
            resolved_mdin_path = Resolve_Path_With_Base(mdin_path, startup_cwd);
        }
        if (In_File == NULL && !mdin_path.empty())
        {
            Open_File_Safely(&In_File, resolved_mdin_path.string().c_str(), "r",
                             true);
        }
        if (In_File != NULL)
        {
            mdin_dir = resolved_mdin_path.parent_path();
            mdin_found = true;
            mdin_path = resolved_mdin_path.string();
            char line[CHAR_LENGTH_MAX];
            char prefix[CHAR_LENGTH_MAX] = {0};
            char ender[CHAR_LENGTH_MAX];
            char* get_ret = fgets(line, CHAR_LENGTH_MAX, In_File);
            line[strlen(line) - 1] = 0;
            commands["md_name"] = line;
            int while_count = 0;
            while (true)
            {
                while_count += 1;
                if (while_count > 100000)
                {
                    Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "CONTROLLER::Commands_From_In_File",
                        "Possible reasons : \n\t1.The coding of the format is "
                        "not ASCII\n\t2.The file is created in one OS but used "
                        "in another OS(Windows / Unix / MacOS)");
                }
                int t = read_one_line(In_File, line, ender);
                if (t == EOF)
                {
                    break;
                }
                if (line[0] == '#')
                {
                    if (line[1] == '#')
                    {
                        if (strchr(ender, '{') != NULL)
                        {
                            int scanf_ret = sscanf(line, "%s", prefix);
                        }
                        if (strchr(ender, '}') != NULL)
                        {
                            prefix[0] = 0;
                        }
                    }
                    if (strchr(ender, '\n') == NULL)
                    {
                        int scanf_ret = fscanf(In_File, "%*[^\n]%*[\n]");
                        fseek(In_File, -1, SEEK_CUR);
                    }
                }
                else if (strchr(ender, '{') != NULL)
                {
                    int scanf_ret = sscanf(line, "%s", prefix);
                }
                else
                {
                    Get_Command(line, prefix);
                    line[0] = 0;
                }
                if (strchr(ender, '}') != NULL)
                {
                    prefix[0] = 0;
                }
            }
        }
    }

    if (!commands.count("md_name"))
    {
        commands["md_name"] = "Default SPONGE MD Task Name";
    }

    // Resolve workspace after mdin has been parsed so mdin-provided workspace
    // is honored. Workspace is resolved relative to mdin directory if present,
    // otherwise relative to startup cwd.
    fs::path workspace_dir;
    bool workspace_set = false;
    if (Command_Exist("workspace"))
    {
        const fs::path workspace_base =
            (workspace_from_cli || !mdin_found) ? startup_cwd : mdin_dir;
        workspace_dir =
            Resolve_Path_With_Base(Command("workspace"), workspace_base);
        workspace_set = true;
    }

    fs::path target_workdir = startup_cwd;
    if (workspace_set)
    {
        target_workdir = workspace_dir;
    }
    else if (mdin_found)
    {
        target_workdir = mdin_dir;
    }

    if (target_workdir != fs::current_path())
    {
        try
        {
            fs::current_path(target_workdir);
        }
        catch (const fs::filesystem_error& e)
        {
            std::string error_reason = string_format(
                "Reason:\n\tfail to change working directory to '%PATH%': "
                "%DESC%",
                {{"PATH", target_workdir.string()},
                 {"DESC", std::string(e.what())}});
            Throw_SPONGE_Error(spongeErrorOpenFileFailed,
                               "CONTROLLER::Commands_From_In_File",
                               error_reason.c_str());
        }
    }

    mdinfo = this->Get_Output_File(false, MDINFO_COMMAND, ".info",
                                   MDINFO_DEFAULT_FILENAME);
    setvbuf(mdinfo, NULL, _IONBF, 0);
    mdout = this->Get_Output_File(false, MDOUT_COMMAND, ".out",
                                  MDOUT_DEFAULT_FILENAME);
    printf("%s\n", SPONGE_ASCII_ART);
    printf("SPONGE Version:\n    %s\n\n", SPONGE_VERSION);
    printf("Sub-package:\n    %s\n\n", subpackage_hint);
    printf(
        "Citation:\n    %s\n",
        "Huang, Y. - P., Xia, Y., Yang, L., Wei, J., Yang, Y.I.and Gao, Y.Q. "
        "(2022), SPONGE: A GPU - Accelerated Molecular Dynamics Package with "
        "Enhanced Sampling and AI - Driven Algorithms.Chin.J.Chem., 40 : 160 - "
        "168. https ://doi.org/10.1002/cjoc.202100456\n\n");
    printf("Working Directory:\n    %s\n\n",
           Get_Current_Working_Directory().c_str());
    printf("SPONGE Path:\n    %s\n\n", Get_SPONGE_Directory().c_str());
    printf("Start Wall Time:\n    %s\n", Get_Wall_Time().c_str());
    printf("MD TASK NAME:\n    %s\n\n", commands["md_name"].c_str());
    int scanf_ret = fprintf(mdinfo, "Terminal Commands:\n    ");
    for (int i = 0; i < argc; i++)
    {
        scanf_ret = fprintf(mdinfo, "%s ", argv[i]);
    }
    scanf_ret = fprintf(mdinfo, "\n\n");
    if (command_only)
    {
        scanf_ret = fprintf(
            mdinfo,
            "Mdin File:\n    command_only mode: no mdin file is loaded.\n\n");
    }
    else if (mdin_format == MdinInputFormat::Toml)
    {
        scanf_ret = fprintf(mdinfo, "Mdin File:\n");
        std::istringstream content_stream(toml_content);
        std::string content_line;
        while (std::getline(content_stream, content_line))
        {
            scanf_ret = fprintf(mdinfo, "    %s\n", content_line.c_str());
        }
        scanf_ret = fprintf(mdinfo, "\n\n");
    }
    else if (In_File != NULL)
    {
        scanf_ret = fprintf(mdinfo, "Mdin File:\n");
        fseek(In_File, 0, SEEK_SET);
        char temp[CHAR_LENGTH_MAX];
        while (fgets(temp, CHAR_LENGTH_MAX, In_File) != NULL)
        {
            scanf_ret = fprintf(mdinfo, "    %s", temp);
        }
        scanf_ret = fprintf(mdinfo, "\n\n");
        fclose(In_File);
    }
}

void CONTROLLER::Set_Command(const char* Flag, const char* Value, int Check,
                             const char* prefix)
{
    if (prefix && strcmp(prefix, "comments") == 0) return;
    char temp[CHAR_LENGTH_MAX] = {0}, temp2[CHAR_LENGTH_MAX];
    if (prefix && prefix[0] != 0 && strcmp(prefix, "main") != 0)
    {
        strcpy(temp, prefix);
        strcat(temp, "_");
    }
    strcat(temp, Flag);
    if (commands.count(temp))
    {
        sprintf(temp2, "Reason:\n\t'%s' is set more than once\n", temp);
        Throw_SPONGE_Error(spongeErrorConflictingCommand,
                           "CONTROLLER::Set_Command", temp2);
    }
    strcpy(temp2, Value);
    char* real_value = strtok(temp2, "#");
    original_commands[temp] = real_value;
    if (sscanf(real_value, "%s", temp2))
        commands[temp] = temp2;
    else
        commands[temp] = "";

    command_check[temp] = Check;
}

void CONTROLLER::Default_Set()
{
    srand(0);
    buffer_frame = 10;
    if (Command_Exist("buffer_frame"))
    {
        Check_Int("buffer_frame", "CONTROLLER::Default_Set");
        buffer_frame = atoi(Command("buffer_frame"));
    }
}

// 初始化设备
void CONTROLLER::Init_Device()
{
#ifdef USE_GPU
    printf("    Start initializing GPU\n");

#ifdef USE_CUDA
#ifdef CUDA_VERSION
    int cuda_major_version = CUDA_VERSION / 1000;
    int cuda_minor_version = (CUDA_VERSION - 1000 * cuda_major_version) / 10;
    printf("        Compiled by CUDA %d.%d\n", cuda_major_version,
           cuda_minor_version);
#else
    printf("        Compiled by unknown CUDA version\n");
#endif  // CUDA_VERSION
#elif defined(USE_HIP)
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR)
#ifdef HIP_VERSION_PATCH
    printf("        Compiled by HIP %d.%d.%d\n", HIP_VERSION_MAJOR,
           HIP_VERSION_MINOR, HIP_VERSION_PATCH);
#else
    printf("        Compiled by HIP %d.%d\n", HIP_VERSION_MAJOR,
           HIP_VERSION_MINOR);
#endif
#else
    printf("        Compiled by unknown HIP/ROCm version\n");
#endif
#endif

    if (deviceInit(0) != DEVICE_INIT_SUCCESS)
    {
        std::string error_reason =
            string_format("Reason:\n\tFail to initialize %backend% runtime",
                          {{"backend", GPU_ARCH_NAME}});
        Throw_SPONGE_Error(spongeErrorMallocFailed, "CONTROLLER::Init_Device",
                           error_reason.c_str());
    }

    if (Command_Exist("device"))
    {
        std::string device_command = Original_Command("device");
        size_t comment_pos = device_command.find('#');
        if (comment_pos != std::string::npos)
        {
            device_command = device_command.substr(0, comment_pos);
        }
        std::vector<std::string> words =
            string_split(string_strip(device_command), " ");
        if (words.size() != MPI_size && words.size() != 1)
        {
            char error_reason[CHAR_LENGTH_MAX];
            int ret = sprintf(
                error_reason,
                "Reason:\n\tthe number of words (%zu) in the command 'device' is \
neither equal to the size of MPI ranks (%d) nor 1\n",
                words.size(), MPI_size);
            Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                               "CONTROLLER::Init_Device", error_reason);
        }
        if (words.size() == 1)
            working_device = atoi(words[0].c_str());
        else
            working_device = atoi(words[MPI_rank].c_str());
    }
    else
    {
        working_device = MPI_rank;
    }
    if (Command_Exist("device_optimized_block"))
    {
        Check_Int("device_optimized_block", "CONTROLLER::Init_Device");
        device_optimized_block = atoi(Command("device_optimized_block"));
    }
    int count;
    deviceGetDeviceCount(&count);
    printf("        %d device(s) found:\n", count);

    deviceProp prop;
    float GlobalMem;
    for (int i = 0; i < count; i++)
    {
        getDeviceProperties(&prop, i);
        GlobalMem = (float)prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f;
#ifdef USE_HIP
        std::string runtime_arch = Get_Device_Runtime_Arch_Name(prop);
        printf(
            "            Device %d:\n                Name: %s\n                "
            "Memory: %.1f GB\n                Architecture: %s\n",
            i, prop.name, GlobalMem, runtime_arch.c_str());
#else
        printf(
            "            Device %d:\n                Name: %s\n                "
            "Memory: %.1f GB\n                Compute Capability: %d%d\n",
            i, prop.name, GlobalMem, prop.major, prop.minor);
#endif
        if (i == working_device)
        {
            device_max_thread = prop.maxThreadsPerBlock;
            device_warp = prop.warpSize;
        }
    }
    if (count <= working_device)
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason,
                "Reason:\n\tthe available device count %d is less than the "
                "setting working_device %d.",
                count, working_device);
        this->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                 "CONTROLLER::Init_Device", error_reason);
    }
    MPI_printf("        Rank %d: Device %d\n", MPI_rank, working_device);
    setWorkingDevice(working_device);

// 暂时只检查CUDA架构
#ifdef USE_CUDA
    int built_arch_bin = Get_Built_Arch();
    getDeviceProperties(&prop, working_device);
    int runtime_arch_bin = prop.major * 10 + prop.minor;
    if (runtime_arch_bin < built_arch_bin)
    {
        char error_reason[CHAR_LENGTH_MAX];
        int ret =
            sprintf(error_reason,
                    "Reason:\n\tthe compute compacity (%d) of the working GPU "
                    "device (%s) is less than the minimum required compute "
                    "compacity (%d) for the compiled SPONGE version\n",
                    runtime_arch_bin, prop.name, built_arch_bin);
        Throw_SPONGE_Error(spongeErrorMallocFailed, "CONTROLLER::Init_Device",
                           error_reason);
    }
    printf("        Compiled by CUDA ARCH %d.%d\n", built_arch_bin / 10,
           built_arch_bin % 10);
    printf("        Runtime CUDA ARCH %d.%d\n", runtime_arch_bin / 10,
           runtime_arch_bin % 10);
#endif  // USE_CUDA
#ifdef USE_HIP
    getDeviceProperties(&prop, working_device);
    printf("        Runtime HIP ARCH %s\n",
           Get_Device_Runtime_Arch_Name(prop).c_str());
    printf("        Runtime backend HIP/ROCm\n");
#endif
    printf("    End initializing GPU\n");
#else
    printf("    Start initializing OpenMP\n");
    if (Command_Exist("device"))
    {
        std::string device_command = Original_Command("device");
        size_t comment_pos = device_command.find('#');
        if (comment_pos != std::string::npos)
        {
            device_command = device_command.substr(0, comment_pos);
        }
        std::vector<std::string> words =
            string_split(string_strip(device_command), " ");
        if (words.size() != MPI_size && words.size() != 1)
        {
            char error_reason[CHAR_LENGTH_MAX];
            int ret = sprintf(
                error_reason,
                "Reason:\n\tthe number of words (%ld) in the command 'device' is \
neither equal to the size of MPI ranks (%d) nor 1\n",
                words.size(), MPI_size);
            Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                               "CONTROLLER::Init_Device", error_reason);
        }
        if (words.size() == 1)
            working_device = atoi(words[0].c_str());
        else
            working_device = atoi(words[MPI_rank].c_str());
    }
    else
    {
        working_device = omp_get_max_threads();
    }
    MPI_printf("        Rank %d: %d OpenMP Thread(s)\n", MPI_rank,
               working_device);
    omp_set_num_threads(working_device);
    printf("    End initializing OpenMP\n");
#endif  // USE_GPU
}

void CONTROLLER::Init_Host_MPI()
{
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
#endif
}

void CONTROLLER::Init_Device_MPI()
{
#ifdef USE_MPI
    printf("    Start initializing Device MPI\n");
    printf("        Total %d MPI process(es)\n", MPI_size);

#ifdef USE_XCCL
    xcclUniqueId id;
    if (MPI_rank == 0) xcclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, SPONGE_MPI_ROOT, MPI_COMM_WORLD);
    xcclCommInitRank(&D_MPI_COMM_WORLD, MPI_size, id, MPI_rank);
#else
    D_MPI_COMM_WORLD = MPI_COMM_WORLD;
#endif

    printf("    End initializing Device MPI\n");
#else
    printf("    MPI is not enabled\n");
#endif
}

void CONTROLLER::Input_Check()
{
    if (!(Command_Exist("dont_check_input") &&
          Get_Bool("dont_check_input", "CONTROLLER::Input_Check")))
    {
        int no_warning = 0;
        for (CheckMap::iterator iter = command_check.begin();
             iter != command_check.end(); iter++)
        {
            if (iter->second == 1)
            {
                printf("Warning: '%s' is set, but never used.\n",
                       iter->first.c_str());
                no_warning += 1;
            }
        }
        for (CheckMap::iterator iter = choice_check.begin();
             iter != choice_check.end(); iter++)
        {
            if (iter->second == 2)
            {
                char error_reason[CHAR_LENGTH_MAX];
                sprintf(error_reason,
                        "Reason:\n\tthe value '%s' of command '%s' matches "
                        "none of the choices.\n",
                        this->commands[iter->first].c_str(),
                        iter->first.c_str());
                this->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                         "CONTROLLER::Input_Check",
                                         error_reason);
            }
            else if (iter->second == 3)
            {
                printf("Warning: command '%s' is not set.\n",
                       iter->first.c_str());
                no_warning += 1;
            }
        }
        for (int i = 0; i < warnings.size(); i++)
        {
            printf("Warning: %s\n", warnings[i].c_str());
            no_warning += 1;
        }
        if (no_warning)
        {
            printf(
                "\nWarning: inputs raised %d warning(s). If You know WHAT YOU "
                "ARE DOING, press any key to continue. Set dont_check_input = "
                "1 to disable this warning.\n",
                no_warning);
            getchar();
        }
    }
}

void CONTROLLER::Initial(int argc, char** argv, const char* subpackage_hint)
{
    Init_Host_MPI();
    if (argc == 2 && is_str_equal(argv[1], "-v", 1))
    {
        std::cout << SPONGE_ASCII_ART << std::endl;
        std::cout << SPONGE_VERSION << std::endl
                  << subpackage_hint << std::endl;
        exit(0);
    }
    warn_of_initialization = 1;
    Arguments_Parse(argc, argv);
    Commands_From_In_File(argc, argv, subpackage_hint);
    printf("START INITIALIZING CONTROLLER\n");
    Default_Set();
    Init_Device();
    Init_Device_MPI();
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        printf("    structure last modify date is %d\n", last_modify_date);
    }
    Command_Exist("end_pause");
    Get_Time_Recorder("Initialization")->Start();
    Get_Time_Recorder("Neighbor Searching");
    Get_Time_Recorder("Force Calculation");
    Get_Time_Recorder("Iteration");
    Get_Time_Recorder("Printing & Dumping");
    Get_Time_Recorder("Communication");
    printf("END INITIALIZING CONTROLLER\n\n");
}

void CONTROLLER::Clear()
{
    if (is_initialized)
    {
        fcloseall();
        if (Command_Exist("end_pause") && atoi(Command("end_pause")) != 0)
        {
            printf("End Pause\n");
            getchar();
        }
#ifdef USE_MPI
#ifdef USE_XCCL
        if (CONTROLLER::MPI_size > 1)
        {
            xcclCommDestroy(CONTROLLER::D_MPI_COMM_WORLD);
            xcclCommDestroy(CONTROLLER::d_pp_comm);
            if (CONTROLLER::PM_MPI_size > 0)
            {
                xcclCommDestroy(CONTROLLER::d_pm_comm);
            }
        }
#endif
        MPI_Finalize();
#endif
    }
}

TIME_RECORDER* CONTROLLER::Get_Time_Recorder(const char* name)
{
    if (time_recorders.count(name)) return &time_recorders[name];
    time_recorder_names.push_back(name);
    TIME_RECORDER t;
    time_recorders[name] = t;
    return &time_recorders[name];
}

static std::string _get_time_unit(float time, float* factor)
{
    std::string print_time_unit;
    if (time > 86400)
    {
        factor[0] = 1.0f / 86400.0f;
        print_time_unit = "day";
    }
    else if (time > 3600)
    {
        factor[0] = 1.0f / 3600.0f;
        print_time_unit = "hour";
    }
    else if (time > 60)
    {
        factor[0] = 1.0f / 60.0f;
        print_time_unit = "minute";
    }
    else
    {
        factor[0] = 1.0f;
        print_time_unit = "second";
    }
    return print_time_unit;
}

void CONTROLLER::Final_Time_Summary(int steps, float time_factor,
                                    const char* unit_name, const int mode)
{
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    core_time.Stop();
    float print_time_unit_factor = 1.0f;
    std::string print_time_unit = "second";
    int left_padding = 0, right_padding = 0;

    printf("Time Summary:\n");
    TIME_RECORDER* recorder;
    for (int i = 0; i < this->time_recorder_names.size(); i++)
    {
        std::string name = this->time_recorder_names[i];
        // printf("%s\n", name.c_str());
        recorder = &this->time_recorders[name];
        if (recorder->time / steps <
            1e-6f)  // 不打印平均每步小于1微秒计算的计时器
        {
            MPI_printf("");
            continue;
        }
        print_time_unit =
            _get_time_unit(recorder->time, &print_time_unit_factor);
        left_padding = (30 - name.size()) / 2;
        right_padding = 30 - left_padding - name.size();
        MPI_printf("Rank % 3d    | %*s%s%*s |    %10.6f %ss\n",
                   CONTROLLER::MPI_rank, left_padding, "", name.c_str(),
                   right_padding, "", recorder->time * print_time_unit_factor,
                   print_time_unit.c_str());
    }
    printf(
        "----------------------------------------------------------------------"
        "--------------------------------------\n");
    printf("Stop Wall Time: %s", Get_Wall_Time().c_str());

    print_time_unit = _get_time_unit(core_time.time, &print_time_unit_factor);
    if (print_time_unit != "second")
        print_time_unit =
            " (" + std::to_string(core_time.time * print_time_unit_factor) +
            " " + print_time_unit + "s)";
    else
        print_time_unit = "";

    printf("Core Run Wall Time: %f seconds%s\n", core_time.time,
           print_time_unit.c_str());

    if (mode >= 0)
    {
        simulation_speed = steps * time_factor / core_time.time;
        printf("Core Run Speed: %f %s\n", simulation_speed, unit_name);
    }
    else if (mode == -1)
    {
        simulation_speed = steps / core_time.time;
        printf("Core Run Speed: %f step/second\n", simulation_speed);
    }
    else if (mode == -2)
    {
        simulation_speed = time_factor / core_time.time;
        printf("Core Run Speed: %f frame/second\n", simulation_speed);
    }
}
