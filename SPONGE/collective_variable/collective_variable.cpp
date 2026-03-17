#include "collective_variable.h"

CV_MAP_TYPE* CV_MAP = new CV_MAP_TYPE;
CV_INSTANCE_TYPE* CV_INSTANCE_MAP = new CV_INSTANCE_TYPE;

void COLLECTIVE_VARIABLE_CONTROLLER::Initial(
    CONTROLLER* controller, int* no_direct_interaction_virtual_atom_numbers)
{
    controller->printf("START INITIALIZING CV CONTROLLER:\n");
    strcpy(module_name, "cv_controller");
    this->controller = controller;
    mdinfo = controller->mdinfo;
    if (controller->Command_Exist("cv_in_file"))
    {
        int CV_numbers = 0;
        Commands_From_In_File(controller);
        int cv_vatom_count = 0;
        for (StringMap::iterator iter = this->commands.begin();
             iter != this->commands.end(); iter++)
        {
            int i = iter->first.rfind("vatom_type");
            if (i > 0 && i == (iter->first.length() - 10))
            {
                cv_vatom_name[iter->first.substr(0, i - 1)] = cv_vatom_count;
                cv_vatom_index[cv_vatom_count] = iter->first.substr(0, i - 1);
                cv_vatom_count += 1;
                no_direct_interaction_virtual_atom_numbers[0]++;
            }
            i = iter->first.rfind("CV_type");
            if (i > 0 && i == (iter->first.length() - 7))
            {
                CV_numbers++;
            }
        }
        printf("    %d CV defined\n", CV_numbers);
        printf("    %d cv virtual atoms\n",
               no_direct_interaction_virtual_atom_numbers[0]);
        is_initialized = 1;
        controller->printf("END INITIALIZING CV CONTROLLER\n\n");
    }
    else
    {
        controller->printf("CV CONTROLLER IS NOT INITIALIZING\n\n");
    }
}

static int read_one_line(FILE* In_File, char* line, char* ender)
{
    int line_count = 0;
    int ender_count = 0;
    int c;
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

void COLLECTIVE_VARIABLE_CONTROLLER::Commands_From_In_File(
    CONTROLLER* controller)
{
    FILE* In_File = NULL;
    if (controller->Command_Exist("cv_in_file"))
    {
        std::string cv_path = controller->Command("cv_in_file");
        std::string ext = to_lower_copy(fs::path(cv_path).extension().string());
        if (ext == ".toml")
        {
            std::string toml_content = Read_File_To_String(cv_path, controller);
            Load_Toml_Commands(
                toml_content, cv_path, this,
                "COLLECTIVE_VARIABLE_CONTROLLER::Commands_From_In_File");
            return;
        }
        Open_File_Safely(&In_File, cv_path.c_str(), "r", true);
    }
    if (In_File != NULL)
    {
        char line[CHAR_LENGTH_MAX];
        char prefix[CHAR_LENGTH_MAX] = {0};
        char ender[CHAR_LENGTH_MAX];
        while (true)
        {
            if (read_one_line(In_File, line, ender) == EOF)
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

void COLLECTIVE_VARIABLE_CONTROLLER::Input_Check()
{
    if (!is_initialized) return;
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        controller->Step_Print_Initial(print_cv_list[i]->module_name, "%.4f");
    }
    if (!(Command_Exist("dont_check_input") &&
          atoi(Command("dont_check_input"))))
    {
        int no_warning = 0;
        for (CheckMap::iterator iter = command_check.begin();
             iter != command_check.end(); iter++)
        {
            if (iter->second == 1)
            {
                controller->printf(
                    "Warning: CV command '%s' is set, but never used.\n",
                    iter->first.c_str());
                no_warning += 1;
            }
        }
        for (CheckMap::iterator iter = choice_check.begin();
             iter != choice_check.end(); iter++)
        {
            if (iter->second == 2)
            {
                controller->printf(
                    "Warning: the value '%s' of CV command '%s' matches none "
                    "of the choices.\n",
                    this->commands[iter->first].c_str(), iter->first.c_str());
                no_warning += 1;
            }
            else if (iter->second == 3)
            {
                controller->printf("Warning: CV command '%s' is not set.\n",
                                   iter->first.c_str());
                no_warning += 1;
            }
        }
        if (no_warning)
        {
            controller->printf(
                "\nWarning: CV inputs raised %d warning(s). If You know WHAT "
                "YOU ARE DOING, press any key to continue. Set "
                "dont_check_input = 1 to disable this warning.\n",
                no_warning);
            getchar();
        }
    }
}

void COLLECTIVE_VARIABLE_CONTROLLER::Print_Initial()
{
    if (!is_initialized) return;
    controller->printf("START INITIALIZING CV PRINTER:\n");
    print_cv_list = Ask_For_CV("print", 0);
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        if (controller->outputs_content.count(print_cv_list[i]->module_name))
        {
            std::string error_reason = "Reason:\n\tthe name of the CV '";
            error_reason += print_cv_list[i]->module_name;
            error_reason += "' to print is the same with a built-in output\n";
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Print_Initial",
                error_reason.c_str());
        }
    }
    controller->printf("END INITIALIZING CV PRINTER\n\n");
}

void COLLECTIVE_VARIABLE_CONTROLLER::Compute_CV_For_Print(
    int atom_numbers, VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell, int steps,
    int interval, bool print_zeroth_frame)
{
    if (!((print_zeroth_frame || steps) && (steps % interval == 0))) return;
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        print_cv_list[i]->Compute(atom_numbers, crd, cell, rcell,
                                  CV_NEED_CPU_VALUE, steps);
    }
}

void COLLECTIVE_VARIABLE_CONTROLLER::Step_Print()
{
    if (!is_initialized) return;
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        for (int i = 0; i < print_cv_list.size(); i++)
        {
            controller->Step_Print(print_cv_list[i]->module_name,
                                   print_cv_list[i]->value);
        }
        return;
    }
    else  // 把最后一个进程号的信息发给0号进程
    {
#ifdef USE_MPI
        for (int i = 0; i < print_cv_list.size(); i++)
        {
            if (CONTROLLER::MPI_rank == 0)
            {
                MPI_Recv(&print_cv_list[i]->value, 1, MPI_FLOAT,
                         CONTROLLER::MPI_size - 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                controller->Step_Print(print_cv_list[i]->module_name,
                                       print_cv_list[i]->value);
            }
            if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
            {
                MPI_Send(&print_cv_list[i]->value, 1, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
#endif
    }
}

COLLECTIVE_VARIABLE_PROTOTYPE* COLLECTIVE_VARIABLE_CONTROLLER::get_CV(
    const char* cv_name)
{
    if (!is_initialized)
    {
        this->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
            "Reason:\n\tcommand 'cv_in_file' is not set\n");
    }
    if (CV_INSTANCE_MAP->count(cv_name))
    {
        return CV_INSTANCE_MAP[0][cv_name];
    }
    if (Command_Exist(cv_name, "CV_type"))
    {
        std::string cv_type = Command(cv_name, "CV_type");
        if (CV_MAP->count(cv_type))
        {
            COLLECTIVE_VARIABLE_PROTOTYPE* cv =
                CV_MAP[0][cv_type](this, cv_name);
            CV_INSTANCE_MAP[0][cv_name] = cv;
            return CV_INSTANCE_MAP[0][cv_name];
        }
        else
        {
            std::string error_reason = string_format(
                "Reason:\n\tthe type '%CV_TYPE%' of the CV '%CV_NAME%' is "
                "undefined",
                {{"CV_TYPE", cv_type}, {"CV_NAME", cv_name}});
            this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand,
                                     "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
                                     error_reason.c_str());
        }
    }
    else
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe type of the CV '%CV_NAME%' is required",
            {{"CV_NAME", cv_name}});
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand,
                                 "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
                                 error_reason.c_str());
    }
    return 0;
}

CV_LIST COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV(const char* name, int N,
                                                   float verbose_level,
                                                   int layout)
{
    CV_LIST cv_list;
    int CV_numbers = 0;
    std::vector<std::string> cv_names;
    std::string command = string_format("%NAME%_CV", {{"NAME", name}});
    if (Command_Exist(command.c_str()))
    {
        command = string_strip(Original_Command(command.c_str()));
        cv_names = string_split(command, " ");
        CV_numbers = cv_names.size();
    }
    if (N > 0 && CV_numbers != N)
    {
        std::string error_reason = string_format(
            "Reason:\n\t%N_NEED% CV(s) should be given to %NAME%, but %N_CV% "
            "found",
            {{"N_NEED", std::to_string(N)},
             {"NAME", name},
             {"N_CV", std::to_string(CV_numbers)}});
        Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                           "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV",
                           error_reason.c_str());
    }
    else if (N <= 0 && CV_numbers < -N)
    {
        std::string error_reason = string_format(
            "Reason:\n\tat least %N_NEED% CV(s) should be given to %NAME%, but "
            "only %N_CV% found",
            {{"N_NEED", std::to_string(N)},
             {"NAME", name},
             {"N_CV", std::to_string(CV_numbers)}});
        Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                           "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV",
                           error_reason.c_str());
    }
    if (verbose_level > -2)
    {
        for (int i = 0; i < layout; i++)
        {
            printf("    ");
        }
        printf("%d CV(s) found for %s\n", CV_numbers, name);
    }
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    for (int i = 0; i < CV_numbers; i++)
    {
        command = cv_names[i];
        if (verbose_level > -1)
        {
            for (int ii = 0; ii < layout; ii++)
            {
                printf("    ");
            }
            printf("    CV %d: %s\n", i, command.c_str());
        }

        cv = get_CV(command.c_str());
        if (verbose_level > -1)
        {
            for (int ii = 0; ii < layout; ii++)
            {
                printf("    ");
            }
            printf("        type of '%s' is '%s'\n", command.c_str(),
                   cv->type_name);
        }
        cv_list.push_back(cv);
    }
    return cv_list;
}

int* COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, int default_value, float verbose_level,
    const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    int* t;
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    Malloc_Safely((void**)&t, sizeof(int) * N);
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe number of parameters '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"SIZE", std::to_string(parameters.size())}});
        Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
            error_reason.c_str());
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = string_format(
                "%f%%s% (from default value)",
                {{"f", std::to_string(default_value)}, {"s", unit_string}});
        }
        else if (controller->Command_Exist(parameters[i].c_str()))
        {
            controller->Check_Int(
                parameters[i].c_str(),
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter");
            t[i] = atoi(controller->Command(parameters[i].c_str()));
            hint = string_format("%f%%s% (from command %C%)",
                                 {{"f", std::to_string(t[i])},
                                  {"s", unit_string},
                                  {"C", parameters[i]}});
        }
        else if (cv_vatom_name.count(parameters[i]))
        {
            t[i] = cv_vatom_name[parameters[i]] + atom_numbers;
            hint = string_format("%f%%s% (from cv virtual atom)",
                                 {{"f", parameters[i]}, {"s", unit_string}});
        }
        else if (!is_str_int(parameters[i].c_str()))
        {
            std::string error_reason = string_format(
                "Reason:\n\t%VALUE% (the %i%-th value of '%PARAMETER_NAME%'\
for CV '%CV_NAME%') is not an int",
                {{"CV_NAME", name},
                 {"PARAMETER_NAME", parameter_name},
                 {"VALUE", parameters[i]},
                 {"i", std::to_string(i)}});
        }
        else
        {
            t[i] = atoi(parameters[i].c_str());
            hint = string_format(
                "%f%%s%", {{"f", std::to_string(t[i])}, {"s", unit_string}});
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

float* COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, float default_value, float verbose_level,
    const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    float* t;
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    Malloc_Safely((void**)&t, sizeof(float) * N);
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe number of parameters '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"SIZE", std::to_string(parameters.size())}});
        Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter",
            error_reason.c_str());
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = string_format(
                "%f%%s% (from default value)",
                {{"f", std::to_string(default_value)}, {"s", unit_string}});
        }
        else if (controller->Command_Exist(parameters[i].c_str()))
        {
            controller->Check_Float(
                parameters[i].c_str(),
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter");
            t[i] = (float)atof(controller->Command(parameters[i].c_str()));
            hint = string_format("%f%%s% (from command %C%)",
                                 {{"f", std::to_string(t[i])},
                                  {"s", unit_string},
                                  {"C", parameters[i]}});
        }
        else if (!is_str_float(parameters[i].c_str()))
        {
            std::string error_reason = string_format(
                "Reason:\n\t%VALUE% (the %i%-th value of '%PARAMETER_NAME%'\
for CV '%CV_NAME%') is not a float",
                {{"CV_NAME", name},
                 {"PARAMETER_NAME", parameter_name},
                 {"VALUE", parameters[i]},
                 {"i", std::to_string(i)}});
        }
        else
        {
            t[i] = (float)atof(parameters[i].c_str());
            hint = string_format(
                "%f%%s%", {{"f", std::to_string(t[i])}, {"s", unit_string}});
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

std::vector<std::string>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_String_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, const char* default_value,
    float verbose_level, const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    std::vector<std::string> t(N);
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\t%N% number of parameter '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"%SIZE", std::to_string(parameters.size())}});
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = t[i] + unit_string + " (from default value)";
        }
        else
        {
            t[i] = parameters[i];
            hint = t[i] + unit_string;
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

std::vector<int>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_Int_Parameter(
    const char* name, const char* parameter_name)
{
    std::vector<int> ints;
    std::string out;
    std::string file_name = parameter_name;
    file_name += "_in_file";
    std::istream* ss = NULL;
    if (Command_Exist(name, parameter_name))
    {
        std::string strs = Original_Command(name, parameter_name);
        ss = new std::istringstream(strs);
    }
    else if (Command_Exist(name, file_name.c_str()))
    {
        ss = new std::ifstream(Command(name, file_name.c_str()));
    }
    if (ss != NULL)
    {
        while (ss[0] >> out)
        {
            if (cv_vatom_name.count(out))
            {
                ints.push_back(cv_vatom_name[out] + atom_numbers);
            }
            else if (controller->Command_Exist(out.c_str()))
            {
                controller->Check_Int(out.c_str(),
                                      "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_"
                                      "Indefinite_Length_Int_Parameter");
                ints.push_back(atoi(controller->Command(out.c_str())));
            }
            else
            {
                if (!is_str_int(out.c_str()))
                {
                    Throw_SPONGE_Error(
                        spongeErrorTypeErrorCommand,
                        "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_"
                        "Length_Int_Parameter",
                        "Reason:\n\tone of the value is not an int\n");
                }
                ints.push_back(atoi(out.c_str()));
            }
        }
    }
    return ints;
}

std::vector<float>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_Float_Parameter(
    const char* name, const char* parameter_name)
{
    std::vector<float> floats;
    std::string out;
    std::string file_name = parameter_name;
    file_name += "_in_file";
    std::istream* ss = NULL;
    if (Command_Exist(name, parameter_name))
    {
        std::string strs = Original_Command(name, parameter_name);
        ss = new std::istringstream(strs);
    }
    else if (Command_Exist(name, file_name.c_str()))
    {
        ss = new std::ifstream(Command(name, file_name.c_str()));
    }
    if (ss != NULL)
    {
        while (ss[0] >> out)
        {
            if (cv_vatom_name.count(out))
            {
                floats.push_back(cv_vatom_name[out] + atom_numbers);
            }
            else if (controller->Command_Exist(out.c_str()))
            {
                controller->Check_Float(
                    out.c_str(),
                    "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_"
                    "Float_Parameter");
                floats.push_back(atof(controller->Command(out.c_str())));
            }
            else
            {
                if (!is_str_float(out.c_str()))
                {
                    Throw_SPONGE_Error(
                        spongeErrorTypeErrorCommand,
                        "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_"
                        "Length_Float_Parameter",
                        "Reason:\n\tone of the value is not a float\n");
                }
                floats.push_back(atof(out.c_str()));
            }
        }
    }
    return floats;
}

int COLLECTIVE_VARIABLE_PROTOTYPE::Check_Whether_Computed_At_This_Step(int step,
                                                                       int need)
{
    if ((need & CV_NEED_CPU_VALUE) &&
        (last_update_step[CV_NEED_CPU_VALUE] == step))
        need &= ~CV_NEED_CPU_VALUE;
    if ((need & CV_NEED_GPU_VALUE) &&
        (last_update_step[CV_NEED_GPU_VALUE] == step))
        need &= ~CV_NEED_GPU_VALUE;
    if ((need & CV_NEED_CRD_GRADS) &&
        (last_update_step[CV_NEED_CRD_GRADS] == step))
        need &= ~CV_NEED_CRD_GRADS;
    if ((need & CV_NEED_VIRIAL) && (last_update_step[CV_NEED_VIRIAL] == step))
        need &= ~CV_NEED_VIRIAL;
    return need;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Record_Update_Step_Of_Slow_Computing_CV(
    int step, int need)
{
    if (need & CV_NEED_CPU_VALUE) last_update_step[CV_NEED_CPU_VALUE] = step;
    if (need & CV_NEED_CRD_GRADS) last_update_step[CV_NEED_CRD_GRADS] = step;
    if (need & CV_NEED_GPU_VALUE) last_update_step[CV_NEED_GPU_VALUE] = step;
    if (need & CV_NEED_VIRIAL) last_update_step[CV_NEED_VIRIAL] = step;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Record_Update_Step_Of_Fast_Computing_CV(
    int step, int need)
{
    last_update_step[CV_NEED_VIRIAL] = step;
    last_update_step[CV_NEED_CRD_GRADS] = step;
    last_update_step[CV_NEED_GPU_VALUE] = step;
    last_update_step[CV_NEED_CPU_VALUE] = step;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Super_Initial(
    COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
    const char* module_name)
{
    strcpy(this->module_name, module_name);
    int total_atom_numbers = atom_numbers;
    if (manager != NULL)
    {
        total_atom_numbers += manager->cv_vatom_name.size();
    }
    Device_Malloc_Safely((void**)&crd_grads,
                         sizeof(VECTOR) * total_atom_numbers);
    Device_Malloc_Safely((void**)&virial, sizeof(LTMatrix3));
    Device_Malloc_Safely((void**)&d_value, sizeof(float));
    deviceMemset(crd_grads, 0, sizeof(VECTOR) * total_atom_numbers);
    deviceMemset(virial, 0, sizeof(LTMatrix3));
#ifdef GPU_ARCH_NAME
    deviceStreamCreate(&device_stream);
#endif
    last_update_step[CV_NEED_GPU_VALUE] = -1;
    last_update_step[CV_NEED_CPU_VALUE] = -1;
    last_update_step[CV_NEED_CRD_GRADS] = -1;
    last_update_step[CV_NEED_VIRIAL] = -1;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Initial(
    COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
    const char* module_name)
{
    Super_Initial(manager, atom_numbers, module_name);
}
