#include "../meta.h"
void META::Initial(CONTROLLER* controller,
                   COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                   char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "meta");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!cv_controller->Command_Exist(this->module_name, "CV"))
    {
        controller->printf("META IS NOT INITIALIZED\n\n");
        return;
    }
    else
    {
        std::vector<std::string> cv_str =
            cv_controller->Ask_For_String_Parameter(this->module_name, "CV",
                                                    ndim);
        std::string cvv =
            std::accumulate(cv_str.begin(), cv_str.end(), std::string(""));
        printf("%s contains %d dimension META\n", cvv.c_str(), ndim);
    }
    if (cv_controller->Command_Exist(this->module_name, "dip"))
    {
        dip = cv_controller->Ask_For_Float_Parameter(this->module_name, "dip",
                                                     1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "welltemp_factor"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "welltemp_factor");
        welltemp_factor = temp_value[0];
        free(temp_value);
        if (welltemp_factor > 1)
        {
            is_welltemp = 1;
        }
        else
        {
            printf("\tWell-tempered Factor must larger than 1!\n");
            getchar();
        }
    }
    cvs = cv_controller->Ask_For_CV(this->module_name, -1);
    if (cv_controller->Command_Exist(this->module_name, "Ndim"))
    {
        ndim = *cv_controller->Ask_For_Int_Parameter(this->module_name, "Ndim");
        if (ndim != cvs.size())
        {
            controller->printf("%d D-META IS NOT CONSISTANT CV size %d\n\n",
                               ndim, cvs.size());
            return;
        }
    }
    else
    {
        ndim = cvs.size();
    }
    controller->printf("START INITIALIZING %dD-META:\n", ndim);
    // initialize Dpotential_local
    Malloc_Safely((void**)&Dpotential_local, sizeof(float) * ndim);
    sprintf(read_potential_file_name, "Meta_Potential.txt");
    sprintf(write_potential_file_name, "Meta_Potential.txt");
    if (controller->Command_Exist("default_in_file_prefix"))
    {
        sprintf(read_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_in_file_prefix"));
    }
    else
    {
        sprintf(read_potential_file_name, "Meta_Potential.txt");
    }
    if (controller->Command_Exist("default_out_file_prefix"))
    {
        sprintf(write_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_out_file_prefix"));
    }
    else
    {
        sprintf(write_potential_file_name, "Meta_Potential.txt");
    }
    sprintf(write_directly_file_name, "Meta_directly.txt");
    sprintf(edge_file_name, "sumhill.log");
    if (cv_controller->Command_Exist(this->module_name, "edge_in_file"))
    {
        strcpy(edge_file_name, cv_controller
                                   ->Ask_For_String_Parameter(this->module_name,
                                                              "edge_in_file")[0]
                                   .c_str());
    }
    if (cv_controller->Command_Exist(this->module_name, "subhill"))
    {
        subhill = true;
        printf("reading subhill for meta: 1\n");
    }
    if (cv_controller->Command_Exist(this->module_name, "kde"))
    {
        int kde_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "kde", 1)[0];
        if (kde_dim)
        {
            kde = true;
            subhill = true;
            printf("reading kde's subhill for meta: %d\n", kde_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "mask"))
    {
        mask = cv_controller->Ask_For_Int_Parameter(this->module_name, "mask",
                                                    1)[0];
        if (mask)
        {
            printf("reading mask dimension meta: %d\n", mask);
            if (cv_controller->Command_Exist(this->module_name, "maxforce"))
            {
                maxforce = cv_controller->Ask_For_Float_Parameter(
                    this->module_name, "maxforce", 1)[0];
            }
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sink"))
    {
        int sub_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "sink", 1)[0];
        if (sub_dim > 0)
        {
            do_negative = true;
            printf("reading sink/submarine dimension for meta: %d\n", sub_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sumhill_freq"))
    {
        history_freq = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "sumhill_freq", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "catheter"))
    {
        usegrid = false;
        use_scatter = true;
        do_negative = true;
        catheter =
            3;  // cv_controller->Ask_For_Int_Parameter(this->module_name,
                // "catheter", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "convmeta"))
    {
        // use_scatter = true;
        do_negative = true;
        convmeta = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                        "convmeta", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "grw"))
    {
        // use_scatter = true;
        do_negative = true;
        grw = cv_controller->Ask_For_Int_Parameter(this->module_name, "grw",
                                                   1)[0];
    }
    cv_periods = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_period", cvs.size(), 1, false);
    cv_sigmas = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                       "CV_sigma", cvs.size());
    cutoff = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_sigma", cvs.size(), 1, false, 0., -3);
    if (cv_controller->Command_Exist(this->module_name, "cutoff"))
    {
        do_cutoff = true;
        cutoff = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                        "cutoff", cvs.size());
    }
    for (int i = 0; i < cvs.size(); i++)
    {
        if (cv_sigmas[i] <= 0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                "CV_sigma should always be greater than 0");
        }
        if (!do_cutoff)
        {
            cutoff[i] = 3 * cv_sigmas[i];
        }
        if (kde)
        {
            cv_sigmas[i] =
                1.414f / cv_sigmas[i];  ///  inverted sigma!!!!!!!!!!!!!!!!!!!
        }
        else
        {
            cv_sigmas[i] =
                1.0 / cv_sigmas[i];  /// inverted sigma!!!!!!!!!!!!!!!!!!!
        }
    }
    float sqrtpi = sqrtf(CONSTANT_Pi * 2);
    for (int i = 0; i < ndim; i++)
    {
        sigmas.push_back(cv_sigmas[i]);
        periods.push_back(cv_periods[i]);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_in_file"))
    {
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else if (cv_controller->Command_Exist(this->module_name, "scatter_in_file"))
    {
        usegrid = false;
        use_scatter = true;
        printf("Use %d scatter point for CV!\n", scatter_size);
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "scatter_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else
    {
        if (cv_controller->Command_Exist(this->module_name, "scatter"))
        {
            scatter_size = *(cv_controller->Ask_For_Int_Parameter(
                this->module_name, "scatter", 1));
            if (scatter_size > 0)
            {
                usegrid = false;
                use_scatter = true;
                printf("Use %d scatter point for CV!\n", scatter_size);
                for (int i = 0; i < cvs.size(); i++)
                {
                    tcoor.push_back(cv_controller->Ask_For_Float_Parameter(
                        cvs[i]->module_name, "CV_point", scatter_size, 1,
                        false));
                }
            }
            else
            {
                printf("Not using scatter point for CV\n");
                use_scatter = false;
            }
        }

        cv_mins = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_minimal", cvs.size());
        cv_maxs = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_maximum", cvs.size());
        n_grids = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                       "CV_grid", cvs.size());
        for (int i = 0; i < cvs.size(); ++i)
        {
            if (cv_maxs[i] <= cv_mins[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_maximum should always be greater than CV_minimal");
            }
            if (n_grids[i] <= 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_grid should always be greater than 1");
            }
            cv_deltas.push_back((cv_maxs[i] - cv_mins[i]) / n_grids[i]);
        }
        Setgrid(controller);
    }
    height_0 = 1.0;
    if (cv_controller->Command_Exist(this->module_name, "height"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "height");
        height_0 = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "wall_height"))
    {
        do_borderwall = true;
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "wall_height");
        border_potential_height = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_out_file"))
    {
        strcpy(write_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_out_file")[0]
                   .c_str());
    }
    bool has_potential_update_interval = false;
    if (cv_controller->Command_Exist(this->module_name,
                                     "potential_update_interval"))
    {
        int* temp_value = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "potential_update_interval");
        potential_update_interval = temp_value[0];
        has_potential_update_interval = true;
        free(temp_value);
    }
    if (controller->Command_Exist("write_information_interval"))
    {
        write_information_interval =
            atoi(controller->Command("write_information_interval"));
    }
    else
    {
        write_information_interval = 1000;
    }
    if (write_information_interval <= 0)
    {
        write_information_interval = 1000;
    }
    if (!has_potential_update_interval)
    {
        controller->printf(
            "    Potential update interval is set to "
            "write_information_interval by default\n");
        potential_update_interval = write_information_interval;
    }
    if (potential_update_interval <= 0)
    {
        potential_update_interval = 1000;
    }
    controller->Step_Print_Initial("meta", "%f");
    controller->Step_Print_Initial("rbias", "%f");
    controller->Step_Print_Initial("rct", "%f");
    controller->printf("    potential output file: %s\n",
                       write_potential_file_name);
    controller->printf("    edge effect file: %s\n", edge_file_name);
    is_initialized = 1;
    controller->printf("END INITIALIZING META\n\n");
}
