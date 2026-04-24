#include "basis/basis.h"
#include "ecp/ecp_library.h"
#include "guess/minao.h"
#include "guess/sap.h"
#include "quantum_chemistry.h"

static void Init_ERI_Workspace_Params(QUANTUM_CHEMISTRY* qc,
                                      CONTROLLER* controller, int max_l)
{
    const int max_total_l = 4 * max_l;
    qc->task_ctx.params.eri_hr_base = max_total_l + 1;
    if (qc->task_ctx.params.eri_hr_base > HR_BASE_MAX)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorOverflow, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    basis angular momentum too high (max l=%d, required "
            "hr_base=%d, supported <=%d)\n",
            max_l, qc->task_ctx.params.eri_hr_base, HR_BASE_MAX);
    }
    qc->task_ctx.params.eri_hr_size =
        qc->task_ctx.params.eri_hr_base * qc->task_ctx.params.eri_hr_base *
        qc->task_ctx.params.eri_hr_base * qc->task_ctx.params.eri_hr_base;

    const int max_cart = (max_l + 1) * (max_l + 2) / 2;
    qc->task_ctx.params.eri_shell_buf_size =
        max_cart * max_cart * max_cart * max_cart;
    qc->task_ctx.params.eri_shell_buf_size = std::max(
        1, std::min(MAX_SHELL_ERI, qc->task_ctx.params.eri_shell_buf_size));
}

static void Build_Shell_Pairs_And_Pair_Types(QUANTUM_CHEMISTRY* qc, int max_l)
{
    auto& mol = qc->mol;
    auto& task_ctx = qc->task_ctx;

    task_ctx.topo.h_shell_pairs.clear();
    for (int i = 0; i < mol.nbas; i++)
        for (int j = 0; j <= i; j++)
            task_ctx.topo.h_shell_pairs.push_back({i, j});
    task_ctx.topo.n_shell_pairs = task_ctx.topo.h_shell_pairs.size();

    const int stride = max_l + 1;
    const int n_types = stride * stride;

    std::vector<std::vector<int>> type_lists(n_types);
    for (int pid = 0; pid < task_ctx.topo.n_shell_pairs; pid++)
    {
        const auto& p = task_ctx.topo.h_shell_pairs[pid];
        int tid = mol.h_l_list[p.x] * stride + mol.h_l_list[p.y];
        type_lists[tid].push_back(pid);
    }

    task_ctx.topo.h_sorted_pair_ids.clear();
    task_ctx.topo.h_sorted_pair_ids.reserve(task_ctx.topo.n_shell_pairs);
    task_ctx.topo.n_pair_types = 0;
    for (int tid = 0; tid < n_types; tid++)
    {
        if (type_lists[tid].empty()) continue;
        int slot = task_ctx.topo.n_pair_types++;
        task_ctx.topo.pair_type_offset[slot] =
            (int)task_ctx.topo.h_sorted_pair_ids.size();
        task_ctx.topo.pair_type_count[slot] = (int)type_lists[tid].size();
        task_ctx.topo.pair_type_l0[slot] = tid / stride;
        task_ctx.topo.pair_type_l1[slot] = tid % stride;
        for (int pid : type_lists[tid])
            task_ctx.topo.h_sorted_pair_ids.push_back(pid);
    }

    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.buffers.d_sorted_pair_ids,
        (void*)task_ctx.topo.h_sorted_pair_ids.data(),
        sizeof(int) * task_ctx.topo.h_sorted_pair_ids.size());
}

static void Build_Screening_Combos_And_Task_Buffers(QUANTUM_CHEMISTRY* qc)
{
    auto& task_ctx = qc->task_ctx;
    auto& mol = qc->mol;

    const int npt = task_ctx.topo.n_pair_types;
    task_ctx.topo.n_combos = 0;
    for (int tA = 0; tA < npt; tA++)
    {
        for (int tB = 0; tB <= tA; tB++)
        {
            const int nA = task_ctx.topo.pair_type_count[tA];
            const int nB = task_ctx.topo.pair_type_count[tB];
            const bool same = (tA == tB);
            const int nq = same ? nA * (nA + 1) / 2 : nA * nB;
            if (nq == 0) continue;

            auto& c = task_ctx.topo.h_combos[task_ctx.topo.n_combos];
            c.pair_base_A = task_ctx.topo.pair_type_offset[tA];
            c.n_A = nA;
            c.pair_base_B = task_ctx.topo.pair_type_offset[tB];
            c.n_B = nB;
            c.n_quartets = nq;
            c.output_offset = 0;
            c.same_type = same ? 1 : 0;
            c.l0 = task_ctx.topo.pair_type_l0[tA];
            c.l1 = task_ctx.topo.pair_type_l1[tA];
            c.l2 = task_ctx.topo.pair_type_l0[tB];
            c.l3 = task_ctx.topo.pair_type_l1[tB];
            task_ctx.topo.n_combos++;
        }
    }

    task_ctx.topo.combo_prefix[0] = 0;
    for (int i = 0; i < task_ctx.topo.n_combos; i++)
        task_ctx.topo.combo_prefix[i + 1] =
            task_ctx.topo.combo_prefix[i] +
            task_ctx.topo.h_combos[i].n_quartets;
    task_ctx.topo.total_quartets =
        task_ctx.topo.combo_prefix[task_ctx.topo.n_combos];

    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.buffers.d_combos, (void*)task_ctx.topo.h_combos,
        sizeof(QC_INTEGRAL_TASKS::ScreenCombo) * task_ctx.topo.n_combos);

    task_ctx.buffers.screened_buf_capacity =
        std::max(1, task_ctx.topo.total_quartets);
    for (int i = 0; i < task_ctx.topo.n_combos; i++)
    {
        task_ctx.topo.h_combos[i].output_offset = task_ctx.topo.combo_prefix[i];
    }
    Device_Malloc_Safely(
        (void**)&task_ctx.buffers.d_screened_tasks,
        sizeof(QC_ERI_TASK) * task_ctx.buffers.screened_buf_capacity);
    Device_Malloc_Safely((void**)&task_ctx.buffers.d_screen_counts,
                         sizeof(int) * std::max(1, task_ctx.topo.n_combos));
    if (task_ctx.buffers.d_combos != NULL)
        deviceMemcpy(
            task_ctx.buffers.d_combos, task_ctx.topo.h_combos,
            sizeof(QC_INTEGRAL_TASKS::ScreenCombo) * task_ctx.topo.n_combos,
            deviceMemcpyHostToDevice);

    for (int i = 0; i < mol.nbas; i++)
        for (int j = 0; j < mol.nbas; j++)
            task_ctx.topo.h_1e_tasks.push_back({i, j});
    task_ctx.topo.n_1e_tasks = task_ctx.topo.h_1e_tasks.size();

    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.buffers.d_1e_tasks,
        (void*)task_ctx.topo.h_1e_tasks.data(),
        sizeof(QC_ONE_E_TASK) * task_ctx.topo.h_1e_tasks.size());
    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.buffers.d_shell_pairs,
        (void*)task_ctx.topo.h_shell_pairs.data(),
        sizeof(QC_ONE_E_TASK) * task_ctx.topo.h_shell_pairs.size());
}

bool QUANTUM_CHEMISTRY::Parsing_Arguments(CONTROLLER* controller,
                                          const int atom_numbers,
                                          const char*& qc_type_file,
                                          std::string& basis_set_name)
{
    if (!controller->Command_Exist("qc_type_in_file"))
    {
        is_initialized = 0;
        return false;
    }
    qc_type_file = controller->Command("qc_type_in_file");

    std::string model_chemistry = "HF/6-31g";
    if (controller->Command_Exist("qc_model_chemistry"))
    {
        model_chemistry = controller->Command("qc_model_chemistry");
    }
    int slash_pos = model_chemistry.find('/');
    if (slash_pos == std::string::npos)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    qc_model_chemistry format error: expected "
            "\"METHOD/<basis>\", got \"%s\"\n",
            model_chemistry.c_str());
    }
    std::string method_name =
        string_strip(model_chemistry.substr(0, slash_pos));
    basis_set_name = string_strip(model_chemistry.substr(slash_pos + 1));

    if (is_str_equal(method_name.c_str(), "HF", 0))
    {
        method = QC_METHOD::HF;
        dft.exx_fraction = 1.0f;
        dft.enable_dft = 0;
    }
    else if (is_str_equal(method_name.c_str(), "LDA", 0))
    {
        method = QC_METHOD::LDA;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (is_str_equal(method_name.c_str(), "PBE", 0))
    {
        method = QC_METHOD::PBE;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (is_str_equal(method_name.c_str(), "BLYP", 0))
    {
        method = QC_METHOD::BLYP;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (is_str_equal(method_name.c_str(), "PBE0", 0))
    {
        method = QC_METHOD::PBE0;
        dft.exx_fraction = 0.25f;
        dft.enable_dft = 1;
    }
    else if (is_str_equal(method_name.c_str(), "B3LYP", 0))
    {
        method = QC_METHOD::B3LYP;
        dft.exx_fraction = 0.20f;
        dft.enable_dft = 1;
    }
    else
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    qc_model_chemistry \"%s\" not supported. Supported "
            "methods: HF, LDA, PBE, BLYP, PBE0, B3LYP.\n",
            model_chemistry.c_str());
    }

    if (controller->Command_Exist("qc_need_gradient"))
    {
        need_gradient = atoi(controller->Command("qc_need_gradient"));
    }

    task_ctx.params.eri_prim_screen_tol = 1e-12f;
    if (controller->Command_Exist("qc_eri_prim_screen_tol"))
    {
        controller->Check_Float("qc_eri_prim_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.params.eri_prim_screen_tol =
            atof(controller->Command("qc_eri_prim_screen_tol"));
        if (task_ctx.params.eri_prim_screen_tol < 0.0f)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_eri_prim_screen_tol must be >= 0, got %g\n",
                (double)task_ctx.params.eri_prim_screen_tol);
        }
    }

    task_ctx.params.direct_eri_prim_screen_tol = 1e-10f;
    if (controller->Command_Exist("qc_direct_eri_prim_screen_tol"))
    {
        controller->Check_Float("qc_direct_eri_prim_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.params.direct_eri_prim_screen_tol =
            atof(controller->Command("qc_direct_eri_prim_screen_tol"));
        if (task_ctx.params.direct_eri_prim_screen_tol < 0.0f)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_direct_eri_prim_screen_tol must be >= 0, got "
                "%g\n",
                (double)task_ctx.params.direct_eri_prim_screen_tol);
        }
    }

    task_ctx.params.eri_shell_screen_tol = 1e-10f;
    if (controller->Command_Exist("qc_eri_shell_screen_tol"))
    {
        controller->Check_Float("qc_eri_shell_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.params.eri_shell_screen_tol =
            atof(controller->Command("qc_eri_shell_screen_tol"));
        if (task_ctx.params.eri_shell_screen_tol < 0.0f)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_eri_shell_screen_tol must be >= 0, got %g\n",
                (double)task_ctx.params.eri_shell_screen_tol);
        }
    }

    scf_ws.runtime.unrestricted = false;
    if (controller->Command_Exist("qc_restricted"))
    {
        controller->Check_Int("qc_restricted", "QUANTUM_CHEMISTRY::Initial");
        const int qc_restricted = atoi(controller->Command("qc_restricted"));
        if (qc_restricted != 0 && qc_restricted != 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_restricted must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_restricted"));
        }
        scf_ws.runtime.unrestricted = (qc_restricted == 0);
    }

    scf_ws.runtime.max_scf_iter = 100;
    if (controller->Command_Exist("qc_scf_max_iter"))
    {
        controller->Check_Int("qc_scf_max_iter", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.runtime.max_scf_iter =
            atoi(controller->Command("qc_scf_max_iter"));
        if (scf_ws.runtime.max_scf_iter < 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_scf_max_iter must be >= 1, got \"%s\"\n",
                controller->Command("qc_scf_max_iter"));
        }
    }

    scf_ws.runtime.use_diis = true;
    if (controller->Command_Exist("qc_diis"))
    {
        controller->Check_Int("qc_diis", "QUANTUM_CHEMISTRY::Initial");
        const int qc_diis = atoi(controller->Command("qc_diis"));
        if (qc_diis != 0 && qc_diis != 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_diis must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_diis"));
        }
        scf_ws.runtime.use_diis = (qc_diis != 0);
    }

    scf_ws.runtime.diis_start_iter = 2;
    if (controller->Command_Exist("qc_diis_start"))
    {
        controller->Check_Int("qc_diis_start", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.runtime.diis_start_iter =
            atoi(controller->Command("qc_diis_start"));
        if (scf_ws.runtime.diis_start_iter < 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_diis_start must be >= 1, got \"%s\"\n",
                controller->Command("qc_diis_start"));
        }
    }

    scf_ws.runtime.diis_space = 6;
    if (controller->Command_Exist("qc_diis_space"))
    {
        controller->Check_Int("qc_diis_space", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.runtime.diis_space = atoi(controller->Command("qc_diis_space"));
        if (scf_ws.runtime.diis_space < 2)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_diis_space must be >= 2, got \"%s\"\n",
                controller->Command("qc_diis_space"));
        }
    }

    scf_ws.runtime.diis_reg = 1e-10;
    if (controller->Command_Exist("qc_diis_reg"))
    {
        controller->Check_Float("qc_diis_reg", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.runtime.diis_reg = atof(controller->Command("qc_diis_reg"));
        if (scf_ws.runtime.diis_reg < 0.0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_diis_reg must be >= 0, got \"%s\"\n",
                controller->Command("qc_diis_reg"));
        }
    }

    scf_ws.runtime.energy_tol = 1e-6;
    if (controller->Command_Exist("qc_scf_energy_tol"))
    {
        controller->Check_Float("qc_scf_energy_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        scf_ws.runtime.energy_tol =
            atof(controller->Command("qc_scf_energy_tol"));
        if (scf_ws.runtime.energy_tol <= 0.0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_scf_energy_tol must be > 0, got \"%s\"\n",
                controller->Command("qc_scf_energy_tol"));
        }
    }

    scf_ws.runtime.print_iter = false;
    if (controller->Command_Exist("qc_scf_print_iter"))
    {
        controller->Check_Int("qc_scf_print_iter",
                              "QUANTUM_CHEMISTRY::Initial");
        const int qc_scf_print_iter =
            atoi(controller->Command("qc_scf_print_iter"));
        if (qc_scf_print_iter != 0 && qc_scf_print_iter != 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_scf_print_iter must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_scf_print_iter"));
        }
        scf_ws.runtime.print_iter = (qc_scf_print_iter != 0);
    }

    if (controller->Command_Exist("qc_level_shift"))
    {
        scf_ws.runtime.level_shift =
            atof(controller->Command("qc_level_shift"));
    }

    if (controller->Command_Exist("qc_scf_output"))
    {
        const char* fname = controller->Command("qc_scf_output");
        Open_File_Safely(&scf_output_file, fname, "w");
    }

    dft.dft_radial_points = 60;
    if (controller->Command_Exist("qc_dft_radial_points"))
    {
        controller->Check_Int("qc_dft_radial_points",
                              "QUANTUM_CHEMISTRY::Initial");
        dft.dft_radial_points = std::max(
            10,
            std::min(200, atoi(controller->Command("qc_dft_radial_points"))));
    }

    dft.dft_angular_points = 194;
    if (controller->Command_Exist("qc_dft_angular_points"))
    {
        controller->Check_Int("qc_dft_angular_points",
                              "QUANTUM_CHEMISTRY::Initial");
        dft.dft_angular_points = std::max(
            26,
            std::min(590, atoi(controller->Command("qc_dft_angular_points"))));
    }

    initial_guess = QC_INITIAL_GUESS::SAP;
    if (controller->Command_Exist("qc_initial_guess"))
    {
        std::string guess_str = controller->Command("qc_initial_guess");
        std::transform(guess_str.begin(), guess_str.end(), guess_str.begin(),
                       ::tolower);
        if (guess_str == "none")
            initial_guess = QC_INITIAL_GUESS::NONE;
        else if (guess_str == "minao")
            initial_guess = QC_INITIAL_GUESS::MINAO;
        else if (guess_str == "sap")
            initial_guess = QC_INITIAL_GUESS::SAP;
        else
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_initial_guess must be \"none\", \"minao\","
                " or \"sap\", got \"%s\"\n",
                controller->Command("qc_initial_guess"));
        }
    }

    // Density Fitting (RI-JK) 开关
    scf_ws.ri.enabled = false;
    if (controller->Command_Exist("qc_density_fit"))
    {
        controller->Check_Int("qc_density_fit", "QUANTUM_CHEMISTRY::Initial");
        const int qc_df = atoi(controller->Command("qc_density_fit"));
        if (qc_df != 0 && qc_df != 1)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_density_fit must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_density_fit"));
        }
        scf_ws.ri.enabled = (qc_df != 0);
    }

    // DF 模式选择: auto（默认）/ stored / direct
    scf_ws.ri.mode = QC_RI_WORKSPACE::DF_AUTO;
    scf_ws.ri.direct = false;
    if (controller->Command_Exist("qc_density_fitting_mode"))
    {
        std::string mode_str(controller->Command("qc_density_fitting_mode"));
        std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(),
                       ::tolower);
        if (mode_str == "auto")
            scf_ws.ri.mode = QC_RI_WORKSPACE::DF_AUTO;
        else if (mode_str == "stored")
            scf_ws.ri.mode = QC_RI_WORKSPACE::DF_STORED;
        else if (mode_str == "direct")
            scf_ws.ri.mode = QC_RI_WORKSPACE::DF_DIRECT;
        else
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_density_fitting_mode must be "
                "\"auto\", \"stored\", or \"direct\", got \"%s\"\n",
                mode_str.c_str());
        }
    }

    // ECP 选择: auto (默认, 按基组匹配), none, def2-ecp, lanl2dz
    ecp_type = QC_ECP_TYPE::AUTO;
    if (controller->Command_Exist("qc_ecp"))
    {
        std::string ecp_name_str = controller->Command("qc_ecp");
        std::transform(ecp_name_str.begin(), ecp_name_str.end(),
                       ecp_name_str.begin(), ::tolower);
        if (ecp_name_str == "auto")
            ecp_type = QC_ECP_TYPE::AUTO;
        else if (ecp_name_str == "none")
            ecp_type = QC_ECP_TYPE::NONE;
        else if (ecp_name_str == "def2-ecp")
            ecp_type = QC_ECP_TYPE::DEF2_ECP;
        else if (ecp_name_str == "lanl2dz")
            ecp_type = QC_ECP_TYPE::LANL2DZ;
        else
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_ecp must be \"auto\", \"none\", \"def2-ecp\""
                ", or \"lanl2dz\", got \"%s\"\n",
                ecp_name_str.c_str());
        }
    }

    this->atom_numbers = atom_numbers;
    return true;
}

void QUANTUM_CHEMISTRY::Initial_Molecule(CONTROLLER* controller,
                                         const char* qc_type_file,
                                         const std::string& basis_set_name)
{
    static QC_BASIS_SET* all_bases[] = {
        QC_BASIS_STO_3G_PTR,
        QC_BASIS_3_21G_PTR,
        QC_BASIS_631G_PTR,
        QC_BASIS_631G_STAR_PTR,
        QC_BASIS_631G_STARSTAR_PTR,
        QC_BASIS_6311G_PTR,
        QC_BASIS_6311G_STAR_PTR,
        QC_BASIS_6311G_STARSTAR_PTR,
        QC_BASIS_631PG_PTR,
        QC_BASIS_631PPG_PTR,
        QC_BASIS_631PG_STAR_PTR,
        QC_BASIS_631PG_STARSTAR_PTR,
        QC_BASIS_631PPG_STARSTAR_PTR,
        QC_BASIS_6311PG_STAR_PTR,
        QC_BASIS_6311PPG_STARSTAR_PTR,
        QC_BASIS_DEF2_SVP_PTR,
        QC_BASIS_DEF2_TZVP_PTR,
        QC_BASIS_DEF2_TZVPP_PTR,
        QC_BASIS_DEF2_QZVP_PTR,
        QC_BASIS_DEF2_SVPD_PTR,
        QC_BASIS_DEF2_TZVPD_PTR,
        QC_BASIS_MA_DEF2_SVP_PTR,
        QC_BASIS_MA_DEF2_TZVP_PTR,
        QC_BASIS_CC_PVDZ_PTR,
        QC_BASIS_CC_PVTZ_PTR,
        QC_BASIS_AUG_CC_PVDZ_PTR,
        QC_BASIS_AUG_CC_PVTZ_PTR,
    };

    QC_BASIS_SET* basis = nullptr;
    for (auto* b : all_bases)
    {
        if (is_str_equal(basis_set_name.c_str(), b->name, 0))
        {
            basis = b;
            break;
        }
    }
    if (!basis)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    Basis set \"%s\" is not supported.\n",
            basis_set_name.c_str());
    }
    basis->Initialize();
    mol.is_spherical = basis->spherical ? 1 : 0;

    std::vector<std::string> atom_symbols;
    {
        std::ifstream ifs(qc_type_file);
        if (!ifs.is_open())
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    Cannot open %s\n", qc_type_file);
        }
        std::string line;
        std::getline(ifs, line);
        {
            std::istringstream iss(line);
            if (!(iss >> mol.natm >> mol.charge >> mol.multiplicity))
            {
                controller->Throw_Formatted_SPONGE_Error(
                    spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
                    "Reason:\n    Failed to read first line of %s\n",
                    qc_type_file);
            }
        }
        atom_local.reserve(mol.natm);
        for (int i = 0; i < mol.natm; i++)
        {
            std::getline(ifs, line);
            std::istringstream iss(line);
            int idx;
            std::string sym;
            if (!(iss >> idx >> sym))
            {
                controller->Throw_Formatted_SPONGE_Error(
                    spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
                    "Reason:\n    Failed to read atom line %d of %s\n", i,
                    qc_type_file);
            }
            atom_local.push_back(idx);
            atom_symbols.push_back(sym);
        }
    }

    // ECP 查找
    QC_ECP_SET* ecp_set = nullptr;
    switch (ecp_type)
    {
        case QC_ECP_TYPE::AUTO:
            ecp_set = QC_Get_Auto_ECP(basis_set_name.c_str());
            break;
        case QC_ECP_TYPE::DEF2_ECP:
            ecp_set = QC_ECP_DEF2_PTR;
            break;
        case QC_ECP_TYPE::LANL2DZ:
            ecp_set = QC_ECP_LANL2DZ_PTR;
            break;
        case QC_ECP_TYPE::NONE:
            break;
    }

    if (ecp_set) ecp_set->Initialize();

    mol.nelectron = -mol.charge;
    mol.h_Z.resize(mol.natm);
    mol.h_ecp_n_core.resize(mol.natm, 0);
    mol.h_ecp_l_max.resize(mol.natm, -1);
    for (int i = 0; i < mol.natm; ++i)
    {
        auto it_sym = QC_Z_FROM_SYMBOL.find(atom_symbols[i]);
        if (it_sym == QC_Z_FROM_SYMBOL.end())
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    Unknown element symbol %s at atom line %d in "
                "%s\n",
                atom_symbols[i].c_str(), i, qc_type_file);
        }
        int Z = it_sym->second;
        int md_idx = atom_local[i];
        if (md_idx < 0 || md_idx >= this->atom_numbers)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorOverflow, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    MD index %d out of bounds [0, %d)\n", md_idx,
                this->atom_numbers);
        }

        // ECP: 用有效核电荷替代全电荷
        // auto 模式下按基组规范过滤（def2-ECP 只对 Z>=37 生效）
        const bool apply_ecp =
            ecp_set && (ecp_type != QC_ECP_TYPE::AUTO ||
                        QC_Auto_ECP_Applies(basis_set_name.c_str(), Z));
        if (apply_ecp)
        {
            auto it_ecp = ecp_set->data.find(atom_symbols[i]);
            if (it_ecp != ecp_set->data.end())
            {
                const auto& ecp_data = it_ecp->second;
                mol.h_ecp_n_core[i] = ecp_data.n_core;
                mol.h_ecp_l_max[i] = ecp_data.l_max;
                Z -= ecp_data.n_core;
                mol.has_ecp = true;
            }
        }
        mol.h_Z[i] = Z;
        mol.nelectron += Z;
    }

    Device_Malloc_And_Copy_Safely((void**)&mol.d_Z, (void*)mol.h_Z.data(),
                                  sizeof(int) * (int)mol.natm);

    const int spin_e = mol.multiplicity - 1;
    if (spin_e < 0)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    multiplicity must be >= 1, got %d\n",
            mol.multiplicity);
    }
    if (((mol.nelectron + spin_e) & 1) != 0)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorBadFileFormat, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    Inconsistent electron number/multiplicity: N=%d, "
            "multiplicity=%d\n",
            mol.nelectron, mol.multiplicity);
    }
    if (!scf_ws.runtime.unrestricted)
    {
        if (spin_e != 0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_restricted=1 requires closed-shell "
                "multiplicity=1, got multiplicity=%d\n",
                mol.multiplicity);
        }
        if ((mol.nelectron & 1) != 0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    qc_restricted=1 requires even electron number, "
                "got N=%d\n",
                mol.nelectron);
        }
    }

    mol.nao_cart = 0;
    mol.nbas = 0;
    mol.nao_sph = 0;
    mol.nao = 0;
    mol.nao2 = 0;
    for (int i = 0; i < mol.natm; ++i)
    {
        int Z = mol.h_Z[i];
        std::string sym = atom_symbols[i];

        int ptr_coord = mol.h_env.size();
        mol.h_env.push_back(0.0f);
        mol.h_env.push_back(0.0f);
        mol.h_env.push_back(0.0f);
        mol.h_atm.push_back(Z);
        mol.h_atm.push_back(ptr_coord);
        mol.h_atm.push_back(1);
        mol.h_atm.push_back(0);
        mol.h_atm.push_back(0);
        mol.h_atm.push_back(0);

        const std::vector<QC_SHELL_DATA>* shells_ptr = NULL;
        auto it_basis = basis->data.find(sym);
        if (it_basis != basis->data.end()) shells_ptr = &(it_basis->second);

        if (shells_ptr == NULL)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    Basis set %s not available for element %s\n",
                basis_set_name.c_str(), sym.c_str());
        }
        const auto& shells = *shells_ptr;
        for (const auto& shell : shells)
        {
            int ptr_exp = mol.h_env.size();
            mol.h_env.insert(mol.h_env.end(), shell.exps.begin(),
                             shell.exps.end());
            int ptr_coeff = mol.h_env.size();
            mol.h_env.insert(mol.h_env.end(), shell.coeffs.begin(),
                             shell.coeffs.end());

            mol.h_bas.push_back(i);
            mol.h_bas.push_back(shell.l);
            mol.h_bas.push_back(shell.exps.size());
            mol.h_bas.push_back(1);
            mol.h_bas.push_back(0);
            mol.h_bas.push_back(ptr_exp);
            mol.h_bas.push_back(ptr_coeff);
            mol.h_bas.push_back(0);

            int ao_dim = (shell.l + 1) * (shell.l + 2) / 2;
            mol.nao_cart += ao_dim;
            mol.nao_sph += (2 * shell.l + 1);

            mol.h_l_list.push_back(shell.l);
            mol.h_shell_sizes.push_back(shell.exps.size());
            mol.h_shell_offsets.push_back(mol.h_exps.size());

            mol.h_exps.insert(mol.h_exps.end(), shell.exps.begin(),
                              shell.exps.end());
            mol.h_coeffs.insert(mol.h_coeffs.end(), shell.coeffs.begin(),
                                shell.coeffs.end());
            mol.h_centers.push_back(VECTOR(0.0f));

            mol.nbas++;
        }
    }
    if (!mol.is_spherical)
        mol.nao_sph = mol.nao_cart;
    else
        Build_Cart2Sph_Matrix();
    mol.nao = mol.is_spherical ? mol.nao_sph : mol.nao_cart;
    mol.nao2 = (int)((int)mol.nao * (int)mol.nao);
    mol.h_ao_offsets.clear();
    mol.h_ao_offsets_sph.clear();
    int acc = 0;
    int acc_sph = 0;
    for (int k = 0; k < mol.h_l_list.size(); k++)
    {
        const int l = mol.h_l_list[k];
        const int cart_dim = (l + 1) * (l + 2) / 2;
        const int sph_dim = mol.is_spherical ? (2 * l + 1) : cart_dim;
        mol.h_ao_offsets.push_back(acc);
        mol.h_ao_offsets_sph.push_back(acc_sph);
        acc += cart_dim;
        acc_sph += sph_dim;
    }
    mol.h_shell_offsets.clear();
    int exp_acc = 0;
    for (int k = 0; k < mol.h_shell_sizes.size(); k++)
    {
        mol.h_shell_offsets.push_back(exp_acc);
        exp_acc += mol.h_shell_sizes[k];
    }

    Device_Malloc_And_Copy_Safely((void**)&mol.d_atm, (void*)mol.h_atm.data(),
                                  sizeof(int) * mol.h_atm.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_bas, (void*)mol.h_bas.data(),
                                  sizeof(int) * mol.h_bas.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_env, (void*)mol.h_env.data(),
                                  sizeof(float) * mol.h_env.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_centers,
                                  (void*)mol.h_centers.data(),
                                  sizeof(VECTOR) * mol.h_centers.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_l_list,
                                  (void*)mol.h_l_list.data(),
                                  sizeof(int) * mol.h_l_list.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_exps, (void*)mol.h_exps.data(),
                                  sizeof(float) * mol.h_exps.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_coeffs,
                                  (void*)mol.h_coeffs.data(),
                                  sizeof(float) * mol.h_coeffs.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_shell_offsets,
                                  (void*)mol.h_shell_offsets.data(),
                                  sizeof(int) * mol.h_shell_offsets.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_shell_sizes,
                                  (void*)mol.h_shell_sizes.data(),
                                  sizeof(int) * mol.h_shell_sizes.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_ao_offsets,
                                  (void*)mol.h_ao_offsets.data(),
                                  sizeof(int) * mol.h_ao_offsets.size());
    Device_Malloc_And_Copy_Safely((void**)&mol.d_ao_offsets_sph,
                                  (void*)mol.h_ao_offsets_sph.data(),
                                  sizeof(int) * mol.h_ao_offsets_sph.size());
    Device_Malloc_And_Copy_Safely((void**)&d_atom_local,
                                  (void*)atom_local.data(),
                                  sizeof(int) * atom_local.size());

    // 原子坐标数组 (初始为零, 由 Update_Coordinates_From_MD 更新)
    mol.h_atom_coords.resize(mol.natm, VECTOR(0.0f));
    Device_Malloc_And_Copy_Safely((void**)&mol.d_atom_coords,
                                  (void*)mol.h_atom_coords.data(),
                                  sizeof(VECTOR) * mol.natm);

    // ECP 数据扁平化并拷贝到 device
    if (mol.has_ecp && ecp_set)
    {
        mol.ecp_total_channels = 0;
        mol.ecp_total_terms = 0;
        mol.h_ecp_atom_channel_range.resize(mol.natm + 1);

        for (int i = 0; i < mol.natm; i++)
        {
            mol.h_ecp_atom_channel_range[i] = mol.ecp_total_channels;
            if (mol.h_ecp_l_max[i] < 0) continue;

            const auto& ecp_data = ecp_set->data.at(atom_symbols[i]);
            for (const auto& ch : ecp_data.channels)
            {
                mol.h_ecp_l.push_back(ch.l);
                mol.h_ecp_channel_offsets.push_back(mol.ecp_total_terms);
                mol.h_ecp_channel_sizes.push_back((int)ch.terms.size());
                for (const auto& t : ch.terms)
                {
                    mol.h_ecp_d.push_back(t.d_k);
                    mol.h_ecp_zeta.push_back(t.zeta_k);
                    mol.h_ecp_n.push_back(t.n_k);
                    mol.ecp_total_terms++;
                }
                mol.ecp_total_channels++;
            }
        }
        mol.h_ecp_atom_channel_range[mol.natm] = mol.ecp_total_channels;

        // Device 拷贝
        Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_l_max,
                                      (void*)mol.h_ecp_l_max.data(),
                                      sizeof(int) * mol.natm);
        Device_Malloc_And_Copy_Safely(
            (void**)&mol.d_ecp_atom_channel_range,
            (void*)mol.h_ecp_atom_channel_range.data(),
            sizeof(int) * (mol.natm + 1));
        if (mol.ecp_total_channels > 0)
        {
            Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_l,
                                          (void*)mol.h_ecp_l.data(),
                                          sizeof(int) * mol.ecp_total_channels);
            Device_Malloc_And_Copy_Safely(
                (void**)&mol.d_ecp_channel_offsets,
                (void*)mol.h_ecp_channel_offsets.data(),
                sizeof(int) * mol.ecp_total_channels);
            Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_channel_sizes,
                                          (void*)mol.h_ecp_channel_sizes.data(),
                                          sizeof(int) * mol.ecp_total_channels);
        }
        if (mol.ecp_total_terms > 0)
        {
            Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_d,
                                          (void*)mol.h_ecp_d.data(),
                                          sizeof(float) * mol.ecp_total_terms);
            Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_zeta,
                                          (void*)mol.h_ecp_zeta.data(),
                                          sizeof(float) * mol.ecp_total_terms);
            Device_Malloc_And_Copy_Safely((void**)&mol.d_ecp_n,
                                          (void*)mol.h_ecp_n.data(),
                                          sizeof(int) * mol.ecp_total_terms);
        }
    }
}

void QUANTUM_CHEMISTRY::Initial_Integral_Tasks(CONTROLLER* controller)
{
    const int max_l = *std::max_element(mol.h_l_list.begin(),
                                        mol.h_l_list.begin() + mol.nbas);

    Init_ERI_Workspace_Params(this, controller, max_l);
    Build_Shell_Pairs_And_Pair_Types(this, max_l);
    Build_Screening_Combos_And_Task_Buffers(this);
}

void QUANTUM_CHEMISTRY::Initial(CONTROLLER* controller, const int atom_numbers,
                                const VECTOR* crd, const char* module_name)
{
    (void)crd;
    if (module_name == NULL)
    {
        strcpy(this->module_name, "quantum_chemistry");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (is_initialized) return;

    const char* qc_type_file = NULL;
    std::string basis_set_name;
    const bool need_qc = Parsing_Arguments(controller, atom_numbers,
                                           qc_type_file, basis_set_name);
    if (!need_qc) return;

    Initial_Molecule(controller, qc_type_file, basis_set_name);
    orbital_basis_name = basis_set_name;

    if (scf_ws.ri.enabled) Initial_Auxiliary_Basis(controller);

    Initial_Integral_Tasks(controller);

    is_initialized = 1;
    deviceBlasCreate(&blas_handle);
    deviceSolverCreate(&solver_handle);
    Memory_Allocate(controller);

    controller->Step_Print_Initial("QC", "%e");
    if (scf_ws.runtime.unrestricted)
        controller->Step_Print_Initial("QC_S_sq", "%.4f");
}

// 将 d_F 中的 F_guess 对角化，按 aufbau 填充 alpha/beta 轨道构建初始 P
void QUANTUM_CHEMISTRY::Diag_Guess_And_Build_P()
{
    const int nao2 = mol.nao2;

    // 同步到 d_F_double（Diagonalize 优先读 d_F_double）
    if (scf_ws.alpha.d_F_double)
        QC_Float_To_Double(nao2, scf_ws.alpha.d_F, scf_ws.alpha.d_F_double);
    if (scf_ws.runtime.unrestricted)
    {
        deviceMemcpy(scf_ws.beta.d_F, scf_ws.alpha.d_F, sizeof(float) * nao2,
                     deviceMemcpyDeviceToDevice);
        if (scf_ws.beta.d_F_double)
            QC_Float_To_Double(nao2, scf_ws.beta.d_F, scf_ws.beta.d_F_double);
    }

    // 对角化（跳过 level shift）
    double saved_ls = scf_ws.runtime.level_shift;
    scf_ws.runtime.level_shift = 0.0;
    Diagonalize_And_Build_Density();
    scf_ws.runtime.level_shift = saved_ls;

    // P = P_new
    deviceMemcpy(scf_ws.alpha.d_P, scf_ws.alpha.d_P_new, sizeof(float) * nao2,
                 deviceMemcpyDeviceToDevice);
    if (scf_ws.runtime.unrestricted)
    {
        deviceMemcpy(scf_ws.beta.d_P, scf_ws.beta.d_P_new, sizeof(float) * nao2,
                     deviceMemcpyDeviceToDevice);
    }
}

void QUANTUM_CHEMISTRY::Build_Initial_Guess()
{
    if (initial_guess == QC_INITIAL_GUESS::NONE) return;

    const int nao2 = mol.nao2;

    if (initial_guess == QC_INITIAL_GUESS::MINAO)
    {
        QC_Build_Minao_Guess(mol, scf_ws.runtime, scf_ws.alpha.d_P,
                             scf_ws.beta.d_P);
    }
    else if (initial_guess == QC_INITIAL_GUESS::SAP)
    {
        const int nao_c = mol.nao_cart;
        const int nao = mol.nao;

        // 1. 计算 V_SAP（笛卡尔基下）
        float* d_V_SAP = nullptr;
        Device_Malloc_Safely((void**)&d_V_SAP, sizeof(float) * nao_c * nao_c);
        QC_Compute_V_SAP(mol, task_ctx, d_V_SAP);

        // 2. 球谐变换（如需要），结果写入 d_F
        if (mol.is_spherical)
            Cart2Sph_Single_Matrix(d_V_SAP, scf_ws.alpha.d_F);
        else
            deviceMemcpy(scf_ws.alpha.d_F, d_V_SAP, sizeof(float) * nao2,
                         deviceMemcpyDeviceToDevice);
        deviceFree(d_V_SAP);

        // 3. 归一化 + F_guess = T + V_SAP
        QC_Scale_Matrix_By_Norms(nao, scf_ws.ortho.d_norms, scf_ws.alpha.d_F);
        QC_Add_Matrix(nao2, scf_ws.core.d_T, scf_ws.alpha.d_F,
                      scf_ws.alpha.d_F);

        // 4. 对角化 + aufbau 构建 P
        Diag_Guess_And_Build_P();
    }

    if (scf_ws.runtime.unrestricted)
    {
        QC_Add_Matrix((int)mol.nao2, scf_ws.alpha.d_P, scf_ws.beta.d_P,
                      scf_ws.direct.d_Ptot);
    }
}

void QUANTUM_CHEMISTRY::Memory_Allocate(CONTROLLER* controller)
{
    Device_Malloc_Safely((void**)&scf_ws.core.d_S, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.core.d_T, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.core.d_V, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.core.d_H_core,
                         sizeof(float) * mol.nao2);
    if (mol.has_ecp)
    {
        Device_Malloc_Safely((void**)&scf_ws.core.d_V_ECP,
                             sizeof(float) * mol.nao2);
    }
    Device_Malloc_Safely((void**)&scf_ws.core.d_scf_energy, sizeof(double));
    Device_Malloc_Safely((void**)&scf_ws.core.d_nuc_energy_dev, sizeof(double));
    Device_Malloc_Safely((void**)&dft.d_exc_total, sizeof(double));
    deviceMemset(scf_ws.core.d_scf_energy, 0, sizeof(double));
    deviceMemset(scf_ws.core.d_nuc_energy_dev, 0, sizeof(double));
    deviceMemset(dft.d_exc_total, 0, sizeof(double));

    if (mol.is_spherical)
    {
        int nao_c = mol.nao_cart;
        int nao_s = mol.nao_sph;
        Device_Malloc_Safely((void**)&cart2sph.d_S_cart,
                             sizeof(float) * nao_c * nao_c);
        Device_Malloc_Safely((void**)&cart2sph.d_T_cart,
                             sizeof(float) * nao_c * nao_c);
        Device_Malloc_Safely((void**)&cart2sph.d_V_cart,
                             sizeof(float) * nao_c * nao_c);
        Device_Malloc_Safely((void**)&cart2sph.d_cart2sph_1e_tmp,
                             sizeof(float) * (int)nao_c * (int)nao_s);
    }
#ifdef USE_GPU
    // GPU: scratch 池槽数与 bounds kernel 的 launch 线程总数一致，避免
    // O(n_pairs) 膨胀
    int hr_pool_tasks = QC_BOUNDS_POOL_SLOTS;
#else
    int hr_pool_tasks = std::max(1, omp_get_max_threads());
#endif
    Device_Malloc_Safely((void**)&scf_ws.direct.d_hr_pool,
                         (int)hr_pool_tasks *
                             (task_ctx.params.eri_hr_size +
                              2 * task_ctx.params.eri_shell_buf_size) *
                             sizeof(float));
    Device_Malloc_Safely((void**)&task_ctx.buffers.d_shell_pair_bounds,
                         sizeof(float) * task_ctx.topo.n_shell_pairs);
    deviceMemset(task_ctx.buffers.d_shell_pair_bounds, 0,
                 sizeof(float) * task_ctx.topo.n_shell_pairs);
    if (dft.enable_dft)
    {
        dft.max_grid_capacity =
            mol.natm * dft.dft_radial_points * dft.dft_angular_points;
        dft.max_grid_size = 0;
        if (dft.max_grid_capacity <= 0)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    invalid DFT grid capacity: %d (natm=%d, "
                "radial=%d, "
                "angular=%d)\n",
                dft.max_grid_capacity, mol.natm, dft.dft_radial_points,
                dft.dft_angular_points);
        }

        dft.h_grid_coords.assign((int)dft.max_grid_capacity * 3, 0.0f);
        dft.h_grid_weights.assign((int)dft.max_grid_capacity, 0.0f);
        Device_Malloc_And_Copy_Safely((void**)&dft.d_grid_coords,
                                      (void*)dft.h_grid_coords.data(),
                                      sizeof(float) * dft.h_grid_coords.size());
        Device_Malloc_And_Copy_Safely(
            (void**)&dft.d_grid_weights, (void*)dft.h_grid_weights.data(),
            sizeof(float) * dft.h_grid_weights.size());
        Device_Malloc_Safely((void**)&dft.d_Vxc, sizeof(float) * mol.nao2);
        if (scf_ws.runtime.unrestricted)
        {
            Device_Malloc_Safely((void**)&dft.d_Vxc_beta,
                                 sizeof(float) * mol.nao2);
        }

        Device_Malloc_Safely((void**)&dft.d_ao_vals,
                             sizeof(float) * dft.grid_batch_size * mol.nao);
        Device_Malloc_Safely((void**)&dft.d_ao_grad_x,
                             sizeof(float) * dft.grid_batch_size * mol.nao);
        Device_Malloc_Safely((void**)&dft.d_ao_grad_y,
                             sizeof(float) * dft.grid_batch_size * mol.nao);
        Device_Malloc_Safely((void**)&dft.d_ao_grad_z,
                             sizeof(float) * dft.grid_batch_size * mol.nao);
        if (mol.is_spherical)
        {
            const int nao_c = mol.nao_cart;
            Device_Malloc_Safely((void**)&dft.d_ao_vals_cart,
                                 sizeof(float) * dft.grid_batch_size * nao_c);
            Device_Malloc_Safely((void**)&dft.d_ao_grad_x_cart,
                                 sizeof(float) * dft.grid_batch_size * nao_c);
            Device_Malloc_Safely((void**)&dft.d_ao_grad_y_cart,
                                 sizeof(float) * dft.grid_batch_size * nao_c);
            Device_Malloc_Safely((void**)&dft.d_ao_grad_z_cart,
                                 sizeof(float) * dft.grid_batch_size * nao_c);
        }
        Device_Malloc_Safely((void**)&dft.d_rho,
                             sizeof(double) * dft.grid_batch_size);
        Device_Malloc_Safely((void**)&dft.d_sigma,
                             sizeof(double) * dft.grid_batch_size);
        Device_Malloc_Safely((void**)&dft.d_exc,
                             sizeof(double) * dft.grid_batch_size);
        Device_Malloc_Safely((void**)&dft.d_vrho,
                             sizeof(double) * dft.grid_batch_size);
        Device_Malloc_Safely((void**)&dft.d_vsigma,
                             sizeof(double) * dft.grid_batch_size);

        // VXC BLAS 优化缓冲
        const int nao_alloc = mol.nao;
        const int bs = dft.grid_batch_size;
        Device_Malloc_Safely((void**)&dft.d_ao_norm,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_gx_norm,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_gy_norm,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_gz_norm,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_Pao,
                             sizeof(float) * nao_alloc * bs);
        Device_Malloc_Safely((void**)&dft.d_W_full,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_W_sigma,
                             sizeof(float) * bs * nao_alloc);
        Device_Malloc_Safely((void**)&dft.d_grad_rho_x, sizeof(double) * bs);
        Device_Malloc_Safely((void**)&dft.d_grad_rho_y, sizeof(double) * bs);
        Device_Malloc_Safely((void**)&dft.d_grad_rho_z, sizeof(double) * bs);

        // UKS 额外缓冲
        if (scf_ws.runtime.unrestricted)
        {
            Device_Malloc_Safely((void**)&dft.d_Pao_b,
                                 sizeof(float) * nao_alloc * bs);
            Device_Malloc_Safely((void**)&dft.d_rho_a, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_rho_b, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_sigma_aa, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_sigma_ab, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_sigma_bb, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_grb_x, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_grb_y, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_grb_z, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_exc_buf, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_vra, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_vrb, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_vsaa, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_vsab, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_vsbb, sizeof(double) * bs);
            Device_Malloc_Safely((void**)&dft.d_Wb_full,
                                 sizeof(float) * bs * nao_alloc);
            Device_Malloc_Safely((void**)&dft.d_Wb_sigma,
                                 sizeof(float) * bs * nao_alloc);
        }

        // AO screening 半径: r2_screen = -ln(tol) / alpha_min
        {
            const float screen_tol = 1e-15f;
            const float neg_ln_tol = -logf(screen_tol);  // ~34.5
            std::vector<float> h_r2_screen(mol.nbas);
            for (int ish = 0; ish < mol.nbas; ish++)
            {
                float alpha_min = 1e30f;
                for (int ip = 0; ip < mol.h_shell_sizes[ish]; ip++)
                {
                    float a = mol.h_exps[mol.h_shell_offsets[ish] + ip];
                    if (a < alpha_min) alpha_min = a;
                }
                h_r2_screen[ish] = neg_ln_tol / fmaxf(alpha_min, 1e-10f);
            }
            Device_Malloc_Safely((void**)&dft.d_shell_r2_screen,
                                 sizeof(float) * mol.nbas);
            deviceMemcpy(dft.d_shell_r2_screen, h_r2_screen.data(),
                         sizeof(float) * mol.nbas, deviceMemcpyHostToDevice);
        }
    }
    Build_SCF_Workspace();

    // RI 内存分配在 Build_SCF_Workspace 之后，因为需要 n_alpha/n_beta
    if (scf_ws.ri.enabled) RI_Memory_Allocate();

    // 分配梯度工作空间
    Device_Malloc_Safely((void**)&grad_ws.d_grad,
                         sizeof(double) * mol.natm * 3);
    Device_Malloc_Safely((void**)&grad_ws.d_W_density,
                         sizeof(float) * mol.nao2);
    if (scf_ws.runtime.unrestricted)
    {
        Device_Malloc_Safely((void**)&grad_ws.d_W_density_beta,
                             sizeof(float) * mol.nao2);
    }
    // 壳层到原子映射
    {
        std::vector<int> h_shell_atom(mol.nbas);
        for (int ish = 0; ish < mol.nbas; ish++)
            h_shell_atom[ish] = mol.h_bas[ish * 8 + 0];
        Device_Malloc_Safely((void**)&grad_ws.d_shell_atom,
                             sizeof(int) * mol.nbas);
        deviceMemcpy(grad_ws.d_shell_atom, h_shell_atom.data(),
                     sizeof(int) * mol.nbas, deviceMemcpyHostToDevice);
    }
    // 预分配笛卡尔密度缓冲 (球谐 1e 梯度 + ECP 梯度共用)
    if (mol.is_spherical || mol.has_ecp)
    {
        const int nc2 = mol.nao_cart * mol.nao_cart;
        Device_Malloc_Safely((void**)&grad_ws.d_P_cart, sizeof(float) * nc2);
    }
    if (mol.is_spherical)
    {
        const int nc2 = mol.nao_cart * mol.nao_cart;
        Device_Malloc_Safely((void**)&grad_ws.d_W_cart, sizeof(float) * nc2);
        Device_Malloc_Safely((void**)&grad_ws.d_norms_ones,
                             sizeof(float) * mol.nao_cart);
        std::vector<float> h_ones(mol.nao_cart, 1.0f);
        deviceMemcpy(grad_ws.d_norms_ones, h_ones.data(),
                     sizeof(float) * mol.nao_cart, deviceMemcpyHostToDevice);
    }
    {
        int max_l_cart = 0;
        for (int sh = 0; sh < mol.nbas; sh++)
            if (mol.h_l_list[sh] > max_l_cart) max_l_cart = mol.h_l_list[sh];
        const int max_dim_cart = (max_l_cart + 1) * (max_l_cart + 2) / 2;
        grad_ws.grad_gamma_buf_size =
            max_dim_cart * max_dim_cart * max_dim_cart * max_dim_cart;
        // 限制 gamma pool 占用 (qzvp: 15^4 * 8B/slot = 405 KB; 4096 slots = 1.6
        // GB) 高 L 时减少 slots, 启动时配套缩减 blocks (kernel worker stride
        // 自适应)
        const size_t bytes_per_slot =
            (size_t)(2 * grad_ws.grad_gamma_buf_size) * sizeof(float);
        const size_t pool_budget_bytes = (size_t)512 * 1024 * 1024;  // 512 MB
        int slots = QC_GRAD_GAMMA_POOL_SLOTS;
        const int budget_slots = (int)(pool_budget_bytes / bytes_per_slot);
        const int min_slots = QC_GRAD_ERI_THREADS;  // 至少 1 block
        if (slots > budget_slots) slots = budget_slots;
        if (slots < min_slots) slots = min_slots;
        const size_t gamma_pool_elems =
            (size_t)slots * (size_t)(2 * grad_ws.grad_gamma_buf_size);
        Device_Malloc_Safely((void**)&grad_ws.d_grad_gamma_pool,
                             sizeof(float) * gamma_pool_elems);
        grad_ws.grad_gamma_pool_slots = slots;
    }
    // 辅助基壳层到原子映射 (RI 梯度用)
    if (scf_ws.ri.enabled)
    {
        const auto& ri = scf_ws.ri;
        std::vector<int> h_shell_atom_aux(ri.naux_bas);
        for (int ish = 0; ish < ri.naux_bas; ish++)
            h_shell_atom_aux[ish] = ri.h_aux_bas[ish * 8 + 0];
        Device_Malloc_Safely((void**)&grad_ws.d_shell_atom_aux,
                             sizeof(int) * ri.naux_bas);
        deviceMemcpy(grad_ws.d_shell_atom_aux, h_shell_atom_aux.data(),
                     sizeof(int) * ri.naux_bas, deviceMemcpyHostToDevice);
    }
    // GPU ERI 梯度持久化缓冲 (combo 前缀 + gradient 多副本累加)
    {
        const int cp_size = task_ctx.topo.n_combos + 1;
        Device_Malloc_Safely((void**)&grad_ws.d_combo_prefix_grad,
                             sizeof(int) * cp_size);
        const size_t copies_elems =
            (size_t)QC_GRAD_N_COPIES * (size_t)(mol.natm * 3);
        Device_Malloc_Safely((void**)&grad_ws.d_grad_copies,
                             sizeof(double) * copies_elems);
    }
}

void QUANTUM_CHEMISTRY::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    if (scf_ws.core.d_scf_energy)
    {
        double h_energy = 0.0;
        deviceMemcpy(&h_energy, scf_ws.core.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        scf_energy = (float)h_energy;
    }
    controller->Step_Print("QC", scf_energy * CONSTANT_HARTREE_TO_KCAL_MOL);

    if (scf_ws.runtime.unrestricted)
    {
        Compute_Spin_Square();
        controller->Step_Print("QC_S_sq", scf_ws.runtime.spin_square);
    }
}
