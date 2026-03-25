#include <fstream>
#include <sstream>

#include "basis/basis.h"
#include "quantum_chemistry.h"

static inline bool Equals_Ignore_Case(const std::string& lhs, const char* rhs)
{
    return is_str_equal(lhs.c_str(), rhs, 0);
}

static void Throw_QC_Initial_Error(CONTROLLER* controller, int error_number,
                                   const char* format, ...)
{
    char error_reason[CHAR_LENGTH_MAX];
    va_list args;
    va_start(args, format);
    vsnprintf(error_reason, sizeof(error_reason), format, args);
    va_end(args);
    controller->Throw_SPONGE_Error(error_number, "QUANTUM_CHEMISTRY::Initial",
                                   error_reason);
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
        Throw_QC_Initial_Error(
            controller, spongeErrorValueErrorCommand,
            "Reason:\n    qc_model_chemistry format error: expected "
            "\"METHOD/<basis>\", got \"%s\"\n",
            model_chemistry.c_str());
    }
    std::string method_name =
        string_strip(model_chemistry.substr(0, slash_pos));
    basis_set_name = string_strip(model_chemistry.substr(slash_pos + 1));

    if (Equals_Ignore_Case(method_name, "HF"))
    {
        method = QC_METHOD::HF;
        dft.exx_fraction = 1.0f;
        dft.enable_dft = 0;
    }
    else if (Equals_Ignore_Case(method_name, "LDA"))
    {
        method = QC_METHOD::LDA;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (Equals_Ignore_Case(method_name, "PBE"))
    {
        method = QC_METHOD::PBE;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (Equals_Ignore_Case(method_name, "BLYP"))
    {
        method = QC_METHOD::BLYP;
        dft.exx_fraction = 0.0f;
        dft.enable_dft = 1;
    }
    else if (Equals_Ignore_Case(method_name, "PBE0"))
    {
        method = QC_METHOD::PBE0;
        dft.exx_fraction = 0.25f;
        dft.enable_dft = 1;
    }
    else if (Equals_Ignore_Case(method_name, "B3LYP"))
    {
        method = QC_METHOD::B3LYP;
        dft.exx_fraction = 0.20f;
        dft.enable_dft = 1;
    }
    else
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorValueErrorCommand,
            "Reason:\n    qc_model_chemistry \"%s\" not supported. Supported "
            "methods: HF, LDA, PBE, BLYP, PBE0, B3LYP.\n",
            model_chemistry.c_str());
    }

    task_ctx.eri_prim_screen_tol = 1e-12f;
    if (controller->Command_Exist("qc_eri_prim_screen_tol"))
    {
        controller->Check_Float("qc_eri_prim_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.eri_prim_screen_tol =
            atof(controller->Command("qc_eri_prim_screen_tol"));
        if (task_ctx.eri_prim_screen_tol < 0.0f)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_eri_prim_screen_tol must be >= 0, got %g\n",
                (double)task_ctx.eri_prim_screen_tol);
        }
    }

    task_ctx.direct_eri_prim_screen_tol = 1e-10f;
    if (controller->Command_Exist("qc_direct_eri_prim_screen_tol"))
    {
        controller->Check_Float("qc_direct_eri_prim_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.direct_eri_prim_screen_tol =
            atof(controller->Command("qc_direct_eri_prim_screen_tol"));
        if (task_ctx.direct_eri_prim_screen_tol < 0.0f)
        {
            Throw_QC_Initial_Error(controller, spongeErrorValueErrorCommand,
                                   "Reason:\n    qc_direct_eri_prim_screen_tol "
                                   "must be >= 0, got %g\n",
                                   (double)task_ctx.direct_eri_prim_screen_tol);
        }
    }

    task_ctx.eri_shell_screen_tol = 1e-10f;
    if (controller->Command_Exist("qc_eri_shell_screen_tol"))
    {
        controller->Check_Float("qc_eri_shell_screen_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        task_ctx.eri_shell_screen_tol =
            atof(controller->Command("qc_eri_shell_screen_tol"));
        if (task_ctx.eri_shell_screen_tol < 0.0f)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_eri_shell_screen_tol must be >= 0, got %g\n",
                (double)task_ctx.eri_shell_screen_tol);
        }
    }

    scf_ws.unrestricted = false;
    if (controller->Command_Exist("qc_restricted"))
    {
        controller->Check_Int("qc_restricted", "QUANTUM_CHEMISTRY::Initial");
        const int qc_restricted = atoi(controller->Command("qc_restricted"));
        if (qc_restricted != 0 && qc_restricted != 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_restricted must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_restricted"));
        }
        scf_ws.unrestricted = (qc_restricted == 0);
    }

    scf_ws.max_scf_iter = 100;
    if (controller->Command_Exist("qc_scf_max_iter"))
    {
        controller->Check_Int("qc_scf_max_iter", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.max_scf_iter = atoi(controller->Command("qc_scf_max_iter"));
        if (scf_ws.max_scf_iter < 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_scf_max_iter must be >= 1, got \"%s\"\n",
                controller->Command("qc_scf_max_iter"));
        }
    }

    scf_ws.use_diis = true;
    if (controller->Command_Exist("qc_diis"))
    {
        controller->Check_Int("qc_diis", "QUANTUM_CHEMISTRY::Initial");
        const int qc_diis = atoi(controller->Command("qc_diis"));
        if (qc_diis != 0 && qc_diis != 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_diis must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_diis"));
        }
        scf_ws.use_diis = (qc_diis != 0);
    }

    scf_ws.diis_start_iter = 2;
    if (controller->Command_Exist("qc_diis_start"))
    {
        controller->Check_Int("qc_diis_start", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.diis_start_iter = atoi(controller->Command("qc_diis_start"));
        if (scf_ws.diis_start_iter < 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_diis_start must be >= 1, got \"%s\"\n",
                controller->Command("qc_diis_start"));
        }
    }

    scf_ws.diis_space = 6;
    if (controller->Command_Exist("qc_diis_space"))
    {
        controller->Check_Int("qc_diis_space", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.diis_space = atoi(controller->Command("qc_diis_space"));
        if (scf_ws.diis_space < 2)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_diis_space must be >= 2, got \"%s\"\n",
                controller->Command("qc_diis_space"));
        }
    }

    scf_ws.density_mixing = 0.20f;
    if (controller->Command_Exist("qc_diis_damp"))
    {
        controller->Check_Float("qc_diis_damp", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.density_mixing = atof(controller->Command("qc_diis_damp"));
        if (scf_ws.density_mixing < 0.0f || scf_ws.density_mixing > 1.0f)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_diis_damp must be in [0, 1], got \"%s\"\n",
                controller->Command("qc_diis_damp"));
        }
    }

    scf_ws.diis_reg = 1e-10;
    if (controller->Command_Exist("qc_diis_reg"))
    {
        controller->Check_Float("qc_diis_reg", "QUANTUM_CHEMISTRY::Initial");
        scf_ws.diis_reg = atof(controller->Command("qc_diis_reg"));
        if (scf_ws.diis_reg < 0.0)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_diis_reg must be >= 0, got \"%s\"\n",
                controller->Command("qc_diis_reg"));
        }
    }

    scf_ws.energy_tol = 1e-6;
    if (controller->Command_Exist("qc_scf_energy_tol"))
    {
        controller->Check_Float("qc_scf_energy_tol",
                                "QUANTUM_CHEMISTRY::Initial");
        scf_ws.energy_tol = atof(controller->Command("qc_scf_energy_tol"));
        if (scf_ws.energy_tol <= 0.0)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_scf_energy_tol must be > 0, got \"%s\"\n",
                controller->Command("qc_scf_energy_tol"));
        }
    }

    scf_ws.print_iter = false;
    if (controller->Command_Exist("qc_scf_print_iter"))
    {
        controller->Check_Int("qc_scf_print_iter",
                              "QUANTUM_CHEMISTRY::Initial");
        const int qc_scf_print_iter =
            atoi(controller->Command("qc_scf_print_iter"));
        if (qc_scf_print_iter != 0 && qc_scf_print_iter != 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_scf_print_iter must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_scf_print_iter"));
        }
        scf_ws.print_iter = (qc_scf_print_iter != 0);
    }

    if (controller->Command_Exist("qc_level_shift"))
    {
        scf_ws.level_shift = atof(controller->Command("qc_level_shift"));
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

    this->atom_numbers = atom_numbers;
    return true;
}

void QUANTUM_CHEMISTRY::Initial_Molecule(CONTROLLER* controller,
                                         const char* qc_type_file,
                                         const std::string& basis_set_name)
{
    static QC_BASIS_SET* all_bases[] = {
        QC_BASIS_STO_3G_PTR,        QC_BASIS_3_21G_PTR,
        QC_BASIS_631G_PTR,          QC_BASIS_631G_STAR_PTR,
        QC_BASIS_631G_STARSTAR_PTR, QC_BASIS_6311G_PTR,
        QC_BASIS_6311G_STAR_PTR,    QC_BASIS_6311G_STARSTAR_PTR,
        QC_BASIS_DEF2_SVP_PTR,      QC_BASIS_DEF2_TZVP_PTR,
        QC_BASIS_DEF2_TZVPP_PTR,    QC_BASIS_DEF2_QZVP_PTR,
        QC_BASIS_CC_PVDZ_PTR,       QC_BASIS_CC_PVTZ_PTR,
    };

    QC_BASIS_SET* basis = nullptr;
    for (auto* b : all_bases)
    {
        if (Equals_Ignore_Case(basis_set_name, b->name))
        {
            basis = b;
            break;
        }
    }
    if (!basis)
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorValueErrorCommand,
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
            Throw_QC_Initial_Error(controller, spongeErrorBadFileFormat,
                                   "Reason:\n    Cannot open %s\n",
                                   qc_type_file);
        }
        std::string line;
        std::getline(ifs, line);
        {
            std::istringstream iss(line);
            if (!(iss >> mol.natm >> mol.charge >> mol.multiplicity))
            {
                Throw_QC_Initial_Error(
                    controller, spongeErrorBadFileFormat,
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
                Throw_QC_Initial_Error(
                    controller, spongeErrorBadFileFormat,
                    "Reason:\n    Failed to read atom line %d of %s\n", i,
                    qc_type_file);
            }
            atom_local.push_back(idx);
            atom_symbols.push_back(sym);
        }
    }

    mol.nelectron = -mol.charge;
    mol.h_Z.resize(mol.natm);
    for (int i = 0; i < mol.natm; ++i)
    {
        auto it_sym = QC_Z_FROM_SYMBOL.find(atom_symbols[i]);
        if (it_sym == QC_Z_FROM_SYMBOL.end())
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorBadFileFormat,
                "Reason:\n    Unknown element symbol %s at atom line %d in "
                "%s\n",
                atom_symbols[i].c_str(), i, qc_type_file);
        }
        int Z = it_sym->second;
        mol.h_Z[i] = Z;
        int md_idx = atom_local[i];
        if (md_idx < 0 || md_idx >= this->atom_numbers)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorOverflow,
                "Reason:\n    MD index %d out of bounds [0, %d)\n", md_idx,
                this->atom_numbers);
        }
        mol.nelectron += Z;
    }

    Device_Malloc_And_Copy_Safely((void**)&mol.d_Z, (void*)mol.h_Z.data(),
                                  sizeof(int) * (int)mol.natm);

    const int spin_e = mol.multiplicity - 1;
    if (spin_e < 0)
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorBadFileFormat,
            "Reason:\n    multiplicity must be >= 1, got %d\n",
            mol.multiplicity);
    }
    if (((mol.nelectron + spin_e) & 1) != 0)
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorBadFileFormat,
            "Reason:\n    Inconsistent electron number/multiplicity: N=%d, "
            "multiplicity=%d\n",
            mol.nelectron, mol.multiplicity);
    }
    if (!scf_ws.unrestricted)
    {
        if (spin_e != 0)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_restricted=1 requires closed-shell "
                "multiplicity=1, got multiplicity=%d\n",
                mol.multiplicity);
        }
        if ((mol.nelectron & 1) != 0)
        {
            Throw_QC_Initial_Error(controller, spongeErrorValueErrorCommand,
                                   "Reason:\n    qc_restricted=1 requires even "
                                   "electron number, got "
                                   "N=%d\n",
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
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
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

            mol.h_ao_loc.push_back(mol.nao_cart);
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
    mol.h_ao_loc.push_back(mol.nao_cart);
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
    Device_Malloc_And_Copy_Safely((void**)&mol.d_ao_loc,
                                  (void*)mol.h_ao_loc.data(),
                                  sizeof(int) * mol.h_ao_loc.size());

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
}

void QUANTUM_CHEMISTRY::Initial_Integral_Tasks(CONTROLLER* controller)
{
    int max_l = 0;
    for (int k = 0; k < mol.h_l_list.size(); k++)
    {
        if (mol.h_l_list[k] > max_l) max_l = mol.h_l_list[k];
    }
    const int max_total_l = 4 * max_l;
    task_ctx.eri_hr_base = max_total_l + 1;
    if (task_ctx.eri_hr_base > HR_BASE_MAX)
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorOverflow,
            "Reason:\n    basis angular momentum too high (max l=%d, required "
            "hr_base=%d, supported <=%d)\n",
            max_l, task_ctx.eri_hr_base, HR_BASE_MAX);
    }
    task_ctx.eri_hr_size = task_ctx.eri_hr_base * task_ctx.eri_hr_base *
                           task_ctx.eri_hr_base * task_ctx.eri_hr_base;
    {
        const int max_cart = (max_l + 1) * (max_l + 2) / 2;
        task_ctx.eri_shell_buf_size = max_cart * max_cart * max_cart * max_cart;
        task_ctx.eri_shell_buf_size =
            std::max(1, std::min(MAX_SHELL_ERI, task_ctx.eri_shell_buf_size));
    }

    task_ctx.h_shell_pairs.clear();
    for (int i = 0; i < mol.nbas; i++)
        for (int j = 0; j <= i; j++) task_ctx.h_shell_pairs.push_back({i, j});
    task_ctx.n_shell_pairs = task_ctx.h_shell_pairs.size();

    // Build pair type index for on-the-fly dispatch
    {
        const int max_l = *std::max_element(mol.h_l_list.begin(),
                                            mol.h_l_list.begin() + mol.nbas);
        const int stride = max_l + 1;
        const int n_types = stride * stride;

        // Count pairs per type
        std::vector<std::vector<int>> type_lists(n_types);
        for (int pid = 0; pid < task_ctx.n_shell_pairs; pid++)
        {
            const auto& p = task_ctx.h_shell_pairs[pid];
            int tid = mol.h_l_list[p.x] * stride + mol.h_l_list[p.y];
            type_lists[tid].push_back(pid);
        }

        // Build sorted pair ids and type boundaries
        task_ctx.h_sorted_pair_ids.clear();
        task_ctx.h_sorted_pair_ids.reserve(task_ctx.n_shell_pairs);
        task_ctx.n_pair_types = 0;
        for (int tid = 0; tid < n_types; tid++)
        {
            if (type_lists[tid].empty()) continue;
            int slot = task_ctx.n_pair_types++;
            task_ctx.pair_type_offset[slot] =
                (int)task_ctx.h_sorted_pair_ids.size();
            task_ctx.pair_type_count[slot] = (int)type_lists[tid].size();
            task_ctx.pair_type_l0[slot] = tid / stride;
            task_ctx.pair_type_l1[slot] = tid % stride;
            for (int pid : type_lists[tid])
                task_ctx.h_sorted_pair_ids.push_back(pid);
        }

        Device_Malloc_And_Copy_Safely(
            (void**)&task_ctx.d_sorted_pair_ids,
            (void*)task_ctx.h_sorted_pair_ids.data(),
            sizeof(int) * task_ctx.h_sorted_pair_ids.size());
    }

    // Build screening combos: one per pair-type combination (A >= B)
    {
        const int npt = task_ctx.n_pair_types;
        task_ctx.n_combos = 0;
        for (int tA = 0; tA < npt; tA++)
        {
            for (int tB = 0; tB <= tA; tB++)
            {
                const int nA = task_ctx.pair_type_count[tA];
                const int nB = task_ctx.pair_type_count[tB];
                const bool same = (tA == tB);
                const int nq = same ? nA * (nA + 1) / 2 : nA * nB;
                if (nq == 0) continue;

                auto& c = task_ctx.h_combos[task_ctx.n_combos];
                c.pair_base_A = task_ctx.pair_type_offset[tA];
                c.n_A = nA;
                c.pair_base_B = task_ctx.pair_type_offset[tB];
                c.n_B = nB;
                c.n_quartets = nq;
                c.output_offset = 0;  // set below after counting combos
                c.same_type = same ? 1 : 0;
                c.l0 = task_ctx.pair_type_l0[tA];
                c.l1 = task_ctx.pair_type_l1[tA];
                c.l2 = task_ctx.pair_type_l0[tB];
                c.l3 = task_ctx.pair_type_l1[tB];
                task_ctx.n_combos++;
            }
        }
        // Prefix sum
        task_ctx.combo_prefix[0] = 0;
        for (int i = 0; i < task_ctx.n_combos; i++)
            task_ctx.combo_prefix[i + 1] =
                task_ctx.combo_prefix[i] + task_ctx.h_combos[i].n_quartets;
        task_ctx.total_quartets = task_ctx.combo_prefix[task_ctx.n_combos];

        // Allocate device buffers
        Device_Malloc_And_Copy_Safely(
            (void**)&task_ctx.d_combos, (void*)task_ctx.h_combos,
            sizeof(QC_INTEGRAL_TASKS::ScreenCombo) * task_ctx.n_combos);
        // Output offsets and buffer allocated after task list (see below)
    }

    for (int i = 0; i < mol.nbas; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            int pair_ij = i * (i + 1) / 2 + j;
            for (int k = 0; k < mol.nbas; k++)
            {
                for (int l = 0; l <= k; l++)
                {
                    int pair_kl = k * (k + 1) / 2 + l;
                    if (pair_ij < pair_kl) continue;
                    task_ctx.h_eri_tasks.push_back({i, j, k, l});
                }
            }
        }
    }
    task_ctx.n_eri_tasks = task_ctx.h_eri_tasks.size();

    // Allocate screening output buffer and assign per-combo offsets.
    // Buffer sized to n_eri_tasks (post-screening active tasks <= this).
    // Combos share buffer proportionally to their n_quartets.
    {
        task_ctx.screened_buf_capacity = task_ctx.n_eri_tasks;
        const long long total_q =
            task_ctx.total_quartets > 0 ? task_ctx.total_quartets : 1;
        int output_off = 0;
        for (int i = 0; i < task_ctx.n_combos; i++)
        {
            task_ctx.h_combos[i].output_offset = output_off;
            // Proportional share: this combo gets (n_quartets/total_quartets) *
            // capacity
            const int share = (int)((long long)task_ctx.h_combos[i].n_quartets *
                                    task_ctx.screened_buf_capacity / total_q);
            output_off += std::max(share, 1);
        }
        // Clamp to capacity (rounding might exceed slightly)
        if (output_off > task_ctx.screened_buf_capacity)
            task_ctx.screened_buf_capacity = output_off;
        Device_Malloc_Safely(
            (void**)&task_ctx.d_screened_tasks,
            sizeof(QC_ERI_TASK) * task_ctx.screened_buf_capacity);
        Device_Malloc_Safely((void**)&task_ctx.d_screen_counts,
                             sizeof(int) * QC_INTEGRAL_TASKS::MAX_COMBOS);
        // Re-upload combos with updated offsets
        if (task_ctx.d_combos != NULL)
            deviceMemcpy(
                task_ctx.d_combos, task_ctx.h_combos,
                sizeof(QC_INTEGRAL_TASKS::ScreenCombo) * task_ctx.n_combos,
                deviceMemcpyHostToDevice);
    }

    // Build screening combos from pair types (after pair type index is ready)
    // Deferred to after pair type construction below.

    // Pre-bin ERI tasks by shell type and sort in-place.
    // Bucket layout: 4s(1) | 3s1p×4 | 2s2p×6 | 1s3p×4 | 4p(1) | generic(1)
    {
        auto get_bucket = [&](const QC_ERI_TASK& t) -> int
        {
            const int la = mol.h_l_list[t.x], lb = mol.h_l_list[t.y];
            const int lc = mol.h_l_list[t.z], ld = mol.h_l_list[t.w];
            const int l_sum = la + lb + lc + ld;
            const int l_max = std::max({la, lb, lc, ld});
            if (l_sum == 0) return 0;  // 4s
            if (l_sum == 1)
            {
                // 3s1p: bucket 1-4 by p position
                if (la == 1) return 1;
                if (lb == 1) return 2;
                if (lc == 1) return 3;
                return 4;
            }
            if (l_sum == 2 && l_max <= 1)
            {
                // 2s2p: bucket 5-10 by (p0,p1) pair
                // positions of the two p shells
                int pp[2], pi = 0;
                if (la == 1) pp[pi++] = 0;
                if (lb == 1) pp[pi++] = 1;
                if (lc == 1) pp[pi++] = 2;
                if (ld == 1) pp[pi++] = 3;
                static const int pair_idx[4][4] = {
                    {-1, 0, 1, 2}, {0, -1, 3, 4}, {1, 3, -1, 5}, {2, 4, 5, -1}};
                return 5 + pair_idx[pp[0]][pp[1]];
            }
            if (l_sum == 3 && l_max <= 1)
            {
                // 1s3p: bucket 11-14 by s position
                if (la == 0) return 11;
                if (lb == 0) return 12;
                if (lc == 0) return 13;
                return 14;
            }
            if (l_sum == 4 && l_max <= 1) return 15;  // 4p
            return 16;                                // generic
        };

        // Count tasks per bucket
        for (int b = 0; b < QC_INTEGRAL_TASKS::N_BUCKETS; b++)
            task_ctx.bucket_count[b] = 0;
        for (const auto& t : task_ctx.h_eri_tasks)
            task_ctx.bucket_count[get_bucket(t)]++;

        // Compute offsets (prefix sum)
        task_ctx.bucket_offset[0] = 0;
        for (int b = 1; b < QC_INTEGRAL_TASKS::N_BUCKETS; b++)
            task_ctx.bucket_offset[b] =
                task_ctx.bucket_offset[b - 1] + task_ctx.bucket_count[b - 1];

        // Sort by bucket using a temporary array
        std::vector<QC_ERI_TASK> sorted(task_ctx.n_eri_tasks);
        std::vector<int> pos(QC_INTEGRAL_TASKS::N_BUCKETS);
        for (int b = 0; b < QC_INTEGRAL_TASKS::N_BUCKETS; b++)
            pos[b] = task_ctx.bucket_offset[b];
        for (const auto& t : task_ctx.h_eri_tasks)
            sorted[pos[get_bucket(t)]++] = t;
        task_ctx.h_eri_tasks = std::move(sorted);

    }

    for (int i = 0; i < mol.nbas; i++)
        for (int j = 0; j < mol.nbas; j++)
            task_ctx.h_1e_tasks.push_back({i, j});
    task_ctx.n_1e_tasks = task_ctx.h_1e_tasks.size();

    // d_eri_tasks now holds the pre-sorted task list permanently
    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.d_eri_tasks, (void*)task_ctx.h_eri_tasks.data(),
        sizeof(QC_ERI_TASK) * task_ctx.h_eri_tasks.size());
    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.d_1e_tasks, (void*)task_ctx.h_1e_tasks.data(),
        sizeof(QC_ONE_E_TASK) * task_ctx.h_1e_tasks.size());
    Device_Malloc_And_Copy_Safely(
        (void**)&task_ctx.d_shell_pairs, (void*)task_ctx.h_shell_pairs.data(),
        sizeof(QC_ONE_E_TASK) * task_ctx.h_shell_pairs.size());
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
    Initial_Integral_Tasks(controller);

    is_initialized = 1;
    deviceBlasCreate(&blas_handle);
    deviceSolverCreate(&solver_handle);
    Memory_Allocate(controller);
    controller->Step_Print_Initial("QC", "%e");
}

void QUANTUM_CHEMISTRY::Memory_Allocate(CONTROLLER* controller)
{
    Device_Malloc_Safely((void**)&scf_ws.d_S, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.d_T, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.d_V, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.d_H_core, sizeof(float) * mol.nao2);
    Device_Malloc_Safely((void**)&scf_ws.d_scf_energy, sizeof(double));
    Device_Malloc_Safely((void**)&scf_ws.d_nuc_energy_dev, sizeof(double));
    Device_Malloc_Safely((void**)&dft.d_exc_total, sizeof(double));
    deviceMemset(scf_ws.d_scf_energy, 0, sizeof(double));
    deviceMemset(scf_ws.d_nuc_energy_dev, 0, sizeof(double));
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
    int hr_pool_tasks = ERI_BATCH_SIZE;
#ifndef USE_GPU
    hr_pool_tasks = std::max(1, omp_get_max_threads());
#endif
    Device_Malloc_Safely(
        (void**)&d_hr_pool,
        (int)hr_pool_tasks *
            (task_ctx.eri_hr_size + 2 * task_ctx.eri_shell_buf_size) *
            sizeof(float));
    Device_Malloc_Safely((void**)&task_ctx.d_shell_pair_bounds,
                         sizeof(float) * task_ctx.n_shell_pairs);
    deviceMemset(task_ctx.d_shell_pair_bounds, 0,
                 sizeof(float) * task_ctx.n_shell_pairs);
    if (dft.enable_dft)
    {
        dft.max_grid_capacity =
            mol.natm * dft.dft_radial_points * dft.dft_angular_points;
        dft.max_grid_size = 0;
        if (dft.max_grid_capacity <= 0)
        {
            Throw_QC_Initial_Error(controller, spongeErrorValueErrorCommand,
                                   "Reason:\n    invalid DFT grid capacity: %d "
                                   "(natm=%d, radial=%d, "
                                   "angular=%d)\n",
                                   dft.max_grid_capacity, mol.natm,
                                   dft.dft_radial_points,
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
        if (scf_ws.unrestricted)
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
    }
    Build_SCF_Workspace();
}

void QUANTUM_CHEMISTRY::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    if (scf_ws.d_scf_energy)
    {
        double h_energy = 0.0;
        deviceMemcpy(&h_energy, scf_ws.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        scf_energy = (float)h_energy;
    }
    controller->Step_Print("QC", scf_energy * CONSTANT_HARTREE_TO_KCAL_MOL);
}
