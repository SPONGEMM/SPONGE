#include "quantum_chemistry.h"

#include "basis/basis_3_21g.hpp"
#include "basis/basis_631g.hpp"
#include "basis/basis_6_311g.hpp"
#include "basis/basis_6_311gstar.hpp"
#include "basis/basis_6_311gstarstar.hpp"
#include "basis/basis_6_31gstar.hpp"
#include "basis/basis_6_31gstarstar.hpp"
#include "basis/basis_cc_pvdz.hpp"
#include "basis/basis_cc_pvtz.hpp"
#include "basis/basis_def2_qzvp.hpp"
#include "basis/basis_def2_svp.hpp"
#include "basis/basis_def2_tzvp.hpp"
#include "basis/basis_def2_tzvpp.hpp"
#include "basis/basis_sto_3g.hpp"

// 计算单电子积分的批大小
#define ONE_E_BATCH_SIZE 4096

#define PI_25 17.4934183276248628469f
#define HR_BASE_MAX 17
#define HR_SIZE_MAX 83521
#define ONEE_MD_BASE 9
#define ONEE_MD_IDX(t, u, v, n) \
    ((((t) * ONEE_MD_BASE + (u)) * ONEE_MD_BASE + (v)) * ONEE_MD_BASE + (n))
#define ERI_BATCH_SIZE 128
#define MAX_CART_SHELL 15
#define MAX_SHELL_ERI \
    (MAX_CART_SHELL * MAX_CART_SHELL * MAX_CART_SHELL * MAX_CART_SHELL)

#include "dft/dft.hpp"
#include "integrals/eri.hpp"
#include "integrals/one_e.hpp"
#include "scf/matrix.hpp"

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
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_direct_eri_prim_screen_tol must be >= 0, got %g\n",
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

    scf_ws.overlap_eig_floor = 1e-10f;
    if (controller->Command_Exist("qc_overlap_eig_floor"))
    {
        controller->Check_Float("qc_overlap_eig_floor",
                                "QUANTUM_CHEMISTRY::Initial");
        scf_ws.overlap_eig_floor =
            atof(controller->Command("qc_overlap_eig_floor"));
        if (scf_ws.overlap_eig_floor <= 0.0f)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_overlap_eig_floor must be > 0, got \"%s\"\n",
                controller->Command("qc_overlap_eig_floor"));
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

    scf_ws.profile_build_fock = false;
    if (controller->Command_Exist("qc_scf_profile_build_fock"))
    {
        controller->Check_Int("qc_scf_profile_build_fock",
                              "QUANTUM_CHEMISTRY::Initial");
        const int qc_scf_profile_build_fock =
            atoi(controller->Command("qc_scf_profile_build_fock"));
        if (qc_scf_profile_build_fock != 0 &&
            qc_scf_profile_build_fock != 1)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorValueErrorCommand,
                "Reason:\n    qc_scf_profile_build_fock must be 0 or 1, got \"%s\"\n",
                controller->Command("qc_scf_profile_build_fock"));
        }
        scf_ws.profile_build_fock = (qc_scf_profile_build_fock != 0);
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
    using BasisMap = std::map<std::string, std::vector<ShellData>>;
    struct BasisSpec
    {
        BasisMap* basis;
        void (*init_fn)();
        bool spherical;
    };

    BasisSpec spec = {nullptr, nullptr, false};

    if (Equals_Ignore_Case(basis_set_name, "6-31g"))
        spec = {&BASIS_631G, Initialize_Basis_631G, false};
    else if (Equals_Ignore_Case(basis_set_name, "def2-svp"))
        spec = {&BASIS_DEF2_SVP, Initialize_Basis_Def2_SVP, true};
    else if (Equals_Ignore_Case(basis_set_name, "sto-3g"))
        spec = {&BASIS_STO_3G, Initialize_Basis_Sto3g, false};
    else if (Equals_Ignore_Case(basis_set_name, "3-21g"))
        spec = {&BASIS_3_21G, Initialize_Basis_321g, false};
    else if (Equals_Ignore_Case(basis_set_name, "6-311g"))
        spec = {&BASIS_6_311G, Initialize_Basis_6311g, false};
    else if (Equals_Ignore_Case(basis_set_name, "6-31g*"))
        spec = {&BASIS_6_31GSTAR, Initialize_Basis_631gstar, true};
    else if (Equals_Ignore_Case(basis_set_name, "6-31g**"))
        spec = {&BASIS_6_31GSTARSTAR, Initialize_Basis_631gstarstar, true};
    else if (Equals_Ignore_Case(basis_set_name, "6-311g*"))
        spec = {&BASIS_6_311GSTAR, Initialize_Basis_6311gstar, true};
    else if (Equals_Ignore_Case(basis_set_name, "6-311g**"))
        spec = {&BASIS_6_311GSTARSTAR, Initialize_Basis_6311gstarstar, true};
    else if (Equals_Ignore_Case(basis_set_name, "def2-tzvp"))
        spec = {&BASIS_DEF2_TZVP, Initialize_Basis_Def2Tzvp, true};
    else if (Equals_Ignore_Case(basis_set_name, "def2-tzvpp"))
        spec = {&BASIS_DEF2_TZVPP, Initialize_Basis_Def2Tzvpp, true};
    else if (Equals_Ignore_Case(basis_set_name, "def2-qzvp"))
        spec = {&BASIS_DEF2_QZVP, Initialize_Basis_Def2Qzvp, true};
    else if (Equals_Ignore_Case(basis_set_name, "cc-pvdz"))
        spec = {&BASIS_CC_PVDZ, Initialize_Basis_CcPvdz, true};
    else if (Equals_Ignore_Case(basis_set_name, "cc-pvtz"))
        spec = {&BASIS_CC_PVTZ, Initialize_Basis_CcPvtz, true};
    else
    {
        Throw_QC_Initial_Error(
            controller, spongeErrorValueErrorCommand,
            "Reason:\n    Basis set \"%s\" is not supported.\n",
            basis_set_name.c_str());
    }
    spec.init_fn();
    mol.is_spherical = spec.spherical ? 1 : 0;

    std::vector<std::string> atom_symbols;
    FILE* fp = NULL;
    Open_File_Safely(&fp, qc_type_file, "r");
    if (fscanf(fp, "%d %d %d", &mol.natm, &mol.charge, &mol.multiplicity) != 3)
    {
        Throw_QC_Initial_Error(controller, spongeErrorBadFileFormat,
                               "Reason:\n    Failed to read first line of %s\n",
                               qc_type_file);
    }
    atom_local.reserve(mol.natm);
    for (int i = 0; i < mol.natm; i++)
    {
        int idx;
        char sym[16];
        if (fscanf(fp, "%d %s", &idx, sym) != 2)
        {
            Throw_QC_Initial_Error(
                controller, spongeErrorBadFileFormat,
                "Reason:\n    Failed to read atom line %d of %s\n", i,
                qc_type_file);
        }
        atom_local.push_back(idx);
        atom_symbols.push_back(std::string(sym));
    }
    fclose(fp);

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

        const std::vector<ShellData>* shells_ptr = NULL;
        auto it_basis = spec.basis->find(sym);
        if (it_basis != spec.basis->end()) shells_ptr = &(it_basis->second);

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

    for (int i = 0; i < mol.nbas; i++)
        for (int j = 0; j < mol.nbas; j++)
            task_ctx.h_1e_tasks.push_back({i, j});
    task_ctx.n_1e_tasks = task_ctx.h_1e_tasks.size();

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

#include "dft/ao.hpp"
#include "dft/grid.hpp"
#include "dft/vxc.hpp"
#include "dft/xc.hpp"
#include "integrals/cart2sph.hpp"
#include "scf/scf.hpp"
