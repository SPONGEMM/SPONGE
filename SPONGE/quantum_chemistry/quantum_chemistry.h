#ifndef QUANTUM_CHEMISTRY_H
#define QUANTUM_CHEMISTRY_H

#include "../common.h"
#include "../control.h"
#include "structure/cart2sph.h"
#include "structure/dft.h"
#include "structure/integral_tasks.h"
#include "structure/matrix.h"
#include "structure/method.h"
#include "structure/molecule.h"
#include "structure/scf_workspace.h"

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

struct QUANTUM_CHEMISTRY
{
   public:
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int last_modify_date = 20260216;
    int atom_numbers = 0;

    float scf_energy = 0.0f;
    FILE* scf_output_file = NULL;

    // 本地原子映射
    std::vector<int> atom_local;
    int* d_atom_local = NULL;

    // 计算方法
    QC_METHOD method = QC_METHOD::HF;
    // 初始猜测
    QC_INITIAL_GUESS initial_guess = QC_INITIAL_GUESS::SAP;
    bool need_initial_guess = true;
    // DFT信息
    QC_DFT dft;
    // 分子信息
    QC_MOLECULE mol;
    // 积分任务
    QC_INTEGRAL_TASKS task_ctx;
    // SCF计算内容
    QC_SCF_WORKSPACE scf_ws;

    BLAS_HANDLE blas_handle;
    SOLVER_HANDLE solver_handle;

    // 笛卡尔基组转球形基组
    QC_CARTESIAN_TO_SPHERICAL cart2sph;

    // 外部入口
    void Initial(CONTROLLER* controller, const int atom_numbers,
                 const VECTOR* crd, const char* module_name = NULL);
    void Solve_SCF(const VECTOR* crd, const VECTOR box_length,
                   bool need_energy = true, int md_step = -1);

    // 外部查询与输出
    void Step_Print(CONTROLLER* controller);

   private:
    // 初始化内部流程
    bool Parsing_Arguments(CONTROLLER* controller, const int atom_numbers,
                           const char*& qc_type_file,
                           std::string& basis_set_name);
    void Initial_Molecule(CONTROLLER* controller, const char* qc_type_file,
                          const std::string& basis_set_name);
    void Initial_Integral_Tasks(CONTROLLER* controller);
    void Memory_Allocate(CONTROLLER* controller);
    void Build_SCF_Workspace();

    // 积分与基组变换内部流程
    void Build_Cart2Sph_Matrix();
    void Cart2Sph_OneE_Integrals();
    void Cart2Sph_Single_Matrix(float* d_cart, float* d_sph);

    // 坐标更新
    void Update_Coordinates_From_MD(const VECTOR* crd, const VECTOR box_length);

    // 积分
    void Compute_OneE_Integrals();
    void Compute_Nuclear_Repulsion(const VECTOR box_length);
    void Prepare_Integrals();

    // DFT VXC 构建
    void Update_DFT_Grid();
    void Build_DFT_VXC();

    // SCF 循环内部流程
    void Build_Fock(int iter);
    void Accumulate_SCF_Energy(int iter);
    void Apply_DIIS(int iter);
    void Diagonalize_And_Build_Density();
    bool Check_Convergence(int iter, int md_step);
    void Build_Overlap_X();
    void Reset_SCF_State();
    void Build_Initial_Guess();
    void Diag_Guess_And_Build_P();
    void Compute_Spin_Square();
};

#endif
