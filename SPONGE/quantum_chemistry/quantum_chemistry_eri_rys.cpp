// Rys quadrature ERI kernel definitions (d/f/g shells).
// Compiled independently. Includes Rys data tables + per-L_sum kernels.

#include "quantum_chemistry.h"
#include "gpu_eri/eri_kernel_utils.hpp"
#include "../common.h"

#include "gpu_eri/eri_common.hpp"
#include "gpu_eri/eri_rys.hpp"

// Screening kernel
#include "gpu_eri/eri_screen_compact.hpp"

// Rys per-L_sum kernels (L2..L16)
// ERI_MAX_G = max (ij_am+1)*(kl_am+1) over all (la,lb,lc,ld) with la+lb+lc+ld=L_sum
// = ceil((L+2)/2) * floor((L+2)/2)  (constrained by l_max=4 → ij,kl <= 8)
#define ERI_NRYS 2
#define ERI_MAX_G 4
#define ERI_MAX_CART 6
#define KERNEL_NAME QC_Fock_Rys_L2_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 2
#define ERI_MAX_G 6
#define ERI_MAX_CART 18
#define KERNEL_NAME QC_Fock_Rys_L3_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 3
#define ERI_MAX_G 9
#define ERI_MAX_CART 54
#define KERNEL_NAME QC_Fock_Rys_L4_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 3
#define ERI_MAX_G 12
#define ERI_MAX_CART 162
#define KERNEL_NAME QC_Fock_Rys_L5_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 4
#define ERI_MAX_G 16
#define ERI_MAX_CART 324
#define KERNEL_NAME QC_Fock_Rys_L6_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 4
#define ERI_MAX_G 20
#define ERI_MAX_CART 648
#define KERNEL_NAME QC_Fock_Rys_L7_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 5
#define ERI_MAX_G 25
#define ERI_MAX_CART 1296
#define KERNEL_NAME QC_Fock_Rys_L8_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 5
#define ERI_MAX_G 30
#define ERI_MAX_CART 2160
#define KERNEL_NAME QC_Fock_Rys_L9_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 6
#define ERI_MAX_G 36
#define ERI_MAX_CART 3600
#define KERNEL_NAME QC_Fock_Rys_L10_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 6
#define ERI_MAX_G 42
#define ERI_MAX_CART 6000
#define KERNEL_NAME QC_Fock_Rys_L11_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 7
#define ERI_MAX_G 49
#define ERI_MAX_CART 10000
#define KERNEL_NAME QC_Fock_Rys_L12_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 7
#define ERI_MAX_G 56
#define ERI_MAX_CART 15000
#define KERNEL_NAME QC_Fock_Rys_L13_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 8
#define ERI_MAX_G 64
#define ERI_MAX_CART 22500
#define KERNEL_NAME QC_Fock_Rys_L14_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 8
#define ERI_MAX_G 72
#define ERI_MAX_CART 33750
#define KERNEL_NAME QC_Fock_Rys_L15_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 9
#define ERI_MAX_G 81
#define ERI_MAX_CART 50625
#define KERNEL_NAME QC_Fock_Rys_L16_Kernel
#include "gpu_eri/eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS

#include "gpu_eri/eri_launch.hpp"
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L2, QC_Fock_Rys_L2_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L3, QC_Fock_Rys_L3_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L4, QC_Fock_Rys_L4_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L5, QC_Fock_Rys_L5_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L6, QC_Fock_Rys_L6_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L7, QC_Fock_Rys_L7_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L8, QC_Fock_Rys_L8_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L9, QC_Fock_Rys_L9_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L10, QC_Fock_Rys_L10_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L11, QC_Fock_Rys_L11_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L12, QC_Fock_Rys_L12_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L13, QC_Fock_Rys_L13_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L14, QC_Fock_Rys_L14_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L15, QC_Fock_Rys_L15_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_Rys_L16, QC_Fock_Rys_L16_Kernel)

// Screening kernel wrapper
void QC_Launch_Screen(
    int n_total,
    const QC_INTEGRAL_TASKS::ScreenCombo* combos,
    const int* combo_prefix,
    int n_combos,
    const int* sorted_pair_ids,
    const QC_ONE_E_TASK* shell_pairs,
    const float* shell_pair_bounds,
    const float* pair_density_coul,
    const float* pair_density_exx_a,
    const float* pair_density_exx_b,
    float shell_screen_tol,
    float exx_scale_a, float exx_scale_b,
    QC_ERI_TASK* output_tasks,
    int* output_counts)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Screen_All_Combos_Kernel,
        (n_total + threads - 1) / threads, threads, 0, 0,
        n_total, combos, combo_prefix, n_combos,
        sorted_pair_ids, shell_pairs, shell_pair_bounds,
        pair_density_coul, pair_density_exx_a, pair_density_exx_b,
        shell_screen_tol, exx_scale_a, exx_scale_b,
        output_tasks, output_counts);
}
