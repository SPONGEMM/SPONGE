// McMurchie-Davidson d-shell ERI kernel definitions (l_max <= 2).
// Compiled independently.

// clang-format off
// Include order matters: quantum_chemistry.h provides macros/types needed by
// ERI GPU headers.
#include "../../../../quantum_chemistry.h"
#include "../../common/eri_kernel_utils.hpp"
#include "../../../../../common.h"
#include "../../common/eri_common.hpp"
// clang-format on

// MD per-L_sum kernels for d-containing quartets
#define ERI_LSUM 2
#define ERI_F_SIZE 3
#define ERI_R0_SIZE 10
#define ERI_RW_SIZE 15
#define ERI_MAX_CART 6
#define KERNEL_NAME QC_Fock_D_L2_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 3
#define ERI_F_SIZE 4
#define ERI_R0_SIZE 20
#define ERI_RW_SIZE 35
#define ERI_MAX_CART 18
#define KERNEL_NAME QC_Fock_D_L3_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 4
#define ERI_F_SIZE 5
#define ERI_R0_SIZE 35
#define ERI_RW_SIZE 70
#define ERI_MAX_CART 54
#define KERNEL_NAME QC_Fock_D_L4_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 5
#define ERI_F_SIZE 6
#define ERI_R0_SIZE 56
#define ERI_RW_SIZE 126
#define ERI_MAX_CART 162
#define KERNEL_NAME QC_Fock_D_L5_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 6
#define ERI_F_SIZE 7
#define ERI_R0_SIZE 84
#define ERI_RW_SIZE 210
#define ERI_MAX_CART 324
#define KERNEL_NAME QC_Fock_D_L6_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 7
#define ERI_F_SIZE 8
#define ERI_R0_SIZE 120
#define ERI_RW_SIZE 330
#define ERI_MAX_CART 648
#define KERNEL_NAME QC_Fock_D_L7_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM
#define ERI_LSUM 8
#define ERI_F_SIZE 9
#define ERI_R0_SIZE 165
#define ERI_RW_SIZE 495
#define ERI_MAX_CART 1296
#define KERNEL_NAME QC_Fock_D_L8_Kernel
#include "eri_d_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_RW_SIZE
#undef ERI_R0_SIZE
#undef ERI_F_SIZE
#undef ERI_LSUM

#include "../launch.hpp"
DEFINE_ERI_LAUNCH(QC_Launch_D_L2, QC_Fock_D_L2_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L3, QC_Fock_D_L3_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L4, QC_Fock_D_L4_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L5, QC_Fock_D_L5_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L6, QC_Fock_D_L6_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L7, QC_Fock_D_L7_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_D_L8, QC_Fock_D_L8_Kernel)
