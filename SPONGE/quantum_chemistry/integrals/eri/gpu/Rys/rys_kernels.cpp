// Rys quadrature ERI kernel definitions (d/f/g shells).
// Compiled independently. Includes Rys data tables + per-L_sum kernels.

// clang-format off
// Include order matters: quantum_chemistry.h provides macros/types needed by
// ERI GPU headers.
#include "../../../../quantum_chemistry.h"
#include "../../common/eri_kernel_utils.hpp"
#include "../../../../../common.h"
#include "../../common/eri_common.hpp"
#include "../../common/eri_rys.hpp"
// clang-format on

// Rys per-L_sum kernels (L2..L16)
// ERI_MAX_G = max (ij_am+1)*(kl_am+1) over all (la,lb,lc,ld) with
// la+lb+lc+ld=L_sum = ceil((L+2)/2) * floor((L+2)/2)  (constrained by l_max=4 →
// ij,kl <= 8)
#define ERI_NRYS 2
#define ERI_MAX_G 4
#define ERI_MAX_CART 6
#define KERNEL_NAME QC_Fock_Rys_L2_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 2
#define ERI_MAX_G 6
#define ERI_MAX_CART 18
#define KERNEL_NAME QC_Fock_Rys_L3_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 3
#define ERI_MAX_G 9
#define ERI_MAX_CART 54
#define KERNEL_NAME QC_Fock_Rys_L4_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 3
#define ERI_MAX_G 12
#define ERI_MAX_CART 162
#define KERNEL_NAME QC_Fock_Rys_L5_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 4
#define ERI_MAX_G 16
#define ERI_MAX_CART 324
#define KERNEL_NAME QC_Fock_Rys_L6_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 4
#define ERI_MAX_G 20
#define ERI_MAX_CART 648
#define KERNEL_NAME QC_Fock_Rys_L7_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 5
#define ERI_MAX_G 25
#define ERI_MAX_CART 1296
#define KERNEL_NAME QC_Fock_Rys_L8_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 5
#define ERI_MAX_G 30
#define ERI_MAX_CART 2160
#define KERNEL_NAME QC_Fock_Rys_L9_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 6
#define ERI_MAX_G 36
#define ERI_MAX_CART 3600
#define KERNEL_NAME QC_Fock_Rys_L10_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 6
#define ERI_MAX_G 42
#define ERI_MAX_CART 6000
#define KERNEL_NAME QC_Fock_Rys_L11_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 7
#define ERI_MAX_G 49
#define ERI_MAX_CART 10000
#define KERNEL_NAME QC_Fock_Rys_L12_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 7
#define ERI_MAX_G 56
#define ERI_MAX_CART 15000
#define KERNEL_NAME QC_Fock_Rys_L13_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 8
#define ERI_MAX_G 64
#define ERI_MAX_CART 22500
#define KERNEL_NAME QC_Fock_Rys_L14_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 8
#define ERI_MAX_G 72
#define ERI_MAX_CART 33750
#define KERNEL_NAME QC_Fock_Rys_L15_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS
#define ERI_NRYS 9
#define ERI_MAX_G 81
#define ERI_MAX_CART 50625
#define KERNEL_NAME QC_Fock_Rys_L16_Kernel
#include "eri_rys_Lsum.hpp"
#undef KERNEL_NAME
#undef ERI_MAX_CART
#undef ERI_MAX_G
#undef ERI_NRYS

#include "../launch.hpp"
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
