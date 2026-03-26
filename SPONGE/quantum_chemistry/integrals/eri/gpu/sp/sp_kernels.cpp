// SP-shell ERI kernel definitions (s/p only, l_max <= 1).
// Compiled independently from the SCF dispatch code.

// clang-format off
// Include order matters: quantum_chemistry.h provides macros/types needed by
// ERI GPU headers.
#include "../../../../quantum_chemistry.h"
#include "../../common/eri_kernel_utils.hpp"
#include "../../../../../common.h"
// clang-format on

// Common utilities
#include "../../common/eri_common.hpp"

// ssss: hand-written specialized kernel
#include "eri_ssss.hpp"

// 3s1p: 4 permutations by p-shell position
#define P_POS 0
#define KERNEL_NAME QC_Fock_psss_Kernel
#include "eri_3s1p.hpp"
#undef KERNEL_NAME
#undef P_POS
#define P_POS 1
#define KERNEL_NAME QC_Fock_spss_Kernel
#include "eri_3s1p.hpp"
#undef KERNEL_NAME
#undef P_POS
#define P_POS 2
#define KERNEL_NAME QC_Fock_ssps_Kernel
#include "eri_3s1p.hpp"
#undef KERNEL_NAME
#undef P_POS
#define P_POS 3
#define KERNEL_NAME QC_Fock_sssp_New_Kernel
#include "eri_3s1p.hpp"
#undef KERNEL_NAME
#undef P_POS

// pppp
#include "eri_pppp.hpp"

// 2s2p: 6 permutations
#define P_POS0 0
#define P_POS1 1
#define KERNEL_NAME QC_Fock_ppss_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0
#define P_POS0 0
#define P_POS1 2
#define KERNEL_NAME QC_Fock_psps_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0
#define P_POS0 0
#define P_POS1 3
#define KERNEL_NAME QC_Fock_pssp_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0
#define P_POS0 1
#define P_POS1 2
#define KERNEL_NAME QC_Fock_spps_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0
#define P_POS0 1
#define P_POS1 3
#define KERNEL_NAME QC_Fock_spsp_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0
#define P_POS0 2
#define P_POS1 3
#define KERNEL_NAME QC_Fock_sspp_New_Kernel
#include "eri_2s2p.hpp"
#undef KERNEL_NAME
#undef P_POS1
#undef P_POS0

// 1s3p: 4 permutations
#define S_POS 0
#define KERNEL_NAME QC_Fock_sppp_New_Kernel
#include "eri_1s3p.hpp"
#undef KERNEL_NAME
#undef S_POS
#define S_POS 1
#define KERNEL_NAME QC_Fock_pspp_Kernel
#include "eri_1s3p.hpp"
#undef KERNEL_NAME
#undef S_POS
#define S_POS 2
#define KERNEL_NAME QC_Fock_ppsp_Kernel
#include "eri_1s3p.hpp"
#undef KERNEL_NAME
#undef S_POS
#define S_POS 3
#define KERNEL_NAME QC_Fock_ppps_Kernel
#include "eri_1s3p.hpp"
#undef KERNEL_NAME
#undef S_POS

// Wrapper functions for cross-TU kernel launching
#include "../launch.hpp"
DEFINE_ERI_LAUNCH(QC_Launch_ssss, QC_Fock_ssss_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_psss, QC_Fock_psss_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_spss, QC_Fock_spss_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_ssps, QC_Fock_ssps_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_sssp, QC_Fock_sssp_New_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_ppss, QC_Fock_ppss_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_psps, QC_Fock_psps_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_pssp, QC_Fock_pssp_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_spps, QC_Fock_spps_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_spsp, QC_Fock_spsp_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_sspp, QC_Fock_sspp_New_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_sppp, QC_Fock_sppp_New_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_pspp, QC_Fock_pspp_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_ppsp, QC_Fock_ppsp_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_ppps, QC_Fock_ppps_Kernel)
DEFINE_ERI_LAUNCH(QC_Launch_pppp, QC_Fock_pppp_Kernel)
