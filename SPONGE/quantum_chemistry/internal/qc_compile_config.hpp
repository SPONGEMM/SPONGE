#ifndef SPONGE_QUANTUM_CHEMISTRY_INTERNAL_QC_COMPILE_CONFIG_HPP
#define SPONGE_QUANTUM_CHEMISTRY_INTERNAL_QC_COMPILE_CONFIG_HPP

// Shared compile-time constants for split quantum chemistry translation units.
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

#endif
