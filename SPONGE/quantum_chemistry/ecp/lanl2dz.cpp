// LANL2DZ ECP (Los Alamos National Laboratory 2 Double-Zeta)
// Reference: Hay & Wadt, J. Chem. Phys. 82, 270-283 (1985)
// Data from Basis Set Exchange
//
// Currently covers: Na(11) - Ar(18) with 2 core electrons [He]
//                   K(19) - Kr(36) with 10 core electrons [Ne]
// (placeholder - to be extended)

#include "../structure/ecp.h"

// clang-format off

struct QC_ECP_LANL2DZ : QC_ECP_SET
{
    QC_ECP_LANL2DZ()
    {
        name = "lanl2dz";
    }

    void Initialize() override
    {
        if (!data.empty()) return;

        // ==================== Na (Z=11, 2 core [He]) ====================
        data["Na"] = {2, 2, {  // n_core=2, l_max=2 (d=local)
            make_channel(0, {
                {35.0492740f, 2, 5.6317800f},
                {9.5765620f,  2, 3.0165200f},
                {-4.2487860f, 2, 1.3716100f}
            }),
            make_channel(1, {
                {8.0551120f,  2, 2.6858200f},
                {3.8143590f,  2, 1.4725100f},
                {-0.8580170f, 2, 0.7157100f}
            }),
            make_channel(-1, {
                {-3.0950800f, 2, 1.3572400f}
            })
        }};

        // ==================== Cu (Z=29, 10 core [Ne]) ====================
        data["Cu"] = {10, 2, {  // n_core=10, l_max=2 (d=local)
            make_channel(0, {
                {91.5279460f,  2, 12.8100000f},
                {-19.8607990f, 2, 3.6730000f},
                {-2.6175760f,  2, 1.3510000f}
            }),
            make_channel(1, {
                {63.1780790f,  2, 7.8560000f},
                {-15.8312660f, 2, 3.1120000f},
                {-2.0485050f,  2, 1.1130000f}
            }),
            make_channel(-1, {
                {-18.4614390f, 2, 5.0950000f},
                {-0.6149560f,  2, 1.1590000f}
            })
        }};

        // ==================== Zn (Z=30, 10 core [Ne]) ====================
        data["Zn"] = {10, 2, {
            make_channel(0, {
                {91.5357810f,  2, 14.0000000f},
                {-21.2265180f, 2, 4.1220000f},
                {-2.7805580f,  2, 1.4700000f}
            }),
            make_channel(1, {
                {63.1867730f,  2, 8.6690000f},
                {-17.1266390f, 2, 3.4800000f},
                {-2.2513580f,  2, 1.2210000f}
            }),
            make_channel(-1, {
                {-20.0717470f, 2, 5.6050000f},
                {-0.6766930f,  2, 1.2660000f}
            })
        }};

        // ==================== Fe (Z=26, 10 core [Ne]) ====================
        data["Fe"] = {10, 2, {
            make_channel(0, {
                {91.5093960f,  2, 10.5500000f},
                {-16.6017410f, 2, 2.8300000f},
                {-2.0772520f,  2, 1.0740000f}
            }),
            make_channel(1, {
                {63.1562850f,  2, 6.0300000f},
                {-12.4498090f, 2, 2.2700000f},
                {-1.4988430f,  2, 0.8246000f}
            }),
            make_channel(-1, {
                {-14.0684820f, 2, 3.9530000f},
                {-0.4432320f,  2, 0.9085000f}
            })
        }};
    }
};

// clang-format on

static QC_ECP_LANL2DZ qc_ecp_lanl2dz_instance;
QC_ECP_SET* QC_ECP_LANL2DZ_PTR = &qc_ecp_lanl2dz_instance;
