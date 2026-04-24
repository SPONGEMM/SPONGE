#ifndef REAXFF_H
#define REAXFF_H

#include "../../Domain_decomposition/Domain_decomposition.h"
#include "../../control.h"
#include "../../neighbor_list/neighbor_list.h"
#include "bond.h"
#include "bond_order.h"
#include "eeq.h"
#include "hydrogen_bond.h"
#include "over_under.h"
#include "torsion.h"
#include "valence_angle.h"
#include "vdw.h"

struct REAXFF
{
    int is_initialized = 0;

    REAXFF_EEQ eeq;
    REAXFF_BOND_ORDER bond_order;
    REAXFF_BOND bond;
    REAXFF_VDW vdw;
    REAXFF_OVER_UNDER ovun;
    REAXFF_VALENCE_ANGLE angle;
    REAXFF_TORSION torsion;
    REAXFF_HYDROGEN_BOND hb;

    void Initial(CONTROLLER* controller, int atom_numbers, float cutoff,
                 float* cutoff_full, bool* need_full_nl_flag);
    void Calculate_Force(DOMAIN_INFORMATION* dd, MD_INFORMATION* md_info,
                         NEIGHBOR_LIST* neighbor_list);
    void Step_Print(CONTROLLER* controller, const float* d_charge);

   private:
    void Wire_Shared_State();
};

#endif
