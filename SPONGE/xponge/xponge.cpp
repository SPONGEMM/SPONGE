#include "xponge.h"

#include "load/amber.hpp"
#include "load/gromacs.hpp"
#include "load/native.hpp"

void Xponge::System::Load_Inputs(CONTROLLER* controller)
{
    if (controller->Command_Exist("gromacs_top") ||
        controller->Command_Exist("gromacs_gro"))
    {
        Load_Gromacs_Inputs(this, controller);
    }
    else if (controller->Command_Exist("amber_parm7") ||
             controller->Command_Exist("amber_rst7"))
    {
        Load_Amber_Inputs(this, controller);
    }
    else
    {
        Load_Native_Inputs(this, controller);
    }
}
