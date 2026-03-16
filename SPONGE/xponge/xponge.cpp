#include "xponge.h"

#include "load/amber.hpp"
#include "load/native.hpp"

void Xponge::System::Load_Inputs(CONTROLLER* controller)
{
    if (controller->Command_Exist("amber_parm7") ||
        controller->Command_Exist("amber_rst7"))
    {
        Load_Amber_Inputs(this, controller);
        return;
    }
    Load_Native_Inputs(this, controller);
}
