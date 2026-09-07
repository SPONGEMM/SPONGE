#include <cstdio>
#include <cstdlib>

#include "control.h"

int CONTROLLER::MPI_rank = 0;

namespace
{
// The probes initialize constraints directly. These definitions satisfy linkers
// that resolve references before discarding unused initialization functions.
[[noreturn]] void Unexpected_Configuration_Access()
{
    std::fputs(
        "constraint probe unexpectedly accessed controller configuration\n",
        stderr);
    std::abort();
}
}  // namespace

bool CONTROLLER::Command_Exist(const char*)
{
    Unexpected_Configuration_Access();
}

bool CONTROLLER::Command_Exist(const char*, const char*)
{
    Unexpected_Configuration_Access();
}

const char* CONTROLLER::Command(const char*)
{
    Unexpected_Configuration_Access();
}

const char* CONTROLLER::Command(const char*, const char*)
{
    Unexpected_Configuration_Access();
}
