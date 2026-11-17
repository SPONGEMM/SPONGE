#pragma once

#include <cstdio>
#include <cstring>

// This query intentionally runs before controller, MPI or device initialization.
// Capability support does not assert scientific convergence or unrestricted DD.
inline bool Print_FEP_Capabilities_If_Requested(int argc, char** argv)
{
    if (argc != 2 || std::strcmp(argv[1], "--fep-capabilities-json") != 0)
        return false;
    std::puts(
        "{\"schema\":\"sponge.fep-capabilities.v1\","
        "\"capabilities\":[\"native_nb14_softcore.v1\","
        "\"independent_sterics.v1\",\"fep_core_ordinary.v1\"],"
        "\"inputFormats\":[\"raw\",\"h5\"],"
        "\"scope\":\"fixed-window-energy-force-virial-rerun\","
        "\"limits\":[\"pbc-pme\",\"single-pp-rank\",\"no-sits\",\"no-ti\"]}");
    return true;
}
