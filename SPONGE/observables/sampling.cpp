#include "sampling.h"

#include <stdexcept>

#include "../MD_core/energy_ledger.h"
#include "../utils/control/component_options.hpp"
namespace SpongeDynamics
{
void SamplingPlan::Initial(CONTROLLER* c, VECTOR box, VECTOR angles,
                           bool periodic)
{
    try
    {
        Component_Keys(c, "sampling",
                       {"stress", "profile", "interval", "output", "flow",
                        "gradient", "bins", "mode", "velocity_time_layer"});
        stress = Option_Bool(c, "sampling_stress");
        std::string kind = c->Command_Exist("sampling_profile")
                               ? c->Command("sampling_profile")
                               : "none";
        if (kind != "none" && kind != "slab" && kind != "cosine")
            throw std::runtime_error(
                "sampling.profile must be none/slab/cosine");
        profile = kind != "none";
        mode.enabled = profile;
        mode.cosine = kind == "cosine";
        interval = Option_Integer(c, "sampling_interval", 1);
        if (c->Command_Exist("sampling_output"))
            output = c->Command("sampling_output");
        if (output.empty())
            throw std::runtime_error("sampling.output cannot be empty");
        if (c->Command_Exist("sampling_velocity_time_layer"))
        {
            std::string s = c->Command("sampling_velocity_time_layer");
            if (s != "stored" && s != "force_centered")
                throw std::runtime_error(
                    "sampling velocity time layer must be "
                    "stored/force_centered");
            centered = s == "force_centered";
        }
        if (profile)
        {
            Orthogonal_Box(box, angles, periodic);
            mode.flow = Option_Axis(c, "sampling_flow", 0);
            mode.gradient = Option_Axis(c, "sampling_gradient", 2);
            if (mode.flow == mode.gradient)
                throw std::runtime_error(
                    "sampling flow and gradient must differ");
            mode.bins = Option_Integer(c, "sampling_bins", 20, 4, 10000);
            mode.length = mode.gradient == 0   ? box.x
                          : mode.gradient == 1 ? box.y
                                               : box.z;
            mode.wave_number = 2 * 3.14159265358979323846 *
                               Option_Integer(c, "sampling_mode", 1, 1, 1000) /
                               mode.length;
        }
    }
    catch (const std::exception& e)
    {
        c->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                              "SamplingPlan::Initial", e.what());
    }
}
}  // namespace SpongeDynamics
