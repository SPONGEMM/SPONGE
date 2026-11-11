#pragma once

#include "../Domain_decomposition/Domain_decomposition.h"
#include "../MD_core/MD_core.h"
#include "../MD_core/energy_accounting.h"
#include "../MD_core/integrator/constrained_middle.h"
#include "../MD_core/integrator/rattle_state.h"
#include "../SITS/SITS.h"
#include "../barostat/MC_barostat.h"
#include "../barostat/pressure_based_barostat.h"
#include "../bias/sinkmeta.h"
#include "../external_force/external_force.h"
#include "../manybody/reaxff/reaxff.h"
#include "../momentum_exchange/momentum_exchange.h"
#include "../observables/profile_sampler.h"
#include "../observables/sampling.h"
#include "../observables/stress_sampler.h"
#include "../restrain/restrain.h"
#include "../thermostat/Andersen_thermostat.h"
#include "../thermostat/Bussi_thermostat.h"
#include "../thermostat/Middle_Langevin_MD.h"
#include "../thermostat/Nose_Hoover_Chain.h"
#include "../thermostat/velocity_bias/velocity_bias.h"
#include "../xponge/xponge.h"

namespace SpongeIO
{
// Borrowed runtime objects must outlive both synchronous sessions.
struct RuntimeState
{
    CONTROLLER& controller;
    Xponge::System& system;
    MD_INFORMATION& md_info;
    DOMAIN_INFORMATION& dd;
    MIDDLE_Langevin_INFORMATION& middle_langevin;
    ANDERSEN_THERMOSTAT_INFORMATION& ad_thermo;
    BUSSI_THERMOSTAT_INFORMATION& bussi_thermo;
    NOSE_HOOVER_CHAIN_INFORMATION& nhc;
    PRESSURE_BASED_BAROSTAT_INFORMATION& press_baro;
    MC_BAROSTAT_INFORMATION& mc_baro;
    RESTRAIN_INFORMATION& restrain;
    COLLECTIVE_VARIABLE_CONTROLLER& cv_controller;
    META& meta;
    SITS_INFORMATION& sits;
    REAXFF& reaxff;
    SpongeDynamics::VelocityBias& velocity_bias;
    SpongeDynamics::ExternalForce& external_force;
    SpongeDynamics::EnergyAccounting& energy_accounting;
    SpongeDynamics::ConstrainedNVEState& rattle_nve;
    SpongeDynamics::SamplingPlan& sampling;
    SpongeDynamics::StressSampler& stress_sampler;
    SpongeDynamics::ProfileSampler& profile_sampler;
    SpongeDynamics::MomentumExchangeController& momentum_exchange;
    SpongeDynamics::ConstrainedMiddle& constrained_middle;
    deviceStream_t& main_stream;
};

inline std::string Current_MD_Mode_Name(const MD_INFORMATION& md_info)
{
    if (md_info.mode == md_info.RERUN) return "rerun";
    if (md_info.mode == md_info.MINIMIZATION) return "minimization";
    if (md_info.mode == md_info.NVE) return "nve";
    if (md_info.mode == md_info.NVT) return "nvt";
    if (md_info.mode == md_info.NPT) return "npt";
    return "unknown";
}
}  // namespace SpongeIO
