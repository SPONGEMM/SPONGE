#pragma once
#include "../common.h"
#include "../control.h"
#include "dynamics_types.h"
#include "energy_ledger.h"
namespace SpongeDynamics
{
class EnergyAccounting
{
   public:
    void Initial(CONTROLLER*, int, bool, const std::string&, double, double,
                 bool = false);
    void Begin_Step(int, const VECTOR*, const VECTOR*, double,
                    const std::vector<Vec3>* = nullptr);
    void Complete_Step(int, const VECTOR*, const VECTOR*, double,
                       const std::vector<Vec3>* = nullptr);
    void Begin_Thermal(int, const VECTOR*);
    void End_Thermal(int, const VECTOR*, const float*);
    void Begin_Kicked_Thermal(int, const VECTOR*, const VECTOR*, const float*,
                              double);
    void Record_Host_Thermal(double);
    bool Heat_Enabled() const { return ledger_.heat_enabled; }
    double Thermostat_Heat() const { return ledger_.heat; }
    double Coupling_Work() const { return ledger_.work; }
    std::map<std::string, std::string> Export_Restart() const;
    void Restore_Restart(const std::map<std::string, std::string>&, double);
    void Step_Print();
    void Clear();

   private:
    bool enabled_ = false, pending_ = false;
    int atom_count_ = 0;
    CONTROLLER* controller_ = nullptr;
    VECTOR *previous_coordinates_ = nullptr, *previous_force_ = nullptr,
           *previous_velocity_ = nullptr, *zero_external_ = nullptr;
    double* partial_work_ = nullptr;
    std::vector<double> host_work_;
    EnergyLedger ledger_;
    bool ou_pending_ = false, ou_recorded_ = false;
    double heat_increment_ = 0, reported_heat_ = 0, reported_work_ = 0;
    std::vector<Vec3> previous_precise_coordinates_;
    std::vector<VECTOR> previous_host_external_, current_host_external_;
};
}  // namespace SpongeDynamics
