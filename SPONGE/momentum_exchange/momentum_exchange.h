#pragma once
#include "../common.h"
#include "../control.h"
#include "exchange_math.h"
namespace SpongeDynamics
{
class MomentumExchangeController
{
   public:
    void Initial(CONTROLLER*, int, const float*, VECTOR, VECTOR, bool,
                 const std::string&, double, double);
    bool enabled() const { return parameters_.enabled; }
    const ExchangeParameters& Parameters() const { return parameters_; }
    bool Due(int step) const
    {
        return enabled() && step > 0 &&
               step % parameters_.exchange_interval == 0;
    }
    void Begin_Step(double);
    void Complete_Step(double);
    MomentumExchange Apply(int, const int*, const VECTOR*, VECTOR*,
                           const VECTOR*, double);
    void Record(const MomentumExchange&, double, int);
    std::map<std::string, std::string> Export_Restart() const;
    void Restore_Restart(const std::map<std::string, std::string>&, double);
    std::map<std::string, double> Values() const;
    std::map<std::string, std::string> Units() const;
    void Clear() { *this = MomentumExchangeController{}; }

   private:
    CONTROLLER* controller_ = nullptr;
    int atom_count_ = 0;
    ExchangeParameters parameters_;
    ExchangeLedger ledger_;
    bool pending_ = false;
    std::vector<int> ids_;
    std::vector<VECTOR> coordinates_, exchange_velocity_, exchange_force_;
    std::vector<double> masses_, exchange_mass_, exchange_position_,
        exchange_centered_;
};
}  // namespace SpongeDynamics
