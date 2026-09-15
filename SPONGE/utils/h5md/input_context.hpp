#pragma once

#include <optional>

#include "input_validation.hpp"
#include "topology_native_h5_reader.hpp"

namespace SpongeH5MD
{
// Static launch payloads are shared across initialization phases. Rerun keeps
// its own streaming reader; it must not be cached as a static launch payload.
class InputContext
{
   public:
    template <typename ControllerType>
    bool Initial(ControllerType* controller)
    {
        Clear();
        plan_ = SpongeH5InputPlan::Resolve_Input_Plan(controller);
        const auto validation =
            SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan_);
        error_ = validation.error_message;
        ready_ = validation.valid;
        return ready_;
    }

    const SpongeH5InputPlan::ResolvedInputPlan& Plan() const { return plan_; }
    const std::string& Last_Error() const { return error_; }

    const NativeTopologyCoreState* Topology()
    {
        if (!Check_Binding(plan_.topology.enabled, "topology")) return nullptr;
        if (!topology_)
        {
            TopologyNativeH5Reader reader;
            NativeTopologyCoreState state;
            if (!reader.Open(plan_.topology.path) ||
                !reader.Read_Core_State(&state))
            {
                error_ = reader.Last_Error();
                return nullptr;
            }
            topology_ = std::move(state);
        }
        error_.clear();
        return &*topology_;
    }

    const RestartProtocolState* Protocol_Restart()
    {
        if (!Check_Binding(plan_.restart.binding.enabled, "restart"))
            return nullptr;
        if (!protocol_)
        {
            if (!Open_Restart()) return nullptr;
            RestartProtocolState state;
            if (!restart_->Read_Protocol_State(&state))
            {
                error_ = restart_->Last_Error();
                return nullptr;
            }
            protocol_ = std::move(state);
        }
        error_.clear();
        return &*protocol_;
    }

    const RestartDynamicState* Dynamic_Restart()
    {
        if (!Check_Binding(plan_.restart.binding.enabled, "restart"))
            return nullptr;
        if (!dynamic_)
        {
            if (!Open_Restart()) return nullptr;
            RestartDynamicState state;
            if (!restart_->Read_Dynamic_State(&state))
            {
                error_ = restart_->Last_Error();
                return nullptr;
            }
            dynamic_ = std::move(state);
        }
        error_.clear();
        return &*dynamic_;
    }

    void Clear()
    {
        restart_.reset();
        topology_.reset();
        protocol_.reset();
        dynamic_.reset();
        plan_ = {};
        error_.clear();
        ready_ = false;
    }

   private:
    bool Check_Binding(bool enabled, const char* role)
    {
        if (!ready_)
        {
            if (error_.empty()) error_ = "H5 input context is not initialized";
            return false;
        }
        if (!enabled)
        {
            error_ = std::string("H5 input binding is disabled: ") + role;
            return false;
        }
        return true;
    }

    bool Open_Restart()
    {
        if (restart_) return true;
        auto reader = std::make_unique<RestartH5Reader>();
        if (!reader->Open(plan_.restart.binding.path))
        {
            error_ = reader->Last_Error();
            return false;
        }
        restart_ = std::move(reader);
        return true;
    }

    SpongeH5InputPlan::ResolvedInputPlan plan_;
    std::optional<NativeTopologyCoreState> topology_;
    std::optional<RestartProtocolState> protocol_;
    std::optional<RestartDynamicState> dynamic_;
    std::unique_ptr<RestartH5Reader> restart_;
    std::string error_;
    bool ready_ = false;
};
}  // namespace SpongeH5MD
