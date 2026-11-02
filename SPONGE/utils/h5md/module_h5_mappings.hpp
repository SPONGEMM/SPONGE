#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
namespace module_path
{
static constexpr const char* nhc_root =
    "/observables/all/thermostat/nose_hoover_chain";
static constexpr const char* nhc_step =
    "/observables/all/thermostat/nose_hoover_chain/step";
static constexpr const char* nhc_time =
    "/observables/all/thermostat/nose_hoover_chain/time";
static constexpr const char* nhc_coordinate_value =
    "/observables/all/thermostat/nose_hoover_chain/coordinate/value";
static constexpr const char* nhc_coordinate_step =
    "/observables/all/thermostat/nose_hoover_chain/coordinate/step";
static constexpr const char* nhc_coordinate_time =
    "/observables/all/thermostat/nose_hoover_chain/coordinate/time";
static constexpr const char* nhc_velocity_value =
    "/observables/all/thermostat/nose_hoover_chain/velocity/value";
static constexpr const char* nhc_velocity_step =
    "/observables/all/thermostat/nose_hoover_chain/velocity/step";
static constexpr const char* nhc_velocity_time =
    "/observables/all/thermostat/nose_hoover_chain/velocity/time";
static constexpr const char* sits_root = "/observables/all/sits";
static constexpr const char* metad_root = "/observables/all/metadynamics";
static constexpr const char* metad_step = "/observables/all/metadynamics/step";
static constexpr const char* metad_time = "/observables/all/metadynamics/time";
static constexpr const char* metad_parameter_root =
    "/parameters/sponge/metadynamics";
static constexpr const char* reaxff_root = "/observables/all/reaxff";
static constexpr const char* reaxff_step = "/observables/all/reaxff/step";
static constexpr const char* reaxff_time = "/observables/all/reaxff/time";
static constexpr const char* reaxff_parameter_root =
    "/parameters/sponge/reaxff";
static constexpr const char* reaxff_eeq_charges_root =
    "/parameters/sponge/reaxff/eeq_charges";
static constexpr const char* reaxff_eeq_charges_value =
    "/parameters/sponge/reaxff/eeq_charges/value";
}  // namespace module_path

inline std::string Nose_Hoover_Chain_Coordinate_Root()
{
    return std::string(module_path::nhc_root) + "/coordinate";
}

inline std::string Nose_Hoover_Chain_Velocity_Root()
{
    return std::string(module_path::nhc_root) + "/velocity";
}

inline std::string Sits_Module_Root(const std::string& module_name)
{
    return std::string(module_path::sits_root) + "/" + module_name;
}

inline std::string Sits_Nk_Root(const std::string& module_name)
{
    return Sits_Module_Root(module_name) + "/nk";
}

inline std::string Scalar_Observable_Value_Path(const std::string& root)
{
    return root + "/value";
}

inline std::string Scalar_Observable_Step_Path(const std::string& root)
{
    return root + "/step";
}

inline std::string Scalar_Observable_Time_Path(const std::string& root)
{
    return root + "/time";
}

inline std::string Sits_Nk_Value_Path(const std::string& module_name)
{
    return Scalar_Observable_Value_Path(Sits_Nk_Root(module_name));
}

inline std::string Sits_Nk_Step_Path(const std::string& module_name)
{
    return Scalar_Observable_Step_Path(Sits_Nk_Root(module_name));
}

inline std::string Sits_Nk_Time_Path(const std::string& module_name)
{
    return Scalar_Observable_Time_Path(Sits_Nk_Root(module_name));
}

inline std::string Metadynamics_Scalar_Root(const std::string& name)
{
    return std::string(module_path::metad_root) + "/" + name;
}

inline std::string Metadynamics_Scalar_Value_Path(const std::string& name)
{
    return Scalar_Observable_Value_Path(Metadynamics_Scalar_Root(name));
}

inline std::string Metadynamics_Scalar_Step_Path(const std::string& name)
{
    return Scalar_Observable_Step_Path(Metadynamics_Scalar_Root(name));
}

inline std::string Metadynamics_Scalar_Time_Path(const std::string& name)
{
    return Scalar_Observable_Time_Path(Metadynamics_Scalar_Root(name));
}

inline std::string Metadynamics_Diagnostic_Root(const std::string& name)
{
    return std::string(module_path::metad_parameter_root) + "/" + name;
}

inline std::string Metadynamics_Diagnostic_Path(const std::string& name,
                                                const std::string& component)
{
    return Metadynamics_Diagnostic_Root(name) + "/" + component;
}

inline std::string Reaxff_Term_Root(const std::string& term)
{
    return std::string(module_path::reaxff_root) + "/" + term;
}

inline std::string Reaxff_Term_Value_Path(const std::string& term)
{
    return Scalar_Observable_Value_Path(Reaxff_Term_Root(term));
}

inline std::string Reaxff_Term_Step_Path(const std::string& term)
{
    return Scalar_Observable_Step_Path(Reaxff_Term_Root(term));
}

inline std::string Reaxff_Term_Time_Path(const std::string& term)
{
    return Scalar_Observable_Time_Path(Reaxff_Term_Root(term));
}

class ModuleH5MappingWriter
{
   public:
    explicit ModuleH5MappingWriter(H5MDWriter* writer) : writer_(writer) {}

    bool Is_Attached() const { return writer_ != nullptr; }

    bool Ensure_Nose_Hoover_Chain_Observables(std::size_t chain_length)
    {
        if (!writer_->Ensure_Group(module_path::nhc_root)) return false;
        if (!writer_->Ensure_Group(Nose_Hoover_Chain_Coordinate_Root()))
        {
            return false;
        }
        if (!writer_->Ensure_Group(Nose_Hoover_Chain_Velocity_Root()))
        {
            return false;
        }
        if (!writer_->Create_Dataset({module_path::nhc_step,
                                      DataType::int64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({module_path::nhc_time,
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        return Create_Vector_Observable(module_path::nhc_coordinate_value,
                                        chain_length) &&
               writer_->Create_Hard_Link(module_path::nhc_step,
                                         module_path::nhc_coordinate_step) &&
               writer_->Create_Hard_Link(module_path::nhc_time,
                                         module_path::nhc_coordinate_time) &&
               Create_Vector_Observable(module_path::nhc_velocity_value,
                                        chain_length) &&
               writer_->Create_Hard_Link(module_path::nhc_step,
                                         module_path::nhc_velocity_step) &&
               writer_->Create_Hard_Link(module_path::nhc_time,
                                         module_path::nhc_velocity_time);
    }

    bool Append_Nose_Hoover_Chain_Frame(const int64_t step, const double time,
                                        const float* coordinates,
                                        const float* velocities,
                                        std::size_t chain_length)
    {
        return writer_->Append_Int64(module_path::nhc_step, &step, 1) &&
               writer_->Append_Float64(module_path::nhc_time, &time, 1) &&
               writer_->Append_Float32(module_path::nhc_coordinate_value,
                                       coordinates, chain_length) &&
               writer_->Append_Float32(module_path::nhc_velocity_value,
                                       velocities, chain_length);
    }

    bool Ensure_Sits_Nk_Observable(const std::string& module_name,
                                   std::size_t k_count)
    {
        const std::string root = Sits_Module_Root(module_name);
        const std::string nk_root = Sits_Nk_Root(module_name);
        if (!writer_->Ensure_Group(module_path::sits_root)) return false;
        if (!writer_->Ensure_Group(root)) return false;
        if (!writer_->Ensure_Group(nk_root)) return false;
        if (!writer_->Create_Dataset({Sits_Nk_Step_Path(module_name),
                                      DataType::int64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({Sits_Nk_Time_Path(module_name),
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        return Create_Vector_Observable(Sits_Nk_Value_Path(module_name),
                                        k_count);
    }

    bool Append_Sits_Nk_Frame(const int64_t step, const double time,
                              const std::string& module_name,
                              const float* values, std::size_t k_count)
    {
        return writer_->Append_Int64(Sits_Nk_Step_Path(module_name), &step,
                                     1) &&
               writer_->Append_Float64(Sits_Nk_Time_Path(module_name), &time,
                                       1) &&
               writer_->Append_Float32(Sits_Nk_Value_Path(module_name), values,
                                       k_count);
    }

    bool Ensure_Metadynamics_Scalars()
    {
        if (!writer_->Ensure_Group(module_path::metad_root)) return false;
        if (!writer_->Create_Dataset({module_path::metad_step,
                                      DataType::int64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({module_path::metad_time,
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        return Create_Scalar_Observable_With_Axis(
                   Metadynamics_Scalar_Root("meta"), module_path::metad_step,
                   module_path::metad_time) &&
               Create_Scalar_Observable_With_Axis(
                   Metadynamics_Scalar_Root("rbias"), module_path::metad_step,
                   module_path::metad_time) &&
               Create_Scalar_Observable_With_Axis(
                   Metadynamics_Scalar_Root("rct"), module_path::metad_step,
                   module_path::metad_time);
    }

    bool Append_Metadynamics_Scalar_Frame(const int64_t step, const double time,
                                          double meta, double rbias, double rct)
    {
        return writer_->Append_Int64(module_path::metad_step, &step, 1) &&
               writer_->Append_Float64(module_path::metad_time, &time, 1) &&
               writer_->Append_Float64(Metadynamics_Scalar_Value_Path("meta"),
                                       &meta, 1) &&
               writer_->Append_Float64(Metadynamics_Scalar_Value_Path("rbias"),
                                       &rbias, 1) &&
               writer_->Append_Float64(Metadynamics_Scalar_Value_Path("rct"),
                                       &rct, 1);
    }

    bool Write_Metadynamics_Diagnostic(const std::string& name,
                                       const std::string& component,
                                       const std::string& text)
    {
        const std::string root = Metadynamics_Diagnostic_Root(name);
        if (!writer_->Ensure_Group(module_path::metad_parameter_root))
        {
            return false;
        }
        if (!writer_->Ensure_Group(root)) return false;
        return writer_->Write_String(
            Metadynamics_Diagnostic_Path(name, component), text);
    }

    bool Write_Metadynamics_Potential_Export(const std::string& name,
                                             const std::string& text)
    {
        return Write_Metadynamics_Diagnostic(name, "potential_export", text);
    }

    bool Write_Metadynamics_Direct_Export(const std::string& name,
                                          const std::string& text)
    {
        return Write_Metadynamics_Diagnostic(name, "direct_export", text);
    }

    bool Write_Metadynamics_Hills(const std::string& name,
                                  const std::string& text)
    {
        return Write_Metadynamics_Diagnostic(name, "hills", text);
    }

    bool Write_Metadynamics_History(const std::string& name,
                                    const std::string& text)
    {
        return Write_Metadynamics_Diagnostic(name, "history", text);
    }

    bool Write_Metadynamics_Edge(const std::string& name,
                                 const std::string& text)
    {
        return Write_Metadynamics_Diagnostic(name, "edge", text);
    }

    bool Ensure_Reaxff_Energy_Terms(const std::vector<std::string>& terms)
    {
        if (!writer_->Ensure_Group(module_path::reaxff_root)) return false;
        if (!writer_->Create_Dataset({module_path::reaxff_step,
                                      DataType::int64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({module_path::reaxff_time,
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        for (const std::string& term : terms)
        {
            if (!Create_Scalar_Observable_With_Axis(Reaxff_Term_Root(term),
                                                    module_path::reaxff_step,
                                                    module_path::reaxff_time))
            {
                return false;
            }
        }
        reaxff_terms_ = terms;
        return true;
    }

    bool Append_Reaxff_Frame(
        const int64_t step, const double time,
        const std::map<std::string, double>& values_by_term)
    {
        if (!writer_->Append_Int64(module_path::reaxff_step, &step, 1))
        {
            return false;
        }
        if (!writer_->Append_Float64(module_path::reaxff_time, &time, 1))
        {
            return false;
        }
        for (const std::string& term : reaxff_terms_)
        {
            const auto iter = values_by_term.find(term);
            if (iter == values_by_term.end())
            {
                last_error_ = "missing ReaxFF term: " + term;
                return false;
            }
            const double value = iter->second;
            if (!writer_->Append_Float64(Reaxff_Term_Value_Path(term), &value,
                                         1))
            {
                return false;
            }
        }
        return true;
    }

    bool Append_Reaxff_Frame(
        const int64_t step, const double time,
        const std::vector<std::string>& terms,
        const std::map<std::string, double>& values_by_term)
    {
        reaxff_terms_ = terms;
        return Append_Reaxff_Frame(step, time, values_by_term);
    }

    bool Ensure_Reaxff_Eeq_Charge_Snapshot(std::size_t atom_count)
    {
        if (writer_ == nullptr || atom_count == 0)
        {
            last_error_ =
                "ReaxFF EEQ charge snapshot requires a positive atom count";
            return false;
        }
        if (!writer_->Ensure_Group(module_path::reaxff_parameter_root) ||
            !writer_->Ensure_Group(module_path::reaxff_eeq_charges_root))
        {
            return false;
        }
        if (!writer_->Create_Dataset(
                {module_path::reaxff_eeq_charges_value,
                 DataType::float32,
                 {{atom_count}, {atom_count}, {atom_count}},
                 false}))
        {
            return false;
        }
        return writer_->Set_String_Attribute(
            module_path::reaxff_eeq_charges_value, "unit", "e");
    }

    bool Write_Reaxff_Eeq_Charge_Snapshot(const float* values,
                                          std::size_t atom_count)
    {
        if (writer_ == nullptr || values == nullptr || atom_count == 0)
        {
            last_error_ = "invalid ReaxFF EEQ charge snapshot";
            return false;
        }
        return writer_->Write_Float32(module_path::reaxff_eeq_charges_value,
                                      values, atom_count);
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_Observable_Frame_Axis()
    {
        if (writer_ == nullptr)
        {
            last_error_ = "ModuleH5MappingWriter has no attached H5MDWriter";
            return false;
        }
        if (observable_axis_defined_)
        {
            return true;
        }
        if (!writer_->Ensure_Group(path::observables_all)) return false;
        if (!writer_->Create_Dataset({path::observables_all_step,
                                      DataType::int64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({path::observables_all_time,
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        observable_axis_defined_ = true;
        return true;
    }

    bool Append_Observable_Frame_Axis(const int64_t step, const double time)
    {
        if (!Ensure_Observable_Frame_Axis()) return false;
        return writer_->Append_Int64(path::observables_all_step, &step, 1) &&
               writer_->Append_Float64(path::observables_all_time, &time, 1);
    }

    bool Create_Scalar_Observable(const std::string& group)
    {
        return Create_Scalar_Observable_With_Axis(
            group, path::observables_all_step, path::observables_all_time);
    }

    bool Create_Scalar_Observable_With_Axis(const std::string& group,
                                            const std::string& step_path,
                                            const std::string& time_path)
    {
        if (!writer_->Ensure_Group(group)) return false;
        if (!writer_->Create_Dataset({Scalar_Observable_Value_Path(group),
                                      DataType::float64,
                                      {{0}, {0}, {0}},
                                      true}))
        {
            return false;
        }
        return writer_->Create_Hard_Link(step_path,
                                         Scalar_Observable_Step_Path(group)) &&
               writer_->Create_Hard_Link(time_path,
                                         Scalar_Observable_Time_Path(group));
    }

    bool Create_Vector_Observable(const std::string& value_path,
                                  std::size_t width)
    {
        return writer_->Create_Dataset({value_path,
                                        DataType::float32,
                                        {{0, width}, {0, 0}, {0, width}},
                                        true});
    }

    H5MDWriter* writer_ = nullptr;
    bool observable_axis_defined_ = false;
    std::vector<std::string> reaxff_terms_;
    std::string last_error_;
};
}  // namespace SpongeH5MD
