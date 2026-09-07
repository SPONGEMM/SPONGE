#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <highfive/highfive.hpp>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../utils/float_classification.hpp"
#include "../../ir/protocol.h"

namespace SpongeH5MD
{
class NativeRestraintH5Reader
{
   public:
    bool Open_Protocol(const std::string& file_path)
    {
        last_error_.clear();
        restraint_name_.clear();
        has_single_weight_default_ = false;
        single_weight_default_ = 0.0f;
        has_refcoord_scaling_default_ = false;
        refcoord_scaling_default_.clear();
        has_calc_virial_default_ = false;
        calc_virial_default_ = true;
        try
        {
            protocol_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            if (!Resolve_Positional_Restraint() || !Read_Defaults())
                return false;
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to open protocol H5 file: ") +
                        error.what());
        }
    }

    bool Open_Restart(const std::string& file_path)
    {
        try
        {
            restart_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to open restart H5 file: ") +
                        error.what());
        }
    }

    bool Has_Positional_Restraint() const { return !restraint_name_.empty(); }

    const std::string& Restraint_Name() const { return restraint_name_; }
    bool Has_Single_Weight_Default() const
    {
        return has_single_weight_default_;
    }
    float Single_Weight_Default() const { return single_weight_default_; }
    bool Has_Refcoord_Scaling_Default() const
    {
        return has_refcoord_scaling_default_;
    }
    const std::string& Refcoord_Scaling_Default() const
    {
        return refcoord_scaling_default_;
    }
    bool Has_Calc_Virial_Default() const { return has_calc_virial_default_; }
    bool Calc_Virial_Default() const { return calc_virial_default_; }

    bool Read(std::size_t atom_count, Xponge::PositionalRestraint* state)
    {
        if (state == nullptr)
        {
            return Fail("native restraint state output pointer is null");
        }
        *state = {};
        if (protocol_ == nullptr)
        {
            return Fail("native restraint protocol H5 reader is not open");
        }
        if (!Has_Positional_Restraint()) return true;
        if (atom_count == 0)
        {
            return Fail("native restraint requires a positive atom count");
        }

        try
        {
            const std::string restraint_root = "/restraint/" + restraint_name_;
            const std::string atom_path = restraint_root + "/atom_indices";
            const auto atom_dimensions = Dimensions(*protocol_, atom_path);
            if (atom_dimensions.size() != 1 || atom_dimensions[0] == 0)
            {
                return Fail(atom_path +
                            " must have shape [n] with n greater than zero");
            }
            const auto atom_indices =
                Read_Flat<int>(*protocol_, atom_path, atom_dimensions,
                               "restraint atom indices");
            std::set<int> unique_indices;
            for (std::size_t row = 0; row < atom_indices.size(); ++row)
            {
                if (atom_indices[row] < 0 ||
                    static_cast<std::size_t>(atom_indices[row]) >= atom_count)
                {
                    std::ostringstream message;
                    message << atom_path << " contains out-of-range atom "
                            << atom_indices[row] << " at row " << row;
                    return Fail(message.str());
                }
                if (!unique_indices.insert(atom_indices[row]).second)
                {
                    std::ostringstream message;
                    message << atom_path << " contains duplicate atom "
                            << atom_indices[row] << " at row " << row;
                    return Fail(message.str());
                }
            }

            std::vector<float> weight;
            const std::string weight_path = restraint_root + "/weight";
            if (Protocol_Exists(weight_path))
            {
                weight = Read_Flat<float>(
                    *protocol_, weight_path,
                    {atom_indices.size(), static_cast<std::size_t>(3)},
                    "restraint weight");
                Validate_Finite(weight, weight_path);
                if (std::any_of(weight.begin(), weight.end(),
                                [](float value) { return value < 0.0f; }))
                {
                    return Fail(weight_path +
                                " contains a negative force constant");
                }
            }

            std::vector<float> reference;
            const std::string reference_path =
                "/parameters/restart/references/restraint/" + restraint_name_ +
                "/coordinate";
            if (Restart_Exists(reference_path))
            {
                reference =
                    Read_Flat<float>(*restart_, reference_path,
                                     {atom_count, static_cast<std::size_t>(3)},
                                     "restraint reference coordinate");
                Validate_Finite(reference, reference_path);
            }
            if (has_single_weight_default_ &&
                (!SpongeFloat::Is_Finite(single_weight_default_) ||
                 single_weight_default_ < 0.0f))
            {
                return Fail(restraint_root +
                            "/single_weight_default must be finite and "
                            "non-negative");
            }
            if (has_refcoord_scaling_default_)
            {
                if (refcoord_scaling_default_ == "none")
                {
                    refcoord_scaling_default_ = "no";
                }
                const std::set<std::string> allowed = {"no", "all", "com_ug",
                                                       "com_res", "com_mol"};
                if (allowed.count(refcoord_scaling_default_) == 0)
                {
                    return Fail(restraint_root +
                                "/refcoord_scaling_default has an unsupported "
                                "value");
                }
            }

            state->present = true;
            state->name = restraint_name_;
            state->atom_indices = atom_indices;
            state->weight = weight;
            state->reference_coordinates = reference;
            state->has_single_weight_default = has_single_weight_default_;
            state->single_weight_default = single_weight_default_;
            state->has_refcoord_scaling_default = has_refcoord_scaling_default_;
            state->refcoord_scaling_default = refcoord_scaling_default_;
            state->has_calc_virial_default = has_calc_virial_default_;
            state->calc_virial_default = calc_virial_default_;
            if (state->weight.empty() && !state->has_single_weight_default)
            {
                return Fail(restraint_root +
                            " requires weight or single_weight_default");
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read native restraint: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    bool Read_Defaults()
    {
        if (restraint_name_.empty()) return true;
        try
        {
            const std::string root = "/restraint/" + restraint_name_;
            Read_Optional_Scalar(root, "single_weight_default",
                                 &has_single_weight_default_,
                                 &single_weight_default_);
            Read_Optional_String(root, "refcoord_scaling_default",
                                 &has_refcoord_scaling_default_,
                                 &refcoord_scaling_default_);
            int calc_virial = 0;
            Read_Optional_Scalar(root, "calc_virial_default",
                                 &has_calc_virial_default_, &calc_virial);
            calc_virial_default_ = calc_virial != 0;
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read positional restraint "
                                    "defaults: ") +
                        error.what());
        }
    }

    template <typename T>
    void Read_Optional_Scalar(const std::string& root, const std::string& field,
                              bool* present, T* value)
    {
        const std::string path = root + "/" + field;
        if (Protocol_Exists(path))
        {
            protocol_->getDataSet(path).read(*value);
            *present = true;
            return;
        }
        auto group = protocol_->getGroup(root);
        if (group.hasAttribute(field))
        {
            group.getAttribute(field).read(*value);
            *present = true;
        }
    }

    void Read_Optional_String(const std::string& root, const std::string& field,
                              bool* present, std::string* value)
    {
        const std::string path = root + "/" + field;
        if (Protocol_Exists(path))
        {
            protocol_->getDataSet(path).read(*value);
            *present = true;
            return;
        }
        auto group = protocol_->getGroup(root);
        if (group.hasAttribute(field))
        {
            group.getAttribute(field).read(*value);
            *present = true;
        }
    }

    bool Resolve_Positional_Restraint()
    {
        restraint_name_.clear();
        if (!Protocol_Exists("/restraint")) return true;
        std::vector<std::string> candidates;
        for (const auto& name :
             protocol_->getGroup("/restraint").listObjectNames())
        {
            const std::string root = "/restraint/" + name;
            const std::string atom_path = root + "/atom_indices";
            if (!Protocol_Exists(atom_path)) continue;
            bool has_enabled = false;
            int enabled = 1;
            Read_Optional_Scalar(root, "enabled_default", &has_enabled,
                                 &enabled);
            if (has_enabled && enabled == 0) continue;
            bool has_type = false;
            std::string type;
            Read_Optional_String(root, "type", &has_type, &type);
            if (has_type && type != "harmonic_positional")
            {
                return Fail(root +
                            "/type must be 'harmonic_positional' when "
                            "atom_indices is present");
            }
            candidates.push_back(name);
        }
        std::sort(candidates.begin(), candidates.end());
        if (candidates.size() > 1)
        {
            return Fail(
                "sponge.input.v2 currently supports one active positional "
                "restraint object; found " +
                std::to_string(candidates.size()));
        }
        if (!candidates.empty()) restraint_name_ = candidates.front();
        return true;
    }

    bool Protocol_Exists(const std::string& path) const
    {
        return protocol_ != nullptr && protocol_->exist(path);
    }

    bool Restart_Exists(const std::string& path) const
    {
        return restart_ != nullptr && restart_->exist(path);
    }

    static std::vector<std::size_t> Dimensions(HighFive::File& file,
                                               const std::string& path)
    {
        if (!file.exist(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        return file.getDataSet(path).getSpace().getDimensions();
    }

    template <typename T>
    static std::vector<T> Read_Flat(HighFive::File& file,
                                    const std::string& path,
                                    const std::vector<std::size_t>& expected,
                                    const std::string& label)
    {
        const auto dimensions = Dimensions(file, path);
        if (dimensions != expected)
        {
            std::ostringstream message;
            message << label << " dataset " << path << " must have shape [";
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                if (index != 0) message << ',';
                message << expected[index];
            }
            message << ']';
            throw std::runtime_error(message.str());
        }
        std::size_t count = 1;
        for (std::size_t value : dimensions) count *= value;
        std::vector<T> values(count);
        HighFive::DataSet dataset = file.getDataSet(path);
        if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
        {
            throw std::runtime_error(label + " failed to read dataset " + path);
        }
        return values;
    }

    static void Validate_Finite(const std::vector<float>& values,
                                const std::string& path)
    {
        for (float value : values)
        {
            if (!SpongeFloat::Is_Finite(value))
            {
                throw std::runtime_error(path + " contains a non-finite value");
            }
        }
    }

    template <typename T>
    static hid_t Native_H5_Type();

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> protocol_;
    std::unique_ptr<HighFive::File> restart_;
    std::string restraint_name_;
    bool has_single_weight_default_ = false;
    float single_weight_default_ = 0.0f;
    bool has_refcoord_scaling_default_ = false;
    std::string refcoord_scaling_default_;
    bool has_calc_virial_default_ = false;
    bool calc_virial_default_ = true;
    std::string last_error_;
};

template <>
inline hid_t NativeRestraintH5Reader::Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t NativeRestraintH5Reader::Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}
}  // namespace SpongeH5MD
