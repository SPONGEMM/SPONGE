#include "xponge.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../utils/float_classification.hpp"
#include "load/amber.hpp"
#include "load/gromacs.hpp"
#include "load/native.hpp"
#include "load/native/nb14_extra_h5.hpp"
#include "load/native/restraint_h5.hpp"
#include "utils/h5md/topology_h5_reader.hpp"

namespace
{
bool Is_Supported_Topology_Schema_Version(const std::string& version)
{
    return version == "0" || version == "1" ||
           version == "xponge.legacy_to_bundle.v1" ||
           version == "sponge.input.v2";
}

void Validate_H5_Topology_Schema_Version(CONTROLLER* controller)
{
    constexpr const char* input_key = "input_h5_topology_path";
    if (!controller->Command_Exist(input_key))
    {
        return;
    }

    SpongeH5MD::TopologyH5Reader reader;
    if (!reader.Open(controller->Command(input_key)))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Validate_H5_Topology_Schema_Version",
            reader.Last_Error().c_str());
    }
    SpongeH5InputMetadata::TopologyMetadata metadata;
    if (!reader.Read_Metadata(&metadata))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Validate_H5_Topology_Schema_Version",
            reader.Last_Error().c_str());
    }
    if (!Is_Supported_Topology_Schema_Version(metadata.schema_version))
    {
        const std::string message = std::string("Reason:\n\t") + input_key +
                                    " has unsupported /schema/version '" +
                                    metadata.schema_version +
                                    "'\n\tSupported versions: 0, 1, "
                                    "xponge.legacy_to_bundle.v1, "
                                    "sponge.input.v2\n";
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Xponge::Validate_H5_Topology_Schema_Version", message.c_str());
    }
}

void Materialize_H5_Native_NB14_Extra(CONTROLLER* controller,
                                      Xponge::System* system)
{
    constexpr const char* input_key = "input_h5_topology_path";
    if (!controller->Command_Exist(input_key))
    {
        return;
    }

    int atom_count = 0;
    if (!system->atoms.mass.empty())
    {
        atom_count = static_cast<int>(system->atoms.mass.size());
    }
    else if (!system->atoms.charge.empty())
    {
        atom_count = static_cast<int>(system->atoms.charge.size());
    }
    else if (!system->atoms.coordinate.empty())
    {
        atom_count = static_cast<int>(system->atoms.coordinate.size() / 3);
    }

    SpongeH5MD::NativeNB14ExtraH5Reader reader;
    if (!reader.Open(controller->Command(input_key)))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Materialize_H5_Native_NB14_Extra",
                                       reader.Last_Error().c_str());
    }
    SpongeH5MD::NativeNB14ExtraState state;
    if (!reader.Read(atom_count, &state))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Materialize_H5_Native_NB14_Extra",
                                       reader.Last_Error().c_str());
    }
    if (!state.present)
    {
        return;
    }
    if (controller->Command_Exist("nb14_in_file") ||
        controller->Command_Exist("nb14_extra_in_file"))
    {
        return;
    }

    Xponge::NB14& nb14 = system->classical_force_field.nb14;
    nb14.atom_a = state.atom_a;
    nb14.atom_b = state.atom_b;
    nb14.A = state.A;
    nb14.B = state.B;
    nb14.cf_scale_factor = state.cf_scale_factor;
}

void Load_H5_Native_Positional_Restraint(CONTROLLER* controller,
                                         std::size_t atom_count,
                                         Xponge::PositionalRestraint* restraint)
{
    constexpr const char* protocol_key = "input_h5_protocol_path";
    if (restraint == nullptr)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Load_H5_Native_Positional_Restraint",
            "Reason:\n\tnative positional restraint output is null\n");
    }
    *restraint = {};
    if (!controller->Command_Exist(protocol_key))
    {
        return;
    }

    SpongeH5MD::NativeRestraintH5Reader reader;
    if (!reader.Open_Protocol(controller->Command(protocol_key)))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Load_H5_Native_Positional_Restraint",
                                       reader.Last_Error().c_str());
    }
    if (!reader.Has_Positional_Restraint())
    {
        return;
    }
    if (controller->Command_Exist("restrain", "atom_id") ||
        controller->Command_Exist("restrain", "weight_in_file") ||
        controller->Command_Exist("restrain", "coordinate_in_file") ||
        controller->Command_Exist("restrain", "amber_rst7"))
    {
        return;
    }
    if (controller->Command_Exist("input_h5_restart_path") &&
        !reader.Open_Restart(controller->Command("input_h5_restart_path")))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Load_H5_Native_Positional_Restraint",
                                       reader.Last_Error().c_str());
    }
    if (!reader.Read(atom_count, restraint))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Load_H5_Native_Positional_Restraint",
                                       reader.Last_Error().c_str());
    }
}

std::vector<int> Read_H5_Residue_Atom_Numbers(CONTROLLER* controller,
                                              std::size_t atom_count)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* dataset_path = "/atoms/residue_index";
    constexpr const char* offset_path = "/residues/atom_offset";
    if (!controller->Command_Exist(input_key))
    {
        return {};
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(dataset_path))
        {
            return {};
        }
        if (controller->Command_Exist("residue_in_file"))
        {
            const std::string message =
                std::string("Reason:\n\tnative residue mapping ") +
                dataset_path +
                " conflicts with residue_in_file; select exactly one residue "
                "input owner\n";
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Xponge::Read_H5_Residue_Atom_Numbers", message.c_str());
        }

        HighFive::DataSet dataset = file.getDataSet(dataset_path);
        const auto dimensions = dataset.getSpace().getDimensions();
        if (dimensions.size() != 1)
        {
            throw std::runtime_error(
                "/atoms/residue_index must be a one-dimensional dataset");
        }
        if (dimensions[0] != atom_count)
        {
            std::ostringstream message;
            message << "/atoms/residue_index length " << dimensions[0]
                    << " does not match atom count " << atom_count;
            throw std::runtime_error(message.str());
        }

        std::vector<std::int64_t> residue_indices;
        dataset.read(residue_indices);
        if (residue_indices.empty() || residue_indices.front() != 0)
        {
            throw std::runtime_error(
                "/atoms/residue_index must start with residue index 0");
        }

        std::vector<int> residue_atom_numbers;
        residue_atom_numbers.push_back(1);
        for (std::size_t atom = 1; atom < residue_indices.size(); ++atom)
        {
            const std::int64_t previous = residue_indices[atom - 1];
            const std::int64_t current = residue_indices[atom];
            if (current == previous)
            {
                ++residue_atom_numbers.back();
            }
            else if (current == previous + 1)
            {
                residue_atom_numbers.push_back(1);
            }
            else
            {
                std::ostringstream message;
                message << "/atoms/residue_index must contain contiguous, "
                           "nondecreasing labels; atom "
                        << atom << " changes from " << previous << " to "
                        << current;
                throw std::runtime_error(message.str());
            }
        }

        if (file.exist(offset_path))
        {
            HighFive::DataSet offsets_dataset = file.getDataSet(offset_path);
            const auto offset_dimensions =
                offsets_dataset.getSpace().getDimensions();
            if (offset_dimensions.size() != 1)
            {
                throw std::runtime_error(
                    "/residues/atom_offset must be a one-dimensional dataset");
            }
            std::vector<std::int64_t> atom_offsets;
            offsets_dataset.read(atom_offsets);
            std::vector<std::int64_t> expected_offsets = {0};
            expected_offsets.reserve(residue_atom_numbers.size() + 1);
            for (const int residue_atom_number : residue_atom_numbers)
            {
                expected_offsets.push_back(expected_offsets.back() +
                                           residue_atom_number);
            }
            if (atom_offsets != expected_offsets)
            {
                throw std::runtime_error(
                    "/atoms/residue_index and /residues/atom_offset describe "
                    "different residue mappings");
            }
        }
        return residue_atom_numbers;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed residue membership ") +
            "from " + dataset_path + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Xponge::Read_H5_Residue_Atom_Numbers",
                                       message.c_str());
    }
    return {};
}

bool Read_H5_Distance_Constraints(CONTROLLER* controller,
                                  std::size_t atom_count,
                                  Xponge::DistanceConstraints* constraints)
{
    constexpr const char* input_key = "input_h5_protocol_path";
    constexpr const char* atoms_path = "/constraint/default/pairs/atoms";
    constexpr const char* distance_path = "/constraint/default/pairs/r0";
    if (!controller->Command_Exist(input_key))
    {
        return false;
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        const bool has_atoms = file.exist(atoms_path);
        const bool has_distances = file.exist(distance_path);
        if (!has_atoms && !has_distances)
        {
            return false;
        }
        if (controller->Command_Exist("constrain_in_file"))
        {
            return false;
        }
        if (!has_atoms || !has_distances)
        {
            throw std::runtime_error(
                "typed constraints require both atoms and r0 datasets");
        }
        HighFive::DataSet atoms_dataset = file.getDataSet(atoms_path);
        const auto atom_dimensions = atoms_dataset.getSpace().getDimensions();
        if (atom_dimensions.size() != 2 || atom_dimensions[1] != 2 ||
            atom_dimensions[0] == 0)
        {
            throw std::runtime_error(
                "/constraint/default/pairs/atoms must have shape [n,2] with "
                "n > 0");
        }
        HighFive::DataSet distance_dataset = file.getDataSet(distance_path);
        const auto distance_dimensions =
            distance_dataset.getSpace().getDimensions();
        if (distance_dimensions.size() != 1 ||
            distance_dimensions[0] != atom_dimensions[0])
        {
            throw std::runtime_error(
                "/constraint/default/pairs/r0 must have shape [n] matching "
                "the atoms dataset");
        }

        std::vector<std::array<std::int64_t, 2>> atoms;
        std::vector<float> distances;
        atoms_dataset.read(atoms);
        distance_dataset.read(distances);
        Xponge::DistanceConstraints result;
        result.atom_a.reserve(distances.size());
        result.atom_b.reserve(distances.size());
        result.r0.reserve(distances.size());
        for (std::size_t pair = 0; pair < distances.size(); ++pair)
        {
            const std::int64_t atom_a = atoms[pair][0];
            const std::int64_t atom_b = atoms[pair][1];
            if (atom_a < 0 || atom_b < 0 ||
                atom_a >= static_cast<std::int64_t>(atom_count) ||
                atom_b >= static_cast<std::int64_t>(atom_count) ||
                atom_a == atom_b)
            {
                std::ostringstream message;
                message << atoms_path << " contains invalid pair [" << atom_a
                        << ", " << atom_b << "] at row " << pair;
                throw std::runtime_error(message.str());
            }
            if (!SpongeFloat::Is_Finite(distances[pair]) ||
                distances[pair] <= 0.0f)
            {
                std::ostringstream message;
                message << distance_path
                        << " contains a non-positive or non-finite distance at "
                           "row "
                        << pair;
                throw std::runtime_error(message.str());
            }
            result.atom_a.push_back(static_cast<int>(atom_a));
            result.atom_b.push_back(static_cast<int>(atom_b));
            result.r0.push_back(distances[pair]);
        }
        *constraints = result;
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed constraints from ") +
            atoms_path + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Xponge::Read_H5_Distance_Constraints",
                                       message.c_str());
    }
    return false;
}
}  // namespace

void Xponge::System::Load_Inputs(CONTROLLER* controller)
{
    Validate_H5_Topology_Schema_Version(controller);
    if (controller->Command_Exist("gromacs_top") ||
        controller->Command_Exist("gromacs_gro"))
    {
        Load_Gromacs_Inputs(this, controller);
    }
    else if (controller->Command_Exist("amber_parm7") ||
             controller->Command_Exist("amber_rst7"))
    {
        Load_Amber_Inputs(this, controller);
    }
    else
    {
        Load_Native_Inputs(this, controller);
        Load_H5_Native_Positional_Restraint(controller, this->atoms.mass.size(),
                                            &this->positional_restraint);
        Materialize_H5_Native_NB14_Extra(controller, this);
        const auto residue_atom_numbers =
            Read_H5_Residue_Atom_Numbers(controller, this->atoms.mass.size());
        if (!residue_atom_numbers.empty())
        {
            this->residues.atom_numbers = residue_atom_numbers;
        }
        Xponge::DistanceConstraints constraints;
        if (Read_H5_Distance_Constraints(controller, this->atoms.mass.size(),
                                         &constraints))
        {
            this->classical_force_field.constraints = constraints;
        }
    }
}
