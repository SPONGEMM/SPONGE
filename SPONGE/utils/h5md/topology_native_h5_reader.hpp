#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <highfive/highfive.hpp>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace SpongeH5MD
{
struct NativeTopologyExclusionsState
{
    std::vector<std::vector<int>> excluded_atoms;
};

struct NativeTopologyBondState
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<float> k;
    std::vector<float> r0;
};

struct NativeTopologyAngleState
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<float> k;
    std::vector<float> theta0;
};

struct NativeTopologyTorsionState
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<int> atom_d;
    std::vector<float> pk;
    std::vector<float> pn;
    std::vector<int> ipn;
    std::vector<float> gamc;
    std::vector<float> gams;
};

struct NativeTopologyNB14State
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> cf_scale_factor;
};

struct NativeTopologyGBState
{
    std::vector<float> radius;
    std::vector<float> scale_factor;
};

struct NativeTopologyVirtualAtomRecord
{
    int type = -1;
    int virtual_atom = -1;
    std::vector<int> from;
    std::vector<float> parameter;
};

struct NativeTopologyVirtualAtomsState
{
    std::vector<NativeTopologyVirtualAtomRecord> records;
};

struct NativeTopologyUreyBradleyState
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<float> angle_k;
    std::vector<float> angle_theta0;
    std::vector<float> bond_k;
    std::vector<float> bond_r0;
};

struct NativeTopologyCMapState
{
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<int> atom_c;
    std::vector<int> atom_d;
    std::vector<int> atom_e;
    std::vector<int> cmap_type;
    std::vector<int> resolution;
    std::vector<float> grid_value;
    std::vector<float> interpolation_coeff;
    std::vector<int> type_offset;
    int unique_type_numbers = 0;
    int unique_gridpoint_numbers = 0;
};

struct NativeTopologyLJSoftCoreState
{
    int atom_numbers = 0;
    int atom_type_numbers_A = 0;
    int atom_type_numbers_B = 0;
    std::vector<float> LJ_AA;
    std::vector<float> LJ_AB;
    std::vector<float> LJ_BA;
    std::vector<float> LJ_BB;
    std::vector<int> atom_LJ_type_A;
    std::vector<int> atom_LJ_type_B;
    std::vector<int> subsystem_division;
};

struct NativeTopologyLJState
{
    std::vector<int> atom_type;
    std::vector<float> pair_A;
    std::vector<float> pair_B;
    int atom_type_numbers = 0;
};

struct NativeTopologyCoreState
{
    std::int64_t atom_count = 0;
    bool has_mass = false;
    bool has_charge = false;
    bool has_exclusions = false;
    bool has_bonds = false;
    bool has_angles = false;
    bool has_dihedrals = false;
    bool has_impropers = false;
    bool has_lj = false;
    bool has_nb14 = false;
    bool has_gb = false;
    bool has_virtual_atoms = false;
    bool has_urey_bradley = false;
    bool has_cmap = false;
    bool has_lj_soft_core = false;
    std::vector<float> mass;
    std::vector<float> charge;
    NativeTopologyExclusionsState exclusions;
    NativeTopologyBondState bonds;
    NativeTopologyAngleState angles;
    NativeTopologyTorsionState dihedrals;
    NativeTopologyTorsionState impropers;
    NativeTopologyLJState lj;
    NativeTopologyNB14State nb14;
    NativeTopologyGBState gb;
    NativeTopologyVirtualAtomsState virtual_atoms;
    NativeTopologyUreyBradleyState urey_bradley;
    NativeTopologyCMapState cmap;
    NativeTopologyLJSoftCoreState lj_soft_core;
};

class TopologyNativeH5Reader
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open topology H5 file: ") +
                        err.what());
        }
    }

    bool Read_Core_State(NativeTopologyCoreState* state)
    {
        if (state == nullptr)
        {
            return Fail("native topology core state output pointer is null");
        }
        if (!Ensure_File()) return false;

        NativeTopologyCoreState result;
        try
        {
            result.atom_count = Read_Optional_Int64("/topology/atom_count");
            if (Exists("/atoms/mass"))
            {
                result.mass = Read_Float_Vector("/atoms/mass", "atom mass");
                result.has_mass = true;
                if (!Validate_Positive_Mass(result.mass)) return false;
            }
            if (Exists("/atoms/charge"))
            {
                result.charge =
                    Read_Float_Vector("/atoms/charge", "atom charge");
                result.has_charge = true;
            }
            if (!Read_Optional_Exclusions(&result)) return false;
            if (!Read_Optional_LJ(&result)) return false;
            if (!Read_Optional_Bonds(&result)) return false;
            if (!Read_Optional_Angles(&result)) return false;
            if (!Read_Optional_Dihedrals(&result)) return false;
            if (!Read_Optional_Impropers(&result)) return false;
            if (!Read_Optional_NB14(&result)) return false;
            if (!Read_Optional_GB(&result)) return false;
            if (!Read_Optional_Virtual_Atoms(&result)) return false;
            if (!Read_Optional_Urey_Bradley(&result)) return false;
            if (!Read_Optional_CMap(&result)) return false;
            if (!Read_Optional_LJ_Soft_Core(&result)) return false;
            if (!Validate_Lengths(result)) return false;
            if (!Validate_Atom_Indices(result)) return false;
            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read native topology H5 core "
                                    "state: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::int64_t Read_Optional_Int64(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            return 0;
        }
        std::int64_t value = 0;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    int Read_Optional_Int(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            return 0;
        }
        int value = 0;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    std::vector<float> Read_Float_Vector(const std::string& dataset_path,
                                         const std::string& label)
    {
        const auto dataset = file_->getDataSet(dataset_path);
        const auto dimensions = dataset.getSpace().getDimensions();
        if (dimensions.size() != 1)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path
                << " must be one-dimensional";
            throw std::runtime_error(out.str());
        }
        std::vector<float> values;
        dataset.read(values);
        return values;
    }

    std::vector<int> Read_Int_Vector(const std::string& dataset_path,
                                     const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 1)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path
                << " must be one-dimensional";
            throw std::runtime_error(out.str());
        }
        return Read_Selection<int>(dataset_path, std::vector<std::size_t>{0},
                                   dims, label);
    }

    std::vector<std::int64_t> Read_Int64_Vector(const std::string& dataset_path,
                                                const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 1)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path
                << " must be one-dimensional";
            throw std::runtime_error(out.str());
        }
        return Read_Selection<std::int64_t>(
            dataset_path, std::vector<std::size_t>{0}, dims, label);
    }

    template <typename T>
    std::vector<T> Read_Matrix(const std::string& dataset_path,
                               std::size_t columns, const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2 || dims[1] != columns)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path << " must have shape "
                << "[n," << columns << "]";
            throw std::runtime_error(out.str());
        }
        return Read_Selection<T>(dataset_path, std::vector<std::size_t>{0, 0},
                                 dims, label);
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    template <typename T>
    std::vector<T> Read_Selection(const std::string& dataset_path,
                                  const std::vector<std::size_t>& offsets,
                                  const std::vector<std::size_t>& counts,
                                  const std::string& label)
    {
        const std::size_t value_count = Product(counts);
        std::vector<T> values(value_count);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const std::vector<hsize_t> h_offsets = To_HSize(offsets);
        const std::vector<hsize_t> h_counts = To_HSize(counts);
        hid_t file_space = H5Dget_space(dataset.getId());
        if (file_space < 0)
        {
            throw std::runtime_error(label + " failed to get dataspace at " +
                                     dataset_path);
        }
        const herr_t select_rc =
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offsets.data(),
                                nullptr, h_counts.data(), nullptr);
        if (select_rc < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label + " failed to select hyperslab at " +
                                     dataset_path);
        }
        hid_t mem_space = H5Screate_simple(static_cast<int>(h_counts.size()),
                                           h_counts.data(), nullptr);
        if (mem_space < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), Native_H5_Type<T>(), mem_space, file_space,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        H5Sclose(file_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read dataset at " +
                                     dataset_path);
        }
        return values;
    }

    bool Read_Optional_Exclusions(NativeTopologyCoreState* state)
    {
        const bool has_offset = Exists("/topology/exclusions/offset");
        const bool has_list = Exists("/topology/exclusions/list");
        if (!has_offset && !has_list)
        {
            return true;
        }
        if (!has_offset || !has_list)
        {
            return Fail(
                "native topology exclusions require both "
                "/topology/exclusions/offset and "
                "/topology/exclusions/list");
        }
        const auto offsets = Read_Int64_Vector("/topology/exclusions/offset",
                                               "topology exclusions offset");
        const auto list = Read_Int_Vector("/topology/exclusions/list",
                                          "topology exclusions list");
        if (offsets.empty() || offsets[0] != 0)
        {
            return Fail("/topology/exclusions/offset must start with 0");
        }
        if (offsets.back() != static_cast<std::int64_t>(list.size()))
        {
            return Fail(
                "/topology/exclusions/offset final value must match "
                "/topology/exclusions/list length");
        }
        for (std::size_t i = 1; i < offsets.size(); ++i)
        {
            if (offsets[i] < offsets[i - 1])
            {
                return Fail("/topology/exclusions/offset must be monotonic");
            }
        }
        if (state->atom_count == 0)
        {
            state->atom_count = static_cast<std::int64_t>(offsets.size() - 1);
        }
        if (state->atom_count != static_cast<std::int64_t>(offsets.size() - 1))
        {
            return Fail(
                "/topology/exclusions/offset length does not match "
                "/topology/atom_count");
        }
        state->exclusions.excluded_atoms.assign(
            static_cast<std::size_t>(state->atom_count), {});
        for (std::size_t atom = 0;
             atom < state->exclusions.excluded_atoms.size(); ++atom)
        {
            const auto begin = static_cast<std::size_t>(offsets[atom]);
            const auto end = static_cast<std::size_t>(offsets[atom + 1]);
            state->exclusions.excluded_atoms[atom].assign(list.begin() + begin,
                                                          list.begin() + end);
        }
        state->has_exclusions = true;
        return true;
    }

    bool Read_Optional_LJ(NativeTopologyCoreState* state)
    {
        const bool has_type = Exists("/forcefield/lj/type");
        const bool has_params = Exists("/forcefield/lj/params");
        const bool has_pair_a = Exists("/forcefield/lj/pair_A_12");
        const bool has_pair_b = Exists("/forcefield/lj/pair_B_6");
        if (!has_type && !has_params && !has_pair_a && !has_pair_b)
        {
            return true;
        }
        if (!has_type)
        {
            return Fail("native LJ parameters require /forcefield/lj/type");
        }
        if (!has_params && (!has_pair_a || !has_pair_b))
        {
            return Fail(
                "native LJ parameters require either "
                "/forcefield/lj/params or both "
                "/forcefield/lj/pair_A_12 and /forcefield/lj/pair_B_6");
        }
        state->lj.atom_type =
            Read_Int_Vector("/forcefield/lj/type", "LJ atom type");
        state->lj.atom_type_numbers =
            Read_Optional_Int("/forcefield/lj/atom_type_count");
        if (state->lj.atom_type_numbers <= 0 && !state->lj.atom_type.empty())
        {
            state->lj.atom_type_numbers =
                *std::max_element(state->lj.atom_type.begin(),
                                  state->lj.atom_type.end()) +
                1;
        }
        if (state->lj.atom_type_numbers <= 0)
        {
            return Fail(
                "/forcefield/lj/atom_type_count must be positive when "
                "LJ params are present");
        }
        const std::size_t pair_count =
            static_cast<std::size_t>(state->lj.atom_type_numbers) *
            static_cast<std::size_t>(state->lj.atom_type_numbers + 1) / 2;
        if (has_params)
        {
            const auto params =
                Read_Matrix<float>("/forcefield/lj/params", 2, "LJ params");
            if (params.size() != pair_count * 2)
            {
                return Fail(
                    "/forcefield/lj/params row count does not match "
                    "/forcefield/lj/atom_type_count");
            }
            state->lj.pair_A.resize(pair_count);
            state->lj.pair_B.resize(pair_count);
            for (std::size_t i = 0; i < pair_count; ++i)
            {
                state->lj.pair_A[i] = params[2 * i];
                state->lj.pair_B[i] = params[2 * i + 1];
            }
        }
        else
        {
            state->lj.pair_A =
                Read_Float_Vector("/forcefield/lj/pair_A_12", "LJ pair A");
            state->lj.pair_B =
                Read_Float_Vector("/forcefield/lj/pair_B_6", "LJ pair B");
            if (state->lj.pair_A.size() != pair_count ||
                state->lj.pair_B.size() != pair_count)
            {
                return Fail(
                    "native LJ pair_A_12/pair_B_6 lengths do not match "
                    "/forcefield/lj/atom_type_count");
            }
        }
        if (state->atom_count == 0)
        {
            state->atom_count =
                static_cast<std::int64_t>(state->lj.atom_type.size());
        }
        state->has_lj = true;
        return true;
    }

    bool Read_Optional_Bonds(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/bond/atoms"))
        {
            return true;
        }
        if (!Exists("/forcefield/bond/k") || !Exists("/forcefield/bond/r0"))
        {
            return Fail(
                "native bond parameters require atoms, k, and r0 "
                "datasets");
        }
        const auto atoms =
            Read_Matrix<int>("/forcefield/bond/atoms", 2, "bond atoms");
        state->bonds.k = Read_Float_Vector("/forcefield/bond/k", "bond k");
        state->bonds.r0 = Read_Float_Vector("/forcefield/bond/r0", "bond r0");
        const std::size_t bond_count = atoms.size() / 2;
        if (state->bonds.k.size() != bond_count ||
            state->bonds.r0.size() != bond_count)
        {
            return Fail(
                "native bond k/r0 lengths must match "
                "/forcefield/bond/atoms row count");
        }
        state->bonds.atom_a.resize(bond_count);
        state->bonds.atom_b.resize(bond_count);
        for (std::size_t i = 0; i < bond_count; ++i)
        {
            state->bonds.atom_a[i] = atoms[2 * i];
            state->bonds.atom_b[i] = atoms[2 * i + 1];
        }
        state->has_bonds = true;
        return true;
    }

    bool Read_Optional_Angles(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/angle/atoms"))
        {
            return true;
        }
        if (!Exists("/forcefield/angle/k") ||
            !Exists("/forcefield/angle/theta0"))
        {
            return Fail(
                "native angle parameters require atoms, k, and theta0 "
                "datasets");
        }
        const auto atoms =
            Read_Matrix<int>("/forcefield/angle/atoms", 3, "angle atoms");
        state->angles.k = Read_Float_Vector("/forcefield/angle/k", "angle k");
        state->angles.theta0 =
            Read_Float_Vector("/forcefield/angle/theta0", "angle theta0");
        const std::size_t angle_count = atoms.size() / 3;
        if (state->angles.k.size() != angle_count ||
            state->angles.theta0.size() != angle_count)
        {
            return Fail(
                "native angle k/theta0 lengths must match "
                "/forcefield/angle/atoms row count");
        }
        state->angles.atom_a.resize(angle_count);
        state->angles.atom_b.resize(angle_count);
        state->angles.atom_c.resize(angle_count);
        for (std::size_t i = 0; i < angle_count; ++i)
        {
            state->angles.atom_a[i] = atoms[3 * i];
            state->angles.atom_b[i] = atoms[3 * i + 1];
            state->angles.atom_c[i] = atoms[3 * i + 2];
        }
        state->has_angles = true;
        return true;
    }

    bool Read_Torsion_Atoms_And_Runtime_State(
        const std::string& group_path, NativeTopologyTorsionState* torsions,
        const std::string& label)
    {
        const auto atoms =
            Read_Matrix<int>(group_path + "/atoms", 4, label + " atoms");
        torsions->pk = Read_Float_Vector(group_path + "/pk", label + " pk");
        torsions->pn = Read_Float_Vector(group_path + "/pn", label + " pn");
        torsions->ipn = Read_Int_Vector(group_path + "/ipn", label + " ipn");
        torsions->gamc =
            Read_Float_Vector(group_path + "/gamc", label + " gamc");
        torsions->gams =
            Read_Float_Vector(group_path + "/gams", label + " gams");
        const std::size_t torsion_count = atoms.size() / 4;
        if (torsions->pk.size() != torsion_count ||
            torsions->pn.size() != torsion_count ||
            torsions->ipn.size() != torsion_count ||
            torsions->gamc.size() != torsion_count ||
            torsions->gams.size() != torsion_count)
        {
            return Fail("native " + label +
                        " runtime parameter lengths must match atoms row "
                        "count");
        }
        torsions->atom_a.resize(torsion_count);
        torsions->atom_b.resize(torsion_count);
        torsions->atom_c.resize(torsion_count);
        torsions->atom_d.resize(torsion_count);
        for (std::size_t i = 0; i < torsion_count; ++i)
        {
            torsions->atom_a[i] = atoms[4 * i];
            torsions->atom_b[i] = atoms[4 * i + 1];
            torsions->atom_c[i] = atoms[4 * i + 2];
            torsions->atom_d[i] = atoms[4 * i + 3];
        }
        return true;
    }

    bool Read_Optional_Dihedrals(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/dihedral/atoms"))
        {
            return true;
        }
        const bool has_runtime = Exists("/forcefield/dihedral/pk") &&
                                 Exists("/forcefield/dihedral/pn") &&
                                 Exists("/forcefield/dihedral/ipn") &&
                                 Exists("/forcefield/dihedral/gamc") &&
                                 Exists("/forcefield/dihedral/gams");
        const bool has_legacy = Exists("/forcefield/dihedral/k") &&
                                Exists("/forcefield/dihedral/periodicity") &&
                                Exists("/forcefield/dihedral/phi0");
        if (!has_runtime && !has_legacy)
        {
            return Fail(
                "native dihedral parameters require either runtime "
                "pk/pn/ipn/gamc/gams datasets or legacy "
                "k/periodicity/phi0 datasets");
        }
        if (has_runtime)
        {
            if (!Read_Torsion_Atoms_And_Runtime_State(
                    "/forcefield/dihedral", &state->dihedrals, "dihedral"))
            {
                return false;
            }
        }
        else
        {
            const auto atoms = Read_Matrix<int>("/forcefield/dihedral/atoms", 4,
                                                "dihedral atoms");
            state->dihedrals.pk =
                Read_Float_Vector("/forcefield/dihedral/k", "dihedral k");
            state->dihedrals.ipn = Read_Int_Vector(
                "/forcefield/dihedral/periodicity", "dihedral periodicity");
            const auto phi0 =
                Read_Float_Vector("/forcefield/dihedral/phi0", "dihedral phi0");
            const std::size_t torsion_count = atoms.size() / 4;
            if (state->dihedrals.pk.size() != torsion_count ||
                state->dihedrals.ipn.size() != torsion_count ||
                phi0.size() != torsion_count)
            {
                return Fail(
                    "native dihedral k/periodicity/phi0 lengths must "
                    "match atoms row count");
            }
            state->dihedrals.pn.resize(torsion_count);
            state->dihedrals.gamc.resize(torsion_count);
            state->dihedrals.gams.resize(torsion_count);
            state->dihedrals.atom_a.resize(torsion_count);
            state->dihedrals.atom_b.resize(torsion_count);
            state->dihedrals.atom_c.resize(torsion_count);
            state->dihedrals.atom_d.resize(torsion_count);
            for (std::size_t i = 0; i < torsion_count; ++i)
            {
                state->dihedrals.atom_a[i] = atoms[4 * i];
                state->dihedrals.atom_b[i] = atoms[4 * i + 1];
                state->dihedrals.atom_c[i] = atoms[4 * i + 2];
                state->dihedrals.atom_d[i] = atoms[4 * i + 3];
                state->dihedrals.pn[i] =
                    static_cast<float>(state->dihedrals.ipn[i]);
                state->dihedrals.gamc[i] =
                    std::cos(phi0[i]) * state->dihedrals.pk[i];
                state->dihedrals.gams[i] =
                    std::sin(phi0[i]) * state->dihedrals.pk[i];
            }
        }
        state->has_dihedrals = true;
        return true;
    }

    bool Read_Optional_Impropers(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/improper/atoms"))
        {
            return true;
        }
        if (!Exists("/forcefield/improper/pk") ||
            !Exists("/forcefield/improper/phi0"))
        {
            return Fail(
                "native improper parameters require atoms, pk, and "
                "phi0 datasets");
        }
        const auto atoms =
            Read_Matrix<int>("/forcefield/improper/atoms", 4, "improper atoms");
        state->impropers.pk =
            Read_Float_Vector("/forcefield/improper/pk", "improper pk");
        state->impropers.gamc =
            Read_Float_Vector("/forcefield/improper/phi0", "improper phi0");
        const std::size_t improper_count = atoms.size() / 4;
        if (state->impropers.pk.size() != improper_count ||
            state->impropers.gamc.size() != improper_count)
        {
            return Fail(
                "native improper pk/phi0 lengths must match "
                "/forcefield/improper/atoms row count");
        }
        state->impropers.atom_a.resize(improper_count);
        state->impropers.atom_b.resize(improper_count);
        state->impropers.atom_c.resize(improper_count);
        state->impropers.atom_d.resize(improper_count);
        state->impropers.pn.assign(improper_count, 0.0f);
        state->impropers.ipn.assign(improper_count, 0);
        state->impropers.gams.assign(improper_count, 0.0f);
        for (std::size_t i = 0; i < improper_count; ++i)
        {
            state->impropers.atom_a[i] = atoms[4 * i];
            state->impropers.atom_b[i] = atoms[4 * i + 1];
            state->impropers.atom_c[i] = atoms[4 * i + 2];
            state->impropers.atom_d[i] = atoms[4 * i + 3];
        }
        state->has_impropers = true;
        return true;
    }

    bool Read_Optional_NB14(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/nb14/atoms"))
        {
            return true;
        }
        if (!Exists("/forcefield/nb14/params"))
        {
            return Fail("native nb14 parameters require atoms and params");
        }
        const auto atoms =
            Read_Matrix<int>("/forcefield/nb14/atoms", 2, "nb14 atoms");
        const auto param_dims = Dimensions("/forcefield/nb14/params");
        if (param_dims.size() != 2 ||
            (param_dims[1] != 2 && param_dims[1] != 3))
        {
            std::ostringstream out;
            out << "nb14 params dataset /forcefield/nb14/params must have "
                   "shape [n,2] for legacy lj/cf scale factors or [n,3] "
                   "for materialized A/B/cf parameters";
            throw std::runtime_error(out.str());
        }
        const std::size_t param_columns = param_dims[1];
        const auto params = Read_Selection<float>(
            "/forcefield/nb14/params", std::vector<std::size_t>{0, 0},
            param_dims, "nb14 params");
        const std::size_t nb14_count = atoms.size() / 2;
        if (params.size() != nb14_count * param_columns)
        {
            return Fail(
                "native nb14 params row count must match atoms row count");
        }
        if (param_columns == 2 &&
            (!state->has_lj || state->lj.atom_type.empty() ||
             state->lj.pair_A.empty() || state->lj.pair_B.empty()))
        {
            return Fail(
                "native nb14 legacy scale params require initialized "
                "native LJ atom types and pair parameters");
        }
        state->nb14.atom_a.resize(nb14_count);
        state->nb14.atom_b.resize(nb14_count);
        state->nb14.A.resize(nb14_count);
        state->nb14.B.resize(nb14_count);
        state->nb14.cf_scale_factor.resize(nb14_count);
        for (std::size_t i = 0; i < nb14_count; ++i)
        {
            state->nb14.atom_a[i] = atoms[2 * i];
            state->nb14.atom_b[i] = atoms[2 * i + 1];
            if (param_columns == 2)
            {
                const int atom_a = state->nb14.atom_a[i];
                const int atom_b = state->nb14.atom_b[i];
                if (atom_a < 0 || atom_b < 0 ||
                    static_cast<std::size_t>(atom_a) >=
                        state->lj.atom_type.size() ||
                    static_cast<std::size_t>(atom_b) >=
                        state->lj.atom_type.size())
                {
                    return Fail(
                        "native nb14 atoms must index initialized LJ "
                        "atom types when using legacy scale params");
                }
                int small_type = state->lj.atom_type[atom_a];
                int large_type = state->lj.atom_type[atom_b];
                if (small_type > large_type)
                {
                    std::swap(small_type, large_type);
                }
                const std::size_t pair_type =
                    static_cast<std::size_t>(large_type) *
                        static_cast<std::size_t>(large_type + 1) / 2 +
                    static_cast<std::size_t>(small_type);
                if (small_type < 0 || large_type < 0 ||
                    pair_type >= state->lj.pair_A.size() ||
                    pair_type >= state->lj.pair_B.size())
                {
                    return Fail(
                        "native nb14 legacy scale params reference an "
                        "LJ pair type outside native LJ parameters");
                }
                const float lj_scale_factor = params[param_columns * i];
                state->nb14.A[i] =
                    lj_scale_factor * state->lj.pair_A[pair_type] * 12.0f;
                state->nb14.B[i] =
                    lj_scale_factor * state->lj.pair_B[pair_type] * 6.0f;
                state->nb14.cf_scale_factor[i] = params[param_columns * i + 1];
            }
            else
            {
                state->nb14.A[i] = params[param_columns * i];
                state->nb14.B[i] = params[param_columns * i + 1];
                state->nb14.cf_scale_factor[i] = params[param_columns * i + 2];
            }
        }
        state->has_nb14 = true;
        return true;
    }

    bool Read_Optional_GB(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/gb/params"))
        {
            return true;
        }
        const auto params =
            Read_Matrix<float>("/forcefield/gb/params", 2, "GB params");
        const std::size_t atom_count = params.size() / 2;
        state->gb.radius.resize(atom_count);
        state->gb.scale_factor.resize(atom_count);
        for (std::size_t i = 0; i < atom_count; ++i)
        {
            state->gb.radius[i] = params[2 * i];
            state->gb.scale_factor[i] = params[2 * i + 1];
        }
        if (state->atom_count == 0)
        {
            state->atom_count = static_cast<std::int64_t>(atom_count);
        }
        state->has_gb = true;
        return true;
    }

    bool Read_Optional_Virtual_Atoms(NativeTopologyCoreState* state)
    {
        const bool has_type = Exists("/forcefield/virtual_atom/type");
        const bool has_atom = Exists("/forcefield/virtual_atom/atom");
        const bool has_from_offset =
            Exists("/forcefield/virtual_atom/from_offset");
        const bool has_from = Exists("/forcefield/virtual_atom/from");
        const bool has_parameter_offset =
            Exists("/forcefield/virtual_atom/parameter_offset");
        const bool has_parameter = Exists("/forcefield/virtual_atom/parameter");
        if (!has_type && !has_atom && !has_from_offset && !has_from &&
            !has_parameter_offset && !has_parameter)
        {
            return true;
        }
        if (!has_type || !has_atom || !has_from_offset || !has_from ||
            !has_parameter_offset || !has_parameter)
        {
            return Fail(
                "native virtual atom records require type, atom, "
                "from_offset, from, parameter_offset, and parameter "
                "datasets");
        }
        const auto type = Read_Int_Vector("/forcefield/virtual_atom/type",
                                          "virtual atom type");
        const auto atom =
            Read_Int_Vector("/forcefield/virtual_atom/atom", "virtual atom");
        const auto from_offset = Read_Int64_Vector(
            "/forcefield/virtual_atom/from_offset", "virtual atom from offset");
        const auto from = Read_Int_Vector("/forcefield/virtual_atom/from",
                                          "virtual atom from");
        const auto parameter_offset =
            Read_Int64_Vector("/forcefield/virtual_atom/parameter_offset",
                              "virtual atom parameter offset");
        const auto parameter = Read_Float_Vector(
            "/forcefield/virtual_atom/parameter", "virtual atom parameter");
        if (type.size() != atom.size())
        {
            return Fail("native virtual atom type and atom lengths must match");
        }
        if (from_offset.size() != type.size() + 1 ||
            parameter_offset.size() != type.size() + 1)
        {
            return Fail(
                "native virtual atom offset lengths must be record_count "
                "+ 1");
        }
        if (from_offset.empty() || from_offset[0] != 0 ||
            from_offset.back() != static_cast<std::int64_t>(from.size()))
        {
            return Fail(
                "native virtual atom from offsets must start at 0 and "
                "end at from length");
        }
        if (parameter_offset.empty() || parameter_offset[0] != 0 ||
            parameter_offset.back() !=
                static_cast<std::int64_t>(parameter.size()))
        {
            return Fail(
                "native virtual atom parameter offsets must start at 0 "
                "and end at parameter length");
        }
        state->virtual_atoms.records.resize(type.size());
        for (std::size_t i = 0; i < type.size(); ++i)
        {
            if (from_offset[i + 1] < from_offset[i] ||
                parameter_offset[i + 1] < parameter_offset[i])
            {
                return Fail("native virtual atom offsets must be monotonic");
            }
            const std::size_t from_begin =
                static_cast<std::size_t>(from_offset[i]);
            const std::size_t from_end =
                static_cast<std::size_t>(from_offset[i + 1]);
            const std::size_t parameter_begin =
                static_cast<std::size_t>(parameter_offset[i]);
            const std::size_t parameter_end =
                static_cast<std::size_t>(parameter_offset[i + 1]);
            NativeTopologyVirtualAtomRecord& record =
                state->virtual_atoms.records[i];
            record.type = type[i];
            record.virtual_atom = atom[i];
            record.from.assign(from.begin() + from_begin,
                               from.begin() + from_end);
            record.parameter.assign(parameter.begin() + parameter_begin,
                                    parameter.begin() + parameter_end);
            if (!Validate_Virtual_Atom_Arity(record))
            {
                return false;
            }
        }
        state->has_virtual_atoms = true;
        return true;
    }

    bool Read_Optional_Urey_Bradley(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/urey_bradley/atoms"))
        {
            return true;
        }
        if (!Exists("/forcefield/urey_bradley/angle_k") ||
            !Exists("/forcefield/urey_bradley/angle_theta0") ||
            !Exists("/forcefield/urey_bradley/bond_k") ||
            !Exists("/forcefield/urey_bradley/bond_r0"))
        {
            return Fail(
                "native urey-bradley parameters require atoms, "
                "angle_k, angle_theta0, bond_k, and bond_r0 datasets");
        }
        const auto atoms = Read_Matrix<int>("/forcefield/urey_bradley/atoms", 3,
                                            "urey-bradley atoms");
        state->urey_bradley.angle_k = Read_Float_Vector(
            "/forcefield/urey_bradley/angle_k", "urey-bradley angle k");
        state->urey_bradley.angle_theta0 =
            Read_Float_Vector("/forcefield/urey_bradley/angle_theta0",
                              "urey-bradley angle theta0");
        state->urey_bradley.bond_k = Read_Float_Vector(
            "/forcefield/urey_bradley/bond_k", "urey-bradley bond k");
        state->urey_bradley.bond_r0 = Read_Float_Vector(
            "/forcefield/urey_bradley/bond_r0", "urey-bradley bond r0");
        const std::size_t term_count = atoms.size() / 3;
        if (state->urey_bradley.angle_k.size() != term_count ||
            state->urey_bradley.angle_theta0.size() != term_count ||
            state->urey_bradley.bond_k.size() != term_count ||
            state->urey_bradley.bond_r0.size() != term_count)
        {
            return Fail(
                "native urey-bradley parameter lengths must match "
                "/forcefield/urey_bradley/atoms row count");
        }
        state->urey_bradley.atom_a.resize(term_count);
        state->urey_bradley.atom_b.resize(term_count);
        state->urey_bradley.atom_c.resize(term_count);
        for (std::size_t i = 0; i < term_count; ++i)
        {
            state->urey_bradley.atom_a[i] = atoms[3 * i];
            state->urey_bradley.atom_b[i] = atoms[3 * i + 1];
            state->urey_bradley.atom_c[i] = atoms[3 * i + 2];
        }
        state->has_urey_bradley = true;
        return true;
    }

    bool Read_Optional_CMap(NativeTopologyCoreState* state)
    {
        const bool has_atoms = Exists("/forcefield/cmap/atoms");
        const bool has_type = Exists("/forcefield/cmap/type");
        const bool has_resolution = Exists("/forcefield/cmap/resolution");
        const bool has_grid_value = Exists("/forcefield/cmap/grid_value");
        if (!has_atoms && !has_type && !has_resolution && !has_grid_value)
        {
            return true;
        }
        if (!has_atoms || !has_type || !has_resolution || !has_grid_value)
        {
            return Fail(
                "native CMAP parameters require atoms, type, "
                "resolution, and grid_value datasets");
        }
        const auto atoms =
            Read_Matrix<int>("/forcefield/cmap/atoms", 5, "cmap atoms");
        state->cmap.cmap_type =
            Read_Int_Vector("/forcefield/cmap/type", "cmap type");
        state->cmap.resolution =
            Read_Int_Vector("/forcefield/cmap/resolution", "cmap resolution");
        state->cmap.grid_value =
            Read_Float_Vector("/forcefield/cmap/grid_value", "cmap grid value");
        const std::size_t cmap_count = atoms.size() / 5;
        if (state->cmap.cmap_type.size() != cmap_count)
        {
            return Fail(
                "native cmap type length must match "
                "/forcefield/cmap/atoms row count");
        }
        state->cmap.unique_type_numbers =
            static_cast<int>(state->cmap.resolution.size());
        state->cmap.type_offset.resize(state->cmap.resolution.size());
        state->cmap.unique_gridpoint_numbers = 0;
        for (std::size_t i = 0; i < state->cmap.resolution.size(); ++i)
        {
            if (state->cmap.resolution[i] <= 0)
            {
                return Fail("native cmap resolution values must be positive");
            }
            state->cmap.type_offset[i] =
                16 * state->cmap.unique_gridpoint_numbers;
            state->cmap.unique_gridpoint_numbers +=
                state->cmap.resolution[i] * state->cmap.resolution[i];
        }
        if (state->cmap.grid_value.size() !=
            static_cast<std::size_t>(state->cmap.unique_gridpoint_numbers))
        {
            return Fail(
                "native cmap grid_value length does not match "
                "resolution-derived gridpoint count");
        }
        state->cmap.atom_a.resize(cmap_count);
        state->cmap.atom_b.resize(cmap_count);
        state->cmap.atom_c.resize(cmap_count);
        state->cmap.atom_d.resize(cmap_count);
        state->cmap.atom_e.resize(cmap_count);
        for (std::size_t i = 0; i < cmap_count; ++i)
        {
            state->cmap.atom_a[i] = atoms[5 * i];
            state->cmap.atom_b[i] = atoms[5 * i + 1];
            state->cmap.atom_c[i] = atoms[5 * i + 2];
            state->cmap.atom_d[i] = atoms[5 * i + 3];
            state->cmap.atom_e[i] = atoms[5 * i + 4];
        }
        state->has_cmap = true;
        return true;
    }

    bool Read_Optional_LJ_Soft_Core(NativeTopologyCoreState* state)
    {
        if (!Exists("/forcefield/lj_soft_core/atom_type_A") &&
            !Exists("/forcefield/lj_soft_core/atom_type_B"))
        {
            return true;
        }
        if (!Exists("/forcefield/lj_soft_core/atom_type_A") ||
            !Exists("/forcefield/lj_soft_core/atom_type_B") ||
            !Exists("/forcefield/lj_soft_core/pair_AA") ||
            !Exists("/forcefield/lj_soft_core/pair_AB") ||
            !Exists("/forcefield/lj_soft_core/pair_BA") ||
            !Exists("/forcefield/lj_soft_core/pair_BB"))
        {
            return Fail(
                "native LJ soft-core parameters require atom_type_A, "
                "atom_type_B, pair_AA, pair_AB, pair_BA, and pair_BB "
                "datasets");
        }
        NativeTopologyLJSoftCoreState& lj_soft = state->lj_soft_core;
        lj_soft.atom_LJ_type_A = Read_Int_Vector(
            "/forcefield/lj_soft_core/atom_type_A", "LJ soft atom type A");
        lj_soft.atom_LJ_type_B = Read_Int_Vector(
            "/forcefield/lj_soft_core/atom_type_B", "LJ soft atom type B");
        lj_soft.LJ_AA = Read_Float_Vector("/forcefield/lj_soft_core/pair_AA",
                                          "LJ soft pair AA");
        lj_soft.LJ_AB = Read_Float_Vector("/forcefield/lj_soft_core/pair_AB",
                                          "LJ soft pair AB");
        lj_soft.LJ_BA = Read_Float_Vector("/forcefield/lj_soft_core/pair_BA",
                                          "LJ soft pair BA");
        lj_soft.LJ_BB = Read_Float_Vector("/forcefield/lj_soft_core/pair_BB",
                                          "LJ soft pair BB");
        lj_soft.atom_numbers = static_cast<int>(lj_soft.atom_LJ_type_A.size());
        if (lj_soft.atom_LJ_type_B.size() !=
            static_cast<std::size_t>(lj_soft.atom_numbers))
        {
            return Fail(
                "native LJ soft-core atom_type_A and atom_type_B "
                "lengths must match");
        }
        lj_soft.atom_type_numbers_A =
            Read_Optional_Int("/forcefield/lj_soft_core/atom_type_count_A");
        lj_soft.atom_type_numbers_B =
            Read_Optional_Int("/forcefield/lj_soft_core/atom_type_count_B");
        if (lj_soft.atom_type_numbers_A <= 0 && !lj_soft.atom_LJ_type_A.empty())
        {
            lj_soft.atom_type_numbers_A =
                *std::max_element(lj_soft.atom_LJ_type_A.begin(),
                                  lj_soft.atom_LJ_type_A.end()) +
                1;
        }
        if (lj_soft.atom_type_numbers_B <= 0 && !lj_soft.atom_LJ_type_B.empty())
        {
            lj_soft.atom_type_numbers_B =
                *std::max_element(lj_soft.atom_LJ_type_B.begin(),
                                  lj_soft.atom_LJ_type_B.end()) +
                1;
        }
        if (lj_soft.atom_type_numbers_A <= 0 ||
            lj_soft.atom_type_numbers_B <= 0)
        {
            return Fail(
                "native LJ soft-core atom type counts must be positive");
        }
        const std::size_t pair_count_A =
            static_cast<std::size_t>(lj_soft.atom_type_numbers_A) *
            static_cast<std::size_t>(lj_soft.atom_type_numbers_A + 1) / 2;
        const std::size_t pair_count_B =
            static_cast<std::size_t>(lj_soft.atom_type_numbers_B) *
            static_cast<std::size_t>(lj_soft.atom_type_numbers_B + 1) / 2;
        if (lj_soft.LJ_AA.size() != pair_count_A ||
            lj_soft.LJ_AB.size() != pair_count_A ||
            lj_soft.LJ_BA.size() != pair_count_B ||
            lj_soft.LJ_BB.size() != pair_count_B)
        {
            return Fail(
                "native LJ soft-core pair parameter lengths do not "
                "match atom type counts");
        }
        if (Exists("/forcefield/subsys_division"))
        {
            lj_soft.subsystem_division = Read_Int_Vector(
                "/forcefield/subsys_division", "subsystem division");
            if (lj_soft.subsystem_division.size() !=
                static_cast<std::size_t>(lj_soft.atom_numbers))
            {
                return Fail(
                    "native /forcefield/subsys_division length must "
                    "match LJ soft-core atom count");
            }
        }
        if (state->atom_count == 0)
        {
            state->atom_count = lj_soft.atom_numbers;
        }
        state->has_lj_soft_core = true;
        return true;
    }

    bool Validate_Virtual_Atom_Arity(
        const NativeTopologyVirtualAtomRecord& record)
    {
        std::size_t expected_from = 0;
        std::size_t expected_parameter = 0;
        switch (record.type)
        {
            case 0:
                expected_from = 1;
                expected_parameter = 1;
                break;
            case 1:
                expected_from = 2;
                expected_parameter = 1;
                break;
            case 2:
            case 3:
                expected_from = 3;
                expected_parameter = 2;
                break;
            default:
                return Fail("native virtual atom contains unsupported type");
        }
        if (record.from.size() != expected_from ||
            record.parameter.size() != expected_parameter)
        {
            return Fail(
                "native virtual atom record arity does not match its "
                "type");
        }
        for (float parameter : record.parameter)
        {
            std::uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(parameter),
                          "float must be 32 bit");
            std::memcpy(&bits, &parameter, sizeof(bits));
            if ((bits & 0x7f800000u) == 0x7f800000u)
            {
                return Fail(
                    "native virtual atom record contains a non-finite "
                    "parameter");
            }
        }
        return true;
    }

    bool Validate_Positive_Mass(const std::vector<float>& mass)
    {
        for (std::size_t i = 0; i < mass.size(); ++i)
        {
            if (!(mass[i] > 0.0f))
            {
                std::ostringstream out;
                out << "atom mass dataset /atoms/mass contains a non-positive "
                       "value at index "
                    << i;
                return Fail(out.str());
            }
        }
        return true;
    }

    template <typename T>
    bool Validate_Length(const std::vector<T>& values,
                         std::int64_t expected_count,
                         const std::string& dataset_path)
    {
        if (expected_count <= 0)
        {
            return true;
        }
        if (values.size() == static_cast<std::size_t>(expected_count))
        {
            return true;
        }
        std::ostringstream out;
        out << dataset_path << " length " << values.size()
            << " does not match /topology/atom_count " << expected_count;
        return Fail(out.str());
    }

    bool Validate_Lengths(NativeTopologyCoreState& state)
    {
        if (state.has_mass &&
            !Validate_Length(state.mass, state.atom_count, "/atoms/mass"))
        {
            return false;
        }
        if (state.has_charge &&
            !Validate_Length(state.charge, state.atom_count, "/atoms/charge"))
        {
            return false;
        }
        if (state.atom_count == 0)
        {
            if (state.has_mass)
            {
                state.atom_count = static_cast<std::int64_t>(state.mass.size());
            }
            else if (state.has_charge)
            {
                state.atom_count =
                    static_cast<std::int64_t>(state.charge.size());
            }
        }
        if (state.has_mass && state.has_charge &&
            state.mass.size() != state.charge.size())
        {
            std::ostringstream out;
            out << "/atoms/mass length " << state.mass.size()
                << " does not match /atoms/charge length "
                << state.charge.size();
            return Fail(out.str());
        }
        if (state.has_lj &&
            !Validate_Length(state.lj.atom_type, state.atom_count,
                             "/forcefield/lj/type"))
        {
            return false;
        }
        if (state.has_gb &&
            (!Validate_Length(state.gb.radius, state.atom_count,
                              "/forcefield/gb/params") ||
             !Validate_Length(state.gb.scale_factor, state.atom_count,
                              "/forcefield/gb/params")))
        {
            return false;
        }
        if (state.has_lj_soft_core &&
            !Validate_Length(state.lj_soft_core.atom_LJ_type_A,
                             state.atom_count,
                             "/forcefield/lj_soft_core/atom_type_A"))
        {
            return false;
        }
        return true;
    }

    template <typename Container>
    bool Validate_Index_Vector(const Container& values, std::int64_t atom_count,
                               const std::string& dataset_path)
    {
        if (atom_count <= 0)
        {
            return true;
        }
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (values[i] < 0 || values[i] >= atom_count)
            {
                std::ostringstream out;
                out << dataset_path << " contains atom index " << values[i]
                    << " outside [0, " << atom_count << ") at flat index " << i;
                return Fail(out.str());
            }
        }
        return true;
    }

    bool Validate_Atom_Indices(const NativeTopologyCoreState& state)
    {
        if (state.has_exclusions)
        {
            for (std::size_t i = 0; i < state.exclusions.excluded_atoms.size();
                 ++i)
            {
                if (!Validate_Index_Vector(state.exclusions.excluded_atoms[i],
                                           state.atom_count,
                                           "/topology/exclusions/list"))
                {
                    return false;
                }
            }
        }
        if (state.has_lj)
        {
            for (std::size_t i = 0; i < state.lj.atom_type.size(); ++i)
            {
                if (state.lj.atom_type[i] < 0 ||
                    state.lj.atom_type[i] >= state.lj.atom_type_numbers)
                {
                    return Fail(
                        "/forcefield/lj/type contains atom type outside "
                        "[0, atom_type_count)");
                }
            }
        }
        if (state.has_bonds)
        {
            if (!Validate_Index_Vector(state.bonds.atom_a, state.atom_count,
                                       "/forcefield/bond/atoms"))
                return false;
            if (!Validate_Index_Vector(state.bonds.atom_b, state.atom_count,
                                       "/forcefield/bond/atoms"))
                return false;
        }
        if (state.has_angles)
        {
            if (!Validate_Index_Vector(state.angles.atom_a, state.atom_count,
                                       "/forcefield/angle/atoms"))
                return false;
            if (!Validate_Index_Vector(state.angles.atom_b, state.atom_count,
                                       "/forcefield/angle/atoms"))
                return false;
            if (!Validate_Index_Vector(state.angles.atom_c, state.atom_count,
                                       "/forcefield/angle/atoms"))
                return false;
        }
        if (state.has_dihedrals)
        {
            if (!Validate_Torsion_Indices(state.dihedrals,
                                          "/forcefield/dihedral/atoms",
                                          state.atom_count))
                return false;
        }
        if (state.has_impropers)
        {
            if (!Validate_Torsion_Indices(state.impropers,
                                          "/forcefield/improper/atoms",
                                          state.atom_count))
                return false;
        }
        if (state.has_nb14)
        {
            if (!Validate_Index_Vector(state.nb14.atom_a, state.atom_count,
                                       "/forcefield/nb14/atoms"))
                return false;
            if (!Validate_Index_Vector(state.nb14.atom_b, state.atom_count,
                                       "/forcefield/nb14/atoms"))
                return false;
        }
        if (state.has_virtual_atoms)
        {
            for (const auto& record : state.virtual_atoms.records)
            {
                if (record.virtual_atom < 0 ||
                    (state.atom_count > 0 &&
                     record.virtual_atom >= state.atom_count))
                {
                    return Fail(
                        "/forcefield/virtual_atom/atom contains index "
                        "outside runtime atom range");
                }
                if (!Validate_Index_Vector(record.from, state.atom_count,
                                           "/forcefield/virtual_atom/from"))
                {
                    return false;
                }
            }
        }
        if (state.has_urey_bradley)
        {
            if (!Validate_Index_Vector(state.urey_bradley.atom_a,
                                       state.atom_count,
                                       "/forcefield/urey_bradley/atoms"))
                return false;
            if (!Validate_Index_Vector(state.urey_bradley.atom_b,
                                       state.atom_count,
                                       "/forcefield/urey_bradley/atoms"))
                return false;
            if (!Validate_Index_Vector(state.urey_bradley.atom_c,
                                       state.atom_count,
                                       "/forcefield/urey_bradley/atoms"))
                return false;
        }
        if (state.has_cmap)
        {
            if (!Validate_Index_Vector(state.cmap.atom_a, state.atom_count,
                                       "/forcefield/cmap/atoms"))
                return false;
            if (!Validate_Index_Vector(state.cmap.atom_b, state.atom_count,
                                       "/forcefield/cmap/atoms"))
                return false;
            if (!Validate_Index_Vector(state.cmap.atom_c, state.atom_count,
                                       "/forcefield/cmap/atoms"))
                return false;
            if (!Validate_Index_Vector(state.cmap.atom_d, state.atom_count,
                                       "/forcefield/cmap/atoms"))
                return false;
            if (!Validate_Index_Vector(state.cmap.atom_e, state.atom_count,
                                       "/forcefield/cmap/atoms"))
                return false;
            for (const int cmap_type : state.cmap.cmap_type)
            {
                if (cmap_type < 0 ||
                    cmap_type >= state.cmap.unique_type_numbers)
                {
                    return Fail(
                        "native cmap type index is outside "
                        "[0, unique_type_numbers)");
                }
            }
        }
        if (state.has_lj_soft_core)
        {
            for (const int atom_type : state.lj_soft_core.atom_LJ_type_A)
            {
                if (atom_type < 0 ||
                    atom_type >= state.lj_soft_core.atom_type_numbers_A)
                {
                    return Fail(
                        "native LJ soft-core atom_type_A contains "
                        "index outside atom_type_count_A");
                }
            }
            for (const int atom_type : state.lj_soft_core.atom_LJ_type_B)
            {
                if (atom_type < 0 ||
                    atom_type >= state.lj_soft_core.atom_type_numbers_B)
                {
                    return Fail(
                        "native LJ soft-core atom_type_B contains "
                        "index outside atom_type_count_B");
                }
            }
        }
        return true;
    }

    bool Validate_Torsion_Indices(const NativeTopologyTorsionState& torsions,
                                  const std::string& dataset_path,
                                  std::int64_t atom_count)
    {
        if (!Validate_Index_Vector(torsions.atom_a, atom_count, dataset_path))
            return false;
        if (!Validate_Index_Vector(torsions.atom_b, atom_count, dataset_path))
            return false;
        if (!Validate_Index_Vector(torsions.atom_c, atom_count, dataset_path))
            return false;
        if (!Validate_Index_Vector(torsions.atom_d, atom_count, dataset_path))
            return false;
        return true;
    }

    static std::size_t Product(const std::vector<std::size_t>& values)
    {
        return std::accumulate(
            values.begin(), values.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    static std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values)
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (const std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    template <typename T>
    static hid_t Native_H5_Type()
    {
        if constexpr (std::is_same<T, float>::value)
        {
            return H5T_NATIVE_FLOAT;
        }
        else if constexpr (std::is_same<T, int>::value)
        {
            return H5T_NATIVE_INT;
        }
        else if constexpr (std::is_same<T, std::int64_t>::value)
        {
            return H5T_NATIVE_INT64;
        }
        else
        {
            static_assert(std::is_same<T, float>::value ||
                              std::is_same<T, int>::value ||
                              std::is_same<T, std::int64_t>::value,
                          "unsupported HDF5 numeric read type");
        }
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};
}  // namespace SpongeH5MD
