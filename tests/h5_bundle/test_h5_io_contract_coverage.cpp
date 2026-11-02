#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "h5_input_matrix_fixture.hpp"
#include "utils/h5md/h5_legacy_sidecar.hpp"

namespace
{
struct ManifestRequirement
{
    const char* bucket;
    const char* contract_id;
    const char* status;
};

struct BundlePathRequirement
{
    const char* file;
    const char* path;
};

struct OutputPlanRequirement
{
    const char* contract_id;
    const char* source_key;
};

struct Phase6BucketEvidence
{
    const char* bucket;
    const char* evidence;
};

std::string Read_Text_File(const std::filesystem::path& path)
{
    std::ifstream in(path);
    REQUIRE_TRUE(in.good());
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

void Require_Manifest_Entry(const std::string& manifest,
                            const ManifestRequirement& requirement)
{
    REQUIRE_TRUE(requirement.bucket != nullptr);
    REQUIRE_TRUE(std::string(requirement.bucket).size() > 0);
    const std::string contract_field =
        std::string("\"contract_id\": \"") + requirement.contract_id + "\"";
    const auto contract_pos = manifest.find(contract_field);
    REQUIRE_TRUE(contract_pos != std::string::npos);

    const auto entry_begin = manifest.rfind('{', contract_pos);
    const auto entry_end = manifest.find('}', contract_pos);
    REQUIRE_TRUE(entry_begin != std::string::npos);
    REQUIRE_TRUE(entry_end != std::string::npos);
    const auto entry =
        manifest.substr(entry_begin, entry_end - entry_begin + 1);

    const std::string status_field =
        std::string("\"status\": \"") + requirement.status + "\"";
    REQUIRE_TRUE(entry.find(status_field) != std::string::npos);
}

std::string Manifest_Entry_Text(const std::string& manifest,
                                const char* contract_id)
{
    const std::string contract_field =
        std::string("\"contract_id\": \"") + contract_id + "\"";
    const auto contract_pos = manifest.find(contract_field);
    REQUIRE_TRUE(contract_pos != std::string::npos);

    const auto entry_begin = manifest.rfind('{', contract_pos);
    const auto entry_end = manifest.find('}', contract_pos);
    REQUIRE_TRUE(entry_begin != std::string::npos);
    REQUIRE_TRUE(entry_end != std::string::npos);
    return manifest.substr(entry_begin, entry_end - entry_begin + 1);
}

std::string Manifest_String_Field(const std::string& entry,
                                  const std::string& field)
{
    const std::string prefix = "\"" + field + "\": \"";
    const auto value_begin = entry.find(prefix);
    REQUIRE_TRUE(value_begin != std::string::npos);
    const auto start = value_begin + prefix.size();
    const auto end = entry.find('"', start);
    REQUIRE_TRUE(end != std::string::npos);
    const auto value = entry.substr(start, end - start);
    REQUIRE_TRUE(!value.empty());
    return value;
}

void Require_Manifest_Entry_Fields(const std::string& entry,
                                   const ManifestRequirement& requirement,
                                   const std::filesystem::path& legacy_root)
{
    const std::vector<std::string> common_fields = {
        "contract_id",  "status",          "component",   "direction",
        "payload_kind", "override_policy", "bundle_file", "bundle_path"};
    for (const auto& field : common_fields)
    {
        (void)Manifest_String_Field(entry, field);
    }

    const std::string status = Manifest_String_Field(entry, "status");
    REQUIRE_EQ(status, std::string(requirement.status));
    if (status == "converted" || status == "typed_converted" ||
        status == "sidecar_embedded")
    {
        const auto source_key = Manifest_String_Field(entry, "source_key");
        const auto source_path = Manifest_String_Field(entry, "source_path");
        REQUIRE_TRUE(!source_key.empty());
        const std::filesystem::path manifest_source(source_path);
        // Checked-in manifests may record POSIX absolute paths on Windows.
        REQUIRE_TRUE(manifest_source.is_absolute() ||
                     (!source_path.empty() && source_path.front() == '/'));
        REQUIRE_TRUE(
            std::filesystem::exists(legacy_root / manifest_source.filename()));
    }

    if (status == "sidecar_embedded")
    {
        const auto sidecar_key = Manifest_String_Field(entry, "sidecar_key");
        const auto sidecar_path = Manifest_String_Field(entry, "sidecar_path");
        const std::filesystem::path relative(sidecar_path);
        REQUIRE_TRUE(!relative.is_absolute());
        REQUIRE_TRUE(relative.begin() != relative.end());
        REQUIRE_TRUE(relative.begin()->string() == "legacy_sidecars");
        auto key_component = relative.begin();
        ++key_component;
        REQUIRE_TRUE(key_component != relative.end());
        REQUIRE_TRUE(key_component->string() == sidecar_key);
        REQUIRE_TRUE(Manifest_String_Field(entry, "source_key") == sidecar_key);
    }
}

std::string Flat_Mdin_Value(const std::string& mdin, const std::string& key)
{
    std::istringstream input(mdin);
    std::string line;
    while (std::getline(input, line))
    {
        const auto comment = line.find('#');
        if (comment != std::string::npos)
        {
            line = line.substr(0, comment);
        }
        const auto equal = line.find('=');
        if (equal == std::string::npos)
        {
            continue;
        }

        auto lhs = line.substr(0, equal);
        auto rhs = line.substr(equal + 1);
        auto trim = [](std::string text)
        {
            while (!text.empty() &&
                   (text.front() == ' ' || text.front() == '\t'))
            {
                text.erase(text.begin());
            }
            while (!text.empty() &&
                   (text.back() == ' ' || text.back() == '\t' ||
                    text.back() == '\r'))
            {
                text.pop_back();
            }
            return text;
        };
        lhs = trim(lhs);
        rhs = trim(rhs);
        if (lhs == key)
        {
            return rhs;
        }
    }
    throw TestFailure("missing mdin key: " + key);
}

std::vector<ManifestRequirement> Full_Contract_Manifest_Requirements()
{
    return {
        {"restart structural state", "restart.coordinate", "converted"},
        {"restart structural state", "restart.velocity", "converted"},
        {"restart structural state", "restart.box", "converted"},
        {"rerun trajectory input", "trajectory.crd", "typed_converted"},
        {"rerun trajectory input", "trajectory.box", "typed_converted"},
        {"rerun trajectory input", "trajectory.vel", "typed_converted"},
        {"topology typed datasets", "topology.mass", "typed_converted"},
        {"topology typed datasets", "topology.charge", "typed_converted"},
        {"topology typed datasets", "topology.residue", "typed_converted"},
        {"topology typed datasets", "topology.bond", "typed_converted"},
        {"ReaxFF sidecars", "topology.REAXFF", "typed_converted"},
        {"ReaxFF sidecars", "topology.REAXFF_type", "typed_converted"},
        {"protocol typed datasets", "protocol.cv", "typed_converted"},
        {"protocol typed datasets", "protocol.constrain", "typed_converted"},
        {"protocol typed datasets", "protocol.restrain", "typed_converted"},
        {"protocol typed datasets", "protocol.restrain_cv", "typed_converted"},
        {"protocol typed datasets", "protocol.soft_walls", "typed_converted"},
        {"protocol typed datasets", "protocol.steer_cv", "typed_converted"},
        {"protocol sidecars", "restart.protocol_sidecar.cv_in_file",
         "sidecar_embedded"},
        {"protocol sidecars", "restart.protocol_sidecar.SITS_in_file",
         "sidecar_embedded"},
        {"protocol sidecars", "restart.protocol_sidecar.meta_potential_in_file",
         "sidecar_embedded"},
        {"SITS state and sidecars", "protocol.SITS", "typed_converted"},
        {"SITS state and sidecars", "protocol.SITS_atom", "typed_converted"},
        {"SITS state and sidecars", "restart.SITS_nk", "typed_converted"},
        {"restraint typed datasets", "protocol.restrain_atom_id",
         "typed_converted"},
        {"restraint typed datasets", "protocol.restrain_weight",
         "typed_converted"},
        {"restraint typed datasets", "restart.restrain_coordinate",
         "typed_converted"},
        {"metadynamics state and sidecars", "protocol.meta_edge",
         "typed_converted"},
        {"metadynamics state and sidecars", "restart.meta_potential",
         "typed_converted"},
        {"metadynamics state and sidecars", "restart.meta_scatter",
         "typed_converted"},
        {"metadynamics state and sidecars", "restart.hills", "typed_converted"},
        {"restart dynamic state", "restart.nose_hoover_chain_restart_input",
         "typed_converted"},
        {"custom force payloads", "topology.pairwise_force", "typed_converted"},
        {"custom force payloads", "topology.listed_forces", "typed_converted"},
        {"custom force payloads", "topology.pairwise_force_data.custom_pair",
         "typed_converted"},
        {"custom force payloads", "topology.listed_force_data.custom_bond",
         "typed_converted"},
        {"bundled output paths", "output.h5.output_h5_trajectory_path",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_trajectory_vds",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_restart_path",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_restart_topology_hash",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_restart_atom_order_hash",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_restart_protocol_hash",
         "output_plan_preserved"},
        {"bundled output paths", "output.h5.output_h5_observable_path",
         "output_plan_preserved"},
    };
}

std::vector<OutputPlanRequirement> Full_Contract_Output_Plan_Requirements()
{
    return {
        {"output.h5.output_h5_trajectory_path", "output_h5_trajectory_path"},
        {"output.h5.output_h5_trajectory_vds", "output_h5_trajectory_vds"},
        {"output.h5.output_h5_restart_path", "output_h5_restart_path"},
        {"output.h5.output_h5_restart_topology_hash",
         "output_h5_restart_topology_hash"},
        {"output.h5.output_h5_restart_atom_order_hash",
         "output_h5_restart_atom_order_hash"},
        {"output.h5.output_h5_restart_protocol_hash",
         "output_h5_restart_protocol_hash"},
        {"output.h5.output_h5_observable_path", "output_h5_observable_path"},
    };
}

std::vector<OutputPlanRequirement>
Full_Contract_Legacy_Output_Sidecar_Requirements()
{
    return {
        {"output.legacy_sidecar.mdout", "mdout"},
        {"output.legacy_sidecar.crd", "crd"},
        {"output.legacy_sidecar.box", "box"},
        {"output.legacy_sidecar.vel", "vel"},
    };
}

std::vector<Phase6BucketEvidence> Phase6_Coverage_Bucket_Evidence()
{
    return {
        {"Topology typed datasets", "topology.mass"},
        {"Restart structural state", "restart.coordinate"},
        {"Restart dynamic state", "restart.nose_hoover_chain_restart_input"},
        {"Protocol sidecars", "restart.protocol_sidecar.cv_in_file"},
        {"SITS state and sidecars", "protocol.SITS"},
        {"Metadynamics state and sidecars", "restart.meta_potential"},
        {"Custom pairwise/listed force payloads", "topology.pairwise_force"},
        {"ReaxFF sidecars", "topology.REAXFF"},
        {"Rerun trajectory input", "trajectory.crd"},
        {"Legacy sidecar key/path tables",
         "Test_Full_Contract_Rerun_H5_Files_Cover_Sidecar_Tables"},
        {"Bundled output trajectory/restart/observable paths",
         "Test_Full_Contract_Rerun_Output_H5_Plan_Is_Preserved"},
    };
}

void Require_Legacy_Sidecar_Table(const std::filesystem::path& path,
                                  const std::set<std::string>& expected_keys)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(path.string(),
                                                          &sidecars, &error));
    REQUIRE_EQ(sidecars.size(), expected_keys.size());
    const auto sidecar_root = path.parent_path() / "legacy_sidecars";
    REQUIRE_TRUE(std::filesystem::is_directory(sidecar_root));
    std::set<std::string> actual_keys;
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(!sidecar.key.empty());
        REQUIRE_TRUE(actual_keys.insert(sidecar.key).second);
        const std::filesystem::path sidecar_path(sidecar.path);
        REQUIRE_TRUE(sidecar_path.is_absolute());
        REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
        std::error_code error_code;
        const auto relative =
            std::filesystem::relative(sidecar_path, sidecar_root, error_code);
        REQUIRE_TRUE(!error_code);
        REQUIRE_TRUE(!relative.empty());
        REQUIRE_TRUE(relative.begin() != relative.end());
        REQUIRE_TRUE(relative.begin()->string() == sidecar.key);
    }
    REQUIRE_TRUE(actual_keys == expected_keys);
}

void Test_Full_Contract_Rerun_H5_Files_Cover_Sidecar_Tables()
{
    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path() /
                      "bundled_input_with_legacy_sidecar";
    const auto bundle = full / "bundle";
    const auto manifest_path = full / "manifest.json";
    SpongeH5InputMatrix::Require_Path_Exists(manifest_path);

    SpongeH5InputMatrix::Require_Path_Exists(bundle / "trajectory.spg.h5md");
    Require_Legacy_Sidecar_Table(bundle / "topology.spgt.h5",
                                 {"pairwise_force_in_file",
                                  "listed_forces_in_file",
                                  "mass_in_file",
                                  "charge_in_file",
                                  "exclude_in_file",
                                  "bond_in_file",
                                  "angle_in_file",
                                  "dihedral_in_file",
                                  "LJ_in_file",
                                  "nb14_extra_in_file",
                                  "urey_bradley_in_file",
                                  "cmap_in_file",
                                  "gb_in_file",
                                  "virtual_atom_in_file",
                                  "LJ_soft_core_in_file",
                                  "subsys_division_in_file",
                                  "EAM_in_file",
                                  "EAM_atom_type_in_file",
                                  "SW_in_file",
                                  "EDIP_in_file",
                                  "TERSOFF_in_file",
                                  "REAXFF_in_file",
                                  "REAXFF_type_in_file"});
    Require_Legacy_Sidecar_Table(
        bundle / "protocol.spgp.h5",
        {"cv_in_file", "constrain_in_file", "restrain_in_file",
         "soft_walls_in_file", "SITS_in_file", "SITS_atom_in_file",
         "restrain_atom_id", "restrain_weight_in_file", "meta_edge_in_file",
         "restrain_cv_in_file", "steer_cv_in_file"});
    Require_Legacy_Sidecar_Table(
        bundle / "restart.spgr.h5",
        {"SITS_nk_in_file", "restrain_coordinate_in_file",
         "meta_potential_in_file", "meta_scatter_in_file",
         "nose_hoover_chain_restart_input", "hills_in_file", "cv_in_file",
         "constrain_in_file", "restrain_in_file", "pairwise_force_in_file",
         "listed_forces_in_file", "soft_walls_in_file", "SITS_in_file",
         "SITS_atom_in_file", "restrain_atom_id", "restrain_weight_in_file",
         "meta_edge_in_file", "restrain_cv_in_file", "steer_cv_in_file"});
}

void Test_Full_Contract_Rerun_H5_Files_Cover_Required_Bundle_Paths()
{
    const auto bundle = SpongeH5InputMatrix::Full_Contract_Rerun_Path() /
                        "bundled_input_with_legacy_sidecar" / "bundle";
    const std::vector<BundlePathRequirement> requirements = {
        {"restart.spgr.h5", "/particles/all/position/value"},
        {"restart.spgr.h5", "/particles/all/velocity/value"},
        {"restart.spgr.h5", "/particles/all/box/edges/value"},
        {"restart.spgr.h5", "/parameters/restart/thermostat/nose_hoover_chain"},
        {"restart.spgr.h5", "/parameters/restart/bias/sits/SITS/nk"},
        {"restart.spgr.h5",
         "/parameters/restart/bias/meta/default/potential_export"},
        {"restart.spgr.h5", "/parameters/restart/bias/meta/default/scatter"},
        {"restart.spgr.h5", "/parameters/restart/bias/meta/default/hills"},
        {"restart.spgr.h5",
         "/parameters/restart/references/restraint/default/coordinate"},
        {"trajectory.spg.h5md", "/particles/all/position/value"},
        {"trajectory.spg.h5md", "/particles/all/velocity/value"},
        {"trajectory.spg.h5md", "/particles/all/box/edges/value"},
        {"trajectory.spg.h5md", "/particles/all/step"},
        {"trajectory.spg.h5md", "/particles/all/time"},
        {"topology.spgt.h5", "/atoms/mass"},
        {"topology.spgt.h5", "/atoms/charge"},
        {"topology.spgt.h5", "/atoms/residue_index"},
        {"topology.spgt.h5", "/residues/atom_offset"},
        {"topology.spgt.h5", "/forcefield/bond"},
        {"topology.spgt.h5", "/forcefield/custom_force/pairwise"},
        {"topology.spgt.h5",
         "/forcefield/custom_force/pairwise/data/custom_pair"},
        {"topology.spgt.h5", "/forcefield/custom_force/listed"},
        {"topology.spgt.h5",
         "/forcefield/custom_force/listed/data/custom_bond"},
        {"topology.spgt.h5", "/manybody/reaxff/parameters"},
        {"topology.spgt.h5", "/manybody/reaxff/type"},
        {"protocol.spgp.h5", "/cv"},
        {"protocol.spgp.h5", "/cv/config/section/name"},
        {"protocol.spgp.h5", "/constraint/default/pairs/atoms"},
        {"protocol.spgp.h5", "/constraint/default/pairs/r0"},
        {"protocol.spgp.h5", "/sits"},
        {"protocol.spgp.h5", "/sits/atom_indices"},
        {"protocol.spgp.h5", "/restraint/config/section/name"},
        {"protocol.spgp.h5", "/restraint/cv/config/section/name"},
        {"protocol.spgp.h5", "/restraint/default/atom_indices"},
        {"protocol.spgp.h5", "/restraint/default/weight"},
        {"protocol.spgp.h5", "/meta/default/grid"},
        {"protocol.spgp.h5", "/wall/soft/potential"},
        {"protocol.spgp.h5", "/steer/config/section/name"},
    };

    for (const auto& requirement : requirements)
    {
        HighFive::File file((bundle / requirement.file).string(),
                            HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(requirement.path));
    }
}

void Test_Full_Contract_Rerun_Manifest_Covers_Planned_Buckets()
{
    const auto fixture_root = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto full = fixture_root / "bundled_input_with_legacy_sidecar";
    const auto legacy_root = fixture_root / "legacy_input";
    const auto manifest = Read_Text_File(full / "manifest.json");

    for (const auto& requirement : Full_Contract_Manifest_Requirements())
    {
        Require_Manifest_Entry(manifest, requirement);
        Require_Manifest_Entry_Fields(
            Manifest_Entry_Text(manifest, requirement.contract_id), requirement,
            legacy_root);
    }
}

void Test_Full_Contract_Rerun_Output_H5_Plan_Is_Preserved()
{
    const auto fixture_root = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto full = fixture_root / "bundled_input_with_legacy_sidecar";
    const auto bundle = full / "bundle";
    const auto legacy = fixture_root / "legacy_input";
    const auto manifest = Read_Text_File(full / "manifest.json");
    const auto bundled_mdin = Read_Text_File(bundle / "mdin.bundled.spg.toml");
    const auto legacy_mdin = Read_Text_File(legacy / "mdin.spg.toml");

    for (const auto& requirement : Full_Contract_Output_Plan_Requirements())
    {
        Require_Manifest_Entry(manifest,
                               {"bundled output paths", requirement.contract_id,
                                "output_plan_preserved"});
        const auto entry =
            Manifest_Entry_Text(manifest, requirement.contract_id);
        REQUIRE_TRUE(entry.find("\"bundle_file\": \"run.mdin\"") !=
                     std::string::npos);
        REQUIRE_TRUE(entry.find(std::string("\"bundle_path\": \"") +
                                requirement.source_key + "\"") !=
                     std::string::npos);
        REQUIRE_TRUE(entry.find(std::string("\"source_key\": \"") +
                                requirement.source_key + "\"") !=
                     std::string::npos);
        REQUIRE_TRUE(Flat_Mdin_Value(bundled_mdin, requirement.source_key) ==
                     Flat_Mdin_Value(legacy_mdin, requirement.source_key));
    }
}

void Test_Full_Contract_Rerun_Legacy_Output_Sidecar_Plan_Is_Preserved()
{
    const auto fixture_root = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto full = fixture_root / "bundled_input_with_legacy_sidecar";
    const auto bundle = full / "bundle";
    const auto legacy = fixture_root / "legacy_input";
    const auto manifest = Read_Text_File(full / "manifest.json");
    const auto bundled_mdin = Read_Text_File(bundle / "mdin.bundled.spg.toml");
    const auto legacy_mdin = Read_Text_File(legacy / "mdin.spg.toml");

    std::set<std::string> active_bundled_legacy_outputs;
    for (const auto& requirement :
         Full_Contract_Legacy_Output_Sidecar_Requirements())
    {
        Require_Manifest_Entry(
            manifest, {"legacy output sidecars", requirement.contract_id,
                       "legacy_output_sidecar_preserved"});
        const auto entry =
            Manifest_Entry_Text(manifest, requirement.contract_id);
        Require_Manifest_Entry_Fields(
            entry,
            {"legacy output sidecars", requirement.contract_id,
             "legacy_output_sidecar_preserved"},
            legacy);
        REQUIRE_EQ(Manifest_String_Field(entry, "component"),
                   std::string("output"));
        REQUIRE_EQ(Manifest_String_Field(entry, "direction"),
                   std::string("output"));
        REQUIRE_EQ(Manifest_String_Field(entry, "payload_kind"),
                   std::string("path"));
        REQUIRE_EQ(Manifest_String_Field(entry, "override_policy"),
                   std::string("explicit"));
        REQUIRE_EQ(Manifest_String_Field(entry, "bundle_file"),
                   std::string("*.legacy"));
        REQUIRE_EQ(
            Manifest_String_Field(entry, "bundle_path"),
            std::string("/parameters/sponge/files/") + requirement.source_key);
        REQUIRE_EQ(Manifest_String_Field(entry, "source_key"),
                   std::string(requirement.source_key));
        (void)Flat_Mdin_Value(legacy_mdin, requirement.source_key);
        try
        {
            (void)Flat_Mdin_Value(bundled_mdin, requirement.source_key);
            active_bundled_legacy_outputs.insert(requirement.source_key);
        }
        catch (const TestFailure&)
        {
        }
    }
    REQUIRE_TRUE(active_bundled_legacy_outputs ==
                 std::set<std::string>{"mdout"});
}

void Test_Full_Contract_Rerun_Manifest_Buckets_Are_Explicit()
{
    const std::set<std::string> expected_buckets = {
        "ReaxFF sidecars",
        "SITS state and sidecars",
        "bundled output paths",
        "custom force payloads",
        "metadynamics state and sidecars",
        "protocol sidecars",
        "protocol typed datasets",
        "rerun trajectory input",
        "restart dynamic state",
        "restart structural state",
        "restraint typed datasets",
        "topology typed datasets",
    };

    std::set<std::string> actual_buckets;
    for (const auto& requirement : Full_Contract_Manifest_Requirements())
    {
        REQUIRE_TRUE(requirement.bucket != nullptr);
        actual_buckets.insert(requirement.bucket);
    }
    REQUIRE_EQ(actual_buckets.size(), expected_buckets.size());
    REQUIRE_TRUE(actual_buckets == expected_buckets);
}

void Test_Phase6_Plan_Buckets_Are_Represented()
{
    const std::set<std::string> expected_plan_buckets = {
        "Topology typed datasets",
        "Restart structural state",
        "Restart dynamic state",
        "Protocol sidecars",
        "SITS state and sidecars",
        "Metadynamics state and sidecars",
        "Custom pairwise/listed force payloads",
        "ReaxFF sidecars",
        "Rerun trajectory input",
        "Legacy sidecar key/path tables",
        "Bundled output trajectory/restart/observable paths",
    };

    std::set<std::string> actual_plan_buckets;
    for (const auto& evidence : Phase6_Coverage_Bucket_Evidence())
    {
        REQUIRE_TRUE(evidence.bucket != nullptr);
        REQUIRE_TRUE(evidence.evidence != nullptr);
        REQUIRE_TRUE(std::string(evidence.evidence).size() > 0);
        actual_plan_buckets.insert(evidence.bucket);
    }
    REQUIRE_TRUE(actual_plan_buckets == expected_plan_buckets);
}
}  // namespace

int main()
{
    return SpongeH5Test::Run_Test(
        []
        {
            Test_Full_Contract_Rerun_H5_Files_Cover_Sidecar_Tables();
            Test_Full_Contract_Rerun_H5_Files_Cover_Required_Bundle_Paths();
            Test_Full_Contract_Rerun_Manifest_Covers_Planned_Buckets();
            Test_Full_Contract_Rerun_Output_H5_Plan_Is_Preserved();
            Test_Full_Contract_Rerun_Legacy_Output_Sidecar_Plan_Is_Preserved();
            Test_Full_Contract_Rerun_Manifest_Buckets_Are_Explicit();
            Test_Phase6_Plan_Buckets_Are_Represented();
        });
}
