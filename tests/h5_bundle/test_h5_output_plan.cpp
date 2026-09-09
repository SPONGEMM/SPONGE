#include "h5_bundle_test_common.hpp"
#include "utils/control/h5_output_contract.hpp"
#include "utils/h5md/output_plan.hpp"

using namespace SpongeH5Test;

static void Test_Defaults_And_Legacy_Gating()
{
    CONTROLLER controller;
    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(!plan.any_h5_output_enabled);
    REQUIRE_TRUE(!plan.trajectory.enabled);
    REQUIRE_TRUE(!plan.trajectory.vds);
    REQUIRE_EQ(plan.trajectory.chunk_size, 20);
    REQUIRE_EQ(plan.trajectory.repair_policy, std::string("strict"));
    REQUIRE_TRUE(!plan.trajectory.allow_complete_prefix_repair);
    REQUIRE_TRUE(plan.legacy.default_enabled);
    REQUIRE_TRUE(plan.legacy.Enabled("mdout"));
}

static void Test_Contract_Helper_Functions()
{
    REQUIRE_TRUE(!SpongeH5OutputContract::Any_H5_Output_Enabled(nullptr));
    REQUIRE_TRUE(
        !SpongeH5OutputContract::Legacy_Sidecar_Requested(nullptr, "mdout"));
    REQUIRE_TRUE(
        SpongeH5OutputContract::Legacy_Sidecars_Default_Enabled(nullptr));

    CONTROLLER controller;
    REQUIRE_TRUE(!SpongeH5OutputContract::Any_H5_Output_Enabled(&controller));
    REQUIRE_TRUE(!SpongeH5OutputContract::Command_Has_Non_Empty_Value(
        &controller, SpongeH5OutputContract::kTrajectoryPathKey));
    REQUIRE_TRUE(
        SpongeH5OutputContract::Legacy_Sidecars_Default_Enabled(&controller));
    REQUIRE_TRUE(
        SpongeH5OutputContract::Legacy_Sidecar_Enabled(&controller, "mdout"));
    REQUIRE_TRUE(!SpongeH5OutputContract::Legacy_Sidecar_Requested(&controller,
                                                                   "mdout"));

    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    REQUIRE_TRUE(SpongeH5OutputContract::Command_Has_Non_Empty_Value(
        &controller, SpongeH5OutputContract::kTrajectoryPathKey));
    REQUIRE_TRUE(SpongeH5OutputContract::Any_H5_Output_Enabled(&controller));
    REQUIRE_TRUE(
        !SpongeH5OutputContract::Legacy_Sidecars_Default_Enabled(&controller));
    REQUIRE_TRUE(
        !SpongeH5OutputContract::Legacy_Sidecar_Enabled(&controller, "mdout"));
    controller.Set("mdout", "legacy.out");
    REQUIRE_TRUE(
        SpongeH5OutputContract::Legacy_Sidecar_Requested(&controller, "mdout"));
    REQUIRE_TRUE(
        SpongeH5OutputContract::Legacy_Sidecar_Enabled(&controller, "mdout"));
}

static void Test_Grouped_Output_H5_Key_Names_And_Legacy_Gating()
{
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kTrajectoryPathKey),
               std::string("output_h5_trajectory_path"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kTrajectoryVdsKey),
               std::string("output_h5_trajectory_vds"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kTrajectoryChunkSizeKey),
               std::string("output_h5_trajectory_chunk_size"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kTrajectoryRepairPolicyKey),
               std::string("output_h5_trajectory_repair_policy"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kRestartPathKey),
               std::string("output_h5_restart_path"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kRestartTopologyHashKey),
               std::string("output_h5_restart_topology_hash"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kRestartAtomOrderHashKey),
               std::string("output_h5_restart_atom_order_hash"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kRestartProtocolHashKey),
               std::string("output_h5_restart_protocol_hash"));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::kObservablePathKey),
               std::string("output_h5_observable_path"));

    {
        CONTROLLER controller;
        controller.Set("output_trajectory_h5_path", "legacy_name.spg.h5md");
        controller.Set("output_traj_h5", "legacy_name.spg.h5md");
        controller.Set("output_restart_h5", "legacy_name.spgr.h5");
        controller.Set("output_observable_h5", "legacy_name.obs.spg.h5md");

        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(
            !SpongeH5OutputContract::Any_H5_Output_Enabled(&controller));
        REQUIRE_TRUE(!plan.any_h5_output_enabled);
        REQUIRE_TRUE(!plan.trajectory.enabled);
        REQUIRE_TRUE(!plan.restart.enabled);
        REQUIRE_TRUE(!plan.observable.enabled);
        REQUIRE_TRUE(plan.legacy.default_enabled);
    }

    static constexpr const char* h5_path_keys[] = {
        SpongeH5OutputContract::kTrajectoryPathKey,
        SpongeH5OutputContract::kRestartPathKey,
        SpongeH5OutputContract::kObservablePathKey};
    static constexpr const char* h5_path_values[] = {
        "prod.spg.h5md", "prod.spgr.h5", "prod.obs.spg.h5md"};

    for (std::size_t i = 0; i < 3; ++i)
    {
        CONTROLLER controller;
        controller.Set(h5_path_keys[i], h5_path_values[i]);
        REQUIRE_TRUE(
            SpongeH5OutputContract::Any_H5_Output_Enabled(&controller));
        REQUIRE_TRUE(!SpongeH5OutputContract::Legacy_Sidecars_Default_Enabled(
            &controller));
        REQUIRE_TRUE(!SpongeH5OutputContract::Legacy_Sidecar_Enabled(
            &controller, "mdout"));

        controller.Set("mdout", "explicit.out");
        REQUIRE_TRUE(SpongeH5OutputContract::Legacy_Sidecar_Requested(
            &controller, "mdout"));
        REQUIRE_TRUE(SpongeH5OutputContract::Legacy_Sidecar_Enabled(&controller,
                                                                    "mdout"));
    }
}

static void Test_Raw_Input_Restart_Lineage_Plan()
{
    const std::string topology_hash = "sha256:" + std::string(64, '1');
    const std::string atom_order_hash = "sha256:" + std::string(64, '2');
    const std::string protocol_hash = "sha256:" + std::string(64, '3');
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kRestartPathKey, "restart.spgr.h5");
    controller.Set(SpongeH5OutputContract::kRestartTopologyHashKey,
                   topology_hash.c_str());
    controller.Set(SpongeH5OutputContract::kRestartAtomOrderHashKey,
                   atom_order_hash.c_str());
    controller.Set(SpongeH5OutputContract::kRestartProtocolHashKey,
                   protocol_hash.c_str());

    const auto plan =
        SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.restart.enabled);
    REQUIRE_EQ(plan.restart.topology_hash, topology_hash);
    REQUIRE_EQ(plan.restart.atom_order_hash, atom_order_hash);
    REQUIRE_EQ(plan.restart.protocol_hash, protocol_hash);
    REQUIRE_TRUE(SpongeH5OutputContract::Is_Canonical_Sha256(topology_hash));
    REQUIRE_TRUE(!SpongeH5OutputContract::Is_Canonical_Sha256("sha256:ABCDEF"));
    REQUIRE_TRUE(!SpongeH5OutputContract::Is_Canonical_Sha256(
        "sha256:" + std::string(63, '0')));
}

static void Test_Empty_H5_Output_Paths_Do_Not_Enable_Bundles()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "");
    controller.Set(SpongeH5OutputContract::kRestartPathKey, "");
    controller.Set(SpongeH5OutputContract::kObservablePathKey, "");
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "true");
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "20");
    controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                   "strict");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(!SpongeH5OutputContract::Any_H5_Output_Enabled(&controller));
    REQUIRE_TRUE(!SpongeH5OutputContract::Command_Has_Non_Empty_Value(
        &controller, SpongeH5OutputContract::kTrajectoryPathKey));
    REQUIRE_TRUE(!plan.any_h5_output_enabled);
    REQUIRE_TRUE(!plan.trajectory.enabled);
    REQUIRE_TRUE(!plan.restart.enabled);
    REQUIRE_TRUE(!plan.observable.enabled);
    REQUIRE_TRUE(plan.trajectory.vds);
    REQUIRE_EQ(plan.trajectory.chunk_size, 20);
    REQUIRE_TRUE(plan.legacy.default_enabled);
    REQUIRE_TRUE(plan.legacy.Enabled("mdout"));
}

static void Test_Null_Controller_Resolver()
{
    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(nullptr, false);
    REQUIRE_TRUE(!plan.valid);
    REQUIRE_EQ(plan.error_message, std::string("CONTROLLER is null"));
}

static void Test_Output_Selectors_Do_Not_Enable_H5_Without_Path()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "true");
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "5");
    controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                   "strict");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(!plan.any_h5_output_enabled);
    REQUIRE_TRUE(!plan.trajectory.enabled);
    REQUIRE_TRUE(plan.trajectory.vds);
    REQUIRE_EQ(plan.trajectory.chunk_size, 5);
    REQUIRE_TRUE(plan.legacy.default_enabled);
}

static void Test_H5_Output_Path_Keys_Enable_Only_Their_Bundle()
{
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");

        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.any_h5_output_enabled);
        REQUIRE_TRUE(plan.trajectory.enabled);
        REQUIRE_TRUE(!plan.restart.enabled);
        REQUIRE_TRUE(!plan.observable.enabled);
        REQUIRE_EQ(plan.trajectory.path, std::string("prod.spg.h5md"));
        REQUIRE_EQ(plan.trajectory.derived_shard_root,
                   std::string("prod.spg.shards"));
        REQUIRE_TRUE(!plan.legacy.default_enabled);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kRestartPathKey, "prod.spgr.h5");

        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.any_h5_output_enabled);
        REQUIRE_TRUE(!plan.trajectory.enabled);
        REQUIRE_TRUE(plan.restart.enabled);
        REQUIRE_TRUE(!plan.observable.enabled);
        REQUIRE_EQ(plan.restart.path, std::string("prod.spgr.h5"));
        REQUIRE_TRUE(!plan.legacy.default_enabled);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kObservablePathKey,
                       "prod.obs.spg.h5md");

        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.any_h5_output_enabled);
        REQUIRE_TRUE(!plan.trajectory.enabled);
        REQUIRE_TRUE(!plan.restart.enabled);
        REQUIRE_TRUE(plan.observable.enabled);
        REQUIRE_EQ(plan.observable.path, std::string("prod.obs.spg.h5md"));
        REQUIRE_TRUE(!plan.legacy.default_enabled);
    }
}

static void Test_All_H5_Output_Bundles_Can_Be_Enabled_Together()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    controller.Set(SpongeH5OutputContract::kRestartPathKey, "prod.spgr.h5");
    controller.Set(SpongeH5OutputContract::kObservablePathKey,
                   "prod.obs.spg.h5md");
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "off");
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "20");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_output_enabled);
    REQUIRE_TRUE(plan.trajectory.enabled);
    REQUIRE_TRUE(plan.restart.enabled);
    REQUIRE_TRUE(plan.observable.enabled);
    REQUIRE_TRUE(!plan.trajectory.vds);
    REQUIRE_EQ(plan.trajectory.chunk_size, 20);
    REQUIRE_EQ(plan.trajectory.path, std::string("prod.spg.h5md"));
    REQUIRE_EQ(plan.restart.path, std::string("prod.spgr.h5"));
    REQUIRE_EQ(plan.observable.path, std::string("prod.obs.spg.h5md"));
    REQUIRE_TRUE(plan.trajectory.has_recommended_suffix);
    REQUIRE_TRUE(plan.restart.has_recommended_suffix);
    REQUIRE_TRUE(plan.observable.has_recommended_suffix);
    REQUIRE_TRUE(!plan.legacy.default_enabled);
    REQUIRE_TRUE(!plan.legacy.Enabled("mdout"));
}

static void Test_Bool_Parsing_Text_Variants()
{
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool("1"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool("true"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool("TRUE"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool("yes"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool("on"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool("0"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool("false"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool("maybe"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool(""));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool("off"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool("no"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Parse_Bool(nullptr, true));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Parse_Bool(nullptr, false));
    REQUIRE_TRUE(SpongeH5OutputPlan::Is_Bool_Text("true"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Is_Bool_Text("FALSE"));
    REQUIRE_TRUE(SpongeH5OutputPlan::Is_Bool_Text("0"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Is_Bool_Text("maybe"));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Is_Bool_Text(""));
    REQUIRE_TRUE(!SpongeH5OutputPlan::Is_Bool_Text(nullptr));
    REQUIRE_EQ(SpongeH5OutputPlan::Lowercase("CoMpLeTe_PrEfIx"),
               std::string("complete_prefix"));
    REQUIRE_EQ(SpongeH5OutputPlan::Lowercase("STRICT"), std::string("strict"));
}

static void Test_Trajectory_Vds_Repair_Policy()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "true");
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "7");
    controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                   "complete_prefix");
    controller.Set("mdout", "legacy.out");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_output_enabled);
    REQUIRE_TRUE(plan.trajectory.enabled);
    REQUIRE_TRUE(plan.trajectory.vds);
    REQUIRE_EQ(plan.trajectory.path, std::string("prod.spg.h5md"));
    REQUIRE_EQ(plan.trajectory.chunk_size, 7);
    REQUIRE_EQ(plan.trajectory.derived_shard_root,
               std::string("prod.spg.shards"));
    REQUIRE_TRUE(plan.trajectory.allow_complete_prefix_repair);
    REQUIRE_TRUE(!plan.legacy.default_enabled);
    REQUIRE_TRUE(plan.legacy.Enabled("mdout"));
    REQUIRE_TRUE(plan.legacy.Explicitly_Requested("mdout"));
}

static void Test_Repair_Policy_Alias()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "on");
    controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                   "allow_complete_prefix");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.trajectory.vds);
    REQUIRE_TRUE(plan.trajectory.allow_complete_prefix_repair);
    REQUIRE_EQ(plan.trajectory.repair_policy,
               std::string("allow_complete_prefix"));
}

static void Test_Repair_Policy_Is_Case_Insensitive()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "TRUE");
    controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                   "COMPLETE_PREFIX");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_EQ(plan.trajectory.repair_policy, std::string("complete_prefix"));
    REQUIRE_TRUE(plan.trajectory.allow_complete_prefix_repair);
}

static void Test_Restart_And_Observable_Paths()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kRestartPathKey, "prod.spgr.h5");
    controller.Set(SpongeH5OutputContract::kObservablePathKey,
                   "prod.obs.spg.h5md");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.any_h5_output_enabled);
    REQUIRE_TRUE(plan.restart.enabled);
    REQUIRE_TRUE(plan.restart.has_recommended_suffix);
    REQUIRE_TRUE(plan.observable.enabled);
    REQUIRE_TRUE(plan.observable.has_recommended_suffix);
    REQUIRE_TRUE(!plan.legacy.default_enabled);
}

static void Test_Invalid_Values()
{
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "0");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "-3");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("chunk_size") !=
                     std::string::npos);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey,
                       "not_an_integer");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("chunk_size") !=
                     std::string::npos);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey,
                       "12frames");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("chunk_size") !=
                     std::string::npos);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryVdsKey, "maybe");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("output_h5_trajectory_vds") !=
                     std::string::npos);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                       "truncate");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(
            plan.error_message.find("output_h5_trajectory_repair_policy") !=
            std::string::npos);
    }
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                       "complete_prefix");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("output_h5_trajectory_vds=true") !=
                     std::string::npos);
    }
}

static void Test_Throw_On_Error_Uses_Controller_Error_Path()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "prod.spg.h5md");
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "0");

    bool threw = false;
    try
    {
        (void)SpongeH5OutputPlan::Resolve_Output_Plan(&controller, true);
    }
    catch (const std::runtime_error& err)
    {
        threw = true;
        REQUIRE_TRUE(
            std::string(err.what())
                .find(
                    "output_h5_trajectory_chunk_size must be greater than 0") !=
            std::string::npos);
    }
    REQUIRE_TRUE(threw);
}

static void Test_Suffix_And_Shard_Derivation()
{
    REQUIRE_TRUE(SpongeH5OutputContract::Has_Recommended_Suffix(
        "x.spg.h5md", SpongeH5OutputContract::kTrajectorySuffix));
    REQUIRE_TRUE(!SpongeH5OutputContract::Ends_With("h5md", ".spg.h5md"));
    REQUIRE_TRUE(!SpongeH5OutputContract::Ends_With("x.spg.h5m", ".spg.h5md"));
    REQUIRE_TRUE(
        !SpongeH5OutputContract::Ends_With("x.spg.h5md.tmp", ".spg.h5md"));
    REQUIRE_TRUE(!SpongeH5OutputContract::Has_Recommended_Suffix(
        "x.h5", SpongeH5OutputContract::kTrajectorySuffix));
    REQUIRE_TRUE(SpongeH5OutputContract::Has_Recommended_Suffix(
        "x.spgr.h5", SpongeH5OutputContract::kRestartSuffix));
    REQUIRE_TRUE(SpongeH5OutputContract::Has_Recommended_Suffix(
        "x.obs.spg.h5md", SpongeH5OutputContract::kObservableSuffix));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::Recommended_Suffix_For_Key(
                   SpongeH5OutputContract::kTrajectoryPathKey)),
               std::string(SpongeH5OutputContract::kTrajectorySuffix));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::Recommended_Suffix_For_Key(
                   SpongeH5OutputContract::kRestartPathKey)),
               std::string(SpongeH5OutputContract::kRestartSuffix));
    REQUIRE_EQ(std::string(SpongeH5OutputContract::Recommended_Suffix_For_Key(
                   SpongeH5OutputContract::kObservablePathKey)),
               std::string(SpongeH5OutputContract::kObservableSuffix));
    REQUIRE_TRUE(SpongeH5OutputContract::Recommended_Suffix_For_Key(
                     "unknown_output_key") == nullptr);
    REQUIRE_EQ(SpongeH5OutputPlan::Derive_Shards_Root("x.spg.h5md"),
               std::string("x.spg.shards"));
    REQUIRE_EQ(SpongeH5OutputPlan::Derive_Shards_Root("runs/prod.spg.h5md"),
               std::string("runs/prod.spg.shards"));
    REQUIRE_EQ(SpongeH5OutputPlan::Derive_Shards_Root("x.h5"),
               std::string("x.h5.shards"));
}

static void Test_Contract_Helper_Edge_Cases()
{
    REQUIRE_EQ(SpongeH5OutputPlan::Command_String(nullptr, "missing"),
               std::string(""));
    CONTROLLER command_controller;
    REQUIRE_EQ(SpongeH5OutputPlan::Command_String(&command_controller, nullptr),
               std::string(""));
    REQUIRE_EQ(
        SpongeH5OutputPlan::Command_String(&command_controller, "missing"),
        std::string(""));
    command_controller.Set("present", "value");
    REQUIRE_EQ(
        SpongeH5OutputPlan::Command_String(&command_controller, "present"),
        std::string("value"));

    REQUIRE_TRUE(SpongeH5OutputContract::Recommended_Suffix_For_Key(nullptr) ==
                 nullptr);
    REQUIRE_TRUE(!SpongeH5OutputContract::Has_Recommended_Suffix(
        nullptr, SpongeH5OutputContract::kTrajectorySuffix));
    REQUIRE_TRUE(!SpongeH5OutputContract::Has_Recommended_Suffix(
        "prod.spg.h5md", nullptr));

    CONTROLLER controller;
    REQUIRE_EQ(SpongeH5OutputContract::Trajectory_Chunk_Size(&controller),
               SpongeH5OutputContract::kDefaultTrajectoryChunkSize);
    controller.Set(SpongeH5OutputContract::kTrajectoryChunkSizeKey, "13");
    REQUIRE_EQ(SpongeH5OutputContract::Trajectory_Chunk_Size(&controller), 13);
}

static void Test_Legacy_Output_Plan_All_Keys()
{
    static constexpr const char* legacy_keys[] = {
        "mdout", "mdinfo", "crd", "box", "vel", "frc", "rst"};

    {
        CONTROLLER controller;
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(!plan.any_h5_output_enabled);
        REQUIRE_TRUE(plan.legacy.default_enabled);
        for (const char* key : legacy_keys)
        {
            REQUIRE_TRUE(plan.legacy.Enabled(key));
            REQUIRE_TRUE(!plan.legacy.Explicitly_Requested(key));
        }
        REQUIRE_TRUE(!plan.legacy.Enabled("not_a_legacy_sidecar"));
    }

    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kObservablePathKey,
                       "analysis.obs.spg.h5md");
        controller.Set("vel", "legacy.vel");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(plan.valid);
        REQUIRE_TRUE(plan.any_h5_output_enabled);
        REQUIRE_TRUE(!plan.legacy.default_enabled);
        REQUIRE_TRUE(!plan.legacy.Enabled("mdout"));
        REQUIRE_TRUE(plan.legacy.Enabled("vel"));
        REQUIRE_TRUE(plan.legacy.Explicitly_Requested("vel"));
        REQUIRE_EQ(plan.legacy.sidecars.size(), static_cast<std::size_t>(7));
        bool found_vel = false;
        for (const auto& sidecar : plan.legacy.sidecars)
        {
            if (sidecar.key == "vel")
            {
                found_vel = true;
                REQUIRE_EQ(sidecar.path, std::string("legacy.vel"));
            }
        }
        REQUIRE_TRUE(found_vel);
    }
}

static void Test_Resolve_Legacy_Output_Plan_Matrix()
{
    static constexpr const char* legacy_keys[] = {
        "mdout", "mdinfo", "crd", "box", "vel", "frc", "rst"};
    static constexpr const char* legacy_paths[] = {
        "legacy.mdout", "legacy.mdinfo", "legacy.crd", "legacy.box",
        "legacy.vel",   "legacy.frc",    "legacy.rst"};

    {
        auto legacy = SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(nullptr);
        REQUIRE_TRUE(legacy.default_enabled);
        REQUIRE_EQ(legacy.sidecars.size(), static_cast<std::size_t>(7));
        for (const char* key : legacy_keys)
        {
            REQUIRE_TRUE(legacy.Enabled(key));
            REQUIRE_TRUE(!legacy.Explicitly_Requested(key));
        }
        REQUIRE_TRUE(!legacy.Enabled(nullptr));
        REQUIRE_TRUE(!legacy.Explicitly_Requested(nullptr));
        REQUIRE_TRUE(!legacy.Enabled("not_a_legacy_sidecar"));
        REQUIRE_TRUE(!legacy.Explicitly_Requested("not_a_legacy_sidecar"));
    }

    {
        CONTROLLER controller;
        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        REQUIRE_TRUE(legacy.default_enabled);
        REQUIRE_EQ(legacy.sidecars.size(), static_cast<std::size_t>(7));
        for (const char* key : legacy_keys)
        {
            REQUIRE_TRUE(legacy.Enabled(key));
            REQUIRE_TRUE(!legacy.Explicitly_Requested(key));
        }
        REQUIRE_TRUE(!legacy.Enabled(nullptr));
        REQUIRE_TRUE(!legacy.Explicitly_Requested(nullptr));
        REQUIRE_TRUE(!legacy.Enabled("not_a_legacy_sidecar"));
        REQUIRE_TRUE(!legacy.Explicitly_Requested("not_a_legacy_sidecar"));
    }

    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        REQUIRE_TRUE(!legacy.default_enabled);
        REQUIRE_EQ(legacy.sidecars.size(), static_cast<std::size_t>(7));
        for (const char* key : legacy_keys)
        {
            REQUIRE_TRUE(!legacy.Enabled(key));
            REQUIRE_TRUE(!legacy.Explicitly_Requested(key));
        }
        REQUIRE_TRUE(!legacy.Enabled(nullptr));
        REQUIRE_TRUE(!legacy.Explicitly_Requested(nullptr));
        REQUIRE_TRUE(!legacy.Enabled("not_a_legacy_sidecar"));
        REQUIRE_TRUE(!legacy.Explicitly_Requested("not_a_legacy_sidecar"));
    }

    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kObservablePathKey,
                       "analysis.obs.spg.h5md");
        for (std::size_t i = 0;
             i < sizeof(legacy_keys) / sizeof(legacy_keys[0]); ++i)
        {
            controller.Set(legacy_keys[i], legacy_paths[i]);
        }

        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        REQUIRE_TRUE(!legacy.default_enabled);
        REQUIRE_EQ(legacy.sidecars.size(), static_cast<std::size_t>(7));
        for (std::size_t i = 0;
             i < sizeof(legacy_keys) / sizeof(legacy_keys[0]); ++i)
        {
            REQUIRE_TRUE(legacy.Enabled(legacy_keys[i]));
            REQUIRE_TRUE(legacy.Explicitly_Requested(legacy_keys[i]));
            REQUIRE_EQ(legacy.sidecars[i].key, std::string(legacy_keys[i]));
            REQUIRE_EQ(legacy.sidecars[i].path, std::string(legacy_paths[i]));
        }
        REQUIRE_TRUE(!legacy.Enabled(nullptr));
        REQUIRE_TRUE(!legacy.Explicitly_Requested(nullptr));
        REQUIRE_TRUE(!legacy.Enabled("not_a_legacy_sidecar"));
        REQUIRE_TRUE(!legacy.Explicitly_Requested("not_a_legacy_sidecar"));
    }
}

static void Test_Explicit_Legacy_Sidecar_Collection()
{
    {
        CONTROLLER controller;
        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        std::vector<std::string> keys = {"stale_key"};
        std::vector<std::string> paths = {"stale_path"};

        SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(legacy, &keys,
                                                             &paths);

        REQUIRE_TRUE(keys.empty());
        REQUIRE_TRUE(paths.empty());
    }

    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set("rst", "legacy.rst");
        controller.Set("mdout", "legacy.mdout");

        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        std::vector<std::string> keys = {"stale_key"};
        std::vector<std::string> paths = {"stale_path"};

        SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(legacy, &keys,
                                                             &paths);

        REQUIRE_EQ(keys.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(paths.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(keys[0], std::string("mdout"));
        REQUIRE_EQ(paths[0], std::string("legacy.mdout"));
        REQUIRE_EQ(keys[1], std::string("rst"));
        REQUIRE_EQ(paths[1], std::string("legacy.rst"));
    }

    {
        CONTROLLER controller;
        controller.Set("mdinfo", "legacy.mdinfo");
        auto legacy =
            SpongeH5OutputPlan::Resolve_Legacy_Output_Plan(&controller);
        std::vector<std::string> paths = {"unchanged"};

        SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(legacy, nullptr,
                                                             &paths);

        REQUIRE_EQ(paths.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(paths[0], std::string("unchanged"));
    }
}

static void Test_Output_Path_Suffix_Flags_Are_Non_Fatal()
{
    CONTROLLER controller;
    controller.Set(SpongeH5OutputContract::kTrajectoryPathKey, "traj.h5");
    controller.Set(SpongeH5OutputContract::kRestartPathKey, "restart.h5");
    controller.Set(SpongeH5OutputContract::kObservablePathKey,
                   "observables.h5");

    auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);

    REQUIRE_TRUE(plan.valid);
    REQUIRE_TRUE(plan.trajectory.enabled);
    REQUIRE_TRUE(plan.restart.enabled);
    REQUIRE_TRUE(plan.observable.enabled);
    REQUIRE_TRUE(!plan.trajectory.has_recommended_suffix);
    REQUIRE_TRUE(!plan.restart.has_recommended_suffix);
    REQUIRE_TRUE(!plan.observable.has_recommended_suffix);
    REQUIRE_EQ(plan.trajectory.derived_shard_root,
               std::string("traj.h5.shards"));
}

static void Test_Complete_Prefix_Repair_Requires_Vds_Trajectory()
{
    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                       "complete_prefix");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(plan.error_message.find("requires") != std::string::npos);
    }

    {
        CONTROLLER controller;
        controller.Set(SpongeH5OutputContract::kTrajectoryPathKey,
                       "prod.spg.h5md");
        controller.Set(SpongeH5OutputContract::kTrajectoryRepairPolicyKey,
                       "complete_prefix");
        auto plan = SpongeH5OutputPlan::Resolve_Output_Plan(&controller, false);
        REQUIRE_TRUE(!plan.valid);
        REQUIRE_TRUE(!plan.trajectory.vds);
        REQUIRE_TRUE(plan.error_message.find("requires") != std::string::npos);
    }
}

int main()
{
    return Run_Test(
        []
        {
            Test_Defaults_And_Legacy_Gating();
            Test_Contract_Helper_Functions();
            Test_Grouped_Output_H5_Key_Names_And_Legacy_Gating();
            Test_Raw_Input_Restart_Lineage_Plan();
            Test_Empty_H5_Output_Paths_Do_Not_Enable_Bundles();
            Test_Null_Controller_Resolver();
            Test_Output_Selectors_Do_Not_Enable_H5_Without_Path();
            Test_H5_Output_Path_Keys_Enable_Only_Their_Bundle();
            Test_All_H5_Output_Bundles_Can_Be_Enabled_Together();
            Test_Bool_Parsing_Text_Variants();
            Test_Trajectory_Vds_Repair_Policy();
            Test_Repair_Policy_Alias();
            Test_Repair_Policy_Is_Case_Insensitive();
            Test_Restart_And_Observable_Paths();
            Test_Invalid_Values();
            Test_Throw_On_Error_Uses_Controller_Error_Path();
            Test_Suffix_And_Shard_Derivation();
            Test_Contract_Helper_Edge_Cases();
            Test_Legacy_Output_Plan_All_Keys();
            Test_Resolve_Legacy_Output_Plan_Matrix();
            Test_Explicit_Legacy_Sidecar_Collection();
            Test_Output_Path_Suffix_Flags_Are_Non_Fatal();
            Test_Complete_Prefix_Repair_Requires_Vds_Trajectory();
        });
}
