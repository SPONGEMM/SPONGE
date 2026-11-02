#include "h5_bundle_test_common.hpp"
#include "utils/h5md/completion_tracker.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static void Test_State_Machine_Without_Writer()
{
    OutputCompletionTracker tracker(nullptr);

    REQUIRE_TRUE(tracker.Mark_Open());
    REQUIRE_EQ(tracker.State().status, FileStatus::open);
    REQUIRE_TRUE(tracker.Begin_Frame(10, 0.5));
    REQUIRE_TRUE(tracker.Has_Incomplete_Frame());
    REQUIRE_TRUE(!tracker.Begin_Frame(11, 0.6));
    REQUIRE_EQ(
        tracker.Last_Error(),
        std::string("cannot begin a new frame before completing current"));
    REQUIRE_TRUE(!tracker.Mark_Finalized());
    REQUIRE_EQ(tracker.Last_Error(),
               std::string("cannot finalize while a frame is incomplete"));
    REQUIRE_TRUE(tracker.Complete_Frame());
    REQUIRE_TRUE(!tracker.Has_Incomplete_Frame());
    REQUIRE_EQ(tracker.State().frame_count, static_cast<int64_t>(1));
    REQUIRE_EQ(tracker.State().last_complete_step, static_cast<int64_t>(10));
    REQUIRE_TRUE(tracker.Mark_Closing());
    REQUIRE_EQ(tracker.State().status, FileStatus::closing);
    REQUIRE_TRUE(tracker.Mark_Finalized());
    REQUIRE_EQ(tracker.State().status, FileStatus::finalized);
}

static void Test_State_Machine_Error_Paths_Without_Writer()
{
    {
        OutputCompletionTracker tracker(nullptr);
        REQUIRE_TRUE(!tracker.Complete_Frame());
        REQUIRE_EQ(
            tracker.Last_Error(),
            std::string("cannot complete frame because no frame is open"));
        REQUIRE_EQ(tracker.State().frame_count, static_cast<int64_t>(0));
        REQUIRE_EQ(tracker.State().last_complete_step,
                   static_cast<int64_t>(-1));
    }
    {
        OutputCompletionTracker tracker(nullptr);
        REQUIRE_TRUE(tracker.Mark_Open());
        REQUIRE_TRUE(tracker.Begin_Frame(20, 1.0));
        REQUIRE_TRUE(!tracker.Mark_Closing());
        REQUIRE_EQ(tracker.Last_Error(),
                   std::string("cannot close while a frame is incomplete"));
        REQUIRE_EQ(tracker.State().status, FileStatus::open);
        REQUIRE_TRUE(tracker.Has_Incomplete_Frame());
        REQUIRE_TRUE(tracker.Complete_Frame());
        REQUIRE_TRUE(tracker.Mark_Closing());
        REQUIRE_EQ(tracker.State().status, FileStatus::closing);
    }
    {
        OutputCompletionTracker tracker(nullptr);
        REQUIRE_TRUE(tracker.Mark_Open());
        REQUIRE_TRUE(tracker.Mark_Failed("manual failure"));
        REQUIRE_EQ(tracker.State().status, FileStatus::failed);
        REQUIRE_EQ(tracker.Last_Error(), std::string("manual failure"));
        REQUIRE_TRUE(!tracker.Has_Incomplete_Frame());
    }
}

static void Test_State_Machine_Writes_Metadata()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    H5MDWriter writer(&backend);
    WriterOptions options;
    options.path = "completion.spg.h5md";
    REQUIRE_TRUE(writer.Open(options));

    OutputCompletionTracker tracker(&writer);
    REQUIRE_TRUE(tracker.Mark_Open());
    REQUIRE_TRUE(tracker.Begin_Frame(5, 0.25));
    REQUIRE_TRUE(tracker.Complete_Frame());
    REQUIRE_TRUE(tracker.Mark_Failed("synthetic failure"));

    REQUIRE_EQ(log->strings[path::output_status], std::string("failed"));
    REQUIRE_EQ(log->strings[path::output_error],
               std::string("synthetic failure"));
    REQUIRE_TRUE(log->datasets.count(path::output_frame_count) != 0);
    REQUIRE_TRUE(log->datasets.count(path::output_last_complete_step) != 0);
    REQUIRE_TRUE(log->datasets.count(path::output_last_complete_time) != 0);
    REQUIRE_TRUE(log->append_counts[path::output_frame_count] >= 3);
}

static VdsShardManifestEntry Manifest_Entry(int64_t index, int64_t start,
                                            int64_t count,
                                            const std::string& status)
{
    VdsShardManifestEntry entry;
    entry.index = index;
    entry.frame_start = start;
    entry.frame_count = count;
    entry.status = status;
    return entry;
}

static void Test_Manifest_Validation_Strict()
{
    {
        std::vector<VdsShardManifestEntry> manifest = {
            Manifest_Entry(0, 0, 2, "complete"),
            Manifest_Entry(1, 2, 3, "complete")};
        auto report = Validate_Complete_Manifest(manifest, false);
        REQUIRE_TRUE(report.valid);
        REQUIRE_EQ(report.complete_shard_count, static_cast<int64_t>(2));
        REQUIRE_EQ(report.complete_frame_count, static_cast<int64_t>(5));
    }
    {
        std::vector<VdsShardManifestEntry> manifest = {
            Manifest_Entry(0, 0, 2, "open")};
        auto report = Validate_Complete_Manifest(manifest, false);
        REQUIRE_TRUE(!report.valid);
        REQUIRE_EQ(report.error_message,
                   std::string("manifest contains incomplete shard"));
    }
    {
        std::vector<VdsShardManifestEntry> manifest = {
            Manifest_Entry(0, 0, 2, "complete"),
            Manifest_Entry(2, 2, 1, "complete")};
        auto report = Validate_Complete_Manifest(manifest, false);
        REQUIRE_TRUE(!report.valid);
        REQUIRE_EQ(report.error_message,
                   std::string("manifest shard indices are not contiguous"));
    }
    {
        std::vector<VdsShardManifestEntry> manifest = {
            Manifest_Entry(0, 0, 2, "complete"),
            Manifest_Entry(1, 3, 1, "complete")};
        auto report = Validate_Complete_Manifest(manifest, false);
        REQUIRE_TRUE(!report.valid);
        REQUIRE_EQ(report.error_message,
                   std::string("manifest frame ranges are not contiguous"));
    }
    {
        std::vector<VdsShardManifestEntry> manifest = {
            Manifest_Entry(0, 0, 0, "complete")};
        auto report = Validate_Complete_Manifest(manifest, false);
        REQUIRE_TRUE(!report.valid);
        REQUIRE_EQ(
            report.error_message,
            std::string(
                "manifest shard must contain at least one stream frame"));
    }
    {
        VdsShardManifestEntry observable_only =
            Manifest_Entry(0, 0, 0, "complete");
        observable_only.observable_frame_count = 2;
        auto report = Validate_Complete_Manifest({observable_only}, false);
        REQUIRE_TRUE(report.valid);
        REQUIRE_EQ(report.complete_shard_count, static_cast<int64_t>(1));
        REQUIRE_EQ(report.complete_frame_count, static_cast<int64_t>(0));
    }
}

static void Test_Manifest_Validation_Repair_Prefix()
{
    std::vector<VdsShardManifestEntry> manifest = {
        Manifest_Entry(0, 0, 2, "complete"),
        Manifest_Entry(1, 2, 3, "complete"), Manifest_Entry(2, 5, 1, "open")};
    auto report = Validate_Complete_Manifest(manifest, true);
    REQUIRE_TRUE(report.valid);
    REQUIRE_EQ(report.complete_shard_count, static_cast<int64_t>(2));
    REQUIRE_EQ(report.complete_frame_count, static_cast<int64_t>(5));
}

int main()
{
    return Run_Test(
        []
        {
            Test_State_Machine_Without_Writer();
            Test_State_Machine_Error_Paths_Without_Writer();
            Test_State_Machine_Writes_Metadata();
            Test_Manifest_Validation_Strict();
            Test_Manifest_Validation_Repair_Prefix();
        });
}
