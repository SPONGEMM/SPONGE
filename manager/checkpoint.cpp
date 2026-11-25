#include "checkpoint.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace sponge_exchange
{
namespace
{
void Sync_File(const std::filesystem::path& path)
{
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
        throw std::runtime_error(
            "cannot open checkpoint for synchronization: " + path.string());
    int result;
    do
    {
        result = fsync(descriptor);
    } while (result < 0 && errno == EINTR);
    close(descriptor);
    if (result != 0)
        throw std::runtime_error("cannot synchronize checkpoint: " +
                                 path.string());
}

void Verify_Snapshot(const SavedReplica& replica)
{
    if (!replica.snapshot.is_absolute() || !replica.bytes ||
        !std::filesystem::is_regular_file(replica.snapshot) ||
        std::filesystem::file_size(replica.snapshot) != replica.bytes ||
        !std::isfinite(replica.temperature) || replica.temperature <= 0 ||
        !std::isfinite(replica.pressure))
        throw std::runtime_error("invalid replica checkpoint: " +
                                 replica.snapshot.string());
}
}  // namespace

Checkpoints::Checkpoints(const std::filesystem::path& manifest, bool resume)
    : manifest_(std::filesystem::absolute(manifest))
{
    auto pattern =
        (manifest_.parent_path() / "replica-checkpoint-XXXXXX").string();
    lock_ = open((manifest_.string() + ".lock").c_str(),
                 O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_ < 0 || flock(lock_, LOCK_EX | LOCK_NB))
    {
        if (lock_ >= 0) close(lock_);
        throw std::runtime_error("cannot lock replica checkpoint manifest");
    }
    try
    {
        if (std::filesystem::exists(manifest_) != resume)
            throw std::runtime_error(
                resume ? "resume manifest does not exist"
                       : "checkpoint manifest already exists; use --resume");
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error(
                "cannot create replica checkpoint directory");
        directory_ = pattern;
    }
    catch (...)
    {
        close(lock_);
        throw;
    }
}

Checkpoints::~Checkpoints() { close(lock_); }

std::filesystem::path Checkpoints::Snapshot(std::size_t rank, int step) const
{
    return directory_ / ("rank-" + std::to_string(rank) + "-step-" +
                         std::to_string(step) + ".spgr.h5");
}

SavedGroup Checkpoints::Load() const
{
    std::ifstream stream(manifest_);
    std::string magic;
    std::size_t count;
    SavedGroup group;
    if (!(stream >> magic >> group.npt >> group.step >> group.interval >>
          group.exchange >> group.completed_rounds >> count) ||
        magic != "sponge-exchange-3" || group.npt < 0 || group.npt > 1 ||
        group.step <= 0 || group.step == INT_MAX || group.interval <= 0 ||
        group.step % group.interval || group.completed_rounds == 0 ||
        count < 2 || count > 100000 || !(stream >> group.random))
        throw std::runtime_error("invalid replica checkpoint manifest");
    group.walkers.resize(count);
    group.replicas.resize(count);
    std::vector<bool> seen(count);
    for (auto& rank : group.walkers)
    {
        if (!(stream >> rank) || rank >= count || seen[rank])
            throw std::runtime_error("invalid checkpoint label map");
        seen[rank] = true;
    }
    for (auto& replica : group.replicas)
    {
        std::string name;
        if (!(stream >> replica.temperature >> replica.pressure >>
              replica.bytes >> replica.phase_space >> std::quoted(name)))
            throw std::runtime_error("incomplete replica checkpoint manifest");
        replica.snapshot = name;
        Verify_Snapshot(replica);
    }
    stream >> std::ws;
    if (!stream.eof()) throw std::runtime_error("trailing checkpoint data");
    return group;
}

void Checkpoints::Publish(const SavedGroup& group) const
{
    for (const auto& replica : group.replicas)
    {
        Verify_Snapshot(replica);
        Sync_File(replica.snapshot);
    }
    Sync_File(directory_);
    const auto temporary = directory_ / "manifest.tmp";
    std::ofstream stream(temporary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "sponge-exchange-3 " << group.npt << ' ' << group.step << ' '
           << group.interval << ' ' << group.exchange << ' '
           << group.completed_rounds << ' ' << group.replicas.size() << '\n'
           << group.random << '\n';
    for (auto rank : group.walkers) stream << rank << ' ';
    stream << '\n' << std::setprecision(17);
    for (const auto& replica : group.replicas)
        stream << replica.temperature << ' ' << replica.pressure << ' '
               << replica.bytes << ' ' << replica.phase_space << ' '
               << std::quoted(replica.snapshot.string()) << '\n';
    stream.close();
    Sync_File(temporary);
    std::filesystem::rename(temporary, manifest_);
    Sync_File(manifest_.parent_path());
}
}  // namespace sponge_exchange
