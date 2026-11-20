#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include "protocol.h"

namespace sponge_exchange
{
struct SavedReplica
{
    double temperature;
    double pressure;
    std::filesystem::path snapshot;
    std::uint64_t bytes;
    std::uint64_t phase_space;
};

struct SavedGroup
{
    int npt = 0;
    int step = 0;
    int interval = 0;
    bool exchange = true;
    std::size_t completed_rounds = 0;
    std::mt19937_64 random;
    std::vector<std::size_t> walkers;
    std::vector<SavedReplica> replicas;
};

class Checkpoints
{
   public:
    Checkpoints(const std::filesystem::path& manifest, bool resume);
    ~Checkpoints();
    Checkpoints(const Checkpoints&) = delete;
    Checkpoints& operator=(const Checkpoints&) = delete;

    const std::filesystem::path& Directory() const { return directory_; }
    SavedGroup Load() const;
    void Publish(const SavedGroup& group) const;
    std::filesystem::path Snapshot(std::size_t rank, int step) const;

   private:
    std::filesystem::path manifest_;
    std::filesystem::path directory_;
    int lock_ = -1;
};
}  // namespace sponge_exchange
