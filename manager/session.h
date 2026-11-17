#pragma once

#include "checkpoint.h"

namespace sponge_exchange
{
class ReplicaGroup;
void Run(ReplicaGroup& group, int interval, int timeout_ms, std::uint64_t seed,
         bool exchange, const Checkpoints* checkpoints, int checkpoint_rounds,
         const SavedGroup* resume, int coordinate_fd);
}  // namespace sponge_exchange
