#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

namespace sponge_exchange
{
struct Replica
{
    pid_t pid;
    int fd;
};

class ReplicaGroup
{
   public:
    ReplicaGroup() = default;
    ReplicaGroup(const ReplicaGroup&) = delete;
    ReplicaGroup& operator=(const ReplicaGroup&) = delete;
    ~ReplicaGroup();

    void Start(const std::string& binary, const std::string& directory,
               const std::string& gpu, const std::string& interval);
    void Reap(std::size_t index);

    std::vector<Replica> replicas;
};

bool Interrupted();
void Register_Stop_Handlers();
}  // namespace sponge_exchange
