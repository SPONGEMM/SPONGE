#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "process_group.h"
#include "session.h"

using namespace sponge_exchange;

namespace
{
struct Coordinate_Channel
{
    int fd = -1;
    ~Coordinate_Channel()
    {
        if (fd >= 0) close(fd);
    }
};

int Positive_Int(const std::string& value)
{
    std::size_t length = 0;
    const long parsed = std::stol(value, &length);
    if (length != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max())
        throw std::runtime_error("expected positive integer: " + value);
    return static_cast<int>(parsed);
}

std::uint64_t Parse_Seed(const std::string& value)
{
    std::size_t length = 0;
    if (value.empty() || value.front() < '0' || value.front() > '9')
        throw std::runtime_error("expected unsigned exchange seed");
    const auto seed = std::stoull(value, &length);
    if (length != value.size())
        throw std::runtime_error("expected unsigned exchange seed");
    return seed;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        std::filesystem::path manifest;
        int checkpoint_rounds = 0;
        bool resume_requested = false;
        while (argc > 1)
        {
            const std::string option = argv[1];
            if ((option == "--checkpoint" || option == "--resume") && argc > 3)
            {
                if (checkpoint_rounds)
                    throw std::runtime_error(
                        "specify one checkpoint or resume option");
                manifest = argv[2];
                checkpoint_rounds = Positive_Int(argv[3]);
                resume_requested = option == "--resume";
                argv += 3;
                argc -= 3;
            }
            else
                break;
        }
        if (argc < 10 || (argc - 6) % 2)
        {
            std::cerr << "Usage: SPONGE_MANAGER [--checkpoint MANIFEST ROUNDS "
                         "| --resume MANIFEST ROUNDS] "
                         "SPONGE INTERVAL TIMEOUT_MS SEED "
                         "exchange|sync GPU DIRECTORY GPU DIRECTORY [GPU "
                         "DIRECTORY ...]\n"
                         "Use - for GPU to inherit CUDA_VISIBLE_DEVICES. "
                         "Each directory needs its own mdin.spg.toml.\n";
            return 2;
        }
        const auto binary = std::filesystem::canonical(argv[1]).string();
        const auto interval = Positive_Int(argv[2]);
        const auto timeout_ms = Positive_Int(argv[3]);
        const auto seed = Parse_Seed(argv[4]);
        const std::string mode = argv[5];
        if (mode != "exchange" && mode != "sync")
            throw std::runtime_error("mode must be exchange or sync");
        std::vector<std::filesystem::path> directories;
        for (int i = 7; i < argc; i += 2)
        {
            auto directory = std::filesystem::canonical(argv[i]);
            if (!std::filesystem::is_regular_file(directory /
                                                  "mdin.spg.toml") ||
                std::find(directories.begin(), directories.end(), directory) !=
                    directories.end())
                throw std::runtime_error(
                    "each replica needs a distinct directory and "
                    "mdin.spg.toml");
            directories.push_back(std::move(directory));
        }

        std::optional<Checkpoints> checkpoints;
        std::optional<SavedGroup> resume;
        if (checkpoint_rounds)
        {
            checkpoints.emplace(manifest, resume_requested);
            if (resume_requested)
            {
                resume = checkpoints->Load();
                if (resume->interval != interval ||
                    resume->exchange != (mode == "exchange") ||
                    resume->replicas.size() != directories.size())
                    throw std::runtime_error(
                        "resume settings disagree with checkpoint manifest");
            }
            if (setenv("SPONGE_EXCHANGE_CHECKPOINT_DIR",
                       checkpoints->Directory().c_str(), 1))
                throw std::runtime_error("cannot export checkpoint directory");
        }
        else
            unsetenv("SPONGE_EXCHANGE_CHECKPOINT_DIR");
        Coordinate_Channel channel;
        channel.fd = memfd_create("sponge-replica-coordinates", 0);
        if (channel.fd < 0 || setenv("SPONGE_EXCHANGE_COORDINATE_FD",
                                     std::to_string(channel.fd).c_str(), 1))
            throw std::runtime_error(
                "cannot create shared replica coordinate channel");

        Register_Stop_Handlers();
        ReplicaGroup group;
        group.replicas.reserve(directories.size());
        unsetenv("SPONGE_EXCHANGE_RESUME_PATH");
        for (std::size_t i = 0; i < directories.size(); ++i)
        {
            if (resume && setenv("SPONGE_EXCHANGE_RESUME_PATH",
                                 resume->replicas[i].snapshot.c_str(), 1))
                throw std::runtime_error(
                    "cannot export the replica restart path");
            group.Start(binary, directories[i].string(), argv[6 + 2 * i],
                        argv[2]);
        }
        Run(group, interval, timeout_ms, seed, mode == "exchange",
            checkpoints ? &*checkpoints : nullptr, checkpoint_rounds,
            resume ? &*resume : nullptr, channel.fd);
    }
    catch (const std::exception& error)
    {
        std::cerr << "SPONGE_MANAGER: " << error.what() << '\n';
        return 1;
    }
}
