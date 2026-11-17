#include "client.h"
#if defined(__linux__)
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

namespace sponge_exchange
{
namespace
{
int Nonnegative(const char* value)
{
    if (!value) throw std::runtime_error("missing replica exchange integer");
    std::size_t length = 0;
    const auto number = std::stol(value, &length);
    if (value[length] || number < 0 || number > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid replica exchange integer");
    return static_cast<int>(number);
}

void Transfer(int fd, std::vector<float>& data, int rank, bool write)
{
    const auto bytes = data.size() * sizeof(float);
    auto* buffer = reinterpret_cast<char*>(data.data());
    const auto offset = static_cast<off_t>(rank) * bytes;
    std::size_t done = 0;
    while (done < bytes)
    {
        const auto result =
            write ? pwrite(fd, buffer + done, bytes - done, offset + done)
                  : pread(fd, buffer + done, bytes - done, offset + done);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0)
            throw std::runtime_error("shared coordinate transfer failed");
        done += result;
    }
}

}  // namespace

void Client::Initialize()
{
    const char* descriptor = std::getenv("SPONGE_EXCHANGE_FD");
    if (!descriptor) return;
    socket_ = Nonnegative(descriptor);
    if (const char* shared = std::getenv("SPONGE_EXCHANGE_COORDINATE_FD"))
    {
        coordinate_fd_ = Nonnegative(shared);
        coordinate_rank_ = Nonnegative(std::getenv("SPONGE_EXCHANGE_RANK"));
    }
    if (const char* directory = std::getenv("SPONGE_EXCHANGE_CHECKPOINT_DIR"))
    {
        checkpoint_directory_ = directory;
        const char* rank = std::getenv("SPONGE_EXCHANGE_RANK");
        if (!rank || !*rank ||
            std::string(rank).find_first_not_of("0123456789") !=
                std::string::npos)
            throw std::runtime_error("invalid checkpoint rank");
        rank_ = rank;
    }
}

void Client::Handshake(const Window& window)
{
    const auto& config = window.configuration;
    if (coordinate_fd_ < 0)
        throw std::runtime_error("missing replica coordinate channel");
    original_.resize(static_cast<std::size_t>(config.atom_count) * 6 +
                     box_values);
    incoming_.resize(original_.size());
    Send(socket_, config);
    next_ = Receive<Assignment>(socket_);
    if (next_.step < config.first_step || next_.step >= config.final_step)
        throw std::runtime_error("invalid initial exchange assignment");
}

void Client::Exchange(int step, const Window& window)
{
    if (step != next_.step)
        throw std::runtime_error("invalid replica exchange step");
    const double current = window.energy();
    const bool paired = next_.peer >= 0;
    if (paired)
    {
        window.capture(original_);
        Transfer(coordinate_fd_, original_, coordinate_rank_, true);
    }
    Send(socket_, Coordinates_Ready{step});
    if (Receive<Coordinates_Ready>(socket_).step != step)
        throw std::runtime_error("invalid coordinate probe signal");
    double other = current;
    if (paired)
    {
        if (next_.peer == coordinate_rank_ ||
            !std::isfinite(next_.peer_temperature) ||
            next_.peer_temperature <= 0)
            throw std::runtime_error("invalid replica exchange partner");
        Transfer(coordinate_fd_, incoming_, next_.peer, false);
        window.install(incoming_, std::sqrt(window.configuration.temperature /
                                            next_.peer_temperature));
        other = window.energy();
    }
    Send(socket_, Report{step, current, other});
    const auto decision = Receive<Decision>(socket_);
    if (decision.step != step || decision.checkpoint < 0 ||
        decision.checkpoint > 1 ||
        (decision.checkpoint && !Checkpoints_Enabled()) ||
        (decision.transfer != 0 && (decision.transfer != 1 || !paired)) ||
        (decision.next.step != 0 && decision.next.step <= step))
        throw std::runtime_error("invalid replica exchange decision");
    if (paired && !decision.transfer) window.install(original_, 1.0);
    next_ = decision.next;
    if (decision.checkpoint) pending_checkpoint_ = step;
}

std::filesystem::path Client::Checkpoint_Path(int step) const
{
    return checkpoint_directory_ /
           ("rank-" + rank_ + "-step-" + std::to_string(step) + ".spgr.h5");
}

void Client::Acknowledge_Checkpoint(int step)
{
    if (!Checkpoint_Due(step))
        throw std::runtime_error("unexpected replica checkpoint");
    const auto path = Checkpoint_Path(step);
    const auto bytes = std::filesystem::file_size(path);
    if (!bytes) throw std::runtime_error("empty replica checkpoint");
    Send(socket_, CheckpointAck{step, bytes});
    pending_checkpoint_ = 0;
}
}  // namespace sponge_exchange
#endif
