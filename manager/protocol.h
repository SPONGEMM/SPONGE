#pragma once

#include <sys/socket.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sponge_exchange
{
constexpr std::uint32_t protocol_version = 5;
constexpr std::size_t box_values = 3;

struct Hello
{
    std::uint32_t version;
    int npt;
    int first_step;
    int final_step;
    int atom_count;
    double timestep;
    double temperature;
    double pressure;
    double box_y_over_x;
    double box_z_over_x;
    std::uint64_t phase_space;
};

struct Assignment
{
    int step;
    int peer;
    double peer_temperature;
};

struct Report
{
    int step;
    double current;
    double partner;
};

struct Decision
{
    int step;
    int checkpoint;
    int transfer;
    Assignment next;
};

struct Coordinates_Ready
{
    int step;
};

struct CheckpointAck
{
    int step;
    std::uint64_t bytes;
};

template <typename Message>
void Send(int socket, const Message& message)
{
    ssize_t result;
    do
    {
        result = send(socket, &message, sizeof(message), MSG_NOSIGNAL);
    } while (result < 0 && errno == EINTR);
    if (result != sizeof(message))
        throw std::runtime_error("replica exchange send failed");
}

template <typename Message>
Message Receive(int socket)
{
    Message message{};
    ssize_t result;
    do
    {
        result = recv(socket, &message, sizeof(message), MSG_TRUNC);
    } while (result < 0 && errno == EINTR);
    if (result != sizeof(message))
        throw std::runtime_error(
            "replica exchange connection closed or invalid message");
    return message;
}
}  // namespace sponge_exchange
