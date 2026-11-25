#include "session.h"

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "exchange.h"
#include "process_group.h"

namespace sponge_exchange
{
namespace
{
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

void Wait(std::vector<pollfd>& sockets, int timeout_ms)
{
    for (;;)
    {
        if (Interrupted()) throw std::runtime_error("interrupted");
        const int status = poll(sockets.data(), sockets.size(), timeout_ms);
        if (status < 0 && errno == EINTR) continue;
        if (status <= 0)
            throw std::runtime_error(
                "replica exchange poll failed or timed out");
        return;
    }
}

template <typename Message>
double Gather(ReplicaGroup& group, int timeout_ms,
              std::vector<Message>& messages)
{
    std::vector<pollfd> sockets;
    for (const auto& replica : group.replicas)
        sockets.push_back({replica.fd, POLLIN, 0});

    auto initial = Clock::now();
    std::size_t received = 0;
    while (received < messages.size())
    {
        Wait(sockets, timeout_ms);
        for (std::size_t i = 0; i < sockets.size(); ++i)
        {
            auto& socket = sockets[i];
            if (socket.revents == 0) continue;
            char byte;
            if (!(socket.revents & POLLIN) ||
                recv(socket.fd, &byte, 1, MSG_PEEK) <= 0)
            {
                group.Reap(i);
                throw std::runtime_error("replica " + std::to_string(i) +
                                         " exited before reporting");
            }
            messages[i] = Receive<Message>(socket.fd);
            socket.fd = -1;
            if (received++ == 0) initial = Clock::now();
        }
    }
    return Milliseconds(Clock::now() - initial).count();
}

std::vector<Assignment> Assign(int step, std::size_t round,
                               const std::vector<Hello>& windows)
{
    std::vector<Assignment> assignments(windows.size(), {step, -1, 0});
    for (const auto& [left, right] : Pair_Labels(windows.size(), round))
    {
        assignments[left] = {step, static_cast<int>(right),
                             windows[right].temperature};
        assignments[right] = {step, static_cast<int>(left),
                              windows[left].temperature};
    }
    return assignments;
}

void Validate_Hello(const Hello& message, const Hello& reference)
{
    if (message.version != protocol_version || message.npt != reference.npt ||
        message.npt < 0 || message.npt > 1 ||
        message.first_step != reference.first_step ||
        message.final_step != reference.final_step || message.atom_count <= 0 ||
        message.atom_count != reference.atom_count ||
        !std::isfinite(message.timestep) || message.timestep <= 0 ||
        message.timestep != reference.timestep || message.first_step < 0 ||
        message.final_step <= message.first_step ||
        !std::isfinite(message.temperature) || message.temperature <= 0 ||
        !std::isfinite(message.pressure) || !message.phase_space ||
        message.phase_space != reference.phase_space ||
        (message.npt &&
         (!std::isfinite(message.box_y_over_x) ||
          !std::isfinite(message.box_z_over_x) ||
          std::abs(message.box_y_over_x - reference.box_y_over_x) > 1e-4 ||
          std::abs(message.box_z_over_x - reference.box_z_over_x) > 1e-4)))
        throw std::runtime_error("incompatible replica handshake");
}
}  // namespace

void Run(ReplicaGroup& group, int interval, int timeout_ms, std::uint64_t seed,
         bool exchange, const Checkpoints* checkpoints, int checkpoint_rounds,
         const SavedGroup* resume, int coordinate_fd)
{
    const auto count = group.replicas.size();
    if (coordinate_fd < 0 || count > 100000)
        throw std::runtime_error("invalid replica coordinate channel");
    std::vector<Hello> handshakes(count);
    Gather(group, timeout_ms, handshakes);
    const auto& first = handshakes.front();
    for (const auto& hello : handshakes) Validate_Hello(hello, first);
    const auto values = static_cast<off_t>(first.atom_count) * 6 + box_values;
    if (ftruncate(coordinate_fd, count * values * sizeof(float)))
        throw std::runtime_error("cannot size the shared coordinate channel");

    std::vector<std::size_t> walkers(count);
    std::iota(walkers.begin(), walkers.end(), 0);
    if (resume)
    {
        if (resume->npt != first.npt || resume->step + 1 != first.first_step)
            throw std::runtime_error(
                "replicas did not resume at the checkpoint boundary");
        walkers = resume->walkers;
        for (std::size_t rank = 0; rank < count; ++rank)
        {
            const auto& saved = resume->replicas[rank];
            const auto& actual = handshakes[rank];
            if (saved.temperature != actual.temperature ||
                saved.pressure != actual.pressure ||
                saved.phase_space != actual.phase_space)
                throw std::runtime_error(
                    "replica did not restore its checkpoint state");
        }
    }
    else if (first.first_step > 1)
        throw std::runtime_error("replica restart requires a manager manifest");
    std::int64_t step =
        ((std::max(1, first.first_step) + std::int64_t{interval} - 1) /
         interval) *
        interval;
    if (step >= first.final_step)
        throw std::runtime_error("run has no exchange step");
    std::size_t round = resume ? resume->completed_rounds : 0;
    auto assignments = Assign(static_cast<int>(step), round, handshakes);
    for (std::size_t i = 0; i < count; ++i)
        Send(group.replicas[i].fd, assignments[i]);

    std::vector<Report> reports(count);
    std::mt19937_64 random = resume ? resume->random : std::mt19937_64(seed);
    std::cout << std::setprecision(17);
    for (; step < first.final_step; step += interval, ++round)
    {
        std::vector<Coordinates_Ready> published(count);
        Gather(group, timeout_ms, published);
        for (const auto& message : published)
            if (message.step != step)
                throw std::runtime_error("misaligned coordinate publication");
        for (const auto& replica : group.replicas)
            Send(replica.fd, Coordinates_Ready{static_cast<int>(step)});
        const double arrival = Gather(group, timeout_ms, reports);
        const auto ready = Clock::now();
        for (const auto& report : reports)
        {
            if (report.step != step || !std::isfinite(report.current) ||
                !std::isfinite(report.partner))
                throw std::runtime_error("invalid reduced potential report");
        }
        const auto attempts = Decide(reports, round, random, exchange);
        std::vector<bool> transfers(count, false);
        for (const auto& attempt : attempts)
        {
            if (!attempt.accepted) continue;
            transfers[attempt.left] = transfers[attempt.right] = true;
            std::swap(walkers[attempt.left], walkers[attempt.right]);
        }
        const auto upcoming = step + interval;
        assignments =
            Assign(upcoming < first.final_step ? static_cast<int>(upcoming) : 0,
                   round + 1, handshakes);
        const bool save = checkpoints && (round + 1) % checkpoint_rounds == 0;
        for (std::size_t rank = 0; rank < count; ++rank)
        {
            Send(group.replicas[rank].fd,
                 Decision{static_cast<int>(step), save, transfers[rank],
                          assignments[rank]});
        }
        const double service = Milliseconds(Clock::now() - ready).count();
        for (const auto& attempt : attempts)
            std::cout << "pair " << step << ' ' << attempt.left << ' '
                      << attempt.right << ' ' << attempt.log_acceptance << ' '
                      << attempt.draw << ' ' << attempt.accepted << '\n';
        for (std::size_t rank = 0; rank < count; ++rank)
            std::cout << "label " << step << ' ' << walkers[rank] << ' ' << rank
                      << ' ' << handshakes[rank].temperature << '\n';
        std::cout << "timing " << step << ' ' << arrival << ' ' << service
                  << '\n';
        std::cout.flush();
        if (save)
        {
            std::vector<CheckpointAck> acknowledgements(count);
            Gather(group, timeout_ms, acknowledgements);
            SavedGroup group_state{first.npt, static_cast<int>(step),
                                   interval,  exchange,
                                   round + 1, random,
                                   walkers,   {}};
            group_state.replicas.resize(count);
            for (std::size_t rank = 0; rank < count; ++rank)
            {
                if (acknowledgements[rank].step != step)
                    throw std::runtime_error("misaligned replica checkpoint");
                group_state.replicas[rank] = {
                    handshakes[rank].temperature, handshakes[rank].pressure,
                    checkpoints->Snapshot(rank, static_cast<int>(step)),
                    acknowledgements[rank].bytes, handshakes[rank].phase_space};
            }
            checkpoints->Publish(group_state);
            std::cout << "checkpoint " << step << std::endl;
        }
    }

    std::vector<pollfd> sockets;
    for (const auto& replica : group.replicas)
        sockets.push_back({replica.fd, POLLIN, 0});
    std::size_t finished = 0;
    while (finished != count)
    {
        Wait(sockets, timeout_ms);
        for (std::size_t i = 0; i < count; ++i)
        {
            if (sockets[i].revents == 0) continue;
            char byte;
            if ((sockets[i].revents & POLLIN) &&
                recv(sockets[i].fd, &byte, 1, MSG_PEEK) > 0)
                throw std::runtime_error(
                    "unexpected report after final exchange");
            group.Reap(i);
            sockets[i].fd = -1;
            ++finished;
        }
    }
}
}  // namespace sponge_exchange
