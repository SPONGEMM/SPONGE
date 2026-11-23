#include "process_group.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace sponge_exchange
{
namespace
{
volatile std::sig_atomic_t interrupted = 0;
void Request_Stop(int) { interrupted = 1; }
}  // namespace

bool Interrupted() { return interrupted != 0; }

void Register_Stop_Handlers()
{
    struct sigaction action{};
    action.sa_handler = Request_Stop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) ||
        sigaction(SIGTERM, &action, nullptr))
        throw std::runtime_error("cannot register manager signal handlers");
}

ReplicaGroup::~ReplicaGroup()
{
    for (const auto& replica : replicas)
    {
        if (replica.fd >= 0) close(replica.fd);
        if (replica.pid > 0) kill(replica.pid, SIGKILL);
    }
    for (const auto& replica : replicas)
    {
        if (replica.pid > 0)
            while (waitpid(replica.pid, nullptr, 0) < 0 && errno == EINTR)
            {
            }
    }
}

void ReplicaGroup::Start(const std::string& binary,
                         const std::string& directory, const std::string& gpu,
                         const std::string& interval)
{
    if (Interrupted()) throw std::runtime_error("interrupted");
    const std::string rank = std::to_string(replicas.size());
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets))
        throw std::runtime_error("cannot create replica socket");
    const auto pid = fork();
    if (pid == 0)
    {
        close(sockets[0]);
        const auto fd = std::to_string(sockets[1]);
        if (fcntl(sockets[1], F_SETFD, 0) ||
            setenv("SPONGE_EXCHANGE_FD", fd.c_str(), 1) ||
            setenv("SPONGE_EXCHANGE_INTERVAL", interval.c_str(), 1) ||
            setenv("SPONGE_EXCHANGE_RANK", rank.c_str(), 1) ||
            (gpu != "-" && setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1)) ||
            chdir(directory.c_str()))
        {
            std::perror("prepare replica");
            _exit(126);
        }
        const int log = open("sponge.log", O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (log < 0 || dup2(log, STDOUT_FILENO) < 0 ||
            dup2(log, STDERR_FILENO) < 0)
        {
            std::perror("open replica log");
            _exit(126);
        }
        if (log > STDERR_FILENO) close(log);
        execl(binary.c_str(), binary.c_str(), "-mdin", "mdin.spg.toml",
              static_cast<char*>(nullptr));
        std::perror("launch SPONGE");
        _exit(127);
    }
    close(sockets[1]);
    if (pid < 0)
    {
        close(sockets[0]);
        throw std::runtime_error("cannot launch replica");
    }
    replicas.push_back({pid, sockets[0]});
}

void ReplicaGroup::Reap(std::size_t index)
{
    auto& replica = replicas[index];
    int status = 0;
    pid_t result;
    do
    {
        result = waitpid(replica.pid, &status, 0);
    } while (result < 0 && errno == EINTR && !Interrupted());
    if (result < 0) throw std::runtime_error("cannot wait for replica");
    replica.pid = -1;
    close(replica.fd);
    replica.fd = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("replica " + std::to_string(index) +
                                 " failed; see its sponge.log");
}

}  // namespace sponge_exchange
