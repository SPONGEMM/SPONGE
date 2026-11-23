#pragma once
#if defined(__linux__)
#include <filesystem>

#include "state.h"

namespace sponge_exchange
{
class Client
{
   public:
    void Initialize();
    bool Enabled() const { return socket_ >= 0; }
    void Handshake(const Window& window);
    bool Due(int step) const
    {
        return Enabled() && step > 0 && step == next_.step;
    }
    void Exchange(int step, const Window& window);
    bool Checkpoints_Enabled() const { return !checkpoint_directory_.empty(); }
    bool Checkpoint_Due(int step) const
    {
        return pending_checkpoint_ == step && step > 0;
    }
    std::filesystem::path Checkpoint_Path(int step) const;
    void Acknowledge_Checkpoint(int step);

   private:
    int socket_ = -1;
    int coordinate_fd_ = -1;
    int coordinate_rank_ = -1;
    std::vector<float> original_;
    std::vector<float> incoming_;
    Assignment next_{};
    int pending_checkpoint_ = 0;
    std::filesystem::path checkpoint_directory_;
    std::string rank_;
};
}  // namespace sponge_exchange
#endif
