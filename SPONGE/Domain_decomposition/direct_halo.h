#pragma once
#include <cstdint>
#include <vector>

#include "../common.h"

struct DOMAIN_INFORMATION;
struct MD_INFORMATION;
struct CONTROLLER;

// One plan per DD layout. Owned rows are immutable during its lifetime;
// receives are disjoint owner-ordered spans. Reverse is the additive transpose.
class DIRECT_HALO_PLAN
{
   public:
    bool active = false;
    void Build(DOMAIN_INFORMATION& dd, CONTROLLER* controller,
               MD_INFORMATION* md, float radius, bool preserve_classic,
               float classic_radius);
    void Forward(float* rows, int width);
    void Reverse(float* rows, int width);
    void Clear();

   private:
    int rank_ = 0, size_ = 1, owned_ = 0;
    std::vector<int> send_counts_, recv_counts_, send_offsets_, recv_offsets_;
    std::vector<int> peers_;
    int send_total_ = 0;
    int* send_rows_ = nullptr;
    size_t send_rows_capacity_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_capacity_ = 0;
#ifdef USE_MPI
    D_MPI_Request requests_[2];
    bool streams_ready_ = false;
#endif
    void Invalidate_Mapping();
    void Reserve_Send_Rows(size_t rows);
    void Reserve(size_t values);
};
