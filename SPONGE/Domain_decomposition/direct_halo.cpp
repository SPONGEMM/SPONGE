#include "direct_halo.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../neighbor_list/neighbor_list.h"
#include "../utils/mlip_geometry.hpp"
#include "Domain_decomposition.h"

namespace
{
struct Bounds
{
    VECTOR lo, hi, head;
    int empty;
};
__host__ __device__ double Axis_Gap(float alo, float ahi, float blo, float bhi,
                                    float length)
{
    // Canonical intervals lie in [0,L]. The nearest periodic intervals are
    // attained at shifts -L, 0, L regardless of radius; this is a distance
    // lower bound, not an assumption on the number of edge images.
    double gap = length;
    for (int q = -1; q <= 1; ++q)
    {
        double low = double(blo) + q * double(length),
               high = double(bhi) + q * double(length);
        double d = alo > high ? alo - high : (low > ahi ? low - ahi : 0.0);
        gap = fmin(gap, d);
    }
    return gap;
}
__host__ __device__ bool Overlap(Bounds a, Bounds b, VECTOR box, float radius)
{
    return !a.empty && !b.empty &&
           Axis_Gap(a.lo.x, a.hi.x, b.lo.x, b.hi.x, box.x) <= radius &&
           Axis_Gap(a.lo.y, a.hi.y, b.lo.y, b.hi.y, box.y) <= radius &&
           Axis_Gap(a.lo.z, a.hi.z, b.lo.z, b.hi.z, box.z) <= radius;
}
__device__ VECTOR Classic_Canonical(VECTOR v, VECTOR box)
{
    v.x -= floorf(v.x / box.x) * box.x;
    v.y -= floorf(v.y / box.y) * box.y;
    v.z -= floorf(v.z / box.z) * box.z;
    return v;
}
__device__ Bounds Point_Bounds(VECTOR raw, VECTOR box)
{
    Bounds b;
    b.empty = 0;
    b.head = {0, 0, 0};
    const double x = MlipGeometry::Canonical(raw.x, box.x);
    const double y = MlipGeometry::Canonical(raw.y, box.y);
    const double z = MlipGeometry::Canonical(raw.z, box.z);
    // Enclose the canonical value, including the double addition on wrapping
    // negative remainders. nextafter is a representation bound, not a
    // tolerance.
    b.lo = {MlipGeometry::Lower(nextafter(x, -HUGE_VAL)),
            MlipGeometry::Lower(nextafter(y, -HUGE_VAL)),
            MlipGeometry::Lower(nextafter(z, -HUGE_VAL))};
    b.hi = {MlipGeometry::Upper(nextafter(x, HUGE_VAL)),
            MlipGeometry::Upper(nextafter(y, HUGE_VAL)),
            MlipGeometry::Upper(nextafter(z, HUGE_VAL))};
    return b;
}
__host__ __device__ void Include(Bounds& b, const Bounds& p)
{
    b.lo.x = fminf(b.lo.x, p.lo.x);
    b.hi.x = fmaxf(b.hi.x, p.hi.x);
    b.lo.y = fminf(b.lo.y, p.lo.y);
    b.hi.y = fmaxf(b.hi.y, p.hi.y);
    b.lo.z = fminf(b.lo.z, p.lo.z);
    b.hi.z = fmaxf(b.hi.z, p.hi.z);
}
__global__ void Group_Bounds(int groups, const int* starts, const int* lengths,
                             const VECTOR* crd, VECTOR box, Bounds* bounds)
{
    SIMPLE_DEVICE_FOR(g, groups)
    {
        Bounds b;
        b.empty = lengths[g] == 0;
        b.lo = box;
        b.hi = {0, 0, 0};
        b.head = {0, 0, 0};
        for (int j = 0; j < lengths[g]; ++j)
        {
            VECTOR raw = crd[starts[g] + j];
            if (j == 0) b.head = Classic_Canonical(raw, box);
            Include(b, Point_Bounds(raw, box));
        }
        bounds[g] = b;
    }
}
// Query current owned-only buckets, marking each target atom once across all
// candidate groups from a given source. Replies never become resident ghosts.
__global__ void Query_Buckets(int grids, int capacity, const int* counts,
                              const int* atoms, const VECTOR* crd, int queries,
                              const Bounds* bounds, VECTOR box, float radius,
                              int* selected)
{
    SIMPLE_DEVICE_FOR(bin, grids)
    {
        if (!counts[bin])
        {
#ifdef USE_GPU
            return;
#else
            continue;
#endif
        }
        // Bounds of actual bucket contents avoid assumptions about division,
        // terminal-bin width, or the representation of canonical coordinates.
        Bounds b;
        b.empty = 0;
        b.lo = box;
        b.hi = {0, 0, 0};
        for (int j = 0; j < counts[bin]; ++j)
            Include(b, Point_Bounds(crd[atoms[bin * capacity + j]], box));
        bool candidate = false;
        for (int q = 0; q < queries && !candidate; ++q)
            candidate = Overlap(b, bounds[q], box, radius);
        if (!candidate)
        {
#ifdef USE_GPU
            return;
#else
            continue;
#endif
        }
        for (int j = 0; j < counts[bin]; ++j)
        {
            int row = atoms[bin * capacity + j];
            Bounds p = Point_Bounds(crd[row], box);
            for (int q = 0; q < queries; ++q)
                if (Overlap(p, bounds[q], box, radius))
                {
                    selected[row] = 1;
                    break;
                }
        }
    }
}
__global__ void Pack_Replies(int owned, const int* selected, const VECTOR* crd,
                             VECTOR* out, int* count)
{
    SIMPLE_DEVICE_FOR(i, owned)
    {
        if (selected[i]) out[atomicAdd(count, 1)] = crd[i];
    }
}
__global__ void Exact_Groups(int groups, const int* starts, const int* lengths,
                             const VECTOR* crd, const Bounds* bounds,
                             Bounds target_bounds, const VECTOR* target,
                             VECTOR box, float radius, int nx, int ny, int nz,
                             int capacity, const int* counts, const int* atoms,
                             int* selected)
{
    SIMPLE_DEVICE_FOR(g, groups)
    {
        bool hit = false;
        if (!Overlap(bounds[g], target_bounds, box, radius))
        {
            selected[g] = 0;
#ifdef USE_GPU
            return;
#else
            continue;
#endif
        }
        for (int j = 0; j < lengths[g] && !hit; ++j)
        {
            VECTOR a = crd[starts[g] + j];
            int bx = MlipGeometry::Bin(MlipGeometry::Canonical(a.x, box.x),
                                       radius, nx);
            int by = MlipGeometry::Bin(MlipGeometry::Canonical(a.y, box.y),
                                       radius, ny);
            int bz = MlipGeometry::Bin(MlipGeometry::Canonical(a.z, box.z),
                                       radius, nz);
            // floor(L/R) bins: the terminal bin may be wider than R.
            // Two cyclic bins each way conservatively cover the R interval;
            // clamp iteration counts so small dimensions never repeat bins.
            for (int ix = 0; ix < (nx < 5 ? nx : 5) && !hit; ++ix)
                for (int iy = 0; iy < (ny < 5 ? ny : 5) && !hit; ++iy)
                    for (int iz = 0; iz < (nz < 5 ? nz : 5) && !hit; ++iz)
                    {
                        int x = (bx - 2 + nx + ix) % nx,
                            y = (by - 2 + ny + iy) % ny,
                            z = (bz - 2 + nz + iz) % nz;
                        // Adding nx once is insufficient only for nx=1; handle
                        // it.
                        if (x < 0) x += nx;
                        if (y < 0) y += ny;
                        if (z < 0) z += nz;
                        int bin = (x * ny + y) * nz + z;
                        for (int k = 0; k < counts[bin]; ++k)
                        {
                            VECTOR b = target[atoms[bin * capacity + k]];
                            double dx = double(a.x) - b.x,
                                   dy = double(a.y) - b.y,
                                   dz = double(a.z) - b.z;
                            dx -= nearbyint(dx / box.x) * box.x;
                            dy -= nearbyint(dy / box.y) * box.y;
                            dz -= nearbyint(dz / box.z) * box.z;
                            if (dx * dx + dy * dy + dz * dz <=
                                double(radius) * radius)
                            {
                                hit = true;
                                break;
                            }
                        }
                    }
        }
        selected[g] = hit;
    }
}
__global__ void Pack_Rows(int n, int width, const int* indices,
                          const float* rows, float* out)
{
    SIMPLE_DEVICE_FOR(i, n * width)
    out[i] = rows[indices[i / width] * width + i % width];
}
__global__ void Add_Rows(int n, int width, const int* indices, float* rows,
                         const float* input)
{
    SIMPLE_DEVICE_FOR(i, n * width)
    atomicAdd(rows + indices[i / width] * width + i % width, input[i]);
}
template <class T>
struct Device
{
    T* p = nullptr;
    explicit Device(size_t n)
    {
        Device_Malloc_Safely((void**)&p, std::max(size_t(1), n) * sizeof(T));
    }
    ~Device() { Free_Single_Device_Pointer((void**)&p); }
    void Set(const std::vector<T>& v)
    {
        if (!v.empty())
            deviceMemcpy(p, v.data(), v.size() * sizeof(T),
                         deviceMemcpyHostToDevice);
    }
};
#ifdef USE_MPI
template <class T>
std::vector<T> Exchange(const std::vector<T>& send, int peer, int tag)
{
    int n = static_cast<int>(send.size()), m = 0;
    MPI_Sendrecv(&n, 1, MPI_INT, peer, tag, &m, 1, MPI_INT, peer, tag,
                 CONTROLLER::pp_comm, MPI_STATUS_IGNORE);
    if (m < 0 || size_t(m) > size_t(INT_MAX) / sizeof(T) ||
        size_t(n) > size_t(INT_MAX) / sizeof(T))
        throw std::overflow_error("direct halo discovery count overflow");
    std::vector<T> receive(m);
    MPI_Sendrecv(send.data(), n * sizeof(T), MPI_BYTE, peer, tag + 1,
                 receive.data(), m * sizeof(T), MPI_BYTE, peer, tag + 1,
                 CONTROLLER::pp_comm, MPI_STATUS_IGNORE);
    return receive;
}
#endif
}  // namespace

void DIRECT_HALO_PLAN::Build(DOMAIN_INFORMATION& dd, CONTROLLER* controller,
                             MD_INFORMATION* md, float radius,
                             bool preserve_classic, float classic_radius)
{
#ifdef USE_MPI
    Invalidate_Mapping();
    rank_ = dd.pp_rank;
    size_ = controller->PP_MPI_size;
    owned_ = dd.atom_numbers;
    const VECTOR box = md->sys.box_length;
    if (!(radius > 0) || !std::isfinite(radius))
        throw std::invalid_argument("invalid direct halo radius");
    // Direct halo selection assumes orthogonal cell geometry.
    Device<Bounds> db(dd.res_numbers);
    if (dd.res_numbers)
        Launch_Device_Kernel(Group_Bounds, (dd.res_numbers + 255) / 256, 256, 0,
                             NULL, dd.res_numbers, dd.res_start, dd.res_len,
                             dd.crd, box, db.p);
    std::vector<Bounds> groups(dd.res_numbers);
    if (!groups.empty())
        deviceMemcpy(groups.data(), db.p, groups.size() * sizeof(Bounds),
                     deviceMemcpyDeviceToHost);
    Bounds local;
    local.empty = owned_ == 0;
    local.lo = box;
    local.hi = {0, 0, 0};
    local.head = {0, 0, 0};
    for (auto b : groups)
        if (!b.empty)
        {
            local.lo.x = std::min(local.lo.x, b.lo.x);
            local.hi.x = std::max(local.hi.x, b.hi.x);
            local.lo.y = std::min(local.lo.y, b.lo.y);
            local.hi.y = std::max(local.hi.y, b.hi.y);
            local.lo.z = std::min(local.lo.z, b.lo.z);
            local.hi.z = std::max(local.hi.z, b.hi.z);
        }
    std::vector<Bounds> ranks(size_), domains(size_);
    MPI_Allgather(&local, sizeof(Bounds), MPI_BYTE, ranks.data(),
                  sizeof(Bounds), MPI_BYTE, CONTROLLER::pp_comm);
    Bounds domain;
    domain.empty = 0;
    domain.lo = dd.min_corner;
    domain.hi = dd.max_corner;
    domain.head = {0, 0, 0};
    MPI_Allgather(&domain, sizeof(Bounds), MPI_BYTE, domains.data(),
                  sizeof(Bounds), MPI_BYTE, CONTROLLER::pp_comm);
    std::vector<int> starts(dd.res_numbers), lengths(dd.res_numbers),
        ids(owned_);
    if (dd.res_numbers)
    {
        deviceMemcpy(starts.data(), dd.res_start, starts.size() * sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(lengths.data(), dd.res_len, lengths.size() * sizeof(int),
                     deviceMemcpyDeviceToHost);
    }
    if (owned_)
        deviceMemcpy(ids.data(), dd.atom_local, ids.size() * sizeof(int),
                     deviceMemcpyDeviceToHost);
    NEIGHBOR_LIST::GRIDS bins;
    int nx = std::max(1, int(floor(double(box.x) / radius))),
        ny = std::max(1, int(floor(double(box.y) / radius))),
        nz = std::max(1, int(floor(double(box.z) / radius)));
    Device<int> overflow(1), unused(1), marked(owned_), count(1);
    Device<VECTOR> replies(owned_);
    int capacity = 2;
    for (;;)
    {
        bins.Initial_Buckets(nx, ny, nz, capacity, 1);
        bins.Populate_Buckets(owned_, 0, dd.crd, md->pbc.boundary, radius,
                              overflow.p, unused.p, true);
        int over = 0;
        deviceMemcpy(&over, overflow.p, sizeof(int), deviceMemcpyDeviceToHost);
        if (!over) break;
        bins.Clear();
        if (capacity >= std::max(1, owned_))
            throw std::overflow_error("direct halo bucket overflow");
        capacity = std::min(std::max(1, owned_), capacity * 2);
    }
    send_counts_.assign(size_, 0);
    recv_counts_.assign(size_, 0);
    send_offsets_.assign(size_, 0);
    recv_offsets_.assign(size_, 0);
    std::vector<int> all_send, all_receive, receive_lengths;
    Device<int> exact(dd.res_numbers);
    NEIGHBOR_LIST::GRIDS target_bins;
    int target_capacity = 2;
    bool target_bins_ready = false;
    // Lexicographic unordered pairs avoid rendezvous cycles and require no
    // second communicator. Nonparticipants advance to their next pair.
    for (int a = 0; a < size_; ++a)
        for (int b = a + 1; b < size_; ++b)
        {
            if (rank_ != a && rank_ != b) continue;
            int peer = rank_ == a ? b : a;
            std::vector<Bounds> queries;
            for (auto g : groups)
                if (Overlap(g, ranks[peer], box, radius)) queries.push_back(g);
            auto incoming = Exchange(queries, peer, 310);
            Device<Bounds> query(incoming.size());
            query.Set(incoming);
            deviceMemset(marked.p, 0, std::max(1, owned_) * sizeof(int));
            if (!incoming.empty())
                Launch_Device_Kernel(
                    Query_Buckets, (bins.grid_numbers + 255) / 256, 256, 0,
                    NULL, bins.grid_numbers, capacity, bins.d_grid_atom_numbers,
                    bins.d_grid_atoms, dd.crd, int(incoming.size()), query.p,
                    box, radius, marked.p);
            deviceMemset(count.p, 0, sizeof(int));
            if (owned_ && !incoming.empty())
                Launch_Device_Kernel(Pack_Replies, (owned_ + 255) / 256, 256, 0,
                                     NULL, owned_, marked.p, dd.crd, replies.p,
                                     count.p);
            int reply_count = 0;
            deviceMemcpy(&reply_count, count.p, sizeof(int),
                         deviceMemcpyDeviceToHost);
            std::vector<VECTOR> reply(reply_count);
            if (reply_count)
                deviceMemcpy(reply.data(), replies.p,
                             reply.size() * sizeof(VECTOR),
                             deviceMemcpyDeviceToHost);
            auto targets = Exchange(reply, peer, 312);
            Device<VECTOR> target(targets.size());
            target.Set(targets);
            if (!targets.empty())
                for (;;)
                {
                    if (!target_bins_ready)
                    {
                        target_bins.Initial_Buckets(nx, ny, nz, target_capacity,
                                                    1);
                        target_bins_ready = true;
                    }
                    target_bins.Populate_Buckets(targets.size(), 0, target.p,
                                                 md->pbc.boundary, radius,
                                                 overflow.p, unused.p, true);
                    int over = 0;
                    deviceMemcpy(&over, overflow.p, sizeof(int),
                                 deviceMemcpyDeviceToHost);
                    if (!over) break;
                    target_bins.Clear();
                    target_bins_ready = false;
                    if (target_capacity >= std::max(1, int(targets.size())))
                        throw std::overflow_error(
                            "direct target bucket overflow");
                    target_capacity = std::min(std::max(1, int(targets.size())),
                                               target_capacity * 2);
                }
            deviceMemset(exact.p, 0, std::max(1, dd.res_numbers) * sizeof(int));
            if (dd.res_numbers && !targets.empty())
                Launch_Device_Kernel(
                    Exact_Groups, (dd.res_numbers + 255) / 256, 256, 0, NULL,
                    dd.res_numbers, dd.res_start, dd.res_len, dd.crd, db.p,
                    ranks[peer], target.p, box, radius, nx, ny, nz,
                    target_capacity, target_bins.d_grid_atom_numbers,
                    target_bins.d_grid_atoms, exact.p);
            std::vector<int> selected(dd.res_numbers);
            if (dd.res_numbers)
                deviceMemcpy(selected.data(), exact.p,
                             selected.size() * sizeof(int),
                             deviceMemcpyDeviceToHost);
            std::vector<int> send_ids, send_lengths;
            send_offsets_[peer] = all_send.size();
            for (int g = 0; g < dd.res_numbers; ++g)
            {
                Bounds head = groups[g];
                head.lo = head.hi = head.head;
                bool classic = preserve_classic && Overlap(head, domains[peer],
                                                           box, classic_radius);
                if (!selected[g] && !classic) continue;
                send_lengths.push_back(lengths[g]);
                for (int j = 0; j < lengths[g]; ++j)
                {
                    int row = starts[g] + j;
                    all_send.push_back(row);
                    send_ids.push_back(ids[row]);
                }
            }
            auto received = Exchange(send_ids, peer, 314);
            auto received_groups = Exchange(send_lengths, peer, 316);
            send_counts_[peer] = send_ids.size();
            recv_counts_[peer] = received.size();
            recv_offsets_[peer] = owned_ + all_receive.size();
            all_receive.insert(all_receive.end(), received.begin(),
                               received.end());
            receive_lengths.insert(receive_lengths.end(),
                                   received_groups.begin(),
                                   received_groups.end());
        }
    bins.Clear();
    target_bins.Clear();
    if (owned_ + all_receive.size() > size_t(dd.max_atom_numbers) ||
        starts.size() + receive_lengths.size() > size_t(dd.max_res_numbers))
        throw std::overflow_error("direct halo exceeds DD resident capacity");
    std::vector<int> seen(dd.max_atom_numbers, 0);
    for (int id : ids)
    {
        if (id < 0 || id >= dd.max_atom_numbers || seen[id]++)
            throw std::runtime_error("duplicate/invalid owned ID");
    }
    for (int id : all_receive)
    {
        if (id < 0 || id >= dd.max_atom_numbers || seen[id]++)
            throw std::runtime_error("duplicate/invalid direct ghost ID");
    }
    peers_.clear();
    send_total_ = all_send.size();
    for (int peer = 0; peer < size_; ++peer)
        if (send_counts_[peer] || recv_counts_[peer]) peers_.push_back(peer);
    Reserve_Send_Rows(all_send.size());
    if (!all_send.empty())
        deviceMemcpy(send_rows_, all_send.data(), all_send.size() * sizeof(int),
                     deviceMemcpyHostToDevice);
    dd.ghost_numbers = all_receive.size();
    dd.ghost_res_numbers = receive_lengths.size();
    if (!all_receive.empty())
        deviceMemcpy(dd.atom_local + owned_, all_receive.data(),
                     all_receive.size() * sizeof(int),
                     deviceMemcpyHostToDevice);
    int offset = owned_;
    for (int length : receive_lengths)
    {
        starts.push_back(offset);
        lengths.push_back(length);
        offset += length;
    }
    if (offset != owned_ + dd.ghost_numbers)
        throw std::runtime_error(
            "direct group lengths disagree with received rows");
    if (!starts.empty())
    {
        deviceMemcpy(dd.res_start, starts.data(), starts.size() * sizeof(int),
                     deviceMemcpyHostToDevice);
        deviceMemcpy(dd.res_len, lengths.data(), lengths.size() * sizeof(int),
                     deviceMemcpyHostToDevice);
    }
    deviceMemcpy(dd.d_ghost_numbers, &dd.ghost_numbers, sizeof(int),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(dd.d_ghost_res_numbers, &dd.ghost_res_numbers, sizeof(int),
                 deviceMemcpyHostToDevice);
    if (!peers_.empty())
    {
        if (size_t(send_total_) > size_t(INT_MAX) / (3 * sizeof(float)))
            throw std::overflow_error(
                "direct halo coordinate count exceeds native int range");
        Reserve(size_t(send_total_) * 3);
        if (!streams_ready_)
        {
#ifdef USE_GPU
            if (deviceStreamCreate(&requests_[0]) != DEVICE_MALLOC_SUCCESS)
                throw std::runtime_error(
                    "direct halo send stream creation failed");
            if (deviceStreamCreate(&requests_[1]) != DEVICE_MALLOC_SUCCESS)
            {
                deviceStreamDestroy(requests_[0]);
                throw std::runtime_error(
                    "direct halo receive stream creation failed");
            }
#endif
            streams_ready_ = true;
        }
    }
    active = true;
    try
    {
        Forward(reinterpret_cast<float*>(dd.crd), 3);
    }
    catch (...)
    {
        active = false;
        throw;
    }
#endif
}

void DIRECT_HALO_PLAN::Invalidate_Mapping()
{
    active = false;
    send_counts_.clear();
    recv_counts_.clear();
    send_offsets_.clear();
    recv_offsets_.clear();
    peers_.clear();
    send_total_ = 0;
}
void DIRECT_HALO_PLAN::Reserve_Send_Rows(size_t rows)
{
    if (rows <= send_rows_capacity_) return;
    int* next = nullptr;
    if (!Device_Malloc_Safely((void**)&next, rows * sizeof(int)) || !next)
        throw std::runtime_error("direct halo send-row allocation failed");
    Free_Single_Device_Pointer((void**)&send_rows_);
    send_rows_ = next;
    send_rows_capacity_ = rows;
}
void DIRECT_HALO_PLAN::Reserve(size_t values)
{
    values = std::max(size_t(1), values);
    if (values <= scratch_capacity_) return;
    float* next = nullptr;
    if (!Device_Malloc_Safely((void**)&next, values * sizeof(float)) || !next)
        throw std::runtime_error("direct halo scratch allocation failed");
    Free_Single_Device_Pointer((void**)&scratch_);
    scratch_ = next;
    scratch_capacity_ = values;
}
void DIRECT_HALO_PLAN::Forward(float* rows, int width)
{
#ifdef USE_MPI
    if (!active || width <= 0)
        throw std::runtime_error("invalid direct halo forward");
    if (peers_.empty()) return;
    if (size_t(send_total_) * width > size_t(INT_MAX) / sizeof(float))
        throw std::overflow_error(
            "direct halo tensor count exceeds native int range");
    Reserve(size_t(send_total_) * width);
    if (send_total_)
        Launch_Device_Kernel(Pack_Rows, (send_total_ * width + 255) / 256, 256,
                             0, NULL, send_total_, width, send_rows_, rows,
                             scratch_);
    deviceStreamSynchronize(0);
#ifndef USE_XCCL
    std::vector<D_MPI_Request> requests(peers_.size() * 2);
    std::vector<D_MPI_Status> statuses(peers_.size() * 2);
    int request = 0;
#endif
    D_MPI_GroupStart();
    for (int peer : peers_)
    {
        const int n = send_counts_[peer], m = recv_counts_[peer];
#ifdef USE_XCCL
        auto send_request = requests_[0], recv_request = requests_[1];
#else
        auto& send_request = requests[request++];
        auto& recv_request = requests[request++];
#endif
        D_MPI_Isend(n ? scratch_ + send_offsets_[peer] * width : scratch_,
                    n * width * sizeof(float), D_MPI_BYTE, peer, 0,
                    CONTROLLER::d_pp_comm, send_request);
        D_MPI_Irecv(m ? rows + recv_offsets_[peer] * width : scratch_,
                    m * width * sizeof(float), D_MPI_BYTE, peer, 0,
                    CONTROLLER::d_pp_comm, recv_request);
    }
    D_MPI_GroupEnd();
#ifdef USE_XCCL
    D_MPI_Status statuses[2];
    D_MPI_Waitall(2, requests_, statuses);
#else
    D_MPI_Waitall(request, requests.data(), statuses.data());
#endif
#endif
}
void DIRECT_HALO_PLAN::Reverse(float* rows, int width)
{
#ifdef USE_MPI
    if (!active || width <= 0)
        throw std::runtime_error("invalid direct halo reverse");
    if (peers_.empty()) return;
    if (size_t(send_total_) * width > size_t(INT_MAX) / sizeof(float))
        throw std::overflow_error(
            "direct halo tensor count exceeds native int range");
    Reserve(size_t(send_total_) * width);
    deviceStreamSynchronize(0);
#ifndef USE_XCCL
    std::vector<D_MPI_Request> requests(peers_.size() * 2);
    std::vector<D_MPI_Status> statuses(peers_.size() * 2);
    int request = 0;
#endif
    D_MPI_GroupStart();
    for (int peer : peers_)
    {
        const int n = send_counts_[peer], m = recv_counts_[peer];
#ifdef USE_XCCL
        auto send_request = requests_[0], recv_request = requests_[1];
#else
        auto& send_request = requests[request++];
        auto& recv_request = requests[request++];
#endif
        D_MPI_Isend(m ? rows + recv_offsets_[peer] * width : scratch_,
                    m * width * sizeof(float), D_MPI_BYTE, peer, 0,
                    CONTROLLER::d_pp_comm, send_request);
        D_MPI_Irecv(n ? scratch_ + send_offsets_[peer] * width : scratch_,
                    n * width * sizeof(float), D_MPI_BYTE, peer, 0,
                    CONTROLLER::d_pp_comm, recv_request);
    }
    D_MPI_GroupEnd();
#ifdef USE_XCCL
    D_MPI_Status statuses[2];
    D_MPI_Waitall(2, requests_, statuses);
#else
    D_MPI_Waitall(request, requests.data(), statuses.data());
#endif
    if (send_total_)
        Launch_Device_Kernel(Add_Rows, (send_total_ * width + 255) / 256, 256,
                             0, NULL, send_total_, width, send_rows_, rows,
                             scratch_);
    deviceStreamSynchronize(0);
#endif
}
void DIRECT_HALO_PLAN::Clear()
{
    Invalidate_Mapping();
#ifdef USE_MPI
    if (streams_ready_)
    {
#ifdef USE_GPU
        for (int i = 0; i < 2; ++i)
        {
            deviceStreamSynchronize(requests_[i]);
            deviceStreamDestroy(requests_[i]);
        }
#endif
        streams_ready_ = false;
    }
#endif
    Free_Single_Device_Pointer((void**)&send_rows_);
    send_rows_capacity_ = 0;
    Free_Single_Device_Pointer((void**)&scratch_);
    scratch_capacity_ = 0;
}
