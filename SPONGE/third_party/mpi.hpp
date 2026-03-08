#pragma once

#define SPONGE_MPI_ROOT 0

#if defined(USE_CUDA) && defined(USE_MPI)
#define USE_XCCL
#endif

#ifdef USE_MPI
#include <mpi.h>
#ifdef USE_XCCL
#include <nccl.h>
#define xcclUniqueId ncclUniqueId
#define xcclGetUniqueId ncclGetUniqueId
#define xcclCommInitRank ncclCommInitRank
#define xcclCommDestroy ncclCommDestroy
#define xcclAllReduce ncclAllReduce
#define xcclBroadcast ncclBroadcast
#define xcclGroupStart ncclGroupStart
#define xcclGroupEnd ncclGroupEnd
#define xcclChar ncclChar
#define xcclFloat ncclFloat
#define xcclSum ncclSum
#define D_MPI_Comm ncclComm_t
#define xcclSend ncclSend
#define xcclRecv ncclRecv

#endif
#else  // FAKE MPI FOR SINGLE PROCESSOR
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_COMM_WORLD 0
#define MPI_DOUBLE 0
#define MPI_SUM 0
#define MPI_SUCCESS 0

static inline int MPI_Init(int* argc, char*** argv) { return MPI_SUCCESS; }
static inline int MPI_Finalize(void) { return MPI_SUCCESS; }
static inline int MPI_Comm_split(MPI_Comm comm, int color, int key,
                                 MPI_Comm* newcomm)
{
    *newcomm = 0;
    return MPI_SUCCESS;
}
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank)
{
    *rank = 0;
    return MPI_SUCCESS;
}
static inline int MPI_Comm_size(MPI_Comm comm, int* size)
{
    *size = 1;
    return MPI_SUCCESS;
}
static inline int MPI_Barrier(MPI_Comm comm) { return MPI_SUCCESS; }
#endif  // USE_MPI

#ifndef D_MPI_Comm
#define D_MPI_Comm MPI_Comm
#endif

// 进程通信封装
struct SPONGE_MPI_WRAPPER
{
    static __forceinline__ void Device_Sum(void* ptr, int count,
                                           D_MPI_Comm comm)
    {
#ifndef USE_MPI

#elif defined(USE_XCCL)
        xcclAllReduce(ptr, ptr, count, xcclFloat, xcclSum, comm, NULL);
#else
        MPI_Allreduce(MPI_IN_PLACE, ptr, count, MPI_FLOAT, MPI_SUM, comm);
#endif
    }
    static __forceinline__ void Device_Gatherv(void* ptr, int* counts,
                                               int* displs, int MPI_size,
                                               D_MPI_Comm comm)
    {
#ifndef USE_MPI

#elif defined(USE_XCCL)
        float* temp;
        xcclGroupStart();
        for (int i = 0; i < MPI_size; i++)
        {
            temp = ((float*)ptr) + displs[i];
            xcclBroadcast(temp, temp, counts[i], xcclFloat, i, comm, NULL);
        }
        xcclGroupEnd();
#else
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, ptr, counts, displs,
                       MPI_FLOAT, MPI_COMM_WORLD);
#endif
    }
};

// 兼容DEVICE点对点通信封装
// xccl 中，没有tag，而是需要在同一个stream中依次发送和接收
#ifdef USE_MPI
//-------------------USE_NCCL-------------------
#ifdef USE_XCCL
#define D_MPI_Request deviceStream_t
#define D_MPI_Status int
#define D_MPI_Isend(buf, count, datatype, dest, tag, comm, request) \
    xcclSend(buf, count, datatype, dest, comm, request)
#define D_MPI_Irecv(buf, count, datatype, source, tag, comm, request) \
    xcclRecv(buf, count, datatype, source, comm, request)
#define D_MPI_Waitall(count, requests, statuses) \
    for (int i = 0; i < count; i++) deviceStreamSynchronize(requests[i])

#define D_MPI_Send D_MPI_Isend
#define D_MPI_Recv D_MPI_Irecv

#define D_MPI_Barrier(comm, stream) deviceStreamSynchronize(stream)

#define D_MPI_Allreduce_IN_PLACE(buf, count, datatype, op, comm, stream) \
    xcclAllReduce(buf, buf, count, datatype, op, comm, stream)
#define D_MPI_Allreduce(buf, recvbuf, count, datatype, op, comm, stream) \
    xcclAllReduce(buf, recvbuf, count, datatype, op, comm, stream)

#define D_MPI_BYTE xcclChar
#define D_MPI_FLOAT xcclFloat
#define D_MPI_SUM xcclSum

// Explicit group start/end macros for GPU path
#define D_MPI_GroupStart() xcclGroupStart()
#define D_MPI_GroupEnd() xcclGroupEnd()

// -------------------USE_MPI---------------------
#else
#define D_MPI_Request MPI_Request
#define D_MPI_Status MPI_Status
#define D_MPI_Isend(buf, count, datatype, dest, tag, comm, request) \
    MPI_Isend(buf, count, datatype, dest, tag, comm, &request)
#define D_MPI_Irecv(buf, count, datatype, source, tag, comm, request) \
    MPI_Irecv(buf, count, datatype, source, tag, comm, &request)

#define D_MPI_Send(buf, count, datatype, dest, tag, comm, request) \
    MPI_Send(buf, count, datatype, dest, tag, comm)
#define D_MPI_Recv(buf, count, datatype, source, tag, comm, request) \
    MPI_Recv(buf, count, datatype, source, tag, comm, MPI_STATUS_IGNORE)

#define D_MPI_Waitall(count, requests, statuses) \
    MPI_Waitall(count, requests, statuses)
#define D_MPI_Barrier(comm, stream) MPI_Barrier(comm)

#define D_MPI_Allreduce_IN_PLACE(buf, count, datatype, op, comm, stream) \
    MPI_Allreduce(MPI_IN_PLACE, buf, count, datatype, op, comm)
#define D_MPI_Allreduce(buf, recvbuf, count, datatype, op, comm, stream) \
    MPI_Allreduce(buf, recvbuf, count, datatype, op, comm)

#define D_MPI_BYTE MPI_BYTE
#define D_MPI_FLOAT MPI_FLOAT
#define D_MPI_SUM MPI_SUM

// For non-GPU path these are no-ops (keeps call sites uniform)
#define D_MPI_GroupStart()
#define D_MPI_GroupEnd()
#endif

#endif
