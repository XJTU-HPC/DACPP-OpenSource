#ifndef DACPP_MPI_STENCIL_TYPES_H
#define DACPP_MPI_STENCIL_TYPES_H

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "CoreTypes.h"

namespace dacpp {
namespace mpi {

struct GatheredIndexLayout {
    int local_count = 0;
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int> byte_counts;
    std::vector<int> byte_displs;
    std::vector<int64_t> globals;
};

struct AllRankIndexLayout {
    int local_count = 0;
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int64_t> globals;
};

struct PeerSlotExchange {
    int peer_rank = -1;
    std::vector<int32_t> local_slots;
    std::vector<int64_t> globals;
};

struct SlotSpan {
    int32_t begin = 0;
    int32_t count = 0;
    int32_t stride = 1;
};

struct ContiguousRowCopyBlock {
    int32_t source_begin = 0;
    int32_t target_begin = 0;
    int32_t row_width = 0;
    int32_t row_count = 0;
    int32_t source_row_stride = 0;
    int32_t target_row_stride = 0;
};

struct PeerHaloExchange {
    int peer_rank = -1;
    std::vector<SlotSpan> local_spans;
};

struct ExchangePlan {
    bool supported = false;
    std::string unsupported_reason;
    std::vector<PeerSlotExchange> send_transfers;
    std::vector<PeerSlotExchange> recv_transfers;
};

struct HaloExchangePlan {
    bool supported = false;
    std::string unsupported_reason;
    std::vector<PeerHaloExchange> send_transfers;
    std::vector<PeerHaloExchange> recv_transfers;
};

struct WaveRouteFastPathState {
    std::vector<SlotSpan> local_write_spans;
    std::vector<SlotSpan> local_target_spans;
    std::vector<ContiguousRowCopyBlock> local_row_copy_blocks;
    bool use_span_pairs = false;
    bool use_row_copy_blocks = false;
};

struct WaveReadCacheTransitionFastPathState {
    std::vector<SlotSpan> source_spans;
    std::vector<SlotSpan> target_spans;
    std::vector<ContiguousRowCopyBlock> row_copy_blocks;
    bool use_span_pairs = false;
    bool use_row_copy_blocks = false;
};

struct WaveDirectKernelState {
    std::vector<int32_t> slots;
    std::unique_ptr<sycl::buffer<int32_t, 1>> slots_buffer;
    std::vector<int32_t> next_stale_slots;
    bool can_sparse_clear = false;
};

struct WaveSpecializationState {
    bool use_span_pairs = false;
    bool use_direct_kernel = false;
    std::vector<std::vector<WaveRouteFastPathState>> route_fast_paths_by_param;
    std::vector<WaveReadCacheTransitionFastPathState>
        read_cache_transition_fast_paths;
    WaveDirectKernelState direct_kernel;
};

template <typename T>
struct HaloExchangeRuntime {
    std::vector<std::vector<T>> send_buffers;
    std::vector<std::vector<T>> recv_buffers;
    std::vector<MPI_Request> requests;
};

template <typename T>
struct DistributedTensorState {
    bool enabled = false;
    bool seeded = false;
    bool authoritative_source = false;

    std::vector<T> local_cache;
    std::vector<T> local_write_values;
    std::vector<int32_t> local_write_slots;
    std::vector<int32_t> local_target_slots;
    std::vector<std::vector<int32_t>> local_target_slots_by_route;
    std::vector<int64_t> local_write_globals;
    AllRankIndexLayout read_layout;
    AllRankIndexLayout write_layout;
    AllRankIndexLayout root_bridge_layout;
    ExchangePlan exchange_plan;
    std::vector<ExchangePlan> exchange_plans_by_route;
    ExchangePlan root_bridge_plan;
    HaloExchangePlan halo_plan;
    HaloExchangeRuntime<T> halo_runtime;
    std::vector<HaloExchangePlan> halo_plans_by_route;
    std::vector<HaloExchangeRuntime<T>> halo_runtimes_by_route;
    PackMap root_bridge_pack;
};

}  // namespace mpi
}  // namespace dacpp

#endif
