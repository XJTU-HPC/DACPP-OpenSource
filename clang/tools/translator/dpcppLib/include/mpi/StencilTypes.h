#ifndef DACPP_MPI_STENCIL_TYPES_H
#define DACPP_MPI_STENCIL_TYPES_H

#include <cstdint>
#include <string>
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
    std::vector<std::vector<SlotSpan>> local_write_spans_by_route;
    std::vector<std::vector<SlotSpan>> local_target_spans_by_route;
    std::vector<bool> use_span_pairs_by_route;
    std::vector<std::vector<SlotSpan>> read_cache_transition_source_spans;
    std::vector<std::vector<SlotSpan>> read_cache_transition_target_spans;
    std::vector<bool> read_cache_transition_use_span_pairs;
    std::vector<int64_t> local_write_globals;
    AllRankIndexLayout read_layout;
    AllRankIndexLayout write_layout;
    AllRankIndexLayout root_bridge_layout;
    ExchangePlan exchange_plan;
    std::vector<ExchangePlan> exchange_plans_by_route;
    ExchangePlan root_bridge_plan;
    HaloExchangePlan halo_plan;
    std::vector<HaloExchangePlan> halo_plans_by_route;
    PackMap root_bridge_pack;
};

}  // namespace mpi
}  // namespace dacpp

#endif
