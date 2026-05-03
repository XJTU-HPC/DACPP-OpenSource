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

struct ExchangePlan {
    bool supported = false;
    std::string unsupported_reason;
    std::vector<PeerSlotExchange> send_transfers;
    std::vector<PeerSlotExchange> recv_transfers;
};

template <typename T>
struct DistributedTensorState {
    bool enabled = false;
    bool seeded = false;
    bool authoritative_source = false;

    std::vector<T> local_cache;
    AllRankIndexLayout read_layout;
    AllRankIndexLayout write_layout;
    AllRankIndexLayout root_bridge_layout;
    ExchangePlan exchange_plan;
    ExchangePlan root_bridge_plan;
    PackMap root_bridge_pack;
};

}  // namespace mpi
}  // namespace dacpp

#endif
