#ifndef DACPP_MPI_STENCIL_EXCHANGE_PLAN_H
#define DACPP_MPI_STENCIL_EXCHANGE_PLAN_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "PackMap.h"
#include "StencilLayout.h"

namespace dacpp {
namespace mpi {

inline PackMap make_dense_cover_pack(std::size_t dense_size) {
    std::vector<int64_t> globals(dense_size);
    for (std::size_t idx = 0; idx < dense_size; ++idx) {
        globals[idx] = static_cast<int64_t>(idx);
    }
    PackMap pack = make_pack_map_from_globals(
        std::move(globals), PackLayoutKind::DenseCover, dense_size);
    return pack;
}

inline void build_local_slots_for_globals(const PackMap& pack,
                                          std::vector<int32_t>& local_slots) {
    const std::vector<int64_t>& globals =
        pack.writeback_globals.empty() ? pack.globals : pack.writeback_globals;
    local_slots.clear();
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        local_slots.push_back(
            lookup_local_slot_or_throw(
                pack, global_idx, "build_local_slots_for_globals"));
    }
}

inline void build_target_slots_for_globals(const PackMap& pack,
                                           const std::vector<int64_t>& globals,
                                           std::vector<int32_t>& local_slots) {
    local_slots.clear();
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        local_slots.push_back(try_lookup_local_slot(pack, global_idx));
    }
}

inline void build_target_slots_for_globals_with_offset(
    const PackMap& pack,
    const std::vector<int64_t>& globals,
    int64_t target_offset,
    std::vector<int32_t>& local_slots) {
    local_slots.clear();
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        local_slots.push_back(try_lookup_local_slot(pack, global_idx + target_offset));
    }
}

inline int64_t map_2d_global_with_offset(int64_t source_global,
                                         int64_t source_cols,
                                         int64_t target_cols,
                                         int64_t row_offset,
                                         int64_t col_offset) {
    if (source_global < 0 || source_cols <= 0 || target_cols <= 0) {
        return -1;
    }
    const int64_t source_row = source_global / source_cols;
    const int64_t source_col = source_global % source_cols;
    const int64_t target_row = source_row + row_offset;
    const int64_t target_col = source_col + col_offset;
    if (target_row < 0 || target_col < 0 || target_col >= target_cols) {
        return -1;
    }
    return target_row * target_cols + target_col;
}

inline int64_t inverse_map_2d_global_with_offset(int64_t target_global,
                                                 int64_t source_cols,
                                                 int64_t target_cols,
                                                 int64_t row_offset,
                                                 int64_t col_offset) {
    if (target_global < 0 || source_cols <= 0 || target_cols <= 0) {
        return -1;
    }
    const int64_t target_row = target_global / target_cols;
    const int64_t target_col = target_global % target_cols;
    const int64_t source_row = target_row - row_offset;
    const int64_t source_col = target_col - col_offset;
    if (source_row < 0 || source_col < 0 || source_col >= source_cols) {
        return -1;
    }
    return source_row * source_cols + source_col;
}

inline void build_target_slots_for_globals_2d_offset(
    const PackMap& pack,
    const std::vector<int64_t>& globals,
    int64_t source_cols,
    int64_t target_cols,
    int64_t row_offset,
    int64_t col_offset,
    std::vector<int32_t>& local_slots) {
    local_slots.clear();
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        const int64_t target_global =
            map_2d_global_with_offset(global_idx, source_cols, target_cols,
                                      row_offset, col_offset);
        local_slots.push_back(
            target_global >= 0 ? try_lookup_local_slot(pack, target_global) : -1);
    }
}

inline ExchangePlan build_exchange_plan_from_layouts(
    const PackMap& writer_pack,
    const AllRankIndexLayout& writer_layout,
    const PackMap& reader_pack,
    const AllRankIndexLayout& reader_layout,
    int mpi_rank,
    int mpi_size) {
    ExchangePlan plan;
    std::string unique_writer_reason;
    if (!validate_unique_writers(writer_layout, mpi_size, &unique_writer_reason)) {
        plan.supported = false;
        plan.unsupported_reason =
            "writer layout has overlapping ownership: " + unique_writer_reason;
        return plan;
    }

    std::unordered_map<int64_t, int> writer_owner;
    writer_owner.reserve(writer_layout.globals.size());
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = writer_layout.counts[static_cast<std::size_t>(rank)];
        const int displ = writer_layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            writer_owner.emplace(
                writer_layout.globals[static_cast<std::size_t>(displ + idx)], rank);
        }
    }

    std::unordered_map<int, PeerSlotExchange> recv_by_peer;
    const int local_read_count =
        reader_layout.counts.empty()
            ? 0
            : reader_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_read_displ =
        reader_layout.displs.empty()
            ? 0
            : reader_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_read_count; ++idx) {
        const int64_t global_idx =
            reader_layout.globals[static_cast<std::size_t>(local_read_displ + idx)];
        const auto ownerIt = writer_owner.find(global_idx);
        if (ownerIt == writer_owner.end()) {
            continue;
        }
        const int owner_rank = ownerIt->second;
        if (owner_rank == mpi_rank) {
            continue;
        }
        const int32_t slot =
            try_lookup_local_slot(reader_pack, global_idx);
        if (slot < 0) {
            plan.supported = false;
            plan.unsupported_reason =
                "reader pack missing global " + std::to_string(global_idx);
            return plan;
        }
        auto& transfer = recv_by_peer[owner_rank];
        transfer.peer_rank = owner_rank;
        transfer.globals.push_back(global_idx);
        transfer.local_slots.push_back(slot);
    }

    std::unordered_map<int, PeerSlotExchange> send_by_peer;
    const int local_write_count =
        writer_layout.counts.empty()
            ? 0
            : writer_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_write_displ =
        writer_layout.displs.empty()
            ? 0
            : writer_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_write_count; ++idx) {
        const int64_t global_idx =
            writer_layout.globals[static_cast<std::size_t>(local_write_displ + idx)];
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank == mpi_rank) {
                continue;
            }
            const int peer_count =
                reader_layout.counts[static_cast<std::size_t>(rank)];
            const int peer_displ =
                reader_layout.displs[static_cast<std::size_t>(rank)];
            const auto begin =
                reader_layout.globals.begin() + peer_displ;
            const auto end = begin + peer_count;
            if (!std::binary_search(begin, end, global_idx)) {
                continue;
            }
            const int32_t slot = try_lookup_local_slot(writer_pack, global_idx);
            if (slot < 0) {
                plan.supported = false;
                plan.unsupported_reason =
                    "writer pack missing global " + std::to_string(global_idx);
                return plan;
            }
            auto& transfer = send_by_peer[rank];
            transfer.peer_rank = rank;
            transfer.globals.push_back(global_idx);
            transfer.local_slots.push_back(slot);
        }
    }

    plan.supported = true;
    for (auto& entry : send_by_peer) {
        plan.send_transfers.push_back(std::move(entry.second));
    }
    for (auto& entry : recv_by_peer) {
        plan.recv_transfers.push_back(std::move(entry.second));
    }
    std::sort(plan.send_transfers.begin(), plan.send_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    std::sort(plan.recv_transfers.begin(), plan.recv_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    return plan;
}

inline ExchangePlan build_exchange_plan_from_layouts_with_target_offset(
    const PackMap& writer_pack,
    const AllRankIndexLayout& writer_layout,
    const PackMap& reader_pack,
    const AllRankIndexLayout& reader_layout,
    int64_t target_offset,
    int mpi_rank,
    int mpi_size) {
    ExchangePlan plan;
    std::string unique_writer_reason;
    if (!validate_unique_writers(writer_layout, mpi_size, &unique_writer_reason)) {
        plan.supported = false;
        plan.unsupported_reason =
            "writer layout has overlapping ownership: " + unique_writer_reason;
        return plan;
    }

    std::unordered_map<int64_t, int> writer_owner;
    writer_owner.reserve(writer_layout.globals.size());
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = writer_layout.counts[static_cast<std::size_t>(rank)];
        const int displ = writer_layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            writer_owner.emplace(
                writer_layout.globals[static_cast<std::size_t>(displ + idx)], rank);
        }
    }

    std::unordered_map<int, PeerSlotExchange> recv_by_peer;
    const int local_read_count =
        reader_layout.counts.empty()
            ? 0
            : reader_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_read_displ =
        reader_layout.displs.empty()
            ? 0
            : reader_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_read_count; ++idx) {
        const int64_t target_global =
            reader_layout.globals[static_cast<std::size_t>(local_read_displ + idx)];
        const int64_t writer_global = target_global - target_offset;
        const auto ownerIt = writer_owner.find(writer_global);
        if (ownerIt == writer_owner.end()) {
            continue;
        }
        const int owner_rank = ownerIt->second;
        if (owner_rank == mpi_rank) {
            continue;
        }
        const int32_t slot = try_lookup_local_slot(reader_pack, target_global);
        if (slot < 0) {
            plan.supported = false;
            plan.unsupported_reason =
                "reader pack missing global " + std::to_string(target_global);
            return plan;
        }
        auto& transfer = recv_by_peer[owner_rank];
        transfer.peer_rank = owner_rank;
        transfer.globals.push_back(target_global);
        transfer.local_slots.push_back(slot);
    }

    std::unordered_map<int, PeerSlotExchange> send_by_peer;
    const int local_write_count =
        writer_layout.counts.empty()
            ? 0
            : writer_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_write_displ =
        writer_layout.displs.empty()
            ? 0
            : writer_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_write_count; ++idx) {
        const int64_t writer_global =
            writer_layout.globals[static_cast<std::size_t>(local_write_displ + idx)];
        const int64_t target_global = writer_global + target_offset;
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank == mpi_rank) {
                continue;
            }
            const int peer_count =
                reader_layout.counts[static_cast<std::size_t>(rank)];
            const int peer_displ =
                reader_layout.displs[static_cast<std::size_t>(rank)];
            const auto begin =
                reader_layout.globals.begin() + peer_displ;
            const auto end = begin + peer_count;
            if (!std::binary_search(begin, end, target_global)) {
                continue;
            }
            const int32_t slot = try_lookup_local_slot(writer_pack, writer_global);
            if (slot < 0) {
                plan.supported = false;
                plan.unsupported_reason =
                    "writer pack missing global " + std::to_string(writer_global);
                return plan;
            }
            auto& transfer = send_by_peer[rank];
            transfer.peer_rank = rank;
            transfer.globals.push_back(writer_global);
            transfer.local_slots.push_back(slot);
        }
    }

    plan.supported = true;
    for (auto& entry : send_by_peer) {
        plan.send_transfers.push_back(std::move(entry.second));
    }
    for (auto& entry : recv_by_peer) {
        plan.recv_transfers.push_back(std::move(entry.second));
    }
    std::sort(plan.send_transfers.begin(), plan.send_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    std::sort(plan.recv_transfers.begin(), plan.recv_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    return plan;
}

inline ExchangePlan build_exchange_plan_from_layouts_2d_offset(
    const PackMap& writer_pack,
    const AllRankIndexLayout& writer_layout,
    const PackMap& reader_pack,
    const AllRankIndexLayout& reader_layout,
    int64_t writer_cols,
    int64_t reader_cols,
    int64_t row_offset,
    int64_t col_offset,
    int mpi_rank,
    int mpi_size) {
    ExchangePlan plan;
    std::string unique_writer_reason;
    if (!validate_unique_writers(writer_layout, mpi_size, &unique_writer_reason)) {
        plan.supported = false;
        plan.unsupported_reason =
            "writer layout has overlapping ownership: " + unique_writer_reason;
        return plan;
    }

    std::unordered_map<int64_t, int> writer_owner;
    writer_owner.reserve(writer_layout.globals.size());
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = writer_layout.counts[static_cast<std::size_t>(rank)];
        const int displ = writer_layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            writer_owner.emplace(
                writer_layout.globals[static_cast<std::size_t>(displ + idx)], rank);
        }
    }

    std::unordered_map<int, PeerSlotExchange> recv_by_peer;
    const int local_read_count =
        reader_layout.counts.empty()
            ? 0
            : reader_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_read_displ =
        reader_layout.displs.empty()
            ? 0
            : reader_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_read_count; ++idx) {
        const int64_t target_global =
            reader_layout.globals[static_cast<std::size_t>(local_read_displ + idx)];
        const int64_t writer_global =
            inverse_map_2d_global_with_offset(target_global, writer_cols,
                                              reader_cols, row_offset,
                                              col_offset);
        if (writer_global < 0) {
            continue;
        }
        const auto ownerIt = writer_owner.find(writer_global);
        if (ownerIt == writer_owner.end()) {
            continue;
        }
        const int owner_rank = ownerIt->second;
        if (owner_rank == mpi_rank) {
            continue;
        }
        const int32_t slot = try_lookup_local_slot(reader_pack, target_global);
        if (slot < 0) {
            plan.supported = false;
            plan.unsupported_reason =
                "reader pack missing global " + std::to_string(target_global);
            return plan;
        }
        auto& transfer = recv_by_peer[owner_rank];
        transfer.peer_rank = owner_rank;
        transfer.globals.push_back(target_global);
        transfer.local_slots.push_back(slot);
    }

    std::unordered_map<int, PeerSlotExchange> send_by_peer;
    const int local_write_count =
        writer_layout.counts.empty()
            ? 0
            : writer_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_write_displ =
        writer_layout.displs.empty()
            ? 0
            : writer_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_write_count; ++idx) {
        const int64_t writer_global =
            writer_layout.globals[static_cast<std::size_t>(local_write_displ + idx)];
        const int64_t target_global =
            map_2d_global_with_offset(writer_global, writer_cols, reader_cols,
                                      row_offset, col_offset);
        if (target_global < 0) {
            continue;
        }
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank == mpi_rank) {
                continue;
            }
            const int peer_count =
                reader_layout.counts[static_cast<std::size_t>(rank)];
            const int peer_displ =
                reader_layout.displs[static_cast<std::size_t>(rank)];
            const auto begin =
                reader_layout.globals.begin() + peer_displ;
            const auto end = begin + peer_count;
            if (!std::binary_search(begin, end, target_global)) {
                continue;
            }
            const int32_t slot = try_lookup_local_slot(writer_pack, writer_global);
            if (slot < 0) {
                plan.supported = false;
                plan.unsupported_reason =
                    "writer pack missing global " + std::to_string(writer_global);
                return plan;
            }
            auto& transfer = send_by_peer[rank];
            transfer.peer_rank = rank;
            transfer.globals.push_back(writer_global);
            transfer.local_slots.push_back(slot);
        }
    }

    plan.supported = true;
    for (auto& entry : send_by_peer) {
        plan.send_transfers.push_back(std::move(entry.second));
    }
    for (auto& entry : recv_by_peer) {
        plan.recv_transfers.push_back(std::move(entry.second));
    }
    std::sort(plan.send_transfers.begin(), plan.send_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    std::sort(plan.recv_transfers.begin(), plan.recv_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    return plan;
}

inline std::vector<SlotSpan> compact_slots_to_spans(
    const std::vector<int32_t>& local_slots,
    std::string* unsupported_reason = nullptr) {
    std::vector<SlotSpan> spans;
    if (local_slots.empty()) {
        return spans;
    }

    int32_t span_begin = local_slots.front();
    if (span_begin < 0) {
        if (unsupported_reason) {
            *unsupported_reason = "halo transfer contains negative slot";
        }
        return {};
    }
    int32_t previous = span_begin;
    int32_t stride = 1;
    int32_t count = 1;
    for (std::size_t idx = 1; idx < local_slots.size(); ++idx) {
        const int32_t slot = local_slots[idx];
        if (slot < 0) {
            if (unsupported_reason) {
                *unsupported_reason = "halo transfer contains negative slot";
            }
            return {};
        }
        if (slot < previous) {
            if (unsupported_reason) {
                *unsupported_reason = "halo transfer slots are not monotonic";
            }
            return {};
        }
        if (count == 1) {
            stride = slot - previous;
            if (stride <= 0) {
                if (unsupported_reason) {
                    *unsupported_reason = "halo transfer slots are not monotonic";
                }
                return {};
            }
            ++count;
        } else if (slot == previous + stride) {
            ++count;
        } else if (slot == previous) {
            if (unsupported_reason) {
                *unsupported_reason = "halo transfer slots contain duplicates";
            }
            return {};
        } else {
            spans.push_back(SlotSpan{span_begin, count, stride});
            span_begin = slot;
            stride = 1;
            count = 1;
        }
        previous = slot;
    }
    spans.push_back(SlotSpan{span_begin, count, stride});
    return spans;
}

inline HaloExchangePlan build_halo_plan_from_exchange_plan(
    const ExchangePlan& exchange_plan) {
    HaloExchangePlan halo_plan;
    if (!exchange_plan.supported) {
        halo_plan.supported = false;
        halo_plan.unsupported_reason =
            exchange_plan.unsupported_reason.empty()
                ? "exchange plan is unsupported"
                : exchange_plan.unsupported_reason;
        return halo_plan;
    }

    halo_plan.send_transfers.reserve(exchange_plan.send_transfers.size());
    for (const auto& transfer : exchange_plan.send_transfers) {
        std::string reason;
        PeerHaloExchange halo_transfer;
        halo_transfer.peer_rank = transfer.peer_rank;
        halo_transfer.local_spans =
            compact_slots_to_spans(transfer.local_slots, &reason);
        if (!reason.empty()) {
            halo_plan.supported = false;
            halo_plan.unsupported_reason =
                "send peer " + std::to_string(transfer.peer_rank) +
                ": " + reason;
            return halo_plan;
        }
        halo_plan.send_transfers.push_back(std::move(halo_transfer));
    }

    halo_plan.recv_transfers.reserve(exchange_plan.recv_transfers.size());
    for (const auto& transfer : exchange_plan.recv_transfers) {
        std::string reason;
        PeerHaloExchange halo_transfer;
        halo_transfer.peer_rank = transfer.peer_rank;
        halo_transfer.local_spans =
            compact_slots_to_spans(transfer.local_slots, &reason);
        if (!reason.empty()) {
            halo_plan.supported = false;
            halo_plan.unsupported_reason =
                "recv peer " + std::to_string(transfer.peer_rank) +
                ": " + reason;
            return halo_plan;
        }
        halo_plan.recv_transfers.push_back(std::move(halo_transfer));
    }

    halo_plan.supported = true;
    return halo_plan;
}

}  // namespace mpi
}  // namespace dacpp

#endif
