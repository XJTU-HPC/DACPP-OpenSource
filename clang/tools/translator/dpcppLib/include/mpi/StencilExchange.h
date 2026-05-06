#ifndef DACPP_MPI_STENCIL_EXCHANGE_H
#define DACPP_MPI_STENCIL_EXCHANGE_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

#include "MpiTypes.h"
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

inline bool build_span_pairs_from_slots(
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<SlotSpan>& source_spans,
    std::vector<SlotSpan>& target_spans,
    std::string* unsupported_reason = nullptr) {
    source_spans.clear();
    target_spans.clear();

    if (source_slots.size() != target_slots.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target slot counts differ";
        }
        return false;
    }
    if (source_slots.empty()) {
        return true;
    }

    std::size_t idx = 0;
    while (idx < source_slots.size()) {
        const int32_t source_begin = source_slots[idx];
        const int32_t target_begin = target_slots[idx];
        if (source_begin < 0 || target_begin < 0) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair contains negative slot";
            }
            return false;
        }

        int32_t source_stride = 1;
        int32_t target_stride = 1;
        std::size_t count = 1;
        if (idx + 1 < source_slots.size()) {
            source_stride = source_slots[idx + 1] - source_slots[idx];
            target_stride = target_slots[idx + 1] - target_slots[idx];
            if (source_stride <= 0 || target_stride <= 0) {
                if (unsupported_reason) {
                    *unsupported_reason = "span pair contains non-monotonic slots";
                }
                return false;
            }
            count = 2;
            idx += 2;
            while (idx < source_slots.size() &&
                   source_slots[idx] == source_slots[idx - 1] + source_stride &&
                   target_slots[idx] == target_slots[idx - 1] + target_stride) {
                ++count;
                ++idx;
            }
        } else {
            ++idx;
        }

        source_spans.push_back(
            SlotSpan{source_begin, static_cast<int32_t>(count), source_stride});
        target_spans.push_back(
            SlotSpan{target_begin, static_cast<int32_t>(count), target_stride});
    }

    return true;
}

inline bool build_local_span_pairs_from_slots(
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<SlotSpan>& source_spans,
    std::vector<SlotSpan>& target_spans,
    std::string* unsupported_reason = nullptr) {
    source_spans.clear();
    target_spans.clear();

    if (source_slots.size() != target_slots.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target slot counts differ";
        }
        return false;
    }

    std::vector<int32_t> filtered_source_slots;
    std::vector<int32_t> filtered_target_slots;
    filtered_source_slots.reserve(source_slots.size());
    filtered_target_slots.reserve(target_slots.size());
    for (std::size_t idx = 0; idx < source_slots.size(); ++idx) {
        const int32_t source_slot = source_slots[idx];
        const int32_t target_slot = target_slots[idx];
        if (target_slot < 0) {
            continue;
        }
        if (source_slot < 0) {
            if (unsupported_reason) {
                *unsupported_reason = "local span pair contains negative source slot";
            }
            return false;
        }
        filtered_source_slots.push_back(source_slot);
        filtered_target_slots.push_back(target_slot);
    }

    return build_span_pairs_from_slots(filtered_source_slots, filtered_target_slots,
                                       source_spans, target_spans,
                                       unsupported_reason);
}

inline bool build_contiguous_row_copy_blocks_from_span_pairs(
    const std::vector<SlotSpan>& source_spans,
    const std::vector<SlotSpan>& target_spans,
    std::vector<ContiguousRowCopyBlock>& blocks,
    std::string* unsupported_reason = nullptr) {
    blocks.clear();

    if (source_spans.size() != target_spans.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target span counts differ";
        }
        return false;
    }
    if (source_spans.empty()) {
        return true;
    }

    for (std::size_t idx = 0; idx < source_spans.size(); ++idx) {
        const SlotSpan& source_span = source_spans[idx];
        const SlotSpan& target_span = target_spans[idx];
        if (source_span.count != target_span.count) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair count differs";
            }
            blocks.clear();
            return false;
        }
        if (source_span.count <= 0) {
            continue;
        }
        if (source_span.stride <= 0 || target_span.stride <= 0) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair contains non-positive stride";
            }
            blocks.clear();
            return false;
        }

        if (blocks.empty()) {
            blocks.push_back(ContiguousRowCopyBlock{
                source_span.begin,
                target_span.begin,
                source_span.count,
                1,
                source_span.stride,
                target_span.stride,
            });
            continue;
        }

        ContiguousRowCopyBlock& last = blocks.back();
        const int32_t expected_source_begin =
            last.source_begin + last.row_count * last.source_row_stride;
        const int32_t expected_target_begin =
            last.target_begin + last.row_count * last.target_row_stride;
        if (last.row_width == source_span.count &&
            last.source_row_stride == source_span.stride &&
            last.target_row_stride == target_span.stride &&
            expected_source_begin == source_span.begin &&
            expected_target_begin == target_span.begin) {
            ++last.row_count;
            continue;
        }

        blocks.push_back(ContiguousRowCopyBlock{
            source_span.begin,
            target_span.begin,
            source_span.count,
            1,
            source_span.stride,
            target_span.stride,
        });
    }

    return true;
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

template <typename T>
inline void pack_values_by_slots_into(const std::vector<T>& local_data,
                                      const std::vector<int32_t>& local_slots,
                                      std::vector<T>& values) {
    values.resize(local_slots.size());
    for (std::size_t idx = 0; idx < local_slots.size(); ++idx) {
        values[idx] = local_data[static_cast<std::size_t>(local_slots[idx])];
    }
}

template <typename T>
inline void unpack_values_by_slots_into(const std::vector<T>& values,
                                        const std::vector<int32_t>& local_slots,
                                        std::vector<T>& target) {
    const std::size_t count = std::min(values.size(), local_slots.size());
    for (std::size_t idx = 0; idx < count; ++idx) {
        const int32_t slot = local_slots[idx];
        if (slot < 0 ||
            static_cast<std::size_t>(slot) >= target.size()) {
            continue;
        }
        target[static_cast<std::size_t>(slot)] = values[idx];
    }
}

template <typename T>
inline void pack_values_by_slots_parallel_into(
    const std::vector<T>& local_data,
    const std::vector<int32_t>& local_slots,
    std::vector<T>& values,
    std::size_t threshold = 1 << 18) {
    if (local_slots.size() < threshold) {
        pack_values_by_slots_into(local_data, local_slots, values);
        return;
    }

    values.resize(local_slots.size());
    if (local_slots.empty()) {
        return;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> local_buf(
            const_cast<T*>(local_data.data()),
            sycl::range<1>(local_data.size()));
        sycl::buffer<int32_t, 1> slots_buf(
            const_cast<int32_t*>(local_slots.data()),
            sycl::range<1>(local_slots.size()));
        sycl::buffer<T, 1> values_buf(
            values.data(),
            sycl::range<1>(values.size()));

        q.submit([&](sycl::handler& h) {
            auto local_acc = local_buf.template get_access<sycl::access::mode::read>(h);
            auto slots_acc = slots_buf.template get_access<sycl::access::mode::read>(h);
            auto values_acc = values_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(values.size()), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                values_acc[i] = local_acc[static_cast<std::size_t>(slots_acc[i])];
            });
        });
        q.wait();
    }
}

template <typename T>
inline void scatter_values_by_slots_parallel_into(
    const std::vector<T>& source,
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<T>& target,
    sycl::queue& q,
    std::size_t threshold = 1 << 18) {
    const std::size_t count = std::min(source_slots.size(), target_slots.size());
    if (count == 0) {
        return;
    }

    if (count < threshold || source.data() == target.data()) {
        for (std::size_t idx = 0; idx < count; ++idx) {
            const int32_t source_slot = source_slots[idx];
            const int32_t target_slot = target_slots[idx];
            if (source_slot < 0 || target_slot < 0 ||
                static_cast<std::size_t>(source_slot) >= source.size() ||
                static_cast<std::size_t>(target_slot) >= target.size()) {
                continue;
            }
            target[static_cast<std::size_t>(target_slot)] =
                source[static_cast<std::size_t>(source_slot)];
        }
        return;
    }

    {
        sycl::buffer<T, 1> source_buf(
            const_cast<T*>(source.data()),
            sycl::range<1>(source.size()));
        sycl::buffer<int32_t, 1> source_slots_buf(
            const_cast<int32_t*>(source_slots.data()),
            sycl::range<1>(source_slots.size()));
        sycl::buffer<int32_t, 1> target_slots_buf(
            const_cast<int32_t*>(target_slots.data()),
            sycl::range<1>(target_slots.size()));
        sycl::buffer<T, 1> target_buf(
            target.data(),
            sycl::range<1>(target.size()));

        q.submit([&](sycl::handler& h) {
            auto source_acc =
                source_buf.template get_access<sycl::access::mode::read>(h);
            auto source_slots_acc =
                source_slots_buf.template get_access<sycl::access::mode::read>(h);
            auto target_slots_acc =
                target_slots_buf.template get_access<sycl::access::mode::read>(h);
            auto target_acc =
                target_buf.template get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                const int32_t source_slot = source_slots_acc[i];
                const int32_t target_slot = target_slots_acc[i];
                if (source_slot < 0 || target_slot < 0 ||
                    static_cast<std::size_t>(source_slot) >= source_acc.get_count() ||
                    static_cast<std::size_t>(target_slot) >= target_acc.get_count()) {
                    return;
                }
                target_acc[static_cast<std::size_t>(target_slot)] =
                    source_acc[static_cast<std::size_t>(source_slot)];
            });
        });
        q.wait();
    }
}

template <typename T>
inline void scatter_values_by_slots_parallel_into(
    const std::vector<T>& source,
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<T>& target,
    std::size_t threshold = 1 << 18) {
    sycl::queue q(sycl::default_selector_v);
    scatter_values_by_slots_parallel_into(source, source_slots, target_slots,
                                          target, q, threshold);
}

template <typename T>
inline std::size_t halo_value_count(
    const std::vector<SlotSpan>& spans) {
    std::size_t count = 0;
    for (const SlotSpan& span : spans) {
        if (span.count > 0) {
            count += static_cast<std::size_t>(span.count);
        }
    }
    return count;
}

template <typename T>
inline void prepare_halo_exchange_runtime(const HaloExchangePlan& plan,
                                          HaloExchangeRuntime<T>& runtime) {
    runtime.send_buffers.resize(plan.send_transfers.size());
    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        runtime.send_buffers[idx].resize(
            halo_value_count<T>(plan.send_transfers[idx].local_spans));
    }
    runtime.recv_buffers.resize(plan.recv_transfers.size());
    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        runtime.recv_buffers[idx].resize(
            halo_value_count<T>(plan.recv_transfers[idx].local_spans));
    }
    runtime.requests.resize(plan.send_transfers.size() + plan.recv_transfers.size());
}

template <typename T>
inline void pack_values_by_spans_into(const std::vector<T>& local_data,
                                      const std::vector<SlotSpan>& spans,
                                      std::vector<T>& values) {
    values.resize(halo_value_count<T>(spans));
    std::size_t out_idx = 0;
    for (const SlotSpan& span : spans) {
        if (span.count <= 0) {
            continue;
        }
        if (span.stride == 1 &&
            span.begin >= 0 &&
            static_cast<std::size_t>(span.begin) + static_cast<std::size_t>(span.count) <=
                local_data.size()) {
            std::copy_n(local_data.begin() + span.begin, span.count,
                        values.begin() + out_idx);
            out_idx += static_cast<std::size_t>(span.count);
            continue;
        }
        for (int32_t offset = 0; offset < span.count; ++offset) {
            values[out_idx++] = local_data[static_cast<std::size_t>(
                span.begin + offset * span.stride)];
        }
    }
}

template <typename T>
inline void unpack_values_by_spans_into(const std::vector<T>& values,
                                        const std::vector<SlotSpan>& spans,
                                        std::vector<T>& target) {
    std::size_t in_idx = 0;
    for (const SlotSpan& span : spans) {
        if (span.count <= 0) {
            continue;
        }
        if (span.stride == 1 &&
            span.begin >= 0 &&
            in_idx + static_cast<std::size_t>(span.count) <= values.size() &&
            static_cast<std::size_t>(span.begin) + static_cast<std::size_t>(span.count) <=
                target.size()) {
            std::copy_n(values.begin() + in_idx, span.count,
                        target.begin() + span.begin);
            in_idx += static_cast<std::size_t>(span.count);
            continue;
        }
        for (int32_t offset = 0; offset < span.count; ++offset) {
            if (in_idx >= values.size()) {
                return;
            }
            const int32_t slot = span.begin + offset * span.stride;
            if (slot >= 0 && static_cast<std::size_t>(slot) < target.size()) {
                target[static_cast<std::size_t>(slot)] = values[in_idx];
            }
            ++in_idx;
        }
    }
}

template <typename T>
inline void scatter_values_by_span_pairs_into(
    const std::vector<T>& source,
    const std::vector<SlotSpan>& source_spans,
    const std::vector<SlotSpan>& target_spans,
    std::vector<T>& target) {
    const std::size_t count =
        std::min(source_spans.size(), target_spans.size());
    for (std::size_t idx = 0; idx < count; ++idx) {
        const SlotSpan& source_span = source_spans[idx];
        const SlotSpan& target_span = target_spans[idx];
        const int32_t span_count =
            std::min(source_span.count, target_span.count);
        if (span_count <= 0) {
            continue;
        }
        if (source.data() != target.data() &&
            source_span.stride == 1 &&
            target_span.stride == 1 &&
            source_span.begin >= 0 &&
            target_span.begin >= 0 &&
            static_cast<std::size_t>(source_span.begin) + static_cast<std::size_t>(span_count) <=
                source.size() &&
            static_cast<std::size_t>(target_span.begin) + static_cast<std::size_t>(span_count) <=
                target.size()) {
            std::copy_n(source.begin() + source_span.begin, span_count,
                        target.begin() + target_span.begin);
            continue;
        }
        for (int32_t offset = 0; offset < span_count; ++offset) {
            const int32_t source_slot =
                source_span.begin + offset * source_span.stride;
            const int32_t target_slot =
                target_span.begin + offset * target_span.stride;
            if (source_slot < 0 || target_slot < 0 ||
                static_cast<std::size_t>(source_slot) >= source.size() ||
                static_cast<std::size_t>(target_slot) >= target.size()) {
                continue;
            }
            target[static_cast<std::size_t>(target_slot)] =
                source[static_cast<std::size_t>(source_slot)];
        }
    }
}

template <typename T>
inline void scatter_values_by_row_copy_blocks_into(
    const std::vector<T>& source,
    const std::vector<ContiguousRowCopyBlock>& blocks,
    std::vector<T>& target) {
    for (const ContiguousRowCopyBlock& block : blocks) {
        if (block.row_width <= 0 || block.row_count <= 0 ||
            block.source_begin < 0 || block.target_begin < 0 ||
            block.source_row_stride <= 0 || block.target_row_stride <= 0) {
            continue;
        }
        for (int32_t row = 0; row < block.row_count; ++row) {
            const int32_t source_row_begin =
                block.source_begin + row * block.source_row_stride;
            const int32_t target_row_begin =
                block.target_begin + row * block.target_row_stride;
            if (static_cast<std::size_t>(source_row_begin) +
                    static_cast<std::size_t>(block.row_width) > source.size() ||
                static_cast<std::size_t>(target_row_begin) +
                    static_cast<std::size_t>(block.row_width) > target.size()) {
                break;
            }
            std::copy_n(source.begin() + source_row_begin, block.row_width,
                        target.begin() + target_row_begin);
        }
    }
}

template <typename T>
inline void exchange_values_by_slots(const std::vector<T>& send_source,
                                     const ExchangePlan& plan,
                                     std::vector<T>& recv_target,
                                     MPI_Comm comm = MPI_COMM_WORLD) {
    if (!plan.supported) {
        return;
    }

    const MPI_Datatype mpi_type = mpi_datatype_for_value<T>();
    std::vector<std::vector<T>> send_buffers(plan.send_transfers.size());
    std::vector<std::vector<T>> recv_buffers(plan.recv_transfers.size());
    std::vector<MPI_Request> requests;
    requests.reserve(plan.send_transfers.size() + plan.recv_transfers.size());

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        const auto& transfer = plan.recv_transfers[idx];
        recv_buffers[idx].resize(transfer.local_slots.size());
        if (recv_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Irecv(reinterpret_cast<unsigned char*>(recv_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Irecv(recv_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        const auto& transfer = plan.send_transfers[idx];
        pack_values_by_slots_into(send_source, transfer.local_slots, send_buffers[idx]);
        if (send_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Isend(reinterpret_cast<unsigned char*>(send_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Isend(send_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
    }

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        unpack_values_by_slots_into(recv_buffers[idx],
                                    plan.recv_transfers[idx].local_slots,
                                    recv_target);
    }
}

template <typename T>
inline void exchange_values_by_halo_spans(const std::vector<T>& send_source,
                                          const HaloExchangePlan& plan,
                                          HaloExchangeRuntime<T>& runtime,
                                          std::vector<T>& recv_target,
                                          MPI_Comm comm = MPI_COMM_WORLD) {
    if (!plan.supported) {
        return;
    }

    if (runtime.send_buffers.size() != plan.send_transfers.size() ||
        runtime.recv_buffers.size() != plan.recv_transfers.size() ||
        runtime.requests.size() !=
            plan.send_transfers.size() + plan.recv_transfers.size()) {
        prepare_halo_exchange_runtime(plan, runtime);
    }

    const MPI_Datatype mpi_type = mpi_datatype_for_value<T>();
    int request_count = 0;

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        const auto& transfer = plan.recv_transfers[idx];
        auto& recv_buffer = runtime.recv_buffers[idx];
        const std::size_t recv_count = halo_value_count<T>(transfer.local_spans);
        if (recv_buffer.size() != recv_count) {
            recv_buffer.resize(recv_count);
        }
        if (recv_buffer.empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Irecv(reinterpret_cast<unsigned char*>(recv_buffer.data()),
                      mpi_payload_count_for_values<T>(recv_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Irecv(recv_buffer.data(),
                      mpi_payload_count_for_values<T>(recv_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        runtime.requests[static_cast<std::size_t>(request_count++)] = req;
    }

    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        const auto& transfer = plan.send_transfers[idx];
        auto& send_buffer = runtime.send_buffers[idx];
        const std::size_t send_count = halo_value_count<T>(transfer.local_spans);
        if (send_buffer.size() != send_count) {
            send_buffer.resize(send_count);
        }
        pack_values_by_spans_into(send_source, transfer.local_spans, send_buffer);
        if (send_buffer.empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Isend(reinterpret_cast<unsigned char*>(send_buffer.data()),
                      mpi_payload_count_for_values<T>(send_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Isend(send_buffer.data(),
                      mpi_payload_count_for_values<T>(send_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        runtime.requests[static_cast<std::size_t>(request_count++)] = req;
    }

    if (request_count > 0) {
        MPI_Waitall(request_count, runtime.requests.data(), MPI_STATUSES_IGNORE);
    }

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        unpack_values_by_spans_into(runtime.recv_buffers[idx],
                                    plan.recv_transfers[idx].local_spans,
                                    recv_target);
    }
}

template <typename T>
inline void exchange_values_by_halo_spans(const std::vector<T>& send_source,
                                          const HaloExchangePlan& plan,
                                          std::vector<T>& recv_target,
                                          MPI_Comm comm = MPI_COMM_WORLD) {
    if (!plan.supported) {
        return;
    }

    const MPI_Datatype mpi_type = mpi_datatype_for_value<T>();
    std::vector<std::vector<T>> send_buffers(plan.send_transfers.size());
    std::vector<std::vector<T>> recv_buffers(plan.recv_transfers.size());
    std::vector<MPI_Request> requests;
    requests.reserve(plan.send_transfers.size() + plan.recv_transfers.size());

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        const auto& transfer = plan.recv_transfers[idx];
        recv_buffers[idx].resize(halo_value_count<T>(transfer.local_spans));
        if (recv_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Irecv(reinterpret_cast<unsigned char*>(recv_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Irecv(recv_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        const auto& transfer = plan.send_transfers[idx];
        pack_values_by_spans_into(send_source, transfer.local_spans,
                                  send_buffers[idx]);
        if (send_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Isend(reinterpret_cast<unsigned char*>(send_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Isend(send_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
    }

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        unpack_values_by_spans_into(recv_buffers[idx],
                                    plan.recv_transfers[idx].local_spans,
                                    recv_target);
    }
}

template <typename T>
inline void publish_local_writes_with_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        pack_values_by_slots_parallel_into(local_write_source, local_write_slots,
                                           local_write_values);
        unpack_values_by_slots_into(local_write_values, local_target_slots,
                                    local_cache);
    } else {
        local_write_values.clear();
    }

    if (!plan.supported) {
        return;
    }

    exchange_values_by_slots(local_write_source, plan, local_cache, comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        pack_values_by_slots_parallel_into(local_write_source, local_write_slots,
                                           local_write_values);
        scatter_values_by_slots_parallel_into(
            local_write_source, local_write_slots, local_target_slots,
            local_cache, q);
    } else {
        local_write_values.clear();
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, local_cache,
                                      comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_halo_or_exchange(
        local_write_source, local_write_slots, local_target_slots, local_cache,
        local_write_values, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        scatter_values_by_slots_parallel_into(
            local_write_source, local_write_slots, local_target_slots,
            local_cache, q);
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, local_cache,
                                      comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    HaloExchangeRuntime<T>& halo_runtime,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_row_copy_blocks.empty()) {
        scatter_values_by_row_copy_blocks_into(local_write_source,
                                               local_row_copy_blocks,
                                               local_cache);
    } else if (!local_write_spans.empty() && !local_target_spans.empty()) {
        scatter_values_by_span_pairs_into(local_write_source, local_write_spans,
                                          local_target_spans, local_cache);
    } else if (!local_write_slots.empty()) {
        scatter_values_by_slots_parallel_into(local_write_source,
                                              local_write_slots,
                                              local_target_slots,
                                              local_cache,
                                              q);
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, halo_runtime,
                                      local_cache, comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    HaloExchangeRuntime<T>& halo_runtime,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, halo_runtime, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    HaloExchangeRuntime<T> halo_runtime;
    if (halo_plan.supported) {
        prepare_halo_exchange_runtime(halo_plan, halo_runtime);
    }
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, local_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, halo_runtime, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, local_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_halo_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots, local_cache,
        exchange_plan, halo_plan, q, comm);
}

}  // namespace mpi
}  // namespace dacpp

#endif
