#ifndef DACPP_MPI_PACK_H
#define DACPP_MPI_PACK_H

#include "Common.h"

namespace dacpp {
namespace mpi {

inline std::vector<LinearSegment> build_segments(const std::vector<int64_t>& globals) {
    std::vector<LinearSegment> segments;
    if (globals.empty()) {
        return segments;
    }

    LinearSegment cur{globals[0], 1, 0};
    for (std::size_t idx = 1; idx < globals.size(); ++idx) {
        if (globals[idx] == globals[idx - 1] + 1) {
            ++cur.len;
            continue;
        }
        segments.push_back(cur);
        cur = LinearSegment{globals[idx], 1, static_cast<int64_t>(idx)};
    }
    segments.push_back(cur);
    return segments;
}

inline PackMap make_pack_map_from_globals(
    std::vector<int64_t> globals,
    PackLayoutKind layout_kind = PackLayoutKind::ItemDerived,
    std::size_t dense_extent = 0) {
    std::sort(globals.begin(), globals.end());
    globals.erase(std::unique(globals.begin(), globals.end()), globals.end());

    PackMap pack;
    pack.layout_kind = layout_kind;
    pack.dense_extent = dense_extent;
    pack.globals = std::move(globals);
    pack.segments = build_segments(pack.globals);

    // Dense or near-dense packs are common for matrix rows/columns after
    // compaction.  Segment lookup avoids building a large hash table in those
    // cases; keep the hash only for highly fragmented layouts.
    const bool fragmented =
        pack.segments.size() > 64 &&
        pack.segments.size() * 4 > pack.globals.size();
    if (fragmented) {
        pack.g2l.reserve(pack.globals.size());
        for (std::size_t idx = 0; idx < pack.globals.size(); ++idx) {
            pack.g2l.emplace(pack.globals[idx], static_cast<int32_t>(idx));
        }
    }
    return pack;
}

inline PackMap make_dense_cover_pack(std::size_t dense_size) {
    std::vector<int64_t> globals(dense_size);
    for (std::size_t idx = 0; idx < dense_size; ++idx) {
        globals[idx] = static_cast<int64_t>(idx);
    }
    PackMap pack = make_pack_map_from_globals(
        std::move(globals), PackLayoutKind::DenseCover, dense_size);
    return pack;
}

inline int32_t try_lookup_local_slot(const PackMap& pack, int64_t global_idx) {
    if (pack.layout_kind == PackLayoutKind::DenseCover &&
        global_idx >= 0 &&
        static_cast<std::size_t>(global_idx) < pack.dense_extent) {
        return static_cast<int32_t>(global_idx);
    }

    if (!pack.g2l.empty()) {
        const auto it = pack.g2l.find(global_idx);
        if (it != pack.g2l.end()) {
            return it->second;
        }
    }

    auto it = std::upper_bound(
        pack.segments.begin(),
        pack.segments.end(),
        global_idx,
        [](int64_t value, const LinearSegment& segment) {
            return value < segment.global_begin;
        });
    if (it != pack.segments.begin()) {
        --it;
        if (global_idx >= it->global_begin &&
            global_idx < it->global_begin + it->len) {
            return static_cast<int32_t>(
                it->local_begin + (global_idx - it->global_begin));
        }
    }

    return -1;
}

inline int32_t lookup_local_slot_or_throw(const PackMap& pack,
                                          int64_t global_idx,
                                          const char* where) {
    const int32_t slot = try_lookup_local_slot(pack, global_idx);
    if (slot >= 0) {
        return slot;
    }

    std::string message = where ? where : "lookup_local_slot_or_throw";
    message += ": global ";
    message += std::to_string(global_idx);
    message += " not found in pack layout mismatch (kind=";
    message += pack_layout_kind_name(pack.layout_kind);
    message += ")";
    throw std::runtime_error(message);
}

inline std::vector<int32_t> build_dense_global_to_local_lut(
    const PackMap& pack,
    std::size_t dense_size,
    const char* where) {
    if (pack.layout_kind == PackLayoutKind::DenseCover &&
        pack.dense_extent != 0 && pack.dense_extent != dense_size) {
        std::string message = where ? where : "build_dense_global_to_local_lut";
        message += ": dense extent mismatch, pack dense_extent=";
        message += std::to_string(pack.dense_extent);
        message += ", requested=";
        message += std::to_string(dense_size);
        throw std::runtime_error(message);
    }

    std::vector<int32_t> lut(dense_size, -1);
    for (std::size_t local_idx = 0; local_idx < pack.globals.size(); ++local_idx) {
        const int64_t global_idx = pack.globals[local_idx];
        if (global_idx < 0 ||
            static_cast<std::size_t>(global_idx) >= dense_size) {
            std::string message =
                where ? where : "build_dense_global_to_local_lut";
            message += ": global ";
            message += std::to_string(global_idx);
            message += " outside dense extent ";
            message += std::to_string(dense_size);
            message += " — pack layout mismatch";
            throw std::runtime_error(message);
        }
        lut[static_cast<std::size_t>(global_idx)] =
            static_cast<int32_t>(local_idx);
    }
    return lut;
}

inline std::vector<int> build_item_bind_key(int64_t item_id,
                                            const AccessPattern& pattern) {
    const std::vector<int64_t> bind_splits =
        pattern.bind_split_sizes.empty() ? init_bind_split_sizes(pattern)
                                         : pattern.bind_split_sizes;
    const std::vector<int> bind_indices = decode_item_id(item_id, bind_splits);

    std::vector<int> key(bind_splits.size(), 0);
    std::vector<bool> used(bind_splits.size(), false);
    for (int op_idx = 0; op_idx < pattern.param_ops.size; ++op_idx) {
        if (op_idx >= static_cast<int>(pattern.bind_set_id.size())) {
            continue;
        }
        const int bind_id = pattern.bind_set_id[op_idx];
        if (bind_id < 0 || bind_id >= static_cast<int>(key.size())) {
            continue;
        }
        key[bind_id] =
            bind_id < static_cast<int>(bind_indices.size()) ? bind_indices[bind_id] : 0;
        used[bind_id] = true;
    }

    for (std::size_t idx = 0; idx < key.size(); ++idx) {
        if (!used[idx]) {
            key[idx] = -1;
        }
    }
    return key;
}

inline PackPlan build_pack_plan(ItemRange range,
                                const AccessPattern& pattern,
                                bool include_writeback) {
    PackPlan plan;
    const int64_t item_count = range.size();
    const int64_t elem_count = partition_element_count(pattern);

    std::vector<std::vector<int64_t>> unique_positions;
    std::vector<int32_t> item_key_indices;
    item_key_indices.reserve(static_cast<std::size_t>(std::max<int64_t>(item_count, 0)));

    std::unordered_map<std::vector<int>, int32_t, VectorIntHash> key_to_index;
    key_to_index.reserve(static_cast<std::size_t>(std::max<int64_t>(item_count, 0)));

    for (int64_t item = range.begin; item < range.end; ++item) {
        const std::vector<int> key = build_item_bind_key(item, pattern);
        auto it = key_to_index.find(key);
        if (it == key_to_index.end()) {
            const int32_t key_index = static_cast<int32_t>(unique_positions.size());
            it = key_to_index.emplace(key, key_index).first;
            unique_positions.push_back(collect_positions_for_item(item, pattern));
        }
        item_key_indices.push_back(it->second);
    }

    std::vector<int64_t> globals;
    globals.reserve(unique_positions.size() * static_cast<std::size_t>(elem_count));
    for (const auto& positions : unique_positions) {
        globals.insert(globals.end(), positions.begin(), positions.end());
    }

    plan.pack = make_pack_map_from_globals(std::move(globals));
    (void)include_writeback;

    plan.compact_slots.reserve(unique_positions.size() * static_cast<std::size_t>(elem_count));
    for (const auto& positions : unique_positions) {
        for (int64_t global_idx : positions) {
            plan.compact_slots.push_back(
                lookup_local_slot_or_throw(plan.pack, global_idx, "build_pack_plan"));
        }
    }

    plan.item_key_offsets.reserve(item_key_indices.size());
    for (int32_t key_index : item_key_indices) {
        plan.item_key_offsets.push_back(
            key_index * static_cast<int32_t>(elem_count));
    }

    return plan;
}

inline PackPlan build_input_pack_plan(ItemRange range,
                                      const AccessPattern& pattern) {
    return build_pack_plan(range, pattern, false);
}

inline PackPlan build_output_pack_plan(ItemRange range,
                                       const AccessPattern& pattern) {
    return build_pack_plan(range, pattern, true);
}

inline PackPlan build_rw_pack_plan(ItemRange range,
                                   const AccessPattern& pattern) {
    return build_pack_plan(range, pattern, true);
}

inline void init_gathered_index_layout(GatheredIndexLayout& layout,
                                       const std::vector<int64_t>& local_globals,
                                       int mpi_rank,
                                       int mpi_size,
                                       MPI_Comm comm = MPI_COMM_WORLD) {
    layout.local_count = static_cast<int>(local_globals.size());
    layout.counts.clear();
    layout.displs.clear();
    layout.byte_counts.clear();
    layout.byte_displs.clear();
    layout.globals.clear();

    if (mpi_rank == 0) {
        layout.counts.resize(mpi_size);
        layout.displs.resize(mpi_size);
    }

    MPI_Gather(&layout.local_count, 1, MPI_INT,
               mpi_rank == 0 ? layout.counts.data() : nullptr, 1, MPI_INT,
               0, comm);

    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int rank = 0; rank < mpi_size; ++rank) {
            layout.displs[rank] = current_displ;
            current_displ += layout.counts[rank];
        }
        layout.globals.resize(current_displ);
    }

    MPI_Gatherv(const_cast<int64_t*>(local_globals.data()), layout.local_count,
                MPI_LONG_LONG,
                mpi_rank == 0 ? layout.globals.data() : nullptr,
                mpi_rank == 0 ? layout.counts.data() : nullptr,
                mpi_rank == 0 ? layout.displs.data() : nullptr,
                MPI_LONG_LONG, 0, comm);
}

inline void init_all_rank_index_layout(AllRankIndexLayout& layout,
                                       const std::vector<int64_t>& local_globals,
                                       int mpi_rank,
                                       int mpi_size,
                                       MPI_Comm comm = MPI_COMM_WORLD) {
    (void)mpi_rank;
    layout.local_count = static_cast<int>(local_globals.size());
    layout.counts.assign(static_cast<std::size_t>(mpi_size), 0);
    layout.displs.assign(static_cast<std::size_t>(mpi_size), 0);
    layout.globals.clear();

    MPI_Allgather(&layout.local_count, 1, MPI_INT,
                  layout.counts.data(), 1, MPI_INT, comm);

    int current_displ = 0;
    for (int rank = 0; rank < mpi_size; ++rank) {
        layout.displs[static_cast<std::size_t>(rank)] = current_displ;
        current_displ += layout.counts[static_cast<std::size_t>(rank)];
    }
    layout.globals.resize(static_cast<std::size_t>(current_displ));

    MPI_Allgatherv(const_cast<int64_t*>(local_globals.data()), layout.local_count,
                   MPI_LONG_LONG, layout.globals.data(), layout.counts.data(),
                   layout.displs.data(), MPI_LONG_LONG, comm);
}

inline void init_layout_byte_counts(GatheredIndexLayout& layout,
                                    std::size_t value_size) {
    layout.byte_counts = layout.counts;
    layout.byte_displs = layout.displs;
    for (std::size_t idx = 0; idx < layout.byte_counts.size(); ++idx) {
        layout.byte_counts[idx] *= static_cast<int>(value_size);
        layout.byte_displs[idx] *= static_cast<int>(value_size);
    }
}

inline bool validate_unique_writers(const AllRankIndexLayout& layout,
                                    int mpi_size,
                                    std::string* reason = nullptr) {
    std::unordered_map<int64_t, int> owner_by_global;
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = layout.counts[static_cast<std::size_t>(rank)];
        const int displ = layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            const int64_t global_idx =
                layout.globals[static_cast<std::size_t>(displ + idx)];
            auto [it, inserted] = owner_by_global.emplace(global_idx, rank);
            if (!inserted && it->second != rank) {
                if (reason) {
                    *reason = "global " + std::to_string(global_idx) +
                              " written by ranks " +
                              std::to_string(it->second) + " and " +
                              std::to_string(rank);
                }
                return false;
            }
        }
    }
    return true;
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

template <typename T>
inline std::vector<T> pack_values_by_globals(const std::vector<T>& global_data,
                                             const std::vector<int64_t>& globals) {
    std::vector<T> packed;
    packed.reserve(globals.size());
    for (int64_t global_idx : globals) {
        packed.push_back(global_data[static_cast<std::size_t>(global_idx)]);
    }
    return packed;
}

template <typename T>
inline std::vector<T> pack_values_by_globals_range(const std::vector<T>& global_data,
                                                   const int64_t* globals,
                                                   std::size_t count) {
    std::vector<T> packed;
    packed.reserve(count);
    for (std::size_t idx = 0; idx < count; ++idx) {
        packed.push_back(global_data[static_cast<std::size_t>(globals[idx])]);
    }
    return packed;
}

template <typename T>
inline void pack_values_by_globals_range_into(
    const std::vector<T>& global_data,
    const int64_t* globals,
    std::size_t count,
    std::vector<T>& packed) {
    packed.resize(count);
    for (std::size_t idx = 0; idx < count; ++idx) {
        packed[idx] = global_data[static_cast<std::size_t>(globals[idx])];
    }
}

template <typename T>
inline std::vector<T> pack_values_by_globals_parallel(
    const std::vector<T>& global_data,
    const std::vector<int64_t>& globals,
    std::size_t threshold = 1 << 18) {
    if (globals.size() < threshold) {
        return pack_values_by_globals(global_data, globals);
    }

    std::vector<T> packed(globals.size());
    if (globals.empty()) {
        return packed;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> global_buf(
            const_cast<T*>(global_data.data()),
            sycl::range<1>(global_data.size()));
        sycl::buffer<int64_t, 1> globals_buf(
            const_cast<int64_t*>(globals.data()),
            sycl::range<1>(globals.size()));
        sycl::buffer<T, 1> packed_buf(
            packed.data(),
            sycl::range<1>(packed.size()));

        q.submit([&](sycl::handler& h) {
            auto global_acc = global_buf.template get_access<sycl::access::mode::read>(h);
            auto globals_acc = globals_buf.template get_access<sycl::access::mode::read>(h);
            auto packed_acc = packed_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(globals.size()), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                packed_acc[i] = global_acc[static_cast<std::size_t>(globals_acc[i])];
            });
        });
        q.wait();
    }
    return packed;
}

template <typename T>
inline std::vector<T> pack_values_by_globals_parallel_range(
    const std::vector<T>& global_data,
    const int64_t* globals,
    std::size_t count,
    std::size_t threshold = 1 << 18) {
    if (count < threshold) {
        return pack_values_by_globals_range(global_data, globals, count);
    }

    std::vector<T> packed(count);
    if (count == 0) {
        return packed;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> global_buf(
            const_cast<T*>(global_data.data()),
            sycl::range<1>(global_data.size()));
        sycl::buffer<int64_t, 1> globals_buf(
            const_cast<int64_t*>(globals),
            sycl::range<1>(count));
        sycl::buffer<T, 1> packed_buf(
            packed.data(),
            sycl::range<1>(packed.size()));

        q.submit([&](sycl::handler& h) {
            auto global_acc = global_buf.template get_access<sycl::access::mode::read>(h);
            auto globals_acc = globals_buf.template get_access<sycl::access::mode::read>(h);
            auto packed_acc = packed_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                packed_acc[i] = global_acc[static_cast<std::size_t>(globals_acc[i])];
            });
        });
        q.wait();
    }
    return packed;
}

template <typename T>
inline void pack_values_by_globals_parallel_range_into(
    const std::vector<T>& global_data,
    const int64_t* globals,
    std::size_t count,
    std::vector<T>& packed,
    std::size_t threshold = 1 << 18) {
    if (count < threshold) {
        pack_values_by_globals_range_into(global_data, globals, count, packed);
        return;
    }

    packed.resize(count);
    if (count == 0) {
        return;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> global_buf(
            const_cast<T*>(global_data.data()),
            sycl::range<1>(global_data.size()));
        sycl::buffer<int64_t, 1> globals_buf(
            const_cast<int64_t*>(globals),
            sycl::range<1>(count));
        sycl::buffer<T, 1> packed_buf(
            packed.data(),
            sycl::range<1>(packed.size()));

        q.submit([&](sycl::handler& h) {
            auto global_acc = global_buf.template get_access<sycl::access::mode::read>(h);
            auto globals_acc = globals_buf.template get_access<sycl::access::mode::read>(h);
            auto packed_acc = packed_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                packed_acc[i] = global_acc[static_cast<std::size_t>(globals_acc[i])];
            });
        });
        q.wait();
    }
}

template <typename T>
inline void apply_writeback_by_globals(const std::vector<T>& local_data,
                                       const std::vector<int64_t>& globals,
                                       std::vector<T>& global_data) {
    for (std::size_t idx = 0; idx < globals.size(); ++idx) {
        global_data[static_cast<std::size_t>(globals[idx])] = local_data[idx];
    }
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

template <typename T>
inline std::vector<T> build_writeback_values(const std::vector<T>& local_data,
                                             const PackMap& pack) {
    const std::vector<int64_t>& globals =
        pack.writeback_globals.empty() ? pack.globals : pack.writeback_globals;
    std::vector<T> values;
    values.reserve(globals.size());
    for (int64_t global_idx : globals) {
        values.push_back(local_data[static_cast<std::size_t>(
            lookup_local_slot_or_throw(
                pack, global_idx, "build_writeback_values"))]);
    }
    return values;
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
inline std::vector<T> build_writeback_values_parallel(
    const std::vector<T>& local_data,
    const PackMap& pack,
    std::size_t threshold = 1 << 18) {
    const std::vector<int64_t>& globals =
        pack.writeback_globals.empty() ? pack.globals : pack.writeback_globals;
    if (globals.size() < threshold) {
        return build_writeback_values(local_data, pack);
    }

    std::vector<int32_t> local_slots;
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        local_slots.push_back(
            lookup_local_slot_or_throw(
                pack, global_idx, "build_writeback_values_parallel"));
    }

    std::vector<T> values(globals.size());
    if (globals.empty()) {
        return values;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> local_buf(
            const_cast<T*>(local_data.data()),
            sycl::range<1>(local_data.size()));
        sycl::buffer<int32_t, 1> slots_buf(
            local_slots.data(),
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
    return values;
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


}  // namespace mpi
}  // namespace dacpp

#endif
