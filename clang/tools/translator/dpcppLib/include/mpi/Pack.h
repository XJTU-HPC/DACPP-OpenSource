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
    for (std::size_t idx = 0; idx < pack.globals.size(); ++idx) {
        pack.g2l.emplace(pack.globals[idx], static_cast<int32_t>(idx));
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
    pack.writeback_globals = pack.globals;
    pack.writeback_segments = pack.segments;
    return pack;
}

inline int32_t try_lookup_local_slot(const PackMap& pack, int64_t global_idx) {
    const auto it = pack.g2l.find(global_idx);
    if (it == pack.g2l.end()) {
        return -1;
    }
    return it->second;
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
    for (const auto& g2l_entry : pack.g2l) {
        if (g2l_entry.first < 0 ||
            static_cast<std::size_t>(g2l_entry.first) >= dense_size) {
            std::string message =
                where ? where : "build_dense_global_to_local_lut";
            message += ": global ";
            message += std::to_string(g2l_entry.first);
            message += " outside dense extent ";
            message += std::to_string(dense_size);
            message += " — pack layout mismatch";
            throw std::runtime_error(message);
        }
        lut[static_cast<std::size_t>(g2l_entry.first)] = g2l_entry.second;
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
    if (include_writeback) {
        plan.pack.writeback_globals = plan.pack.globals;
        plan.pack.writeback_segments = plan.pack.segments;
    }

    std::vector<std::vector<int32_t>> key_slots;
    key_slots.reserve(unique_positions.size());
    for (const auto& positions : unique_positions) {
        std::vector<int32_t> slots_for_key;
        slots_for_key.reserve(positions.size());
        for (int64_t global_idx : positions) {
            slots_for_key.push_back(
                lookup_local_slot_or_throw(plan.pack, global_idx, "build_pack_plan"));
        }
        key_slots.push_back(std::move(slots_for_key));
    }

    plan.compact_slots.reserve(unique_positions.size() * static_cast<std::size_t>(elem_count));
    for (const auto& slots_for_key : key_slots) {
        plan.compact_slots.insert(plan.compact_slots.end(),
                                   slots_for_key.begin(), slots_for_key.end());
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
inline void apply_writeback_by_globals(const std::vector<T>& local_data,
                                       const std::vector<int64_t>& globals,
                                       std::vector<T>& global_data) {
    for (std::size_t idx = 0; idx < globals.size(); ++idx) {
        global_data[static_cast<std::size_t>(globals[idx])] = local_data[idx];
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


}  // namespace mpi
}  // namespace dacpp

#endif
