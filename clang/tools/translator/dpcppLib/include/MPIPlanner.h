#ifndef DACPP_MPI_PLANNER_H
#define DACPP_MPI_PLANNER_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataReconstructor.new.h"
#include "dacInfo.h"

namespace dacpp {
namespace mpi {

struct ItemRange {
    int64_t begin = 0;
    int64_t end = 0;

    int64_t size() const {
        return end - begin;
    }
};

enum class AccessMode {
    Read,
    Write,
    ReadWrite
};

struct LinearSegment {
    int64_t global_begin = 0;
    int64_t len = 0;
    int64_t local_begin = 0;
};

struct PackMap {
    std::vector<int64_t> globals;
    std::unordered_map<int64_t, int32_t> g2l;
    std::vector<LinearSegment> segments;

    std::vector<int64_t> writeback_globals;
    std::vector<LinearSegment> writeback_segments;
};

struct AccessPattern {
    int param_id = -1;
    std::string name;
    AccessMode mode = AccessMode::Read;

    DataInfo data_info;
    Dac_Ops param_ops;
    std::vector<int> partition_shape;

    std::vector<int> bind_set_id;
    std::vector<std::string> bind_offset_expr;
    std::vector<bool> is_index_op;
    std::vector<int64_t> bind_split_sizes;
};

inline ItemRange get_rank_item_range(int64_t total_items, int rank, int mpi_size) {
    const int64_t base = total_items / mpi_size;
    const int64_t rem = total_items % mpi_size;
    const int64_t begin = static_cast<int64_t>(rank) * base + std::min<int64_t>(rank, rem);
    const int64_t end = begin + base + (rank < rem ? 1 : 0);
    return ItemRange{begin, end};
}

inline int64_t linearize(const std::vector<int>& pos, const DataInfo& info) {
    if (static_cast<int>(pos.size()) != info.dim) {
        throw std::runtime_error("linearize: position rank mismatch");
    }

    int64_t linear = 0;
    for (int dim = 0; dim < info.dim; ++dim) {
        linear = linear * info.dimLength[dim] + pos[dim];
    }
    return linear;
}

inline std::vector<int> decode_item_id(int64_t item_id,
                                       const std::vector<int64_t>& split_sizes) {
    std::vector<int> result(split_sizes.size(), 0);
    int64_t divisor = 1;

    for (int idx = static_cast<int>(split_sizes.size()) - 1; idx >= 0; --idx) {
        const int64_t split = split_sizes[idx] <= 0 ? 1 : split_sizes[idx];
        result[idx] = static_cast<int>((item_id / divisor) % split);
        divisor *= split;
    }
    return result;
}

inline int64_t eval_bind_offset_expr(const std::string& expr) {
    std::string compact;
    compact.reserve(expr.size());
    for (char ch : expr) {
        if (!std::isspace(static_cast<unsigned char>(ch)) && ch != '(' && ch != ')') {
            compact.push_back(ch);
        }
    }

    if (compact.empty()) {
        return 0;
    }

    bool simple = true;
    for (char ch : compact) {
        if (!std::isdigit(static_cast<unsigned char>(ch)) && ch != '+' && ch != '-') {
            simple = false;
            break;
        }
    }
    if (!simple) {
        return 0;
    }

    int64_t value = 0;
    std::size_t idx = 0;
    while (idx < compact.size()) {
        int sign = 1;
        while (idx < compact.size() && (compact[idx] == '+' || compact[idx] == '-')) {
            if (compact[idx] == '-') {
                sign = -sign;
            }
            ++idx;
        }

        if (idx >= compact.size() || !std::isdigit(static_cast<unsigned char>(compact[idx]))) {
            continue;
        }

        int64_t number = 0;
        while (idx < compact.size() && std::isdigit(static_cast<unsigned char>(compact[idx]))) {
            number = number * 10 + (compact[idx] - '0');
            ++idx;
        }
        value += sign * number;
    }

    return value;
}

inline std::vector<int> init_partition_shape(const AccessPattern& pattern) {
    std::vector<int> shape = pattern.data_info.dimLength;
    for (int op_idx = 0; op_idx < pattern.param_ops.size; ++op_idx) {
        const Dac_Op& op = pattern.param_ops[op_idx];
        if (op.dimId < 0 || op.dimId >= static_cast<int>(shape.size())) {
            continue;
        }
        if (op_idx < static_cast<int>(pattern.is_index_op.size()) && pattern.is_index_op[op_idx]) {
            shape[op.dimId] = 1;
        } else {
            shape[op.dimId] = op.size;
        }
    }
    return shape;
}

inline std::vector<int64_t> init_bind_split_sizes(const AccessPattern& pattern) {
    int max_bind = -1;
    for (int bind_id : pattern.bind_set_id) {
        max_bind = std::max(max_bind, bind_id);
    }
    std::vector<int64_t> split_sizes(max_bind + 1, 1);
    for (int op_idx = 0; op_idx < pattern.param_ops.size; ++op_idx) {
        if (op_idx >= static_cast<int>(pattern.bind_set_id.size())) {
            continue;
        }
        const int bind_id = pattern.bind_set_id[op_idx];
        if (bind_id < 0) {
            continue;
        }
        if (split_sizes[bind_id] == 1) {
            split_sizes[bind_id] = pattern.param_ops[op_idx].split_size;
        }
    }
    return split_sizes;
}

inline int64_t partition_element_count(const AccessPattern& pattern) {
    const std::vector<int> shape =
        pattern.partition_shape.empty() ? init_partition_shape(pattern) : pattern.partition_shape;
    int64_t count = 1;
    for (int dim_len : shape) {
        count *= dim_len;
    }
    return count;
}

inline std::vector<int64_t> collect_positions_for_item(int64_t item_id,
                                                       const AccessPattern& pattern) {
    const std::vector<int> part_shape =
        pattern.partition_shape.empty() ? init_partition_shape(pattern) : pattern.partition_shape;
    const std::vector<int64_t> bind_splits =
        pattern.bind_split_sizes.empty() ? init_bind_split_sizes(pattern) : pattern.bind_split_sizes;
    const std::vector<int> bind_indices = decode_item_id(item_id, bind_splits);

    std::vector<int> base_pos(pattern.data_info.dim, 0);
    for (int op_idx = 0; op_idx < pattern.param_ops.size; ++op_idx) {
        const Dac_Op& op = pattern.param_ops[op_idx];
        if (op.dimId < 0 || op.dimId >= pattern.data_info.dim) {
            continue;
        }

        int bind_id = 0;
        if (op_idx < static_cast<int>(pattern.bind_set_id.size()) && pattern.bind_set_id[op_idx] >= 0) {
            bind_id = pattern.bind_set_id[op_idx];
        }

        int64_t bind_index = 0;
        if (bind_id < static_cast<int>(bind_indices.size())) {
            bind_index = bind_indices[bind_id];
        }

        int64_t offset = 0;
        if (op_idx < static_cast<int>(pattern.bind_offset_expr.size())) {
            offset = eval_bind_offset_expr(pattern.bind_offset_expr[op_idx]);
        }

        const bool is_index =
            op_idx < static_cast<int>(pattern.is_index_op.size()) && pattern.is_index_op[op_idx];
        if (is_index) {
            base_pos[op.dimId] = static_cast<int>(bind_index + offset);
        } else {
            base_pos[op.dimId] = static_cast<int>(bind_index * op.stride + offset);
        }
    }

    int64_t elem_count = 1;
    for (int dim_len : part_shape) {
        elem_count *= dim_len;
    }

    std::vector<int64_t> globals;
    globals.reserve(elem_count);

    for (int64_t linear_idx = 0; linear_idx < elem_count; ++linear_idx) {
        std::vector<int> global_pos = base_pos;
        int64_t carry = linear_idx;
        for (int dim = pattern.data_info.dim - 1; dim >= 0; --dim) {
            const int width = part_shape[dim];
            const int offset = static_cast<int>(carry % width);
            carry /= width;
            global_pos[dim] += offset;
        }
        globals.push_back(linearize(global_pos, pattern.data_info));
    }
    return globals;
}

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

inline PackMap make_pack_map_from_globals(std::vector<int64_t> globals) {
    std::sort(globals.begin(), globals.end());
    globals.erase(std::unique(globals.begin(), globals.end()), globals.end());

    PackMap pack;
    pack.globals = std::move(globals);
    pack.segments = build_segments(pack.globals);
    for (std::size_t idx = 0; idx < pack.globals.size(); ++idx) {
        pack.g2l.emplace(pack.globals[idx], static_cast<int32_t>(idx));
    }
    return pack;
}

inline PackMap build_input_pack_map(ItemRange range, const AccessPattern& pattern) {
    std::vector<int64_t> globals;
    for (int64_t item = range.begin; item < range.end; ++item) {
        const auto item_globals = collect_positions_for_item(item, pattern);
        globals.insert(globals.end(), item_globals.begin(), item_globals.end());
    }
    return make_pack_map_from_globals(std::move(globals));
}

inline PackMap build_output_pack_map(ItemRange range, const AccessPattern& pattern) {
    PackMap pack = build_input_pack_map(range, pattern);
    pack.writeback_globals = pack.globals;
    pack.writeback_segments = pack.segments;
    return pack;
}

inline PackMap build_rw_pack_map(ItemRange range, const AccessPattern& pattern) {
    PackMap pack = build_input_pack_map(range, pattern);
    pack.writeback_globals = pack.globals;
    pack.writeback_segments = pack.segments;
    return pack;
}

inline std::vector<int32_t> build_item_slots(ItemRange range,
                                             const AccessPattern& pattern,
                                             const PackMap& pack) {
    std::vector<int32_t> slots;
    const int64_t item_count = range.size();
    const int64_t elem_count = partition_element_count(pattern);
    slots.reserve(item_count * elem_count);

    for (int64_t item = range.begin; item < range.end; ++item) {
        const auto globals = collect_positions_for_item(item, pattern);
        for (int64_t global_idx : globals) {
            auto it = pack.g2l.find(global_idx);
            if (it == pack.g2l.end()) {
                throw std::runtime_error("build_item_slots: missing global to local mapping");
            }
            slots.push_back(it->second);
        }
    }
    return slots;
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
        auto it = pack.g2l.find(global_idx);
        if (it == pack.g2l.end()) {
            throw std::runtime_error("build_writeback_values: missing global to local mapping");
        }
        values.push_back(local_data[static_cast<std::size_t>(it->second)]);
    }
    return values;
}

template <typename T>
struct View1D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2DRow {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;
    int cols = 0;

    View2DRow<T> operator[](int row) const {
        return View2DRow<T>{data, slots, offset + row * cols};
    }
};

}  // namespace mpi
}  // namespace dacpp

#endif
