#ifndef DACPP_MPI_PATTERN_H
#define DACPP_MPI_PATTERN_H

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/CoreTypes.h"
#include "DataReconstructor.new.h"
#include "../common/Profile.h"
#include "dacInfo.h"

namespace dacpp {
namespace mpi {

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

struct SimpleStrideRange {
    bool valid = false;
    int64_t start = 0;
    int64_t stride = 0;
    int64_t count = 0;
};

inline SimpleStrideRange try_build_simple_stride_range(
    const std::vector<int>& base_pos,
    const AccessPattern& pattern,
    const std::vector<int>& part_shape) {
    SimpleStrideRange result;
    if (pattern.data_info.dim <= 0 ||
        static_cast<int>(base_pos.size()) != pattern.data_info.dim ||
        static_cast<int>(part_shape.size()) != pattern.data_info.dim) {
        return result;
    }

    int d_first = -1;
    int d_last = -1;
    int64_t count = 1;

    for (int dim = 0; dim < pattern.data_info.dim; ++dim) {
        if (part_shape[dim] > 1) {
            if (d_first < 0) {
                d_first = dim;
            }
            d_last = dim;
            count *= part_shape[dim];
        }
    }

    if (d_first < 0) {
        result.valid = true;
        result.start = linearize(base_pos, pattern.data_info);
        result.stride = 0;
        result.count = 1;
        return result;
    }

    bool contiguous = true;
    for (int dim = d_first + 1; dim <= d_last; ++dim) {
        if (part_shape[dim] != pattern.data_info.dimLength[dim]) {
            contiguous = false;
            break;
        }
    }

    if (!contiguous) {
        return result;
    }

    int64_t stride = 1;
    for (int dim = d_last + 1; dim < pattern.data_info.dim; ++dim) {
        stride *= pattern.data_info.dimLength[dim];
    }

    result.valid = true;
    result.start = linearize(base_pos, pattern.data_info);
    result.stride = stride;
    result.count = count;
    return result;
}

inline std::vector<int64_t> collect_positions_for_item(int64_t item_id,
                                                       const AccessPattern& pattern) {
    const auto profile_begin = profilingEnabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
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

    const SimpleStrideRange simple_range =
        try_build_simple_stride_range(base_pos, pattern, part_shape);
    if (simple_range.valid) {
        for (int64_t idx = 0; idx < simple_range.count; ++idx) {
            globals.push_back(simple_range.start + idx * simple_range.stride);
        }
        if (profilingEnabled()) {
            const auto profile_end = std::chrono::steady_clock::now();
            const long long nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                profile_end - profile_begin).count();
            recordCollectPositionsSample(nanos);
        }
        return globals;
    }

    std::vector<int64_t> element_strides(pattern.data_info.dim, 1);
    for (int dim = pattern.data_info.dim - 2; dim >= 0; --dim) {
        element_strides[dim] = element_strides[dim + 1] * pattern.data_info.dimLength[dim + 1];
    }

    int64_t current_linear = linearize(base_pos, pattern.data_info);
    std::vector<int> current_pos(pattern.data_info.dim, 0);

    for (int64_t idx = 0; idx < elem_count; ++idx) {
        globals.push_back(current_linear);

        for (int dim = pattern.data_info.dim - 1; dim >= 0; --dim) {
            if (++current_pos[dim] < part_shape[dim]) {
                current_linear += element_strides[dim];
                break;
            }
            current_pos[dim] = 0;
            current_linear -= element_strides[dim] * static_cast<int64_t>(part_shape[dim] - 1);
        }
    }
    if (profilingEnabled()) {
        const auto profile_end = std::chrono::steady_clock::now();
        const long long nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
            profile_end - profile_begin).count();
        recordCollectPositionsSample(nanos);
    }
    return globals;
}

}  // namespace mpi
}  // namespace dacpp

#endif
