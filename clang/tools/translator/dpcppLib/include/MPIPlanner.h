#ifndef DACPP_MPI_PLANNER_H
#define DACPP_MPI_PLANNER_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mpi.h>

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

template <typename T>
class PackedElementRef {
public:
    using ValueType = std::remove_const_t<T>;

    PackedElementRef(T* data,
                     const int32_t* lookup,
                     std::size_t lookup_size,
                     int64_t global_index,
                     unsigned char* dirty = nullptr,
                     const ValueType* dense_fallback = nullptr,
                     ValueType* dense_shadow = nullptr)
        : data_(data),
          lookup_(lookup),
          lookup_size_(lookup_size),
          global_index_(global_index),
          dirty_(dirty),
          dense_fallback_(dense_fallback),
          dense_shadow_(dense_shadow) {}

    operator ValueType() const {
        const int32_t slot = lookup_slot();
        if (slot < 0 || !data_) {
            if (dense_shadow_ && global_index_ >= 0 &&
                static_cast<std::size_t>(global_index_) < lookup_size_) {
                return dense_shadow_[static_cast<std::size_t>(global_index_)];
            }
            if (dense_fallback_ && global_index_ >= 0 &&
                static_cast<std::size_t>(global_index_) < lookup_size_) {
                return dense_fallback_[static_cast<std::size_t>(global_index_)];
            }
            return ValueType{};
        }
        return data_[static_cast<std::size_t>(slot)];
    }

    PackedElementRef& operator=(const PackedElementRef& other) {
        if constexpr (std::is_const_v<T>) {
            return *this;
        } else {
            return assign_value(static_cast<ValueType>(other));
        }
    }

    template <typename U,
              typename V = T,
              typename = std::enable_if_t<!std::is_const_v<V>>>
    PackedElementRef& operator=(const U& value) {
        return assign_value(static_cast<ValueType>(value));
    }

private:
    PackedElementRef& assign_value(const ValueType& casted) {
        if (dense_shadow_ && global_index_ >= 0 &&
            static_cast<std::size_t>(global_index_) < lookup_size_) {
            dense_shadow_[static_cast<std::size_t>(global_index_)] = casted;
        }
        const int32_t slot = lookup_slot();
        if (slot >= 0 && data_) {
            data_[static_cast<std::size_t>(slot)] = casted;
        }
        mark_dirty();
        return *this;
    }
public:
    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator+=(const ValueType& value) {
        return *this = static_cast<ValueType>(*this) + value;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator-=(const ValueType& value) {
        return *this = static_cast<ValueType>(*this) - value;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator*=(const ValueType& value) {
        return *this = static_cast<ValueType>(*this) * value;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator/=(const ValueType& value) {
        return *this = static_cast<ValueType>(*this) / value;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator++() {
        *this += static_cast<ValueType>(1);
        return *this;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    ValueType operator++(int) {
        const ValueType old = static_cast<ValueType>(*this);
        ++(*this);
        return old;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    PackedElementRef& operator--() {
        *this -= static_cast<ValueType>(1);
        return *this;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    ValueType operator--(int) {
        const ValueType old = static_cast<ValueType>(*this);
        --(*this);
        return old;
    }

private:
    void mark_dirty() {
        if (!dirty_ || global_index_ < 0 ||
            static_cast<std::size_t>(global_index_) >= lookup_size_) {
            return;
        }
        dirty_[static_cast<std::size_t>(global_index_)] = 1;
    }

    int32_t lookup_slot() const {
        if (!lookup_ || global_index_ < 0 ||
            static_cast<std::size_t>(global_index_) >= lookup_size_) {
            return -1;
        }
        return lookup_[static_cast<std::size_t>(global_index_)];
    }

    T* data_ = nullptr;
    const int32_t* lookup_ = nullptr;
    std::size_t lookup_size_ = 0;
    int64_t global_index_ = -1;
    unsigned char* dirty_ = nullptr;
    const ValueType* dense_fallback_ = nullptr;
    ValueType* dense_shadow_ = nullptr;
};

template <typename T>
class PackedVectorView {
public:
    using ValueType = std::remove_const_t<T>;

    PackedVectorView(T* data,
                     const int32_t* lookup,
                     std::size_t lookup_size,
                     unsigned char* dirty = nullptr,
                     const ValueType* dense_fallback = nullptr,
                     ValueType* dense_shadow = nullptr)
        : data_(data),
          lookup_(lookup),
          lookup_size_(lookup_size),
          dirty_(dirty),
          dense_fallback_(dense_fallback),
          dense_shadow_(dense_shadow) {}

    PackedElementRef<T> operator[](int idx) {
        return PackedElementRef<T>(data_, lookup_, lookup_size_, idx, dirty_,
                                   dense_fallback_, dense_shadow_);
    }

    ValueType operator[](int idx) const {
        return static_cast<ValueType>(
            PackedElementRef<const ValueType>(data_, lookup_, lookup_size_,
                                              static_cast<int64_t>(idx),
                                              nullptr, dense_fallback_));
    }

private:
    T* data_ = nullptr;
    const int32_t* lookup_ = nullptr;
    std::size_t lookup_size_ = 0;
    unsigned char* dirty_ = nullptr;
    const ValueType* dense_fallback_ = nullptr;
    ValueType* dense_shadow_ = nullptr;
};

template <typename T>
class PackedMatrixRowView {
public:
    using ValueType = std::remove_const_t<T>;

    PackedMatrixRowView(T* data,
                        const int32_t* lookup,
                        std::size_t lookup_size,
                        int row,
                        int cols,
                        unsigned char* dirty = nullptr,
                        const ValueType* dense_fallback = nullptr,
                        ValueType* dense_shadow = nullptr)
        : data_(data),
          lookup_(lookup),
          lookup_size_(lookup_size),
          row_(row),
          cols_(cols),
          dirty_(dirty),
          dense_fallback_(dense_fallback),
          dense_shadow_(dense_shadow) {}

    PackedElementRef<T> operator[](int col) {
        const int64_t global_index =
            static_cast<int64_t>(row_) * static_cast<int64_t>(cols_) + col;
        return PackedElementRef<T>(data_, lookup_, lookup_size_, global_index,
                                   dirty_, dense_fallback_, dense_shadow_);
    }

    ValueType operator[](int col) const {
        const int64_t global_index =
            static_cast<int64_t>(row_) * static_cast<int64_t>(cols_) + col;
        return static_cast<ValueType>(PackedElementRef<const ValueType>(
            data_, lookup_, lookup_size_, global_index, nullptr,
            dense_fallback_));
    }

private:
    T* data_ = nullptr;
    const int32_t* lookup_ = nullptr;
    std::size_t lookup_size_ = 0;
    int row_ = 0;
    int cols_ = 0;
    unsigned char* dirty_ = nullptr;
    const ValueType* dense_fallback_ = nullptr;
    ValueType* dense_shadow_ = nullptr;
};

template <typename T>
class PackedMatrixView {
public:
    PackedMatrixView(T* data,
                     const int32_t* lookup,
                     std::size_t lookup_size,
                     int cols,
                     unsigned char* dirty = nullptr,
                     const std::remove_const_t<T>* dense_fallback = nullptr,
                     std::remove_const_t<T>* dense_shadow = nullptr)
        : data_(data),
          lookup_(lookup),
          lookup_size_(lookup_size),
          cols_(cols),
          dirty_(dirty),
          dense_fallback_(dense_fallback),
          dense_shadow_(dense_shadow) {}

    PackedMatrixRowView<T> operator[](int row) {
        return PackedMatrixRowView<T>(data_, lookup_, lookup_size_, row,
                                      cols_, dirty_, dense_fallback_,
                                      dense_shadow_);
    }

    const PackedMatrixRowView<const std::remove_const_t<T>> operator[](
        int row) const {
        return PackedMatrixRowView<const std::remove_const_t<T>>(
            data_, lookup_, lookup_size_, row, cols_, nullptr,
            dense_fallback_);
    }

private:
    T* data_ = nullptr;
    const int32_t* lookup_ = nullptr;
    std::size_t lookup_size_ = 0;
    int cols_ = 0;
    unsigned char* dirty_ = nullptr;
    const std::remove_const_t<T>* dense_fallback_ = nullptr;
    std::remove_const_t<T>* dense_shadow_ = nullptr;
};

// Dense host-side views used by MPI region sibling-loop lowering.  The region
// path stores rank-local packed arrays; sibling loops are written against the
// original dense tensor indexing syntax, so these lightweight views provide
// Vector-style and Matrix-style operator[] access over a dense temporary.
template <typename T>
class DenseElementRef {
public:
    DenseElementRef(std::vector<T>& data,
                    std::size_t index,
                    std::vector<unsigned char>* dirty = nullptr)
        : data_(data), index_(index), dirty_(dirty) {}

    operator T() const {
        return data_[index_];
    }

    DenseElementRef& operator=(const DenseElementRef& other) {
        return *this = static_cast<T>(other);
    }

    template <typename U>
    DenseElementRef& operator=(const U& value) {
        data_[index_] = static_cast<T>(value);
        mark_dirty();
        return *this;
    }

    template <typename U>
    DenseElementRef& operator+=(const U& value) {
        return *this = static_cast<T>(*this) + value;
    }

    template <typename U>
    DenseElementRef& operator-=(const U& value) {
        return *this = static_cast<T>(*this) - value;
    }

    template <typename U>
    DenseElementRef& operator*=(const U& value) {
        return *this = static_cast<T>(*this) * value;
    }

    template <typename U>
    DenseElementRef& operator/=(const U& value) {
        return *this = static_cast<T>(*this) / value;
    }

    DenseElementRef& operator++() {
        *this += 1;
        return *this;
    }

    T operator++(int) {
        T old = static_cast<T>(*this);
        ++(*this);
        return old;
    }

    DenseElementRef& operator--() {
        *this -= 1;
        return *this;
    }

    T operator--(int) {
        T old = static_cast<T>(*this);
        --(*this);
        return old;
    }

private:
    void mark_dirty() {
        if (dirty_ && index_ < dirty_->size()) {
            (*dirty_)[index_] = 1;
        }
    }

    std::vector<T>& data_;
    std::size_t index_ = 0;
    std::vector<unsigned char>* dirty_ = nullptr;
};

template <typename T>
class DenseVectorView {
public:
    DenseVectorView(std::vector<T>& data,
                    std::vector<unsigned char>* dirty = nullptr)
        : data_(data), dirty_(dirty) {}

    DenseVectorView(std::vector<T>& data,
                    const std::vector<int>& shape,
                    std::vector<unsigned char>* dirty = nullptr)
        : data_(data), dirty_(dirty) {
        (void)shape;
    }

    DenseElementRef<T> operator[](int idx) {
        return DenseElementRef<T>(
            data_, static_cast<std::size_t>(idx), dirty_);
    }

    T operator[](int idx) const {
        return data_[static_cast<std::size_t>(idx)];
    }

private:
    std::vector<T>& data_;
    std::vector<unsigned char>* dirty_ = nullptr;
};

template <typename T>
class DenseMatrixRowView {
public:
    DenseMatrixRowView(std::vector<T>& data,
                       std::size_t row_offset,
                       int cols,
                       std::vector<unsigned char>* dirty = nullptr)
        : data_(data), row_offset_(row_offset), cols_(cols), dirty_(dirty) {}

    DenseElementRef<T> operator[](int col) {
        return DenseElementRef<T>(
            data_, row_offset_ + static_cast<std::size_t>(col), dirty_);
    }

    T operator[](int col) const {
        return data_[row_offset_ + static_cast<std::size_t>(col)];
    }

private:
    std::vector<T>& data_;
    std::size_t row_offset_ = 0;
    int cols_ = 0;
    std::vector<unsigned char>* dirty_ = nullptr;
};

template <typename T>
class DenseMatrixView {
public:
    DenseMatrixView(std::vector<T>& data,
                    const std::vector<int>& shape,
                    std::vector<unsigned char>* dirty = nullptr)
        : data_(data), shape_(shape), dirty_(dirty) {}

    DenseMatrixRowView<T> operator[](int row) {
        return DenseMatrixRowView<T>(
            data_,
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols()),
            cols(),
            dirty_);
    }

    const DenseMatrixRowView<T> operator[](int row) const {
        return DenseMatrixRowView<T>(
            const_cast<std::vector<T>&>(data_),
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols()),
            cols(),
            dirty_);
    }

private:
    int cols() const {
        return shape_.size() > 1 ? shape_[1] : 1;
    }

    std::vector<T>& data_;
    const std::vector<int>& shape_;
    std::vector<unsigned char>* dirty_ = nullptr;
};

// ---------------------------------------------------------------------------
// Halo exchange support for MPI stencil optimization
// ---------------------------------------------------------------------------

/// Information for a single neighbor direction, scoped to one parameter.
struct HaloRegion {
    int neighbor_rank = -1;
    /// Global positions I need to receive from this neighbor.
    std::vector<int64_t> recv_globals;
    /// Corresponding slots in the local array (via g2l).
    std::vector<int32_t> recv_local_slots;
    /// Global positions I need to send to this neighbor.
    std::vector<int64_t> send_globals;
    /// Corresponding slots in the local array.
    std::vector<int32_t> send_local_slots;
};

/// Complete halo information for one parameter (up to 2 neighbors).
struct ParamHalo {
    std::vector<HaloRegion> regions;
};

/// Compute halo regions for a single parameter on the current rank.
/// The caller provides the parameter's AccessPattern, its effective mode,
/// and the current rank's item range.
///
/// Algorithm:
///   For each neighbor (rank-1, rank+1):
///     recv = my_input_globals ∩ neighbor_output_globals
///     send = my_output_globals ∩ neighbor_input_globals
inline ParamHalo computeParamHalo(
    const AccessPattern& pattern,
    AccessMode mode,
    ItemRange my_range,
    int64_t total_items,
    int mpi_rank,
    int mpi_size)
{
    ParamHalo halo;

    // Build my own input/output position sets.
    // For "output" we use the same collect_positions_for_item — the output
    // positions of an item are the positions its partition covers.
    std::vector<int64_t> my_input_globals;
    std::vector<int64_t> my_output_globals;
    {
        std::unordered_set<int64_t> seen_in, seen_out;
        for (int64_t item = my_range.begin; item < my_range.end; ++item) {
            auto positions = collect_positions_for_item(item, pattern);
            for (auto g : positions) {
                if (seen_in.insert(g).second) {
                    my_input_globals.push_back(g);
                }
                // For output, all partition positions are potential outputs.
                // A more precise analysis could restrict to write-mode dims,
                // but using the full partition is safe for correctness.
                if (seen_out.insert(g).second) {
                    my_output_globals.push_back(g);
                }
            }
        }
        std::sort(my_input_globals.begin(), my_input_globals.end());
        std::sort(my_output_globals.begin(), my_output_globals.end());
    }

    for (int delta : {-1, 1}) {
        const int neighbor = mpi_rank + delta;
        if (neighbor < 0 || neighbor >= mpi_size) continue;

        HaloRegion region;
        region.neighbor_rank = neighbor;

        ItemRange nb_range = get_rank_item_range(total_items, neighbor, mpi_size);

        // Collect neighbor's input and output positions.
        std::vector<int64_t> nb_input_globals;
        std::vector<int64_t> nb_output_globals;
        {
            std::unordered_set<int64_t> seen_in, seen_out;
            for (int64_t item = nb_range.begin; item < nb_range.end; ++item) {
                auto positions = collect_positions_for_item(item, pattern);
                for (auto g : positions) {
                    if (seen_in.insert(g).second) {
                        nb_input_globals.push_back(g);
                    }
                    if (seen_out.insert(g).second) {
                        nb_output_globals.push_back(g);
                    }
                }
            }
            std::sort(nb_input_globals.begin(), nb_input_globals.end());
            std::sort(nb_output_globals.begin(), nb_output_globals.end());
        }

        // Build the pack map's g2l for local slot lookup.
        // We need a unified g2l that covers all positions in my input pack.
        PackMap my_pack = make_pack_map_from_globals(my_input_globals);

        // Recv: positions I read that the neighbor produces (writes).
        // recv = my_input_globals ∩ nb_output_globals
        if (mode != AccessMode::Write) {
            std::vector<int64_t> recv_globals;
            std::set_intersection(
                my_input_globals.begin(), my_input_globals.end(),
                nb_output_globals.begin(), nb_output_globals.end(),
                std::back_inserter(recv_globals));
            for (auto g : recv_globals) {
                auto it = my_pack.g2l.find(g);
                if (it != my_pack.g2l.end()) {
                    region.recv_globals.push_back(g);
                    region.recv_local_slots.push_back(it->second);
                }
            }
        }

        // Send: positions I produce that the neighbor reads.
        // send = my_output_globals ∩ nb_input_globals
        if (mode != AccessMode::Read) {
            std::vector<int64_t> send_globals;
            std::set_intersection(
                my_output_globals.begin(), my_output_globals.end(),
                nb_input_globals.begin(), nb_input_globals.end(),
                std::back_inserter(send_globals));
            for (auto g : send_globals) {
                auto it = my_pack.g2l.find(g);
                if (it != my_pack.g2l.end()) {
                    region.send_globals.push_back(g);
                    region.send_local_slots.push_back(it->second);
                }
            }
        }

        if (!region.recv_globals.empty() || !region.send_globals.empty()) {
            halo.regions.push_back(std::move(region));
        }
    }

    return halo;
}

/// Perform halo exchange for a single parameter, iterating over neighbors.
/// Uses non-blocking send/recv to avoid deadlock.
template <typename T>
inline void exchangeHalo(std::vector<T>& local_data,
                         const ParamHalo& halo,
                         void* mpi_type_ptr) {
    // mpi_type_ptr is a pointer to the MPI_Datatype; we cast it back.
    MPI_Datatype mpi_type = *static_cast<MPI_Datatype*>(mpi_type_ptr);

    for (const auto& region : halo.regions) {
        if (region.neighbor_rank < 0) continue;

        // Pack send buffer.
        std::vector<T> send_buf;
        send_buf.reserve(region.send_local_slots.size());
        for (int32_t slot : region.send_local_slots) {
            send_buf.push_back(local_data[static_cast<std::size_t>(slot)]);
        }

        // Prepare receive buffer.
        std::vector<T> recv_buf(region.recv_local_slots.size());

        MPI_Request reqs[2];
        int req_count = 0;

        if (!send_buf.empty()) {
            MPI_Isend(send_buf.data(),
                      static_cast<int>(send_buf.size()),
                      mpi_type,
                      region.neighbor_rank,
                      /*tag=*/0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (!recv_buf.empty()) {
            MPI_Irecv(recv_buf.data(),
                      static_cast<int>(recv_buf.size()),
                      mpi_type,
                      region.neighbor_rank,
                      /*tag=*/0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Unpack receive buffer into local data.
        for (std::size_t i = 0; i < region.recv_local_slots.size(); ++i) {
            local_data[static_cast<std::size_t>(region.recv_local_slots[i])] = recv_buf[i];
        }
    }
}

}  // namespace mpi
}  // namespace dacpp

#endif
