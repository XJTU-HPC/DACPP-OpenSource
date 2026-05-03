#ifndef DACPP_MPI_REGION_VIEWS_H
#define DACPP_MPI_REGION_VIEWS_H

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace dacpp {
namespace mpi {

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

}  // namespace mpi
}  // namespace dacpp

#endif
