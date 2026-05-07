#ifndef DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H
#define DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace dacpp {
namespace mpi {
namespace operator_resident {

struct RankRange1D {
    int64_t begin = 0;
    int64_t count = 0;
};

inline RankRange1D rank_range_1d(int64_t total, int rank, int size) {
    const int64_t safeSize = std::max(1, size);
    const int64_t base = total / safeSize;
    const int64_t rem = total % safeSize;
    const int64_t count = base + (rank < rem ? 1 : 0);
    const int64_t begin = rank * base + std::min<int64_t>(rank, rem);
    return {begin, count};
}

inline void counts_displs_1d(int64_t total,
                             int size,
                             std::vector<int>& counts,
                             std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    int64_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const RankRange1D range = rank_range_1d(total, r, size);
        counts[r] = static_cast<int>(range.count);
        displs[r] = static_cast<int>(offset);
        offset += range.count;
    }
}

inline void byte_counts_displs(const std::vector<int>& elemCounts,
                               const std::vector<int>& elemDispls,
                               std::size_t elemSize,
                               std::vector<int>& byteCounts,
                               std::vector<int>& byteDispls) {
    byteCounts.resize(elemCounts.size());
    byteDispls.resize(elemDispls.size());
    for (std::size_t idx = 0; idx < elemCounts.size(); ++idx) {
        byteCounts[idx] = static_cast<int>(elemCounts[idx] * elemSize);
        byteDispls[idx] = static_cast<int>(elemDispls[idx] * elemSize);
    }
}

struct ResidencyKey {
    const void* tensor = nullptr;
    std::type_index type = std::type_index(typeid(void));

    bool operator==(const ResidencyKey& other) const {
        return tensor == other.tensor && type == other.type;
    }
};

struct ResidencyKeyHash {
    std::size_t operator()(const ResidencyKey& key) const {
        return std::hash<const void*>{}(key.tensor) ^
               (key.type.hash_code() + 0x9e3779b9);
    }
};

template <typename T>
std::unordered_map<ResidencyKey, std::vector<T>, ResidencyKeyHash>&
typed_resident_storage() {
    static std::unordered_map<ResidencyKey, std::vector<T>, ResidencyKeyHash>
        storage;
    return storage;
}

template <typename T, typename TensorT>
std::vector<T>* find_resident(const TensorT& tensor) {
    ResidencyKey key{static_cast<const void*>(&tensor), std::type_index(typeid(T))};
    auto& storage = typed_resident_storage<T>();
    auto it = storage.find(key);
    return it == storage.end() ? nullptr : &it->second;
}

template <typename T, typename TensorT>
std::vector<T>& ensure_resident(const TensorT& tensor, std::size_t size) {
    ResidencyKey key{static_cast<const void*>(&tensor), std::type_index(typeid(T))};
    auto& buffer = typed_resident_storage<T>()[key];
    buffer.resize(size);
    return buffer;
}

} // namespace operator_resident
} // namespace mpi
} // namespace dacpp

#endif
