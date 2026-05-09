#ifndef DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H
#define DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include <mpi.h>

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

inline int nearest_nonempty_rank_1d(int64_t total,
                                    int rank,
                                    int size,
                                    int direction) {
    if (direction == 0) {
        return MPI_PROC_NULL;
    }
    for (int r = rank + direction; r >= 0 && r < size; r += direction) {
        if (rank_range_1d(total, r, size).count > 0) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

struct ResidentHalo1DLayout {
    RankRange1D owned{};
    int left_halo = 0;
    int right_halo = 0;
    int64_t local_size = 0;
    int64_t owned_offset = 0;
    int64_t global_begin = 0;
};

inline ResidentHalo1DLayout resident_halo_1d_layout(
    int64_t outputTotal,
    int rank,
    int size,
    int windowSize) {
    ResidentHalo1DLayout layout;
    layout.owned = rank_range_1d(outputTotal, rank, size);
    layout.left_halo = 0;
    layout.right_halo =
        layout.owned.count > 0 ? std::max(0, windowSize - 1) : 0;
    layout.owned_offset = 0;
    layout.local_size = layout.owned.count + layout.right_halo;
    layout.global_begin = layout.owned.begin;
    return layout;
}

template <typename T>
void scatter_window_1d(const std::vector<T>& global,
                       std::vector<T>& local,
                       int64_t outputTotal,
                       int windowSize,
                       const ResidentHalo1DLayout& layout,
                       int rank,
                       int size,
                       MPI_Datatype mpiType) {
    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo1DLayout target =
                resident_halo_1d_layout(outputTotal, r, size, windowSize);
            if (target.local_size <= 0) {
                continue;
            }
            const std::size_t begin =
                static_cast<std::size_t>(target.global_begin);
            const std::size_t count =
                static_cast<std::size_t>(target.local_size);
            if (r == 0) {
                std::copy(global.begin() + begin,
                          global.begin() + begin + count,
                          local.begin());
            } else {
                MPI_Send(global.data() + begin,
                         static_cast<int>(target.local_size),
                         mpiType,
                         r,
                         4201,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        MPI_Recv(local.data(),
                 static_cast<int>(layout.local_size),
                 mpiType,
                 0,
                 4201,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

template <typename T>
void apply_followup_1d(std::vector<T>& readerLocal,
                       const std::vector<T>& writerLocal,
                       int followupTargetOffset) {
    for (std::size_t idx = 0; idx < writerLocal.size(); ++idx) {
        const int64_t target =
            static_cast<int64_t>(idx) + followupTargetOffset;
        if (target >= 0 &&
            target < static_cast<int64_t>(readerLocal.size())) {
            readerLocal[static_cast<std::size_t>(target)] = writerLocal[idx];
        }
    }
}

template <typename T>
void exchange_halo_1d(std::vector<T>& local,
                      const std::vector<T>& writerLocal,
                      const ResidentHalo1DLayout& layout,
                      int64_t outputTotal,
                      int windowSize,
                      int followupTargetOffset,
                      int rank,
                      int size,
                      MPI_Datatype mpiType) {
    if (followupTargetOffset != 1 || layout.owned.count <= 0) {
        return;
    }
    const int prev = nearest_nonempty_rank_1d(outputTotal, rank, size, -1);
    const int next = nearest_nonempty_rank_1d(outputTotal, rank, size, 1);
    T recvLeft{};
    T sendRight = writerLocal.empty() ? T{} : writerLocal.back();
    MPI_Sendrecv(&sendRight, 1, mpiType, next, 4101,
                 &recvLeft, 1, mpiType, prev, 4101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (prev != MPI_PROC_NULL && !local.empty()) {
        local[0] = recvLeft;
    }
    if (windowSize > 2) {
        T recvRight{};
        T sendLeft = writerLocal.empty() ? T{} : writerLocal.front();
        MPI_Sendrecv(&sendLeft, 1, mpiType, prev, 4102,
                     &recvRight, 1, mpiType, next, 4102,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        const int64_t rightIdx = layout.owned.count + windowSize - 2;
        if (next != MPI_PROC_NULL && rightIdx >= 0 &&
            rightIdx < static_cast<int64_t>(local.size())) {
            local[static_cast<std::size_t>(rightIdx)] = recvRight;
        }
    }
}

template <typename T>
std::vector<T> owned_slice_1d(const std::vector<T>& local,
                              const ResidentHalo1DLayout& layout) {
    const auto begin =
        local.begin() + static_cast<std::ptrdiff_t>(layout.owned_offset);
    return std::vector<T>(
        begin, begin + static_cast<std::ptrdiff_t>(layout.owned.count));
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
