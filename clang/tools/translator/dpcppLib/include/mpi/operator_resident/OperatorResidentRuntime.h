#ifndef DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H
#define DACPP_MPI_OPERATOR_RESIDENT_RUNTIME_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

[[noreturn]] inline void abort_mpi_count_overflow(const char* what,
                                                  int64_t value) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::fprintf(stderr,
                     "[DACPP][MPI][OR] %s exceeds supported MPI int range: %lld\n",
                     what ? what : "MPI count/displacement",
                     static_cast<long long>(value));
    }
    MPI_Abort(MPI_COMM_WORLD, 5);
    std::abort();
}

inline int64_t checked_mul_int64_or_abort(int64_t lhs,
                                          int64_t rhs,
                                          const char* what) {
    if (lhs < 0 || rhs < 0) {
        abort_mpi_count_overflow(what, lhs < 0 ? lhs : rhs);
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        abort_mpi_count_overflow(what, std::numeric_limits<int64_t>::max());
    }
    return lhs * rhs;
}

inline int narrow_mpi_count_or_abort(int64_t value, const char* what) {
    if (value < 0 ||
        value > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        abort_mpi_count_overflow(what, value);
    }
    return static_cast<int>(value);
}

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
        counts[r] = narrow_mpi_count_or_abort(
            range.count,
            "[DACPP][MPI][OR] shape-derived count exceeds MPI int range");
        displs[r] = narrow_mpi_count_or_abort(
            offset,
            "[DACPP][MPI][OR] shape-derived displacement exceeds MPI int range");
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

struct ResidentHalo2DRowLayout {
    RankRange1D owned_rows{};
    int64_t input_cols = 0;
    int64_t local_row_count = 0;
    int64_t local_size = 0;
    int64_t global_row_begin = 0;
};

inline ResidentHalo2DRowLayout resident_halo_2d_row_layout(
    int64_t outputRows,
    int64_t inputCols,
    int rank,
    int size,
    int windowRows) {
    ResidentHalo2DRowLayout layout;
    layout.owned_rows = rank_range_1d(outputRows, rank, size);
    layout.input_cols = inputCols;
    layout.local_row_count =
        layout.owned_rows.count > 0
            ? layout.owned_rows.count + std::max<int64_t>(0, windowRows - 1)
            : 0;
    layout.local_size = checked_mul_int64_or_abort(
        layout.local_row_count,
        inputCols,
        "[DACPP][MPI][OR] resident halo 2D local size exceeds MPI int range");
    layout.global_row_begin = layout.owned_rows.begin;
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
                const int sendCount = narrow_mpi_count_or_abort(
                    target.local_size,
                    "[DACPP][MPI][OR] resident halo 1D scatter count exceeds MPI int range");
                MPI_Send(global.data() + begin,
                         sendCount,
                         mpiType,
                         r,
                         4201,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] resident halo 1D recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4201,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

template <typename T>
void scatter_window_2d_rows(const std::vector<T>& global,
                            std::vector<T>& local,
                            int64_t outputRows,
                            int64_t inputCols,
                            int windowRows,
                            const ResidentHalo2DRowLayout& layout,
                            int rank,
                            int size,
                            MPI_Datatype mpiType) {
    (void)windowRows;
    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo2DRowLayout target =
                resident_halo_2d_row_layout(outputRows, inputCols, r, size,
                                            windowRows);
            if (target.local_size <= 0) {
                continue;
            }
            const std::size_t begin =
                static_cast<std::size_t>(target.global_row_begin * inputCols);
            const std::size_t count =
                static_cast<std::size_t>(target.local_size);
            if (r == 0) {
                std::copy(global.begin() + begin,
                          global.begin() + begin + count,
                          local.begin());
            } else {
                const int sendCount = narrow_mpi_count_or_abort(
                    target.local_size,
                    "[DACPP][MPI][OR] resident halo 2D scatter count exceeds MPI int range");
                MPI_Send(global.data() + begin,
                         sendCount,
                         mpiType,
                         r,
                         4301,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] resident halo 2D recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4301,
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
void apply_followup_2d(std::vector<T>& readerLocal,
                       const std::vector<T>& writerLocal,
                       int64_t localOutputRows,
                       int64_t outputCols,
                       int64_t inputCols,
                       int followupTargetRowOffset,
                       int followupTargetColOffset) {
    for (int64_t row = 0; row < localOutputRows; ++row) {
        for (int64_t col = 0; col < outputCols; ++col) {
            const int64_t targetRow = row + followupTargetRowOffset;
            const int64_t targetCol = col + followupTargetColOffset;
            if (targetRow < 0 || targetCol < 0 || targetCol >= inputCols) {
                continue;
            }
            const std::size_t writerIdx = static_cast<std::size_t>(
                row * outputCols + col);
            const std::size_t targetIdx = static_cast<std::size_t>(
                targetRow * inputCols + targetCol);
            if (writerIdx < writerLocal.size() && targetIdx < readerLocal.size()) {
                readerLocal[targetIdx] = writerLocal[writerIdx];
            }
        }
    }
}

template <typename T>
void apply_read_cache_transition_2d(std::vector<T>& directReaderLocal,
                                    const std::vector<T>& readerLocal,
                                    int64_t localOutputRows,
                                    int64_t outputCols,
                                    int64_t inputCols,
                                    int targetRowOffset,
                                    int targetColOffset) {
    if (localOutputRows <= 0 || outputCols <= 0 ||
        targetRowOffset != -1 || targetColOffset != -1) {
        return;
    }
    for (int64_t row = 0; row < localOutputRows; ++row) {
        const int64_t sourceRow = row - targetRowOffset;
        if (sourceRow < 0) {
            continue;
        }
        for (int64_t col = 0; col < outputCols; ++col) {
            const int64_t sourceCol = col - targetColOffset;
            if (sourceCol < 0 || sourceCol >= inputCols) {
                continue;
            }
            const std::size_t directIdx = static_cast<std::size_t>(
                row * outputCols + col);
            const std::size_t sourceIdx = static_cast<std::size_t>(
                sourceRow * inputCols + sourceCol);
            if (directIdx < directReaderLocal.size() &&
                sourceIdx < readerLocal.size()) {
                directReaderLocal[directIdx] = readerLocal[sourceIdx];
            }
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
void exchange_halo_1d_inplace(std::vector<T>& local,
                              const ResidentHalo1DLayout& layout,
                              int64_t outputTotal,
                              int windowSize,
                              int interiorOffset,
                              int rank,
                              int size,
                              MPI_Datatype mpiType) {
    if (interiorOffset != 1 || layout.owned.count <= 0 || local.empty()) {
        return;
    }
    const int prev = nearest_nonempty_rank_1d(outputTotal, rank, size, -1);
    const int next = nearest_nonempty_rank_1d(outputTotal, rank, size, 1);
    const std::ptrdiff_t firstInterior =
        static_cast<std::ptrdiff_t>(interiorOffset);
    const std::ptrdiff_t lastInterior = static_cast<std::ptrdiff_t>(
        interiorOffset + layout.owned.count - 1);
    T recvLeft{};
    MPI_Sendrecv(local.data() + lastInterior,
                 1,
                 mpiType,
                 next,
                 4101,
                 &recvLeft,
                 1,
                 mpiType,
                 prev,
                 4101,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    if (prev != MPI_PROC_NULL && firstInterior > 0) {
        local[static_cast<std::size_t>(firstInterior - 1)] = recvLeft;
    }
    if (windowSize > 2) {
        T recvRight{};
        MPI_Sendrecv(local.data() + firstInterior,
                     1,
                     mpiType,
                     prev,
                     4102,
                     &recvRight,
                     1,
                     mpiType,
                     next,
                     4102,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        const int64_t rightIdx = interiorOffset + layout.owned.count;
        if (next != MPI_PROC_NULL && rightIdx >= 0 &&
            rightIdx < static_cast<int64_t>(local.size())) {
            local[static_cast<std::size_t>(rightIdx)] = recvRight;
        }
    }
}

template <typename T>
void exchange_halo_2d_rows(std::vector<T>& local,
                           const std::vector<T>& writerLocal,
                           const ResidentHalo2DRowLayout& layout,
                           int64_t outputRows,
                           int64_t outputCols,
                           int64_t inputCols,
                           int followupTargetRowOffset,
                           int followupTargetColOffset,
                           int rank,
                           int size,
                           MPI_Datatype mpiType) {
    if (layout.owned_rows.count <= 0 || outputCols <= 0 ||
        followupTargetRowOffset != 1 || followupTargetColOffset != 1 ||
        local.empty() || writerLocal.empty()) {
        return;
    }
    const int prev = nearest_nonempty_rank_1d(outputRows, rank, size, -1);
    const int next = nearest_nonempty_rank_1d(outputRows, rank, size, 1);
    const int sendCount = narrow_mpi_count_or_abort(
        outputCols,
        "[DACPP][MPI][OR] resident halo 2D halo width exceeds MPI int range");
    const std::ptrdiff_t topRecvOffset = static_cast<std::ptrdiff_t>(
        followupTargetColOffset);
    const std::ptrdiff_t bottomRecvOffset =
        static_cast<std::ptrdiff_t>(checked_mul_int64_or_abort(
                                        layout.local_row_count - 1,
                                        inputCols,
                                        "[DACPP][MPI][OR] resident halo 2D halo offset overflow") +
                                    followupTargetColOffset);
    MPI_Sendrecv(writerLocal.data() +
                     static_cast<std::ptrdiff_t>((layout.owned_rows.count - 1) *
                                                 outputCols),
                 sendCount,
                 mpiType,
                 next,
                 4203,
                 local.data() + topRecvOffset,
                 sendCount,
                 mpiType,
                 prev,
                 4203,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(writerLocal.data(),
                 sendCount,
                 mpiType,
                 prev,
                 4204,
                 local.data() + bottomRecvOffset,
                 sendCount,
                 mpiType,
                 next,
                 4204,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

template <typename T>
void exchange_halo_2d_rows_inplace(std::vector<T>& local,
                                   const ResidentHalo2DRowLayout& layout,
                                   int64_t outputRows,
                                   int64_t outputCols,
                                   int64_t inputCols,
                                   int interiorRowOffset,
                                   int interiorColOffset,
                                   int rank,
                                   int size,
                                   MPI_Datatype mpiType) {
    if (layout.owned_rows.count <= 0 || outputCols <= 0 ||
        interiorRowOffset != 1 || interiorColOffset != 1 || local.empty()) {
        return;
    }
    const int prev = nearest_nonempty_rank_1d(outputRows, rank, size, -1);
    const int next = nearest_nonempty_rank_1d(outputRows, rank, size, 1);
    const int sendCount = narrow_mpi_count_or_abort(
        outputCols,
        "[DACPP][MPI][OR] resident halo 2D halo width exceeds MPI int range");
    const std::ptrdiff_t topSendOffset = static_cast<std::ptrdiff_t>(
        checked_mul_int64_or_abort(
            interiorRowOffset,
            inputCols,
            "[DACPP][MPI][OR] resident halo 2D top send offset overflow") +
        interiorColOffset);
    const std::ptrdiff_t bottomSendOffset = static_cast<std::ptrdiff_t>(
        checked_mul_int64_or_abort(
            interiorRowOffset + layout.owned_rows.count - 1,
            inputCols,
            "[DACPP][MPI][OR] resident halo 2D bottom send offset overflow") +
        interiorColOffset);
    const std::ptrdiff_t topRecvOffset =
        static_cast<std::ptrdiff_t>(interiorColOffset);
    const std::ptrdiff_t bottomRecvOffset = static_cast<std::ptrdiff_t>(
        checked_mul_int64_or_abort(
            layout.local_row_count - 1,
            inputCols,
            "[DACPP][MPI][OR] resident halo 2D halo offset overflow") +
        interiorColOffset);
    MPI_Sendrecv(local.data() + bottomSendOffset,
                 sendCount,
                 mpiType,
                 next,
                 4203,
                 local.data() + topRecvOffset,
                 sendCount,
                 mpiType,
                 prev,
                 4203,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(local.data() + topSendOffset,
                 sendCount,
                 mpiType,
                 prev,
                 4204,
                 local.data() + bottomRecvOffset,
                 sendCount,
                 mpiType,
                 next,
                 4204,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

template <typename T>
std::vector<T> owned_slice_1d(const std::vector<T>& local,
                              const ResidentHalo1DLayout& layout,
                              int64_t sourceOffset = 0) {
    const int64_t beginIndex = layout.owned_offset + sourceOffset;
    const int64_t endIndex = beginIndex + layout.owned.count;
    if (beginIndex < 0 || endIndex < beginIndex ||
        endIndex > static_cast<int64_t>(local.size())) {
        return {};
    }
    const auto begin = local.begin() + static_cast<std::ptrdiff_t>(beginIndex);
    return std::vector<T>(
        begin, begin + static_cast<std::ptrdiff_t>(layout.owned.count));
}

template <typename T>
void write_owned_slice_2d_rows(std::vector<T>& local,
                               const std::vector<T>& owned,
                               const ResidentHalo2DRowLayout& layout,
                               int64_t outputCols,
                               int64_t inputCols,
                               int targetRowOffset,
                               int targetColOffset) {
    if (layout.owned_rows.count <= 0 || outputCols <= 0 ||
        inputCols <= 0 || local.empty() || owned.empty()) {
        return;
    }
    for (int64_t row = 0; row < layout.owned_rows.count; ++row) {
        const int64_t targetRow = row + targetRowOffset;
        if (targetRow < 0 || targetRow >= layout.local_row_count) {
            continue;
        }
        const int64_t sourceBase = checked_mul_int64_or_abort(
            row,
            outputCols,
            "[DACPP][MPI][OR] resident halo 2D owned source offset overflow");
        const int64_t targetBase = checked_mul_int64_or_abort(
            targetRow,
            inputCols,
            "[DACPP][MPI][OR] resident halo 2D owned target offset overflow");
        for (int64_t col = 0; col < outputCols; ++col) {
            const int64_t targetCol = col + targetColOffset;
            if (targetCol < 0 || targetCol >= inputCols) {
                continue;
            }
            const std::size_t sourceIdx =
                static_cast<std::size_t>(sourceBase + col);
            const std::size_t targetIdx =
                static_cast<std::size_t>(targetBase + targetCol);
            if (sourceIdx < owned.size() && targetIdx < local.size()) {
                local[targetIdx] = owned[sourceIdx];
            }
        }
    }
}

template <typename T>
std::vector<T> owned_slice_2d_rows(const std::vector<T>& local,
                                   const ResidentHalo2DRowLayout& layout,
                                   int64_t outputCols,
                                   int64_t inputCols,
                                   int sourceRowOffset,
                                   int sourceColOffset) {
    const int64_t ownedItemCount = checked_mul_int64_or_abort(
        layout.owned_rows.count,
        outputCols,
        "[DACPP][MPI][OR] resident halo 2D owned slice size overflow");
    std::vector<T> owned(static_cast<std::size_t>(ownedItemCount), T{});
    if (layout.owned_rows.count <= 0 || outputCols <= 0 ||
        inputCols <= 0 || local.empty()) {
        return owned;
    }
    for (int64_t row = 0; row < layout.owned_rows.count; ++row) {
        const int64_t sourceRow = row + sourceRowOffset;
        if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
            continue;
        }
        const int64_t ownedBase = checked_mul_int64_or_abort(
            row,
            outputCols,
            "[DACPP][MPI][OR] resident halo 2D owned slice base overflow");
        const int64_t sourceBase = checked_mul_int64_or_abort(
            sourceRow,
            inputCols,
            "[DACPP][MPI][OR] resident halo 2D source slice base overflow");
        for (int64_t col = 0; col < outputCols; ++col) {
            const int64_t sourceCol = col + sourceColOffset;
            if (sourceCol < 0 || sourceCol >= inputCols) {
                continue;
            }
            const std::size_t ownedIdx =
                static_cast<std::size_t>(ownedBase + col);
            const std::size_t sourceIdx =
                static_cast<std::size_t>(sourceBase + sourceCol);
            if (ownedIdx < owned.size() && sourceIdx < local.size()) {
                owned[ownedIdx] = local[sourceIdx];
            }
        }
    }
    return owned;
}

inline void byte_counts_displs(const std::vector<int>& elemCounts,
                               const std::vector<int>& elemDispls,
                               std::size_t elemSize,
                               std::vector<int>& byteCounts,
                               std::vector<int>& byteDispls) {
    byteCounts.resize(elemCounts.size());
    byteDispls.resize(elemDispls.size());
    for (std::size_t idx = 0; idx < elemCounts.size(); ++idx) {
        byteCounts[idx] = narrow_mpi_count_or_abort(
            checked_mul_int64_or_abort(
                static_cast<int64_t>(elemCounts[idx]),
                static_cast<int64_t>(elemSize),
                "[DACPP][MPI][OR] byte count exceeds MPI int range"),
            "[DACPP][MPI][OR] byte count exceeds MPI int range");
        byteDispls[idx] = narrow_mpi_count_or_abort(
            checked_mul_int64_or_abort(
                static_cast<int64_t>(elemDispls[idx]),
                static_cast<int64_t>(elemSize),
                "[DACPP][MPI][OR] byte displacement exceeds MPI int range"),
            "[DACPP][MPI][OR] byte displacement exceeds MPI int range");
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
