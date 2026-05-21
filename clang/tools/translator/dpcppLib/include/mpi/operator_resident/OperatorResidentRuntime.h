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
#include <utility>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

namespace dacpp {
namespace mpi {
namespace operator_resident {

inline sycl::queue& default_queue() {
    static sycl::queue* queue = new sycl::queue{sycl::default_selector_v};
    return *queue;
}

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

inline int64_t fixed_block_count_1d(int64_t total,
                                    int64_t blockSize,
                                    int64_t blockStride) {
    if (total < blockSize || blockSize <= 0 || blockStride <= 0) {
        return 0;
    }
    return ((total - blockSize) / blockStride) + 1;
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
    int prev_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
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
    layout.prev_rank = nearest_nonempty_rank_1d(outputTotal, rank, size, -1);
    layout.next_rank = nearest_nonempty_rank_1d(outputTotal, rank, size, 1);
    return layout;
}

inline ResidentHalo1DLayout resident_halo_1d_layout_temporal(
    int64_t outputTotal,
    int rank,
    int size,
    int temporalBlockSize,
    int windowSize) {
    ResidentHalo1DLayout layout;
    layout.owned = rank_range_1d(outputTotal, rank, size);
    const int64_t halo = std::max<int64_t>(0, temporalBlockSize - 1);
    const int64_t readerTotal =
        outputTotal + std::max<int64_t>(0, windowSize - 1);
    if (layout.owned.count > 0) {
        const int64_t begin =
            std::max<int64_t>(0, layout.owned.begin - halo);
        const int64_t end =
            std::min<int64_t>(readerTotal,
                              layout.owned.begin + layout.owned.count +
                                  halo + std::max<int64_t>(0, windowSize - 1));
        layout.global_begin = begin;
        layout.local_size = std::max<int64_t>(0, end - begin);
        layout.owned_offset = layout.owned.begin - begin;
        layout.left_halo = static_cast<int>(std::min<int64_t>(
            layout.owned_offset, std::numeric_limits<int>::max()));
        const int64_t rightHalo =
            layout.local_size - layout.owned_offset - layout.owned.count;
        layout.right_halo = static_cast<int>(std::min<int64_t>(
            std::max<int64_t>(0, rightHalo), std::numeric_limits<int>::max()));
    }
    layout.prev_rank = nearest_nonempty_rank_1d(outputTotal, rank, size, -1);
    layout.next_rank = nearest_nonempty_rank_1d(outputTotal, rank, size, 1);
    return layout;
}

struct ResidentHalo2DRowLayout {
    RankRange1D owned_rows{};
    int64_t input_cols = 0;
    int64_t local_row_count = 0;
    int64_t local_size = 0;
    int64_t global_row_begin = 0;
    int64_t owned_row_offset = 0;
    int prev_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
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
    layout.owned_row_offset = 0;
    layout.prev_rank = nearest_nonempty_rank_1d(outputRows, rank, size, -1);
    layout.next_rank = nearest_nonempty_rank_1d(outputRows, rank, size, 1);
    return layout;
}

inline ResidentHalo2DRowLayout resident_halo_2d_row_layout_temporal(
    int64_t outputRows,
    int64_t inputCols,
    int rank,
    int size,
    int temporalBlockRows) {
    ResidentHalo2DRowLayout layout;
    layout.owned_rows = rank_range_1d(outputRows, rank, size);
    layout.input_cols = inputCols;
    const int64_t halo = std::max<int64_t>(0, temporalBlockRows - 1);
    if (layout.owned_rows.count > 0) {
        const int64_t begin =
            std::max<int64_t>(0, layout.owned_rows.begin - halo);
        const int64_t end =
            std::min<int64_t>(outputRows + 2,
                              layout.owned_rows.begin +
                                  layout.owned_rows.count + halo + 2);
        layout.global_row_begin = begin;
        layout.local_row_count = std::max<int64_t>(0, end - begin);
        layout.owned_row_offset = layout.owned_rows.begin - begin;
    }
    layout.local_size = checked_mul_int64_or_abort(
        layout.local_row_count,
        inputCols,
        "[DACPP][MPI][OR] temporal resident halo 2D local size exceeds MPI int range");
    layout.prev_rank = nearest_nonempty_rank_1d(outputRows, rank, size, -1);
    layout.next_rank = nearest_nonempty_rank_1d(outputRows, rank, size, 1);
    return layout;
}

template <typename T>
void scatter_window_2d_rows_temporal(const std::vector<T>& global,
                                     std::vector<T>& local,
                                     int64_t outputRows,
                                     int64_t inputCols,
                                     int temporalBlockRows,
                                     const ResidentHalo2DRowLayout& layout,
                                     int rank,
                                     int size,
                                     MPI_Datatype mpiType) {
    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo2DRowLayout target =
                resident_halo_2d_row_layout_temporal(
                    outputRows, inputCols, r, size, temporalBlockRows);
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
                    "[DACPP][MPI][OR] temporal resident halo 2D scatter count exceeds MPI int range");
                MPI_Send(global.data() + begin,
                         sendCount,
                         mpiType,
                         r,
                         4302,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] temporal resident halo 2D recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4302,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

template <typename T>
void scatter_window_1d(const std::vector<T>& global,
                       std::vector<T>& local,
                       int64_t outputTotal,
                       int64_t inputTotal,
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
            if (static_cast<int64_t>(begin + count) > inputTotal ||
                begin + count > global.size()) {
                std::fprintf(stderr,
                             "[DACPP][MPI][OR] resident halo 1D scatter window exceeds reader size\n");
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
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
void scatter_window_1d_temporal(const std::vector<T>& global,
                                std::vector<T>& local,
                                int64_t outputTotal,
                                int64_t inputTotal,
                                int temporalBlockSize,
                                int windowSize,
                                const ResidentHalo1DLayout& layout,
                                int rank,
                                int size,
                                MPI_Datatype mpiType) {
    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo1DLayout target =
                resident_halo_1d_layout_temporal(
                    outputTotal, r, size, temporalBlockSize, windowSize);
            if (target.local_size <= 0) {
                continue;
            }
            const std::size_t begin =
                static_cast<std::size_t>(target.global_begin);
            const std::size_t count =
                static_cast<std::size_t>(target.local_size);
            if (static_cast<int64_t>(begin + count) > inputTotal ||
                begin + count > global.size()) {
                std::fprintf(stderr,
                             "[DACPP][MPI][OR] temporal resident halo 1D scatter window exceeds reader size\n");
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
            if (r == 0) {
                std::copy(global.begin() + begin,
                          global.begin() + begin + count,
                          local.begin());
            } else {
                const int sendCount = narrow_mpi_count_or_abort(
                    target.local_size,
                    "[DACPP][MPI][OR] temporal resident halo 1D scatter count exceeds MPI int range");
                MPI_Send(global.data() + begin,
                         sendCount,
                         mpiType,
                         r,
                         4202,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] temporal resident halo 1D recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4202,
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
    (void)outputTotal;
    (void)rank;
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
    T sendRight = writerLocal.empty() ? T{} : writerLocal.back();
    T sendLeft = writerLocal.empty() ? T{} : writerLocal.front();
    T recvLeft{};
    T recvRight{};
    MPI_Request requests[4];
    int requestCount = 0;
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(&recvLeft,
                  1,
                  mpiType,
                  prev,
                  4101,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (windowSize > 2) {
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(&recvRight,
                      1,
                      mpiType,
                      next,
                      4102,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(&sendRight,
                  1,
                  mpiType,
                  next,
                  4101,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (windowSize > 2 && prev != MPI_PROC_NULL) {
        MPI_Isend(&sendLeft,
                  1,
                  mpiType,
                  prev,
                  4102,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
    if (prev != MPI_PROC_NULL && !local.empty()) {
        local[0] = recvLeft;
    }
    const int64_t rightIdx = layout.owned.count + windowSize - 2;
    if (windowSize > 2 && next != MPI_PROC_NULL && rightIdx >= 0 &&
        rightIdx < static_cast<int64_t>(local.size())) {
        local[static_cast<std::size_t>(rightIdx)] = recvRight;
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
    (void)outputTotal;
    (void)rank;
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
    const std::ptrdiff_t firstInterior =
        static_cast<std::ptrdiff_t>(interiorOffset);
    const std::ptrdiff_t lastInterior = static_cast<std::ptrdiff_t>(
        interiorOffset + layout.owned.count - 1);
    MPI_Request requests[4];
    int requestCount = 0;
    T recvLeft{};
    T recvRight{};
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(&recvLeft,
                  1,
                  mpiType,
                  prev,
                  4101,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (windowSize > 2) {
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(&recvRight,
                      1,
                      mpiType,
                      next,
                      4102,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(local.data() + lastInterior,
                  1,
                  mpiType,
                  next,
                  4101,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (windowSize > 2 && prev != MPI_PROC_NULL) {
        MPI_Isend(local.data() + firstInterior,
                  1,
                  mpiType,
                  prev,
                  4102,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
    if (prev != MPI_PROC_NULL && firstInterior > 0) {
        local[static_cast<std::size_t>(firstInterior - 1)] = recvLeft;
    }
    const int64_t rightIdx = interiorOffset + layout.owned.count;
    if (windowSize > 2 && next != MPI_PROC_NULL && rightIdx >= 0 &&
        rightIdx < static_cast<int64_t>(local.size())) {
        local[static_cast<std::size_t>(rightIdx)] = recvRight;
    }
}

template <typename T>
void exchange_halo_1d_temporal_inplace(std::vector<T>& local,
                                       const ResidentHalo1DLayout& layout,
                                       int64_t outputTotal,
                                       int windowSize,
                                       int interiorOffset,
                                       int temporalBlockSize,
                                       int rank,
                                       int size,
                                       MPI_Datatype mpiType) {
    if (interiorOffset != 1 || temporalBlockSize <= 0 || windowSize != 3) {
        return;
    }
    (void)rank;
    bool hasNarrowTemporalPartition = false;
    for (int r = 0; r < size; ++r) {
        const RankRange1D range = rank_range_1d(outputTotal, r, size);
        if (range.count > 0 &&
            range.count < static_cast<int64_t>(temporalBlockSize)) {
            hasNarrowTemporalPartition = true;
            break;
        }
    }
    const int64_t firstInterior =
        layout.owned_offset + static_cast<int64_t>(interiorOffset);
    if (hasNarrowTemporalPartition) {
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        int64_t gatheredCount = 0;
        for (int r = 0; r < size; ++r) {
            const RankRange1D range = rank_range_1d(outputTotal, r, size);
            counts[static_cast<std::size_t>(r)] =
                narrow_mpi_count_or_abort(
                    range.count,
                    "[DACPP][MPI][OR] temporal resident halo 1D fallback count exceeds MPI int range");
            displs[static_cast<std::size_t>(r)] =
                narrow_mpi_count_or_abort(
                    gatheredCount,
                    "[DACPP][MPI][OR] temporal resident halo 1D fallback displacement exceeds MPI int range");
            gatheredCount += range.count;
        }
        std::vector<T> ownedSend(
            static_cast<std::size_t>(layout.owned.count), T{});
        for (int64_t idx = 0; idx < layout.owned.count; ++idx) {
            const int64_t sourceIdx = firstInterior + idx;
            if (sourceIdx >= 0 &&
                sourceIdx < static_cast<int64_t>(local.size())) {
                ownedSend[static_cast<std::size_t>(idx)] =
                    local[static_cast<std::size_t>(sourceIdx)];
            }
        }
        std::vector<T> gathered(static_cast<std::size_t>(gatheredCount), T{});
        const int sendCount = narrow_mpi_count_or_abort(
            layout.owned.count,
            "[DACPP][MPI][OR] temporal resident halo 1D fallback send count exceeds MPI int range");
        MPI_Allgatherv(ownedSend.empty() ? nullptr : ownedSend.data(),
                       sendCount,
                       mpiType,
                       gathered.empty() ? nullptr : gathered.data(),
                       counts.data(),
                       displs.data(),
                       mpiType,
                       MPI_COMM_WORLD);
        for (int64_t localIdx = 0;
             localIdx < static_cast<int64_t>(local.size());
             ++localIdx) {
            const int64_t globalReaderIdx = layout.global_begin + localIdx;
            const int64_t outputIdx =
                globalReaderIdx - static_cast<int64_t>(interiorOffset);
            if (outputIdx >= 0 && outputIdx < outputTotal &&
                outputIdx < static_cast<int64_t>(gathered.size())) {
                local[static_cast<std::size_t>(localIdx)] =
                    gathered[static_cast<std::size_t>(outputIdx)];
            }
        }
        return;
    }
    if (layout.owned.count <= 0 || local.empty()) {
        return;
    }
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
    const int sendItemsPerNeighbor = std::min<int64_t>(
        layout.owned.count, static_cast<int64_t>(temporalBlockSize));
    if (sendItemsPerNeighbor <= 0) {
        return;
    }
    const int64_t leftRecvItems = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockSize),
        std::max<int64_t>(0, firstInterior));
    const int64_t rightRecvStart = firstInterior + layout.owned.count;
    const int64_t rightRecvItems = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockSize),
        std::max<int64_t>(
            0, static_cast<int64_t>(local.size()) - rightRecvStart));
    auto packItems = [&](int64_t itemBegin, int64_t itemCount) {
        std::vector<T> packed(static_cast<std::size_t>(itemCount), T{});
        for (int64_t item = 0; item < itemCount; ++item) {
            const int64_t sourceIdx = itemBegin + item;
            if (sourceIdx >= 0 &&
                sourceIdx < static_cast<int64_t>(local.size())) {
                packed[static_cast<std::size_t>(item)] =
                    local[static_cast<std::size_t>(sourceIdx)];
            }
        }
        return packed;
    };
    auto unpackItems = [&](const std::vector<T>& packed, int64_t itemBegin) {
        for (int64_t item = 0;
             item < static_cast<int64_t>(packed.size());
             ++item) {
            const int64_t targetIdx = itemBegin + item;
            if (targetIdx >= 0 &&
                targetIdx < static_cast<int64_t>(local.size())) {
                local[static_cast<std::size_t>(targetIdx)] =
                    packed[static_cast<std::size_t>(item)];
            }
        }
    };
    std::vector<T> leftSendBuffer =
        packItems(firstInterior, sendItemsPerNeighbor);
    std::vector<T> rightSendBuffer =
        packItems(firstInterior + layout.owned.count - sendItemsPerNeighbor,
                  sendItemsPerNeighbor);
    std::vector<T> leftRecvBuffer(static_cast<std::size_t>(leftRecvItems),
                                  T{});
    std::vector<T> rightRecvBuffer(static_cast<std::size_t>(rightRecvItems),
                                   T{});
    const int sendCount = narrow_mpi_count_or_abort(
        sendItemsPerNeighbor,
        "[DACPP][MPI][OR] temporal resident halo 1D send count exceeds MPI int range");
    const int leftRecvCount = narrow_mpi_count_or_abort(
        leftRecvItems,
        "[DACPP][MPI][OR] temporal resident halo 1D left receive count exceeds MPI int range");
    const int rightRecvCount = narrow_mpi_count_or_abort(
        rightRecvItems,
        "[DACPP][MPI][OR] temporal resident halo 1D right receive count exceeds MPI int range");
    MPI_Request requests[4];
    int requestCount = 0;
    if (prev != MPI_PROC_NULL && leftRecvCount > 0) {
        MPI_Irecv(leftRecvBuffer.data(),
                  leftRecvCount,
                  mpiType,
                  prev,
                  4113,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL && rightRecvCount > 0) {
        MPI_Irecv(rightRecvBuffer.data(),
                  rightRecvCount,
                  mpiType,
                  next,
                  4114,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(rightSendBuffer.data(),
                  sendCount,
                  mpiType,
                  next,
                  4113,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(leftSendBuffer.data(),
                  sendCount,
                  mpiType,
                  prev,
                  4114,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
    if (prev != MPI_PROC_NULL && leftRecvCount > 0) {
        unpackItems(leftRecvBuffer, firstInterior - leftRecvItems);
    }
    if (next != MPI_PROC_NULL && rightRecvCount > 0) {
        unpackItems(rightRecvBuffer, rightRecvStart);
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
    (void)outputRows;
    (void)rank;
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
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
    const std::ptrdiff_t bottomSendOffset =
        static_cast<std::ptrdiff_t>((layout.owned_rows.count - 1) *
                                    outputCols);
    MPI_Request requests[4];
    int requestCount = 0;
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(local.data() + topRecvOffset,
                  sendCount,
                  mpiType,
                  prev,
                  4203,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(local.data() + bottomRecvOffset,
                  sendCount,
                  mpiType,
                  next,
                  4204,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(writerLocal.data() + bottomSendOffset,
                  sendCount,
                  mpiType,
                  next,
                  4203,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(writerLocal.data(),
                  sendCount,
                  mpiType,
                  prev,
                  4204,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
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
    (void)outputRows;
    (void)rank;
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
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
    MPI_Request requests[4];
    int requestCount = 0;
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(local.data() + topRecvOffset,
                  sendCount,
                  mpiType,
                  prev,
                  4203,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(local.data() + bottomRecvOffset,
                  sendCount,
                  mpiType,
                  next,
                  4204,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(local.data() + bottomSendOffset,
                  sendCount,
                  mpiType,
                  next,
                  4203,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(local.data() + topSendOffset,
                  sendCount,
                  mpiType,
                  prev,
                  4204,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
}

template <typename T>
void exchange_halo_2d_rows_temporal_inplace(
    std::vector<T>& local,
    const ResidentHalo2DRowLayout& layout,
    int64_t outputRows,
    int64_t outputCols,
    int64_t inputCols,
    int interiorRowOffset,
    int interiorColOffset,
    int temporalBlockRows,
    int rank,
    int size,
    MPI_Datatype mpiType) {
    if (outputCols <= 0 || inputCols <= 0 ||
        interiorRowOffset != 1 || interiorColOffset != 1 ||
        temporalBlockRows <= 1) {
        return;
    }
    (void)rank;
    bool hasNarrowTemporalPartition = false;
    for (int r = 0; r < size; ++r) {
        const RankRange1D range = rank_range_1d(outputRows, r, size);
        if (range.count > 0 &&
            range.count < static_cast<int64_t>(temporalBlockRows)) {
            hasNarrowTemporalPartition = true;
            break;
        }
    }
    if (hasNarrowTemporalPartition) {
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        int64_t gatheredCount = 0;
        for (int r = 0; r < size; ++r) {
            const RankRange1D range = rank_range_1d(outputRows, r, size);
            counts[static_cast<std::size_t>(r)] =
                narrow_mpi_count_or_abort(
                    checked_mul_int64_or_abort(
                        range.count,
                        outputCols,
                        "[DACPP][MPI][OR] temporal resident halo 2D fallback count overflow"),
                    "[DACPP][MPI][OR] temporal resident halo 2D fallback count exceeds MPI int range");
            displs[static_cast<std::size_t>(r)] =
                narrow_mpi_count_or_abort(
                    gatheredCount,
                    "[DACPP][MPI][OR] temporal resident halo 2D fallback displacement exceeds MPI int range");
            gatheredCount += checked_mul_int64_or_abort(
                range.count,
                outputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D fallback total overflow");
        }
        const int64_t sendElemCount = checked_mul_int64_or_abort(
            layout.owned_rows.count,
            outputCols,
            "[DACPP][MPI][OR] temporal resident halo 2D fallback send size overflow");
        std::vector<T> ownedSend(static_cast<std::size_t>(sendElemCount), T{});
        for (int64_t row = 0; row < layout.owned_rows.count; ++row) {
            const int64_t sourceRow =
                layout.owned_row_offset + interiorRowOffset + row;
            if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
                continue;
            }
            const int64_t sourceBase = checked_mul_int64_or_abort(
                sourceRow,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D fallback send row overflow");
            const int64_t sendBase = checked_mul_int64_or_abort(
                row,
                outputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D fallback send base overflow");
            for (int64_t col = 0; col < outputCols; ++col) {
                const int64_t sourceCol = col + interiorColOffset;
                if (sourceCol < 0 || sourceCol >= inputCols) {
                    continue;
                }
                const std::size_t sourceIdx =
                    static_cast<std::size_t>(sourceBase + sourceCol);
                const std::size_t sendIdx =
                    static_cast<std::size_t>(sendBase + col);
                if (sourceIdx < local.size() && sendIdx < ownedSend.size()) {
                    ownedSend[sendIdx] = local[sourceIdx];
                }
            }
        }
        std::vector<T> gathered(static_cast<std::size_t>(gatheredCount), T{});
        const int sendCount = narrow_mpi_count_or_abort(
            sendElemCount,
            "[DACPP][MPI][OR] temporal resident halo 2D fallback send count exceeds MPI int range");
        MPI_Allgatherv(ownedSend.empty() ? nullptr : ownedSend.data(),
                       sendCount,
                       mpiType,
                       gathered.empty() ? nullptr : gathered.data(),
                       counts.data(),
                       displs.data(),
                       mpiType,
                       MPI_COMM_WORLD);
        if (!local.empty()) {
            for (int64_t localRow = 0;
                 localRow < layout.local_row_count;
                 ++localRow) {
                const int64_t globalFullRow =
                    layout.global_row_begin + localRow;
                const int64_t outputRow = globalFullRow - interiorRowOffset;
                if (outputRow < 0 || outputRow >= outputRows) {
                    continue;
                }
                const int64_t targetBase = checked_mul_int64_or_abort(
                    localRow,
                    inputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D fallback target row overflow");
                const int64_t sourceBase = checked_mul_int64_or_abort(
                    outputRow,
                    outputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D fallback source row overflow");
                for (int64_t col = 0; col < outputCols; ++col) {
                    const int64_t targetCol = col + interiorColOffset;
                    if (targetCol < 0 || targetCol >= inputCols) {
                        continue;
                    }
                    const std::size_t targetIdx =
                        static_cast<std::size_t>(targetBase + targetCol);
                    const std::size_t sourceIdx =
                        static_cast<std::size_t>(sourceBase + col);
                    if (targetIdx < local.size() && sourceIdx < gathered.size()) {
                        local[targetIdx] = gathered[sourceIdx];
                    }
                }
            }
        }
        return;
    }
    if (layout.owned_rows.count <= 0 || local.empty()) {
        return;
    }
    (void)size;
    const int prev = layout.prev_rank;
    const int next = layout.next_rank;
    const int sendRowsPerNeighbor = std::min<int64_t>(
        layout.owned_rows.count, static_cast<int64_t>(temporalBlockRows));
    if (sendRowsPerNeighbor <= 0) {
        return;
    }
    const int sendCount = narrow_mpi_count_or_abort(
        checked_mul_int64_or_abort(
            sendRowsPerNeighbor,
            outputCols,
            "[DACPP][MPI][OR] temporal resident halo 2D halo width overflow"),
        "[DACPP][MPI][OR] temporal resident halo 2D halo width exceeds MPI int range");
    const int64_t topRecvRows = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockRows),
        std::max<int64_t>(0, layout.owned_row_offset + interiorRowOffset));
    const int64_t bottomRecvRowStart =
        layout.owned_row_offset + interiorRowOffset + layout.owned_rows.count;
    const int64_t bottomRecvRows = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockRows),
        std::max<int64_t>(0, layout.local_row_count - bottomRecvRowStart));
    const int topRecvCount = narrow_mpi_count_or_abort(
        checked_mul_int64_or_abort(
            topRecvRows,
            outputCols,
            "[DACPP][MPI][OR] temporal resident halo 2D top receive width overflow"),
        "[DACPP][MPI][OR] temporal resident halo 2D top receive width exceeds MPI int range");
    const int bottomRecvCount = narrow_mpi_count_or_abort(
        checked_mul_int64_or_abort(
            bottomRecvRows,
            outputCols,
            "[DACPP][MPI][OR] temporal resident halo 2D bottom receive width overflow"),
        "[DACPP][MPI][OR] temporal resident halo 2D bottom receive width exceeds MPI int range");
    auto packRows = [&](int64_t rowBegin, int64_t rowCount) {
        std::vector<T> packed(static_cast<std::size_t>(
            checked_mul_int64_or_abort(
                rowCount,
                outputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D pack size overflow")),
                              T{});
        for (int64_t row = 0; row < rowCount; ++row) {
            const int64_t sourceRow = rowBegin + row;
            if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
                continue;
            }
            const int64_t sourceBase = checked_mul_int64_or_abort(
                sourceRow,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D pack row overflow");
            const int64_t packedBase = checked_mul_int64_or_abort(
                row,
                outputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D pack base overflow");
            for (int64_t col = 0; col < outputCols; ++col) {
                const int64_t sourceCol = col + interiorColOffset;
                if (sourceCol < 0 || sourceCol >= inputCols) {
                    continue;
                }
                const std::size_t sourceIdx =
                    static_cast<std::size_t>(sourceBase + sourceCol);
                const std::size_t packedIdx =
                    static_cast<std::size_t>(packedBase + col);
                if (sourceIdx < local.size() && packedIdx < packed.size()) {
                    packed[packedIdx] = local[sourceIdx];
                }
            }
        }
        return packed;
    };
    auto unpackRows = [&](const std::vector<T>& packed,
                          int64_t rowBegin,
                          int64_t rowCount) {
        for (int64_t row = 0; row < rowCount; ++row) {
            const int64_t targetRow = rowBegin + row;
            if (targetRow < 0 || targetRow >= layout.local_row_count) {
                continue;
            }
            const int64_t targetBase = checked_mul_int64_or_abort(
                targetRow,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D unpack row overflow");
            const int64_t packedBase = checked_mul_int64_or_abort(
                row,
                outputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D unpack base overflow");
            for (int64_t col = 0; col < outputCols; ++col) {
                const int64_t targetCol = col + interiorColOffset;
                if (targetCol < 0 || targetCol >= inputCols) {
                    continue;
                }
                const std::size_t targetIdx =
                    static_cast<std::size_t>(targetBase + targetCol);
                const std::size_t packedIdx =
                    static_cast<std::size_t>(packedBase + col);
                if (targetIdx < local.size() && packedIdx < packed.size()) {
                    local[targetIdx] = packed[packedIdx];
                }
            }
        }
    };
    std::vector<T> topSendBuffer =
        packRows(layout.owned_row_offset + interiorRowOffset,
                 sendRowsPerNeighbor);
    std::vector<T> bottomSendBuffer = packRows(
        layout.owned_row_offset + interiorRowOffset +
            layout.owned_rows.count - sendRowsPerNeighbor,
        sendRowsPerNeighbor);
    std::vector<T> topRecvBuffer(static_cast<std::size_t>(topRecvCount), T{});
    std::vector<T> bottomRecvBuffer(static_cast<std::size_t>(bottomRecvCount),
                                    T{});
    MPI_Request requests[4];
    int requestCount = 0;
    if (prev != MPI_PROC_NULL && topRecvCount > 0) {
        MPI_Irecv(topRecvBuffer.data(),
                  topRecvCount,
                  mpiType,
                  prev,
                  4213,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL && bottomRecvCount > 0) {
        MPI_Irecv(bottomRecvBuffer.data(),
                  bottomRecvCount,
                  mpiType,
                  next,
                  4214,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(bottomSendBuffer.data(),
                  sendCount,
                  mpiType,
                  next,
                  4213,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(topSendBuffer.data(),
                  sendCount,
                  mpiType,
                  prev,
                  4214,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
    if (prev != MPI_PROC_NULL && topRecvCount > 0) {
        unpackRows(topRecvBuffer, 0, topRecvRows);
    }
    if (next != MPI_PROC_NULL && bottomRecvCount > 0) {
        unpackRows(bottomRecvBuffer, bottomRecvRowStart, bottomRecvRows);
    }
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

template <typename T>
std::vector<T> owned_slice_2d_rows_temporal(
    const std::vector<T>& local,
    const ResidentHalo2DRowLayout& layout,
    int64_t outputCols,
    int64_t inputCols,
    int sourceRowOffset,
    int sourceColOffset) {
    return owned_slice_2d_rows(local,
                               layout,
                               outputCols,
                               inputCols,
                               static_cast<int>(layout.owned_row_offset) +
                                   sourceRowOffset,
                               sourceColOffset);
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

struct FixedBlockPhaseExchangeLayout {
    int64_t total = 0;
    RankRange1D owned{};
    int64_t local_begin = 0;
    int64_t local_count = 0;
};

inline FixedBlockPhaseExchangeLayout fixed_block_phase_exchange_layout(
    int64_t total,
    int rank,
    int size) {
    FixedBlockPhaseExchangeLayout layout;
    layout.total = total;
    layout.owned = rank_range_1d(total, rank, size);
    layout.local_begin = layout.owned.begin;
    layout.local_count = layout.owned.count;
    return layout;
}

inline bool fixed_block_phase_exchange_alignment_ok(
    int64_t total,
    int size,
    int blockSize) {
    if (blockSize <= 0 || total < 0 || size <= 0) {
        return false;
    }
    if (blockSize == 1) {
        return true;
    }
    for (int r = 0; r < size; ++r) {
        const RankRange1D range = rank_range_1d(total, r, size);
        if (range.count <= 0) {
            continue;
        }
        if ((range.begin % blockSize) != 0) {
            return false;
        }
        if ((range.count % blockSize) != 0) {
            return false;
        }
    }
    return true;
}

[[noreturn]] inline void abort_fixed_block_phase_exchange_alignment(
    int64_t total,
    int size,
    int blockSize) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::fprintf(stderr,
                     "[DACPP][MPI][OR][P5] fixed-block phase-exchange alignment failed: total=%lld size=%d blockSize=%d\n",
                     static_cast<long long>(total), size, blockSize);
    }
    MPI_Abort(MPI_COMM_WORLD, 7);
    std::abort();
}

[[noreturn]] inline void abort_fixed_block_phase_exchange_total_mismatch(
    int64_t runtimeTotal,
    int64_t provenEvenTotal) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::fprintf(stderr,
                     "[DACPP][MPI][OR][P5] fixed-block phase-exchange total mismatch: runtime=%lld proven-even=%lld\n",
                     static_cast<long long>(runtimeTotal),
                     static_cast<long long>(provenEvenTotal));
    }
    MPI_Abort(MPI_COMM_WORLD, 8);
    std::abort();
}

inline void check_fixed_block_phase_exchange_total(int64_t runtimeTotal,
                                                   int64_t provenEvenTotal) {
    if (provenEvenTotal <= 0 || (provenEvenTotal % 2) != 0 ||
        runtimeTotal != provenEvenTotal) {
        abort_fixed_block_phase_exchange_total_mismatch(runtimeTotal,
                                                        provenEvenTotal);
    }
}

template <typename T, typename CompareSwap>
inline void fixed_block_phase_exchange_phase(
    std::vector<T>& local,
    int64_t globalBegin,
    int64_t totalGlobal,
    int rank,
    int size,
    int phaseOffset,
    MPI_Datatype mpiType,
    CompareSwap compareSwap) {
    const int64_t localCount = static_cast<int64_t>(local.size());
    if (localCount <= 0 || totalGlobal <= 1) {
        (void)mpiType;
        return;
    }

    for (int64_t pos = 0; pos + 1 < localCount; ++pos) {
        const int64_t globalPos = globalBegin + pos;
        if ((globalPos % 2) == phaseOffset) {
            compareSwap(local.data() + pos);
        }
    }

    const int prev = nearest_nonempty_rank_1d(totalGlobal, rank, size, -1);
    const int next = nearest_nonempty_rank_1d(totalGlobal, rank, size, 1);
    const int sendTag = phaseOffset == 0 ? 4501 : 4503;
    const int recvTag = phaseOffset == 0 ? 4502 : 4504;

    const int64_t rightPairStart = globalBegin + localCount - 1;
    const bool hasRightCross =
        next != MPI_PROC_NULL && localCount > 0 &&
        rightPairStart + 1 < totalGlobal &&
        (rightPairStart % 2) == phaseOffset;
    const int64_t leftPairStart = globalBegin - 1;
    const bool hasLeftCross =
        prev != MPI_PROC_NULL && localCount > 0 &&
        leftPairStart >= 0 && leftPairStart + 1 < totalGlobal &&
        (leftPairStart % 2) == phaseOffset;

    T sendLeft = local[0];
    T sendRight = local[static_cast<std::size_t>(localCount - 1)];
    T recvLeft{};
    T recvRight{};
    MPI_Request requests[4];
    int requestCount = 0;
    if (hasLeftCross) {
        MPI_Irecv(&recvLeft, 1, mpiType, prev, sendTag, MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (hasRightCross) {
        MPI_Irecv(&recvRight, 1, mpiType, next, recvTag, MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (hasLeftCross) {
        MPI_Isend(&sendLeft, 1, mpiType, prev, recvTag, MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (hasRightCross) {
        MPI_Isend(&sendRight, 1, mpiType, next, sendTag, MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }

    if (hasRightCross) {
        T pair[2] = {sendRight, recvRight};
        compareSwap(pair);
        local[static_cast<std::size_t>(localCount - 1)] = pair[0];
    }
    if (hasLeftCross) {
        T pair[2] = {recvLeft, sendLeft};
        compareSwap(pair);
        local[0] = pair[1];
    }
}

template <typename T, typename CompareSwap>
inline void fixed_block_phase_exchange_step(
    std::vector<T>& local,
    int64_t globalBegin,
    int64_t totalGlobal,
    int rank,
    int size,
    int blockSize,
    int phaseShiftOffset,
    MPI_Datatype mpiType,
    CompareSwap compareSwap) {
    if (blockSize != 2 || phaseShiftOffset != 1) {
        (void)mpiType;
        return;
    }
    fixed_block_phase_exchange_phase(local, globalBegin, totalGlobal, rank,
                                     size, 0, mpiType, compareSwap);
    fixed_block_phase_exchange_phase(local, globalBegin, totalGlobal, rank,
                                     size, phaseShiftOffset, mpiType,
                                     compareSwap);
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

template <typename T, typename TensorT>
std::vector<T>& replace_resident(const TensorT& tensor,
                                 std::vector<T>&& value) {
    ResidencyKey key{static_cast<const void*>(&tensor), std::type_index(typeid(T))};
    auto& buffer = typed_resident_storage<T>()[key];
    buffer = std::move(value);
    return buffer;
}

} // namespace operator_resident
} // namespace mpi
} // namespace dacpp

#endif
