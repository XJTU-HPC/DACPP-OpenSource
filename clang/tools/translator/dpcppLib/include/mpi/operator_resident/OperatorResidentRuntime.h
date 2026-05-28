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

struct BlockRange2D {
    RankRange1D rows{};
    RankRange1D cols{};
};

struct BlockCyclic1DLayout {
    int64_t total = 0;
    int64_t block_size = 1;
    int64_t local_count = 0;
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

inline int64_t block_cyclic_count_1d(int64_t total,
                                     int rank,
                                     int size,
                                     int64_t blockSize) {
    if (total <= 0 || size <= 0 || blockSize <= 0 || rank < 0 ||
        rank >= size) {
        return 0;
    }
    const int64_t fullBlocks = total / blockSize;
    const int64_t tail = total % blockSize;
    const int64_t rounds = fullBlocks / size;
    const int64_t extraFullBlocks = fullBlocks % size;
    int64_t count = rounds * blockSize;
    if (rank < extraFullBlocks) {
        count += blockSize;
    }
    if (tail > 0 && rank == extraFullBlocks) {
        count += tail;
    }
    return count;
}

inline BlockCyclic1DLayout block_cyclic_layout_1d(int64_t total,
                                                  int rank,
                                                  int size,
                                                  int64_t blockSize) {
    const int64_t safeBlock = std::max<int64_t>(1, blockSize);
    return {total, safeBlock,
            block_cyclic_count_1d(total, rank, size, safeBlock)};
}

inline int64_t block_cyclic_global_index_1d(int64_t localIndex,
                                            int rank,
                                            int size,
                                            int64_t blockSize) {
    const int64_t safeSize = std::max<int64_t>(1, size);
    const int64_t safeBlock = std::max<int64_t>(1, blockSize);
    const int64_t localBlock = localIndex / safeBlock;
    const int64_t withinBlock = localIndex % safeBlock;
    return (localBlock * safeSize + rank) * safeBlock + withinBlock;
}

inline int block_cyclic_owner_1d(int64_t globalIndex,
                                 int size,
                                 int64_t blockSize) {
    if (globalIndex < 0 || size <= 0 || blockSize <= 0) {
        return -1;
    }
    return static_cast<int>((globalIndex / blockSize) %
                            static_cast<int64_t>(size));
}

inline int64_t block_cyclic_local_index_1d(int64_t globalIndex,
                                           int size,
                                           int64_t blockSize) {
    if (globalIndex < 0 || size <= 0 || blockSize <= 0) {
        return -1;
    }
    const int64_t block = globalIndex / blockSize;
    const int64_t withinBlock = globalIndex % blockSize;
    return (block / static_cast<int64_t>(size)) * blockSize + withinBlock;
}

inline void counts_1d_block_cyclic(int64_t total,
                                   int size,
                                   int64_t blockSize,
                                   std::vector<int>& counts) {
    counts.resize(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = narrow_mpi_count_or_abort(
            block_cyclic_count_1d(total, r, size, blockSize),
            "[DACPP][MPI][OR] block-cyclic count exceeds MPI int range");
    }
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

inline void spatial_2d_process_grid(int size, int& rows, int& cols) {
    const int safeSize = std::max(1, size);
    rows = 1;
    for (int candidate = 1; candidate * candidate <= safeSize; ++candidate) {
        if (safeSize % candidate == 0) {
            rows = candidate;
        }
    }
    cols = safeSize / rows;
}

inline int spatial_2d_rank_from_coords(int procRow,
                                       int procCol,
                                       int gridRows,
                                       int gridCols) {
    if (procRow < 0 || procRow >= gridRows || procCol < 0 ||
        procCol >= gridCols) {
        return MPI_PROC_NULL;
    }
    return procRow * gridCols + procCol;
}

inline BlockRange2D spatial_2d_owned_range(int64_t rows,
                                           int64_t cols,
                                           int rank,
                                           int size) {
    int gridRows = 1;
    int gridCols = 1;
    spatial_2d_process_grid(size, gridRows, gridCols);
    const int procRow = rank / gridCols;
    const int procCol = rank % gridCols;
    return {rank_range_1d(rows, procRow, gridRows),
            rank_range_1d(cols, procCol, gridCols)};
}

inline int spatial_2d_owner_rank(int64_t row,
                                 int64_t col,
                                 int64_t rows,
                                 int64_t cols,
                                 int size) {
    if (row < 0 || row >= rows || col < 0 || col >= cols || size <= 0) {
        return -1;
    }
    for (int rank = 0; rank < size; ++rank) {
        const BlockRange2D owned = spatial_2d_owned_range(rows, cols, rank, size);
        if (row >= owned.rows.begin &&
            row < owned.rows.begin + owned.rows.count &&
            col >= owned.cols.begin &&
            col < owned.cols.begin + owned.cols.count) {
            return rank;
        }
    }
    return -1;
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

struct ResidentHalo2DSpatialLayout {
    RankRange1D owned_rows{};
    RankRange1D owned_cols{};
    int64_t input_rows = 0;
    int64_t input_cols = 0;
    int64_t local_row_count = 0;
    int64_t local_col_count = 0;
    int64_t local_size = 0;
    int64_t global_row_begin = 0;
    int64_t global_col_begin = 0;
    int64_t owned_row_offset = 0;
    int64_t owned_col_offset = 0;
    int grid_rows = 1;
    int grid_cols = 1;
    int proc_row = 0;
    int proc_col = 0;
    int north_rank = MPI_PROC_NULL;
    int south_rank = MPI_PROC_NULL;
    int west_rank = MPI_PROC_NULL;
    int east_rank = MPI_PROC_NULL;
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

inline ResidentHalo2DSpatialLayout resident_halo_2d_spatial_layout(
    int64_t outputRows,
    int64_t outputCols,
    int64_t inputRows,
    int64_t inputCols,
    int rank,
    int size,
    int haloWidth) {
    ResidentHalo2DSpatialLayout layout;
    spatial_2d_process_grid(size, layout.grid_rows, layout.grid_cols);
    layout.proc_row = layout.grid_cols > 0 ? rank / layout.grid_cols : 0;
    layout.proc_col = layout.grid_cols > 0 ? rank % layout.grid_cols : 0;
    layout.owned_rows = rank_range_1d(outputRows,
                                      layout.proc_row,
                                      layout.grid_rows);
    layout.owned_cols = rank_range_1d(outputCols,
                                      layout.proc_col,
                                      layout.grid_cols);
    layout.input_rows = inputRows;
    layout.input_cols = inputCols;
    if (layout.owned_rows.count > 0 && layout.owned_cols.count > 0) {
        const int64_t halo = std::max<int64_t>(0, haloWidth);
        const int64_t followupOffset = 1;
        const int64_t interiorRowBegin =
            layout.owned_rows.begin + followupOffset;
        const int64_t interiorRowEnd =
            layout.owned_rows.begin + layout.owned_rows.count +
            followupOffset;
        const int64_t interiorColBegin =
            layout.owned_cols.begin + followupOffset;
        const int64_t interiorColEnd =
            layout.owned_cols.begin + layout.owned_cols.count +
            followupOffset;
        layout.global_row_begin = std::max<int64_t>(0, interiorRowBegin - halo);
        layout.global_col_begin = std::max<int64_t>(0, interiorColBegin - halo);
        const int64_t globalRowEnd =
            std::min<int64_t>(inputRows, interiorRowEnd + halo);
        const int64_t globalColEnd =
            std::min<int64_t>(inputCols, interiorColEnd + halo);
        layout.local_row_count =
            std::max<int64_t>(0, globalRowEnd - layout.global_row_begin);
        layout.local_col_count =
            std::max<int64_t>(0, globalColEnd - layout.global_col_begin);
        layout.owned_row_offset = interiorRowBegin - layout.global_row_begin;
        layout.owned_col_offset = interiorColBegin - layout.global_col_begin;
    }
    layout.local_size = checked_mul_int64_or_abort(
        layout.local_row_count,
        layout.local_col_count,
        "[DACPP][MPI][OR] spatial resident halo 2D local size overflow");
    layout.north_rank = spatial_2d_rank_from_coords(
        layout.proc_row - 1, layout.proc_col, layout.grid_rows,
        layout.grid_cols);
    layout.south_rank = spatial_2d_rank_from_coords(
        layout.proc_row + 1, layout.proc_col, layout.grid_rows,
        layout.grid_cols);
    layout.west_rank = spatial_2d_rank_from_coords(
        layout.proc_row, layout.proc_col - 1, layout.grid_rows,
        layout.grid_cols);
    layout.east_rank = spatial_2d_rank_from_coords(
        layout.proc_row, layout.proc_col + 1, layout.grid_rows,
        layout.grid_cols);
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
void scatter_window_2d_rows_temporal_offset(
    const std::vector<T>& global,
    std::vector<T>& local,
    int64_t outputRows,
    int64_t outputCols,
    int64_t inputCols,
    int temporalBlockRows,
    int targetRowOffset,
    int targetColOffset,
    const ResidentHalo2DRowLayout& layout,
    int rank,
    int size,
    MPI_Datatype mpiType) {
    auto packTarget = [&](const ResidentHalo2DRowLayout& target,
                          std::vector<T>& packed) {
        packed.assign(static_cast<std::size_t>(target.local_size), T{});
        if (target.local_size <= 0) {
            return;
        }
        for (int64_t localRow = 0; localRow < target.local_row_count;
             ++localRow) {
            const int64_t sourceRow =
                target.global_row_begin + localRow - targetRowOffset;
            if (sourceRow < 0 || sourceRow >= outputRows) {
                continue;
            }
            const int64_t targetColBegin =
                std::max<int64_t>(0, static_cast<int64_t>(targetColOffset));
            const int64_t sourceColBegin =
                std::max<int64_t>(0, -static_cast<int64_t>(targetColOffset));
            const int64_t colsToCopy = std::min<int64_t>(
                outputCols - sourceColBegin, inputCols - targetColBegin);
            if (colsToCopy <= 0) {
                continue;
            }
            const std::size_t sourceBase = static_cast<std::size_t>(
                checked_mul_int64_or_abort(
                    sourceRow,
                    outputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D offset scatter source row overflow") +
                sourceColBegin);
            const std::size_t targetBase = static_cast<std::size_t>(
                checked_mul_int64_or_abort(
                    localRow,
                    inputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D offset scatter target row overflow") +
                targetColBegin);
            std::copy_n(global.data() + sourceBase,
                        static_cast<std::size_t>(colsToCopy),
                        packed.data() + targetBase);
        }
    };

    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo2DRowLayout target =
                resident_halo_2d_row_layout_temporal(
                    outputRows, inputCols, r, size, temporalBlockRows);
            if (target.local_size <= 0) {
                continue;
            }
            std::vector<T> packed;
            packTarget(target, packed);
            if (r == 0) {
                local.swap(packed);
            } else {
                const int sendCount = narrow_mpi_count_or_abort(
                    target.local_size,
                    "[DACPP][MPI][OR] temporal resident halo 2D offset scatter count exceeds MPI int range");
                MPI_Send(packed.data(),
                         sendCount,
                         mpiType,
                         r,
                         4303,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] temporal resident halo 2D offset scatter recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4303,
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
std::vector<T> pack_spatial_window_2d(const std::vector<T>& global,
                                      const ResidentHalo2DSpatialLayout& layout,
                                      int64_t inputCols) {
    std::vector<T> packed(static_cast<std::size_t>(layout.local_size), T{});
    if (layout.local_row_count <= 0 || layout.local_col_count <= 0 ||
        inputCols <= 0 || global.empty()) {
        return packed;
    }
    for (int64_t row = 0; row < layout.local_row_count; ++row) {
        const int64_t globalRow = layout.global_row_begin + row;
        const int64_t globalBase = checked_mul_int64_or_abort(
            globalRow,
            inputCols,
            "[DACPP][MPI][OR] spatial resident halo 2D pack row overflow");
        const int64_t packedBase = checked_mul_int64_or_abort(
            row,
            layout.local_col_count,
            "[DACPP][MPI][OR] spatial resident halo 2D pack base overflow");
        for (int64_t col = 0; col < layout.local_col_count; ++col) {
            const int64_t globalCol = layout.global_col_begin + col;
            const std::size_t globalIdx =
                static_cast<std::size_t>(globalBase + globalCol);
            const std::size_t packedIdx =
                static_cast<std::size_t>(packedBase + col);
            if (globalIdx < global.size() && packedIdx < packed.size()) {
                packed[packedIdx] = global[globalIdx];
            }
        }
    }
    return packed;
}

template <typename T>
void scatter_window_2d_spatial(const std::vector<T>& global,
                               std::vector<T>& local,
                               int64_t outputRows,
                               int64_t outputCols,
                               int64_t inputRows,
                               int64_t inputCols,
                               int haloWidth,
                               const ResidentHalo2DSpatialLayout& layout,
                               int rank,
                               int size,
                               MPI_Datatype mpiType) {
    local.assign(static_cast<std::size_t>(layout.local_size), T{});
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const ResidentHalo2DSpatialLayout target =
                resident_halo_2d_spatial_layout(outputRows, outputCols,
                                                inputRows, inputCols, r, size,
                                                haloWidth);
            if (target.local_size <= 0) {
                continue;
            }
            std::vector<T> packed =
                pack_spatial_window_2d(global, target, inputCols);
            if (r == 0) {
                local = std::move(packed);
            } else {
                const int sendCount = narrow_mpi_count_or_abort(
                    target.local_size,
                    "[DACPP][MPI][OR] spatial resident halo 2D scatter count exceeds MPI int range");
                MPI_Send(packed.data(),
                         sendCount,
                         mpiType,
                         r,
                         4305,
                         MPI_COMM_WORLD);
            }
        }
    } else if (layout.local_size > 0) {
        const int recvCount = narrow_mpi_count_or_abort(
            layout.local_size,
            "[DACPP][MPI][OR] spatial resident halo 2D recv count exceeds MPI int range");
        MPI_Recv(local.data(),
                 recvCount,
                 mpiType,
                 0,
                 4305,
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
    if (interiorOffset != 1 || temporalBlockSize <= 0 ||
        (windowSize != 2 && windowSize != 3)) {
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
    const int64_t leftRecvItems = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockSize),
        std::max<int64_t>(0, firstInterior));
    const int64_t rightRecvStart = firstInterior + layout.owned.count;
    const int64_t rightRecvItems = std::min<int64_t>(
        static_cast<int64_t>(temporalBlockSize),
        std::max<int64_t>(
            0, static_cast<int64_t>(local.size()) - rightRecvStart));
    auto neighborLeftRecvItems = [&](int neighborRank) {
        if (neighborRank == MPI_PROC_NULL) {
            return int64_t{0};
        }
        const ResidentHalo1DLayout target =
            resident_halo_1d_layout_temporal(
                outputTotal, neighborRank, size, temporalBlockSize, windowSize);
        const int64_t targetFirstInterior =
            target.owned_offset + static_cast<int64_t>(interiorOffset);
        return std::min<int64_t>(
            std::min<int64_t>(target.owned.count,
                              static_cast<int64_t>(temporalBlockSize)),
            std::max<int64_t>(0, targetFirstInterior));
    };
    auto neighborRightRecvItems = [&](int neighborRank) {
        if (neighborRank == MPI_PROC_NULL) {
            return int64_t{0};
        }
        const ResidentHalo1DLayout target =
            resident_halo_1d_layout_temporal(
                outputTotal, neighborRank, size, temporalBlockSize, windowSize);
        const int64_t targetFirstInterior =
            target.owned_offset + static_cast<int64_t>(interiorOffset);
        const int64_t targetRightRecvStart =
            targetFirstInterior + target.owned.count;
        return std::min<int64_t>(
            std::min<int64_t>(target.owned.count,
                              static_cast<int64_t>(temporalBlockSize)),
            std::max<int64_t>(0, target.local_size - targetRightRecvStart));
    };
    const int64_t leftSendItems =
        std::min<int64_t>(layout.owned.count, neighborRightRecvItems(prev));
    const int64_t rightSendItems =
        std::min<int64_t>(layout.owned.count, neighborLeftRecvItems(next));
    if (leftRecvItems <= 0 && rightRecvItems <= 0 &&
        leftSendItems <= 0 && rightSendItems <= 0) {
        return;
    }
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
    std::vector<T> leftSendBuffer = packItems(firstInterior, leftSendItems);
    std::vector<T> rightSendBuffer =
        packItems(firstInterior + layout.owned.count - rightSendItems,
                  rightSendItems);
    std::vector<T> leftRecvBuffer(static_cast<std::size_t>(leftRecvItems),
                                  T{});
    std::vector<T> rightRecvBuffer(static_cast<std::size_t>(rightRecvItems),
                                   T{});
    const int leftSendCount = narrow_mpi_count_or_abort(
        leftSendItems,
        "[DACPP][MPI][OR] temporal resident halo 1D left send count exceeds MPI int range");
    const int rightSendCount = narrow_mpi_count_or_abort(
        rightSendItems,
        "[DACPP][MPI][OR] temporal resident halo 1D right send count exceeds MPI int range");
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
    if (next != MPI_PROC_NULL && rightSendCount > 0) {
        MPI_Isend(rightSendBuffer.data(),
                  rightSendCount,
                  mpiType,
                  next,
                  4113,
                  MPI_COMM_WORLD,
                  &requests[requestCount++]);
    }
    if (prev != MPI_PROC_NULL && leftSendCount > 0) {
        MPI_Isend(leftSendBuffer.data(),
                  leftSendCount,
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
void gather_spatial_owned_to_root(const std::vector<T>& owned,
                                  std::vector<T>& global,
                                  int64_t outputRows,
                                  int64_t outputCols,
                                  int rank,
                                  int size,
                                  MPI_Datatype mpiType);

template <typename T>
void exchange_halo_2d_spatial_inplace(
    std::vector<T>& local,
    const ResidentHalo2DSpatialLayout& layout,
    int64_t outputRows,
    int64_t outputCols,
    int64_t inputCols,
    int interiorRowOffset,
    int interiorColOffset,
    int haloWidth,
    int rank,
    int size,
    MPI_Datatype mpiType) {
    if (outputRows <= 0 || outputCols <= 0 || inputCols <= 0 ||
        interiorRowOffset != 1 || interiorColOffset != 1 ||
        haloWidth <= 0 ||
        layout.owned_rows.count <= 0 || layout.owned_cols.count <= 0 ||
        layout.local_row_count <= 0 || layout.local_col_count <= 0 ||
        local.empty()) {
        return;
    }
    (void)rank;
    bool hasNarrowSpatialPartition = false;
    for (int r = 0; r < size; ++r) {
        const BlockRange2D range =
            spatial_2d_owned_range(outputRows, outputCols, r, size);
        if ((range.rows.count > 0 &&
             range.rows.count < static_cast<int64_t>(haloWidth)) ||
            (range.cols.count > 0 &&
             range.cols.count < static_cast<int64_t>(haloWidth))) {
            hasNarrowSpatialPartition = true;
            break;
        }
    }
    const int64_t ownedItemCount = checked_mul_int64_or_abort(
        layout.owned_rows.count,
        layout.owned_cols.count,
        "[DACPP][MPI][OR] spatial resident halo 2D fallback owned size overflow");
    if (hasNarrowSpatialPartition) {
        std::vector<T> ownedSend(static_cast<std::size_t>(ownedItemCount),
                                 T{});
        for (int64_t row = 0; row < layout.owned_rows.count; ++row) {
            const int64_t sourceRow =
                layout.owned_row_offset + row;
            if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
                continue;
            }
            const int64_t sourceBase = checked_mul_int64_or_abort(
                sourceRow,
                layout.local_col_count,
                "[DACPP][MPI][OR] spatial resident halo 2D fallback send row overflow");
            const int64_t sendBase = checked_mul_int64_or_abort(
                row,
                layout.owned_cols.count,
                "[DACPP][MPI][OR] spatial resident halo 2D fallback send base overflow");
            for (int64_t col = 0; col < layout.owned_cols.count; ++col) {
                const int64_t sourceCol =
                    layout.owned_col_offset + col;
                if (sourceCol < 0 || sourceCol >= layout.local_col_count) {
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
        std::vector<T> gathered;
        gather_spatial_owned_to_root(ownedSend,
                                     gathered,
                                     outputRows,
                                     outputCols,
                                     rank,
                                     size,
                                     mpiType);
        const int64_t gatheredCount = checked_mul_int64_or_abort(
            outputRows,
            outputCols,
            "[DACPP][MPI][OR] spatial resident halo 2D fallback dense size overflow");
        if (rank != 0) {
            gathered.resize(static_cast<std::size_t>(gatheredCount), T{});
        }
        MPI_Bcast(gathered.empty() ? nullptr : gathered.data(),
                  narrow_mpi_count_or_abort(
                      gatheredCount,
                      "[DACPP][MPI][OR] spatial resident halo 2D fallback broadcast count exceeds MPI int range"),
                  mpiType,
                  0,
                  MPI_COMM_WORLD);
        for (int64_t row = 0; row < layout.local_row_count; ++row) {
            const int64_t globalInputRow = layout.global_row_begin + row;
            const int64_t outputRow = globalInputRow - interiorRowOffset;
            if (outputRow < 0 || outputRow >= outputRows) {
                continue;
            }
            const int64_t localBase = checked_mul_int64_or_abort(
                row,
                layout.local_col_count,
                "[DACPP][MPI][OR] spatial resident halo 2D fallback local row overflow");
            const int64_t globalBase = checked_mul_int64_or_abort(
                outputRow,
                outputCols,
                "[DACPP][MPI][OR] spatial resident halo 2D fallback global row overflow");
            for (int64_t col = 0; col < layout.local_col_count; ++col) {
                const int64_t globalInputCol = layout.global_col_begin + col;
                const int64_t outputCol = globalInputCol - interiorColOffset;
                if (outputCol < 0 || outputCol >= outputCols) {
                    continue;
                }
                const std::size_t localIdx =
                    static_cast<std::size_t>(localBase + col);
                const std::size_t globalIdx =
                    static_cast<std::size_t>(globalBase + outputCol);
                if (localIdx < local.size() && globalIdx < gathered.size()) {
                    local[localIdx] = gathered[globalIdx];
                }
            }
        }
        return;
    }
    auto packRect = [&](int64_t rowBegin,
                        int64_t rowCount,
                        int64_t colBegin,
                        int64_t colCount) {
        std::vector<T> packed(static_cast<std::size_t>(
            checked_mul_int64_or_abort(
                rowCount,
                colCount,
                "[DACPP][MPI][OR] spatial resident halo 2D pack size overflow")),
                              T{});
        for (int64_t row = 0; row < rowCount; ++row) {
            const int64_t sourceRow = rowBegin + row;
            if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
                continue;
            }
            const int64_t sourceBase = checked_mul_int64_or_abort(
                sourceRow,
                layout.local_col_count,
                "[DACPP][MPI][OR] spatial resident halo 2D pack row overflow");
            const int64_t packedBase = checked_mul_int64_or_abort(
                row,
                colCount,
                "[DACPP][MPI][OR] spatial resident halo 2D pack base overflow");
            for (int64_t col = 0; col < colCount; ++col) {
                const int64_t sourceCol = colBegin + col;
                const std::size_t sourceIdx =
                    static_cast<std::size_t>(sourceBase + sourceCol);
                const std::size_t packedIdx =
                    static_cast<std::size_t>(packedBase + col);
                if (sourceCol >= 0 && sourceCol < layout.local_col_count &&
                    sourceIdx < local.size() && packedIdx < packed.size()) {
                    packed[packedIdx] = local[sourceIdx];
                }
            }
        }
        return packed;
    };
    auto unpackRect = [&](const std::vector<T>& packed,
                          int64_t rowBegin,
                          int64_t rowCount,
                          int64_t colBegin,
                          int64_t colCount) {
        for (int64_t row = 0; row < rowCount; ++row) {
            const int64_t targetRow = rowBegin + row;
            if (targetRow < 0 || targetRow >= layout.local_row_count) {
                continue;
            }
            const int64_t targetBase = checked_mul_int64_or_abort(
                targetRow,
                layout.local_col_count,
                "[DACPP][MPI][OR] spatial resident halo 2D unpack row overflow");
            const int64_t packedBase = checked_mul_int64_or_abort(
                row,
                colCount,
                "[DACPP][MPI][OR] spatial resident halo 2D unpack base overflow");
            for (int64_t col = 0; col < colCount; ++col) {
                const int64_t targetCol = colBegin + col;
                const std::size_t targetIdx =
                    static_cast<std::size_t>(targetBase + targetCol);
                const std::size_t packedIdx =
                    static_cast<std::size_t>(packedBase + col);
                if (targetCol >= 0 && targetCol < layout.local_col_count &&
                    targetIdx < local.size() && packedIdx < packed.size()) {
                    local[targetIdx] = packed[packedIdx];
                }
            }
        }
    };
    const bool hasNorth = layout.north_rank != MPI_PROC_NULL &&
                          layout.owned_row_offset > 0;
    const bool hasSouth =
        layout.south_rank != MPI_PROC_NULL &&
        layout.owned_row_offset + layout.owned_rows.count <
            layout.local_row_count;
    const bool hasWest = layout.west_rank != MPI_PROC_NULL &&
                         layout.owned_col_offset > 0;
    const bool hasEast =
        layout.east_rank != MPI_PROC_NULL &&
        layout.owned_col_offset + layout.owned_cols.count <
            layout.local_col_count;
    const int northWestRank = spatial_2d_rank_from_coords(
        layout.proc_row - 1, layout.proc_col - 1, layout.grid_rows,
        layout.grid_cols);
    const int northEastRank = spatial_2d_rank_from_coords(
        layout.proc_row - 1, layout.proc_col + 1, layout.grid_rows,
        layout.grid_cols);
    const int southWestRank = spatial_2d_rank_from_coords(
        layout.proc_row + 1, layout.proc_col - 1, layout.grid_rows,
        layout.grid_cols);
    const int southEastRank = spatial_2d_rank_from_coords(
        layout.proc_row + 1, layout.proc_col + 1, layout.grid_rows,
        layout.grid_cols);
    const int64_t edgeWidth = std::min<int64_t>(
        static_cast<int64_t>(haloWidth),
        std::min(layout.owned_rows.count, layout.owned_cols.count));
    if (edgeWidth <= 0) {
        return;
    }
    const int64_t topRow = layout.owned_row_offset - edgeWidth;
    const int64_t bottomRow = layout.owned_row_offset + layout.owned_rows.count;
    const int64_t leftCol = layout.owned_col_offset - edgeWidth;
    const int64_t rightCol = layout.owned_col_offset + layout.owned_cols.count;
    const int64_t northRows = std::min<int64_t>(
        edgeWidth, std::max<int64_t>(0, layout.owned_row_offset));
    const int64_t southRows = std::min<int64_t>(
        edgeWidth,
        std::max<int64_t>(0, layout.local_row_count - bottomRow));
    const int64_t westCols = std::min<int64_t>(
        edgeWidth, std::max<int64_t>(0, layout.owned_col_offset));
    const int64_t eastCols = std::min<int64_t>(
        edgeWidth,
        std::max<int64_t>(0, layout.local_col_count - rightCol));
    std::vector<T> northSend = packRect(layout.owned_row_offset, edgeWidth,
                                        layout.owned_col_offset,
                                        layout.owned_cols.count);
    std::vector<T> southSend = packRect(
        layout.owned_row_offset + layout.owned_rows.count - edgeWidth,
        edgeWidth, layout.owned_col_offset, layout.owned_cols.count);
    std::vector<T> westSend = packRect(layout.owned_row_offset,
                                       layout.owned_rows.count,
                                       layout.owned_col_offset, edgeWidth);
    std::vector<T> eastSend = packRect(
        layout.owned_row_offset, layout.owned_rows.count,
        layout.owned_col_offset + layout.owned_cols.count - edgeWidth,
        edgeWidth);
    std::vector<T> northWestSend =
        packRect(layout.owned_row_offset, edgeWidth, layout.owned_col_offset,
                 edgeWidth);
    std::vector<T> northEastSend =
        packRect(layout.owned_row_offset, edgeWidth,
                 layout.owned_col_offset + layout.owned_cols.count -
                     edgeWidth,
                 edgeWidth);
    std::vector<T> southWestSend = packRect(
        layout.owned_row_offset + layout.owned_rows.count - edgeWidth,
        edgeWidth, layout.owned_col_offset, edgeWidth);
    std::vector<T> southEastSend = packRect(
        layout.owned_row_offset + layout.owned_rows.count - edgeWidth,
        edgeWidth,
        layout.owned_col_offset + layout.owned_cols.count - edgeWidth,
        edgeWidth);
    std::vector<T> northRecv(static_cast<std::size_t>(
                                 hasNorth
                                     ? checked_mul_int64_or_abort(
                                           northRows,
                                           layout.owned_cols.count,
                                           "[DACPP][MPI][OR] spatial resident halo 2D north receive size overflow")
                                     : 0),
                             T{});
    std::vector<T> southRecv(static_cast<std::size_t>(
                                 hasSouth
                                     ? checked_mul_int64_or_abort(
                                           southRows,
                                           layout.owned_cols.count,
                                           "[DACPP][MPI][OR] spatial resident halo 2D south receive size overflow")
                                     : 0),
                             T{});
    std::vector<T> westRecv(static_cast<std::size_t>(
                                hasWest
                                    ? checked_mul_int64_or_abort(
                                          layout.owned_rows.count,
                                          westCols,
                                          "[DACPP][MPI][OR] spatial resident halo 2D west receive size overflow")
                                    : 0),
                            T{});
    std::vector<T> eastRecv(static_cast<std::size_t>(
                                hasEast
                                    ? checked_mul_int64_or_abort(
                                          layout.owned_rows.count,
                                          eastCols,
                                          "[DACPP][MPI][OR] spatial resident halo 2D east receive size overflow")
                                    : 0),
                            T{});
    std::vector<T> northWestRecv(static_cast<std::size_t>(
                                     hasNorth && hasWest
                                         ? checked_mul_int64_or_abort(
                                               northRows,
                                               westCols,
                                               "[DACPP][MPI][OR] spatial resident halo 2D northwest receive size overflow")
                                         : 0),
                                 T{});
    std::vector<T> northEastRecv(static_cast<std::size_t>(
                                     hasNorth && hasEast
                                         ? checked_mul_int64_or_abort(
                                               northRows,
                                               eastCols,
                                               "[DACPP][MPI][OR] spatial resident halo 2D northeast receive size overflow")
                                         : 0),
                                 T{});
    std::vector<T> southWestRecv(static_cast<std::size_t>(
                                     hasSouth && hasWest
                                         ? checked_mul_int64_or_abort(
                                               southRows,
                                               westCols,
                                               "[DACPP][MPI][OR] spatial resident halo 2D southwest receive size overflow")
                                         : 0),
                                 T{});
    std::vector<T> southEastRecv(static_cast<std::size_t>(
                                     hasSouth && hasEast
                                         ? checked_mul_int64_or_abort(
                                               southRows,
                                               eastCols,
                                               "[DACPP][MPI][OR] spatial resident halo 2D southeast receive size overflow")
                                         : 0),
                                 T{});
    MPI_Request requests[16];
    int requestCount = 0;
    auto postRecv = [&](std::vector<T>& buffer,
                        int source,
                        int tag) {
        if (source != MPI_PROC_NULL && !buffer.empty()) {
            MPI_Irecv(buffer.data(),
                      narrow_mpi_count_or_abort(
                          static_cast<int64_t>(buffer.size()),
                          "[DACPP][MPI][OR] spatial resident halo 2D receive count exceeds MPI int range"),
                      mpiType,
                      source,
                      tag,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
    };
    auto postSend = [&](const std::vector<T>& buffer,
                        int target,
                        int tag) {
        if (target != MPI_PROC_NULL && !buffer.empty()) {
            MPI_Isend(buffer.data(),
                      narrow_mpi_count_or_abort(
                          static_cast<int64_t>(buffer.size()),
                          "[DACPP][MPI][OR] spatial resident halo 2D send count exceeds MPI int range"),
                      mpiType,
                      target,
                      tag,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
    };
    postRecv(northRecv, layout.north_rank, 4401);
    postRecv(southRecv, layout.south_rank, 4402);
    postRecv(westRecv, layout.west_rank, 4403);
    postRecv(eastRecv, layout.east_rank, 4404);
    postRecv(northWestRecv, northWestRank, 4405);
    postRecv(northEastRecv, northEastRank, 4406);
    postRecv(southWestRecv, southWestRank, 4407);
    postRecv(southEastRecv, southEastRank, 4408);
    postSend(southSend, layout.south_rank, 4401);
    postSend(northSend, layout.north_rank, 4402);
    postSend(eastSend, layout.east_rank, 4403);
    postSend(westSend, layout.west_rank, 4404);
    postSend(southEastSend, southEastRank, 4405);
    postSend(southWestSend, southWestRank, 4406);
    postSend(northEastSend, northEastRank, 4407);
    postSend(northWestSend, northWestRank, 4408);
    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
    if (hasNorth) {
        unpackRect(northRecv, topRow, northRows, layout.owned_col_offset,
                   layout.owned_cols.count);
    }
    if (hasSouth) {
        unpackRect(southRecv, bottomRow, southRows, layout.owned_col_offset,
                   layout.owned_cols.count);
    }
    if (hasWest) {
        unpackRect(westRecv, layout.owned_row_offset,
                   layout.owned_rows.count, leftCol, westCols);
    }
    if (hasEast) {
        unpackRect(eastRecv, layout.owned_row_offset,
                   layout.owned_rows.count, rightCol, eastCols);
    }
    if (hasNorth && hasWest) {
        unpackRect(northWestRecv, topRow, northRows, leftCol, westCols);
    }
    if (hasNorth && hasEast) {
        unpackRect(northEastRecv, topRow, northRows, rightCol, eastCols);
    }
    if (hasSouth && hasWest) {
        unpackRect(southWestRecv, bottomRow, southRows, leftCol, westCols);
    }
    if (hasSouth && hasEast) {
        unpackRect(southEastRecv, bottomRow, southRows, rightCol, eastCols);
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
    const bool fullRowFastPath =
        interiorRowOffset == 1 &&
        interiorColOffset == 1 &&
        inputCols == outputCols + 2 &&
        temporalBlockRows > 1;
    if (fullRowFastPath) {
        const int sendFullRowCount = narrow_mpi_count_or_abort(
            checked_mul_int64_or_abort(
                sendRowsPerNeighbor,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D full-row send width overflow"),
            "[DACPP][MPI][OR] temporal resident halo 2D full-row send width exceeds MPI int range");
        const int topRecvFullRowCount = narrow_mpi_count_or_abort(
            checked_mul_int64_or_abort(
                topRecvRows,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D full-row top receive width overflow"),
            "[DACPP][MPI][OR] temporal resident halo 2D full-row top receive width exceeds MPI int range");
        const int bottomRecvFullRowCount = narrow_mpi_count_or_abort(
            checked_mul_int64_or_abort(
                bottomRecvRows,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D full-row bottom receive width overflow"),
            "[DACPP][MPI][OR] temporal resident halo 2D full-row bottom receive width exceeds MPI int range");
        MPI_Request requests[4];
        int requestCount = 0;
        if (prev != MPI_PROC_NULL && topRecvFullRowCount > 0) {
            MPI_Irecv(local.data(),
                      topRecvFullRowCount,
                      mpiType,
                      prev,
                      4213,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
        if (next != MPI_PROC_NULL && bottomRecvFullRowCount > 0) {
            const int64_t recvOffset =
                checked_mul_int64_or_abort(
                    bottomRecvRowStart,
                    inputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D full-row bottom receive offset overflow");
            MPI_Irecv(local.data() + recvOffset,
                      bottomRecvFullRowCount,
                      mpiType,
                      next,
                      4214,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
        if (next != MPI_PROC_NULL) {
            const int64_t sendOffset =
                checked_mul_int64_or_abort(
                    layout.owned_row_offset + interiorRowOffset +
                        layout.owned_rows.count - sendRowsPerNeighbor,
                    inputCols,
                    "[DACPP][MPI][OR] temporal resident halo 2D full-row bottom send offset overflow");
            MPI_Isend(local.data() + sendOffset,
                      sendFullRowCount,
                      mpiType,
                      next,
                      4213,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
        if (prev != MPI_PROC_NULL) {
            const int64_t sendOffset = checked_mul_int64_or_abort(
                layout.owned_row_offset + interiorRowOffset,
                inputCols,
                "[DACPP][MPI][OR] temporal resident halo 2D full-row top send offset overflow");
            MPI_Isend(local.data() + sendOffset,
                      sendFullRowCount,
                      mpiType,
                      prev,
                      4214,
                      MPI_COMM_WORLD,
                      &requests[requestCount++]);
        }
        if (requestCount > 0) {
            MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        }
        return;
    }
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
std::vector<T> owned_slice_2d_spatial(
    const std::vector<T>& local,
    const ResidentHalo2DSpatialLayout& layout,
    int64_t outputRows,
    int64_t outputCols,
    int sourceRowOffset,
    int sourceColOffset) {
    (void)outputRows;
    const int64_t ownedItemCount = checked_mul_int64_or_abort(
        layout.owned_rows.count,
        layout.owned_cols.count,
        "[DACPP][MPI][OR] spatial resident halo 2D owned slice size overflow");
    std::vector<T> owned(static_cast<std::size_t>(ownedItemCount), T{});
    if (layout.owned_rows.count <= 0 || layout.owned_cols.count <= 0 ||
        layout.local_col_count <= 0 || outputCols <= 0 || local.empty()) {
        return owned;
    }
    for (int64_t row = 0; row < layout.owned_rows.count; ++row) {
        const int64_t sourceRow = layout.owned_row_offset + row + sourceRowOffset - 1;
        if (sourceRow < 0 || sourceRow >= layout.local_row_count) {
            continue;
        }
        const int64_t ownedBase = checked_mul_int64_or_abort(
            row,
            layout.owned_cols.count,
            "[DACPP][MPI][OR] spatial resident halo 2D owned slice base overflow");
        const int64_t sourceBase = checked_mul_int64_or_abort(
            sourceRow,
            layout.local_col_count,
            "[DACPP][MPI][OR] spatial resident halo 2D source slice base overflow");
        for (int64_t col = 0; col < layout.owned_cols.count; ++col) {
            const int64_t sourceCol =
                layout.owned_col_offset + col + sourceColOffset - 1;
            if (sourceCol < 0 || sourceCol >= layout.local_col_count) {
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
void gather_spatial_owned_to_root(const std::vector<T>& owned,
                                  std::vector<T>& global,
                                  int64_t outputRows,
                                  int64_t outputCols,
                                  int rank,
                                  int size,
                                  MPI_Datatype mpiType) {
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    int64_t gatheredCount = 0;
    for (int r = 0; r < size; ++r) {
        const BlockRange2D range =
            spatial_2d_owned_range(outputRows, outputCols, r, size);
        const int64_t rankCount = checked_mul_int64_or_abort(
            range.rows.count,
            range.cols.count,
            "[DACPP][MPI][OR] spatial resident halo 2D gather rank size overflow");
        counts[static_cast<std::size_t>(r)] = narrow_mpi_count_or_abort(
            rankCount,
            "[DACPP][MPI][OR] spatial resident halo 2D gather rank count exceeds MPI int range");
        displs[static_cast<std::size_t>(r)] = narrow_mpi_count_or_abort(
            gatheredCount,
            "[DACPP][MPI][OR] spatial resident halo 2D gather displacement exceeds MPI int range");
        gatheredCount += rankCount;
    }
    std::vector<T> gathered;
    if (rank == 0) {
        gathered.resize(static_cast<std::size_t>(gatheredCount));
    }
    const int sendCount = narrow_mpi_count_or_abort(
        static_cast<int64_t>(owned.size()),
        "[DACPP][MPI][OR] spatial resident halo 2D gather send count exceeds MPI int range");
    MPI_Gatherv(owned.empty() ? nullptr : owned.data(),
                sendCount,
                mpiType,
                rank == 0 && !gathered.empty() ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                mpiType,
                0,
                MPI_COMM_WORLD);
    if (rank != 0) {
        return;
    }
    global.assign(static_cast<std::size_t>(
                      checked_mul_int64_or_abort(
                          outputRows,
                          outputCols,
                          "[DACPP][MPI][OR] spatial resident halo 2D global gather size overflow")),
                  T{});
    for (int r = 0; r < size; ++r) {
        const BlockRange2D range =
            spatial_2d_owned_range(outputRows, outputCols, r, size);
        for (int64_t row = 0; row < range.rows.count; ++row) {
            const int64_t globalRow = range.rows.begin + row;
            const int64_t globalBase = checked_mul_int64_or_abort(
                globalRow,
                outputCols,
                "[DACPP][MPI][OR] spatial resident halo 2D global row overflow");
            const int64_t gatheredBase =
                static_cast<int64_t>(displs[static_cast<std::size_t>(r)]) +
                checked_mul_int64_or_abort(
                    row,
                    range.cols.count,
                    "[DACPP][MPI][OR] spatial resident halo 2D gathered row overflow");
            for (int64_t col = 0; col < range.cols.count; ++col) {
                const int64_t globalCol = range.cols.begin + col;
                const std::size_t globalIdx =
                    static_cast<std::size_t>(globalBase + globalCol);
                const std::size_t gatheredIdx =
                    static_cast<std::size_t>(gatheredBase + col);
                if (globalIdx < global.size() &&
                    gatheredIdx < gathered.size()) {
                    global[globalIdx] = gathered[gatheredIdx];
                }
            }
        }
    }
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
