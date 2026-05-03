#ifndef DACPP_MPI_STENCIL_LAYOUT_H
#define DACPP_MPI_STENCIL_LAYOUT_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "StencilTypes.h"

namespace dacpp {
namespace mpi {

inline void init_gathered_index_layout(GatheredIndexLayout& layout,
                                       const std::vector<int64_t>& local_globals,
                                       int mpi_rank,
                                       int mpi_size,
                                       MPI_Comm comm = MPI_COMM_WORLD) {
    layout.local_count = static_cast<int>(local_globals.size());
    layout.counts.clear();
    layout.displs.clear();
    layout.byte_counts.clear();
    layout.byte_displs.clear();
    layout.globals.clear();

    if (mpi_rank == 0) {
        layout.counts.resize(mpi_size);
        layout.displs.resize(mpi_size);
    }

    MPI_Gather(&layout.local_count, 1, MPI_INT,
               mpi_rank == 0 ? layout.counts.data() : nullptr, 1, MPI_INT,
               0, comm);

    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int rank = 0; rank < mpi_size; ++rank) {
            layout.displs[rank] = current_displ;
            current_displ += layout.counts[rank];
        }
        layout.globals.resize(current_displ);
    }

    MPI_Gatherv(const_cast<int64_t*>(local_globals.data()), layout.local_count,
                MPI_LONG_LONG,
                mpi_rank == 0 ? layout.globals.data() : nullptr,
                mpi_rank == 0 ? layout.counts.data() : nullptr,
                mpi_rank == 0 ? layout.displs.data() : nullptr,
                MPI_LONG_LONG, 0, comm);
}

inline void init_all_rank_index_layout(AllRankIndexLayout& layout,
                                       const std::vector<int64_t>& local_globals,
                                       int mpi_rank,
                                       int mpi_size,
                                       MPI_Comm comm = MPI_COMM_WORLD) {
    (void)mpi_rank;
    layout.local_count = static_cast<int>(local_globals.size());
    layout.counts.assign(static_cast<std::size_t>(mpi_size), 0);
    layout.displs.assign(static_cast<std::size_t>(mpi_size), 0);
    layout.globals.clear();

    MPI_Allgather(&layout.local_count, 1, MPI_INT,
                  layout.counts.data(), 1, MPI_INT, comm);

    int current_displ = 0;
    for (int rank = 0; rank < mpi_size; ++rank) {
        layout.displs[static_cast<std::size_t>(rank)] = current_displ;
        current_displ += layout.counts[static_cast<std::size_t>(rank)];
    }
    layout.globals.resize(static_cast<std::size_t>(current_displ));

    MPI_Allgatherv(const_cast<int64_t*>(local_globals.data()), layout.local_count,
                   MPI_LONG_LONG, layout.globals.data(), layout.counts.data(),
                   layout.displs.data(), MPI_LONG_LONG, comm);
}

inline void init_layout_byte_counts(GatheredIndexLayout& layout,
                                    std::size_t value_size) {
    layout.byte_counts = layout.counts;
    layout.byte_displs = layout.displs;
    for (std::size_t idx = 0; idx < layout.byte_counts.size(); ++idx) {
        layout.byte_counts[idx] *= static_cast<int>(value_size);
        layout.byte_displs[idx] *= static_cast<int>(value_size);
    }
}

inline bool validate_unique_writers(const AllRankIndexLayout& layout,
                                    int mpi_size,
                                    std::string* reason = nullptr) {
    std::unordered_map<int64_t, int> owner_by_global;
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = layout.counts[static_cast<std::size_t>(rank)];
        const int displ = layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            const int64_t global_idx =
                layout.globals[static_cast<std::size_t>(displ + idx)];
            auto [it, inserted] = owner_by_global.emplace(global_idx, rank);
            if (!inserted && it->second != rank) {
                if (reason) {
                    *reason = "global " + std::to_string(global_idx) +
                              " written by ranks " +
                              std::to_string(it->second) + " and " +
                              std::to_string(rank);
                }
                return false;
            }
        }
    }
    return true;
}

}  // namespace mpi
}  // namespace dacpp

#endif
