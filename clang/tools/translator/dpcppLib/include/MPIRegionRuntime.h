#ifndef DACPP_MPI_REGION_RUNTIME_H
#define DACPP_MPI_REGION_RUNTIME_H

#ifndef DACPP_MPI_PLANNER_H
#error "Include MPIPlanner.h instead of MPIRegionRuntime.h directly."
#endif

namespace dacpp {
namespace mpi {

template <typename T>
struct RegionParamState {
    AccessPattern pattern;
    PackMap item_pack;
    PackMap runtime_pack;
    std::vector<int32_t> slots;
    std::vector<T> local;
    std::unique_ptr<sycl::buffer<T, 1>> buf;
    std::unique_ptr<sycl::buffer<int32_t, 1>> slots_buf;
    std::vector<int32_t> global_to_local;
    std::unique_ptr<sycl::buffer<int32_t, 1>> global_to_local_buf;
    int partition_size = 0;
    int cols = 0;
    ParamHalo halo;
};

namespace detail {

inline bool has_preconfigured_runtime_pack(const PackMap& pack) {
    return pack.layout_kind == PackLayoutKind::DenseCover ||
           !pack.globals.empty() || !pack.g2l.empty() || !pack.segments.empty() ||
           !pack.writeback_globals.empty() || !pack.writeback_segments.empty();
}

inline PackMap build_pack_for_mode(ItemRange range,
                                   const AccessPattern& pattern,
                                   AccessMode mode) {
    switch (mode) {
    case AccessMode::Read:
        return build_input_pack_map(range, pattern);
    case AccessMode::Write:
        return build_output_pack_map(range, pattern);
    case AccessMode::ReadWrite:
        return build_rw_pack_map(range, pattern);
    }
    return build_input_pack_map(range, pattern);
}

inline int64_t total_items_from_bind_split_sizes(
    const std::vector<int64_t>& bind_split_sizes) {
    int64_t total_items = 1;
    for (int64_t split_size : bind_split_sizes) {
        total_items *= split_size;
    }
    return total_items;
}

template <typename T>
inline int payload_count_for(int elem_count, MPI_Datatype mpi_type) {
    return mpi_type == MPI_BYTE
               ? static_cast<int>(elem_count * static_cast<int>(sizeof(T)))
               : elem_count;
}

inline std::size_t dense_extent_for_pattern(const AccessPattern& pattern) {
    std::size_t dense_extent = 1;
    for (int dim : pattern.data_info.dimLength) {
        dense_extent *= static_cast<std::size_t>(std::max(0, dim));
    }
    return dense_extent;
}

}  // namespace detail

template <typename Tensor, typename T>
inline void init_region_param_storage(Tensor& tensor,
                                      RegionParamState<T>& state,
                                      ItemRange item_range,
                                      int mpi_rank,
                                      int mpi_size,
                                      bool needs_init_scatter) {
    state.item_pack =
        detail::build_pack_for_mode(item_range, state.pattern, state.pattern.mode);
    if (!detail::has_preconfigured_runtime_pack(state.runtime_pack)) {
        state.runtime_pack = state.item_pack;
    }

    state.slots =
        build_item_slots(item_range, state.pattern, state.runtime_pack);
    state.local.resize(state.runtime_pack.globals.size());

    if (needs_init_scatter) {
        if (state.runtime_pack.layout_kind == PackLayoutKind::DenseCover) {
            if (mpi_rank == 0) {
                std::vector<T> global_values;
                tensor.tensor2Array(global_values);
                if (global_values.size() == state.local.size()) {
                    std::copy(global_values.begin(), global_values.end(),
                              state.local.begin());
                } else {
                    throw std::runtime_error(
                        "MPI dense init(runtime_pack): size mismatch for " +
                        state.pattern.name + " — global_size=" +
                        std::to_string(global_values.size()) +
                        ", local_size=" +
                        std::to_string(state.local.size()));
                }
            }
            if (!state.local.empty()) {
                MPI_Bcast(state.local.data(),
                          static_cast<int>(state.local.size() * sizeof(T)),
                          MPI_BYTE, 0, MPI_COMM_WORLD);
            }
        } else {
            const int64_t total_items = detail::total_items_from_bind_split_sizes(
                state.pattern.bind_split_sizes);
            int recv_count = 0;
            std::vector<int> sendcounts;
            std::vector<int> displs;
            std::vector<T> sendbuf;
            if (mpi_rank == 0) {
                sendcounts.resize(mpi_size);
                displs.resize(mpi_size);
                int current_displ = 0;
                std::vector<T> global_values;
                tensor.tensor2Array(global_values);
                for (int rank = 0; rank < mpi_size; ++rank) {
                    const auto rank_range =
                        get_rank_item_range(total_items, rank, mpi_size);
                    const auto rank_pack = detail::build_pack_for_mode(
                        rank_range, state.pattern, state.pattern.mode);
                    auto rank_values =
                        pack_values_by_globals(global_values, rank_pack.globals);
                    const int rank_count =
                        static_cast<int>(rank_values.size());
                    sendcounts[rank] = rank_count;
                    displs[rank] = current_displ;
                    current_displ += rank_count;
                    sendbuf.insert(sendbuf.end(), rank_values.begin(),
                                   rank_values.end());
                }
            }
            MPI_Scatter(mpi_rank == 0 ? sendcounts.data() : nullptr, 1, MPI_INT,
                        &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
            state.local.resize(static_cast<std::size_t>(recv_count));

            std::vector<int> sc_bytes;
            std::vector<int> ds_bytes;
            if (mpi_rank == 0) {
                sc_bytes = sendcounts;
                ds_bytes = displs;
                for (int rank = 0; rank < mpi_size; ++rank) {
                    sc_bytes[rank] *= static_cast<int>(sizeof(T));
                    ds_bytes[rank] *= static_cast<int>(sizeof(T));
                }
            }
            MPI_Scatterv(
                mpi_rank == 0 ? sendbuf.data() : nullptr,
                mpi_rank == 0 ? sc_bytes.data() : nullptr,
                mpi_rank == 0 ? ds_bytes.data() : nullptr, MPI_BYTE,
                state.local.data(),
                static_cast<int>(state.local.size() * sizeof(T)), MPI_BYTE, 0,
                MPI_COMM_WORLD);
        }
    }

    state.partition_size =
        static_cast<int>(partition_element_count(state.pattern));
    state.cols =
        state.pattern.partition_shape.size() > 1 ? state.pattern.partition_shape[1]
                                                 : 0;

    if (!state.local.empty()) {
        state.buf = std::make_unique<sycl::buffer<T, 1>>(
            state.local.data(), sycl::range<1>(state.local.size()));
    }
    if (!state.slots.empty()) {
        state.slots_buf = std::make_unique<sycl::buffer<int32_t, 1>>(
            state.slots.data(), sycl::range<1>(state.slots.size()));
    }
}

template <typename T>
inline void init_region_lookup(RegionParamState<T>& state,
                               const char* where) {
    const std::size_t dense_extent =
        detail::dense_extent_for_pattern(state.pattern);
    state.global_to_local =
        build_dense_global_to_local_lut(state.runtime_pack, dense_extent, where);
    if (!state.global_to_local.empty()) {
        state.global_to_local_buf =
            std::make_unique<sycl::buffer<int32_t, 1>>(
                state.global_to_local.data(),
                sycl::range<1>(state.global_to_local.size()));
    } else {
        state.global_to_local_buf.reset();
    }
}

template <typename T>
inline void sync_buffer_to_local(RegionParamState<T>& state) {
    if (!state.buf) {
        return;
    }
    sycl::host_accessor acc(*state.buf, sycl::read_only);
    for (std::size_t idx = 0; idx < state.local.size(); ++idx) {
        state.local[idx] = acc[idx];
    }
}

template <typename T>
inline void sync_local_to_buffer(RegionParamState<T>& state) {
    if (!state.buf) {
        return;
    }
    sycl::host_accessor acc(*state.buf, sycl::write_only);
    for (std::size_t idx = 0; idx < state.local.size(); ++idx) {
        acc[idx] = state.local[idx];
    }
}

template <typename T>
inline void exchange_halo_buffered(RegionParamState<T>& state,
                                   void* mpi_type_ptr) {
    sync_buffer_to_local(state);
    exchangeHalo(state.local, state.halo, mpi_type_ptr);
    sync_local_to_buffer(state);
}

template <typename Tensor, typename T>
inline void writeback_region_output(Tensor& tensor,
                                    RegionParamState<T>& state,
                                    bool broadcast_to_all,
                                    int mpi_rank,
                                    int mpi_size,
                                    MPI_Datatype mpi_type) {
    sync_buffer_to_local(state);

    auto writeback_values = build_writeback_values(state.local, state.runtime_pack);
    const auto& writeback_globals = state.runtime_pack.writeback_globals.empty()
                                        ? state.runtime_pack.globals
                                        : state.runtime_pack.writeback_globals;

    const int send_count = static_cast<int>(writeback_globals.size());
    std::vector<int> recvcounts;
    std::vector<int> recvdispls;
    std::vector<int64_t> global_recv_globals;
    std::vector<T> global_recv_values;
    if (mpi_rank == 0) {
        recvcounts.resize(mpi_size);
        recvdispls.resize(mpi_size);
    }
    MPI_Gather(&send_count, 1, MPI_INT,
               mpi_rank == 0 ? recvcounts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int rank = 0; rank < mpi_size; ++rank) {
            recvdispls[rank] = current_displ;
            current_displ += recvcounts[rank];
        }
        global_recv_globals.resize(static_cast<std::size_t>(current_displ));
        global_recv_values.resize(static_cast<std::size_t>(current_displ));
    }

    MPI_Gatherv(const_cast<int64_t*>(writeback_globals.data()), send_count,
                MPI_LONG_LONG,
                mpi_rank == 0 ? global_recv_globals.data() : nullptr,
                mpi_rank == 0 ? recvcounts.data() : nullptr,
                mpi_rank == 0 ? recvdispls.data() : nullptr, MPI_LONG_LONG, 0,
                MPI_COMM_WORLD);

    if (mpi_type == MPI_BYTE) {
        std::vector<int> recvcounts_bytes;
        std::vector<int> recvdispls_bytes;
        if (mpi_rank == 0) {
            recvcounts_bytes = recvcounts;
            recvdispls_bytes = recvdispls;
            for (int rank = 0; rank < mpi_size; ++rank) {
                recvcounts_bytes[rank] *= static_cast<int>(sizeof(T));
                recvdispls_bytes[rank] *= static_cast<int>(sizeof(T));
            }
        }
        MPI_Gatherv(writeback_values.data(),
                    detail::payload_count_for<T>(send_count, mpi_type),
                    mpi_type,
                    mpi_rank == 0 ? global_recv_values.data() : nullptr,
                    mpi_rank == 0 ? recvcounts_bytes.data() : nullptr,
                    mpi_rank == 0 ? recvdispls_bytes.data() : nullptr,
                    mpi_type, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(writeback_values.data(), send_count, mpi_type,
                    mpi_rank == 0 ? global_recv_values.data() : nullptr,
                    mpi_rank == 0 ? recvcounts.data() : nullptr,
                    mpi_rank == 0 ? recvdispls.data() : nullptr, mpi_type, 0,
                    MPI_COMM_WORLD);
    }

    std::vector<T> synced_values;
    if (mpi_rank == 0) {
        std::vector<T> global_output;
        tensor.tensor2Array(global_output);
        apply_writeback_by_globals(global_recv_values, global_recv_globals,
                                   global_output);
        tensor.array2Tensor(global_output);
        if (broadcast_to_all) {
            synced_values = global_output;
        }
    }

    if (!broadcast_to_all) {
        return;
    }

    int synced_count = 0;
    if (mpi_rank == 0) {
        synced_count = static_cast<int>(synced_values.size());
    }
    MPI_Bcast(&synced_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_values.resize(static_cast<std::size_t>(synced_count));
    }
    if (synced_count > 0) {
        if (mpi_type == MPI_BYTE) {
            MPI_Bcast(synced_values.data(),
                      detail::payload_count_for<T>(synced_count, mpi_type),
                      mpi_type, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(synced_values.data(), synced_count, mpi_type, 0,
                      MPI_COMM_WORLD);
        }
    }
    if (mpi_rank != 0) {
        tensor.array2Tensor(synced_values);
    }
}

}  // namespace mpi
}  // namespace dacpp

#endif
