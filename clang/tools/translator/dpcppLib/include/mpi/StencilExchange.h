#ifndef DACPP_MPI_STENCIL_EXCHANGE_H
#define DACPP_MPI_STENCIL_EXCHANGE_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

#include "MpiTypes.h"
#include "PackMap.h"
#include "StencilLayout.h"

namespace dacpp {
namespace mpi {

inline PackMap make_dense_cover_pack(std::size_t dense_size) {
    std::vector<int64_t> globals(dense_size);
    for (std::size_t idx = 0; idx < dense_size; ++idx) {
        globals[idx] = static_cast<int64_t>(idx);
    }
    PackMap pack = make_pack_map_from_globals(
        std::move(globals), PackLayoutKind::DenseCover, dense_size);
    return pack;
}

inline void build_local_slots_for_globals(const PackMap& pack,
                                          std::vector<int32_t>& local_slots) {
    const std::vector<int64_t>& globals =
        pack.writeback_globals.empty() ? pack.globals : pack.writeback_globals;
    local_slots.clear();
    local_slots.reserve(globals.size());
    for (int64_t global_idx : globals) {
        local_slots.push_back(
            lookup_local_slot_or_throw(
                pack, global_idx, "build_local_slots_for_globals"));
    }
}

inline ExchangePlan build_exchange_plan_from_layouts(
    const PackMap& writer_pack,
    const AllRankIndexLayout& writer_layout,
    const PackMap& reader_pack,
    const AllRankIndexLayout& reader_layout,
    int mpi_rank,
    int mpi_size) {
    ExchangePlan plan;
    std::string unique_writer_reason;
    if (!validate_unique_writers(writer_layout, mpi_size, &unique_writer_reason)) {
        plan.supported = false;
        plan.unsupported_reason =
            "writer layout has overlapping ownership: " + unique_writer_reason;
        return plan;
    }

    std::unordered_map<int64_t, int> writer_owner;
    writer_owner.reserve(writer_layout.globals.size());
    for (int rank = 0; rank < mpi_size; ++rank) {
        const int count = writer_layout.counts[static_cast<std::size_t>(rank)];
        const int displ = writer_layout.displs[static_cast<std::size_t>(rank)];
        for (int idx = 0; idx < count; ++idx) {
            writer_owner.emplace(
                writer_layout.globals[static_cast<std::size_t>(displ + idx)], rank);
        }
    }

    std::unordered_map<int, PeerSlotExchange> recv_by_peer;
    const int local_read_count =
        reader_layout.counts.empty()
            ? 0
            : reader_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_read_displ =
        reader_layout.displs.empty()
            ? 0
            : reader_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_read_count; ++idx) {
        const int64_t global_idx =
            reader_layout.globals[static_cast<std::size_t>(local_read_displ + idx)];
        const auto ownerIt = writer_owner.find(global_idx);
        if (ownerIt == writer_owner.end()) {
            continue;
        }
        const int owner_rank = ownerIt->second;
        if (owner_rank == mpi_rank) {
            continue;
        }
        const int32_t slot =
            try_lookup_local_slot(reader_pack, global_idx);
        if (slot < 0) {
            plan.supported = false;
            plan.unsupported_reason =
                "reader pack missing global " + std::to_string(global_idx);
            return plan;
        }
        auto& transfer = recv_by_peer[owner_rank];
        transfer.peer_rank = owner_rank;
        transfer.globals.push_back(global_idx);
        transfer.local_slots.push_back(slot);
    }

    std::unordered_map<int, PeerSlotExchange> send_by_peer;
    const int local_write_count =
        writer_layout.counts.empty()
            ? 0
            : writer_layout.counts[static_cast<std::size_t>(mpi_rank)];
    const int local_write_displ =
        writer_layout.displs.empty()
            ? 0
            : writer_layout.displs[static_cast<std::size_t>(mpi_rank)];
    for (int idx = 0; idx < local_write_count; ++idx) {
        const int64_t global_idx =
            writer_layout.globals[static_cast<std::size_t>(local_write_displ + idx)];
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank == mpi_rank) {
                continue;
            }
            const int peer_count =
                reader_layout.counts[static_cast<std::size_t>(rank)];
            const int peer_displ =
                reader_layout.displs[static_cast<std::size_t>(rank)];
            const auto begin =
                reader_layout.globals.begin() + peer_displ;
            const auto end = begin + peer_count;
            if (!std::binary_search(begin, end, global_idx)) {
                continue;
            }
            const int32_t slot = try_lookup_local_slot(writer_pack, global_idx);
            if (slot < 0) {
                plan.supported = false;
                plan.unsupported_reason =
                    "writer pack missing global " + std::to_string(global_idx);
                return plan;
            }
            auto& transfer = send_by_peer[rank];
            transfer.peer_rank = rank;
            transfer.globals.push_back(global_idx);
            transfer.local_slots.push_back(slot);
        }
    }

    plan.supported = true;
    for (auto& entry : send_by_peer) {
        plan.send_transfers.push_back(std::move(entry.second));
    }
    for (auto& entry : recv_by_peer) {
        plan.recv_transfers.push_back(std::move(entry.second));
    }
    std::sort(plan.send_transfers.begin(), plan.send_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    std::sort(plan.recv_transfers.begin(), plan.recv_transfers.end(),
              [](const PeerSlotExchange& lhs, const PeerSlotExchange& rhs) {
                  return lhs.peer_rank < rhs.peer_rank;
              });
    return plan;
}

template <typename T>
inline void pack_values_by_slots_into(const std::vector<T>& local_data,
                                      const std::vector<int32_t>& local_slots,
                                      std::vector<T>& values) {
    values.resize(local_slots.size());
    for (std::size_t idx = 0; idx < local_slots.size(); ++idx) {
        values[idx] = local_data[static_cast<std::size_t>(local_slots[idx])];
    }
}

template <typename T>
inline void unpack_values_by_slots_into(const std::vector<T>& values,
                                        const std::vector<int32_t>& local_slots,
                                        std::vector<T>& target) {
    const std::size_t count = std::min(values.size(), local_slots.size());
    for (std::size_t idx = 0; idx < count; ++idx) {
        const int32_t slot = local_slots[idx];
        if (slot < 0 ||
            static_cast<std::size_t>(slot) >= target.size()) {
            continue;
        }
        target[static_cast<std::size_t>(slot)] = values[idx];
    }
}

template <typename T>
inline void pack_values_by_slots_parallel_into(
    const std::vector<T>& local_data,
    const std::vector<int32_t>& local_slots,
    std::vector<T>& values,
    std::size_t threshold = 1 << 18) {
    if (local_slots.size() < threshold) {
        pack_values_by_slots_into(local_data, local_slots, values);
        return;
    }

    values.resize(local_slots.size());
    if (local_slots.empty()) {
        return;
    }

    sycl::queue q(sycl::default_selector_v);
    {
        sycl::buffer<T, 1> local_buf(
            const_cast<T*>(local_data.data()),
            sycl::range<1>(local_data.size()));
        sycl::buffer<int32_t, 1> slots_buf(
            const_cast<int32_t*>(local_slots.data()),
            sycl::range<1>(local_slots.size()));
        sycl::buffer<T, 1> values_buf(
            values.data(),
            sycl::range<1>(values.size()));

        q.submit([&](sycl::handler& h) {
            auto local_acc = local_buf.template get_access<sycl::access::mode::read>(h);
            auto slots_acc = slots_buf.template get_access<sycl::access::mode::read>(h);
            auto values_acc = values_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(values.size()), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                values_acc[i] = local_acc[static_cast<std::size_t>(slots_acc[i])];
            });
        });
        q.wait();
    }
}

template <typename T>
inline void exchange_values_by_slots(const std::vector<T>& send_source,
                                     const ExchangePlan& plan,
                                     std::vector<T>& recv_target,
                                     MPI_Comm comm = MPI_COMM_WORLD) {
    if (!plan.supported) {
        return;
    }

    const MPI_Datatype mpi_type = mpi_datatype_for_value<T>();
    std::vector<std::vector<T>> send_buffers(plan.send_transfers.size());
    std::vector<std::vector<T>> recv_buffers(plan.recv_transfers.size());
    std::vector<MPI_Request> requests;
    requests.reserve(plan.send_transfers.size() + plan.recv_transfers.size());

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        const auto& transfer = plan.recv_transfers[idx];
        recv_buffers[idx].resize(transfer.local_slots.size());
        if (recv_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Irecv(reinterpret_cast<unsigned char*>(recv_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Irecv(recv_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(recv_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        const auto& transfer = plan.send_transfers[idx];
        pack_values_by_slots_into(send_source, transfer.local_slots, send_buffers[idx]);
        if (send_buffers[idx].empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Isend(reinterpret_cast<unsigned char*>(send_buffers[idx].data()),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Isend(send_buffers[idx].data(),
                      mpi_payload_count_for_values<T>(send_buffers[idx].size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        requests.push_back(req);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
    }

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        unpack_values_by_slots_into(recv_buffers[idx],
                                    plan.recv_transfers[idx].local_slots,
                                    recv_target);
    }
}

}  // namespace mpi
}  // namespace dacpp

#endif
