#ifndef DACPP_MPI_STENCIL_EXCHANGE_RUNTIME_H
#define DACPP_MPI_STENCIL_EXCHANGE_RUNTIME_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

#include "StencilExchangePlan.h"

namespace dacpp {
namespace mpi {

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
inline void scatter_values_by_slots_parallel_into(
    const std::vector<T>& source,
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<T>& target,
    sycl::queue& q,
    std::size_t threshold = 1 << 18) {
    const std::size_t count = std::min(source_slots.size(), target_slots.size());
    if (count == 0) {
        return;
    }

    if (count < threshold || source.data() == target.data()) {
        for (std::size_t idx = 0; idx < count; ++idx) {
            const int32_t source_slot = source_slots[idx];
            const int32_t target_slot = target_slots[idx];
            if (source_slot < 0 || target_slot < 0 ||
                static_cast<std::size_t>(source_slot) >= source.size() ||
                static_cast<std::size_t>(target_slot) >= target.size()) {
                continue;
            }
            target[static_cast<std::size_t>(target_slot)] =
                source[static_cast<std::size_t>(source_slot)];
        }
        return;
    }

    {
        sycl::buffer<T, 1> source_buf(
            const_cast<T*>(source.data()),
            sycl::range<1>(source.size()));
        sycl::buffer<int32_t, 1> source_slots_buf(
            const_cast<int32_t*>(source_slots.data()),
            sycl::range<1>(source_slots.size()));
        sycl::buffer<int32_t, 1> target_slots_buf(
            const_cast<int32_t*>(target_slots.data()),
            sycl::range<1>(target_slots.size()));
        sycl::buffer<T, 1> target_buf(
            target.data(),
            sycl::range<1>(target.size()));

        q.submit([&](sycl::handler& h) {
            auto source_acc =
                source_buf.template get_access<sycl::access::mode::read>(h);
            auto source_slots_acc =
                source_slots_buf.template get_access<sycl::access::mode::read>(h);
            auto target_slots_acc =
                target_slots_buf.template get_access<sycl::access::mode::read>(h);
            auto target_acc =
                target_buf.template get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                const int32_t source_slot = source_slots_acc[i];
                const int32_t target_slot = target_slots_acc[i];
                if (source_slot < 0 || target_slot < 0 ||
                    static_cast<std::size_t>(source_slot) >= source_acc.get_count() ||
                    static_cast<std::size_t>(target_slot) >= target_acc.get_count()) {
                    return;
                }
                target_acc[static_cast<std::size_t>(target_slot)] =
                    source_acc[static_cast<std::size_t>(source_slot)];
            });
        });
        q.wait();
    }
}

template <typename T>
inline void scatter_values_by_slots_parallel_into(
    const std::vector<T>& source,
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<T>& target,
    std::size_t threshold = 1 << 18) {
    sycl::queue q(sycl::default_selector_v);
    scatter_values_by_slots_parallel_into(source, source_slots, target_slots,
                                          target, q, threshold);
}

template <typename T>
inline std::size_t halo_value_count(
    const std::vector<SlotSpan>& spans) {
    std::size_t count = 0;
    for (const SlotSpan& span : spans) {
        if (span.count > 0) {
            count += static_cast<std::size_t>(span.count);
        }
    }
    return count;
}

template <typename T>
inline void prepare_halo_exchange_runtime(const HaloExchangePlan& plan,
                                          HaloExchangeRuntime<T>& runtime) {
    runtime.send_buffers.resize(plan.send_transfers.size());
    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        runtime.send_buffers[idx].resize(
            halo_value_count<T>(plan.send_transfers[idx].local_spans));
    }
    runtime.recv_buffers.resize(plan.recv_transfers.size());
    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        runtime.recv_buffers[idx].resize(
            halo_value_count<T>(plan.recv_transfers[idx].local_spans));
    }
    runtime.requests.resize(plan.send_transfers.size() + plan.recv_transfers.size());
}

template <typename T>
inline void pack_values_by_spans_into(const std::vector<T>& local_data,
                                      const std::vector<SlotSpan>& spans,
                                      std::vector<T>& values) {
    values.resize(halo_value_count<T>(spans));
    std::size_t out_idx = 0;
    for (const SlotSpan& span : spans) {
        if (span.count <= 0) {
            continue;
        }
        if (span.stride == 1 &&
            span.begin >= 0 &&
            static_cast<std::size_t>(span.begin) + static_cast<std::size_t>(span.count) <=
                local_data.size()) {
            std::copy_n(local_data.begin() + span.begin, span.count,
                        values.begin() + out_idx);
            out_idx += static_cast<std::size_t>(span.count);
            continue;
        }
        for (int32_t offset = 0; offset < span.count; ++offset) {
            values[out_idx++] = local_data[static_cast<std::size_t>(
                span.begin + offset * span.stride)];
        }
    }
}

template <typename T>
inline void unpack_values_by_spans_into(const std::vector<T>& values,
                                        const std::vector<SlotSpan>& spans,
                                        std::vector<T>& target) {
    std::size_t in_idx = 0;
    for (const SlotSpan& span : spans) {
        if (span.count <= 0) {
            continue;
        }
        if (span.stride == 1 &&
            span.begin >= 0 &&
            in_idx + static_cast<std::size_t>(span.count) <= values.size() &&
            static_cast<std::size_t>(span.begin) + static_cast<std::size_t>(span.count) <=
                target.size()) {
            std::copy_n(values.begin() + in_idx, span.count,
                        target.begin() + span.begin);
            in_idx += static_cast<std::size_t>(span.count);
            continue;
        }
        for (int32_t offset = 0; offset < span.count; ++offset) {
            if (in_idx >= values.size()) {
                return;
            }
            const int32_t slot = span.begin + offset * span.stride;
            if (slot >= 0 && static_cast<std::size_t>(slot) < target.size()) {
                target[static_cast<std::size_t>(slot)] = values[in_idx];
            }
            ++in_idx;
        }
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

template <typename T>
inline void exchange_values_by_halo_spans(const std::vector<T>& send_source,
                                          const HaloExchangePlan& plan,
                                          HaloExchangeRuntime<T>& runtime,
                                          std::vector<T>& recv_target,
                                          MPI_Comm comm = MPI_COMM_WORLD) {
    if (!plan.supported) {
        return;
    }

    if (runtime.send_buffers.size() != plan.send_transfers.size() ||
        runtime.recv_buffers.size() != plan.recv_transfers.size() ||
        runtime.requests.size() !=
            plan.send_transfers.size() + plan.recv_transfers.size()) {
        prepare_halo_exchange_runtime(plan, runtime);
    }

    const MPI_Datatype mpi_type = mpi_datatype_for_value<T>();
    int request_count = 0;

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        const auto& transfer = plan.recv_transfers[idx];
        auto& recv_buffer = runtime.recv_buffers[idx];
        const std::size_t recv_count = halo_value_count<T>(transfer.local_spans);
        if (recv_buffer.size() != recv_count) {
            recv_buffer.resize(recv_count);
        }
        if (recv_buffer.empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Irecv(reinterpret_cast<unsigned char*>(recv_buffer.data()),
                      mpi_payload_count_for_values<T>(recv_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Irecv(recv_buffer.data(),
                      mpi_payload_count_for_values<T>(recv_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        runtime.requests[static_cast<std::size_t>(request_count++)] = req;
    }

    for (std::size_t idx = 0; idx < plan.send_transfers.size(); ++idx) {
        const auto& transfer = plan.send_transfers[idx];
        auto& send_buffer = runtime.send_buffers[idx];
        const std::size_t send_count = halo_value_count<T>(transfer.local_spans);
        if (send_buffer.size() != send_count) {
            send_buffer.resize(send_count);
        }
        pack_values_by_spans_into(send_source, transfer.local_spans, send_buffer);
        if (send_buffer.empty()) {
            continue;
        }
        MPI_Request req{};
        if (uses_byte_transport_for_value<T>()) {
            MPI_Isend(reinterpret_cast<unsigned char*>(send_buffer.data()),
                      mpi_payload_count_for_values<T>(send_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        } else {
            MPI_Isend(send_buffer.data(),
                      mpi_payload_count_for_values<T>(send_buffer.size()),
                      mpi_type, transfer.peer_rank, 0, comm, &req);
        }
        runtime.requests[static_cast<std::size_t>(request_count++)] = req;
    }

    if (request_count > 0) {
        MPI_Waitall(request_count, runtime.requests.data(), MPI_STATUSES_IGNORE);
    }

    for (std::size_t idx = 0; idx < plan.recv_transfers.size(); ++idx) {
        unpack_values_by_spans_into(runtime.recv_buffers[idx],
                                    plan.recv_transfers[idx].local_spans,
                                    recv_target);
    }
}

template <typename T>
inline void exchange_values_by_halo_spans(const std::vector<T>& send_source,
                                          const HaloExchangePlan& plan,
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
        recv_buffers[idx].resize(halo_value_count<T>(transfer.local_spans));
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
        pack_values_by_spans_into(send_source, transfer.local_spans,
                                  send_buffers[idx]);
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
        unpack_values_by_spans_into(recv_buffers[idx],
                                    plan.recv_transfers[idx].local_spans,
                                    recv_target);
    }
}

template <typename T>
inline void publish_local_writes_with_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        pack_values_by_slots_parallel_into(local_write_source, local_write_slots,
                                           local_write_values);
        unpack_values_by_slots_into(local_write_values, local_target_slots,
                                    local_cache);
    } else {
        local_write_values.clear();
    }

    if (!plan.supported) {
        return;
    }

    exchange_values_by_slots(local_write_source, plan, local_cache, comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        pack_values_by_slots_parallel_into(local_write_source, local_write_slots,
                                           local_write_values);
        scatter_values_by_slots_parallel_into(
            local_write_source, local_write_slots, local_target_slots,
            local_cache, q);
    } else {
        local_write_values.clear();
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, local_cache,
                                      comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    std::vector<T>& local_write_values,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_halo_or_exchange(
        local_write_source, local_write_slots, local_target_slots, local_cache,
        local_write_values, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_write_slots.empty()) {
        scatter_values_by_slots_parallel_into(
            local_write_source, local_write_slots, local_target_slots,
            local_cache, q);
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, local_cache,
                                      comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_halo_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_halo_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots, local_cache,
        exchange_plan, halo_plan, q, comm);
}

}  // namespace mpi
}  // namespace dacpp

#endif
