#ifndef DACPP_MPI_WAVE_EXCHANGE_SPECIALIZATION_H
#define DACPP_MPI_WAVE_EXCHANGE_SPECIALIZATION_H

#include "StencilExchange.h"

namespace dacpp {
namespace mpi {

inline bool build_span_pairs_from_slots(
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<SlotSpan>& source_spans,
    std::vector<SlotSpan>& target_spans,
    std::string* unsupported_reason = nullptr) {
    source_spans.clear();
    target_spans.clear();

    if (source_slots.size() != target_slots.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target slot counts differ";
        }
        return false;
    }
    if (source_slots.empty()) {
        return true;
    }

    std::size_t idx = 0;
    while (idx < source_slots.size()) {
        const int32_t source_begin = source_slots[idx];
        const int32_t target_begin = target_slots[idx];
        if (source_begin < 0 || target_begin < 0) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair contains negative slot";
            }
            return false;
        }

        int32_t source_stride = 1;
        int32_t target_stride = 1;
        std::size_t count = 1;
        if (idx + 1 < source_slots.size()) {
            source_stride = source_slots[idx + 1] - source_slots[idx];
            target_stride = target_slots[idx + 1] - target_slots[idx];
            if (source_stride <= 0 || target_stride <= 0) {
                if (unsupported_reason) {
                    *unsupported_reason = "span pair contains non-monotonic slots";
                }
                return false;
            }
            count = 2;
            idx += 2;
            while (idx < source_slots.size() &&
                   source_slots[idx] == source_slots[idx - 1] + source_stride &&
                   target_slots[idx] == target_slots[idx - 1] + target_stride) {
                ++count;
                ++idx;
            }
        } else {
            ++idx;
        }

        source_spans.push_back(
            SlotSpan{source_begin, static_cast<int32_t>(count), source_stride});
        target_spans.push_back(
            SlotSpan{target_begin, static_cast<int32_t>(count), target_stride});
    }

    return true;
}

inline bool build_local_span_pairs_from_slots(
    const std::vector<int32_t>& source_slots,
    const std::vector<int32_t>& target_slots,
    std::vector<SlotSpan>& source_spans,
    std::vector<SlotSpan>& target_spans,
    std::string* unsupported_reason = nullptr) {
    source_spans.clear();
    target_spans.clear();

    if (source_slots.size() != target_slots.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target slot counts differ";
        }
        return false;
    }

    std::vector<int32_t> filtered_source_slots;
    std::vector<int32_t> filtered_target_slots;
    filtered_source_slots.reserve(source_slots.size());
    filtered_target_slots.reserve(target_slots.size());
    for (std::size_t idx = 0; idx < source_slots.size(); ++idx) {
        const int32_t source_slot = source_slots[idx];
        const int32_t target_slot = target_slots[idx];
        if (target_slot < 0) {
            continue;
        }
        if (source_slot < 0) {
            if (unsupported_reason) {
                *unsupported_reason = "local span pair contains negative source slot";
            }
            return false;
        }
        filtered_source_slots.push_back(source_slot);
        filtered_target_slots.push_back(target_slot);
    }

    return build_span_pairs_from_slots(filtered_source_slots, filtered_target_slots,
                                       source_spans, target_spans,
                                       unsupported_reason);
}

inline bool build_contiguous_row_copy_blocks_from_span_pairs(
    const std::vector<SlotSpan>& source_spans,
    const std::vector<SlotSpan>& target_spans,
    std::vector<ContiguousRowCopyBlock>& blocks,
    std::string* unsupported_reason = nullptr) {
    blocks.clear();

    if (source_spans.size() != target_spans.size()) {
        if (unsupported_reason) {
            *unsupported_reason = "source/target span counts differ";
        }
        return false;
    }
    if (source_spans.empty()) {
        return true;
    }

    for (std::size_t idx = 0; idx < source_spans.size(); ++idx) {
        const SlotSpan& source_span = source_spans[idx];
        const SlotSpan& target_span = target_spans[idx];
        if (source_span.count != target_span.count) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair count differs";
            }
            blocks.clear();
            return false;
        }
        if (source_span.count <= 0) {
            continue;
        }
        if (source_span.stride <= 0 || target_span.stride <= 0) {
            if (unsupported_reason) {
                *unsupported_reason = "span pair contains non-positive stride";
            }
            blocks.clear();
            return false;
        }

        if (blocks.empty()) {
            blocks.push_back(ContiguousRowCopyBlock{
                source_span.begin,
                target_span.begin,
                source_span.count,
                1,
                source_span.stride,
                target_span.stride,
            });
            continue;
        }

        ContiguousRowCopyBlock& last = blocks.back();
        const int32_t expected_source_begin =
            last.source_begin + last.row_count * last.source_row_stride;
        const int32_t expected_target_begin =
            last.target_begin + last.row_count * last.target_row_stride;
        if (last.row_width == source_span.count &&
            last.source_row_stride == source_span.stride &&
            last.target_row_stride == target_span.stride &&
            expected_source_begin == source_span.begin &&
            expected_target_begin == target_span.begin) {
            ++last.row_count;
            continue;
        }

        blocks.push_back(ContiguousRowCopyBlock{
            source_span.begin,
            target_span.begin,
            source_span.count,
            1,
            source_span.stride,
            target_span.stride,
        });
    }

    return true;
}

template <typename T>
inline void scatter_values_by_span_pairs_into(
    const std::vector<T>& source,
    const std::vector<SlotSpan>& source_spans,
    const std::vector<SlotSpan>& target_spans,
    std::vector<T>& target) {
    const std::size_t count =
        std::min(source_spans.size(), target_spans.size());
    for (std::size_t idx = 0; idx < count; ++idx) {
        const SlotSpan& source_span = source_spans[idx];
        const SlotSpan& target_span = target_spans[idx];
        const int32_t span_count =
            std::min(source_span.count, target_span.count);
        if (span_count <= 0) {
            continue;
        }
        if (source.data() != target.data() &&
            source_span.stride == 1 &&
            target_span.stride == 1 &&
            source_span.begin >= 0 &&
            target_span.begin >= 0 &&
            static_cast<std::size_t>(source_span.begin) +
                    static_cast<std::size_t>(span_count) <=
                source.size() &&
            static_cast<std::size_t>(target_span.begin) +
                    static_cast<std::size_t>(span_count) <=
                target.size()) {
            std::copy_n(source.begin() + source_span.begin, span_count,
                        target.begin() + target_span.begin);
            continue;
        }
        for (int32_t offset = 0; offset < span_count; ++offset) {
            const int32_t source_slot =
                source_span.begin + offset * source_span.stride;
            const int32_t target_slot =
                target_span.begin + offset * target_span.stride;
            if (source_slot < 0 || target_slot < 0 ||
                static_cast<std::size_t>(source_slot) >= source.size() ||
                static_cast<std::size_t>(target_slot) >= target.size()) {
                continue;
            }
            target[static_cast<std::size_t>(target_slot)] =
                source[static_cast<std::size_t>(source_slot)];
        }
    }
}

template <typename T>
inline void scatter_values_by_row_copy_blocks_into(
    const std::vector<T>& source,
    const std::vector<ContiguousRowCopyBlock>& blocks,
    std::vector<T>& target) {
    for (const ContiguousRowCopyBlock& block : blocks) {
        if (block.row_width <= 0 || block.row_count <= 0 ||
            block.source_begin < 0 || block.target_begin < 0 ||
            block.source_row_stride <= 0 || block.target_row_stride <= 0) {
            continue;
        }
        for (int32_t row = 0; row < block.row_count; ++row) {
            const int32_t source_row_begin =
                block.source_begin + row * block.source_row_stride;
            const int32_t target_row_begin =
                block.target_begin + row * block.target_row_stride;
            if (static_cast<std::size_t>(source_row_begin) +
                    static_cast<std::size_t>(block.row_width) > source.size() ||
                static_cast<std::size_t>(target_row_begin) +
                    static_cast<std::size_t>(block.row_width) > target.size()) {
                break;
            }
            std::copy_n(source.begin() + source_row_begin, block.row_width,
                        target.begin() + target_row_begin);
        }
    }
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    HaloExchangeRuntime<T>& halo_runtime,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    if (!local_row_copy_blocks.empty()) {
        scatter_values_by_row_copy_blocks_into(local_write_source,
                                               local_row_copy_blocks,
                                               local_cache);
    } else if (!local_write_spans.empty() && !local_target_spans.empty()) {
        scatter_values_by_span_pairs_into(local_write_source, local_write_spans,
                                          local_target_spans, local_cache);
    } else if (!local_write_slots.empty()) {
        scatter_values_by_slots_parallel_into(local_write_source,
                                              local_write_slots,
                                              local_target_slots,
                                              local_cache,
                                              q);
    }

    if (halo_plan.supported) {
        exchange_values_by_halo_spans(local_write_source, halo_plan, halo_runtime,
                                      local_cache, comm);
        return;
    }

    exchange_values_by_slots(local_write_source, exchange_plan, local_cache,
                             comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    HaloExchangeRuntime<T>& halo_runtime,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, halo_runtime, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    HaloExchangeRuntime<T> halo_runtime;
    if (halo_plan.supported) {
        prepare_halo_exchange_runtime(halo_plan, halo_runtime);
    }
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, local_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, halo_runtime, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    sycl::queue& q,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    const std::vector<ContiguousRowCopyBlock>& local_row_copy_blocks,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, local_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

template <typename T>
inline void publish_local_writes_with_span_pairs_or_exchange_cache_only(
    const std::vector<T>& local_write_source,
    const std::vector<int32_t>& local_write_slots,
    const std::vector<int32_t>& local_target_slots,
    const std::vector<SlotSpan>& local_write_spans,
    const std::vector<SlotSpan>& local_target_spans,
    std::vector<T>& local_cache,
    const ExchangePlan& exchange_plan,
    const HaloExchangePlan& halo_plan,
    MPI_Comm comm = MPI_COMM_WORLD) {
    const std::vector<ContiguousRowCopyBlock> empty_row_copy_blocks;
    sycl::queue q(sycl::default_selector_v);
    publish_local_writes_with_span_pairs_or_exchange_cache_only(
        local_write_source, local_write_slots, local_target_slots,
        local_write_spans, local_target_spans, empty_row_copy_blocks,
        local_cache, exchange_plan, halo_plan, q, comm);
}

}  // namespace mpi
}  // namespace dacpp

#endif  // DACPP_MPI_WAVE_EXCHANGE_SPECIALIZATION_H
