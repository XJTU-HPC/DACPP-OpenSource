#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

static inline bool __dacpp_mpi_is_root_rank();
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

#ifndef GRADIENT_NUM_NEURONS
#define GRADIENT_NUM_NEURONS 8
#endif

#ifndef GRADIENT_INPUT_SIZE
#define GRADIENT_INPUT_SIZE 8
#endif

const int NUM_NEURONS = GRADIENT_NUM_NEURONS;   // 神经元数量（层宽度）
const int INPUT_SIZE  = GRADIENT_INPUT_SIZE;    // 每个神经元输入数
 


// -----------------------------
// DAC calc 函数：规约加法
// -----------------------------


// -----------------------------
// 主函数
// -----------------------------
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"
#include <chrono>
#include <utility>

static inline bool __dacpp_mpi_is_root_rank() {
    int __dacpp_mpi_initialized = 0;
    MPI_Initialized(&__dacpp_mpi_initialized);
    if (!__dacpp_mpi_initialized) {
        return true;
    }
    int __dacpp_mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &__dacpp_mpi_rank);
    return __dacpp_mpi_rank == 0;
}

using namespace sycl;

template <typename __dacpp_view_t0, typename __dacpp_view_t1>
__attribute__((always_inline)) inline void gradSum_mpi_local(__dacpp_view_t0 grads, __dacpp_view_t1 neuronSum) {
    int sum = 0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += grads[j];
    }
    neuronSum[0] = sum;
}


void __dacpp_mpi_or_gradSumShell_gradSum_0(dacpp::Matrix<float> & __or_arg0, dacpp::Matrix<float> & __or_arg1) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::SegmentedProfile dacpp_profile;
    auto& q = dacpp::mpi::operator_resident::default_queue();
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_rows = __or_arg1.getShape(0);
    const int64_t __or_cols = __or_arg1.getShape(1);
    const int64_t __or_total_items = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_rows, __or_cols, "[DACPP][MPI][OR] row-block total item count overflow");
    const auto __or_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, mpi_rank, mpi_size);
    const int64_t __or_local_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_row_range.count, __or_cols, "[DACPP][MPI][OR] row-block local item count overflow");
    std::vector<int> __or_row_counts;
    std::vector<int> __or_row_displs;
    dacpp::mpi::operator_resident::counts_displs_1d(__or_rows, mpi_size, __or_row_counts, __or_row_displs);
    std::vector<int> __or_counts(mpi_size);
    std::vector<int> __or_displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        __or_counts[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_counts[r]), __or_cols, "[DACPP][MPI][OR] row-block scatter count overflow"), "[DACPP][MPI][OR] row-block scatter count exceeds MPI int range");
        __or_displs[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_displs[r]), __or_cols, "[DACPP][MPI][OR] row-block scatter displacement overflow"), "[DACPP][MPI][OR] row-block scatter displacement exceeds MPI int range");
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    const int64_t __or_payload_len_grads = __or_arg0.getShape(1);
    const int64_t __or_payload_index_extent_grads = __or_arg0.getShape(0);
    const int64_t __or_payload_unique_count_grads = __or_row_range.count;
    const int64_t __or_payload_index_begin_grads = __or_row_range.begin;
    if (__or_arg0.getDim() != 2 || __or_payload_len_grads < 0 || __or_payload_unique_count_grads < 0 || __or_payload_index_begin_grads < 0 || (__or_payload_unique_count_grads > 0 && __or_payload_index_begin_grads + __or_payload_unique_count_grads > __or_payload_index_extent_grads)) {
        if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] RowPartitionFullRow input payload/index range out of bounds\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    if ((__or_payload_unique_count_grads * __or_payload_len_grads) > static_cast<int64_t>(2147483647)) {
        if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] RowPartitionFullRow input payload exceeds MPI int count\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    std::vector<float> __or_local_grads(static_cast<std::size_t>(__or_payload_unique_count_grads * __or_payload_len_grads));
    std::vector<int> __or_payload_counts_grads(mpi_size);
    std::vector<int> __or_payload_displs_grads(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        __or_payload_counts_grads[r] = static_cast<int>(static_cast<int64_t>(__or_row_counts[r]) * __or_payload_len_grads);
        __or_payload_displs_grads[r] = static_cast<int>(static_cast<int64_t>(__or_row_displs[r]) * __or_payload_len_grads);
    }
    std::vector<float> __or_sendbuf_grads;
    float* __or_payload_send_ptr_grads = nullptr;
    bool __or_payload_direct_grads = false;
    if (mpi_rank == 0) {
        const int64_t __or_input_rows_grads = __or_arg0.getShape(0);
        const int64_t __or_input_cols_grads = __or_arg0.getShape(1);
        __or_payload_direct_grads = (__or_arg0.getDim() == 2 && __or_arg0.getOffset() >= 0 && __or_arg0.getStride(1) == 1 && __or_arg0.getStride(0) == __or_input_cols_grads && __or_payload_len_grads == __or_input_cols_grads && __or_input_rows_grads >= __or_row_displs[mpi_size - 1] + __or_row_counts[mpi_size - 1]);
        if (__or_payload_direct_grads) {
            __or_payload_send_ptr_grads = __or_arg0.getDataPtr().get() + __or_arg0.getOffset();
        }
    }
    if (mpi_rank == 0) {
        if (!__or_payload_direct_grads) {
        auto dacpp_profile_pack_start_grads = dacpp::mpi::profileSegmentStart();
        std::vector<float> __or_global_grads;
        __or_arg0.tensor2Array(__or_global_grads);
        __or_sendbuf_grads.resize(static_cast<std::size_t>(__or_rows * __or_payload_len_grads));
        const int64_t __or_input_cols_grads = __or_arg0.getShape(1);
        const int64_t __or_input_rows_grads = __or_arg0.getShape(0);
        if ((true && __or_payload_len_grads > __or_input_cols_grads) || (false && __or_payload_len_grads > __or_input_rows_grads)) {
            std::fprintf(stderr, "[DACPP][MPI][OR] RowPartitionFullRow payload length exceeds input dimension\n");
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        for (int r = 0; r < mpi_size; ++r) {
            for (int64_t local_i = 0; local_i < __or_row_counts[r]; ++local_i) {
                const int64_t indexed_value = static_cast<int64_t>(__or_row_displs[r]) + local_i;
                const int64_t dst_base = (__or_payload_displs_grads[r] + local_i * __or_payload_len_grads);
                for (int64_t payload_i = 0; payload_i < __or_payload_len_grads; ++payload_i) {
                    const int64_t src_linear = indexed_value * __or_input_cols_grads + payload_i;
                    __or_sendbuf_grads[static_cast<std::size_t>(dst_base + payload_i)] = __or_global_grads[static_cast<std::size_t>(src_linear)];
                }
            }
        }
        __or_payload_send_ptr_grads = __or_sendbuf_grads.data();
        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Pack, dacpp_profile_pack_start_grads);
        }
    }
    auto dacpp_profile_scatter_start_grads = dacpp::mpi::profileSegmentStart();
    MPI_Scatterv(mpi_rank == 0 ? __or_payload_send_ptr_grads : nullptr, mpi_rank == 0 ? __or_payload_counts_grads.data() : nullptr, mpi_rank == 0 ? __or_payload_displs_grads.data() : nullptr, MPI_FLOAT, __or_local_grads.data(), static_cast<int>(__or_payload_unique_count_grads * __or_payload_len_grads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_grads);
    std::vector<float> __or_local_neuronSum(static_cast<std::size_t>(__or_local_item_count));
    std::fill(__or_local_neuronSum.begin(), __or_local_neuronSum.end(), float{});
    // Output-direct no-read fast path for neuronSum initializes local output and skips root pack/scatter.
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<float, 1> __or_buffer_grads(__or_local_grads.data(), sycl::range<1>(__or_local_grads.size()));
            sycl::buffer<float, 1> __or_buffer_neuronSum(__or_local_neuronSum.data(), sycl::range<1>(__or_local_neuronSum.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_grads = __or_buffer_grads.get_access<sycl::access::mode::read>(h);
                auto __or_acc_neuronSum = __or_buffer_neuronSum.get_access<sycl::access::mode::discard_write>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_grads = __or_acc_grads.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const float> view_grads{__or_data_grads, static_cast<int>((item_linear / static_cast<int>(__or_cols)) * __or_payload_len_grads)};
                    auto* __or_data_neuronSum = __or_acc_neuronSum.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<float> view_neuronSum{__or_data_neuronSum, item_linear};
                    gradSum_mpi_local(view_grads, view_neuronSum);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    auto dacpp_profile_bounded_sync_start_neuronSum = dacpp::mpi::profileSegmentStart();
    {
        std::vector<float> __or_bounded_values_neuronSum(static_cast<std::size_t>(5), float{});
        {
            const int64_t __or_target_row = 0;
            const int64_t __or_target_col = 0;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((0 - __or_row_range.begin) * __or_cols + 0);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_neuronSum.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_neuronSum[static_cast<std::size_t>(0)] = __or_local_neuronSum[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 1;
            const int64_t __or_target_col = 0;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((1 - __or_row_range.begin) * __or_cols + 0);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_neuronSum.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_neuronSum[static_cast<std::size_t>(1)] = __or_local_neuronSum[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 2;
            const int64_t __or_target_col = 0;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((2 - __or_row_range.begin) * __or_cols + 0);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_neuronSum.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_neuronSum[static_cast<std::size_t>(2)] = __or_local_neuronSum[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 3;
            const int64_t __or_target_col = 0;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((3 - __or_row_range.begin) * __or_cols + 0);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_neuronSum.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_neuronSum[static_cast<std::size_t>(3)] = __or_local_neuronSum[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 4;
            const int64_t __or_target_col = 0;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((4 - __or_row_range.begin) * __or_cols + 0);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_neuronSum.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_neuronSum[static_cast<std::size_t>(4)] = __or_local_neuronSum[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 0;
            const int64_t __or_target_col = 0;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_neuronSum[static_cast<std::size_t>(0)], 1, MPI_FLOAT, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_neuronSum[static_cast<std::size_t>(0)], 1, MPI_FLOAT, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        {
            const int64_t __or_target_row = 1;
            const int64_t __or_target_col = 0;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_neuronSum[static_cast<std::size_t>(1)], 1, MPI_FLOAT, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_neuronSum[static_cast<std::size_t>(1)], 1, MPI_FLOAT, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        {
            const int64_t __or_target_row = 2;
            const int64_t __or_target_col = 0;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_neuronSum[static_cast<std::size_t>(2)], 1, MPI_FLOAT, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_neuronSum[static_cast<std::size_t>(2)], 1, MPI_FLOAT, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        {
            const int64_t __or_target_row = 3;
            const int64_t __or_target_col = 0;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_neuronSum[static_cast<std::size_t>(3)], 1, MPI_FLOAT, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_neuronSum[static_cast<std::size_t>(3)], 1, MPI_FLOAT, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        {
            const int64_t __or_target_row = 4;
            const int64_t __or_target_col = 0;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_neuronSum[static_cast<std::size_t>(4)], 1, MPI_FLOAT, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_neuronSum[static_cast<std::size_t>(4)], 1, MPI_FLOAT, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (mpi_rank == 0) {
            __or_arg1.reviseValue(__or_bounded_values_neuronSum[static_cast<std::size_t>(0)], std::vector<int>{0, 0});
            __or_arg1.reviseValue(__or_bounded_values_neuronSum[static_cast<std::size_t>(1)], std::vector<int>{1, 0});
            __or_arg1.reviseValue(__or_bounded_values_neuronSum[static_cast<std::size_t>(2)], std::vector<int>{2, 0});
            __or_arg1.reviseValue(__or_bounded_values_neuronSum[static_cast<std::size_t>(3)], std::vector<int>{3, 0});
            __or_arg1.reviseValue(__or_bounded_values_neuronSum[static_cast<std::size_t>(4)], std::vector<int>{4, 0});
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::FinalSync, dacpp_profile_bounded_sync_start_neuronSum);
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_gradSumShell_gradSum_0", dacpp_profile, MPI_COMM_WORLD);
}

int main() {
    int dacpp_mpi_finalize_needed = 0;
    int dacpp_mpi_initialized = 0;
    MPI_Initialized(&dacpp_mpi_initialized);
    if (!dacpp_mpi_initialized) {
        int dacpp_mpi_argc = 0;
        char** dacpp_mpi_argv = nullptr;
        MPI_Init(&dacpp_mpi_argc, &dacpp_mpi_argv);
        dacpp_mpi_finalize_needed = 1;
    }
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 初始化梯度矩阵（模拟随机梯度）
    vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    // for (auto &v : host_grads) v = dist(gen);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j]=i + j;
        }
    }
    // DAC Tensor 初始化
    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    // 执行 DAC shell -> calc
    __dacpp_mpi_or_gradSumShell_gradSum_0(matGrads, matNeuronSum);

    // 输出结果
    if (__dacpp_mpi_is_root_rank()) {
        std::cout << "First 5 neuron gradient sums:\n";
    }
    for (size_t i = 0; i < std::min(5, NUM_NEURONS) ; ++i)
        if (__dacpp_mpi_is_root_rank()) {
        std::cout << matNeuronSum[i][0] << " ";
    }
    if (__dacpp_mpi_is_root_rank()) {
        std::cout << std::endl;
    }

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
