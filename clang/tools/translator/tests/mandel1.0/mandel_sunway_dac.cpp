#include <iostream>
#include <vector>
#include <complex>
#include "ReconTensor.h"
#include <cmath>

static inline bool __dacpp_mpi_is_root_rank();
namespace dacpp {
    typedef std::vector<std::any> list;
}
using namespace std;


#ifndef MANDEL_N
#define MANDEL_N 8192
#endif

#ifndef MANDEL_MAX_ITERATIONS
#define MANDEL_MAX_ITERATIONS 1000
#endif

// 全局变量定义
constexpr int row_count = MANDEL_N;
constexpr int col_count = MANDEL_N;
constexpr int max_iterations = MANDEL_MAX_ITERATIONS;
vector<complex<float>> complex_points;  // 一维向量表示复数点
vector<int> mandelbrot_flags;           // 一维数组表示是否属于 Mandelbrot 集
int total_points = 0;                   // 总点数
int mandelbrot_count = 0;               // 属于 Mandelbrot 集的点数量

// 初始化复数点向量
void InitializeComplexPoints() {
    total_points = row_count * col_count;  // 总点数
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;  // 一维向量索引
            float real = -1.5f + (i * (2.0f / row_count));  // 将行索引映射到实部
            float imag = -1.0f + (j * (2.0f / col_count));  // 将列索引映射到虚部
            complex_points[index] = complex<float>(real, imag);
        }
    }
}







// 打印统计信息
void PrintStats() {
    if (__dacpp_mpi_is_root_rank()) {
        cout << "Mandelbrot Set Statistics:\n";
    }
    if (__dacpp_mpi_is_root_rank()) {
        cout << "Total points: " << total_points << "\n";
    }
    if (__dacpp_mpi_is_root_rank()) {
        cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
    }
}

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

namespace __dacpp_test {
struct ScopedCommunicationTimer {};

inline void print_e2e_summary(int mpi_rank, double total_time) {
    double total_max = 0.0;
    MPI_Reduce(&total_time, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::cerr << "[MPI_DAC][mandel] e2e_seconds=" << total_max << std::endl;
    }
}
}  // namespace __dacpp_test


template <typename __dacpp_view_t0, typename __dacpp_view_t1>
inline void mandel_mpi_local(__dacpp_view_t0 complex_points, __dacpp_view_t1 mandelbrot_flags) {
    const complex<float> &c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (std::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.F) {
            iterations = i;
            break;
        }
        z = z * z + c;
        iterations = max_iterations;
    }
    if (iterations == max_iterations) {
        mandelbrot_flags[0] = 1;
    }
}


void __dacpp_mpi_or_MANDEL_mandel_0(const dacpp::Vector<complex<float> > & __or_arg0, dacpp::Vector<int> & __or_arg1) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::SegmentedProfile dacpp_profile;
    auto& q = dacpp::mpi::operator_resident::default_queue();
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_total_items = __or_arg1.getShape(0);
    const int64_t __or_block_cyclic_block_size = 64;
    const auto __or_block_cyclic_layout = dacpp::mpi::operator_resident::block_cyclic_layout_1d(__or_total_items, mpi_rank, mpi_size, __or_block_cyclic_block_size);
    const int64_t __or_local_item_count = __or_block_cyclic_layout.local_count;
    std::vector<int> __or_counts;
    dacpp::mpi::operator_resident::counts_1d_block_cyclic(__or_total_items, mpi_size, __or_block_cyclic_block_size, __or_counts);
    std::vector<int> __or_displs(mpi_size, 0);
    int64_t __or_block_cyclic_displ = 0;
    for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {
        __or_displs[__or_r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_block_cyclic_displ, "[DACPP][MPI][OR] block-cyclic displacement exceeds MPI int range");
        __or_block_cyclic_displ += static_cast<int64_t>(__or_counts[__or_r]);
    }
    auto __or_global_index_for_local = [=](int64_t __or_local_i) -> int64_t { return dacpp::mpi::operator_resident::block_cyclic_global_index_1d(__or_local_i, mpi_rank, mpi_size, __or_block_cyclic_block_size); };
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    std::vector<complex<float>> __or_local_complex_points(static_cast<std::size_t>(__or_local_item_count));
    auto dacpp_profile_scatter_start_complex_points = dacpp::mpi::profileSegmentStart();
    if (mpi_rank == 0) {
        const int64_t __or_direct_offset_complex_points = __or_arg0.getOffset();
        const int64_t __or_direct_stride_complex_points = __or_arg0.getStride(0);
        const bool __or_block_cyclic_direct_complex_points = (__or_direct_offset_complex_points >= 0 && __or_direct_stride_complex_points == 1 && __or_arg0.getSize() >= __or_total_items && __or_direct_offset_complex_points + __or_total_items <= __or_arg0.getSize());
        std::vector<complex<float>> __or_global_complex_points;
        const complex<float>* __or_block_cyclic_src_complex_points = nullptr;
        if (__or_block_cyclic_direct_complex_points) {
            __or_block_cyclic_src_complex_points = __or_arg0.getDataPtr().get() + __or_direct_offset_complex_points;
        } else {
            auto dacpp_profile_pack_start_complex_points = dacpp::mpi::profileSegmentStart();
            __or_arg0.tensor2Array(__or_global_complex_points);
            __or_block_cyclic_src_complex_points = __or_global_complex_points.data();
            dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Pack, dacpp_profile_pack_start_complex_points);
        }
        for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {
            const int64_t __or_target_count = dacpp::mpi::operator_resident::block_cyclic_count_1d(__or_total_items, __or_r, mpi_size, __or_block_cyclic_block_size);
            if (__or_r == 0) {
                for (int64_t __or_local_i = 0; __or_local_i < __or_target_count; ++__or_local_i) {
                    const int64_t __or_global_i = dacpp::mpi::operator_resident::block_cyclic_global_index_1d(__or_local_i, 0, mpi_size, __or_block_cyclic_block_size);
                    __or_local_complex_points[static_cast<std::size_t>(__or_local_i)] = __or_block_cyclic_src_complex_points[static_cast<std::size_t>(__or_global_i)];
                }
            } else if (__or_target_count > 0) {
                std::vector<complex<float>> __or_block_cyclic_payload(static_cast<std::size_t>(__or_target_count));
                for (int64_t __or_local_i = 0; __or_local_i < __or_target_count; ++__or_local_i) {
                    const int64_t __or_global_i = dacpp::mpi::operator_resident::block_cyclic_global_index_1d(__or_local_i, __or_r, mpi_size, __or_block_cyclic_block_size);
                    __or_block_cyclic_payload[static_cast<std::size_t>(__or_local_i)] = __or_block_cyclic_src_complex_points[static_cast<std::size_t>(__or_global_i)];
                }
                {
                    __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
                    MPI_Send(__or_block_cyclic_payload.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_target_count, "[DACPP][MPI][OR] block-cyclic scatter count exceeds MPI int range"), MPI_C_FLOAT_COMPLEX, __or_r, 4801, MPI_COMM_WORLD);
                }
            }
        }
        if (mpi_size > 1) {
            {
                __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    } else if (__or_local_item_count > 0) {
        {
            __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
            MPI_Recv(__or_local_complex_points.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_local_item_count, "[DACPP][MPI][OR] block-cyclic scatter recv count exceeds MPI int range"), MPI_C_FLOAT_COMPLEX, 0, 4801, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        {
            __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    } else if (mpi_size > 1) {
        {
            __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_complex_points);
    std::vector<int> __or_local_mandelbrot_flags(static_cast<std::size_t>(__or_local_item_count));
    std::fill(__or_local_mandelbrot_flags.begin(), __or_local_mandelbrot_flags.end(), int{});
    // Output-direct no-read fast path for mandelbrot_flags initializes local output and skips root pack/scatter.
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<complex<float>, 1> __or_buffer_complex_points(__or_local_complex_points.data(), sycl::range<1>(__or_local_complex_points.size()));
            sycl::buffer<int, 1> __or_buffer_mandelbrot_flags(__or_local_mandelbrot_flags.data(), sycl::range<1>(__or_local_mandelbrot_flags.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_complex_points = __or_buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto __or_acc_mandelbrot_flags = __or_buffer_mandelbrot_flags.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class mandel_mpi_dac_sycl_buffer_test_kernel_1>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_complex_points = __or_acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const complex<float>> view_complex_points{__or_data_complex_points, item_linear};
                    auto* __or_data_mandelbrot_flags = __or_acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<int> view_mandelbrot_flags{__or_data_mandelbrot_flags, item_linear};
                    mandel_mpi_local(view_complex_points, view_mandelbrot_flags);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    int64_t __or_local_reduction_count_mandelbrot_flags = 0;
    for (const auto& __or_value : __or_local_mandelbrot_flags) {
        if (__or_value == static_cast<int>(1)) {
            ++__or_local_reduction_count_mandelbrot_flags;
        }
    }
    int64_t __or_global_reduction_count_mandelbrot_flags = 0;
    auto dacpp_profile_reduce_start_mandelbrot_flags = dacpp::mpi::profileSegmentStart();
    {
        __dacpp_test::ScopedCommunicationTimer __dacpp_test_comm_timer;
        MPI_Reduce(&__or_local_reduction_count_mandelbrot_flags, &__or_global_reduction_count_mandelbrot_flags, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::FinalSync, dacpp_profile_reduce_start_mandelbrot_flags);
    if (mpi_rank == 0) {
        mandelbrot_count = static_cast<decltype(mandelbrot_count)>(__or_global_reduction_count_mandelbrot_flags);
    }
    // Post-use reduction for mandelbrot_flags replaces full output materialization.
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_MANDEL_mandel_0", dacpp_profile, MPI_COMM_WORLD);
}

int main() {
    const auto __dacpp_test_e2e_start = std::chrono::steady_clock::now();
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

    // 初始化复数点向量
    InitializeComplexPoints();

    // 计算 Mandelbrot 集
    mandelbrot_flags.resize(total_points, 0);  // 初始化一维数组为 0

    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);


    __dacpp_mpi_or_MANDEL_mandel_0(complex_points_tensor, mandelbrot_flags_tensor);

    // 统计数组中 1 的个数
    /* DACPP MPI post-use reduction scalar set in wrapper */;
    /* DACPP MPI post-use reduction loop elided */

    // 打印统计信息
    // PrintStats();

    MPI_Barrier(MPI_COMM_WORLD);
    const auto __dacpp_test_e2e_end = std::chrono::steady_clock::now();
    const double __dacpp_test_e2e_seconds =
        std::chrono::duration<double>(__dacpp_test_e2e_end -
                                      __dacpp_test_e2e_start)
            .count();
    __dacpp_test::print_e2e_summary(mpi_rank, __dacpp_test_e2e_seconds);
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
