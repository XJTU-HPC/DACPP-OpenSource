#include <iostream>
#include <vector>
#include <complex>
#include "ReconTensor.h"
#include <cmath>

namespace dacpp {
    typedef std::vector<std::any> list;
}
using namespace std;


// 全局变量定义
const int row_count = 8, col_count = 8, max_iterations = 1000;
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
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"
#include <chrono>

using namespace sycl;

inline void mandel_mpi_local(dacpp::mpi::View1D<const complex<float>> complex_points, dacpp::mpi::View1D<int> mandelbrot_flags) {
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


void MANDEL_mandel(const dacpp::Vector<complex<float> > & complex_points, dacpp::Vector<int> & mandelbrot_flags) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::resetCollectPositionsProfile();
    auto dacpp_wrapper_start = std::chrono::steady_clock::now();
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_complex_points;
    pattern_complex_points.param_id = 0;
    pattern_complex_points.name = "complex_points";
    pattern_complex_points.mode = dacpp::mpi::AccessMode::Read;
    pattern_complex_points.data_info.dim = complex_points.getDim();
    for (int dim = 0; dim < complex_points.getDim(); ++dim) pattern_complex_points.data_info.dimLength.push_back(complex_points.getShape(dim));
    Dac_Op pattern_complex_points_op_0;
    pattern_complex_points_op_0.setDimId(0);
    pattern_complex_points_op_0.size = 1;
    pattern_complex_points_op_0.stride = 1;
    pattern_complex_points_op_0.SetSplitSize(complex_points.getShape(0));
    pattern_complex_points.param_ops.push_back(pattern_complex_points_op_0);
    pattern_complex_points.bind_set_id.push_back(0);
    pattern_complex_points.bind_offset_expr.push_back("0");
    pattern_complex_points.is_index_op.push_back(true);
    pattern_complex_points.partition_shape = dacpp::mpi::init_partition_shape(pattern_complex_points);
    pattern_complex_points.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_complex_points);
    if (binding_split_sizes.size() < pattern_complex_points.bind_split_sizes.size()) binding_split_sizes.resize(pattern_complex_points.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_complex_points.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_complex_points.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_mandelbrot_flags;
    pattern_mandelbrot_flags.param_id = 1;
    pattern_mandelbrot_flags.name = "mandelbrot_flags";
    pattern_mandelbrot_flags.mode = dacpp::mpi::AccessMode::Write;
    pattern_mandelbrot_flags.data_info.dim = mandelbrot_flags.getDim();
    for (int dim = 0; dim < mandelbrot_flags.getDim(); ++dim) pattern_mandelbrot_flags.data_info.dimLength.push_back(mandelbrot_flags.getShape(dim));
    Dac_Op pattern_mandelbrot_flags_op_0;
    pattern_mandelbrot_flags_op_0.setDimId(0);
    pattern_mandelbrot_flags_op_0.size = 1;
    pattern_mandelbrot_flags_op_0.stride = 1;
    pattern_mandelbrot_flags_op_0.SetSplitSize(mandelbrot_flags.getShape(0));
    pattern_mandelbrot_flags.param_ops.push_back(pattern_mandelbrot_flags_op_0);
    pattern_mandelbrot_flags.bind_set_id.push_back(0);
    pattern_mandelbrot_flags.bind_offset_expr.push_back("0");
    pattern_mandelbrot_flags.is_index_op.push_back(true);
    pattern_mandelbrot_flags.partition_shape = dacpp::mpi::init_partition_shape(pattern_mandelbrot_flags);
    pattern_mandelbrot_flags.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_mandelbrot_flags);
    if (binding_split_sizes.size() < pattern_mandelbrot_flags.bind_split_sizes.size()) binding_split_sizes.resize(pattern_mandelbrot_flags.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_mandelbrot_flags.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_mandelbrot_flags.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_complex_points.bind_split_sizes = binding_split_sizes;
    pattern_mandelbrot_flags.bind_split_sizes = binding_split_sizes;
    auto plan_complex_points = dacpp::mpi::build_input_pack_plan(item_range, pattern_complex_points);
    auto& pack_complex_points = plan_complex_points.pack;
    auto& slots_complex_points = plan_complex_points.compact_slots;
    auto& key_offsets_complex_points = plan_complex_points.item_key_offsets;
    std::vector<complex<float>> local_complex_points(pack_complex_points.globals.size());
    std::vector<int> sendcounts_complex_points;
    std::vector<int> displs_complex_points;
    int local_global_count_complex_points = static_cast<int>(pack_complex_points.globals.size());
    std::vector<int> global_counts_complex_points;
    std::vector<int> global_displs_complex_points;
    std::vector<int64_t> gathered_globals_complex_points;
    if (mpi_rank == 0) {
        global_counts_complex_points.resize(mpi_size);
        global_displs_complex_points.resize(mpi_size);
    }
    MPI_Gather(&local_global_count_complex_points, 1, MPI_INT, mpi_rank == 0 ? global_counts_complex_points.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_global_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            global_displs_complex_points[r] = current_global_displ;
            current_global_displ += global_counts_complex_points[r];
        }
        gathered_globals_complex_points.resize(current_global_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(pack_complex_points.globals.data()), local_global_count_complex_points, MPI_LONG_LONG, mpi_rank == 0 ? gathered_globals_complex_points.data() : nullptr, mpi_rank == 0 ? global_counts_complex_points.data() : nullptr, mpi_rank == 0 ? global_displs_complex_points.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<complex<float>> sendbuf_complex_points;
    if (mpi_rank == 0) {
        sendcounts_complex_points.resize(mpi_size);
        displs_complex_points.resize(mpi_size);
        int current_displ = 0;
        std::vector<complex<float>> global_complex_points;
        complex_points.tensor2Array(global_complex_points);
        for (int r = 0; r < mpi_size; ++r) {
            sendcounts_complex_points[r] = global_counts_complex_points[r];
            displs_complex_points[r] = current_displ;
            current_displ += sendcounts_complex_points[r];
        }
        sendbuf_complex_points = dacpp::mpi::pack_values_by_globals_parallel_range(global_complex_points, gathered_globals_complex_points.data(), gathered_globals_complex_points.size());
    }
    local_complex_points.resize(static_cast<std::size_t>(local_global_count_complex_points));
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_complex_points.data() : nullptr, mpi_rank == 0 ? sendcounts_complex_points.data() : nullptr, mpi_rank == 0 ? displs_complex_points.data() : nullptr, MPI_C_FLOAT_COMPLEX, local_complex_points.data(), local_global_count_complex_points, MPI_C_FLOAT_COMPLEX, 0, MPI_COMM_WORLD);
    const int complex_points_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_complex_points));
    auto plan_mandelbrot_flags = dacpp::mpi::build_output_pack_plan(item_range, pattern_mandelbrot_flags);
    auto& pack_mandelbrot_flags = plan_mandelbrot_flags.pack;
    auto& slots_mandelbrot_flags = plan_mandelbrot_flags.compact_slots;
    auto& key_offsets_mandelbrot_flags = plan_mandelbrot_flags.item_key_offsets;
    std::vector<int> local_mandelbrot_flags(pack_mandelbrot_flags.globals.size());
    const int mandelbrot_flags_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_mandelbrot_flags));
    if (local_item_count > 0) {
        {
            sycl::buffer<complex<float>, 1> buffer_complex_points(local_complex_points.data(), sycl::range<1>(local_complex_points.size()));
            sycl::buffer<int32_t, 1> slots_buffer_complex_points(slots_complex_points.data(), sycl::range<1>(slots_complex_points.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_complex_points(key_offsets_complex_points.data(), sycl::range<1>(key_offsets_complex_points.size()));
            sycl::buffer<int, 1> buffer_mandelbrot_flags(local_mandelbrot_flags.data(), sycl::range<1>(local_mandelbrot_flags.size()));
            sycl::buffer<int32_t, 1> slots_buffer_mandelbrot_flags(slots_mandelbrot_flags.data(), sycl::range<1>(slots_mandelbrot_flags.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_mandelbrot_flags(key_offsets_mandelbrot_flags.data(), sycl::range<1>(key_offsets_mandelbrot_flags.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_complex_points = buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto slots_acc_complex_points = slots_buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_complex_points = key_offsets_buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto acc_mandelbrot_flags = buffer_mandelbrot_flags.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_mandelbrot_flags = slots_buffer_mandelbrot_flags.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_mandelbrot_flags = key_offsets_buffer_mandelbrot_flags.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_complex_points = acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_complex_points = slots_acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_complex_points = key_offsets_acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const complex<float>> view_complex_points{data_complex_points, slots_complex_points, key_offsets_complex_points[item_linear]};
                    auto* data_mandelbrot_flags = acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_mandelbrot_flags = slots_acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_mandelbrot_flags = key_offsets_acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<int> view_mandelbrot_flags{data_mandelbrot_flags, slots_mandelbrot_flags, key_offsets_mandelbrot_flags[item_linear]};
                    mandel_mpi_local(view_complex_points, view_mandelbrot_flags);
                });
            });
            q.wait();
        }
    }
    const auto& writeback_globals_mandelbrot_flags = pack_mandelbrot_flags.writeback_globals.empty() ? pack_mandelbrot_flags.globals : pack_mandelbrot_flags.writeback_globals;
    int send_count_mandelbrot_flags = static_cast<int>(writeback_globals_mandelbrot_flags.size());
    std::vector<int> recvcounts_mandelbrot_flags;
    std::vector<int> recvdispls_mandelbrot_flags;
    std::vector<int64_t> global_recv_globals_mandelbrot_flags;
    std::vector<int> global_recv_values_mandelbrot_flags;
    if (mpi_rank == 0) {
        recvcounts_mandelbrot_flags.resize(mpi_size);
        recvdispls_mandelbrot_flags.resize(mpi_size);
    }
    MPI_Gather(&send_count_mandelbrot_flags, 1, MPI_INT, mpi_rank == 0 ? recvcounts_mandelbrot_flags.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_mandelbrot_flags[r] = current_displ;
            current_displ += recvcounts_mandelbrot_flags[r];
        }
        global_recv_globals_mandelbrot_flags.resize(current_displ);
        global_recv_values_mandelbrot_flags.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_mandelbrot_flags.data()), send_count_mandelbrot_flags, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_mandelbrot_flags.data() : nullptr, mpi_rank == 0 ? recvcounts_mandelbrot_flags.data() : nullptr, mpi_rank == 0 ? recvdispls_mandelbrot_flags.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_mandelbrot_flags.data(), send_count_mandelbrot_flags, MPI_INT, mpi_rank == 0 ? global_recv_values_mandelbrot_flags.data() : nullptr, mpi_rank == 0 ? recvcounts_mandelbrot_flags.data() : nullptr, mpi_rank == 0 ? recvdispls_mandelbrot_flags.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<int> global_out_mandelbrot_flags;
        mandelbrot_flags.tensor2Array(global_out_mandelbrot_flags);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_mandelbrot_flags, global_recv_globals_mandelbrot_flags, global_out_mandelbrot_flags);
        mandelbrot_flags.array2Tensor(global_out_mandelbrot_flags);
    } else {
    }
    auto dacpp_wrapper_end = std::chrono::steady_clock::now();
    double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();
    double dacpp_wrapper_max_ms = 0.0;
    MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0 && dacpp::mpi::profilingEnabled()) {
        std::fprintf(stderr, "[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\n", "MANDEL_mandel", dacpp_wrapper_max_ms);
    }
    dacpp::mpi::reportCollectPositionsProfile("MANDEL_mandel", MPI_COMM_WORLD);
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

    // 初始化复数点向量
    InitializeComplexPoints();

    // 计算 Mandelbrot 集
    mandelbrot_flags.resize(total_points, 0);  // 初始化一维数组为 0

    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);


    MANDEL_mandel(complex_points_tensor, mandelbrot_flags_tensor);

    // 统计数组中 1 的个数
    mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }

    // 打印统计信息
    PrintStats();

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
