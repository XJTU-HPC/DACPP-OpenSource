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
    auto pack_complex_points = dacpp::mpi::build_input_pack_map(item_range, pattern_complex_points);
    auto slots_complex_points = dacpp::mpi::build_item_slots(item_range, pattern_complex_points, pack_complex_points);
    std::vector<complex<float>> local_complex_points(pack_complex_points.globals.size());
    if (mpi_rank == 0) {
        std::vector<complex<float>> global_complex_points;
        complex_points.tensor2Array(global_complex_points);
        local_complex_points = dacpp::mpi::pack_values_by_globals(global_complex_points, pack_complex_points.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_complex_points);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_complex_points, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_C_FLOAT_COMPLEX, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_complex_points.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_complex_points.data(), recv_count, MPI_C_FLOAT_COMPLEX, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int complex_points_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_complex_points));
    auto pack_mandelbrot_flags = dacpp::mpi::build_output_pack_map(item_range, pattern_mandelbrot_flags);
    auto slots_mandelbrot_flags = dacpp::mpi::build_item_slots(item_range, pattern_mandelbrot_flags, pack_mandelbrot_flags);
    std::vector<int> local_mandelbrot_flags(pack_mandelbrot_flags.globals.size());
    const int mandelbrot_flags_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_mandelbrot_flags));
    if (local_item_count > 0) {
        {
            sycl::buffer<complex<float>, 1> buffer_complex_points(local_complex_points.data(), sycl::range<1>(local_complex_points.size()));
            sycl::buffer<int32_t, 1> slots_buffer_complex_points(slots_complex_points.data(), sycl::range<1>(slots_complex_points.size()));
            sycl::buffer<int, 1> buffer_mandelbrot_flags(local_mandelbrot_flags.data(), sycl::range<1>(local_mandelbrot_flags.size()));
            sycl::buffer<int32_t, 1> slots_buffer_mandelbrot_flags(slots_mandelbrot_flags.data(), sycl::range<1>(slots_mandelbrot_flags.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_complex_points = buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto slots_acc_complex_points = slots_buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto acc_mandelbrot_flags = buffer_mandelbrot_flags.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_mandelbrot_flags = slots_buffer_mandelbrot_flags.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_complex_points = acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_complex_points = slots_acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const complex<float>> view_complex_points{data_complex_points, slots_complex_points, item_linear * complex_points_partition_size};
                    auto* data_mandelbrot_flags = acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_mandelbrot_flags = slots_acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<int> view_mandelbrot_flags{data_mandelbrot_flags, slots_mandelbrot_flags, item_linear * mandelbrot_flags_partition_size};
                    mandel_mpi_local(view_complex_points, view_mandelbrot_flags);
                });
            });
            q.wait();
        }
    }
    auto writeback_mandelbrot_flags = dacpp::mpi::build_writeback_values(local_mandelbrot_flags, pack_mandelbrot_flags);
    const auto& writeback_globals_mandelbrot_flags = pack_mandelbrot_flags.writeback_globals.empty() ? pack_mandelbrot_flags.globals : pack_mandelbrot_flags.writeback_globals;
    std::vector<int> synced_mandelbrot_flags;
    if (mpi_rank == 0) {
        std::vector<int> global_out_mandelbrot_flags;
        mandelbrot_flags.tensor2Array(global_out_mandelbrot_flags);
        dacpp::mpi::apply_writeback_by_globals(writeback_mandelbrot_flags, writeback_globals_mandelbrot_flags, global_out_mandelbrot_flags);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<int> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_INT, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_mandelbrot_flags);
        }
        mandelbrot_flags.array2Tensor(global_out_mandelbrot_flags);
        synced_mandelbrot_flags = global_out_mandelbrot_flags;
    } else {
        int send_count = static_cast<int>(writeback_globals_mandelbrot_flags.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_mandelbrot_flags.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_mandelbrot_flags.data(), send_count, MPI_INT, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_mandelbrot_flags = 0;
    if (mpi_rank == 0) {
        synced_count_mandelbrot_flags = static_cast<int>(synced_mandelbrot_flags.size());
    }
    MPI_Bcast(&synced_count_mandelbrot_flags, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_mandelbrot_flags.resize(synced_count_mandelbrot_flags);
    }
    if (synced_count_mandelbrot_flags > 0) {
        MPI_Bcast(synced_mandelbrot_flags.data(), synced_count_mandelbrot_flags, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        mandelbrot_flags.array2Tensor(synced_mandelbrot_flags);
    }
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
    if (mpi_rank != 0) {
        freopen("/dev/null", "w", stdout);
    }

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
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
