#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1
// 定义矩阵大小
const int N = 10; // 可以修改 N 的值来改变矩阵大小
const int max_iter = 100;
const float tolerance = 1e-6;
namespace dacpp {
    typedef std::vector<std::any> list;
}







#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void jacobi_mpi_local(dacpp::mpi::View1D<const float> a, dacpp::mpi::View1D<const float> b, dacpp::mpi::View1D<const float> x, dacpp::mpi::View1D<float> x_new, dacpp::mpi::View1D<const int> num) {
    float sigma = 0;
    for (int i = 0; i < N; ++i) {
        if (i != num[0]) {
            sigma += a[i] * x[i];
        }
    }
    x_new[0] = (b[0] - sigma) / a[num[0]];
}


void jacobiShell_jacobi(const dacpp::Matrix<float> & A, const dacpp::Vector<float> & b, const dacpp::Vector<float> & x, dacpp::Vector<float> & x_new, const dacpp::Vector<int> & nums) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_a;
    pattern_a.param_id = 0;
    pattern_a.name = "a";
    pattern_a.mode = dacpp::mpi::AccessMode::Read;
    pattern_a.data_info.dim = A.getDim();
    for (int dim = 0; dim < A.getDim(); ++dim) pattern_a.data_info.dimLength.push_back(A.getShape(dim));
    Dac_Op pattern_a_op_0;
    pattern_a_op_0.setDimId(0);
    pattern_a_op_0.size = 1;
    pattern_a_op_0.stride = 1;
    pattern_a_op_0.SetSplitSize(A.getShape(0));
    pattern_a.param_ops.push_back(pattern_a_op_0);
    pattern_a.bind_set_id.push_back(0);
    pattern_a.bind_offset_expr.push_back("0");
    pattern_a.is_index_op.push_back(true);
    pattern_a.partition_shape = dacpp::mpi::init_partition_shape(pattern_a);
    pattern_a.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_a);
    if (binding_split_sizes.size() < pattern_a.bind_split_sizes.size()) binding_split_sizes.resize(pattern_a.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_a.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_a.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_b;
    pattern_b.param_id = 1;
    pattern_b.name = "b";
    pattern_b.mode = dacpp::mpi::AccessMode::Read;
    pattern_b.data_info.dim = b.getDim();
    for (int dim = 0; dim < b.getDim(); ++dim) pattern_b.data_info.dimLength.push_back(b.getShape(dim));
    Dac_Op pattern_b_op_0;
    pattern_b_op_0.setDimId(0);
    pattern_b_op_0.size = 1;
    pattern_b_op_0.stride = 1;
    pattern_b_op_0.SetSplitSize(b.getShape(0));
    pattern_b.param_ops.push_back(pattern_b_op_0);
    pattern_b.bind_set_id.push_back(0);
    pattern_b.bind_offset_expr.push_back("0");
    pattern_b.is_index_op.push_back(true);
    pattern_b.partition_shape = dacpp::mpi::init_partition_shape(pattern_b);
    pattern_b.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_b);
    if (binding_split_sizes.size() < pattern_b.bind_split_sizes.size()) binding_split_sizes.resize(pattern_b.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_b.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_b.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_x;
    pattern_x.param_id = 2;
    pattern_x.name = "x";
    pattern_x.mode = dacpp::mpi::AccessMode::Read;
    pattern_x.data_info.dim = x.getDim();
    for (int dim = 0; dim < x.getDim(); ++dim) pattern_x.data_info.dimLength.push_back(x.getShape(dim));
    pattern_x.partition_shape = dacpp::mpi::init_partition_shape(pattern_x);
    pattern_x.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_x);
    if (binding_split_sizes.size() < pattern_x.bind_split_sizes.size()) binding_split_sizes.resize(pattern_x.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_x.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_x.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_x_new;
    pattern_x_new.param_id = 3;
    pattern_x_new.name = "x_new";
    pattern_x_new.mode = dacpp::mpi::AccessMode::Write;
    pattern_x_new.data_info.dim = x_new.getDim();
    for (int dim = 0; dim < x_new.getDim(); ++dim) pattern_x_new.data_info.dimLength.push_back(x_new.getShape(dim));
    Dac_Op pattern_x_new_op_0;
    pattern_x_new_op_0.setDimId(0);
    pattern_x_new_op_0.size = 1;
    pattern_x_new_op_0.stride = 1;
    pattern_x_new_op_0.SetSplitSize(x_new.getShape(0));
    pattern_x_new.param_ops.push_back(pattern_x_new_op_0);
    pattern_x_new.bind_set_id.push_back(0);
    pattern_x_new.bind_offset_expr.push_back("0");
    pattern_x_new.is_index_op.push_back(true);
    pattern_x_new.partition_shape = dacpp::mpi::init_partition_shape(pattern_x_new);
    pattern_x_new.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_x_new);
    if (binding_split_sizes.size() < pattern_x_new.bind_split_sizes.size()) binding_split_sizes.resize(pattern_x_new.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_x_new.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_x_new.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_num;
    pattern_num.param_id = 4;
    pattern_num.name = "num";
    pattern_num.mode = dacpp::mpi::AccessMode::Read;
    pattern_num.data_info.dim = nums.getDim();
    for (int dim = 0; dim < nums.getDim(); ++dim) pattern_num.data_info.dimLength.push_back(nums.getShape(dim));
    Dac_Op pattern_num_op_0;
    pattern_num_op_0.setDimId(0);
    pattern_num_op_0.size = 1;
    pattern_num_op_0.stride = 1;
    pattern_num_op_0.SetSplitSize(nums.getShape(0));
    pattern_num.param_ops.push_back(pattern_num_op_0);
    pattern_num.bind_set_id.push_back(0);
    pattern_num.bind_offset_expr.push_back("0");
    pattern_num.is_index_op.push_back(true);
    pattern_num.partition_shape = dacpp::mpi::init_partition_shape(pattern_num);
    pattern_num.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_num);
    if (binding_split_sizes.size() < pattern_num.bind_split_sizes.size()) binding_split_sizes.resize(pattern_num.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_num.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_num.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_a.bind_split_sizes = binding_split_sizes;
    pattern_b.bind_split_sizes = binding_split_sizes;
    pattern_x.bind_split_sizes = binding_split_sizes;
    pattern_x_new.bind_split_sizes = binding_split_sizes;
    pattern_num.bind_split_sizes = binding_split_sizes;
    auto pack_a = dacpp::mpi::build_input_pack_map(item_range, pattern_a);
    auto slots_a = dacpp::mpi::build_item_slots(item_range, pattern_a, pack_a);
    std::vector<float> local_a(pack_a.globals.size());
    if (mpi_rank == 0) {
        std::vector<float> global_a;
        A.tensor2Array(global_a);
        local_a = dacpp::mpi::pack_values_by_globals(global_a, pack_a.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_a);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_a, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_FLOAT, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_a.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_a.data(), recv_count, MPI_FLOAT, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int a_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_a));
    auto pack_b = dacpp::mpi::build_input_pack_map(item_range, pattern_b);
    auto slots_b = dacpp::mpi::build_item_slots(item_range, pattern_b, pack_b);
    std::vector<float> local_b(pack_b.globals.size());
    if (mpi_rank == 0) {
        std::vector<float> global_b;
        b.tensor2Array(global_b);
        local_b = dacpp::mpi::pack_values_by_globals(global_b, pack_b.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_b);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_b, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1001, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_FLOAT, peer, 2001, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_b.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_b.data(), recv_count, MPI_FLOAT, 0, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int b_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_b));
    auto pack_x = dacpp::mpi::build_input_pack_map(item_range, pattern_x);
    auto slots_x = dacpp::mpi::build_item_slots(item_range, pattern_x, pack_x);
    std::vector<float> local_x(pack_x.globals.size());
    if (mpi_rank == 0) {
        std::vector<float> global_x;
        x.tensor2Array(global_x);
        local_x = dacpp::mpi::pack_values_by_globals(global_x, pack_x.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_x);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_x, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1002, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_FLOAT, peer, 2002, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_x.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_x.data(), recv_count, MPI_FLOAT, 0, 2002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int x_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_x));
    auto pack_x_new = dacpp::mpi::build_output_pack_map(item_range, pattern_x_new);
    auto slots_x_new = dacpp::mpi::build_item_slots(item_range, pattern_x_new, pack_x_new);
    std::vector<float> local_x_new(pack_x_new.globals.size());
    const int x_new_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_x_new));
    auto pack_num = dacpp::mpi::build_input_pack_map(item_range, pattern_num);
    auto slots_num = dacpp::mpi::build_item_slots(item_range, pattern_num, pack_num);
    std::vector<int> local_num(pack_num.globals.size());
    if (mpi_rank == 0) {
        std::vector<int> global_num;
        nums.tensor2Array(global_num);
        local_num = dacpp::mpi::pack_values_by_globals(global_num, pack_num.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_num);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_num, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1004, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_INT, peer, 2004, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1004, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_num.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_num.data(), recv_count, MPI_INT, 0, 2004, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int num_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_num));
    if (local_item_count > 0) {
        {
            sycl::buffer<float, 1> buffer_a(local_a.data(), sycl::range<1>(local_a.size()));
            sycl::buffer<int32_t, 1> slots_buffer_a(slots_a.data(), sycl::range<1>(slots_a.size()));
            sycl::buffer<float, 1> buffer_b(local_b.data(), sycl::range<1>(local_b.size()));
            sycl::buffer<int32_t, 1> slots_buffer_b(slots_b.data(), sycl::range<1>(slots_b.size()));
            sycl::buffer<float, 1> buffer_x(local_x.data(), sycl::range<1>(local_x.size()));
            sycl::buffer<int32_t, 1> slots_buffer_x(slots_x.data(), sycl::range<1>(slots_x.size()));
            sycl::buffer<float, 1> buffer_x_new(local_x_new.data(), sycl::range<1>(local_x_new.size()));
            sycl::buffer<int32_t, 1> slots_buffer_x_new(slots_x_new.data(), sycl::range<1>(slots_x_new.size()));
            sycl::buffer<int, 1> buffer_num(local_num.data(), sycl::range<1>(local_num.size()));
            sycl::buffer<int32_t, 1> slots_buffer_num(slots_num.data(), sycl::range<1>(slots_num.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_a = buffer_a.get_access<sycl::access::mode::read>(h);
                auto slots_acc_a = slots_buffer_a.get_access<sycl::access::mode::read>(h);
                auto acc_b = buffer_b.get_access<sycl::access::mode::read>(h);
                auto slots_acc_b = slots_buffer_b.get_access<sycl::access::mode::read>(h);
                auto acc_x = buffer_x.get_access<sycl::access::mode::read>(h);
                auto slots_acc_x = slots_buffer_x.get_access<sycl::access::mode::read>(h);
                auto acc_x_new = buffer_x_new.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_x_new = slots_buffer_x_new.get_access<sycl::access::mode::read>(h);
                auto acc_num = buffer_num.get_access<sycl::access::mode::read>(h);
                auto slots_acc_num = slots_buffer_num.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_a = acc_a.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_a = slots_acc_a.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_a{data_a, slots_a, item_linear * a_partition_size};
                    auto* data_b = acc_b.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_b = slots_acc_b.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_b{data_b, slots_b, item_linear * b_partition_size};
                    auto* data_x = acc_x.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_x = slots_acc_x.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_x{data_x, slots_x, item_linear * x_partition_size};
                    auto* data_x_new = acc_x_new.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_x_new = slots_acc_x_new.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<float> view_x_new{data_x_new, slots_x_new, item_linear * x_new_partition_size};
                    auto* data_num = acc_num.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_num = slots_acc_num.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_num{data_num, slots_num, item_linear * num_partition_size};
                    jacobi_mpi_local(view_a, view_b, view_x, view_x_new, view_num);
                });
            });
            q.wait();
        }
    }
    auto writeback_x_new = dacpp::mpi::build_writeback_values(local_x_new, pack_x_new);
    const auto& writeback_globals_x_new = pack_x_new.writeback_globals.empty() ? pack_x_new.globals : pack_x_new.writeback_globals;
    std::vector<float> synced_x_new;
    if (mpi_rank == 0) {
        std::vector<float> global_out_x_new;
        x_new.tensor2Array(global_out_x_new);
        dacpp::mpi::apply_writeback_by_globals(writeback_x_new, writeback_globals_x_new, global_out_x_new);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3003, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<float> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4003, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_FLOAT, peer, 5003, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_x_new);
        }
        x_new.array2Tensor(global_out_x_new);
        synced_x_new = global_out_x_new;
    } else {
        int send_count = static_cast<int>(writeback_globals_x_new.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3003, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_x_new.data()), send_count, MPI_LONG_LONG, 0, 4003, MPI_COMM_WORLD);
            MPI_Send(writeback_x_new.data(), send_count, MPI_FLOAT, 0, 5003, MPI_COMM_WORLD);
        }
    }
    int synced_count_x_new = 0;
    if (mpi_rank == 0) {
        synced_count_x_new = static_cast<int>(synced_x_new.size());
    }
    MPI_Bcast(&synced_count_x_new, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_x_new.resize(synced_count_x_new);
    }
    if (synced_count_x_new > 0) {
        MPI_Bcast(synced_x_new.data(), synced_count_x_new, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        x_new.array2Tensor(synced_x_new);
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



    // 初始化系数矩阵 A 和向量 b
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> vec_b(N, 0.0f);
    std::vector<float> vec_x(N, 0.0f);     // 初始解
    std::vector<float> vec_x_new(N, 0.0f); // 更新后的解

    // 自动初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        mat_A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

        if (i > 0) {
            mat_A[i * N + i - 1] = -1.0f; // 下三角元素
        }
        if (i < N - 1) {
            mat_A[i * N + i + 1] = -1.0f; // 上三角元素
        }

        vec_b[i] = 1.0f; // 初始化向量 b，可根据需要修改
    }

    // std::vector<int> A_shape = {100, 100};
    // std::vector<int> b_shape = {100};
    // std::vector<int> x_shape = {100};
    // std::vector<int> x_new_shape = {100};
    dacpp::Matrix<float> A({N, N}, mat_A);
    dacpp::Vector<float> b(vec_b);
    dacpp::Vector<float> x(vec_x);
    dacpp::Vector<float> x_new(vec_x_new);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums(N);
    // 使用 std::iota 填充 nums，值从 0 开始
    for(int i = 0;i < N;  i++){
        nums[i] = i;
    }
    //std::vector<int> nums_shape = {100};
    dacpp::Vector<int> tensor_nums(nums);
    float* data = new float[1 * N];
    float* data2 = new float[1 * N];

    while (!converged && iter < max_iter) {
        jacobiShell_jacobi(A, b, x, x_new, tensor_nums);
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x
        x=x_new;

        ++iter;
    }


    // 输出结果
    //std::cout << "迭代次数: " << iter << std::endl;
    //std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
