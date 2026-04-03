#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}
// 参数设置
const double A = 1.0;  // 吸引力系数
const double D = 0.1;  // 扩散系数
const double dx = 0.1; // 空间步长
const double dt = 0.01; // 时间步长
const int N = 150;     // 空间网格点数
const int T = 1000;    // 时间步数

// 初始化用户偏好分布
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // 假设初始偏好为高斯分布
        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0); // 初始偏好分布中心在x=5
    }
}

// 归一化函数
void normalize(dacpp::Vector<double>& p) {
    double sum = 0.0;
    for (int i = 0;i < N-2; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N-2; i++) {
        p[i] /= sum; // 归一化    
    }
}





// 数值求解Fokker-Planck方程

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void mdp_mpi_local(dacpp::mpi::View1D<double> p, dacpp::mpi::View1D<double> new_p) {
    double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx);
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion + drift);
}


void mdp_shell_mdp(dacpp::Vector<double> & p, dacpp::Vector<double> & new_p) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_p;
    pattern_p.param_id = 0;
    pattern_p.name = "p";
    pattern_p.mode = dacpp::mpi::AccessMode::ReadWrite;
    pattern_p.data_info.dim = p.getDim();
    for (int dim = 0; dim < p.getDim(); ++dim) pattern_p.data_info.dimLength.push_back(p.getShape(dim));
    Dac_Op pattern_p_op_0;
    pattern_p_op_0.setDimId(0);
    pattern_p_op_0.size = 3;
    pattern_p_op_0.stride = 1;
    pattern_p_op_0.SetSplitSize((p.getShape(0) - 3) / 1 + 1);
    pattern_p.param_ops.push_back(pattern_p_op_0);
    pattern_p.bind_set_id.push_back(0);
    pattern_p.bind_offset_expr.push_back("0");
    pattern_p.is_index_op.push_back(false);
    pattern_p.partition_shape = dacpp::mpi::init_partition_shape(pattern_p);
    pattern_p.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_p);
    if (binding_split_sizes.size() < pattern_p.bind_split_sizes.size()) binding_split_sizes.resize(pattern_p.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_p.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_p.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_new_p;
    pattern_new_p.param_id = 1;
    pattern_new_p.name = "new_p";
    pattern_new_p.mode = dacpp::mpi::AccessMode::ReadWrite;
    pattern_new_p.data_info.dim = new_p.getDim();
    for (int dim = 0; dim < new_p.getDim(); ++dim) pattern_new_p.data_info.dimLength.push_back(new_p.getShape(dim));
    Dac_Op pattern_new_p_op_0;
    pattern_new_p_op_0.setDimId(0);
    pattern_new_p_op_0.size = 1;
    pattern_new_p_op_0.stride = 1;
    pattern_new_p_op_0.SetSplitSize(new_p.getShape(0));
    pattern_new_p.param_ops.push_back(pattern_new_p_op_0);
    pattern_new_p.bind_set_id.push_back(1);
    pattern_new_p.bind_offset_expr.push_back("0");
    pattern_new_p.is_index_op.push_back(true);
    pattern_new_p.partition_shape = dacpp::mpi::init_partition_shape(pattern_new_p);
    pattern_new_p.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_new_p);
    if (binding_split_sizes.size() < pattern_new_p.bind_split_sizes.size()) binding_split_sizes.resize(pattern_new_p.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_new_p.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_new_p.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_p.bind_split_sizes = binding_split_sizes;
    pattern_new_p.bind_split_sizes = binding_split_sizes;
    auto pack_p = dacpp::mpi::build_rw_pack_map(item_range, pattern_p);
    auto slots_p = dacpp::mpi::build_item_slots(item_range, pattern_p, pack_p);
    std::vector<double> local_p(pack_p.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_p;
        p.tensor2Array(global_p);
        local_p = dacpp::mpi::pack_values_by_globals(global_p, pack_p.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_rw_pack_map(peer_range, pattern_p);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_p, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_p.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_p.data(), recv_count, MPI_DOUBLE, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int p_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_p));
    auto pack_new_p = dacpp::mpi::build_rw_pack_map(item_range, pattern_new_p);
    auto slots_new_p = dacpp::mpi::build_item_slots(item_range, pattern_new_p, pack_new_p);
    std::vector<double> local_new_p(pack_new_p.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_new_p;
        new_p.tensor2Array(global_new_p);
        local_new_p = dacpp::mpi::pack_values_by_globals(global_new_p, pack_new_p.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_rw_pack_map(peer_range, pattern_new_p);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_new_p, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1001, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2001, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_new_p.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_new_p.data(), recv_count, MPI_DOUBLE, 0, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int new_p_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_new_p));
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_p(local_p.data(), sycl::range<1>(local_p.size()));
            sycl::buffer<int32_t, 1> slots_buffer_p(slots_p.data(), sycl::range<1>(slots_p.size()));
            sycl::buffer<double, 1> buffer_new_p(local_new_p.data(), sycl::range<1>(local_new_p.size()));
            sycl::buffer<int32_t, 1> slots_buffer_new_p(slots_new_p.data(), sycl::range<1>(slots_new_p.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_p = buffer_p.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_p = slots_buffer_p.get_access<sycl::access::mode::read>(h);
                auto acc_new_p = buffer_new_p.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_new_p = slots_buffer_new_p.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_p = acc_p.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_p = slots_acc_p.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_p{data_p, slots_p, item_linear * p_partition_size};
                    auto* data_new_p = acc_new_p.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_new_p = slots_acc_new_p.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_new_p{data_new_p, slots_new_p, item_linear * new_p_partition_size};
                    mdp_mpi_local(view_p, view_new_p);
                });
            });
            q.wait();
        }
    }
    auto writeback_p = dacpp::mpi::build_writeback_values(local_p, pack_p);
    const auto& writeback_globals_p = pack_p.writeback_globals.empty() ? pack_p.globals : pack_p.writeback_globals;
    std::vector<double> synced_p;
    if (mpi_rank == 0) {
        std::vector<double> global_out_p;
        p.tensor2Array(global_out_p);
        dacpp::mpi::apply_writeback_by_globals(writeback_p, writeback_globals_p, global_out_p);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<double> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_DOUBLE, peer, 5000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_p);
        }
        p.array2Tensor(global_out_p);
        synced_p = global_out_p;
    } else {
        int send_count = static_cast<int>(writeback_globals_p.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3000, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_p.data()), send_count, MPI_LONG_LONG, 0, 4000, MPI_COMM_WORLD);
            MPI_Send(writeback_p.data(), send_count, MPI_DOUBLE, 0, 5000, MPI_COMM_WORLD);
        }
    }
    int synced_count_p = 0;
    if (mpi_rank == 0) {
        synced_count_p = static_cast<int>(synced_p.size());
    }
    MPI_Bcast(&synced_count_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_p.resize(synced_count_p);
    }
    if (synced_count_p > 0) {
        MPI_Bcast(synced_p.data(), synced_count_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        p.array2Tensor(synced_p);
    }
    auto writeback_new_p = dacpp::mpi::build_writeback_values(local_new_p, pack_new_p);
    const auto& writeback_globals_new_p = pack_new_p.writeback_globals.empty() ? pack_new_p.globals : pack_new_p.writeback_globals;
    std::vector<double> synced_new_p;
    if (mpi_rank == 0) {
        std::vector<double> global_out_new_p;
        new_p.tensor2Array(global_out_new_p);
        dacpp::mpi::apply_writeback_by_globals(writeback_new_p, writeback_globals_new_p, global_out_new_p);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<double> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_DOUBLE, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_new_p);
        }
        new_p.array2Tensor(global_out_new_p);
        synced_new_p = global_out_new_p;
    } else {
        int send_count = static_cast<int>(writeback_globals_new_p.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_new_p.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_new_p.data(), send_count, MPI_DOUBLE, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_new_p = 0;
    if (mpi_rank == 0) {
        synced_count_new_p = static_cast<int>(synced_new_p.size());
    }
    MPI_Bcast(&synced_count_new_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_new_p.resize(synced_count_new_p);
    }
    if (synced_count_new_p > 0) {
        MPI_Bcast(synced_new_p.data(), synced_count_new_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        new_p.array2Tensor(synced_new_p);
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

    std::vector<double> p1(N, 0.0); // 存储用户偏好分布
    // 初始化偏好分布
    initialize(p1);
    // 数值求解Fokker-Planck方程
    std::vector<double> new_p1(N-2, 0.0); // 存储下一时间步的分布
    dacpp::Vector<double> p(p1);
    dacpp::Vector<double> new_p(new_p1);
    mdp_shell_mdp(p, new_p);
    
    std::cout << p[2] << std::endl;
    //p.print();
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
