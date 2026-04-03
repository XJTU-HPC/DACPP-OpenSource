#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"


namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 100;       // 路段长度
const double TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q(double rho) {
    double V_max = 30; // 最大速度
    double rho_max = 50; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

// 计算交通流量




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void lwr_mpi_local(dacpp::mpi::View1D<double> rho, dacpp::mpi::View1D<double> new_rho) {
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0., new_rho[0]);
}


void LWR_shell_lwr(dacpp::Vector<double> & rho, dacpp::Vector<double> & new_rho) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_rho;
    pattern_rho.param_id = 0;
    pattern_rho.name = "rho";
    pattern_rho.mode = dacpp::mpi::AccessMode::ReadWrite;
    pattern_rho.data_info.dim = rho.getDim();
    for (int dim = 0; dim < rho.getDim(); ++dim) pattern_rho.data_info.dimLength.push_back(rho.getShape(dim));
    Dac_Op pattern_rho_op_0;
    pattern_rho_op_0.setDimId(0);
    pattern_rho_op_0.size = 2;
    pattern_rho_op_0.stride = 1;
    pattern_rho_op_0.SetSplitSize((rho.getShape(0) - 2) / 1 + 1);
    pattern_rho.param_ops.push_back(pattern_rho_op_0);
    pattern_rho.bind_set_id.push_back(0);
    pattern_rho.bind_offset_expr.push_back("0");
    pattern_rho.is_index_op.push_back(false);
    pattern_rho.partition_shape = dacpp::mpi::init_partition_shape(pattern_rho);
    pattern_rho.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_rho);
    if (binding_split_sizes.size() < pattern_rho.bind_split_sizes.size()) binding_split_sizes.resize(pattern_rho.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_rho.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_rho.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_new_rho;
    pattern_new_rho.param_id = 1;
    pattern_new_rho.name = "new_rho";
    pattern_new_rho.mode = dacpp::mpi::AccessMode::ReadWrite;
    pattern_new_rho.data_info.dim = new_rho.getDim();
    for (int dim = 0; dim < new_rho.getDim(); ++dim) pattern_new_rho.data_info.dimLength.push_back(new_rho.getShape(dim));
    Dac_Op pattern_new_rho_op_0;
    pattern_new_rho_op_0.setDimId(0);
    pattern_new_rho_op_0.size = 1;
    pattern_new_rho_op_0.stride = 1;
    pattern_new_rho_op_0.SetSplitSize(new_rho.getShape(0));
    pattern_new_rho.param_ops.push_back(pattern_new_rho_op_0);
    pattern_new_rho.bind_set_id.push_back(1);
    pattern_new_rho.bind_offset_expr.push_back("0");
    pattern_new_rho.is_index_op.push_back(true);
    pattern_new_rho.partition_shape = dacpp::mpi::init_partition_shape(pattern_new_rho);
    pattern_new_rho.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_new_rho);
    if (binding_split_sizes.size() < pattern_new_rho.bind_split_sizes.size()) binding_split_sizes.resize(pattern_new_rho.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_new_rho.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_new_rho.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_rho.bind_split_sizes = binding_split_sizes;
    pattern_new_rho.bind_split_sizes = binding_split_sizes;
    auto pack_rho = dacpp::mpi::build_rw_pack_map(item_range, pattern_rho);
    auto slots_rho = dacpp::mpi::build_item_slots(item_range, pattern_rho, pack_rho);
    std::vector<double> local_rho(pack_rho.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_rho;
        rho.tensor2Array(global_rho);
        local_rho = dacpp::mpi::pack_values_by_globals(global_rho, pack_rho.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_rw_pack_map(peer_range, pattern_rho);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_rho, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_rho.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_rho.data(), recv_count, MPI_DOUBLE, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int rho_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_rho));
    auto pack_new_rho = dacpp::mpi::build_rw_pack_map(item_range, pattern_new_rho);
    auto slots_new_rho = dacpp::mpi::build_item_slots(item_range, pattern_new_rho, pack_new_rho);
    std::vector<double> local_new_rho(pack_new_rho.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_new_rho;
        new_rho.tensor2Array(global_new_rho);
        local_new_rho = dacpp::mpi::pack_values_by_globals(global_new_rho, pack_new_rho.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_rw_pack_map(peer_range, pattern_new_rho);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_new_rho, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1001, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2001, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_new_rho.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_new_rho.data(), recv_count, MPI_DOUBLE, 0, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int new_rho_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_new_rho));
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_rho(local_rho.data(), sycl::range<1>(local_rho.size()));
            sycl::buffer<int32_t, 1> slots_buffer_rho(slots_rho.data(), sycl::range<1>(slots_rho.size()));
            sycl::buffer<double, 1> buffer_new_rho(local_new_rho.data(), sycl::range<1>(local_new_rho.size()));
            sycl::buffer<int32_t, 1> slots_buffer_new_rho(slots_new_rho.data(), sycl::range<1>(slots_new_rho.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_rho = buffer_rho.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_rho = slots_buffer_rho.get_access<sycl::access::mode::read>(h);
                auto acc_new_rho = buffer_new_rho.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_new_rho = slots_buffer_new_rho.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_rho = acc_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_rho = slots_acc_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_rho{data_rho, slots_rho, item_linear * rho_partition_size};
                    auto* data_new_rho = acc_new_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_new_rho = slots_acc_new_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_new_rho{data_new_rho, slots_new_rho, item_linear * new_rho_partition_size};
                    lwr_mpi_local(view_rho, view_new_rho);
                });
            });
            q.wait();
        }
    }
    auto writeback_rho = dacpp::mpi::build_writeback_values(local_rho, pack_rho);
    const auto& writeback_globals_rho = pack_rho.writeback_globals.empty() ? pack_rho.globals : pack_rho.writeback_globals;
    std::vector<double> synced_rho;
    if (mpi_rank == 0) {
        std::vector<double> global_out_rho;
        rho.tensor2Array(global_out_rho);
        dacpp::mpi::apply_writeback_by_globals(writeback_rho, writeback_globals_rho, global_out_rho);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<double> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_DOUBLE, peer, 5000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_rho);
        }
        rho.array2Tensor(global_out_rho);
        synced_rho = global_out_rho;
    } else {
        int send_count = static_cast<int>(writeback_globals_rho.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3000, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_rho.data()), send_count, MPI_LONG_LONG, 0, 4000, MPI_COMM_WORLD);
            MPI_Send(writeback_rho.data(), send_count, MPI_DOUBLE, 0, 5000, MPI_COMM_WORLD);
        }
    }
    int synced_count_rho = 0;
    if (mpi_rank == 0) {
        synced_count_rho = static_cast<int>(synced_rho.size());
    }
    MPI_Bcast(&synced_count_rho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_rho.resize(synced_count_rho);
    }
    if (synced_count_rho > 0) {
        MPI_Bcast(synced_rho.data(), synced_count_rho, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        rho.array2Tensor(synced_rho);
    }
    auto writeback_new_rho = dacpp::mpi::build_writeback_values(local_new_rho, pack_new_rho);
    const auto& writeback_globals_new_rho = pack_new_rho.writeback_globals.empty() ? pack_new_rho.globals : pack_new_rho.writeback_globals;
    std::vector<double> synced_new_rho;
    if (mpi_rank == 0) {
        std::vector<double> global_out_new_rho;
        new_rho.tensor2Array(global_out_new_rho);
        dacpp::mpi::apply_writeback_by_globals(writeback_new_rho, writeback_globals_new_rho, global_out_new_rho);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<double> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_DOUBLE, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_new_rho);
        }
        new_rho.array2Tensor(global_out_new_rho);
        synced_new_rho = global_out_new_rho;
    } else {
        int send_count = static_cast<int>(writeback_globals_new_rho.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_new_rho.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_new_rho.data(), send_count, MPI_DOUBLE, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_new_rho = 0;
    if (mpi_rank == 0) {
        synced_count_new_rho = static_cast<int>(synced_new_rho.size());
    }
    MPI_Bcast(&synced_count_new_rho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_new_rho.resize(synced_count_new_rho);
    }
    if (synced_count_new_rho > 0) {
        MPI_Bcast(synced_new_rho.data(), synced_count_new_rho, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        new_rho.array2Tensor(synced_new_rho);
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

    // 创建 Tensor 类型对象
    std::vector<double> rho1(WIDTH, 0.0);
    std::vector<double> new_rho1(WIDTH, 0.0);
    initializeDensity(rho1);
    dacpp::Vector<double> rho_tensor(rho1);
    dacpp::Vector<double> new_rho_tensor(new_rho1);
    dacpp::Vector<double> new_rho = new_rho_tensor[{1,WIDTH-1}];
    dacpp::Vector<double> rho = rho_tensor[{0,WIDTH-1}];
    LWR_shell_lwr(rho, new_rho);
    
    std::cout << rho[15] << std::endl;
    

    

    // 释放动态分配的内存

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
