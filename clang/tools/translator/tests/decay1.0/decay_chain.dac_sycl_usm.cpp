#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}


const double dt = 0.1;       // 时间步长
const double T = 5.0;       // 总时间
const size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）




// 计算每种同位素在时间 t 的数量
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void decay_mpi_local(dacpp::mpi::View1D<const double> N0s, dacpp::mpi::View1D<const double> lambdas, dacpp::mpi::View1D<double> local_A, dacpp::mpi::View1D<const double> t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}


void DECAY_decay(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_N0s;
    pattern_N0s.param_id = 0;
    pattern_N0s.name = "N0s";
    pattern_N0s.mode = dacpp::mpi::AccessMode::Read;
    pattern_N0s.data_info.dim = N0s.getDim();
    for (int dim = 0; dim < N0s.getDim(); ++dim) pattern_N0s.data_info.dimLength.push_back(N0s.getShape(dim));
    Dac_Op pattern_N0s_op_0;
    pattern_N0s_op_0.setDimId(0);
    pattern_N0s_op_0.size = 1;
    pattern_N0s_op_0.stride = 1;
    pattern_N0s_op_0.SetSplitSize(N0s.getShape(0));
    pattern_N0s.param_ops.push_back(pattern_N0s_op_0);
    pattern_N0s.bind_set_id.push_back(0);
    pattern_N0s.bind_offset_expr.push_back("0");
    pattern_N0s.is_index_op.push_back(true);
    pattern_N0s.partition_shape = dacpp::mpi::init_partition_shape(pattern_N0s);
    pattern_N0s.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_N0s);
    if (binding_split_sizes.size() < pattern_N0s.bind_split_sizes.size()) binding_split_sizes.resize(pattern_N0s.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_N0s.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_N0s.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_lambdas;
    pattern_lambdas.param_id = 1;
    pattern_lambdas.name = "lambdas";
    pattern_lambdas.mode = dacpp::mpi::AccessMode::Read;
    pattern_lambdas.data_info.dim = lambdas.getDim();
    for (int dim = 0; dim < lambdas.getDim(); ++dim) pattern_lambdas.data_info.dimLength.push_back(lambdas.getShape(dim));
    Dac_Op pattern_lambdas_op_0;
    pattern_lambdas_op_0.setDimId(0);
    pattern_lambdas_op_0.size = 1;
    pattern_lambdas_op_0.stride = 1;
    pattern_lambdas_op_0.SetSplitSize(lambdas.getShape(0));
    pattern_lambdas.param_ops.push_back(pattern_lambdas_op_0);
    pattern_lambdas.bind_set_id.push_back(0);
    pattern_lambdas.bind_offset_expr.push_back("0");
    pattern_lambdas.is_index_op.push_back(true);
    pattern_lambdas.partition_shape = dacpp::mpi::init_partition_shape(pattern_lambdas);
    pattern_lambdas.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_lambdas);
    if (binding_split_sizes.size() < pattern_lambdas.bind_split_sizes.size()) binding_split_sizes.resize(pattern_lambdas.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_lambdas.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_lambdas.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_local_A;
    pattern_local_A.param_id = 2;
    pattern_local_A.name = "local_A";
    pattern_local_A.mode = dacpp::mpi::AccessMode::Write;
    pattern_local_A.data_info.dim = local_A.getDim();
    for (int dim = 0; dim < local_A.getDim(); ++dim) pattern_local_A.data_info.dimLength.push_back(local_A.getShape(dim));
    Dac_Op pattern_local_A_op_0;
    pattern_local_A_op_0.setDimId(0);
    pattern_local_A_op_0.size = 1;
    pattern_local_A_op_0.stride = 1;
    pattern_local_A_op_0.SetSplitSize(local_A.getShape(0));
    pattern_local_A.param_ops.push_back(pattern_local_A_op_0);
    pattern_local_A.bind_set_id.push_back(0);
    pattern_local_A.bind_offset_expr.push_back("0");
    pattern_local_A.is_index_op.push_back(true);
    pattern_local_A.partition_shape = dacpp::mpi::init_partition_shape(pattern_local_A);
    pattern_local_A.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_local_A);
    if (binding_split_sizes.size() < pattern_local_A.bind_split_sizes.size()) binding_split_sizes.resize(pattern_local_A.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_local_A.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_local_A.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_t;
    pattern_t.param_id = 3;
    pattern_t.name = "t";
    pattern_t.mode = dacpp::mpi::AccessMode::Read;
    pattern_t.data_info.dim = t.getDim();
    for (int dim = 0; dim < t.getDim(); ++dim) pattern_t.data_info.dimLength.push_back(t.getShape(dim));
    pattern_t.partition_shape = dacpp::mpi::init_partition_shape(pattern_t);
    pattern_t.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_t);
    if (binding_split_sizes.size() < pattern_t.bind_split_sizes.size()) binding_split_sizes.resize(pattern_t.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_t.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_t.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_N0s.bind_split_sizes = binding_split_sizes;
    pattern_lambdas.bind_split_sizes = binding_split_sizes;
    pattern_local_A.bind_split_sizes = binding_split_sizes;
    pattern_t.bind_split_sizes = binding_split_sizes;
    auto pack_N0s = dacpp::mpi::build_input_pack_map(item_range, pattern_N0s);
    auto slots_N0s = dacpp::mpi::build_item_slots(item_range, pattern_N0s, pack_N0s);
    std::vector<double> local_N0s(pack_N0s.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_N0s;
        N0s.tensor2Array(global_N0s);
        local_N0s = dacpp::mpi::pack_values_by_globals(global_N0s, pack_N0s.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_N0s);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_N0s, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_N0s.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_N0s.data(), recv_count, MPI_DOUBLE, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int N0s_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_N0s));
    auto pack_lambdas = dacpp::mpi::build_input_pack_map(item_range, pattern_lambdas);
    auto slots_lambdas = dacpp::mpi::build_item_slots(item_range, pattern_lambdas, pack_lambdas);
    std::vector<double> local_lambdas(pack_lambdas.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_lambdas;
        lambdas.tensor2Array(global_lambdas);
        local_lambdas = dacpp::mpi::pack_values_by_globals(global_lambdas, pack_lambdas.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_lambdas);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_lambdas, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1001, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2001, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_lambdas.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_lambdas.data(), recv_count, MPI_DOUBLE, 0, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int lambdas_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_lambdas));
    auto pack_local_A = dacpp::mpi::build_output_pack_map(item_range, pattern_local_A);
    auto slots_local_A = dacpp::mpi::build_item_slots(item_range, pattern_local_A, pack_local_A);
    std::vector<double> local_local_A(pack_local_A.globals.size());
    const int local_A_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_local_A));
    auto pack_t = dacpp::mpi::build_input_pack_map(item_range, pattern_t);
    auto slots_t = dacpp::mpi::build_item_slots(item_range, pattern_t, pack_t);
    std::vector<double> local_t(pack_t.globals.size());
    if (mpi_rank == 0) {
        std::vector<double> global_t;
        t.tensor2Array(global_t);
        local_t = dacpp::mpi::pack_values_by_globals(global_t, pack_t.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_t);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_t, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1003, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_DOUBLE, peer, 2003, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1003, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_t.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_t.data(), recv_count, MPI_DOUBLE, 0, 2003, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int t_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_t));
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_N0s(local_N0s.data(), sycl::range<1>(local_N0s.size()));
            sycl::buffer<int32_t, 1> slots_buffer_N0s(slots_N0s.data(), sycl::range<1>(slots_N0s.size()));
            sycl::buffer<double, 1> buffer_lambdas(local_lambdas.data(), sycl::range<1>(local_lambdas.size()));
            sycl::buffer<int32_t, 1> slots_buffer_lambdas(slots_lambdas.data(), sycl::range<1>(slots_lambdas.size()));
            sycl::buffer<double, 1> buffer_local_A(local_local_A.data(), sycl::range<1>(local_local_A.size()));
            sycl::buffer<int32_t, 1> slots_buffer_local_A(slots_local_A.data(), sycl::range<1>(slots_local_A.size()));
            sycl::buffer<double, 1> buffer_t(local_t.data(), sycl::range<1>(local_t.size()));
            sycl::buffer<int32_t, 1> slots_buffer_t(slots_t.data(), sycl::range<1>(slots_t.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_N0s = buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto slots_acc_N0s = slots_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto acc_lambdas = buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto slots_acc_lambdas = slots_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto acc_local_A = buffer_local_A.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_local_A = slots_buffer_local_A.get_access<sycl::access::mode::read>(h);
                auto acc_t = buffer_t.get_access<sycl::access::mode::read>(h);
                auto slots_acc_t = slots_buffer_t.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_N0s = acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_N0s = slots_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_N0s{data_N0s, slots_N0s, item_linear * N0s_partition_size};
                    auto* data_lambdas = acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_lambdas = slots_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_lambdas{data_lambdas, slots_lambdas, item_linear * lambdas_partition_size};
                    auto* data_local_A = acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_local_A = slots_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_local_A{data_local_A, slots_local_A, item_linear * local_A_partition_size};
                    auto* data_t = acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_t = slots_acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_t{data_t, slots_t, item_linear * t_partition_size};
                    decay_mpi_local(view_N0s, view_lambdas, view_local_A, view_t);
                });
            });
            q.wait();
        }
    }
    auto writeback_local_A = dacpp::mpi::build_writeback_values(local_local_A, pack_local_A);
    const auto& writeback_globals_local_A = pack_local_A.writeback_globals.empty() ? pack_local_A.globals : pack_local_A.writeback_globals;
    std::vector<double> synced_local_A;
    if (mpi_rank == 0) {
        std::vector<double> global_out_local_A;
        local_A.tensor2Array(global_out_local_A);
        dacpp::mpi::apply_writeback_by_globals(writeback_local_A, writeback_globals_local_A, global_out_local_A);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<double> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_DOUBLE, peer, 5002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_local_A);
        }
        local_A.array2Tensor(global_out_local_A);
        synced_local_A = global_out_local_A;
    } else {
        int send_count = static_cast<int>(writeback_globals_local_A.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3002, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_local_A.data()), send_count, MPI_LONG_LONG, 0, 4002, MPI_COMM_WORLD);
            MPI_Send(writeback_local_A.data(), send_count, MPI_DOUBLE, 0, 5002, MPI_COMM_WORLD);
        }
    }
    int synced_count_local_A = 0;
    if (mpi_rank == 0) {
        synced_count_local_A = static_cast<int>(synced_local_A.size());
    }
    MPI_Bcast(&synced_count_local_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_local_A.resize(synced_count_local_A);
    }
    if (synced_count_local_A > 0) {
        MPI_Bcast(synced_local_A.data(), synced_count_local_A, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        local_A.array2Tensor(synced_local_A);
    }
}

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> A(T/dt*numIsotopes, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

    while(t_tensor[0] <= T){  
        DECAY_decay(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor[1].print();
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

    

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }


    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
