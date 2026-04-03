#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NUM_NEURONS = 8;   // 神经元数量（层宽度）
const int INPUT_SIZE  = 8;   // 每个神经元输入数
 


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

using namespace sycl;

inline void gradSum_mpi_local(dacpp::mpi::View1D<const float> grads, dacpp::mpi::View1D<float> neuronSum) {
    int sum = 0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += grads[j];
    }
    neuronSum[0] = sum;
}


void gradSumShell_gradSum(dacpp::Matrix<float> & matGrads, dacpp::Matrix<float> & matNeuronSum) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_grads;
    pattern_grads.param_id = 0;
    pattern_grads.name = "grads";
    pattern_grads.mode = dacpp::mpi::AccessMode::Read;
    pattern_grads.data_info.dim = matGrads.getDim();
    for (int dim = 0; dim < matGrads.getDim(); ++dim) pattern_grads.data_info.dimLength.push_back(matGrads.getShape(dim));
    Dac_Op pattern_grads_op_1;
    pattern_grads_op_1.setDimId(1);
    pattern_grads_op_1.size = 1;
    pattern_grads_op_1.stride = 1;
    pattern_grads_op_1.SetSplitSize(matGrads.getShape(1));
    pattern_grads.param_ops.push_back(pattern_grads_op_1);
    pattern_grads.bind_set_id.push_back(0);
    pattern_grads.bind_offset_expr.push_back("0");
    pattern_grads.is_index_op.push_back(true);
    pattern_grads.partition_shape = dacpp::mpi::init_partition_shape(pattern_grads);
    pattern_grads.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_grads);
    if (binding_split_sizes.size() < pattern_grads.bind_split_sizes.size()) binding_split_sizes.resize(pattern_grads.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_grads.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_grads.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_neuronSum;
    pattern_neuronSum.param_id = 1;
    pattern_neuronSum.name = "neuronSum";
    pattern_neuronSum.mode = dacpp::mpi::AccessMode::Write;
    pattern_neuronSum.data_info.dim = matNeuronSum.getDim();
    for (int dim = 0; dim < matNeuronSum.getDim(); ++dim) pattern_neuronSum.data_info.dimLength.push_back(matNeuronSum.getShape(dim));
    Dac_Op pattern_neuronSum_op_0;
    pattern_neuronSum_op_0.setDimId(0);
    pattern_neuronSum_op_0.size = 1;
    pattern_neuronSum_op_0.stride = 1;
    pattern_neuronSum_op_0.SetSplitSize(matNeuronSum.getShape(0));
    pattern_neuronSum.param_ops.push_back(pattern_neuronSum_op_0);
    pattern_neuronSum.bind_set_id.push_back(0);
    pattern_neuronSum.bind_offset_expr.push_back("0");
    pattern_neuronSum.is_index_op.push_back(true);
    Dac_Op pattern_neuronSum_op_1;
    pattern_neuronSum_op_1.setDimId(1);
    pattern_neuronSum_op_1.size = 1;
    pattern_neuronSum_op_1.stride = 1;
    pattern_neuronSum_op_1.SetSplitSize(matNeuronSum.getShape(1));
    pattern_neuronSum.param_ops.push_back(pattern_neuronSum_op_1);
    pattern_neuronSum.bind_set_id.push_back(1);
    pattern_neuronSum.bind_offset_expr.push_back("0");
    pattern_neuronSum.is_index_op.push_back(true);
    pattern_neuronSum.partition_shape = dacpp::mpi::init_partition_shape(pattern_neuronSum);
    pattern_neuronSum.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_neuronSum);
    if (binding_split_sizes.size() < pattern_neuronSum.bind_split_sizes.size()) binding_split_sizes.resize(pattern_neuronSum.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_neuronSum.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_neuronSum.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_grads.bind_split_sizes = binding_split_sizes;
    pattern_neuronSum.bind_split_sizes = binding_split_sizes;
    auto pack_grads = dacpp::mpi::build_input_pack_map(item_range, pattern_grads);
    auto slots_grads = dacpp::mpi::build_item_slots(item_range, pattern_grads, pack_grads);
    std::vector<float> local_grads(pack_grads.globals.size());
    if (mpi_rank == 0) {
        std::vector<float> global_grads;
        matGrads.tensor2Array(global_grads);
        local_grads = dacpp::mpi::pack_values_by_globals(global_grads, pack_grads.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_grads);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_grads, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_FLOAT, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_grads.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_grads.data(), recv_count, MPI_FLOAT, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int grads_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_grads));
    auto pack_neuronSum = dacpp::mpi::build_output_pack_map(item_range, pattern_neuronSum);
    auto slots_neuronSum = dacpp::mpi::build_item_slots(item_range, pattern_neuronSum, pack_neuronSum);
    std::vector<float> local_neuronSum(pack_neuronSum.globals.size());
    const int neuronSum_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_neuronSum));
    if (local_item_count > 0) {
        {
            sycl::buffer<float, 1> buffer_grads(local_grads.data(), sycl::range<1>(local_grads.size()));
            sycl::buffer<int32_t, 1> slots_buffer_grads(slots_grads.data(), sycl::range<1>(slots_grads.size()));
            sycl::buffer<float, 1> buffer_neuronSum(local_neuronSum.data(), sycl::range<1>(local_neuronSum.size()));
            sycl::buffer<int32_t, 1> slots_buffer_neuronSum(slots_neuronSum.data(), sycl::range<1>(slots_neuronSum.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_grads = buffer_grads.get_access<sycl::access::mode::read>(h);
                auto slots_acc_grads = slots_buffer_grads.get_access<sycl::access::mode::read>(h);
                auto acc_neuronSum = buffer_neuronSum.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_neuronSum = slots_buffer_neuronSum.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_grads = acc_grads.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_grads = slots_acc_grads.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_grads{data_grads, slots_grads, item_linear * grads_partition_size};
                    auto* data_neuronSum = acc_neuronSum.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_neuronSum = slots_acc_neuronSum.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<float> view_neuronSum{data_neuronSum, slots_neuronSum, item_linear * neuronSum_partition_size};
                    gradSum_mpi_local(view_grads, view_neuronSum);
                });
            });
            q.wait();
        }
    }
    auto writeback_neuronSum = dacpp::mpi::build_writeback_values(local_neuronSum, pack_neuronSum);
    const auto& writeback_globals_neuronSum = pack_neuronSum.writeback_globals.empty() ? pack_neuronSum.globals : pack_neuronSum.writeback_globals;
    std::vector<float> synced_neuronSum;
    if (mpi_rank == 0) {
        std::vector<float> global_out_neuronSum;
        matNeuronSum.tensor2Array(global_out_neuronSum);
        dacpp::mpi::apply_writeback_by_globals(writeback_neuronSum, writeback_globals_neuronSum, global_out_neuronSum);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<float> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_FLOAT, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_neuronSum);
        }
        matNeuronSum.array2Tensor(global_out_neuronSum);
        synced_neuronSum = global_out_neuronSum;
    } else {
        int send_count = static_cast<int>(writeback_globals_neuronSum.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_neuronSum.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_neuronSum.data(), send_count, MPI_FLOAT, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_neuronSum = 0;
    if (mpi_rank == 0) {
        synced_count_neuronSum = static_cast<int>(synced_neuronSum.size());
    }
    MPI_Bcast(&synced_count_neuronSum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_neuronSum.resize(synced_count_neuronSum);
    }
    if (synced_count_neuronSum > 0) {
        MPI_Bcast(synced_neuronSum.data(), synced_count_neuronSum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        matNeuronSum.array2Tensor(synced_neuronSum);
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

    // 初始化梯度矩阵（模拟随机梯度）
    vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    // for (auto &v : host_grads) v = dist(gen);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j]=j;
        }
    }
    // DAC Tensor 初始化
    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    // 执行 DAC shell -> calc
    gradSumShell_gradSum(matGrads, matNeuronSum);

    // 输出结果
    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min(5, NUM_NEURONS) ; ++i)
        std::cout << matNeuronSum[i][0] << " ";
    std::cout << std::endl;

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
