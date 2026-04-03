#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1

namespace dacpp {
    typedef std::vector<std::any> list;
}
const int N = 8;  // 假设数组的大小为1024



// 交换函数
void swap(vector<int>& array, int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}





// 奇偶归并排序的核心操作
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void oddeven_mpi_local(dacpp::mpi::View1D<const int> array, dacpp::mpi::View1D<int> array_out) {
    if (array[0] > array[1]) {
        array_out[0] = array[1];
        array_out[1] = array[0];
    } else {
        array_out[0] = array[0];
        array_out[1] = array[1];
    }
}


void ODDEVEN_oddeven(const dacpp::Vector<int> & array, dacpp::Vector<int> & array_out) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_array;
    pattern_array.param_id = 0;
    pattern_array.name = "array";
    pattern_array.mode = dacpp::mpi::AccessMode::Read;
    pattern_array.data_info.dim = array.getDim();
    for (int dim = 0; dim < array.getDim(); ++dim) pattern_array.data_info.dimLength.push_back(array.getShape(dim));
    Dac_Op pattern_array_op_0;
    pattern_array_op_0.setDimId(0);
    pattern_array_op_0.size = 2;
    pattern_array_op_0.stride = 2;
    pattern_array_op_0.SetSplitSize((array.getShape(0) - 2) / 2 + 1);
    pattern_array.param_ops.push_back(pattern_array_op_0);
    pattern_array.bind_set_id.push_back(0);
    pattern_array.bind_offset_expr.push_back("0");
    pattern_array.is_index_op.push_back(false);
    pattern_array.partition_shape = dacpp::mpi::init_partition_shape(pattern_array);
    pattern_array.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_array);
    if (binding_split_sizes.size() < pattern_array.bind_split_sizes.size()) binding_split_sizes.resize(pattern_array.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_array.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_array.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_array_out;
    pattern_array_out.param_id = 1;
    pattern_array_out.name = "array_out";
    pattern_array_out.mode = dacpp::mpi::AccessMode::Write;
    pattern_array_out.data_info.dim = array_out.getDim();
    for (int dim = 0; dim < array_out.getDim(); ++dim) pattern_array_out.data_info.dimLength.push_back(array_out.getShape(dim));
    Dac_Op pattern_array_out_op_0;
    pattern_array_out_op_0.setDimId(0);
    pattern_array_out_op_0.size = 2;
    pattern_array_out_op_0.stride = 2;
    pattern_array_out_op_0.SetSplitSize((array_out.getShape(0) - 2) / 2 + 1);
    pattern_array_out.param_ops.push_back(pattern_array_out_op_0);
    pattern_array_out.bind_set_id.push_back(0);
    pattern_array_out.bind_offset_expr.push_back("0");
    pattern_array_out.is_index_op.push_back(false);
    pattern_array_out.partition_shape = dacpp::mpi::init_partition_shape(pattern_array_out);
    pattern_array_out.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_array_out);
    if (binding_split_sizes.size() < pattern_array_out.bind_split_sizes.size()) binding_split_sizes.resize(pattern_array_out.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_array_out.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_array_out.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_array.bind_split_sizes = binding_split_sizes;
    pattern_array_out.bind_split_sizes = binding_split_sizes;
    auto pack_array = dacpp::mpi::build_input_pack_map(item_range, pattern_array);
    auto slots_array = dacpp::mpi::build_item_slots(item_range, pattern_array, pack_array);
    std::vector<int> local_array(pack_array.globals.size());
    if (mpi_rank == 0) {
        std::vector<int> global_array;
        array.tensor2Array(global_array);
        local_array = dacpp::mpi::pack_values_by_globals(global_array, pack_array.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_array);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_array, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_INT, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_array.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_array.data(), recv_count, MPI_INT, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int array_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_array));
    auto pack_array_out = dacpp::mpi::build_output_pack_map(item_range, pattern_array_out);
    auto slots_array_out = dacpp::mpi::build_item_slots(item_range, pattern_array_out, pack_array_out);
    std::vector<int> local_array_out(pack_array_out.globals.size());
    const int array_out_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_array_out));
    if (local_item_count > 0) {
        {
            sycl::buffer<int, 1> buffer_array(local_array.data(), sycl::range<1>(local_array.size()));
            sycl::buffer<int32_t, 1> slots_buffer_array(slots_array.data(), sycl::range<1>(slots_array.size()));
            sycl::buffer<int, 1> buffer_array_out(local_array_out.data(), sycl::range<1>(local_array_out.size()));
            sycl::buffer<int32_t, 1> slots_buffer_array_out(slots_array_out.data(), sycl::range<1>(slots_array_out.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_array = buffer_array.get_access<sycl::access::mode::read>(h);
                auto slots_acc_array = slots_buffer_array.get_access<sycl::access::mode::read>(h);
                auto acc_array_out = buffer_array_out.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_array_out = slots_buffer_array_out.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_array = acc_array.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_array = slots_acc_array.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_array{data_array, slots_array, item_linear * array_partition_size};
                    auto* data_array_out = acc_array_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_array_out = slots_acc_array_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<int> view_array_out{data_array_out, slots_array_out, item_linear * array_out_partition_size};
                    oddeven_mpi_local(view_array, view_array_out);
                });
            });
            q.wait();
        }
    }
    auto writeback_array_out = dacpp::mpi::build_writeback_values(local_array_out, pack_array_out);
    const auto& writeback_globals_array_out = pack_array_out.writeback_globals.empty() ? pack_array_out.globals : pack_array_out.writeback_globals;
    std::vector<int> synced_array_out;
    if (mpi_rank == 0) {
        std::vector<int> global_out_array_out;
        array_out.tensor2Array(global_out_array_out);
        dacpp::mpi::apply_writeback_by_globals(writeback_array_out, writeback_globals_array_out, global_out_array_out);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<int> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_INT, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_array_out);
        }
        array_out.array2Tensor(global_out_array_out);
        synced_array_out = global_out_array_out;
    } else {
        int send_count = static_cast<int>(writeback_globals_array_out.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_array_out.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_array_out.data(), send_count, MPI_INT, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_array_out = 0;
    if (mpi_rank == 0) {
        synced_count_array_out = static_cast<int>(synced_array_out.size());
    }
    MPI_Bcast(&synced_count_array_out, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_array_out.resize(synced_count_array_out);
    }
    if (synced_count_array_out > 0) {
        MPI_Bcast(synced_array_out.data(), synced_count_array_out, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        array_out.array2Tensor(synced_array_out);
    }
}

void oddEvenMergeSort(vector<int>& array, int n) {
    dacpp::Tensor<int, 1> array_tensor(array);
    vector<int> array_out(N);
    dacpp::Tensor<int, 1> array_out_tensor(array_out);

    // 每一轮排序进行多次比较
    for (int phase = 0; phase < N; phase++) {
        // 奇数阶段：比较相邻的奇数索引
        //array_tensor.print();
        ODDEVEN_oddeven(array_tensor, array_out_tensor);
        //array_out_tensor.print();
        // vector<int> array2(N-2, 0);
        //dacpp::Tensor<int, 1> array2_tensor(array2);

        // for(int i = 0;i < N-2; i++){
        //     array2_tensor[i] = array_out_tensor[i+1];
        // }
        dacpp::Tensor<int, 1> array2_tensor = array_out_tensor[{1,N-1}];
        vector<int> array_out2(N-2, 0);
        dacpp::Tensor<int, 1> array_out2_tensor(array_out2);
        //array2_tensor.print();
        ODDEVEN_oddeven(array2_tensor, array_out2_tensor);

        for(int i = 1;i < N-1; i++){
            array_tensor[i] = array_out2_tensor[i-1];
        }
        array_tensor[0] = array_out_tensor[0];
        array_tensor[N-1] = array_out_tensor[N-1];
    }
    array_tensor.print();
}

// 主函数：初始化数据并调用奇偶归并排序
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

    vector<int> array(N);

    // 初始化数据
    // srand(time(0));
    // for (int i = 0; i < N; i++) {
    //     array[i] = rand() % 10;  // 随机生成0到1000之间的整数
    // }
    for (int i = 0; i < N; i++) {
        array[i] = N - i;  // 初始化为递减的数组
    }

    // 打印排序前的数组（前10个）
    //std::cout << "Array before sorting (first 10 elements):" << std::endl;
    for (int i = 0; i < N; i++) {
        //std::cout << array[i] << " ";
    }
    //std::cout << std::endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);


    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
