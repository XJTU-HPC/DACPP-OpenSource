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
#include <chrono>

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
    dacpp::mpi::resetCollectPositionsProfile();
    auto dacpp_wrapper_start = std::chrono::steady_clock::now();
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
    auto plan_array = dacpp::mpi::build_input_pack_plan(item_range, pattern_array);
    auto& pack_array = plan_array.pack;
    auto& slots_array = plan_array.compact_slots;
    auto& key_offsets_array = plan_array.item_key_offsets;
    std::vector<int> local_array(pack_array.globals.size());
    std::vector<int> sendcounts_array;
    std::vector<int> displs_array;
    int local_global_count_array = static_cast<int>(pack_array.globals.size());
    std::vector<int> global_counts_array;
    std::vector<int> global_displs_array;
    std::vector<int64_t> gathered_globals_array;
    if (mpi_rank == 0) {
        global_counts_array.resize(mpi_size);
        global_displs_array.resize(mpi_size);
    }
    MPI_Gather(&local_global_count_array, 1, MPI_INT, mpi_rank == 0 ? global_counts_array.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_global_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            global_displs_array[r] = current_global_displ;
            current_global_displ += global_counts_array[r];
        }
        gathered_globals_array.resize(current_global_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(pack_array.globals.data()), local_global_count_array, MPI_LONG_LONG, mpi_rank == 0 ? gathered_globals_array.data() : nullptr, mpi_rank == 0 ? global_counts_array.data() : nullptr, mpi_rank == 0 ? global_displs_array.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> sendbuf_array;
    if (mpi_rank == 0) {
        sendcounts_array.resize(mpi_size);
        displs_array.resize(mpi_size);
        int current_displ = 0;
        std::vector<int> global_array;
        array.tensor2Array(global_array);
        for (int r = 0; r < mpi_size; ++r) {
            sendcounts_array[r] = global_counts_array[r];
            displs_array[r] = current_displ;
            current_displ += sendcounts_array[r];
        }
        sendbuf_array = dacpp::mpi::pack_values_by_globals_parallel_range(global_array, gathered_globals_array.data(), gathered_globals_array.size());
    }
    local_array.resize(static_cast<std::size_t>(local_global_count_array));
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_array.data() : nullptr, mpi_rank == 0 ? sendcounts_array.data() : nullptr, mpi_rank == 0 ? displs_array.data() : nullptr, MPI_INT, local_array.data(), local_global_count_array, MPI_INT, 0, MPI_COMM_WORLD);
    const int array_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_array));
    auto plan_array_out = dacpp::mpi::build_output_pack_plan(item_range, pattern_array_out);
    auto& pack_array_out = plan_array_out.pack;
    auto& slots_array_out = plan_array_out.compact_slots;
    auto& key_offsets_array_out = plan_array_out.item_key_offsets;
    std::vector<int> local_array_out(pack_array_out.globals.size());
    const int array_out_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_array_out));
    if (local_item_count > 0) {
        {
            sycl::buffer<int, 1> buffer_array(local_array.data(), sycl::range<1>(local_array.size()));
            sycl::buffer<int32_t, 1> slots_buffer_array(slots_array.data(), sycl::range<1>(slots_array.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_array(key_offsets_array.data(), sycl::range<1>(key_offsets_array.size()));
            sycl::buffer<int, 1> buffer_array_out(local_array_out.data(), sycl::range<1>(local_array_out.size()));
            sycl::buffer<int32_t, 1> slots_buffer_array_out(slots_array_out.data(), sycl::range<1>(slots_array_out.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_array_out(key_offsets_array_out.data(), sycl::range<1>(key_offsets_array_out.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_array = buffer_array.get_access<sycl::access::mode::read>(h);
                auto slots_acc_array = slots_buffer_array.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_array = key_offsets_buffer_array.get_access<sycl::access::mode::read>(h);
                auto acc_array_out = buffer_array_out.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_array_out = slots_buffer_array_out.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_array_out = key_offsets_buffer_array_out.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_array = acc_array.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_array = slots_acc_array.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_array = key_offsets_acc_array.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_array{data_array, slots_array, key_offsets_array[item_linear]};
                    auto* data_array_out = acc_array_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_array_out = slots_acc_array_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_array_out = key_offsets_acc_array_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<int> view_array_out{data_array_out, slots_array_out, key_offsets_array_out[item_linear]};
                    oddeven_mpi_local(view_array, view_array_out);
                });
            });
            q.wait();
        }
    }
    const auto& writeback_globals_array_out = pack_array_out.writeback_globals.empty() ? pack_array_out.globals : pack_array_out.writeback_globals;
    int send_count_array_out = static_cast<int>(writeback_globals_array_out.size());
    std::vector<int> recvcounts_array_out;
    std::vector<int> recvdispls_array_out;
    std::vector<int64_t> global_recv_globals_array_out;
    std::vector<int> global_recv_values_array_out;
    if (mpi_rank == 0) {
        recvcounts_array_out.resize(mpi_size);
        recvdispls_array_out.resize(mpi_size);
    }
    MPI_Gather(&send_count_array_out, 1, MPI_INT, mpi_rank == 0 ? recvcounts_array_out.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_array_out[r] = current_displ;
            current_displ += recvcounts_array_out[r];
        }
        global_recv_globals_array_out.resize(current_displ);
        global_recv_values_array_out.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_array_out.data()), send_count_array_out, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_array_out.data() : nullptr, mpi_rank == 0 ? recvcounts_array_out.data() : nullptr, mpi_rank == 0 ? recvdispls_array_out.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_array_out.data(), send_count_array_out, MPI_INT, mpi_rank == 0 ? global_recv_values_array_out.data() : nullptr, mpi_rank == 0 ? recvcounts_array_out.data() : nullptr, mpi_rank == 0 ? recvdispls_array_out.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<int> global_out_array_out;
        array_out.tensor2Array(global_out_array_out);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_array_out, global_recv_globals_array_out, global_out_array_out);
        array_out.array2Tensor(global_out_array_out);
    } else {
    }
    auto dacpp_wrapper_end = std::chrono::steady_clock::now();
    double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();
    double dacpp_wrapper_max_ms = 0.0;
    MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0 && dacpp::mpi::profilingEnabled()) {
        std::fprintf(stderr, "[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\n", "ODDEVEN_oddeven", dacpp_wrapper_max_ms);
    }
    dacpp::mpi::reportCollectPositionsProfile("ODDEVEN_oddeven", MPI_COMM_WORLD);
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
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
