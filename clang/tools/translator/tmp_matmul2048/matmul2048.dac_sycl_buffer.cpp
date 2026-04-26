#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"
#include <chrono>

using namespace sycl;

inline void matrixMultiply_calc_mpi_local(dacpp::mpi::View1D<const int> vecA, dacpp::mpi::View1D<const int> vecB, dacpp::mpi::View1D<long long> dotProduct) {
    for (int i = 0; i < 2048; ++i) {
        dotProduct[0] += static_cast<long long>(vecA[i]) * vecB[i];
    }
}


void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<long long> & matC) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::resetCollectPositionsProfile();
    auto dacpp_wrapper_start = std::chrono::steady_clock::now();
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_vecA;
    pattern_vecA.param_id = 0;
    pattern_vecA.name = "vecA";
    pattern_vecA.mode = dacpp::mpi::AccessMode::Read;
    pattern_vecA.data_info.dim = matA.getDim();
    for (int dim = 0; dim < matA.getDim(); ++dim) pattern_vecA.data_info.dimLength.push_back(matA.getShape(dim));
    Dac_Op pattern_vecA_op_0;
    pattern_vecA_op_0.setDimId(0);
    pattern_vecA_op_0.size = 1;
    pattern_vecA_op_0.stride = 1;
    pattern_vecA_op_0.SetSplitSize(matA.getShape(0));
    pattern_vecA.param_ops.push_back(pattern_vecA_op_0);
    pattern_vecA.bind_set_id.push_back(0);
    pattern_vecA.bind_offset_expr.push_back("0");
    pattern_vecA.is_index_op.push_back(true);
    pattern_vecA.partition_shape = dacpp::mpi::init_partition_shape(pattern_vecA);
    pattern_vecA.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_vecA);
    if (binding_split_sizes.size() < pattern_vecA.bind_split_sizes.size()) binding_split_sizes.resize(pattern_vecA.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_vecA.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_vecA.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_vecB;
    pattern_vecB.param_id = 1;
    pattern_vecB.name = "vecB";
    pattern_vecB.mode = dacpp::mpi::AccessMode::Read;
    pattern_vecB.data_info.dim = matB.getDim();
    for (int dim = 0; dim < matB.getDim(); ++dim) pattern_vecB.data_info.dimLength.push_back(matB.getShape(dim));
    Dac_Op pattern_vecB_op_1;
    pattern_vecB_op_1.setDimId(1);
    pattern_vecB_op_1.size = 1;
    pattern_vecB_op_1.stride = 1;
    pattern_vecB_op_1.SetSplitSize(matB.getShape(1));
    pattern_vecB.param_ops.push_back(pattern_vecB_op_1);
    pattern_vecB.bind_set_id.push_back(1);
    pattern_vecB.bind_offset_expr.push_back("0");
    pattern_vecB.is_index_op.push_back(true);
    pattern_vecB.partition_shape = dacpp::mpi::init_partition_shape(pattern_vecB);
    pattern_vecB.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_vecB);
    if (binding_split_sizes.size() < pattern_vecB.bind_split_sizes.size()) binding_split_sizes.resize(pattern_vecB.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_vecB.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_vecB.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_dotProduct;
    pattern_dotProduct.param_id = 2;
    pattern_dotProduct.name = "dotProduct";
    pattern_dotProduct.mode = dacpp::mpi::AccessMode::Write;
    pattern_dotProduct.data_info.dim = matC.getDim();
    for (int dim = 0; dim < matC.getDim(); ++dim) pattern_dotProduct.data_info.dimLength.push_back(matC.getShape(dim));
    Dac_Op pattern_dotProduct_op_0;
    pattern_dotProduct_op_0.setDimId(0);
    pattern_dotProduct_op_0.size = 1;
    pattern_dotProduct_op_0.stride = 1;
    pattern_dotProduct_op_0.SetSplitSize(matC.getShape(0));
    pattern_dotProduct.param_ops.push_back(pattern_dotProduct_op_0);
    pattern_dotProduct.bind_set_id.push_back(0);
    pattern_dotProduct.bind_offset_expr.push_back("0");
    pattern_dotProduct.is_index_op.push_back(true);
    Dac_Op pattern_dotProduct_op_1;
    pattern_dotProduct_op_1.setDimId(1);
    pattern_dotProduct_op_1.size = 1;
    pattern_dotProduct_op_1.stride = 1;
    pattern_dotProduct_op_1.SetSplitSize(matC.getShape(1));
    pattern_dotProduct.param_ops.push_back(pattern_dotProduct_op_1);
    pattern_dotProduct.bind_set_id.push_back(1);
    pattern_dotProduct.bind_offset_expr.push_back("0");
    pattern_dotProduct.is_index_op.push_back(true);
    pattern_dotProduct.partition_shape = dacpp::mpi::init_partition_shape(pattern_dotProduct);
    pattern_dotProduct.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_dotProduct);
    if (binding_split_sizes.size() < pattern_dotProduct.bind_split_sizes.size()) binding_split_sizes.resize(pattern_dotProduct.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_dotProduct.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_dotProduct.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_vecA.bind_split_sizes = binding_split_sizes;
    pattern_vecB.bind_split_sizes = binding_split_sizes;
    pattern_dotProduct.bind_split_sizes = binding_split_sizes;
    auto plan_vecA = dacpp::mpi::build_input_pack_plan(item_range, pattern_vecA);
    auto& pack_vecA = plan_vecA.pack;
    auto& slots_vecA = plan_vecA.compact_slots;
    auto& key_offsets_vecA = plan_vecA.item_key_offsets;
    std::vector<int> local_vecA(pack_vecA.globals.size());
    std::vector<int> sendcounts_vecA;
    std::vector<int> displs_vecA;
    int local_global_count_vecA = static_cast<int>(pack_vecA.globals.size());
    std::vector<int> global_counts_vecA;
    std::vector<int> global_displs_vecA;
    std::vector<int64_t> gathered_globals_vecA;
    if (mpi_rank == 0) {
        global_counts_vecA.resize(mpi_size);
        global_displs_vecA.resize(mpi_size);
    }
    MPI_Gather(&local_global_count_vecA, 1, MPI_INT, mpi_rank == 0 ? global_counts_vecA.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_global_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            global_displs_vecA[r] = current_global_displ;
            current_global_displ += global_counts_vecA[r];
        }
        gathered_globals_vecA.resize(current_global_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(pack_vecA.globals.data()), local_global_count_vecA, MPI_LONG_LONG, mpi_rank == 0 ? gathered_globals_vecA.data() : nullptr, mpi_rank == 0 ? global_counts_vecA.data() : nullptr, mpi_rank == 0 ? global_displs_vecA.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> sendbuf_vecA;
    if (mpi_rank == 0) {
        sendcounts_vecA.resize(mpi_size);
        displs_vecA.resize(mpi_size);
        int current_displ = 0;
        std::vector<int> global_vecA;
        matA.tensor2Array(global_vecA);
        for (int r = 0; r < mpi_size; ++r) {
            sendcounts_vecA[r] = global_counts_vecA[r];
            displs_vecA[r] = current_displ;
            current_displ += sendcounts_vecA[r];
        }
        sendbuf_vecA = dacpp::mpi::pack_values_by_globals_parallel_range(global_vecA, gathered_globals_vecA.data(), gathered_globals_vecA.size());
    }
    local_vecA.resize(static_cast<std::size_t>(local_global_count_vecA));
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_vecA.data() : nullptr, mpi_rank == 0 ? sendcounts_vecA.data() : nullptr, mpi_rank == 0 ? displs_vecA.data() : nullptr, MPI_INT, local_vecA.data(), local_global_count_vecA, MPI_INT, 0, MPI_COMM_WORLD);
    const int vecA_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_vecA));
    auto plan_vecB = dacpp::mpi::build_input_pack_plan(item_range, pattern_vecB);
    auto& pack_vecB = plan_vecB.pack;
    auto& slots_vecB = plan_vecB.compact_slots;
    auto& key_offsets_vecB = plan_vecB.item_key_offsets;
    std::vector<int> local_vecB(pack_vecB.globals.size());
    std::vector<int> sendcounts_vecB;
    std::vector<int> displs_vecB;
    int local_global_count_vecB = static_cast<int>(pack_vecB.globals.size());
    std::vector<int> global_counts_vecB;
    std::vector<int> global_displs_vecB;
    std::vector<int64_t> gathered_globals_vecB;
    if (mpi_rank == 0) {
        global_counts_vecB.resize(mpi_size);
        global_displs_vecB.resize(mpi_size);
    }
    MPI_Gather(&local_global_count_vecB, 1, MPI_INT, mpi_rank == 0 ? global_counts_vecB.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_global_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            global_displs_vecB[r] = current_global_displ;
            current_global_displ += global_counts_vecB[r];
        }
        gathered_globals_vecB.resize(current_global_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(pack_vecB.globals.data()), local_global_count_vecB, MPI_LONG_LONG, mpi_rank == 0 ? gathered_globals_vecB.data() : nullptr, mpi_rank == 0 ? global_counts_vecB.data() : nullptr, mpi_rank == 0 ? global_displs_vecB.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> sendbuf_vecB;
    if (mpi_rank == 0) {
        sendcounts_vecB.resize(mpi_size);
        displs_vecB.resize(mpi_size);
        int current_displ = 0;
        std::vector<int> global_vecB;
        matB.tensor2Array(global_vecB);
        for (int r = 0; r < mpi_size; ++r) {
            sendcounts_vecB[r] = global_counts_vecB[r];
            displs_vecB[r] = current_displ;
            current_displ += sendcounts_vecB[r];
        }
        sendbuf_vecB = dacpp::mpi::pack_values_by_globals_parallel_range(global_vecB, gathered_globals_vecB.data(), gathered_globals_vecB.size());
    }
    local_vecB.resize(static_cast<std::size_t>(local_global_count_vecB));
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_vecB.data() : nullptr, mpi_rank == 0 ? sendcounts_vecB.data() : nullptr, mpi_rank == 0 ? displs_vecB.data() : nullptr, MPI_INT, local_vecB.data(), local_global_count_vecB, MPI_INT, 0, MPI_COMM_WORLD);
    const int vecB_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_vecB));
    auto plan_dotProduct = dacpp::mpi::build_output_pack_plan(item_range, pattern_dotProduct);
    auto& pack_dotProduct = plan_dotProduct.pack;
    auto& slots_dotProduct = plan_dotProduct.compact_slots;
    auto& key_offsets_dotProduct = plan_dotProduct.item_key_offsets;
    std::vector<long long> local_dotProduct(pack_dotProduct.globals.size());
    const int dotProduct_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_dotProduct));
    double dacpp_kernel_local_ms = 0.0;
    if (local_item_count > 0) {
        auto dacpp_kernel_start = std::chrono::steady_clock::now();
        {
            sycl::buffer<int, 1> buffer_vecA(local_vecA.data(), sycl::range<1>(local_vecA.size()));
            sycl::buffer<int32_t, 1> slots_buffer_vecA(slots_vecA.data(), sycl::range<1>(slots_vecA.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_vecA(key_offsets_vecA.data(), sycl::range<1>(key_offsets_vecA.size()));
            sycl::buffer<int, 1> buffer_vecB(local_vecB.data(), sycl::range<1>(local_vecB.size()));
            sycl::buffer<int32_t, 1> slots_buffer_vecB(slots_vecB.data(), sycl::range<1>(slots_vecB.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_vecB(key_offsets_vecB.data(), sycl::range<1>(key_offsets_vecB.size()));
            sycl::buffer<long long, 1> buffer_dotProduct(local_dotProduct.data(), sycl::range<1>(local_dotProduct.size()));
            sycl::buffer<int32_t, 1> slots_buffer_dotProduct(slots_dotProduct.data(), sycl::range<1>(slots_dotProduct.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_dotProduct(key_offsets_dotProduct.data(), sycl::range<1>(key_offsets_dotProduct.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_vecA = buffer_vecA.get_access<sycl::access::mode::read>(h);
                auto slots_acc_vecA = slots_buffer_vecA.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_vecA = key_offsets_buffer_vecA.get_access<sycl::access::mode::read>(h);
                auto acc_vecB = buffer_vecB.get_access<sycl::access::mode::read>(h);
                auto slots_acc_vecB = slots_buffer_vecB.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_vecB = key_offsets_buffer_vecB.get_access<sycl::access::mode::read>(h);
                auto acc_dotProduct = buffer_dotProduct.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_dotProduct = slots_buffer_dotProduct.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_dotProduct = key_offsets_buffer_dotProduct.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_vecA = acc_vecA.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_vecA = slots_acc_vecA.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_vecA = key_offsets_acc_vecA.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_vecA{data_vecA, slots_vecA, key_offsets_vecA[item_linear]};
                    auto* data_vecB = acc_vecB.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_vecB = slots_acc_vecB.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_vecB = key_offsets_acc_vecB.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_vecB{data_vecB, slots_vecB, key_offsets_vecB[item_linear]};
                    auto* data_dotProduct = acc_dotProduct.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_dotProduct = slots_acc_dotProduct.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_dotProduct = key_offsets_acc_dotProduct.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<long long> view_dotProduct{data_dotProduct, slots_dotProduct, key_offsets_dotProduct[item_linear]};
                    matrixMultiply_calc_mpi_local(view_vecA, view_vecB, view_dotProduct);
                });
            });
            q.wait();
        }
        auto dacpp_kernel_end = std::chrono::steady_clock::now();
        dacpp_kernel_local_ms = std::chrono::duration<double, std::milli>(dacpp_kernel_end - dacpp_kernel_start).count();
    }
    double dacpp_kernel_max_ms = 0.0;
    MPI_Reduce(&dacpp_kernel_local_ms, &dacpp_kernel_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto& writeback_globals_dotProduct = pack_dotProduct.writeback_globals.empty() ? pack_dotProduct.globals : pack_dotProduct.writeback_globals;
    int send_count_dotProduct = static_cast<int>(writeback_globals_dotProduct.size());
    std::vector<int> recvcounts_dotProduct;
    std::vector<int> recvdispls_dotProduct;
    std::vector<int64_t> global_recv_globals_dotProduct;
    std::vector<long long> global_recv_values_dotProduct;
    if (mpi_rank == 0) {
        recvcounts_dotProduct.resize(mpi_size);
        recvdispls_dotProduct.resize(mpi_size);
    }
    MPI_Gather(&send_count_dotProduct, 1, MPI_INT, mpi_rank == 0 ? recvcounts_dotProduct.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_dotProduct[r] = current_displ;
            current_displ += recvcounts_dotProduct[r];
        }
        global_recv_globals_dotProduct.resize(current_displ);
        global_recv_values_dotProduct.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_dotProduct.data()), send_count_dotProduct, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_dotProduct.data() : nullptr, mpi_rank == 0 ? recvcounts_dotProduct.data() : nullptr, mpi_rank == 0 ? recvdispls_dotProduct.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_dotProduct.data(), send_count_dotProduct, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_values_dotProduct.data() : nullptr, mpi_rank == 0 ? recvcounts_dotProduct.data() : nullptr, mpi_rank == 0 ? recvdispls_dotProduct.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<long long> global_out_dotProduct;
        matC.tensor2Array(global_out_dotProduct);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_dotProduct, global_recv_globals_dotProduct, global_out_dotProduct);
        matC.array2Tensor(global_out_dotProduct);
        if (!global_out_dotProduct.empty()) {
            MPI_Bcast(global_out_dotProduct.data(), matC.getSize(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        }
    } else {
        std::vector<long long> global_out_dotProduct(static_cast<std::size_t>(matC.getSize()));
        if (!global_out_dotProduct.empty()) {
            MPI_Bcast(global_out_dotProduct.data(), matC.getSize(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        }
        matC.array2Tensor(global_out_dotProduct);
    }
    auto dacpp_wrapper_end = std::chrono::steady_clock::now();
    double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();
    double dacpp_wrapper_max_ms = 0.0;
    MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0 && dacpp::mpi::profilingEnabled()) {
        std::fprintf(stderr, "[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\n", "matrixMultiply_shell_matrixMultiply_calc", dacpp_wrapper_max_ms);
        std::fprintf(stderr, "[DACPP][PROFILE][%s] kernel_region_ms(max): %.3f\n", "matrixMultiply_shell_matrixMultiply_calc", dacpp_kernel_max_ms);
        std::fprintf(stderr, "[DACPP][PROFILE][%s] non_kernel_ms(derived): %.3f\n", "matrixMultiply_shell_matrixMultiply_calc", dacpp_wrapper_max_ms - dacpp_kernel_max_ms);
    }
    dacpp::mpi::reportCollectPositionsProfile("matrixMultiply_shell_matrixMultiply_calc", MPI_COMM_WORLD);
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

    constexpr int N = 2048;
    std::vector<int> dataA(static_cast<std::size_t>(N) * N, 1);
    std::vector<int> dataB(static_cast<std::size_t>(N) * N, 1);
    std::vector<long long> dataC(static_cast<std::size_t>(N) * N, 0);

    dacpp::Matrix<int> matA({N, N}, dataA);
    dacpp::Matrix<int> matB({N, N}, dataB);
    dacpp::Matrix<long long> matC({N, N}, dataC);

    auto start = std::chrono::steady_clock::now();
    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);
    auto end = std::chrono::steady_clock::now();

    std::vector<long long> result;
    matC.tensor2Array(result);
    long long checksum = 0;
    for (long long value : result) {
        checksum += value;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "checksum=" << checksum << "\n";
    std::cout << "kernel_call_ms=" << elapsed_ms << "\n";
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
