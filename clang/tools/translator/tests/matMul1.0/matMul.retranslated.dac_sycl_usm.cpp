#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void matrixMultiply_calc_mpi_local(dacpp::mpi::View1D<const int> vecA, dacpp::mpi::View1D<const int> vecB, dacpp::mpi::View1D<int> dotProduct) {
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<int> & matC) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
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
    auto pack_vecA = dacpp::mpi::build_input_pack_map(item_range, pattern_vecA);
    auto slots_vecA = dacpp::mpi::build_item_slots(item_range, pattern_vecA, pack_vecA);
    std::vector<int> local_vecA(pack_vecA.globals.size());
    if (mpi_rank == 0) {
        std::vector<int> global_vecA;
        matA.tensor2Array(global_vecA);
        local_vecA = dacpp::mpi::pack_values_by_globals(global_vecA, pack_vecA.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_vecA);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_vecA, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_INT, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_vecA.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_vecA.data(), recv_count, MPI_INT, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int vecA_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_vecA));
    auto pack_vecB = dacpp::mpi::build_input_pack_map(item_range, pattern_vecB);
    auto slots_vecB = dacpp::mpi::build_item_slots(item_range, pattern_vecB, pack_vecB);
    std::vector<int> local_vecB(pack_vecB.globals.size());
    if (mpi_rank == 0) {
        std::vector<int> global_vecB;
        matB.tensor2Array(global_vecB);
        local_vecB = dacpp::mpi::pack_values_by_globals(global_vecB, pack_vecB.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_vecB);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_vecB, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1001, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_INT, peer, 2001, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_vecB.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_vecB.data(), recv_count, MPI_INT, 0, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int vecB_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_vecB));
    auto pack_dotProduct = dacpp::mpi::build_output_pack_map(item_range, pattern_dotProduct);
    auto slots_dotProduct = dacpp::mpi::build_item_slots(item_range, pattern_dotProduct, pack_dotProduct);
    std::vector<int> local_dotProduct(pack_dotProduct.globals.size());
    const int dotProduct_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_dotProduct));
    if (local_item_count > 0) {
        {
            sycl::buffer<int, 1> buffer_vecA(local_vecA.data(), sycl::range<1>(local_vecA.size()));
            sycl::buffer<int32_t, 1> slots_buffer_vecA(slots_vecA.data(), sycl::range<1>(slots_vecA.size()));
            sycl::buffer<int, 1> buffer_vecB(local_vecB.data(), sycl::range<1>(local_vecB.size()));
            sycl::buffer<int32_t, 1> slots_buffer_vecB(slots_vecB.data(), sycl::range<1>(slots_vecB.size()));
            sycl::buffer<int, 1> buffer_dotProduct(local_dotProduct.data(), sycl::range<1>(local_dotProduct.size()));
            sycl::buffer<int32_t, 1> slots_buffer_dotProduct(slots_dotProduct.data(), sycl::range<1>(slots_dotProduct.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_vecA = buffer_vecA.get_access<sycl::access::mode::read>(h);
                auto slots_acc_vecA = slots_buffer_vecA.get_access<sycl::access::mode::read>(h);
                auto acc_vecB = buffer_vecB.get_access<sycl::access::mode::read>(h);
                auto slots_acc_vecB = slots_buffer_vecB.get_access<sycl::access::mode::read>(h);
                auto acc_dotProduct = buffer_dotProduct.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_dotProduct = slots_buffer_dotProduct.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_vecA = acc_vecA.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_vecA = slots_acc_vecA.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_vecA{data_vecA, slots_vecA, item_linear * vecA_partition_size};
                    auto* data_vecB = acc_vecB.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_vecB = slots_acc_vecB.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_vecB{data_vecB, slots_vecB, item_linear * vecB_partition_size};
                    auto* data_dotProduct = acc_dotProduct.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_dotProduct = slots_acc_dotProduct.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<int> view_dotProduct{data_dotProduct, slots_dotProduct, item_linear * dotProduct_partition_size};
                    matrixMultiply_calc_mpi_local(view_vecA, view_vecB, view_dotProduct);
                });
            });
            q.wait();
        }
    }
    auto writeback_dotProduct = dacpp::mpi::build_writeback_values(local_dotProduct, pack_dotProduct);
    const auto& writeback_globals_dotProduct = pack_dotProduct.writeback_globals.empty() ? pack_dotProduct.globals : pack_dotProduct.writeback_globals;
    std::vector<int> synced_dotProduct;
    if (mpi_rank == 0) {
        std::vector<int> global_out_dotProduct;
        matC.tensor2Array(global_out_dotProduct);
        dacpp::mpi::apply_writeback_by_globals(writeback_dotProduct, writeback_globals_dotProduct, global_out_dotProduct);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<int> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_INT, peer, 5002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_dotProduct);
        }
        matC.array2Tensor(global_out_dotProduct);
        synced_dotProduct = global_out_dotProduct;
    } else {
        int send_count = static_cast<int>(writeback_globals_dotProduct.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3002, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_dotProduct.data()), send_count, MPI_LONG_LONG, 0, 4002, MPI_COMM_WORLD);
            MPI_Send(writeback_dotProduct.data(), send_count, MPI_INT, 0, 5002, MPI_COMM_WORLD);
        }
    }
    int synced_count_dotProduct = 0;
    if (mpi_rank == 0) {
        synced_count_dotProduct = static_cast<int>(synced_dotProduct.size());
    }
    MPI_Bcast(&synced_count_dotProduct, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_dotProduct.resize(synced_count_dotProduct);
    }
    if (synced_count_dotProduct > 0) {
        MPI_Bcast(synced_dotProduct.data(), synced_count_dotProduct, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        matC.array2Tensor(synced_dotProduct);
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

    // 初始化两个矩阵 A 和 B
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    dacpp::Matrix<int> matA({4, 5}, dataA);

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};
    dacpp::Matrix<int> matB({5, 4}, dataB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Matrix<int> matC({4, 4}, dataC);
    // for(int i=0;i<1;i++){
    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);
    // }
    matC.print();

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
