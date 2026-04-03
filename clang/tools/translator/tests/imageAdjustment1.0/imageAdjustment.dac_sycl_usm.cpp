#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// 定义图像类型，假设每个像素由 RGB 三个值组成
struct Pixel {
    int r, g, b;
    // 重载 << 操作符，使 Pixel 类型的对象可以被输出
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};




// 色彩调整操作：增加红色分量


// 亮度增强操作：增加每个像素的 RGB 分量




// 打印图像的前几个像素，作为调试
void print_image(const std::vector<std::vector<Pixel>>& image, int num_rows = 5, int num_cols = 5) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << "(" << image[i][j].r << "," << image[i][j].g << "," << image[i][j].b << ") ";
        }
        std::cout << std::endl;
    }
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void image_1_mpi_local(dacpp::mpi::View1D<const Pixel> image_tensor, dacpp::mpi::View1D<Pixel> image_tensor2) {
    image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);
}


void imageAdjustment_image_1(const dacpp::Matrix<Pixel> & image_tensor, dacpp::Matrix<Pixel> & image_tensor2) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_image_tensor;
    pattern_image_tensor.param_id = 0;
    pattern_image_tensor.name = "image_tensor";
    pattern_image_tensor.mode = dacpp::mpi::AccessMode::Read;
    pattern_image_tensor.data_info.dim = image_tensor.getDim();
    for (int dim = 0; dim < image_tensor.getDim(); ++dim) pattern_image_tensor.data_info.dimLength.push_back(image_tensor.getShape(dim));
    Dac_Op pattern_image_tensor_op_0;
    pattern_image_tensor_op_0.setDimId(0);
    pattern_image_tensor_op_0.size = 1;
    pattern_image_tensor_op_0.stride = 1;
    pattern_image_tensor_op_0.SetSplitSize(image_tensor.getShape(0));
    pattern_image_tensor.param_ops.push_back(pattern_image_tensor_op_0);
    pattern_image_tensor.bind_set_id.push_back(0);
    pattern_image_tensor.bind_offset_expr.push_back("0");
    pattern_image_tensor.is_index_op.push_back(true);
    Dac_Op pattern_image_tensor_op_1;
    pattern_image_tensor_op_1.setDimId(1);
    pattern_image_tensor_op_1.size = 1;
    pattern_image_tensor_op_1.stride = 1;
    pattern_image_tensor_op_1.SetSplitSize(image_tensor.getShape(1));
    pattern_image_tensor.param_ops.push_back(pattern_image_tensor_op_1);
    pattern_image_tensor.bind_set_id.push_back(1);
    pattern_image_tensor.bind_offset_expr.push_back("0");
    pattern_image_tensor.is_index_op.push_back(true);
    pattern_image_tensor.partition_shape = dacpp::mpi::init_partition_shape(pattern_image_tensor);
    pattern_image_tensor.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_image_tensor);
    if (binding_split_sizes.size() < pattern_image_tensor.bind_split_sizes.size()) binding_split_sizes.resize(pattern_image_tensor.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_image_tensor.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_image_tensor.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_image_tensor2;
    pattern_image_tensor2.param_id = 1;
    pattern_image_tensor2.name = "image_tensor2";
    pattern_image_tensor2.mode = dacpp::mpi::AccessMode::Write;
    pattern_image_tensor2.data_info.dim = image_tensor2.getDim();
    for (int dim = 0; dim < image_tensor2.getDim(); ++dim) pattern_image_tensor2.data_info.dimLength.push_back(image_tensor2.getShape(dim));
    Dac_Op pattern_image_tensor2_op_0;
    pattern_image_tensor2_op_0.setDimId(0);
    pattern_image_tensor2_op_0.size = 1;
    pattern_image_tensor2_op_0.stride = 1;
    pattern_image_tensor2_op_0.SetSplitSize(image_tensor2.getShape(0));
    pattern_image_tensor2.param_ops.push_back(pattern_image_tensor2_op_0);
    pattern_image_tensor2.bind_set_id.push_back(0);
    pattern_image_tensor2.bind_offset_expr.push_back("0");
    pattern_image_tensor2.is_index_op.push_back(true);
    Dac_Op pattern_image_tensor2_op_1;
    pattern_image_tensor2_op_1.setDimId(1);
    pattern_image_tensor2_op_1.size = 1;
    pattern_image_tensor2_op_1.stride = 1;
    pattern_image_tensor2_op_1.SetSplitSize(image_tensor2.getShape(1));
    pattern_image_tensor2.param_ops.push_back(pattern_image_tensor2_op_1);
    pattern_image_tensor2.bind_set_id.push_back(1);
    pattern_image_tensor2.bind_offset_expr.push_back("0");
    pattern_image_tensor2.is_index_op.push_back(true);
    pattern_image_tensor2.partition_shape = dacpp::mpi::init_partition_shape(pattern_image_tensor2);
    pattern_image_tensor2.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_image_tensor2);
    if (binding_split_sizes.size() < pattern_image_tensor2.bind_split_sizes.size()) binding_split_sizes.resize(pattern_image_tensor2.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_image_tensor2.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_image_tensor2.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_image_tensor.bind_split_sizes = binding_split_sizes;
    pattern_image_tensor2.bind_split_sizes = binding_split_sizes;
    auto pack_image_tensor = dacpp::mpi::build_input_pack_map(item_range, pattern_image_tensor);
    auto slots_image_tensor = dacpp::mpi::build_item_slots(item_range, pattern_image_tensor, pack_image_tensor);
    std::vector<Pixel> local_image_tensor(pack_image_tensor.globals.size());
    if (mpi_rank == 0) {
        std::vector<Pixel> global_image_tensor;
        image_tensor.tensor2Array(global_image_tensor);
        local_image_tensor = dacpp::mpi::pack_values_by_globals(global_image_tensor, pack_image_tensor.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_image_tensor);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_image_tensor, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), static_cast<int>((peer_count) * sizeof(Pixel)), MPI_BYTE, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_image_tensor.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_image_tensor.data(), static_cast<int>((recv_count) * sizeof(Pixel)), MPI_BYTE, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int image_tensor_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_image_tensor));
    auto pack_image_tensor2 = dacpp::mpi::build_output_pack_map(item_range, pattern_image_tensor2);
    auto slots_image_tensor2 = dacpp::mpi::build_item_slots(item_range, pattern_image_tensor2, pack_image_tensor2);
    std::vector<Pixel> local_image_tensor2(pack_image_tensor2.globals.size());
    const int image_tensor2_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_image_tensor2));
    if (local_item_count > 0) {
        {
            sycl::buffer<Pixel, 1> buffer_image_tensor(local_image_tensor.data(), sycl::range<1>(local_image_tensor.size()));
            sycl::buffer<int32_t, 1> slots_buffer_image_tensor(slots_image_tensor.data(), sycl::range<1>(slots_image_tensor.size()));
            sycl::buffer<Pixel, 1> buffer_image_tensor2(local_image_tensor2.data(), sycl::range<1>(local_image_tensor2.size()));
            sycl::buffer<int32_t, 1> slots_buffer_image_tensor2(slots_image_tensor2.data(), sycl::range<1>(slots_image_tensor2.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_image_tensor = buffer_image_tensor.get_access<sycl::access::mode::read>(h);
                auto slots_acc_image_tensor = slots_buffer_image_tensor.get_access<sycl::access::mode::read>(h);
                auto acc_image_tensor2 = buffer_image_tensor2.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_image_tensor2 = slots_buffer_image_tensor2.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_image_tensor = acc_image_tensor.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_image_tensor = slots_acc_image_tensor.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const Pixel> view_image_tensor{data_image_tensor, slots_image_tensor, item_linear * image_tensor_partition_size};
                    auto* data_image_tensor2 = acc_image_tensor2.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_image_tensor2 = slots_acc_image_tensor2.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<Pixel> view_image_tensor2{data_image_tensor2, slots_image_tensor2, item_linear * image_tensor2_partition_size};
                    image_1_mpi_local(view_image_tensor, view_image_tensor2);
                });
            });
            q.wait();
        }
    }
    auto writeback_image_tensor2 = dacpp::mpi::build_writeback_values(local_image_tensor2, pack_image_tensor2);
    const auto& writeback_globals_image_tensor2 = pack_image_tensor2.writeback_globals.empty() ? pack_image_tensor2.globals : pack_image_tensor2.writeback_globals;
    std::vector<Pixel> synced_image_tensor2;
    if (mpi_rank == 0) {
        std::vector<Pixel> global_out_image_tensor2;
        image_tensor2.tensor2Array(global_out_image_tensor2);
        dacpp::mpi::apply_writeback_by_globals(writeback_image_tensor2, writeback_globals_image_tensor2, global_out_image_tensor2);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<Pixel> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), static_cast<int>((recv_count) * sizeof(Pixel)), MPI_BYTE, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_image_tensor2);
        }
        image_tensor2.array2Tensor(global_out_image_tensor2);
        synced_image_tensor2 = global_out_image_tensor2;
    } else {
        int send_count = static_cast<int>(writeback_globals_image_tensor2.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_image_tensor2.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_image_tensor2.data(), static_cast<int>((send_count) * sizeof(Pixel)), MPI_BYTE, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_image_tensor2 = 0;
    if (mpi_rank == 0) {
        synced_count_image_tensor2 = static_cast<int>(synced_image_tensor2.size());
    }
    MPI_Bcast(&synced_count_image_tensor2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_image_tensor2.resize(synced_count_image_tensor2);
    }
    if (synced_count_image_tensor2 > 0) {
        MPI_Bcast(synced_image_tensor2.data(), static_cast<int>((synced_count_image_tensor2) * sizeof(Pixel)), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        image_tensor2.array2Tensor(synced_image_tensor2);
    }
}

inline void image_2_mpi_local(dacpp::mpi::View1D<const Pixel> image_tensor2, dacpp::mpi::View1D<Pixel> image_tensor3) {
    int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);
}


void imageAdjustment_image_2(const dacpp::Matrix<Pixel> & image_tensor, dacpp::Matrix<Pixel> & image_tensor2) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_image_tensor2;
    pattern_image_tensor2.param_id = 0;
    pattern_image_tensor2.name = "image_tensor2";
    pattern_image_tensor2.mode = dacpp::mpi::AccessMode::Read;
    pattern_image_tensor2.data_info.dim = image_tensor.getDim();
    for (int dim = 0; dim < image_tensor.getDim(); ++dim) pattern_image_tensor2.data_info.dimLength.push_back(image_tensor.getShape(dim));
    Dac_Op pattern_image_tensor2_op_0;
    pattern_image_tensor2_op_0.setDimId(0);
    pattern_image_tensor2_op_0.size = 1;
    pattern_image_tensor2_op_0.stride = 1;
    pattern_image_tensor2_op_0.SetSplitSize(image_tensor.getShape(0));
    pattern_image_tensor2.param_ops.push_back(pattern_image_tensor2_op_0);
    pattern_image_tensor2.bind_set_id.push_back(0);
    pattern_image_tensor2.bind_offset_expr.push_back("0");
    pattern_image_tensor2.is_index_op.push_back(true);
    Dac_Op pattern_image_tensor2_op_1;
    pattern_image_tensor2_op_1.setDimId(1);
    pattern_image_tensor2_op_1.size = 1;
    pattern_image_tensor2_op_1.stride = 1;
    pattern_image_tensor2_op_1.SetSplitSize(image_tensor.getShape(1));
    pattern_image_tensor2.param_ops.push_back(pattern_image_tensor2_op_1);
    pattern_image_tensor2.bind_set_id.push_back(1);
    pattern_image_tensor2.bind_offset_expr.push_back("0");
    pattern_image_tensor2.is_index_op.push_back(true);
    pattern_image_tensor2.partition_shape = dacpp::mpi::init_partition_shape(pattern_image_tensor2);
    pattern_image_tensor2.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_image_tensor2);
    if (binding_split_sizes.size() < pattern_image_tensor2.bind_split_sizes.size()) binding_split_sizes.resize(pattern_image_tensor2.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_image_tensor2.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_image_tensor2.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_image_tensor3;
    pattern_image_tensor3.param_id = 1;
    pattern_image_tensor3.name = "image_tensor3";
    pattern_image_tensor3.mode = dacpp::mpi::AccessMode::Write;
    pattern_image_tensor3.data_info.dim = image_tensor2.getDim();
    for (int dim = 0; dim < image_tensor2.getDim(); ++dim) pattern_image_tensor3.data_info.dimLength.push_back(image_tensor2.getShape(dim));
    Dac_Op pattern_image_tensor3_op_0;
    pattern_image_tensor3_op_0.setDimId(0);
    pattern_image_tensor3_op_0.size = 1;
    pattern_image_tensor3_op_0.stride = 1;
    pattern_image_tensor3_op_0.SetSplitSize(image_tensor2.getShape(0));
    pattern_image_tensor3.param_ops.push_back(pattern_image_tensor3_op_0);
    pattern_image_tensor3.bind_set_id.push_back(0);
    pattern_image_tensor3.bind_offset_expr.push_back("0");
    pattern_image_tensor3.is_index_op.push_back(true);
    Dac_Op pattern_image_tensor3_op_1;
    pattern_image_tensor3_op_1.setDimId(1);
    pattern_image_tensor3_op_1.size = 1;
    pattern_image_tensor3_op_1.stride = 1;
    pattern_image_tensor3_op_1.SetSplitSize(image_tensor2.getShape(1));
    pattern_image_tensor3.param_ops.push_back(pattern_image_tensor3_op_1);
    pattern_image_tensor3.bind_set_id.push_back(1);
    pattern_image_tensor3.bind_offset_expr.push_back("0");
    pattern_image_tensor3.is_index_op.push_back(true);
    pattern_image_tensor3.partition_shape = dacpp::mpi::init_partition_shape(pattern_image_tensor3);
    pattern_image_tensor3.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_image_tensor3);
    if (binding_split_sizes.size() < pattern_image_tensor3.bind_split_sizes.size()) binding_split_sizes.resize(pattern_image_tensor3.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_image_tensor3.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_image_tensor3.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_image_tensor2.bind_split_sizes = binding_split_sizes;
    pattern_image_tensor3.bind_split_sizes = binding_split_sizes;
    auto pack_image_tensor2 = dacpp::mpi::build_input_pack_map(item_range, pattern_image_tensor2);
    auto slots_image_tensor2 = dacpp::mpi::build_item_slots(item_range, pattern_image_tensor2, pack_image_tensor2);
    std::vector<Pixel> local_image_tensor2(pack_image_tensor2.globals.size());
    if (mpi_rank == 0) {
        std::vector<Pixel> global_image_tensor2;
        image_tensor.tensor2Array(global_image_tensor2);
        local_image_tensor2 = dacpp::mpi::pack_values_by_globals(global_image_tensor2, pack_image_tensor2.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_image_tensor2);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_image_tensor2, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), static_cast<int>((peer_count) * sizeof(Pixel)), MPI_BYTE, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_image_tensor2.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_image_tensor2.data(), static_cast<int>((recv_count) * sizeof(Pixel)), MPI_BYTE, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int image_tensor2_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_image_tensor2));
    auto pack_image_tensor3 = dacpp::mpi::build_output_pack_map(item_range, pattern_image_tensor3);
    auto slots_image_tensor3 = dacpp::mpi::build_item_slots(item_range, pattern_image_tensor3, pack_image_tensor3);
    std::vector<Pixel> local_image_tensor3(pack_image_tensor3.globals.size());
    const int image_tensor3_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_image_tensor3));
    if (local_item_count > 0) {
        {
            sycl::buffer<Pixel, 1> buffer_image_tensor2(local_image_tensor2.data(), sycl::range<1>(local_image_tensor2.size()));
            sycl::buffer<int32_t, 1> slots_buffer_image_tensor2(slots_image_tensor2.data(), sycl::range<1>(slots_image_tensor2.size()));
            sycl::buffer<Pixel, 1> buffer_image_tensor3(local_image_tensor3.data(), sycl::range<1>(local_image_tensor3.size()));
            sycl::buffer<int32_t, 1> slots_buffer_image_tensor3(slots_image_tensor3.data(), sycl::range<1>(slots_image_tensor3.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_image_tensor2 = buffer_image_tensor2.get_access<sycl::access::mode::read>(h);
                auto slots_acc_image_tensor2 = slots_buffer_image_tensor2.get_access<sycl::access::mode::read>(h);
                auto acc_image_tensor3 = buffer_image_tensor3.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_image_tensor3 = slots_buffer_image_tensor3.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_image_tensor2 = acc_image_tensor2.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_image_tensor2 = slots_acc_image_tensor2.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const Pixel> view_image_tensor2{data_image_tensor2, slots_image_tensor2, item_linear * image_tensor2_partition_size};
                    auto* data_image_tensor3 = acc_image_tensor3.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_image_tensor3 = slots_acc_image_tensor3.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<Pixel> view_image_tensor3{data_image_tensor3, slots_image_tensor3, item_linear * image_tensor3_partition_size};
                    image_2_mpi_local(view_image_tensor2, view_image_tensor3);
                });
            });
            q.wait();
        }
    }
    auto writeback_image_tensor3 = dacpp::mpi::build_writeback_values(local_image_tensor3, pack_image_tensor3);
    const auto& writeback_globals_image_tensor3 = pack_image_tensor3.writeback_globals.empty() ? pack_image_tensor3.globals : pack_image_tensor3.writeback_globals;
    std::vector<Pixel> synced_image_tensor3;
    if (mpi_rank == 0) {
        std::vector<Pixel> global_out_image_tensor3;
        image_tensor2.tensor2Array(global_out_image_tensor3);
        dacpp::mpi::apply_writeback_by_globals(writeback_image_tensor3, writeback_globals_image_tensor3, global_out_image_tensor3);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<Pixel> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), static_cast<int>((recv_count) * sizeof(Pixel)), MPI_BYTE, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_image_tensor3);
        }
        image_tensor2.array2Tensor(global_out_image_tensor3);
        synced_image_tensor3 = global_out_image_tensor3;
    } else {
        int send_count = static_cast<int>(writeback_globals_image_tensor3.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_image_tensor3.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_image_tensor3.data(), static_cast<int>((send_count) * sizeof(Pixel)), MPI_BYTE, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_image_tensor3 = 0;
    if (mpi_rank == 0) {
        synced_count_image_tensor3 = static_cast<int>(synced_image_tensor3.size());
    }
    MPI_Bcast(&synced_count_image_tensor3, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_image_tensor3.resize(synced_count_image_tensor3);
    }
    if (synced_count_image_tensor3 > 0) {
        MPI_Bcast(synced_image_tensor3.data(), static_cast<int>((synced_count_image_tensor3) * sizeof(Pixel)), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        image_tensor2.array2Tensor(synced_image_tensor3);
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

    // Use a fixed image size so every MPI rank sees the same dimensions
    // without competing for stdin.
    const int width = 4;
    const int height = 4;
    std::vector<Pixel> image(height*width, {100, 100, 100});
    std::vector<Pixel> image2(height*width, {100, 100, 100});
    //std::vector<std::vector<Pixel>> image2(height, std::vector<Pixel>(width, {100, 100, 100}));

    // 打印初始图像
    std::cout << "Original Image:" << std::endl;
    //print_image(image);

    dacpp::Matrix<Pixel> image_tensor({height, width}, image);
    dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);

    // 执行色彩调整操作
    imageAdjustment_image_1(image_tensor, image_tensor2);
    std::cout << "\nImage After Color Adjustment:" << std::endl;

    // image2 has been moved into image_tensor2 by ReconTensor's ownership model,
    // so build a fresh output buffer instead of reusing the moved-from vector.
    std::vector<Pixel> image3(height * width, {0, 0, 0});
    dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);


    // 执行亮度增强操作
    imageAdjustment_image_2(image_tensor2, image_tensor3);
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
