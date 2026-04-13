#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;









#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void vadd_mpi_local(dacpp::mpi::View1D<const float> lhs, dacpp::mpi::View1D<const float> rhs, dacpp::mpi::View1D<float> out) {
    out[0] = lhs[0] + rhs[0];
}


void VADD_vadd(const dacpp::Vector<float> & lhs, const dacpp::Vector<float> & rhs, dacpp::Vector<float> & out) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_lhs;
    pattern_lhs.param_id = 0;
    pattern_lhs.name = "lhs";
    pattern_lhs.mode = dacpp::mpi::AccessMode::Read;
    pattern_lhs.data_info.dim = lhs.getDim();
    for (int dim = 0; dim < lhs.getDim(); ++dim) pattern_lhs.data_info.dimLength.push_back(lhs.getShape(dim));
    Dac_Op pattern_lhs_op_0;
    pattern_lhs_op_0.setDimId(0);
    pattern_lhs_op_0.size = 1;
    pattern_lhs_op_0.stride = 1;
    pattern_lhs_op_0.SetSplitSize(lhs.getShape(0));
    pattern_lhs.param_ops.push_back(pattern_lhs_op_0);
    pattern_lhs.bind_set_id.push_back(0);
    pattern_lhs.bind_offset_expr.push_back("0");
    pattern_lhs.is_index_op.push_back(true);
    pattern_lhs.partition_shape = dacpp::mpi::init_partition_shape(pattern_lhs);
    pattern_lhs.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_lhs);
    if (binding_split_sizes.size() < pattern_lhs.bind_split_sizes.size()) binding_split_sizes.resize(pattern_lhs.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_lhs.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_lhs.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_rhs;
    pattern_rhs.param_id = 1;
    pattern_rhs.name = "rhs";
    pattern_rhs.mode = dacpp::mpi::AccessMode::Read;
    pattern_rhs.data_info.dim = rhs.getDim();
    for (int dim = 0; dim < rhs.getDim(); ++dim) pattern_rhs.data_info.dimLength.push_back(rhs.getShape(dim));
    Dac_Op pattern_rhs_op_0;
    pattern_rhs_op_0.setDimId(0);
    pattern_rhs_op_0.size = 1;
    pattern_rhs_op_0.stride = 1;
    pattern_rhs_op_0.SetSplitSize(rhs.getShape(0));
    pattern_rhs.param_ops.push_back(pattern_rhs_op_0);
    pattern_rhs.bind_set_id.push_back(0);
    pattern_rhs.bind_offset_expr.push_back("0");
    pattern_rhs.is_index_op.push_back(true);
    pattern_rhs.partition_shape = dacpp::mpi::init_partition_shape(pattern_rhs);
    pattern_rhs.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_rhs);
    if (binding_split_sizes.size() < pattern_rhs.bind_split_sizes.size()) binding_split_sizes.resize(pattern_rhs.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_rhs.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_rhs.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_out;
    pattern_out.param_id = 2;
    pattern_out.name = "out";
    pattern_out.mode = dacpp::mpi::AccessMode::Write;
    pattern_out.data_info.dim = out.getDim();
    for (int dim = 0; dim < out.getDim(); ++dim) pattern_out.data_info.dimLength.push_back(out.getShape(dim));
    Dac_Op pattern_out_op_0;
    pattern_out_op_0.setDimId(0);
    pattern_out_op_0.size = 1;
    pattern_out_op_0.stride = 1;
    pattern_out_op_0.SetSplitSize(out.getShape(0));
    pattern_out.param_ops.push_back(pattern_out_op_0);
    pattern_out.bind_set_id.push_back(0);
    pattern_out.bind_offset_expr.push_back("0");
    pattern_out.is_index_op.push_back(true);
    pattern_out.partition_shape = dacpp::mpi::init_partition_shape(pattern_out);
    pattern_out.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_out);
    if (binding_split_sizes.size() < pattern_out.bind_split_sizes.size()) binding_split_sizes.resize(pattern_out.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_out.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_out.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_lhs.bind_split_sizes = binding_split_sizes;
    pattern_rhs.bind_split_sizes = binding_split_sizes;
    pattern_out.bind_split_sizes = binding_split_sizes;
    auto pack_lhs = dacpp::mpi::build_input_pack_map(item_range, pattern_lhs);
    auto slots_lhs = dacpp::mpi::build_item_slots(item_range, pattern_lhs, pack_lhs);
    std::vector<float> local_lhs(pack_lhs.globals.size());
    int recv_count_lhs = 0;
    std::vector<int> sendcounts_lhs;
    std::vector<int> displs_lhs;
    std::vector<float> sendbuf_lhs;
    if (mpi_rank == 0) {
        sendcounts_lhs.resize(mpi_size);
        displs_lhs.resize(mpi_size);
        int current_displ = 0;
        std::vector<float> global_lhs;
        lhs.tensor2Array(global_lhs);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_lhs);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_lhs, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_lhs[r] = r_count;
            displs_lhs[r] = current_displ;
            current_displ += r_count;
            sendbuf_lhs.insert(sendbuf_lhs.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_lhs.data() : nullptr, 1, MPI_INT, &recv_count_lhs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_lhs.resize(recv_count_lhs);
    std::vector<int> sendcounts_bytes_lhs = sendcounts_lhs;
    std::vector<int> displs_bytes_lhs = displs_lhs;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_lhs.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_lhs.data() : nullptr, mpi_rank == 0 ? displs_bytes_lhs.data() : nullptr, MPI_FLOAT, local_lhs.data(), recv_count_lhs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    const int lhs_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_lhs));
    auto pack_rhs = dacpp::mpi::build_input_pack_map(item_range, pattern_rhs);
    auto slots_rhs = dacpp::mpi::build_item_slots(item_range, pattern_rhs, pack_rhs);
    std::vector<float> local_rhs(pack_rhs.globals.size());
    int recv_count_rhs = 0;
    std::vector<int> sendcounts_rhs;
    std::vector<int> displs_rhs;
    std::vector<float> sendbuf_rhs;
    if (mpi_rank == 0) {
        sendcounts_rhs.resize(mpi_size);
        displs_rhs.resize(mpi_size);
        int current_displ = 0;
        std::vector<float> global_rhs;
        rhs.tensor2Array(global_rhs);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_rhs);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_rhs, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_rhs[r] = r_count;
            displs_rhs[r] = current_displ;
            current_displ += r_count;
            sendbuf_rhs.insert(sendbuf_rhs.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_rhs.data() : nullptr, 1, MPI_INT, &recv_count_rhs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_rhs.resize(recv_count_rhs);
    std::vector<int> sendcounts_bytes_rhs = sendcounts_rhs;
    std::vector<int> displs_bytes_rhs = displs_rhs;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_rhs.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_rhs.data() : nullptr, mpi_rank == 0 ? displs_bytes_rhs.data() : nullptr, MPI_FLOAT, local_rhs.data(), recv_count_rhs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    const int rhs_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_rhs));
    auto pack_out = dacpp::mpi::build_output_pack_map(item_range, pattern_out);
    auto slots_out = dacpp::mpi::build_item_slots(item_range, pattern_out, pack_out);
    std::vector<float> local_out(pack_out.globals.size());
    const int out_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_out));
    if (local_item_count > 0) {
        {
            sycl::buffer<float, 1> buffer_lhs(local_lhs.data(), sycl::range<1>(local_lhs.size()));
            sycl::buffer<int32_t, 1> slots_buffer_lhs(slots_lhs.data(), sycl::range<1>(slots_lhs.size()));
            sycl::buffer<float, 1> buffer_rhs(local_rhs.data(), sycl::range<1>(local_rhs.size()));
            sycl::buffer<int32_t, 1> slots_buffer_rhs(slots_rhs.data(), sycl::range<1>(slots_rhs.size()));
            sycl::buffer<float, 1> buffer_out(local_out.data(), sycl::range<1>(local_out.size()));
            sycl::buffer<int32_t, 1> slots_buffer_out(slots_out.data(), sycl::range<1>(slots_out.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_lhs = buffer_lhs.get_access<sycl::access::mode::read>(h);
                auto slots_acc_lhs = slots_buffer_lhs.get_access<sycl::access::mode::read>(h);
                auto acc_rhs = buffer_rhs.get_access<sycl::access::mode::read>(h);
                auto slots_acc_rhs = slots_buffer_rhs.get_access<sycl::access::mode::read>(h);
                auto acc_out = buffer_out.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_out = slots_buffer_out.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_lhs = acc_lhs.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_lhs = slots_acc_lhs.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_lhs{data_lhs, slots_lhs, item_linear * lhs_partition_size};
                    auto* data_rhs = acc_rhs.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_rhs = slots_acc_rhs.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_rhs{data_rhs, slots_rhs, item_linear * rhs_partition_size};
                    auto* data_out = acc_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_out = slots_acc_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<float> view_out{data_out, slots_out, item_linear * out_partition_size};
                    vadd_mpi_local(view_lhs, view_rhs, view_out);
                });
            });
            q.wait();
        }
    }
    auto writeback_out = dacpp::mpi::build_writeback_values(local_out, pack_out);
    const auto& writeback_globals_out = pack_out.writeback_globals.empty() ? pack_out.globals : pack_out.writeback_globals;
    std::vector<float> synced_out;
    int send_count_out = static_cast<int>(writeback_globals_out.size());
    std::vector<int> recvcounts_out;
    std::vector<int> recvdispls_out;
    std::vector<int64_t> global_recv_globals_out;
    std::vector<float> global_recv_values_out;
    if (mpi_rank == 0) {
        recvcounts_out.resize(mpi_size);
        recvdispls_out.resize(mpi_size);
    }
    MPI_Gather(&send_count_out, 1, MPI_INT, mpi_rank == 0 ? recvcounts_out.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_out[r] = current_displ;
            current_displ += recvcounts_out[r];
        }
        global_recv_globals_out.resize(current_displ);
        global_recv_values_out.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_out.data()), send_count_out, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_out.data() : nullptr, mpi_rank == 0 ? recvcounts_out.data() : nullptr, mpi_rank == 0 ? recvdispls_out.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> recvcounts_bytes_out = recvcounts_out;
    std::vector<int> recvdispls_bytes_out = recvdispls_out;
    MPI_Gatherv(writeback_out.data(), send_count_out, MPI_FLOAT, mpi_rank == 0 ? global_recv_values_out.data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_out.data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_out.data() : nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<float> global_out_out;
        out.tensor2Array(global_out_out);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_out, global_recv_globals_out, global_out_out);
        out.array2Tensor(global_out_out);
        synced_out = global_out_out;
    }
    int synced_count_out = 0;
    if (mpi_rank == 0) {
        synced_count_out = static_cast<int>(synced_out.size());
    }
    MPI_Bcast(&synced_count_out, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_out.resize(synced_count_out);
    }
    if (synced_count_out > 0) {
        MPI_Bcast(synced_out.data(), synced_count_out, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        out.array2Tensor(synced_out);
    }
}

inline void vshift_mpi_local(dacpp::mpi::View1D<const float> in, dacpp::mpi::View1D<const float> bias, dacpp::mpi::View1D<float> out) {
    out[0] = in[0] + bias[0];
}


void VSHIFT_vshift(const dacpp::Vector<float> & in, const dacpp::Vector<float> & bias, dacpp::Vector<float> & out) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_in;
    pattern_in.param_id = 0;
    pattern_in.name = "in";
    pattern_in.mode = dacpp::mpi::AccessMode::Read;
    pattern_in.data_info.dim = in.getDim();
    for (int dim = 0; dim < in.getDim(); ++dim) pattern_in.data_info.dimLength.push_back(in.getShape(dim));
    Dac_Op pattern_in_op_0;
    pattern_in_op_0.setDimId(0);
    pattern_in_op_0.size = 1;
    pattern_in_op_0.stride = 1;
    pattern_in_op_0.SetSplitSize(in.getShape(0));
    pattern_in.param_ops.push_back(pattern_in_op_0);
    pattern_in.bind_set_id.push_back(0);
    pattern_in.bind_offset_expr.push_back("0");
    pattern_in.is_index_op.push_back(true);
    pattern_in.partition_shape = dacpp::mpi::init_partition_shape(pattern_in);
    pattern_in.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_in);
    if (binding_split_sizes.size() < pattern_in.bind_split_sizes.size()) binding_split_sizes.resize(pattern_in.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_in.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_in.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_bias;
    pattern_bias.param_id = 1;
    pattern_bias.name = "bias";
    pattern_bias.mode = dacpp::mpi::AccessMode::Read;
    pattern_bias.data_info.dim = bias.getDim();
    for (int dim = 0; dim < bias.getDim(); ++dim) pattern_bias.data_info.dimLength.push_back(bias.getShape(dim));
    pattern_bias.partition_shape = dacpp::mpi::init_partition_shape(pattern_bias);
    pattern_bias.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_bias);
    if (binding_split_sizes.size() < pattern_bias.bind_split_sizes.size()) binding_split_sizes.resize(pattern_bias.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_bias.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_bias.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_out;
    pattern_out.param_id = 2;
    pattern_out.name = "out";
    pattern_out.mode = dacpp::mpi::AccessMode::Write;
    pattern_out.data_info.dim = out.getDim();
    for (int dim = 0; dim < out.getDim(); ++dim) pattern_out.data_info.dimLength.push_back(out.getShape(dim));
    Dac_Op pattern_out_op_0;
    pattern_out_op_0.setDimId(0);
    pattern_out_op_0.size = 1;
    pattern_out_op_0.stride = 1;
    pattern_out_op_0.SetSplitSize(out.getShape(0));
    pattern_out.param_ops.push_back(pattern_out_op_0);
    pattern_out.bind_set_id.push_back(0);
    pattern_out.bind_offset_expr.push_back("0");
    pattern_out.is_index_op.push_back(true);
    pattern_out.partition_shape = dacpp::mpi::init_partition_shape(pattern_out);
    pattern_out.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_out);
    if (binding_split_sizes.size() < pattern_out.bind_split_sizes.size()) binding_split_sizes.resize(pattern_out.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_out.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_out.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_in.bind_split_sizes = binding_split_sizes;
    pattern_bias.bind_split_sizes = binding_split_sizes;
    pattern_out.bind_split_sizes = binding_split_sizes;
    auto pack_in = dacpp::mpi::build_input_pack_map(item_range, pattern_in);
    auto slots_in = dacpp::mpi::build_item_slots(item_range, pattern_in, pack_in);
    std::vector<float> local_in(pack_in.globals.size());
    int recv_count_in = 0;
    std::vector<int> sendcounts_in;
    std::vector<int> displs_in;
    std::vector<float> sendbuf_in;
    if (mpi_rank == 0) {
        sendcounts_in.resize(mpi_size);
        displs_in.resize(mpi_size);
        int current_displ = 0;
        std::vector<float> global_in;
        in.tensor2Array(global_in);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_in);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_in, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_in[r] = r_count;
            displs_in[r] = current_displ;
            current_displ += r_count;
            sendbuf_in.insert(sendbuf_in.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_in.data() : nullptr, 1, MPI_INT, &recv_count_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_in.resize(recv_count_in);
    std::vector<int> sendcounts_bytes_in = sendcounts_in;
    std::vector<int> displs_bytes_in = displs_in;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_in.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_in.data() : nullptr, mpi_rank == 0 ? displs_bytes_in.data() : nullptr, MPI_FLOAT, local_in.data(), recv_count_in, MPI_FLOAT, 0, MPI_COMM_WORLD);
    const int in_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_in));
    auto pack_bias = dacpp::mpi::build_input_pack_map(item_range, pattern_bias);
    auto slots_bias = dacpp::mpi::build_item_slots(item_range, pattern_bias, pack_bias);
    std::vector<float> local_bias(pack_bias.globals.size());
    int recv_count_bias = 0;
    std::vector<int> sendcounts_bias;
    std::vector<int> displs_bias;
    std::vector<float> sendbuf_bias;
    if (mpi_rank == 0) {
        sendcounts_bias.resize(mpi_size);
        displs_bias.resize(mpi_size);
        int current_displ = 0;
        std::vector<float> global_bias;
        bias.tensor2Array(global_bias);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_bias);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_bias, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_bias[r] = r_count;
            displs_bias[r] = current_displ;
            current_displ += r_count;
            sendbuf_bias.insert(sendbuf_bias.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_bias.data() : nullptr, 1, MPI_INT, &recv_count_bias, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_bias.resize(recv_count_bias);
    std::vector<int> sendcounts_bytes_bias = sendcounts_bias;
    std::vector<int> displs_bytes_bias = displs_bias;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_bias.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_bias.data() : nullptr, mpi_rank == 0 ? displs_bytes_bias.data() : nullptr, MPI_FLOAT, local_bias.data(), recv_count_bias, MPI_FLOAT, 0, MPI_COMM_WORLD);
    const int bias_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_bias));
    auto pack_out = dacpp::mpi::build_output_pack_map(item_range, pattern_out);
    auto slots_out = dacpp::mpi::build_item_slots(item_range, pattern_out, pack_out);
    std::vector<float> local_out(pack_out.globals.size());
    const int out_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_out));
    if (local_item_count > 0) {
        {
            sycl::buffer<float, 1> buffer_in(local_in.data(), sycl::range<1>(local_in.size()));
            sycl::buffer<int32_t, 1> slots_buffer_in(slots_in.data(), sycl::range<1>(slots_in.size()));
            sycl::buffer<float, 1> buffer_bias(local_bias.data(), sycl::range<1>(local_bias.size()));
            sycl::buffer<int32_t, 1> slots_buffer_bias(slots_bias.data(), sycl::range<1>(slots_bias.size()));
            sycl::buffer<float, 1> buffer_out(local_out.data(), sycl::range<1>(local_out.size()));
            sycl::buffer<int32_t, 1> slots_buffer_out(slots_out.data(), sycl::range<1>(slots_out.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_in = buffer_in.get_access<sycl::access::mode::read>(h);
                auto slots_acc_in = slots_buffer_in.get_access<sycl::access::mode::read>(h);
                auto acc_bias = buffer_bias.get_access<sycl::access::mode::read>(h);
                auto slots_acc_bias = slots_buffer_bias.get_access<sycl::access::mode::read>(h);
                auto acc_out = buffer_out.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_out = slots_buffer_out.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_in = acc_in.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_in = slots_acc_in.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_in{data_in, slots_in, item_linear * in_partition_size};
                    auto* data_bias = acc_bias.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_bias = slots_acc_bias.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const float> view_bias{data_bias, slots_bias, item_linear * bias_partition_size};
                    auto* data_out = acc_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_out = slots_acc_out.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<float> view_out{data_out, slots_out, item_linear * out_partition_size};
                    vshift_mpi_local(view_in, view_bias, view_out);
                });
            });
            q.wait();
        }
    }
    auto writeback_out = dacpp::mpi::build_writeback_values(local_out, pack_out);
    const auto& writeback_globals_out = pack_out.writeback_globals.empty() ? pack_out.globals : pack_out.writeback_globals;
    std::vector<float> synced_out;
    int send_count_out = static_cast<int>(writeback_globals_out.size());
    std::vector<int> recvcounts_out;
    std::vector<int> recvdispls_out;
    std::vector<int64_t> global_recv_globals_out;
    std::vector<float> global_recv_values_out;
    if (mpi_rank == 0) {
        recvcounts_out.resize(mpi_size);
        recvdispls_out.resize(mpi_size);
    }
    MPI_Gather(&send_count_out, 1, MPI_INT, mpi_rank == 0 ? recvcounts_out.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_out[r] = current_displ;
            current_displ += recvcounts_out[r];
        }
        global_recv_globals_out.resize(current_displ);
        global_recv_values_out.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_out.data()), send_count_out, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_out.data() : nullptr, mpi_rank == 0 ? recvcounts_out.data() : nullptr, mpi_rank == 0 ? recvdispls_out.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> recvcounts_bytes_out = recvcounts_out;
    std::vector<int> recvdispls_bytes_out = recvdispls_out;
    MPI_Gatherv(writeback_out.data(), send_count_out, MPI_FLOAT, mpi_rank == 0 ? global_recv_values_out.data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_out.data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_out.data() : nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<float> global_out_out;
        out.tensor2Array(global_out_out);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_out, global_recv_globals_out, global_out_out);
        out.array2Tensor(global_out_out);
        synced_out = global_out_out;
    }
    int synced_count_out = 0;
    if (mpi_rank == 0) {
        synced_count_out = static_cast<int>(synced_out.size());
    }
    MPI_Bcast(&synced_count_out, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_out.resize(synced_count_out);
    }
    if (synced_count_out > 0) {
        MPI_Bcast(synced_out.data(), synced_count_out, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        out.array2Tensor(synced_out);
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

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> c{100, 200, 300, 400, 500, 600, 700, 800};
    std::vector<float> bias{0.5f};
    std::vector<float> tmp(N, 0.0f);
    std::vector<float> shifted(N, 0.0f);
    std::vector<float> out(N, 0.0f);

    dacpp::Vector<float> a_tensor(a);
    dacpp::Vector<float> b_tensor(b);
    dacpp::Vector<float> c_tensor(c);
    dacpp::Vector<float> bias_tensor(bias);
    dacpp::Vector<float> tmp_tensor(tmp);
    dacpp::Vector<float> shifted_tensor(shifted);
    dacpp::Vector<float> out_tensor(out);

    VADD_vadd(a_tensor, b_tensor, tmp_tensor);
    VSHIFT_vshift(tmp_tensor, bias_tensor, shifted_tensor);
    VADD_vadd(shifted_tensor, c_tensor, out_tensor);

    std::vector<float> host_out;
    out_tensor.tensor2Array(host_out);

    for (float value : host_out) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
