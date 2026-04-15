#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数

const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const double Lx = 10.0f; // x方向长度
const double Ly = 10.0f; // y方向长度
const double c = 1.0f;   // 波速
const int TIME_STEPS = 10; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void waveEq_mpi_local(dacpp::mpi::View2D<const double> cur, dacpp::mpi::View1D<const double> prev, dacpp::mpi::View1D<double> next) {
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[2][1] - 2.F * cur[1][1] + cur[0][1]) / (dx * dx);
    double u_yy = (cur[1][2] - 2.F * cur[1][1] + cur[1][0]) / (dy * dy);
    next[0] = 2.F * cur[1][1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}


void waveEqShell_waveEq(dacpp::Matrix<double> & matCur, dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_cur;
    pattern_cur.param_id = 0;
    pattern_cur.name = "cur";
    pattern_cur.mode = dacpp::mpi::AccessMode::Read;
    pattern_cur.data_info.dim = matCur.getDim();
    for (int dim = 0; dim < matCur.getDim(); ++dim) pattern_cur.data_info.dimLength.push_back(matCur.getShape(dim));
    Dac_Op pattern_cur_op_0;
    pattern_cur_op_0.setDimId(0);
    pattern_cur_op_0.size = 3;
    pattern_cur_op_0.stride = 1;
    pattern_cur_op_0.SetSplitSize((matCur.getShape(0) - 3) / 1 + 1);
    pattern_cur.param_ops.push_back(pattern_cur_op_0);
    pattern_cur.bind_set_id.push_back(0);
    pattern_cur.bind_offset_expr.push_back("0");
    pattern_cur.is_index_op.push_back(false);
    Dac_Op pattern_cur_op_1;
    pattern_cur_op_1.setDimId(1);
    pattern_cur_op_1.size = 3;
    pattern_cur_op_1.stride = 1;
    pattern_cur_op_1.SetSplitSize((matCur.getShape(1) - 3) / 1 + 1);
    pattern_cur.param_ops.push_back(pattern_cur_op_1);
    pattern_cur.bind_set_id.push_back(1);
    pattern_cur.bind_offset_expr.push_back("0");
    pattern_cur.is_index_op.push_back(false);
    pattern_cur.partition_shape = dacpp::mpi::init_partition_shape(pattern_cur);
    pattern_cur.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_cur);
    if (binding_split_sizes.size() < pattern_cur.bind_split_sizes.size()) binding_split_sizes.resize(pattern_cur.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_cur.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_cur.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_prev;
    pattern_prev.param_id = 1;
    pattern_prev.name = "prev";
    pattern_prev.mode = dacpp::mpi::AccessMode::Read;
    pattern_prev.data_info.dim = matPrev.getDim();
    for (int dim = 0; dim < matPrev.getDim(); ++dim) pattern_prev.data_info.dimLength.push_back(matPrev.getShape(dim));
    Dac_Op pattern_prev_op_0;
    pattern_prev_op_0.setDimId(0);
    pattern_prev_op_0.size = 1;
    pattern_prev_op_0.stride = 1;
    pattern_prev_op_0.SetSplitSize(matPrev.getShape(0));
    pattern_prev.param_ops.push_back(pattern_prev_op_0);
    pattern_prev.bind_set_id.push_back(0);
    pattern_prev.bind_offset_expr.push_back("0");
    pattern_prev.is_index_op.push_back(true);
    Dac_Op pattern_prev_op_1;
    pattern_prev_op_1.setDimId(1);
    pattern_prev_op_1.size = 1;
    pattern_prev_op_1.stride = 1;
    pattern_prev_op_1.SetSplitSize(matPrev.getShape(1));
    pattern_prev.param_ops.push_back(pattern_prev_op_1);
    pattern_prev.bind_set_id.push_back(1);
    pattern_prev.bind_offset_expr.push_back("0");
    pattern_prev.is_index_op.push_back(true);
    pattern_prev.partition_shape = dacpp::mpi::init_partition_shape(pattern_prev);
    pattern_prev.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_prev);
    if (binding_split_sizes.size() < pattern_prev.bind_split_sizes.size()) binding_split_sizes.resize(pattern_prev.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_prev.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_prev.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_next;
    pattern_next.param_id = 2;
    pattern_next.name = "next";
    pattern_next.mode = dacpp::mpi::AccessMode::Write;
    pattern_next.data_info.dim = matNext.getDim();
    for (int dim = 0; dim < matNext.getDim(); ++dim) pattern_next.data_info.dimLength.push_back(matNext.getShape(dim));
    Dac_Op pattern_next_op_0;
    pattern_next_op_0.setDimId(0);
    pattern_next_op_0.size = 1;
    pattern_next_op_0.stride = 1;
    pattern_next_op_0.SetSplitSize(matNext.getShape(0));
    pattern_next.param_ops.push_back(pattern_next_op_0);
    pattern_next.bind_set_id.push_back(0);
    pattern_next.bind_offset_expr.push_back("0");
    pattern_next.is_index_op.push_back(true);
    Dac_Op pattern_next_op_1;
    pattern_next_op_1.setDimId(1);
    pattern_next_op_1.size = 1;
    pattern_next_op_1.stride = 1;
    pattern_next_op_1.SetSplitSize(matNext.getShape(1));
    pattern_next.param_ops.push_back(pattern_next_op_1);
    pattern_next.bind_set_id.push_back(1);
    pattern_next.bind_offset_expr.push_back("0");
    pattern_next.is_index_op.push_back(true);
    pattern_next.partition_shape = dacpp::mpi::init_partition_shape(pattern_next);
    pattern_next.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_next);
    if (binding_split_sizes.size() < pattern_next.bind_split_sizes.size()) binding_split_sizes.resize(pattern_next.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_next.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_next.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_cur.bind_split_sizes = binding_split_sizes;
    pattern_prev.bind_split_sizes = binding_split_sizes;
    pattern_next.bind_split_sizes = binding_split_sizes;
    auto pack_cur = dacpp::mpi::build_input_pack_map(item_range, pattern_cur);
    auto slots_cur = dacpp::mpi::build_item_slots(item_range, pattern_cur, pack_cur);
    std::vector<double> local_cur(pack_cur.globals.size());
    int recv_count_cur = 0;
    std::vector<int> sendcounts_cur;
    std::vector<int> displs_cur;
    std::vector<double> sendbuf_cur;
    if (mpi_rank == 0) {
        sendcounts_cur.resize(mpi_size);
        displs_cur.resize(mpi_size);
        int current_displ = 0;
        std::vector<double> global_cur;
        matCur.tensor2Array(global_cur);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_cur);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_cur, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_cur[r] = r_count;
            displs_cur[r] = current_displ;
            current_displ += r_count;
            sendbuf_cur.insert(sendbuf_cur.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_cur.data() : nullptr, 1, MPI_INT, &recv_count_cur, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_cur.resize(recv_count_cur);
    std::vector<int> sendcounts_bytes_cur = sendcounts_cur;
    std::vector<int> displs_bytes_cur = displs_cur;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_cur.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_cur.data() : nullptr, mpi_rank == 0 ? displs_bytes_cur.data() : nullptr, MPI_DOUBLE, local_cur.data(), recv_count_cur, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int cur_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_cur));
    const int cur_cols = pattern_cur.partition_shape[1];
    auto pack_prev = dacpp::mpi::build_input_pack_map(item_range, pattern_prev);
    auto slots_prev = dacpp::mpi::build_item_slots(item_range, pattern_prev, pack_prev);
    std::vector<double> local_prev(pack_prev.globals.size());
    int recv_count_prev = 0;
    std::vector<int> sendcounts_prev;
    std::vector<int> displs_prev;
    std::vector<double> sendbuf_prev;
    if (mpi_rank == 0) {
        sendcounts_prev.resize(mpi_size);
        displs_prev.resize(mpi_size);
        int current_displ = 0;
        std::vector<double> global_prev;
        matPrev.tensor2Array(global_prev);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_prev);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_prev, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_prev[r] = r_count;
            displs_prev[r] = current_displ;
            current_displ += r_count;
            sendbuf_prev.insert(sendbuf_prev.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_prev.data() : nullptr, 1, MPI_INT, &recv_count_prev, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_prev.resize(recv_count_prev);
    std::vector<int> sendcounts_bytes_prev = sendcounts_prev;
    std::vector<int> displs_bytes_prev = displs_prev;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_prev.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_prev.data() : nullptr, mpi_rank == 0 ? displs_bytes_prev.data() : nullptr, MPI_DOUBLE, local_prev.data(), recv_count_prev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int prev_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_prev));
    auto pack_next = dacpp::mpi::build_output_pack_map(item_range, pattern_next);
    auto slots_next = dacpp::mpi::build_item_slots(item_range, pattern_next, pack_next);
    std::vector<double> local_next(pack_next.globals.size());
    const int next_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_next));
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_cur(local_cur.data(), sycl::range<1>(local_cur.size()));
            sycl::buffer<int32_t, 1> slots_buffer_cur(slots_cur.data(), sycl::range<1>(slots_cur.size()));
            sycl::buffer<double, 1> buffer_prev(local_prev.data(), sycl::range<1>(local_prev.size()));
            sycl::buffer<int32_t, 1> slots_buffer_prev(slots_prev.data(), sycl::range<1>(slots_prev.size()));
            sycl::buffer<double, 1> buffer_next(local_next.data(), sycl::range<1>(local_next.size()));
            sycl::buffer<int32_t, 1> slots_buffer_next(slots_next.data(), sycl::range<1>(slots_next.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_cur = buffer_cur.get_access<sycl::access::mode::read>(h);
                auto slots_acc_cur = slots_buffer_cur.get_access<sycl::access::mode::read>(h);
                auto acc_prev = buffer_prev.get_access<sycl::access::mode::read>(h);
                auto slots_acc_prev = slots_buffer_prev.get_access<sycl::access::mode::read>(h);
                auto acc_next = buffer_next.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_next = slots_buffer_next.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_cur = acc_cur.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_cur = slots_acc_cur.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View2D<const double> view_cur{data_cur, slots_cur, item_linear * cur_partition_size, cur_cols};
                    auto* data_prev = acc_prev.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_prev = slots_acc_prev.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_prev{data_prev, slots_prev, item_linear * prev_partition_size};
                    auto* data_next = acc_next.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_next = slots_acc_next.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_next{data_next, slots_next, item_linear * next_partition_size};
                    waveEq_mpi_local(view_cur, view_prev, view_next);
                });
            });
            q.wait();
        }
    }
    auto writeback_next = dacpp::mpi::build_writeback_values(local_next, pack_next);
    const auto& writeback_globals_next = pack_next.writeback_globals.empty() ? pack_next.globals : pack_next.writeback_globals;
    std::vector<double> synced_next;
    int send_count_next = static_cast<int>(writeback_globals_next.size());
    std::vector<int> recvcounts_next;
    std::vector<int> recvdispls_next;
    std::vector<int64_t> global_recv_globals_next;
    std::vector<double> global_recv_values_next;
    if (mpi_rank == 0) {
        recvcounts_next.resize(mpi_size);
        recvdispls_next.resize(mpi_size);
    }
    MPI_Gather(&send_count_next, 1, MPI_INT, mpi_rank == 0 ? recvcounts_next.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_next[r] = current_displ;
            current_displ += recvcounts_next[r];
        }
        global_recv_globals_next.resize(current_displ);
        global_recv_values_next.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_next.data()), send_count_next, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_next.data() : nullptr, mpi_rank == 0 ? recvcounts_next.data() : nullptr, mpi_rank == 0 ? recvdispls_next.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> recvcounts_bytes_next = recvcounts_next;
    std::vector<int> recvdispls_bytes_next = recvdispls_next;
    MPI_Gatherv(writeback_next.data(), send_count_next, MPI_DOUBLE, mpi_rank == 0 ? global_recv_values_next.data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_next.data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_next.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<double> global_out_next;
        matNext.tensor2Array(global_out_next);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_next, global_recv_globals_next, global_out_next);
        matNext.array2Tensor(global_out_next);
        synced_next = global_out_next;
    }
    int synced_count_next = 0;
    if (mpi_rank == 0) {
        synced_count_next = static_cast<int>(synced_next.size());
    }
    MPI_Bcast(&synced_count_next, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_next.resize(synced_count_next);
    }
    if (synced_count_next > 0) {
        MPI_Bcast(synced_next.data(), synced_count_next, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        matNext.array2Tensor(synced_next);
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

    // 初始化波场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    // for(int i = 0; i < 1; i++){
    //     for(int j = 0; j < NY; j++){
    //         //std::cout << u_prev[i * NX + j] << " " ;
    //     }
    //     //std::cout << std::endl;
    // }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX-1}][{1,NY-1}];
    
    for(int i = 0;i < TIME_STEPS; i++) {
        waveEqShell_waveEq(matCur, matPrev, matNext);
        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matPrev[i-1][j-1]=matCur[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matCur[i][j]=matNext[i-1][j-1];
            }
        }
        // 处理边界条件（绝热边界：导数为零）
        for (int i = 0; i <= NX-1; ++i) {       
            matCur[i][NY-1]=0;
            matCur[i][0]=0;
        }
        for (int j = 0; j <= NY-1; ++j) {
            matCur[NX - 1][j]=0;
            matCur[0][j]=0;
             // 底部边界
        }
    }
    //
    matCur.print(); 
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}