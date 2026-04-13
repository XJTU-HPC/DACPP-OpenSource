#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <any>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <queue>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1

//vector的相关操作有待实现
namespace dacpp {
    typedef std::vector<std::any> list;
}

double phi(double x) { return x*x*x+x; }

double alpha(double t) { return 0.0; }

double beta(double t) { return 1.0+exp(t); }

double f(double x, double t) { return x*exp(t)-6*x; }

double exact(double x, double t) { return x*(x*x+exp(t)); }

//同样的问题，划分时，一个待计算数据和三个计算数据，一共四个数据要划分到一起




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void pde_mpi_local(dacpp::mpi::View1D<const double> u_kin, dacpp::mpi::View1D<double> u_kout, dacpp::mpi::View1D<const double> r) {
    u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}


void PDE_pde(const dacpp::Vector<double> & u_kin, dacpp::Vector<double> & u_kout, const dacpp::Vector<double> & r) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_u_kin;
    pattern_u_kin.param_id = 0;
    pattern_u_kin.name = "u_kin";
    pattern_u_kin.mode = dacpp::mpi::AccessMode::Read;
    pattern_u_kin.data_info.dim = u_kin.getDim();
    for (int dim = 0; dim < u_kin.getDim(); ++dim) pattern_u_kin.data_info.dimLength.push_back(u_kin.getShape(dim));
    Dac_Op pattern_u_kin_op_0;
    pattern_u_kin_op_0.setDimId(0);
    pattern_u_kin_op_0.size = 3;
    pattern_u_kin_op_0.stride = 1;
    pattern_u_kin_op_0.SetSplitSize((u_kin.getShape(0) - 3) / 1 + 1);
    pattern_u_kin.param_ops.push_back(pattern_u_kin_op_0);
    pattern_u_kin.bind_set_id.push_back(0);
    pattern_u_kin.bind_offset_expr.push_back("0");
    pattern_u_kin.is_index_op.push_back(false);
    pattern_u_kin.partition_shape = dacpp::mpi::init_partition_shape(pattern_u_kin);
    pattern_u_kin.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_u_kin);
    if (binding_split_sizes.size() < pattern_u_kin.bind_split_sizes.size()) binding_split_sizes.resize(pattern_u_kin.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_u_kin.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_u_kin.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_u_kout;
    pattern_u_kout.param_id = 1;
    pattern_u_kout.name = "u_kout";
    pattern_u_kout.mode = dacpp::mpi::AccessMode::Write;
    pattern_u_kout.data_info.dim = u_kout.getDim();
    for (int dim = 0; dim < u_kout.getDim(); ++dim) pattern_u_kout.data_info.dimLength.push_back(u_kout.getShape(dim));
    Dac_Op pattern_u_kout_op_0;
    pattern_u_kout_op_0.setDimId(0);
    pattern_u_kout_op_0.size = 1;
    pattern_u_kout_op_0.stride = 1;
    pattern_u_kout_op_0.SetSplitSize(u_kout.getShape(0));
    pattern_u_kout.param_ops.push_back(pattern_u_kout_op_0);
    pattern_u_kout.bind_set_id.push_back(1);
    pattern_u_kout.bind_offset_expr.push_back("0");
    pattern_u_kout.is_index_op.push_back(true);
    pattern_u_kout.partition_shape = dacpp::mpi::init_partition_shape(pattern_u_kout);
    pattern_u_kout.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_u_kout);
    if (binding_split_sizes.size() < pattern_u_kout.bind_split_sizes.size()) binding_split_sizes.resize(pattern_u_kout.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_u_kout.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_u_kout.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_r;
    pattern_r.param_id = 2;
    pattern_r.name = "r";
    pattern_r.mode = dacpp::mpi::AccessMode::Read;
    pattern_r.data_info.dim = r.getDim();
    for (int dim = 0; dim < r.getDim(); ++dim) pattern_r.data_info.dimLength.push_back(r.getShape(dim));
    pattern_r.partition_shape = dacpp::mpi::init_partition_shape(pattern_r);
    pattern_r.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_r);
    if (binding_split_sizes.size() < pattern_r.bind_split_sizes.size()) binding_split_sizes.resize(pattern_r.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_r.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_r.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_u_kin.bind_split_sizes = binding_split_sizes;
    pattern_u_kout.bind_split_sizes = binding_split_sizes;
    pattern_r.bind_split_sizes = binding_split_sizes;
    auto pack_u_kin = dacpp::mpi::build_input_pack_map(item_range, pattern_u_kin);
    auto slots_u_kin = dacpp::mpi::build_item_slots(item_range, pattern_u_kin, pack_u_kin);
    std::vector<double> local_u_kin(pack_u_kin.globals.size());
    int recv_count_u_kin = 0;
    std::vector<int> sendcounts_u_kin;
    std::vector<int> displs_u_kin;
    std::vector<double> sendbuf_u_kin;
    if (mpi_rank == 0) {
        sendcounts_u_kin.resize(mpi_size);
        displs_u_kin.resize(mpi_size);
        int current_displ = 0;
        std::vector<double> global_u_kin;
        u_kin.tensor2Array(global_u_kin);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_u_kin);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_u_kin, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_u_kin[r] = r_count;
            displs_u_kin[r] = current_displ;
            current_displ += r_count;
            sendbuf_u_kin.insert(sendbuf_u_kin.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_u_kin.data() : nullptr, 1, MPI_INT, &recv_count_u_kin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_u_kin.resize(recv_count_u_kin);
    std::vector<int> sendcounts_bytes_u_kin = sendcounts_u_kin;
    std::vector<int> displs_bytes_u_kin = displs_u_kin;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_u_kin.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_u_kin.data() : nullptr, mpi_rank == 0 ? displs_bytes_u_kin.data() : nullptr, MPI_DOUBLE, local_u_kin.data(), recv_count_u_kin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int u_kin_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_u_kin));
    auto pack_u_kout = dacpp::mpi::build_output_pack_map(item_range, pattern_u_kout);
    auto slots_u_kout = dacpp::mpi::build_item_slots(item_range, pattern_u_kout, pack_u_kout);
    std::vector<double> local_u_kout(pack_u_kout.globals.size());
    const int u_kout_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_u_kout));
    auto pack_r = dacpp::mpi::build_input_pack_map(item_range, pattern_r);
    auto slots_r = dacpp::mpi::build_item_slots(item_range, pattern_r, pack_r);
    std::vector<double> local_r(pack_r.globals.size());
    int recv_count_r = 0;
    std::vector<int> sendcounts_r;
    std::vector<int> displs_r;
    std::vector<double> sendbuf_r;
    if (mpi_rank == 0) {
        sendcounts_r.resize(mpi_size);
        displs_r.resize(mpi_size);
        int current_displ = 0;
        std::vector<double> global_r;
        r.tensor2Array(global_r);
        for (int r = 0; r < mpi_size; ++r) {
            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);
            auto r_pack = dacpp::mpi::build_input_pack_map(r_range, pattern_r);
            auto r_values = dacpp::mpi::pack_values_by_globals(global_r, r_pack.globals);
            int r_count = static_cast<int>(r_values.size());
            sendcounts_r[r] = r_count;
            displs_r[r] = current_displ;
            current_displ += r_count;
            sendbuf_r.insert(sendbuf_r.end(), r_values.begin(), r_values.end());
        }
    }
    MPI_Scatter(mpi_rank == 0 ? sendcounts_r.data() : nullptr, 1, MPI_INT, &recv_count_r, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_r.resize(recv_count_r);
    std::vector<int> sendcounts_bytes_r = sendcounts_r;
    std::vector<int> displs_bytes_r = displs_r;
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_r.data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_r.data() : nullptr, mpi_rank == 0 ? displs_bytes_r.data() : nullptr, MPI_DOUBLE, local_r.data(), recv_count_r, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int r_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_r));
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_u_kin(local_u_kin.data(), sycl::range<1>(local_u_kin.size()));
            sycl::buffer<int32_t, 1> slots_buffer_u_kin(slots_u_kin.data(), sycl::range<1>(slots_u_kin.size()));
            sycl::buffer<double, 1> buffer_u_kout(local_u_kout.data(), sycl::range<1>(local_u_kout.size()));
            sycl::buffer<int32_t, 1> slots_buffer_u_kout(slots_u_kout.data(), sycl::range<1>(slots_u_kout.size()));
            sycl::buffer<double, 1> buffer_r(local_r.data(), sycl::range<1>(local_r.size()));
            sycl::buffer<int32_t, 1> slots_buffer_r(slots_r.data(), sycl::range<1>(slots_r.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_u_kin = buffer_u_kin.get_access<sycl::access::mode::read>(h);
                auto slots_acc_u_kin = slots_buffer_u_kin.get_access<sycl::access::mode::read>(h);
                auto acc_u_kout = buffer_u_kout.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_u_kout = slots_buffer_u_kout.get_access<sycl::access::mode::read>(h);
                auto acc_r = buffer_r.get_access<sycl::access::mode::read>(h);
                auto slots_acc_r = slots_buffer_r.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_u_kin = acc_u_kin.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_u_kin = slots_acc_u_kin.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_u_kin{data_u_kin, slots_u_kin, item_linear * u_kin_partition_size};
                    auto* data_u_kout = acc_u_kout.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_u_kout = slots_acc_u_kout.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_u_kout{data_u_kout, slots_u_kout, item_linear * u_kout_partition_size};
                    auto* data_r = acc_r.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_r = slots_acc_r.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_r{data_r, slots_r, item_linear * r_partition_size};
                    pde_mpi_local(view_u_kin, view_u_kout, view_r);
                });
            });
            q.wait();
        }
    }
    auto writeback_u_kout = dacpp::mpi::build_writeback_values(local_u_kout, pack_u_kout);
    const auto& writeback_globals_u_kout = pack_u_kout.writeback_globals.empty() ? pack_u_kout.globals : pack_u_kout.writeback_globals;
    std::vector<double> synced_u_kout;
    int send_count_u_kout = static_cast<int>(writeback_globals_u_kout.size());
    std::vector<int> recvcounts_u_kout;
    std::vector<int> recvdispls_u_kout;
    std::vector<int64_t> global_recv_globals_u_kout;
    std::vector<double> global_recv_values_u_kout;
    if (mpi_rank == 0) {
        recvcounts_u_kout.resize(mpi_size);
        recvdispls_u_kout.resize(mpi_size);
    }
    MPI_Gather(&send_count_u_kout, 1, MPI_INT, mpi_rank == 0 ? recvcounts_u_kout.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int current_displ = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvdispls_u_kout[r] = current_displ;
            current_displ += recvcounts_u_kout[r];
        }
        global_recv_globals_u_kout.resize(current_displ);
        global_recv_values_u_kout.resize(current_displ);
    }
    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_u_kout.data()), send_count_u_kout, MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_u_kout.data() : nullptr, mpi_rank == 0 ? recvcounts_u_kout.data() : nullptr, mpi_rank == 0 ? recvdispls_u_kout.data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> recvcounts_bytes_u_kout = recvcounts_u_kout;
    std::vector<int> recvdispls_bytes_u_kout = recvdispls_u_kout;
    MPI_Gatherv(writeback_u_kout.data(), send_count_u_kout, MPI_DOUBLE, mpi_rank == 0 ? global_recv_values_u_kout.data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_u_kout.data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_u_kout.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::vector<double> global_out_u_kout;
        u_kout.tensor2Array(global_out_u_kout);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_u_kout, global_recv_globals_u_kout, global_out_u_kout);
        u_kout.array2Tensor(global_out_u_kout);
        synced_u_kout = global_out_u_kout;
    }
    int synced_count_u_kout = 0;
    if (mpi_rank == 0) {
        synced_count_u_kout = static_cast<int>(synced_u_kout.size());
    }
    MPI_Bcast(&synced_count_u_kout, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_u_kout.resize(synced_count_u_kout);
    }
    if (synced_count_u_kout > 0) {
        MPI_Bcast(synced_u_kout.data(), synced_count_u_kout, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        u_kout.array2Tensor(synced_u_kout);
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

    int n = 100; //时间域n等分
    int m = 5; //空间域m等分
    double r = 0.25;
    double a = 1.0;
    double h = 1.0 / m; //空间步长
    double tau = 1.0 / n; //时间步长
    double *x,*t,**u;
    
    //r=a*tau/(h*h);  //网比
    //printf("r=%.4f.\n",r);
    
    x = (double*)malloc(sizeof(double)*(m+1));
    for (int i=0;i<=m;i++) {
        x[i]=i*h;
    }
    t = (double*)malloc(sizeof(double)*(n+1));
    for (int i = 0; i <= n; i++) {
        t[i]=i*tau;
    }
    u = (double**)malloc(sizeof(double*)*(m+1));
    for (int i=0;i<=m;i++) {
        u[i]=(double*)malloc(sizeof(double)*(n+1));
    }
    for (int i = 0; i <= m; i++)
        u[i][0]=phi(x[i]);
    for (int i = 1; i <= n; i++) {
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }
    
    // Flatten the 2D u array into a 1D vector for Tensor creation
    std::vector<double> u_flat;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            u_flat.push_back(static_cast<double>(u[i][j]));  // Cast if needed
        }
    }

    dacpp::Matrix<double> u_tensor({m+1, n+1}, u_flat);
    for (int k = 0; k <= n-1; k++) {
        dacpp::Vector<double> u_kout = u_tensor[{1,m}][k+1];
        std::vector<double> r_data;
        r_data.push_back(r);
        dacpp::Vector<double> r(r_data);
        dacpp::Vector<double> u_kin = u_tensor[{}][k];
        PDE_pde(u_kin, u_kout, r);
        
        //计算完毕后，替换第1到4个点
        for (int i = 1; i <= m-1; i++) {
            u_tensor[i][k+1] = u_kout[i-1];
        }

    }

    // 每个位置需要下，左下，右下，三个位置的元素，串行中从下往上，从左往右遍历计算
    // 那么每一行的元素计算是互不相关的，可以并行执行，所有的行从下往上串行执行
    u_tensor[1].print();
    // double* data = new double[6 * 101];
    // u_tensor.tensor2Array(data);

    // // 将一维数组转换为二维 vector
    // std::vector<std::vector<double>> vec2D;
    // vec2D.resize(6, std::vector<double>(101));

    // // 将一维数组的数据填充到二维数组中
    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 101; ++j) {
    //         vec2D[i][j] = data[i * 101 + j];
    //     }
    // }


    // int j = int(0.2 / tau);
    // int number = int(0.4 / h);
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f, err=%.3e.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]),std::fabs(vec2D[number][k]-exact(x[number],t[k])));
    // }
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]));
    // }


    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}