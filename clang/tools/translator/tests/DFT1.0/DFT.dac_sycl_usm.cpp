#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"
//对于for循环里面的tensor翻译出来需要array2Tensor
namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;  // 复数类型别名
const int N = 8;







// 离散傅里叶变换（DFT）
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"

using namespace sycl;

inline void dft_mpi_local(dacpp::mpi::View1D<const std::complex<double>> input, dacpp::mpi::View1D<std::complex<double>> output, dacpp::mpi::View1D<const int> vec) {
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2. * 3.1415926535897931 * vec[0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));
        sum += input[n] * W_n;
    }
    output[0] = sum;
}


void DFT_dft(const dacpp::Vector<std::complex<double> > & input, dacpp::Vector<std::complex<double> > & output, const dacpp::Vector<int> & vec) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    sycl::queue q(sycl::default_selector_v);
    std::vector<int64_t> binding_split_sizes;
    dacpp::mpi::AccessPattern pattern_input;
    pattern_input.param_id = 0;
    pattern_input.name = "input";
    pattern_input.mode = dacpp::mpi::AccessMode::Read;
    pattern_input.data_info.dim = input.getDim();
    for (int dim = 0; dim < input.getDim(); ++dim) pattern_input.data_info.dimLength.push_back(input.getShape(dim));
    pattern_input.partition_shape = dacpp::mpi::init_partition_shape(pattern_input);
    pattern_input.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_input);
    if (binding_split_sizes.size() < pattern_input.bind_split_sizes.size()) binding_split_sizes.resize(pattern_input.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_input.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_input.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_output;
    pattern_output.param_id = 1;
    pattern_output.name = "output";
    pattern_output.mode = dacpp::mpi::AccessMode::Write;
    pattern_output.data_info.dim = output.getDim();
    for (int dim = 0; dim < output.getDim(); ++dim) pattern_output.data_info.dimLength.push_back(output.getShape(dim));
    Dac_Op pattern_output_op_0;
    pattern_output_op_0.setDimId(0);
    pattern_output_op_0.size = 1;
    pattern_output_op_0.stride = 1;
    pattern_output_op_0.SetSplitSize(output.getShape(0));
    pattern_output.param_ops.push_back(pattern_output_op_0);
    pattern_output.bind_set_id.push_back(0);
    pattern_output.bind_offset_expr.push_back("0");
    pattern_output.is_index_op.push_back(true);
    pattern_output.partition_shape = dacpp::mpi::init_partition_shape(pattern_output);
    pattern_output.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_output);
    if (binding_split_sizes.size() < pattern_output.bind_split_sizes.size()) binding_split_sizes.resize(pattern_output.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_output.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_output.bind_split_sizes[bind_i]);
    }
    dacpp::mpi::AccessPattern pattern_vec;
    pattern_vec.param_id = 2;
    pattern_vec.name = "vec";
    pattern_vec.mode = dacpp::mpi::AccessMode::Read;
    pattern_vec.data_info.dim = vec.getDim();
    for (int dim = 0; dim < vec.getDim(); ++dim) pattern_vec.data_info.dimLength.push_back(vec.getShape(dim));
    Dac_Op pattern_vec_op_0;
    pattern_vec_op_0.setDimId(0);
    pattern_vec_op_0.size = 1;
    pattern_vec_op_0.stride = 1;
    pattern_vec_op_0.SetSplitSize(vec.getShape(0));
    pattern_vec.param_ops.push_back(pattern_vec_op_0);
    pattern_vec.bind_set_id.push_back(0);
    pattern_vec.bind_offset_expr.push_back("0");
    pattern_vec.is_index_op.push_back(true);
    pattern_vec.partition_shape = dacpp::mpi::init_partition_shape(pattern_vec);
    pattern_vec.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(pattern_vec);
    if (binding_split_sizes.size() < pattern_vec.bind_split_sizes.size()) binding_split_sizes.resize(pattern_vec.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < pattern_vec.bind_split_sizes.size(); ++bind_i) {
        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], pattern_vec.bind_split_sizes[bind_i]);
    }
    int64_t total_items = 1;
    for (int64_t split_size : binding_split_sizes) total_items *= split_size;
    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);
    const int64_t local_item_count = item_range.size();
    pattern_input.bind_split_sizes = binding_split_sizes;
    pattern_output.bind_split_sizes = binding_split_sizes;
    pattern_vec.bind_split_sizes = binding_split_sizes;
    auto pack_input = dacpp::mpi::build_input_pack_map(item_range, pattern_input);
    auto slots_input = dacpp::mpi::build_item_slots(item_range, pattern_input, pack_input);
    std::vector<std::complex<double>> local_input(pack_input.globals.size());
    if (mpi_rank == 0) {
        std::vector<std::complex<double>> global_input;
        input.tensor2Array(global_input);
        local_input = dacpp::mpi::pack_values_by_globals(global_input, pack_input.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_input);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_input, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1000, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_C_DOUBLE_COMPLEX, peer, 2000, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_input.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_input.data(), recv_count, MPI_C_DOUBLE_COMPLEX, 0, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int input_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_input));
    auto pack_output = dacpp::mpi::build_output_pack_map(item_range, pattern_output);
    auto slots_output = dacpp::mpi::build_item_slots(item_range, pattern_output, pack_output);
    std::vector<std::complex<double>> local_output(pack_output.globals.size());
    const int output_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_output));
    auto pack_vec = dacpp::mpi::build_input_pack_map(item_range, pattern_vec);
    auto slots_vec = dacpp::mpi::build_item_slots(item_range, pattern_vec, pack_vec);
    std::vector<int> local_vec(pack_vec.globals.size());
    if (mpi_rank == 0) {
        std::vector<int> global_vec;
        vec.tensor2Array(global_vec);
        local_vec = dacpp::mpi::pack_values_by_globals(global_vec, pack_vec.globals);
        for (int peer = 1; peer < mpi_size; ++peer) {
            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);
            auto peer_pack = dacpp::mpi::build_input_pack_map(peer_range, pattern_vec);
            auto peer_values = dacpp::mpi::pack_values_by_globals(global_vec, peer_pack.globals);
            int peer_count = static_cast<int>(peer_values.size());
            MPI_Send(&peer_count, 1, MPI_INT, peer, 1002, MPI_COMM_WORLD);
            if (peer_count > 0) {
                MPI_Send(peer_values.data(), peer_count, MPI_INT, peer, 2002, MPI_COMM_WORLD);
            }
        }
    } else {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 1002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_vec.resize(recv_count);
        if (recv_count > 0) {
            MPI_Recv(local_vec.data(), recv_count, MPI_INT, 0, 2002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const int vec_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(pattern_vec));
    if (local_item_count > 0) {
        {
            sycl::buffer<std::complex<double>, 1> buffer_input(local_input.data(), sycl::range<1>(local_input.size()));
            sycl::buffer<int32_t, 1> slots_buffer_input(slots_input.data(), sycl::range<1>(slots_input.size()));
            sycl::buffer<std::complex<double>, 1> buffer_output(local_output.data(), sycl::range<1>(local_output.size()));
            sycl::buffer<int32_t, 1> slots_buffer_output(slots_output.data(), sycl::range<1>(slots_output.size()));
            sycl::buffer<int, 1> buffer_vec(local_vec.data(), sycl::range<1>(local_vec.size()));
            sycl::buffer<int32_t, 1> slots_buffer_vec(slots_vec.data(), sycl::range<1>(slots_vec.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_input = buffer_input.get_access<sycl::access::mode::read>(h);
                auto slots_acc_input = slots_buffer_input.get_access<sycl::access::mode::read>(h);
                auto acc_output = buffer_output.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_output = slots_buffer_output.get_access<sycl::access::mode::read>(h);
                auto acc_vec = buffer_vec.get_access<sycl::access::mode::read>(h);
                auto slots_acc_vec = slots_buffer_vec.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_input = acc_input.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_input = slots_acc_input.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const std::complex<double>> view_input{data_input, slots_input, item_linear * input_partition_size};
                    auto* data_output = acc_output.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_output = slots_acc_output.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<std::complex<double>> view_output{data_output, slots_output, item_linear * output_partition_size};
                    auto* data_vec = acc_vec.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_vec = slots_acc_vec.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const int> view_vec{data_vec, slots_vec, item_linear * vec_partition_size};
                    dft_mpi_local(view_input, view_output, view_vec);
                });
            });
            q.wait();
        }
    }
    auto writeback_output = dacpp::mpi::build_writeback_values(local_output, pack_output);
    const auto& writeback_globals_output = pack_output.writeback_globals.empty() ? pack_output.globals : pack_output.writeback_globals;
    std::vector<std::complex<double>> synced_output;
    if (mpi_rank == 0) {
        std::vector<std::complex<double>> global_out_output;
        output.tensor2Array(global_out_output);
        dacpp::mpi::apply_writeback_by_globals(writeback_output, writeback_globals_output, global_out_output);
        for (int peer = 1; peer < mpi_size; ++peer) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, peer, 3001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_count <= 0) continue;
            std::vector<int64_t> recv_globals(recv_count);
            std::vector<std::complex<double>> recv_values(recv_count);
            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, 4001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_values.data(), recv_count, MPI_C_DOUBLE_COMPLEX, peer, 5001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, global_out_output);
        }
        output.array2Tensor(global_out_output);
        synced_output = global_out_output;
    } else {
        int send_count = static_cast<int>(writeback_globals_output.size());
        MPI_Send(&send_count, 1, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        if (send_count > 0) {
            MPI_Send(const_cast<int64_t*>(writeback_globals_output.data()), send_count, MPI_LONG_LONG, 0, 4001, MPI_COMM_WORLD);
            MPI_Send(writeback_output.data(), send_count, MPI_C_DOUBLE_COMPLEX, 0, 5001, MPI_COMM_WORLD);
        }
    }
    int synced_count_output = 0;
    if (mpi_rank == 0) {
        synced_count_output = static_cast<int>(synced_output.size());
    }
    MPI_Bcast(&synced_count_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        synced_output.resize(synced_count_output);
    }
    if (synced_count_output > 0) {
        MPI_Bcast(synced_output.data(), synced_count_output, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank != 0) {
        output.array2Tensor(synced_output);
    }
}

void dftfunc(const vector<std::complex<double>>& input, vector<std::complex<double>>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // 使用 for 循环初始化 vector，元素从 1 到 N
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // 赋值为 1 到 N
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT_dft(input_tensor, output_tensor, vec_tensor);
    output_tensor.print();
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

    // 定义一个输入信号（长度为8的复数序列）

    vector<std::complex<double>> input(N);
    
    // 初始化输入数据（可以是任何时间域信号）
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // 以复数形式填充数据，这里只是简单的填充数据
    }

    // 输出原始数据
    //std::cout << "原始数据（时间域）:" << std::endl;
    // for (const auto& val : input) {
    //     //std::cout << val << std::endl;
    // }

    // 计算离散傅里叶变换
    vector<Complex> output(N);
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // 使用 for 循环初始化 vector，元素从 1 到 N
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // 赋值为 1 到 N
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT_dft(input_tensor, output_tensor, vec_tensor);
    output_tensor.print();

    // // 输出傅里叶变换后的数据（频域）
    // cout << "\n傅里叶变换后的数据（频域）:" << endl;
    // for (const auto& val : output) {
    //     cout << val << endl;
    // }

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
return 0;

    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
}
