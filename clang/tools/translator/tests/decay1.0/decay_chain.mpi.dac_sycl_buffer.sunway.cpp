#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"
static inline bool __dacpp_mpi_is_root_rank();
namespace dacpp {
    typedef std::vector<std::any> list;
}


#ifndef DECAY_DT
#define DECAY_DT 0.1
#endif

#ifndef DECAY_TOTAL_TIME
#define DECAY_TOTAL_TIME 100.0
#endif

#ifndef DECAY_NUM_ISOTOPES
#define DECAY_NUM_ISOTOPES 1024576
#endif

const double dt = DECAY_DT;       // 时间步长
const double T = DECAY_TOTAL_TIME;       // 总时间
const size_t numIsotopes = DECAY_NUM_ISOTOPES; // 设定大量同位素（例如，10000个）




// 计算每种同位素在时间 t 的数量
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"
#include <chrono>
#include <utility>

static inline bool __dacpp_mpi_is_root_rank() {
    int __dacpp_mpi_initialized = 0;
    MPI_Initialized(&__dacpp_mpi_initialized);
    if (!__dacpp_mpi_initialized) {
        return true;
    }
    int __dacpp_mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &__dacpp_mpi_rank);
    return __dacpp_mpi_rank == 0;
}

using namespace sycl;

namespace __dacpp_test {
inline void print_e2e_summary(const char* label, int mpi_rank, int mpi_size, double e2e_time) {
    double e2e_max = 0.0;
    double e2e_sum = 0.0;
    MPI_Reduce(&e2e_time, &e2e_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&e2e_time, &e2e_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n",
                    label,
                    mpi_size,
                    e2e_max,
                    e2e_sum / static_cast<double>(mpi_size));
    }
}
}  // namespace __dacpp_test

static inline double __dacpp_decay_fast_exp_device(double x) {
    if (x < -50.0) {
        return 0.0;
    }
    if (x > 50.0) {
        x = 50.0;
    }

    const double inv_ln2 = 1.44269504088896340736;
    const double ln2 = 0.69314718055994530942;
    const int n = static_cast<int>(x * inv_ln2 + (x >= 0.0 ? 0.5 : -0.5));
    const double r = x - static_cast<double>(n) * ln2;
    const double r2 = r * r;
    double y = 1.0 + r + 0.5 * r2 + (r2 * r) / 6.0 +
               (r2 * r2) / 24.0 + (r2 * r2 * r) / 120.0 +
               (r2 * r2 * r2) / 720.0;

    if (n > 0) {
        for (int i = 0; i < n; ++i) {
            y *= 2.0;
        }
    } else {
        for (int i = 0; i < -n; ++i) {
            y *= 0.5;
        }
    }
    return y;
}

template <typename __dacpp_view_t0, typename __dacpp_view_t1, typename __dacpp_view_t2, typename __dacpp_view_t3>
__attribute__((always_inline)) inline void decay_mpi_local(__dacpp_view_t0 N0s, __dacpp_view_t1 lambdas, __dacpp_view_t2 local_A, __dacpp_view_t3 t) {
    local_A[0] = N0s[0] * __dacpp_decay_fast_exp_device(-lambdas[0] * t[0]);
}


struct __dacpp_mpi_or_DECAY_decay_0_ctx {
    int mpi_rank = 0;
    int mpi_size = 1;
    int64_t __or_total_items = 0;
    int64_t __or_local_item_count = 0;
    dacpp::mpi::operator_resident::RankRange1D __or_range{};
    std::vector<int> __or_counts;
    std::vector<int> __or_displs;
    dacpp::mpi::SegmentedProfile __or_profile;
    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();
    std::vector<double> __or_local_N0s;
    std::vector<double> __or_local_lambdas;
    std::vector<double> __or_local_local_A;
    double __or_scalar_t{};
    std::vector<double> __or_local_t;
};
void __dacpp_mpi_or_DECAY_decay_0_init(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
    ctx.__or_total_items = __or_arg2.getShape(0);
    ctx.__or_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_total_items, ctx.mpi_rank, ctx.mpi_size);
    ctx.__or_local_item_count = ctx.__or_range.count;
    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_total_items, ctx.mpi_size, ctx.__or_counts, ctx.__or_displs);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    auto dacpp_profile_scatter_start_N0s = dacpp::mpi::profileSegmentStart();
    ctx.__or_local_N0s.resize(static_cast<std::size_t>(ctx.__or_local_item_count));
    std::vector<double> __or_global_N0s;
    if (ctx.mpi_rank == 0) {
        __or_arg0.tensor2Array(__or_global_N0s);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_global_N0s.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, ctx.__or_local_N0s.data(), static_cast<int>(ctx.__or_local_item_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_N0s);
    auto dacpp_profile_scatter_start_lambdas = dacpp::mpi::profileSegmentStart();
    ctx.__or_local_lambdas.resize(static_cast<std::size_t>(ctx.__or_local_item_count));
    std::vector<double> __or_global_lambdas;
    if (ctx.mpi_rank == 0) {
        __or_arg1.tensor2Array(__or_global_lambdas);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_global_lambdas.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, ctx.__or_local_lambdas.data(), static_cast<int>(ctx.__or_local_item_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_lambdas);
    ctx.__or_local_local_A.assign(static_cast<std::size_t>(ctx.__or_local_item_count), double{});
}
void __dacpp_mpi_or_DECAY_decay_0_run(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    if (__or_arg3.getSize() != 1) {
        if (ctx.mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR][P4.5] scalar parameter t expected size 1\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    ctx.__or_scalar_t = double{};
    // P4.5 loop-lowered direct scalar reader: all ranks execute the host scalar update, so refresh from the local replicated tensor without per-iteration MPI_Bcast.
    std::vector<double> __or_scalar_vec_t;
    __or_arg3.tensor2Array(__or_scalar_vec_t);
    if (!__or_scalar_vec_t.empty()) ctx.__or_scalar_t = __or_scalar_vec_t[0];
    ctx.__or_local_t.assign(1, ctx.__or_scalar_t);
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_local_item_count = ctx.__or_local_item_count;
    auto& q = ctx.q;
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<double, 1> __or_buffer_N0s(ctx.__or_local_N0s.data(), sycl::range<1>(ctx.__or_local_N0s.size()));
            sycl::buffer<double, 1> __or_buffer_lambdas(ctx.__or_local_lambdas.data(), sycl::range<1>(ctx.__or_local_lambdas.size()));
            sycl::buffer<double, 1> __or_buffer_local_A(ctx.__or_local_local_A.data(), sycl::range<1>(ctx.__or_local_local_A.size()));
            sycl::buffer<double, 1> __or_buffer_t(ctx.__or_local_t.data(), sycl::range<1>(ctx.__or_local_t.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_N0s = __or_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto __or_acc_lambdas = __or_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto __or_acc_local_A = __or_buffer_local_A.get_access<sycl::access::mode::read_write>(h);
                auto __or_acc_t = __or_buffer_t.get_access<sycl::access::mode::read>(h);
                h.parallel_for<class __dacpp_sycl_kernel_decay_translated_once_0>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_N0s = __or_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_N0s{__or_data_N0s, item_linear};
                    auto* __or_data_lambdas = __or_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_lambdas{__or_data_lambdas, item_linear};
                    auto* __or_data_local_A = __or_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<double> view_local_A{__or_data_local_A, item_linear};
                    auto* __or_data_t = __or_acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_t{__or_data_t, 0};
                    decay_mpi_local(view_N0s, view_lambdas, view_local_A, view_t);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    // No downstream resident reader for local_A; host materialization below preserves visibility.
    const int64_t __or_selective_materialize_row_local_A = static_cast<int64_t>(10*ctx.__or_scalar_t);
    (void)__or_selective_materialize_row_local_A;
}
bool __dacpp_mpi_or_DECAY_decay_0_run_loop(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    if (__or_arg3.getSize() != 1) {
        if (ctx.mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR][P4.5] scalar parameter t expected size 1\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    ctx.__or_scalar_t = double{};
    // P4.5 loop-lowered direct scalar reader: all ranks execute the host scalar update, so refresh from the local replicated tensor without per-iteration MPI_Bcast.
    std::vector<double> __or_scalar_vec_t;
    __or_arg3.tensor2Array(__or_scalar_vec_t);
    if (!__or_scalar_vec_t.empty()) ctx.__or_scalar_t = __or_scalar_vec_t[0];
    ctx.__or_local_t.assign(1, ctx.__or_scalar_t);
    // P4.5 device time loop: each work-item replays scalar evolution locally and materializes only the selected host row.
    const double __or_initial_scalar_t = ctx.__or_scalar_t;
    const int64_t __or_local_item_count = ctx.__or_local_item_count;
    auto& q = ctx.q;
    const auto __or_selective_target_row = static_cast<int64_t>(1);
    bool __or_selected_row_reached = false;
    {
        double __or_time_scalar_t = __or_initial_scalar_t;
        while (__or_time_scalar_t<=T) {
            const int64_t __or_selective_row = static_cast<int64_t>(10*__or_time_scalar_t);
            if (__or_selective_row == __or_selective_target_row) {
                __or_selected_row_reached = true;
                break;
            }
            __or_time_scalar_t+=dt;
        }
    }
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<double, 1> __or_buffer_N0s(ctx.__or_local_N0s.data(), sycl::range<1>(ctx.__or_local_N0s.size()));
            sycl::buffer<double, 1> __or_buffer_lambdas(ctx.__or_local_lambdas.data(), sycl::range<1>(ctx.__or_local_lambdas.size()));
            sycl::buffer<double, 1> __or_buffer_local_A(ctx.__or_local_local_A.data(), sycl::range<1>(ctx.__or_local_local_A.size()));
            double __or_kernel_initial_scalar_t = __or_initial_scalar_t;
            double __or_kernel_T = T;
            double __or_kernel_dt = dt;
            int64_t __or_kernel_selective_target_row = __or_selective_target_row;
            int __or_kernel_selected_row_reached = __or_selected_row_reached ? 1 : 0;
            q.submit([&](sycl::handler& h) {
                auto __or_acc_N0s = __or_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto __or_acc_lambdas = __or_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto __or_acc_local_A = __or_buffer_local_A.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class __dacpp_sycl_kernel_decay_translated_loop_0>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    int64_t __or_selective_target = __or_kernel_selective_target_row;
                    double __or_time_scalar_t = __or_kernel_initial_scalar_t;
                    double __or_selected_value = double{};
                    while (__or_time_scalar_t<=__or_kernel_T) {
                        const int64_t __or_selective_row = static_cast<int64_t>(10*__or_time_scalar_t);
                        auto* __or_data_N0s = __or_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                        dacpp::mpi::ContiguousView1D<const double> view_N0s{__or_data_N0s, item_linear};
                        auto* __or_data_lambdas = __or_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                        dacpp::mpi::ContiguousView1D<const double> view_lambdas{__or_data_lambdas, item_linear};
                        auto* __or_data_local_A = __or_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                        dacpp::mpi::ContiguousView1D<double> view_local_A{__or_data_local_A, item_linear};
                        double __or_scalar_storage_t[1] = {__or_time_scalar_t};
                        dacpp::mpi::ContiguousView1D<double> view_t{__or_scalar_storage_t, 0};
                        decay_mpi_local(view_N0s, view_lambdas, view_local_A, view_t);
                        if (__or_selective_row == __or_selective_target) {
                            auto* __or_output_data = __or_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                            __or_selected_value = __or_output_data[item_linear];
                        }
                        __or_time_scalar_t+=__or_kernel_dt;
                    }
                    if (__or_kernel_selected_row_reached) {
                        auto* __or_output_data = __or_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                        __or_output_data[item_linear] = __or_selected_value;
                    }
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    auto& __or_resident_out_local_A = dacpp::mpi::operator_resident::ensure_resident<double>(__or_arg2, ctx.__or_local_local_A.size());
    __or_resident_out_local_A = ctx.__or_local_local_A;
    return __or_selected_row_reached;
}
void __dacpp_mpi_or_DECAY_decay_0_materialize(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    (void)ctx;
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_DECAY_decay_0_materialize", ctx.__or_profile, MPI_COMM_WORLD);
}
void __dacpp_mpi_or_DECAY_decay_0(const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    __dacpp_mpi_or_DECAY_decay_0_ctx ctx;
    __dacpp_mpi_or_DECAY_decay_0_init(ctx, __or_arg0, __or_arg1, __or_arg2, __or_arg3);
    __dacpp_mpi_or_DECAY_decay_0_run(ctx, __or_arg0, __or_arg1, __or_arg2, __or_arg3);
}

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    

        __dacpp_mpi_or_DECAY_decay_0_ctx __dacpp_mpi_or_ctx_0;
    __dacpp_mpi_or_DECAY_decay_0_init(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
    __dacpp_mpi_or_DECAY_decay_0_run_loop(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
    __dacpp_mpi_or_DECAY_decay_0_materialize(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
while(false){  
        ((void)0);
        t_tensor[0] += dt;
    }
}

int main() {
    const auto __dacpp_test_e2e_start = std::chrono::steady_clock::now();
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

    

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }


    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    const auto __dacpp_test_e2e_end = std::chrono::steady_clock::now();
    const double __dacpp_test_e2e_time =
        std::chrono::duration<double>(__dacpp_test_e2e_end - __dacpp_test_e2e_start)
            .count();
    __dacpp_test::print_e2e_summary("decay_chain.mpi.dac_sycl_buffer.sunway.cpp",
                                    mpi_rank,
                                    mpi_size,
                                    __dacpp_test_e2e_time);

    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
