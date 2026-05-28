#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

static inline bool __dacpp_mpi_is_root_rank();
namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

const int NX = 512;
const int NY = 512;
const int PATHS_PER_OPTION = 4096;
const int INNER_REPEATS = 2;

const double S0 = 100.0;
const double STRIKE = 100.0;
const double RATE = 0.05;
const double VOLATILITY = 0.2;
const double MATURITY = 1.0;
const double DISCOUNT = 0.95122942450071400909;
const double TWO_PI = 6.2831853071795864769;

static inline double __dacpp_fast_exp_device(double x) {
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

static inline double __dacpp_fast_log_device(double x) {
    if (x <= 0.0) {
        return -745.0;
    }

    union DoubleBits {
        double d;
        unsigned long long u;
    } value;

    value.d = x;
    const int exponent = static_cast<int>((value.u >> 52) & 0x7ffULL) - 1023;
    value.u = (value.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    const double mantissa = value.d;

    const double y = (mantissa - 1.0) / (mantissa + 1.0);
    const double y2 = y * y;
    double term = y;
    double series = term;
    term *= y2;
    series += term / 3.0;
    term *= y2;
    series += term / 5.0;
    term *= y2;
    series += term / 7.0;
    term *= y2;
    series += term / 9.0;
    term *= y2;
    series += term / 11.0;

    return static_cast<double>(exponent) * 0.69314718055994530942 +
           2.0 * series;
}

unsigned int lcgNext(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

double uniform01(unsigned int value) {
    return ((double)(value & 0x00FFFFFFu) + 1.0) / 16777217.0;
}





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
inline void print_e2e_summary(const char* label, int mpi_rank, int mpi_size, double total_time) {
    double total_max = 0.0;
    double total_sum = 0.0;
    MPI_Reduce(&total_time, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::printf("[MPI TEST] %s | ranks=%d | total_max=%.6f s | total_avg=%.6f s\n",
                    label,
                    mpi_size,
                    total_max,
                    total_sum / static_cast<double>(mpi_size));
    }
}
}  // namespace __dacpp_test

template <typename __dacpp_view_t0, typename __dacpp_view_t1>
__attribute__((always_inline)) inline void monteCarloOption_mpi_local(__dacpp_view_t0 seed, __dacpp_view_t1 price) {
    unsigned int state = (unsigned int)seed[0] + 1U;
    double payoff_sum = 0.;
    for (int repeat = 0; repeat < INNER_REPEATS; repeat++) {
        for (int path = 0; path < PATHS_PER_OPTION; path++) {
            state = lcgNext(state);
            double u1 = uniform01(state);
            state = lcgNext(state);
            double u2 = uniform01(state);
            double radius = std::sqrt(-2. * __dacpp_fast_log_device(u1));
            double angle = TWO_PI * u2;
            double normal = radius * std::cos(angle);
            double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
            double diffusion = VOLATILITY * normal;
            double terminal = S0 * __dacpp_fast_exp_device(drift + diffusion);
            double payoff = terminal - STRIKE;
            if (payoff < 0.) {
                payoff = 0.;
            }
            payoff_sum += payoff;
        }
    }
    price[0] = DISCOUNT * payoff_sum / (PATHS_PER_OPTION * INNER_REPEATS);
}


void __dacpp_mpi_or_MONTE_CARLO_OPTION_monteCarloOption_0(const dacpp::Matrix<int> & __or_arg0, dacpp::Matrix<double> & __or_arg1) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::SegmentedProfile dacpp_profile;
    auto& q = dacpp::mpi::operator_resident::default_queue();
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_rows = __or_arg1.getShape(0);
    const int64_t __or_cols = __or_arg1.getShape(1);
    const int64_t __or_total_items = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_rows, __or_cols, "[DACPP][MPI][OR] row-block total item count overflow");
    const auto __or_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, mpi_rank, mpi_size);
    const int64_t __or_local_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_row_range.count, __or_cols, "[DACPP][MPI][OR] row-block local item count overflow");
    std::vector<int> __or_row_counts;
    std::vector<int> __or_row_displs;
    dacpp::mpi::operator_resident::counts_displs_1d(__or_rows, mpi_size, __or_row_counts, __or_row_displs);
    std::vector<int> __or_counts(mpi_size);
    std::vector<int> __or_displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        __or_counts[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_counts[r]), __or_cols, "[DACPP][MPI][OR] row-block scatter count overflow"), "[DACPP][MPI][OR] row-block scatter count exceeds MPI int range");
        __or_displs[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_displs[r]), __or_cols, "[DACPP][MPI][OR] row-block scatter displacement overflow"), "[DACPP][MPI][OR] row-block scatter displacement exceeds MPI int range");
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    std::vector<int> __or_local_seed(static_cast<std::size_t>(__or_local_item_count));
    std::vector<int> __or_global_seed;
    int* __or_rowblock_scatter_src_seed = nullptr;
    int __or_rowblock_direct_seed = 0;
    if (mpi_rank == 0) {
        const int64_t __or_rb_offset_seed = __or_arg0.getOffset();
        int64_t __or_rb_rows_seed = -1;
        int64_t __or_rb_cols_seed = -1;
        if (__or_arg0.getDim() == 2) {
            __or_rb_rows_seed = __or_arg0.getShape(0);
            __or_rb_cols_seed = __or_arg0.getShape(1);
        }
        const int64_t __or_rb_size_seed = __or_arg0.getSize();
        const int64_t __or_rb_last_begin_seed = mpi_size > 0 ? static_cast<int64_t>(__or_row_displs[mpi_size - 1]) : 0;
        const int64_t __or_rb_last_count_seed = mpi_size > 0 ? static_cast<int64_t>(__or_row_counts[mpi_size - 1]) : 0;
        bool __or_rb_ranges_ok_seed = mpi_size >= 0;
        for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {
            const int64_t __or_r_begin = static_cast<int64_t>(__or_row_displs[__or_r]);
            const int64_t __or_r_count = static_cast<int64_t>(__or_row_counts[__or_r]);
            if (__or_r_begin < 0 || __or_r_count < 0 || __or_r_begin > __or_rows || __or_r_count > __or_rows - __or_r_begin) {
                __or_rb_ranges_ok_seed = false;
                break;
            }
        }
        __or_rowblock_direct_seed = (__or_arg0.getDim() == 2 && __or_rb_offset_seed >= 0 && __or_rb_rows_seed == __or_rows && __or_rb_cols_seed == __or_cols && __or_rb_rows_seed >= 0 && __or_rb_cols_seed >= 0 && __or_arg0.getStride(1) == 1 && __or_arg0.getStride(0) == __or_cols && __or_rb_size_seed >= 0 && __or_rb_offset_seed <= __or_rb_size_seed && __or_total_items <= __or_rb_size_seed - __or_rb_offset_seed && __or_rb_last_begin_seed >= 0 && __or_rb_last_count_seed >= 0 && __or_rb_last_begin_seed + __or_rb_last_count_seed <= __or_rows && __or_rb_ranges_ok_seed) ? 1 : 0;
        if (__or_rowblock_direct_seed) {
            __or_rowblock_scatter_src_seed = __or_arg0.getDataPtr().get() + __or_rb_offset_seed;
        }
    }
    MPI_Bcast(&__or_rowblock_direct_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        if (!__or_rowblock_direct_seed) {
        auto dacpp_profile_pack_start_seed = dacpp::mpi::profileSegmentStart();
        __or_arg0.tensor2Array(__or_global_seed);
        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Pack, dacpp_profile_pack_start_seed);
        }
    }
    if (__or_rowblock_direct_seed) {
        auto dacpp_profile_scatter_start_seed = dacpp::mpi::profileSegmentStart();
        MPI_Scatterv(mpi_rank == 0 ? __or_rowblock_scatter_src_seed : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, MPI_INT, __or_local_seed.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), "[DACPP][MPI][OR] row-block direct scatter count exceeds MPI int range"), MPI_INT, 0, MPI_COMM_WORLD);
        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_seed);
    } else {
        if (mpi_rank != 0) {
            __or_global_seed.resize(static_cast<std::size_t>(__or_total_items));
        }
        auto dacpp_profile_bcast_start_seed = dacpp::mpi::profileSegmentStart();
        MPI_Bcast(__or_global_seed.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_total_items), "[DACPP][MPI][OR] row-block broadcast count exceeds MPI int range"), MPI_INT, 0, MPI_COMM_WORLD);
        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_seed);
        const int64_t __or_rowblock_offset_seed = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_row_range.begin, __or_cols, "[DACPP][MPI][OR] row-block local offset overflow");
        const int64_t __or_rowblock_local_bytes_seed = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(int)), "[DACPP][MPI][OR] row-block local byte copy size overflow");
        std::memcpy(__or_local_seed.data(), __or_global_seed.data() + __or_rowblock_offset_seed, static_cast<std::size_t>(__or_rowblock_local_bytes_seed));
    }
    std::vector<double> __or_local_price(static_cast<std::size_t>(__or_local_item_count));
    std::fill(__or_local_price.begin(), __or_local_price.end(), double{});
    // Output-direct no-read fast path for price initializes local output and skips root pack/scatter.
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<int, 1> __or_buffer_seed(__or_local_seed.data(), sycl::range<1>(__or_local_seed.size()));
            sycl::buffer<double, 1> __or_buffer_price(__or_local_price.data(), sycl::range<1>(__or_local_price.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_seed = __or_buffer_seed.get_access<sycl::access::mode::read>(h);
                auto __or_acc_price = __or_buffer_price.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class __dacpp_sycl_kernel_monteCarloOption_translated_rowblock_0>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_seed = __or_acc_seed.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const int> view_seed{__or_data_seed, item_linear};
                    auto* __or_data_price = __or_acc_price.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<double> view_price{__or_data_price, item_linear};
                    monteCarloOption_mpi_local(view_seed, view_price);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    auto dacpp_profile_bounded_sync_start_price = dacpp::mpi::profileSegmentStart();
    {
        std::vector<double> __or_bounded_values_price(static_cast<std::size_t>(1), double{});
        {
            const int64_t __or_target_row = 256;
            const int64_t __or_target_col = 256;
            if ((__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)) {
                const int64_t __or_local_linear = ((256 - __or_row_range.begin) * __or_cols + 256);
                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= __or_local_price.size()) {
                    if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read local index out of range\n");
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                __or_bounded_values_price[static_cast<std::size_t>(0)] = __or_local_price[static_cast<std::size_t>(__or_local_linear)];
            }
        }
        {
            const int64_t __or_target_row = 256;
            const int64_t __or_target_col = 256;
            int __or_owner_rank = -1;
            if (__or_target_col >= 0 && __or_target_col < __or_cols) {
                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {
                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, __or_owner_candidate, mpi_size);
                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {
                        __or_owner_rank = __or_owner_candidate;
                        break;
                    }
                }
            }
            if (__or_owner_rank < 0) {
                if (mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] bounded indexed read has no owner\n");
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {
                MPI_Send(&__or_bounded_values_price[static_cast<std::size_t>(0)], 1, MPI_DOUBLE, 0, 4701, MPI_COMM_WORLD);
            } else if (mpi_rank == 0 && __or_owner_rank != 0) {
                MPI_Recv(&__or_bounded_values_price[static_cast<std::size_t>(0)], 1, MPI_DOUBLE, __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (mpi_rank == 0) {
            __or_arg1.reviseValue(__or_bounded_values_price[static_cast<std::size_t>(0)], std::vector<int>{256, 256});
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::FinalSync, dacpp_profile_bounded_sync_start_price);
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_MONTE_CARLO_OPTION_monteCarloOption_0", dacpp_profile, MPI_COMM_WORLD);
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

    std::vector<int> seeds(NX * NY, 0);
    std::vector<double> prices(NX * NY, 0.0);

    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            seeds[i * NX + j] = i * NX + j;
        }
    }

    dacpp::Matrix<int> seeds_tensor({NY, NX}, seeds);
    dacpp::Matrix<double> prices_tensor({NY, NX}, prices);

    MPI_Barrier(MPI_COMM_WORLD);
    const double __dacpp_test_total_start = MPI_Wtime();
    __dacpp_mpi_or_MONTE_CARLO_OPTION_monteCarloOption_0(seeds_tensor, prices_tensor);
    MPI_Barrier(MPI_COMM_WORLD);
    const double __dacpp_test_total_time = MPI_Wtime() - __dacpp_test_total_start;

    if (__dacpp_mpi_is_root_rank()) {
        std::cout << prices_tensor[NY / 2][NX / 2] << std::endl;
    }

    __dacpp_test::print_e2e_summary("monteCarloOption.dac_sycl_buffer.sunway.cpp", mpi_rank, mpi_size, __dacpp_test_total_time);
    
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}
