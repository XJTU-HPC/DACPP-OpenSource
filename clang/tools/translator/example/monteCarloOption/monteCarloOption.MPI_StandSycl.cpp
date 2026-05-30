#include <sycl/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;

const int NX = 4096;
const int NY = 4096;
const int PATHS_PER_OPTION = 8192;
const int INNER_REPEATS = 2;

const double S0 = 100.0;
const double STRIKE = 100.0;
const double RATE = 0.05;
const double VOLATILITY = 0.2;
const double MATURITY = 1.0;
const double DISCOUNT = 0.95122942450071400909;
const double TWO_PI = 6.2831853071795864769;

static inline double fastExpDevice(double x) {
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

static inline double fastLogDevice(double x) {
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

class MonteCarloOptionMpiRowBlockKernel;

int rows_for_rank(int rank, int size) {
    const int base = NY / size;
    const int rem = NY % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = NY / size;
    const int rem = NY % size;
    return rank * base + std::min(rank, rem);
}

void monteCarloOptionSyclMpi(vector<int>& local_seeds,
                             vector<double>& local_prices,
                             int local_count) {
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    if (local_count <= 0) {
        return;
    }
    {
        sycl::buffer<int, 1> bufferSeeds(local_seeds.data(), sycl::range<1>(static_cast<size_t>(local_count)));
        sycl::buffer<double, 1> bufferPrices(local_prices.data(), sycl::range<1>(static_cast<size_t>(local_count)));

        q.submit([&](sycl::handler& h) {
            auto accSeeds = bufferSeeds.get_access<sycl::access::mode::read>(h);
            auto accPrices = bufferPrices.get_access<sycl::access::mode::write>(h);

            h.parallel_for<MonteCarloOptionMpiRowBlockKernel>(sycl::range<1>(static_cast<size_t>(local_count)), [=](sycl::id<1> idx) {
                const size_t linear = idx[0];
                unsigned int state = (unsigned int)accSeeds[linear] + 1u;
                double payoff_sum = 0.0;

                for (int repeat = 0; repeat < INNER_REPEATS; repeat++) {
                    for (int path = 0; path < PATHS_PER_OPTION; path++) {
                        state = state * 1664525u + 1013904223u;
                        double u1 = ((double)(state & 0x00FFFFFFu) + 1.0) / 16777217.0;
                        state = state * 1664525u + 1013904223u;
                        double u2 = ((double)(state & 0x00FFFFFFu) + 1.0) / 16777217.0;

                        double radius = sycl::sqrt(-2.0 * fastLogDevice(u1));
                        double angle = TWO_PI * u2;
                        double normal = radius * sycl::cos(angle);

                        double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
                        double diffusion = VOLATILITY * normal;
                        double terminal = S0 * fastExpDevice(drift + diffusion);
                        double payoff = terminal - STRIKE;
                        if (payoff < 0.0) {
                            payoff = 0.0;
                        }

                        payoff_sum += payoff;
                    }
                }

                accPrices[linear] = DISCOUNT * payoff_sum / (PATHS_PER_OPTION * INNER_REPEATS);
            });
        });

        q.wait();
    }
}

int main(int argc, char** argv) {
    const auto total_start = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NY) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= NY" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = rows_for_rank(rank, size);
    const int local_row_begin = row_begin_for_rank(rank, size);
    const int local_count = local_rows * NX;

    std::vector<int> counts(static_cast<size_t>(size), 0);
    std::vector<int> displs(static_cast<size_t>(size), 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[static_cast<size_t>(r)] = rows_for_rank(r, size) * NX;
        displs[static_cast<size_t>(r)] = offset;
        offset += counts[static_cast<size_t>(r)];
    }

    std::vector<int> seeds(NX * NY, 0);
    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            seeds[i * NX + j] = i * NX + j;
        }
    }

    std::vector<int> local_seeds(static_cast<size_t>(local_count), 0);
    std::vector<double> local_prices(static_cast<size_t>(local_count), 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    const double timed_start = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? seeds.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_INT,
                 local_seeds.data(),
                 local_count,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);

    const double kernel_start = MPI_Wtime();
    monteCarloOptionSyclMpi(local_seeds, local_prices, local_count);
    const double kernel_seconds_local = MPI_Wtime() - kernel_start;

    const int center_row = NY / 2;
    const int center_col = NX / 2;
    int center_owner = 0;
    for (int r = 0; r < size; ++r) {
        const int begin = row_begin_for_rank(r, size);
        const int end = begin + rows_for_rank(r, size);
        if (center_row >= begin && center_row < end) {
            center_owner = r;
            break;
        }
    }

    double center_value = 0.0;
    if (rank == center_owner) {
        const int local_row = center_row - local_row_begin;
        center_value = local_prices[static_cast<size_t>(local_row) * NX + center_col];
    }
    if (center_owner != 0) {
        if (rank == center_owner) {
            MPI_Send(&center_value, 1, MPI_DOUBLE, 0, 4701, MPI_COMM_WORLD);
        } else if (rank == 0) {
            MPI_Recv(&center_value, 1, MPI_DOUBLE, center_owner, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timed_seconds_local = MPI_Wtime() - timed_start;

    const auto total_end = std::chrono::high_resolution_clock::now();
    const double total_time_seconds = std::chrono::duration<double>(total_end - total_start).count();

    double kernel_seconds_max = 0.0;
    double timed_seconds_max = 0.0;
    double e2e_seconds_max = 0.0;
    MPI_Reduce(&kernel_seconds_local, &kernel_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timed_seconds_local, &timed_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time_seconds, &e2e_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << center_value << std::endl;
        std::cerr << "[MPI_StandardSycl][monteCarloOption][rowblock] kernel_seconds="
                  << kernel_seconds_max << std::endl;
        std::cerr << "[MPI_StandardSycl][monteCarloOption][rowblock] timed_seconds="
                  << timed_seconds_max << std::endl;
        std::cerr << "[MPI_StandardSycl][monteCarloOption][rowblock] e2e_seconds="
                  << e2e_seconds_max << std::endl;
    }

    MPI_Finalize();
    return 0;
}
