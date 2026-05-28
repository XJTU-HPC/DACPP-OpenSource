#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef MDP_N
#define MDP_N 150
#endif

#ifndef MDP_T
#define MDP_T 1000
#endif

class MDPNaiveKernel;

namespace {

constexpr double A = 1.0;
constexpr double D = 0.1;
constexpr double dx = 0.1;
constexpr double dt = 0.01;
constexpr int N = MDP_N;
constexpr int T = MDP_T;
constexpr int interior = N - 2;

int items_for_rank(int rank, int size) {
    const int base = interior / size;
    const int rem = interior % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int rank, int size) {
    const int base = interior / size;
    const int rem = interior % size;
    return rank * base + std::min(rank, rem);
}

} // namespace

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local_count = items_for_rank(rank, size);
    const int output_begin = item_begin_for_rank(rank, size);
    const int global_begin = 1 + output_begin;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = items_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> p(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2.0) / 2.0);
    }
    std::vector<double> local_new(local_count, 0.0);
    std::vector<double> global_new(interior, 0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < T; ++t) {
        std::fill(local_new.begin(), local_new.end(), 0.0);

        if (local_count > 0) {
            sycl_compat::buffer<double, 1> p_buf(
                p.data(), sycl_compat::range<1>(p.size()));
            sycl_compat::buffer<double, 1> out_buf(
                local_new.data(), sycl_compat::range<1>(local_new.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto in = p_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<MDPNaiveKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        const int gi = global_begin + static_cast<int>(idx[0]);
                        const double diffusion =
                            D * (in[gi + 1] - 2.0 * in[gi] + in[gi - 1]) /
                            (dx * dx);
                        const double drift =
                            -A * (in[gi + 1] - in[gi - 1]) / (2.0 * dx);
                        out[idx] = in[gi] + dt * (diffusion + drift);
                    });
            });
            q.wait();
        }

        MPI_Allgatherv(local_new.data(),
                       local_count,
                       MPI_DOUBLE,
                       global_new.data(),
                       counts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for (int i = 0; i < interior; ++i) {
            p[i + 1] = global_new[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mdp][naive] seconds=" << max_seconds
                  << std::endl;
        std::cout << p[2] << std::endl;
    }

    const auto e2e_t1 = std::chrono::steady_clock::now();
    const double e2e_local_seconds =
        std::chrono::duration<double>(e2e_t1 - e2e_t0).count();
    double e2e_seconds = 0.0;
    MPI_Reduce(&e2e_local_seconds,
               &e2e_seconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mdp][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
