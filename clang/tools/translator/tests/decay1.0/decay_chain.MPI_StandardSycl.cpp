#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef DECAY_DT
#define DECAY_DT 0.1
#endif

#ifndef DECAY_TOTAL_TIME
#define DECAY_TOTAL_TIME 5.0
#endif

#ifndef DECAY_NUM_ISOTOPES
#define DECAY_NUM_ISOTOPES 10
#endif

class DecayNaiveKernel;

namespace {

constexpr double dt = DECAY_DT;
constexpr double total_time = DECAY_TOTAL_TIME;
constexpr int num_isotopes = DECAY_NUM_ISOTOPES;
constexpr int steps = static_cast<int>(total_time / dt);

int items_for_rank(int rank, int size) {
    const int base = num_isotopes / size;
    const int rem = num_isotopes % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int rank, int size) {
    const int base = num_isotopes / size;
    const int rem = num_isotopes % size;
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
    const int local_begin = item_begin_for_rank(rank, size);

    std::vector<double> n0s(num_isotopes, 1000.0);
    std::vector<double> lambdas(num_isotopes, 0.0);
    for (int i = 0; i < num_isotopes; ++i) {
        lambdas[i] = 0.01 + 0.01 * i;
    }

    std::vector<double> local_a(local_count, 0.0);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = items_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> gathered_a(num_isotopes, 0.0);
    std::vector<double> all_a(static_cast<std::size_t>(steps) * num_isotopes,
                              0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step <= steps; ++step) {
        const double current_t = step * dt;

        if (local_count > 0) {
            sycl_compat::buffer<double, 1> n0s_buf(
                n0s.data(), sycl_compat::range<1>(n0s.size()));
            sycl_compat::buffer<double, 1> lambdas_buf(
                lambdas.data(), sycl_compat::range<1>(lambdas.size()));
            sycl_compat::buffer<double, 1> local_a_buf(
                local_a.data(), sycl_compat::range<1>(local_a.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto n0 =
                    n0s_buf.get_access<sycl_compat::access::mode::read>(h);
                auto lambda =
                    lambdas_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    local_a_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<DecayNaiveKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        const int global_i =
                            local_begin + static_cast<int>(idx[0]);
                        out[idx] = n0[global_i] *
                                   sycl_compat::exp(-lambda[global_i] *
                                                    current_t);
                    });
            });
            q.wait();
        }

        MPI_Gatherv(local_a.data(),
                    local_count,
                    MPI_DOUBLE,
                    gathered_a.data(),
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        const int row = static_cast<int>(10.0 * current_t);
        if (rank == 0 && row >= 0 && row < steps) {
            for (int i = 0; i < num_isotopes; ++i) {
                all_a[static_cast<std::size_t>(row) * num_isotopes + i] =
                    gathered_a[i];
            }
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
        std::cerr << "[MPI_StandardSycl][decay][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "{";
        for (int i = 0; i < num_isotopes; ++i) {
            std::cout << all_a[static_cast<std::size_t>(1) * num_isotopes + i];
            if (i + 1 < num_isotopes) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
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
        std::cerr << "[MPI_StandardSycl][decay][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
