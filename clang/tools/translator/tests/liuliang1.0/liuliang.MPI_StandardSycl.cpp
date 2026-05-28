#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef LIULIANG_WIDTH
#define LIULIANG_WIDTH 100
#endif

#ifndef LIULIANG_STEPS
#define LIULIANG_STEPS 200
#endif

class LiuliangNaiveKernel;

namespace {

constexpr int WIDTH = LIULIANG_WIDTH;
constexpr int TIME_STEPS = LIULIANG_STEPS;
constexpr int interior = WIDTH - 2;
constexpr double DELTA_T = 0.01;
constexpr double DELTA_X = 1.0;

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

    std::vector<double> rho(WIDTH, 0.0);
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40.0;
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20.0;
        } else {
            rho[i] = 10.0;
        }
    }

    std::vector<double> local_new(local_count, 0.0);
    std::vector<double> global_new(interior, 0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < TIME_STEPS; ++t) {
        std::fill(local_new.begin(), local_new.end(), 0.0);

        if (local_count > 0) {
            sycl_compat::buffer<double, 1> rho_buf(
                rho.data(), sycl_compat::range<1>(rho.size()));
            sycl_compat::buffer<double, 1> out_buf(
                local_new.data(), sycl_compat::range<1>(local_new.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto in =
                    rho_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<LiuliangNaiveKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        const int gi = global_begin + static_cast<int>(idx[0]);
                        const double v_max = 30.0;
                        const double rho_max = 50.0;
                        const double q_right =
                            in[gi] * v_max * (1.0 - in[gi] / rho_max);
                        const double q_left =
                            in[gi - 1] * v_max *
                            (1.0 - in[gi - 1] / rho_max);
                        double next =
                            in[gi] - (DELTA_T / DELTA_X) *
                                         (q_right - q_left);
                        if (next < 0.0) {
                            next = 0.0;
                        }
                        out[idx] = next;
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

        for (int i = 1; i <= WIDTH - 2; ++i) {
            rho[i] = global_new[i - 1];
        }
        if (interior > 0) {
            rho[0] = global_new[0];
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
        std::cerr << "[MPI_StandardSycl][liuliang][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << static_cast<int>(rho[15]) << std::endl;
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
        std::cerr << "[MPI_StandardSycl][liuliang][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
