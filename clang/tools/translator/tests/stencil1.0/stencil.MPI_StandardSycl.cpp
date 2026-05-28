#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef STENCIL_NX
#define STENCIL_NX 1024
#endif

#ifndef STENCIL_NY
#define STENCIL_NY 1024
#endif

#ifndef STENCIL_TIME_STEPS
#define STENCIL_TIME_STEPS 200
#endif

class StencilNaiveKernel;

namespace {

constexpr int NX = STENCIL_NX;
constexpr int NY = STENCIL_NY;
constexpr int TIME_STEPS = STENCIL_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double alpha = 0.01;

int rows_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
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

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);
    const std::size_t global_size = static_cast<std::size_t>(NX) * NY;
    const std::size_t local_size = static_cast<std::size_t>(local_rows) * NY;

    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt_stability =
        (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
    const double delta_t = 0.4 * dt_stability;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size) * NY;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> global_curr(global_size, 0.0);
    std::vector<double> global_next(global_size, 0.0);
    std::vector<double> local_next(local_size, 0.0);

    const double sigma = 1.0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            const double x = i * dx;
            const double y = j * dy;
            global_curr[static_cast<std::size_t>(i) * NY + j] =
                std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                           (y - Ly / 2.0) * (y - Ly / 2.0)) /
                         (2.0 * sigma * sigma));
        }
    }

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < TIME_STEPS; ++step) {
        std::fill(local_next.begin(), local_next.end(), 0.0);

        if (local_size > 0) {
            sycl_compat::buffer<double, 1> curr_buf(
                global_curr.data(), sycl_compat::range<1>(global_curr.size()));
            sycl_compat::buffer<double, 1> next_buf(
                local_next.data(), sycl_compat::range<1>(local_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto curr =
                    curr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto next =
                    next_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<StencilNaiveKernel>(
                    sycl_compat::range<1>(local_next.size()),
                    [=](sycl_compat::id<1> idx) {
                        const int item = static_cast<int>(idx[0]);
                        const int lr = item / NY;
                        const int j = item % NY;
                        const int gi = global_begin + lr;
                        const std::size_t pos =
                            static_cast<std::size_t>(gi) * NY + j;

                        if (gi <= 0 || gi >= NX - 1 || j <= 0 ||
                            j >= NY - 1) {
                            next[idx] = 0.0;
                            return;
                        }

                        const double u_xx =
                            (curr[pos + NY] - 2.0 * curr[pos] +
                             curr[pos - NY]) /
                            (dx * dx);
                        const double u_yy =
                            (curr[pos + 1] - 2.0 * curr[pos] +
                             curr[pos - 1]) /
                            (dy * dy);
                        next[idx] =
                            curr[pos] + alpha * delta_t * (u_xx + u_yy);
                    });
            });
            q.wait();
        }

        MPI_Allgatherv(local_next.data(),
                       static_cast<int>(local_next.size()),
                       MPI_DOUBLE,
                       global_next.data(),
                       counts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for (int j = 0; j <= NY - 1; ++j) {
            global_next[j] = global_next[static_cast<std::size_t>(1) * NY + j];
            global_next[static_cast<std::size_t>(NX - 1) * NY + j] =
                global_next[static_cast<std::size_t>(NX - 2) * NY + j];
        }
        for (int i = 0; i < NX - 1; ++i) {
            global_next[static_cast<std::size_t>(i) * NY] =
                global_next[static_cast<std::size_t>(i) * NY + 1];
            global_next[static_cast<std::size_t>(i) * NY + NY - 1] =
                global_next[static_cast<std::size_t>(i) * NY + NY - 2];
        }

        std::swap(global_curr, global_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][stencil][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "{";
        for (int j = 0; j < NY; ++j) {
            std::cout << global_curr[j];
            if (j + 1 < NY) {
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
        std::cerr << "[MPI_StandardSycl][stencil][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
