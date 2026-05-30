#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

class StencilRowHaloKernel;

namespace {

constexpr int NX = STENCIL_NX;
constexpr int NY = STENCIL_NY;
constexpr int TIME_STEPS = STENCIL_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double alpha = 0.01;

#ifdef STENCIL_ENABLE_RESULT_DUMP
void dump_result_if_requested(const std::vector<double>& values) {
    const char* path = std::getenv("STENCIL_RESULT_DUMP");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t rows = static_cast<std::uint64_t>(NX);
    const std::uint64_t cols = static_cast<std::uint64_t>(NY);
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
}
#endif

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
    const std::size_t local_owned_size =
        static_cast<std::size_t>(local_rows) * NY;
    const int halo_rows = local_rows > 0 ? local_rows + 2 : 0;
    const std::size_t local_halo_size =
        static_cast<std::size_t>(halo_rows) * NY;

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

    std::vector<double> global_curr(rank == 0 ? global_size : 0, 0.0);
    std::vector<double> local_curr(local_halo_size, 0.0);
    std::vector<double> local_next(local_halo_size, 0.0);

    const double sigma = 1.0;
    for (int lr = 0; lr < halo_rows; ++lr) {
        const int gi = global_begin + lr - 1;
        if (gi < 0 || gi >= NX) {
            continue;
        }
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            local_curr[static_cast<std::size_t>(lr) * NY + j] =
                std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                           (y - Ly / 2.0) * (y - Ly / 2.0)) /
                         (2.0 * sigma * sigma));
        }
    }

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < TIME_STEPS; ++step) {
        if (local_rows > 0) {
            const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
            const int next_rank = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
            MPI_Request reqs[4];
            int req_count = 0;
            if (prev != MPI_PROC_NULL) {
                MPI_Irecv(local_curr.data(),
                          NY,
                          MPI_DOUBLE,
                          prev,
                          100,
                          MPI_COMM_WORLD,
                          &reqs[req_count++]);
                MPI_Isend(local_curr.data() + NY,
                          NY,
                          MPI_DOUBLE,
                          prev,
                          101,
                          MPI_COMM_WORLD,
                          &reqs[req_count++]);
            }
            if (next_rank != MPI_PROC_NULL) {
                MPI_Irecv(local_curr.data() +
                              static_cast<std::size_t>(local_rows + 1) * NY,
                          NY,
                          MPI_DOUBLE,
                          next_rank,
                          101,
                          MPI_COMM_WORLD,
                          &reqs[req_count++]);
                MPI_Isend(local_curr.data() +
                              static_cast<std::size_t>(local_rows) * NY,
                          NY,
                          MPI_DOUBLE,
                          next_rank,
                          100,
                          MPI_COMM_WORLD,
                          &reqs[req_count++]);
            }
            if (req_count > 0) {
                MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            }
        }

        if (local_owned_size > 0) {
            sycl_compat::buffer<double, 1> curr_buf(
                local_curr.data(), sycl_compat::range<1>(local_curr.size()));
            sycl_compat::buffer<double, 1> next_buf(
                local_next.data(), sycl_compat::range<1>(local_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto curr =
                    curr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto next =
                    next_buf.get_access<
                        sycl_compat::access::mode::discard_write>(h);

                h.parallel_for<StencilRowHaloKernel>(
                    sycl_compat::range<1>(local_owned_size),
                    [=](sycl_compat::id<1> idx) {
                        const int item = static_cast<int>(idx[0]);
                        const int lr = item / NY;
                        const int j = item % NY;
                        const int gi = global_begin + lr;
                        const std::size_t pos =
                            static_cast<std::size_t>(lr + 1) * NY + j;

                        if (gi <= 0 || gi >= NX - 1 || j <= 0 ||
                            j >= NY - 1) {
                            next[pos] = 0.0;
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
                        next[pos] =
                            curr[pos] + alpha * delta_t * (u_xx + u_yy);
                    });
            });
            q.wait();
        }

        if (local_rows > 0 && global_begin == 0 && local_rows > 1) {
            for (int j = 0; j <= NY - 1; ++j) {
                local_next[static_cast<std::size_t>(1) * NY + j] =
                    local_next[static_cast<std::size_t>(2) * NY + j];
            }
        }
        if (local_rows > 0 && global_begin + local_rows == NX &&
            local_rows > 1) {
            for (int j = 0; j <= NY - 1; ++j) {
                local_next[static_cast<std::size_t>(local_rows) * NY + j] =
                    local_next[static_cast<std::size_t>(local_rows - 1) * NY +
                               j];
            }
        }
        for (int lr = 0; lr < local_rows; ++lr) {
            const int gi = global_begin + lr;
            if (gi >= NX - 1) {
                continue;
            }
            const std::size_t row = static_cast<std::size_t>(lr + 1) * NY;
            local_next[row] = local_next[row + 1];
            local_next[row + NY - 1] = local_next[row + NY - 2];
        }

        std::swap(local_curr, local_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][stencil][row-halo] seconds="
                  << max_seconds << std::endl;
    }

    MPI_Gatherv(local_rows > 0 ? local_curr.data() + NY : nullptr,
                static_cast<int>(local_owned_size),
                MPI_DOUBLE,
                rank == 0 ? global_curr.data() : nullptr,
                counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        // Full result printing is intentionally disabled for Sunway benchmark runs.
#ifdef STENCIL_ENABLE_RESULT_DUMP
        dump_result_if_requested(global_curr);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
        std::cerr << "[MPI_StandardSycl][stencil][row-halo] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
