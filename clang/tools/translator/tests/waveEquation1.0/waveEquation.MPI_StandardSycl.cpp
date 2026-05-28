#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef WAVE_NX
#define WAVE_NX 8
#endif

#ifndef WAVE_NY
#define WAVE_NY 8
#endif

#ifndef WAVE_TIME_STEPS
#define WAVE_TIME_STEPS 10
#endif

class WaveEquationStandardHaloKernel;

namespace {

constexpr int NX = WAVE_NX;
constexpr int NY = WAVE_NY;
constexpr int TIME_STEPS = WAVE_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double c = 1.0;

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

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= WAVE_NX\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);
    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t local_with_halo =
        static_cast<std::size_t>(local_rows + 2) * pitch;

    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt = 0.5 * std::fmin(dx, dy) / c;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[static_cast<std::size_t>(r)] = rows_for_rank(r, size) * NY;
        displs[static_cast<std::size_t>(r)] = offset;
        offset += counts[static_cast<std::size_t>(r)];
    }

    std::vector<double> h_prev(local_with_halo, 0.0);
    std::vector<double> h_curr(local_with_halo, 0.0);
    std::vector<double> h_next(local_with_halo, 0.0);

    const double sigma = 0.5;
    for (int lr = 0; lr < local_rows; ++lr) {
        const int gi = global_begin + lr;
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            h_prev[static_cast<std::size_t>(lr + 1) * NY + j] =
                std::exp(-((x - Lx / 2) * (x - Lx / 2) +
                           (y - Ly / 2) * (y - Ly / 2)) /
                         (2 * sigma * sigma));
        }
    }

    sycl_compat::queue q{sycl_compat::default_selector_v};

    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < TIME_STEPS; ++step) {
        MPI_Request reqs[4];
        int req_count = 0;
        if (up != MPI_PROC_NULL) {
            MPI_Irecv(h_curr.data(),
                      NY,
                      MPI_DOUBLE,
                      up,
                      1,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(h_curr.data() + pitch,
                      NY,
                      MPI_DOUBLE,
                      up,
                      0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(h_curr.data() +
                          static_cast<std::size_t>(local_rows + 1) * pitch,
                      NY,
                      MPI_DOUBLE,
                      down,
                      0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(h_curr.data() +
                          static_cast<std::size_t>(local_rows) * pitch,
                      NY,
                      MPI_DOUBLE,
                      down,
                      1,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        {
            sycl_compat::buffer<double, 1> prev_buf(
                h_prev.data(), sycl_compat::range<1>(h_prev.size()));
            sycl_compat::buffer<double, 1> curr_buf(
                h_curr.data(), sycl_compat::range<1>(h_curr.size()));
            sycl_compat::buffer<double, 1> next_buf(
                h_next.data(), sycl_compat::range<1>(h_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto prev =
                    prev_buf.get_access<sycl_compat::access::mode::read>(h);
                auto curr =
                    curr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto next =
                    next_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<WaveEquationStandardHaloKernel>(
                    sycl_compat::range<2>(static_cast<std::size_t>(local_rows),
                                          static_cast<std::size_t>(NY)),
                    [=](sycl_compat::id<2> idx) {
                        const int lr = static_cast<int>(idx[0]) + 1;
                        const int j = static_cast<int>(idx[1]);
                        const int gi = global_begin + lr - 1;
                        const std::size_t pos =
                            static_cast<std::size_t>(lr) * NY + j;

                        if (gi == 0 || gi == NX - 1 || j == 0 || j == NY - 1) {
                            next[pos] = 0.0;
                            return;
                        }

                        const double center = curr[pos];
                        const double u_xx =
                            (curr[pos + NY] - 2.0 * center + curr[pos - NY]) /
                            (dx * dx);
                        const double u_yy =
                            (curr[pos + 1] - 2.0 * center + curr[pos - 1]) /
                            (dy * dy);
                        next[pos] = 2.0 * center - prev[pos] +
                                    c * c * dt * dt * (u_xx + u_yy);
                    });
            });
            q.wait();
        }

        std::swap(h_prev, h_curr);
        std::swap(h_curr, h_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_out(static_cast<std::size_t>(local_rows) * NY);
    for (int lr = 0; lr < local_rows; ++lr) {
        std::copy(h_curr.begin() +
                      static_cast<std::ptrdiff_t>((lr + 1) * NY),
                  h_curr.begin() +
                      static_cast<std::ptrdiff_t>((lr + 2) * NY),
                  local_out.begin() + static_cast<std::ptrdiff_t>(lr * NY));
    }

    std::vector<double> global_out;
    if (rank == 0) {
        global_out.resize(static_cast<std::size_t>(NX) * NY);
    }
    MPI_Gatherv(local_out.data(),
                static_cast<int>(local_out.size()),
                MPI_DOUBLE,
                rank == 0 ? global_out.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][wave] seconds=" << max_seconds
                  << std::endl;
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
        std::cerr << "[MPI_StandardSycl][wave] e2e_seconds=" << e2e_seconds
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
