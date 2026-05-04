#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef WAVE_NX
#define WAVE_NX 8
#endif

#ifndef WAVE_NY
#define WAVE_NY 8
#endif

#ifndef WAVE_TIME_STEPS
#define WAVE_TIME_STEPS 10
#endif

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

    std::vector<double> h_prev(local_with_halo, 0.0);
    std::vector<double> h_curr(local_with_halo, 0.0);
    std::vector<double> h_next(local_with_halo, 0.0);

    const double sigma = 0.5;
    for (int lr = 0; lr < local_rows; ++lr) {
        const int gi = global_begin + lr;
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            h_prev[static_cast<std::size_t>(lr + 1) * pitch + j] =
                std::exp(-((x - Lx / 2) * (x - Lx / 2) +
                           (y - Ly / 2) * (y - Ly / 2)) /
                         (2 * sigma * sigma));
        }
    }

    sycl::queue q{sycl::default_selector_v};
    double* d_prev = sycl::malloc_device<double>(local_with_halo, q);
    double* d_curr = sycl::malloc_device<double>(local_with_halo, q);
    double* d_next = sycl::malloc_device<double>(local_with_halo, q);

    q.memcpy(d_prev, h_prev.data(), sizeof(double) * local_with_halo).wait();
    q.memcpy(d_curr, h_curr.data(), sizeof(double) * local_with_halo).wait();
    q.memcpy(d_next, h_next.data(), sizeof(double) * local_with_halo).wait();

    std::vector<double> send_top(NY, 0.0);
    std::vector<double> send_bottom(NY, 0.0);
    std::vector<double> recv_top(NY, 0.0);
    std::vector<double> recv_bottom(NY, 0.0);

    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < TIME_STEPS; ++step) {
        q.memcpy(send_top.data(), d_curr + pitch, sizeof(double) * pitch).wait();
        q.memcpy(send_bottom.data(),
                 d_curr + static_cast<std::size_t>(local_rows) * pitch,
                 sizeof(double) * pitch)
            .wait();

        MPI_Sendrecv(send_top.data(),
                     NY,
                     MPI_DOUBLE,
                     up,
                     0,
                     recv_bottom.data(),
                     NY,
                     MPI_DOUBLE,
                     down,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom.data(),
                     NY,
                     MPI_DOUBLE,
                     down,
                     1,
                     recv_top.data(),
                     NY,
                     MPI_DOUBLE,
                     up,
                     1,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        if (up != MPI_PROC_NULL) {
            q.memcpy(d_curr, recv_top.data(), sizeof(double) * pitch).wait();
        }
        if (down != MPI_PROC_NULL) {
            q.memcpy(d_curr + static_cast<std::size_t>(local_rows + 1) * pitch,
                     recv_bottom.data(),
                     sizeof(double) * pitch)
                .wait();
        }

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(
                               static_cast<std::size_t>(local_rows),
                               static_cast<std::size_t>(NY)),
                           [=](sycl::id<2> idx) {
                               const int lr = static_cast<int>(idx[0]) + 1;
                               const int j = static_cast<int>(idx[1]);
                               const int gi = global_begin + lr - 1;
                               const std::size_t pos =
                                   static_cast<std::size_t>(lr) * pitch + j;

                               if (gi == 0 || gi == NX - 1 || j == 0 ||
                                   j == NY - 1) {
                                   d_next[pos] = 0.0;
                                   return;
                               }

                               const double center = d_curr[pos];
                               const double u_xx =
                                   (d_curr[pos + pitch] - 2.0 * center +
                                    d_curr[pos - pitch]) /
                                   (dx * dx);
                               const double u_yy =
                                   (d_curr[pos + 1] - 2.0 * center +
                                    d_curr[pos - 1]) /
                                   (dy * dy);
                               d_next[pos] =
                                   2.0 * center - d_prev[pos] +
                                   c * c * dt * dt * (u_xx + u_yy);
                           });
        }).wait();

        std::swap(d_prev, d_curr);
        std::swap(d_curr, d_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_out(static_cast<std::size_t>(local_rows) * pitch);
    q.memcpy(local_out.data(), d_curr + pitch, sizeof(double) * local_out.size())
        .wait();

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> global_out;
    if (rank == 0) {
        counts.resize(size, 0);
        displs.resize(size, 0);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = rows_for_rank(r, size) * NY;
            displs[r] = offset;
            offset += counts[r];
        }
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
        std::cout << "{";
        for (int i = 0; i < NX; ++i) {
            std::cout << "{";
            for (int j = 0; j < NY; ++j) {
                std::cout << global_out[static_cast<std::size_t>(i) * NY + j];
                if (j + 1 < NY) {
                    std::cout << ", ";
                }
            }
            std::cout << "}";
            if (i + 1 < NX) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }

    sycl::free(d_prev, q);
    sycl::free(d_curr, q);
    sycl::free(d_next, q);

    MPI_Finalize();
    return 0;
}
