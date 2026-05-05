#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef STENCIL_NX
#define STENCIL_NX 1024
#endif

#ifndef STENCIL_NY
#define STENCIL_NY 1024
#endif

#ifndef STENCIL_TIME_STEPS
#define STENCIL_TIME_STEPS 200
#endif

constexpr int NX = STENCIL_NX;
constexpr int NY = STENCIL_NY;
constexpr int TIME_STEPS = STENCIL_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double alpha = 0.01;

static int rows_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return base + (rank < rem ? 1 : 0);
}

static int row_begin_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return rank * base + std::min(rank, rem);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= STENCIL_NX\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int local_rows = rows_for_rank(rank, size);
    const int row_begin = row_begin_for_rank(rank, size);
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt_stability =
        (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
    const double delta_t = 0.4 * dt_stability;
    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t local_extent =
        static_cast<std::size_t>(local_rows + 2) * pitch;

    std::vector<double> local_curr(local_extent, 0.0);
    std::vector<double> local_next(local_extent, 0.0);

    const double sigma = 1.0;
    for (int lr = 1; lr <= local_rows; ++lr) {
        const int gi = row_begin + lr - 1;
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            local_curr[static_cast<std::size_t>(lr) * pitch + j] =
                std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                           (y - Ly / 2.0) * (y - Ly / 2.0)) /
                         (2.0 * sigma * sigma));
        }
    }

    sycl::queue q(sycl::default_selector_v);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < TIME_STEPS; ++step) {
        MPI_Sendrecv(local_curr.data() + pitch, NY, MPI_DOUBLE, prev, 10,
                     local_curr.data(), NY, MPI_DOUBLE, prev, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(local_curr.data() + static_cast<std::size_t>(local_rows) * pitch,
                     NY, MPI_DOUBLE, next, 11,
                     local_curr.data() + static_cast<std::size_t>(local_rows + 1) * pitch,
                     NY, MPI_DOUBLE, next, 10, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        {
            sycl::buffer<double, 1> curr_buf(local_curr.data(),
                                             sycl::range<1>(local_curr.size()));
            sycl::buffer<double, 1> next_buf(local_next.data(),
                                             sycl::range<1>(local_next.size()));
            q.submit([&](sycl::handler& h) {
                auto curr = curr_buf.get_access<sycl::access::mode::read>(h);
                auto next_acc = next_buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::range<2>(
                                   static_cast<std::size_t>(local_rows),
                                   static_cast<std::size_t>(NY - 2)),
                               [=](sycl::id<2> idx) {
                                   const int lr = static_cast<int>(idx[0]) + 1;
                                   const int j = static_cast<int>(idx[1]) + 1;
                                   const int gi = row_begin + lr - 1;
                                   if (gi <= 0 || gi >= NX - 1) {
                                       return;
                                   }
                                   const std::size_t center =
                                       static_cast<std::size_t>(lr) * NY + j;
                                   const double u_xx =
                                       (curr[center + NY] - 2.0 * curr[center] +
                                        curr[center - NY]) /
                                       (dx * dx);
                                   const double u_yy =
                                       (curr[center + 1] - 2.0 * curr[center] +
                                        curr[center - 1]) /
                                       (dy * dy);
                                   next_acc[center] =
                                       curr[center] + alpha * delta_t * (u_xx + u_yy);
                               });
            });
            q.submit([&](sycl::handler& h) {
                auto next_acc =
                    next_buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_rows)),
                               [=](sycl::id<1> idx) {
                                   const int lr = static_cast<int>(idx[0]) + 1;
                                   const int gi = row_begin + lr - 1;
                                   const std::size_t row =
                                       static_cast<std::size_t>(lr) * NY;
                                   if (gi > 0 && gi < NX - 1) {
                                       next_acc[row] = next_acc[row + 1];
                                       next_acc[row + NY - 1] =
                                           next_acc[row + NY - 2];
                                   }
                               });
            });
            q.submit([&](sycl::handler& h) {
                auto next_acc =
                    next_buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(NY)),
                               [=](sycl::id<1> idx) {
                                   const int j = static_cast<int>(idx[0]);
                                   if (row_begin == 0) {
                                       next_acc[pitch + j] =
                                           next_acc[2 * pitch + j];
                                   }
                                   if (row_begin + local_rows == NX) {
                                       next_acc[static_cast<std::size_t>(local_rows) *
                                                    pitch +
                                                j] =
                                           next_acc[static_cast<std::size_t>(
                                                        local_rows - 1) *
                                                        pitch +
                                                    j];
                                   }
                               });
            });
            q.wait();
        }

        std::swap(local_curr, local_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> top_row;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = rows_for_rank(r, size) * NY;
            displs[r] = row_begin_for_rank(r, size) * NY;
        }
        top_row.resize(static_cast<std::size_t>(NX) * NY);
    }

    MPI_Gatherv(local_curr.data() + pitch, local_rows * NY, MPI_DOUBLE,
                rank == 0 ? top_row.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "time_sec=" << elapsed << "\n";
        std::cout << "sample=" << top_row[0] << "\n";
    }

    MPI_Finalize();
    return 0;
}
