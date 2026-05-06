// MDP / Fokker-Planck equation — MPI + SYCL standard implementation
// 1D stencil: diffusion + drift update, time-stepping loop.
// Spatial domain split across ranks with 1-element halo exchange per step.

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

const double A = 1.0;   // attraction coefficient
const double D = 0.1;   // diffusion coefficient
const double dx = 0.1;  // spatial step
const double dt = 0.01; // time step

#ifndef MDP_N
#define MDP_N 150
#endif

#ifndef MDP_T
#define MDP_T 10000
#endif

constexpr int N = MDP_N;
constexpr int T = MDP_T;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // We compute interior points 1..N-2 (N-2 points)
    const int interior = N - 2;
    if (size > interior) {
        if (rank == 0) std::cerr << "MPI size too large\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int base = interior / size;
    const int rem  = interior % size;
    const int local_count = base + (rank < rem ? 1 : 0);
    const int local_begin = 1 + rank * base + std::min(rank, rem);

    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    // Local arrays: 1 left halo + local_count + 1 right halo
    const int local_ext = local_count + 2;
    std::vector<double> p(local_ext, 0.0);
    std::vector<double> new_p(local_ext, 0.0);

    // Initialize with Gaussian distribution
    for (int li = 0; li < local_count; ++li) {
        const int gi = local_begin + li;
        double x = gi * dx;
        p[li + 1] = std::exp(-std::pow(x - 5.0, 2) / 2.0);
    }

    sycl::queue q{sycl::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int t = 0; t < T; ++t) {
        // Halo exchange
        double send_left  = p[1];
        double send_right = p[local_count];
        double recv_left  = 0.0;
        double recv_right = 0.0;

        MPI_Sendrecv(&send_left,  1, MPI_DOUBLE, prev, 0,
                     &recv_right, 1, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, next, 1,
                     &recv_left,  1, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (prev != MPI_PROC_NULL) p[0] = recv_left;
        if (next != MPI_PROC_NULL) p[local_count + 1] = recv_right;

        // SYCL kernel
        {
            sycl::buffer<double, 1> buf_p(p.data(), sycl::range<1>(local_ext));
            sycl::buffer<double, 1> buf_np(new_p.data(), sycl::range<1>(local_ext));

            q.submit([&](sycl::handler& h) {
                auto acc_p  = buf_p.get_access<sycl::access::mode::read>(h);
                auto acc_np = buf_np.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                    const int li = static_cast<int>(idx[0]) + 1;
                    double diffusion = D * (acc_p[li + 1] - 2 * acc_p[li] + acc_p[li - 1]) / (dx * dx);
                    double drift = -A * (acc_p[li + 1] - acc_p[li - 1]) / (2 * dx);
                    acc_np[li] = acc_p[li] + dt * (diffusion + drift);
                });
            });
            q.wait();
        }

        std::swap(p, new_p);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all interior points
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (interior / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> local_out(p.begin() + 1, p.begin() + 1 + local_count);

    std::vector<double> global_p;
    if (rank == 0) global_p.resize(N, 0.0);

    MPI_Gatherv(local_out.data(), local_count, MPI_DOUBLE,
                rank == 0 ? global_p.data() + 1 : nullptr,
                counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mdp] seconds=" << max_elapsed << std::endl;
        // Match dac version output: p[2]
        std::cout << global_p[2] << std::endl;
    }

    MPI_Finalize();
    return 0;
}
