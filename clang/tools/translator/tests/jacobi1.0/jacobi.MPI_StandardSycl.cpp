#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef JACOBI_N
#define JACOBI_N 10
#endif

#ifndef JACOBI_MAX_ITER
#define JACOBI_MAX_ITER 100
#endif

#ifndef JACOBI_TOLERANCE
#define JACOBI_TOLERANCE 1e-6f
#endif

namespace {

constexpr int N = JACOBI_N;
constexpr int max_iter = JACOBI_MAX_ITER;
constexpr float tolerance = JACOBI_TOLERANCE;

int rows_for_rank(int rank, int size) {
    const int base = N / size;
    const int rem = N % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = N / size;
    const int rem = N % size;
    return rank * base + std::min(rank, rem);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > N) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= JACOBI_N\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);

    std::vector<float> local_A(static_cast<std::size_t>(local_rows) * N, 0.0f);
    std::vector<float> local_b(local_rows, 1.0f);
    std::vector<float> x(N, 0.0f);
    std::vector<float> local_x_new(local_rows, 0.0f);
    std::vector<float> local_error(local_rows, 0.0f);

    for (int lr = 0; lr < local_rows; ++lr) {
        const int gi = global_begin + lr;
        local_A[static_cast<std::size_t>(lr) * N + gi] = 4.0f;
        if (gi > 0) {
            local_A[static_cast<std::size_t>(lr) * N + gi - 1] = -1.0f;
        }
        if (gi + 1 < N) {
            local_A[static_cast<std::size_t>(lr) * N + gi + 1] = -1.0f;
        }
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    sycl::queue q{sycl::default_selector_v};
    float* d_A = sycl::malloc_device<float>(local_A.size(), q);
    float* d_b = sycl::malloc_device<float>(local_b.size(), q);
    float* d_x = sycl::malloc_device<float>(x.size(), q);
    float* d_x_new = sycl::malloc_device<float>(local_x_new.size(), q);
    float* d_error = sycl::malloc_device<float>(local_error.size(), q);

    q.memcpy(d_A, local_A.data(), sizeof(float) * local_A.size()).wait();
    q.memcpy(d_b, local_b.data(), sizeof(float) * local_b.size()).wait();

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    bool converged = false;
    int iter = 0;
    while (!converged && iter < max_iter) {
        q.memcpy(d_x, x.data(), sizeof(float) * x.size()).wait();

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_rows)),
                           [=](sycl::id<1> idx) {
                               const int lr = static_cast<int>(idx[0]);
                               const int gi = global_begin + lr;
                               float sigma = 0.0f;
                               for (int j = 0; j < N; ++j) {
                                   if (j != gi) {
                                       sigma += d_A[static_cast<std::size_t>(lr) *
                                                    N +
                                                    j] *
                                                d_x[j];
                                   }
                               }
                               const float value =
                                   (d_b[lr] - sigma) /
                                   d_A[static_cast<std::size_t>(lr) * N + gi];
                               d_x_new[lr] = value;
                               const float diff = value - d_x[gi];
                               d_error[lr] = diff < 0.0f ? -diff : diff;
                           });
        }).wait();

        q.memcpy(local_x_new.data(),
                 d_x_new,
                 sizeof(float) * local_x_new.size())
            .wait();
        q.memcpy(local_error.data(), d_error, sizeof(float) * local_error.size())
            .wait();

        float local_max_error = 0.0f;
        for (float err : local_error) {
            local_max_error = std::max(local_max_error, err);
        }

        float global_max_error = 0.0f;
        MPI_Allreduce(&local_max_error,
                      &global_max_error,
                      1,
                      MPI_FLOAT,
                      MPI_MAX,
                      MPI_COMM_WORLD);

        MPI_Allgatherv(local_x_new.data(),
                       local_rows,
                       MPI_FLOAT,
                       x.data(),
                       counts.data(),
                       displs.data(),
                       MPI_FLOAT,
                       MPI_COMM_WORLD);

        converged = global_max_error < tolerance;
        ++iter;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][jacobi] iter=" << iter
                  << " seconds=" << max_seconds << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }

    sycl::free(d_A, q);
    sycl::free(d_b, q);
    sycl::free(d_x, q);
    sycl::free(d_x_new, q);
    sycl::free(d_error, q);

    MPI_Finalize();
    return 0;
}
