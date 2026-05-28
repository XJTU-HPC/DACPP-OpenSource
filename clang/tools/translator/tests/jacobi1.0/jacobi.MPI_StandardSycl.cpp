#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef JACOBI_N
#define JACOBI_N 10
#endif

#ifndef JACOBI_MAX_ITER
#define JACOBI_MAX_ITER 100
#endif

#ifndef JACOBI_TOLERANCE
#define JACOBI_TOLERANCE 1e-6f
#endif

class JacobiNaiveKernel;

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
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<float> A(static_cast<std::size_t>(N) * N, 0.0f);
    std::vector<float> b(N, 1.0f);
    std::vector<float> x(N, 0.0f);
    std::vector<float> x_new(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        A[static_cast<std::size_t>(i) * N + i] = 4.0f;
        if (i > 0) {
            A[static_cast<std::size_t>(i) * N + i - 1] = -1.0f;
        }
        if (i + 1 < N) {
            A[static_cast<std::size_t>(i) * N + i + 1] = -1.0f;
        }
    }

    std::vector<float> local_x_new(local_rows, 0.0f);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    bool converged = false;
    int iter = 0;
    while (!converged && iter < max_iter) {
        std::fill(local_x_new.begin(), local_x_new.end(), 0.0f);

        if (local_rows > 0) {
            sycl_compat::buffer<float, 1> a_buf(
                A.data(), sycl_compat::range<1>(A.size()));
            sycl_compat::buffer<float, 1> b_buf(
                b.data(), sycl_compat::range<1>(b.size()));
            sycl_compat::buffer<float, 1> x_buf(
                x.data(), sycl_compat::range<1>(x.size()));
            sycl_compat::buffer<float, 1> out_buf(
                local_x_new.data(),
                sycl_compat::range<1>(local_x_new.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto a = a_buf.get_access<sycl_compat::access::mode::read>(h);
                auto rhs = b_buf.get_access<sycl_compat::access::mode::read>(h);
                auto old_x =
                    x_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);

                h.parallel_for<JacobiNaiveKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_rows)),
                    [=](sycl_compat::id<1> idx) {
                        const int row = global_begin + static_cast<int>(idx[0]);
                        float sigma = 0.0f;
                        for (int j = 0; j < N; ++j) {
                            if (j != row) {
                                sigma += a[static_cast<std::size_t>(row) * N +
                                           j] *
                                         old_x[j];
                            }
                        }
                        out[idx] = (rhs[row] - sigma) /
                                   a[static_cast<std::size_t>(row) * N + row];
                    });
            });
            q.wait();
        }

        MPI_Allgatherv(local_x_new.data(),
                       local_rows,
                       MPI_FLOAT,
                       x_new.data(),
                       counts.data(),
                       displs.data(),
                       MPI_FLOAT,
                       MPI_COMM_WORLD);

        float local_error = 0.0f;
        for (int lr = 0; lr < local_rows; ++lr) {
            const int gi = global_begin + lr;
            local_error =
                std::max(local_error, std::fabs(x_new[gi] - x[gi]));
        }
        float max_error = 0.0f;
        MPI_Allreduce(
            &local_error, &max_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        converged = max_error < tolerance;
        x = x_new;
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
        std::cerr << "[MPI_StandardSycl][jacobi][naive] iter=" << iter
                  << " seconds=" << max_seconds << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << x_new[i] << " ";
        }
        std::cout << std::endl;
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
        std::cerr << "[MPI_StandardSycl][jacobi][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
