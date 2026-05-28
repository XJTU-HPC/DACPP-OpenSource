#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef MATMUL_M
#define MATMUL_M 4
#endif

#ifndef MATMUL_K
#define MATMUL_K 5
#endif

#ifndef MATMUL_N
#define MATMUL_N 4
#endif

class MatMulNaiveKernel;

namespace {

constexpr int M = MATMUL_M;
constexpr int K = MATMUL_K;
constexpr int N = MATMUL_N;

int rows_for_rank(int rank, int size) {
    const int base = M / size;
    const int rem = M % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = M / size;
    const int rem = M % size;
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
    const int row_begin = row_begin_for_rank(rank, size);
    const int local_count = local_rows * N;

    std::vector<int> A(static_cast<std::size_t>(M) * K, 0);
    std::vector<int> B(static_cast<std::size_t>(K) * N, 0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            A[static_cast<std::size_t>(i) * K + k] = i * K + k + 1;
        }
    }
    if (K == 5 && N == 4) {
        B = {1,  5,  9,  13, 17, 2,  6,  10, 14, 18,
             3,  7,  11, 15, 19, 4,  8,  12, 16, 20};
    } else {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                B[static_cast<std::size_t>(k) * N + j] = k * N + j + 1;
            }
        }
    }

    std::vector<int> local_result(local_count, 0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    if (local_rows > 0) {
        sycl_compat::buffer<int, 1> a_buf(A.data(), sycl_compat::range<1>(A.size()));
        sycl_compat::buffer<int, 1> b_buf(B.data(), sycl_compat::range<1>(B.size()));
        sycl_compat::buffer<int, 1> c_buf(local_result.data(),
                                          sycl_compat::range<1>(local_result.size()));

        q.submit([&](sycl_compat::handler& h) {
            auto a = a_buf.get_access<sycl_compat::access::mode::read>(h);
            auto b = b_buf.get_access<sycl_compat::access::mode::read>(h);
            auto c = c_buf.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<MatMulNaiveKernel>(
                sycl_compat::range<2>(static_cast<std::size_t>(local_rows),
                                       static_cast<std::size_t>(N)),
                [=](sycl_compat::id<2> idx) {
                    const int lr = static_cast<int>(idx[0]);
                    const int j = static_cast<int>(idx[1]);
                    const int gi = row_begin + lr;
                    int sum = 0;
                    for (int k = 0; k < K; ++k) {
                        sum += a[static_cast<std::size_t>(gi) * K + k] *
                               b[static_cast<std::size_t>(k) * N + j];
                    }
                    c[static_cast<std::size_t>(lr) * N + j] = sum;
                });
        });
        q.wait();
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size) * N;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_result(static_cast<std::size_t>(M) * N, 0);
    MPI_Gatherv(local_result.data(),
                local_count,
                MPI_INT,
                global_result.data(),
                counts.data(),
                displs.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][matmul][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "{";
        for (int i = 0; i < M; ++i) {
            std::cout << "{";
            for (int j = 0; j < N; ++j) {
                std::cout << global_result[static_cast<std::size_t>(i) * N + j];
                if (j + 1 < N) {
                    std::cout << ", ";
                }
            }
            std::cout << "}";
            if (i + 1 < M) {
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
        std::cerr << "[MPI_StandardSycl][matmul][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
