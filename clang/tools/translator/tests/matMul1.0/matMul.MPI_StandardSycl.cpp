// Matrix Multiplication — MPI + SYCL standard implementation
// Split output rows across ranks. Each rank computes its portion of C = A * B.
// A is 4x5, B is 5x4 (column-major), C is 4x4.
//
// Communication aligned with DACPP generated code:
//   - MPI_Scatterv distributes A rows (RowPartitionFullRow)
//   - MPI_Bcast broadcasts full B to all ranks
//   - MPI_Gatherv collects C results

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int M = 4, K = 5, N = 4;

    // A (4x5) row-major — all ranks allocate; rank 0 holds source data
    std::vector<int> dataA{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };

    // B (5x4) column-major storage as in dac version
    std::vector<int> dataB{
        1, 5, 9, 13,
        17, 2, 6, 10,
        14, 18, 3, 7,
        11, 15, 19, 4,
        8, 12, 16, 20
    };

    // Partition C rows across ranks
    const int base = M / size;
    const int rem  = M % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_begin = rank * base + std::min(rank, rem);
    const int local_count = local_rows * N;

    // --- Scatter A rows: rank 0 distributes row-blocks (aligned with DACPP RowPartitionFullRow) ---
    std::vector<int> counts_A(size), displs_A(size);
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        int rb = r * base + std::min(r, rem);
        counts_A[r] = lr * K;
        displs_A[r] = rb * K;
    }
    std::vector<int> localA(local_rows * K);
    MPI_Scatterv(rank == 0 ? dataA.data() : nullptr,
                 counts_A.data(), displs_A.data(), MPI_INT,
                 localA.data(), local_rows * K, MPI_INT,
                 0, MPI_COMM_WORLD);

    // --- Broadcast B: all ranks receive full B matrix (aligned with DACPP Bcast) ---
    MPI_Bcast(dataB.data(), K * N, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Compute C = localA * B ---
    std::vector<int> local_result(local_count, 0);

    if (local_rows > 0) {
        sycl::queue q{sycl::default_selector_v};

        sycl::buffer<int, 1> bufA(localA.data(), sycl::range<1>(local_rows * K));
        sycl::buffer<int, 1> bufB(dataB.data(), sycl::range<1>(K * N));
        sycl::buffer<int, 1> bufC(local_result.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto a = bufA.get_access<sycl::access::mode::read>(h);
            auto b = bufB.get_access<sycl::access::mode::read>(h);
            auto c = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(local_rows, N), [=](sycl::id<2> idx) {
                const int local_i = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                int sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += a[local_i * K + k] * b[k * N + j];
                }
                c[local_i * N + j] = sum;
            });
        });
        q.wait();
    }

    // --- Gather results to rank 0 ---
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int lr = (M / size) + (r < rem ? 1 : 0);
        counts[r] = lr * N;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_result;
    if (rank == 0) global_result.resize(M * N);

    MPI_Gatherv(local_result.data(), local_count, MPI_INT,
                rank == 0 ? global_result.data() : nullptr,
                counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "{";
        for (int i = 0; i < M; ++i) {
            std::cout << "{";
            for (int j = 0; j < N; ++j) {
                std::cout << global_result[i * N + j];
                if (j < N - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (i < M - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
