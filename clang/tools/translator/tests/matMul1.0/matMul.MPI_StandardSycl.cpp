// Matrix Multiplication — MPI + SYCL standard implementation
// Split output rows across ranks. Each rank computes its portion of C = A * B.
// A is 4x5, B is 5x4 (column-major), C is 4x4.

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

    // A (4x5) row-major
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

    // Split rows of C across ranks
    const int base = M / size;
    const int rem  = M % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_begin = rank * base + std::min(rank, rem);
    const int local_count = local_rows * N;

    std::vector<int> local_result(local_count, 0);

    if (local_rows > 0) {
        sycl::queue q{sycl::default_selector_v};

        sycl::buffer<int, 1> bufA(dataA.data(), sycl::range<1>(M * K));
        sycl::buffer<int, 1> bufB(dataB.data(), sycl::range<1>(K * N));
        sycl::buffer<int, 1> bufC(local_result.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto a = bufA.get_access<sycl::access::mode::read>(h);
            auto b = bufB.get_access<sycl::access::mode::read>(h);
            auto c = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(local_rows, N), [=](sycl::id<2> idx) {
                const int local_i = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                const int global_i = row_begin + local_i;
                int sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += a[global_i * K + k] * b[k * N + j];
                }
                c[local_i * N + j] = sum;
            });
        });
        q.wait();
    }

    // Gather results to rank 0
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
