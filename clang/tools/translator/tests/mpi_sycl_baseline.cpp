#include <iostream>
#include <vector>
#include <mpi.h>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 16;
    int local_N = N / size;

    std::vector<float> A, B, C;
    if (rank == 0) {
        A.resize(N);
        B.resize(N);
        C.resize(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            A[i] = i * 1.0f;
            B[i] = i * 2.0f;
        }
    }

    std::vector<float> local_A(local_N);
    std::vector<float> local_B(local_N);
    std::vector<float> local_C(local_N, 0.0f);

    MPI_Scatter(rank == 0 ? A.data() : nullptr, local_N, MPI_FLOAT,
                local_A.data(), local_N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(rank == 0 ? B.data() : nullptr, local_N, MPI_FLOAT,
                local_B.data(), local_N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    {
        // SYCL compute
        sycl::queue q(sycl::default_selector_v);
        sycl::buffer<float, 1> bufA(local_A.data(), sycl::range<1>(local_N));
        sycl::buffer<float, 1> bufB(local_B.data(), sycl::range<1>(local_N));
        sycl::buffer<float, 1> bufC(local_C.data(), sycl::range<1>(local_N));

        q.submit([&](sycl::handler& h) {
            auto accA = bufA.get_access<sycl::access::mode::read>(h);
            auto accB = bufB.get_access<sycl::access::mode::read>(h);
            auto accC = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(local_N), [=](sycl::id<1> idx) {
                accC[idx] = accA[idx] + accB[idx];
            });
        });
        q.wait();
    }

    MPI_Gather(local_C.data(), local_N, MPI_FLOAT,
               rank == 0 ? C.data() : nullptr, local_N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool pass = true;
        for (int i = 0; i < N; ++i) {
            if (C[i] != i * 3.0f) {
                pass = false;
                break;
            }
        }
        if (pass) {
            std::cout << "[PASS] Baseline MPI+SYCL Vector Add produced correct results.\n";
        } else {
            std::cout << "[FAIL] Baseline MPI+SYCL Vector Add produced INCORRECT results.\n";
        }
        std::cout << "Result C: ";
        for (int i = 0; i < N; ++i) {
            std::cout << C[i] << " ";
        }
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
