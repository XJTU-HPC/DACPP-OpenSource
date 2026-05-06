#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef DFT_N
#define DFT_N 8
#endif

constexpr int N = DFT_N;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Prepare full input on all ranks (small N, cheap to replicate)
    std::vector<double> in_real(N), in_imag(N);
    for (int n = 0; n < N; ++n) {
        in_real[n] = static_cast<double>(n);
        in_imag[n] = 0.0;
    }

    // Divide output frequencies across ranks
    const int base = N / size;
    const int rem  = N % size;
    const int local_k_count = base + (rank < rem ? 1 : 0);
    const int k_begin = rank * base + std::min(rank, rem);

    std::vector<double> local_out_real(local_k_count, 0.0);
    std::vector<double> local_out_imag(local_k_count, 0.0);

    if (local_k_count > 0) {
        sycl::queue q{sycl::default_selector_v};

        sycl::buffer<double, 1> buf_in_r(in_real.data(), sycl::range<1>(N));
        sycl::buffer<double, 1> buf_in_i(in_imag.data(), sycl::range<1>(N));
        sycl::buffer<double, 1> buf_out_r(local_out_real.data(), sycl::range<1>(local_k_count));
        sycl::buffer<double, 1> buf_out_i(local_out_imag.data(), sycl::range<1>(local_k_count));

        q.submit([&](sycl::handler& h) {
            auto a_in_r = buf_in_r.get_access<sycl::access::mode::read>(h);
            auto a_in_i = buf_in_i.get_access<sycl::access::mode::read>(h);
            auto a_out_r = buf_out_r.get_access<sycl::access::mode::write>(h);
            auto a_out_i = buf_out_i.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(local_k_count), [=](sycl::id<1> idx) {
                int k = k_begin + static_cast<int>(idx[0]);
                double sum_r = 0.0;
                double sum_i = 0.0;
                for (int n = 0; n < N; ++n) {
                    double angle = -2.0 * M_PI * k * n / static_cast<double>(N);
                    double c = sycl::cos(angle);
                    double s = sycl::sin(angle);
                    double a = a_in_r[n];
                    double b = a_in_i[n];
                    sum_r += a * c - b * s;
                    sum_i += a * s + b * c;
                }
                a_out_r[idx] = sum_r;
                a_out_i[idx] = sum_i;
            });
        });
        q.wait();
    }

    // Gather results to rank 0
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (N / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> global_out_real(N), global_out_imag(N);
    MPI_Gatherv(local_out_real.data(), local_k_count, MPI_DOUBLE,
                global_out_real.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_out_imag.data(), local_k_count, MPI_DOUBLE,
                global_out_imag.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "{";
        for (int k = 0; k < N; ++k) {
            std::cout << "(" << global_out_real[k] << "," << global_out_imag[k] << ")";
            if (k < N - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
