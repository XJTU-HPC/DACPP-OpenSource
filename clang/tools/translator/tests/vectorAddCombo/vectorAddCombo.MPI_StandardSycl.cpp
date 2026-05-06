// Vector Add Combo — MPI + SYCL standard implementation
// Three chained vector additions: tmp = a + b, shifted = tmp + bias, out = shifted + c
// Element-parallel: split elements across ranks.

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef VADD_N
#define VADD_N 8
#endif

constexpr int N = VADD_N;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Full input data (small N, replicate on all ranks)
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> c{100, 200, 300, 400, 500, 600, 700, 800};
    float bias_val = 0.5f;

    // Divide elements across ranks
    const int base_count = N / size;
    const int rem = N % size;
    const int local_count = base_count + (rank < rem ? 1 : 0);
    const int local_begin = rank * base_count + std::min(rank, rem);

    // Extract local portions
    std::vector<float> local_a(a.begin() + local_begin, a.begin() + local_begin + local_count);
    std::vector<float> local_b(b.begin() + local_begin, b.begin() + local_begin + local_count);
    std::vector<float> local_c(c.begin() + local_begin, c.begin() + local_begin + local_count);
    std::vector<float> local_tmp(local_count, 0.0f);
    std::vector<float> local_shifted(local_count, 0.0f);
    std::vector<float> local_out(local_count, 0.0f);

    if (local_count > 0) {
        sycl::queue q{sycl::default_selector_v};

        sycl::buffer<float, 1> buf_a(local_a.data(), sycl::range<1>(local_count));
        sycl::buffer<float, 1> buf_b(local_b.data(), sycl::range<1>(local_count));
        sycl::buffer<float, 1> buf_c(local_c.data(), sycl::range<1>(local_count));
        sycl::buffer<float, 1> buf_tmp(local_tmp.data(), sycl::range<1>(local_count));
        sycl::buffer<float, 1> buf_shifted(local_shifted.data(), sycl::range<1>(local_count));
        sycl::buffer<float, 1> buf_out(local_out.data(), sycl::range<1>(local_count));

        // Step 1: tmp = a + b
        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_tmp = buf_tmp.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                acc_tmp[idx] = acc_a[idx] + acc_b[idx];
            });
        });

        // Step 2: shifted = tmp + bias
        q.submit([&](sycl::handler& h) {
            auto acc_tmp = buf_tmp.get_access<sycl::access::mode::read>(h);
            auto acc_shifted = buf_shifted.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                acc_shifted[idx] = acc_tmp[idx] + bias_val;
            });
        });

        // Step 3: out = shifted + c
        q.submit([&](sycl::handler& h) {
            auto acc_shifted = buf_shifted.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::read>(h);
            auto acc_out = buf_out.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                acc_out[idx] = acc_shifted[idx] + acc_c[idx];
            });
        });

        q.wait();
    }

    // Gather output to rank 0
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (N / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<float> global_out;
    if (rank == 0) global_out.resize(N);

    MPI_Gatherv(local_out.data(), local_count, MPI_FLOAT,
                rank == 0 ? global_out.data() : nullptr,
                counts.data(), displs.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (float v : global_out) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
