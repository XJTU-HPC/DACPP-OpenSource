// Mandelbrot set — MPI + SYCL standard implementation
// Element-parallel: split total_points across ranks, each rank computes its portion.

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef MANDEL_N
#define MANDEL_N 8
#endif

constexpr int row_count = MANDEL_N;
constexpr int col_count = MANDEL_N;
constexpr int max_iterations = 1000;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_points = row_count * col_count;

    // Initialize complex points on all ranks (small data, cheap to replicate)
    std::vector<std::complex<float>> complex_points(total_points);
    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;
            float real = -1.5f + (i * (2.0f / row_count));
            float imag = -1.0f + (j * (2.0f / col_count));
            complex_points[index] = std::complex<float>(real, imag);
        }
    }

    // Divide points across ranks
    const int base = total_points / size;
    const int rem  = total_points % size;
    const int local_count = base + (rank < rem ? 1 : 0);
    const int local_begin = rank * base + std::min(rank, rem);

    std::vector<int> local_flags(local_count, 0);

    if (local_count > 0) {
        sycl::queue q{sycl::default_selector_v};

        // Copy only the needed portion of complex_points to device
        std::vector<std::complex<float>> local_points(
            complex_points.begin() + local_begin,
            complex_points.begin() + local_begin + local_count);

        sycl::buffer<std::complex<float>, 1> buf_pts(local_points.data(), sycl::range<1>(local_count));
        sycl::buffer<int, 1> buf_flags(local_flags.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto pts   = buf_pts.get_access<sycl::access::mode::read>(h);
            auto flags = buf_flags.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                std::complex<float> c = pts[idx];
                std::complex<float> z(0.0f, 0.0f);
                int it = 0;

                for (int i = 0; i < max_iterations; ++i) {
                    if (sycl::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.0f) {
                        it = i;
                        break;
                    }
                    z = std::complex<float>(
                        z.real() * z.real() - z.imag() * z.imag() + c.real(),
                        2.0f * z.real() * z.imag() + c.imag());
                    it = max_iterations;
                }

                if (it == max_iterations) {
                    flags[idx] = 1;
                }
            });
        });
        q.wait();
    }

    // Gather flags to rank 0
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (total_points / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_flags;
    if (rank == 0) global_flags.resize(total_points, 0);

    MPI_Gatherv(local_flags.data(), local_count, MPI_INT,
                rank == 0 ? global_flags.data() : nullptr,
                counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int mandelbrot_count = 0;
        for (int f : global_flags) {
            if (f == 1) mandelbrot_count++;
        }
        std::cout << "Mandelbrot Set Statistics:\n";
        std::cout << "Total points: " << total_points << "\n";
        std::cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
    }

    MPI_Finalize();
    return 0;
}
