#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef MANDEL_N
#define MANDEL_N 8
#endif

#ifndef MANDEL_ROWS
#define MANDEL_ROWS MANDEL_N
#endif

#ifndef MANDEL_COLS
#define MANDEL_COLS MANDEL_N
#endif

#ifndef MANDEL_MAX_ITER
#define MANDEL_MAX_ITER 1000
#endif

class MandelNaiveKernel;

namespace {

constexpr int row_count = MANDEL_ROWS;
constexpr int col_count = MANDEL_COLS;
constexpr int max_iterations = MANDEL_MAX_ITER;
constexpr int total_points = row_count * col_count;

int items_for_rank(int rank, int size) {
    const int base = total_points / size;
    const int rem = total_points % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int rank, int size) {
    const int base = total_points / size;
    const int rem = total_points % size;
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

    const int local_count = items_for_rank(rank, size);
    const int local_begin = item_begin_for_rank(rank, size);

    std::vector<float> points_real(total_points, 0.0f);
    std::vector<float> points_imag(total_points, 0.0f);
    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            const int index = i * col_count + j;
            points_real[index] =
                -1.5f + (static_cast<float>(i) * (2.0f / row_count));
            points_imag[index] =
                -1.0f + (static_cast<float>(j) * (2.0f / col_count));
        }
    }

    std::vector<int> local_flags(local_count, 0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    if (local_count > 0) {
        sycl_compat::buffer<float, 1> real_buf(
            points_real.data(), sycl_compat::range<1>(points_real.size()));
        sycl_compat::buffer<float, 1> imag_buf(
            points_imag.data(), sycl_compat::range<1>(points_imag.size()));
        sycl_compat::buffer<int, 1> flags_buf(
            local_flags.data(), sycl_compat::range<1>(local_flags.size()));

        q.submit([&](sycl_compat::handler& h) {
            auto real = real_buf.get_access<sycl_compat::access::mode::read>(h);
            auto imag = imag_buf.get_access<sycl_compat::access::mode::read>(h);
            auto flags =
                flags_buf.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<MandelNaiveKernel>(
                sycl_compat::range<1>(static_cast<std::size_t>(local_count)),
                [=](sycl_compat::id<1> idx) {
                    const int global_i = local_begin + static_cast<int>(idx[0]);
                    const float c_real = real[global_i];
                    const float c_imag = imag[global_i];
                    float z_real = 0.0f;
                    float z_imag = 0.0f;
                    int iterations = 0;

                    for (int it = 0; it < max_iterations; ++it) {
                        if (sycl_compat::sqrt(z_real * z_real +
                                              z_imag * z_imag) > 2.0f) {
                            iterations = it;
                            break;
                        }
                        const float next_real =
                            z_real * z_real - z_imag * z_imag + c_real;
                        const float next_imag =
                            2.0f * z_real * z_imag + c_imag;
                        z_real = next_real;
                        z_imag = next_imag;
                        iterations = max_iterations;
                    }
                    flags[idx] = iterations == max_iterations ? 1 : 0;
                });
        });
        q.wait();
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = items_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_flags(total_points, 0);
    MPI_Gatherv(local_flags.data(),
                local_count,
                MPI_INT,
                global_flags.data(),
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
        int mandelbrot_count = 0;
        for (int flag : global_flags) {
            if (flag == 1) {
                ++mandelbrot_count;
            }
        }
        std::cerr << "[MPI_StandardSycl][mandel][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "Mandelbrot Set Statistics:\n";
        std::cout << "Total points: " << total_points << "\n";
        std::cout << "Points in the Mandelbrot set: " << mandelbrot_count
                  << "\n";
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
        std::cerr << "[MPI_StandardSycl][mandel][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
