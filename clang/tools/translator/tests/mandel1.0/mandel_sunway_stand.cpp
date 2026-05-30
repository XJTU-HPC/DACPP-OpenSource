// Mandelbrot set — MPI + SYCL standard implementation
// Data model aligned with the DACPP translated version: the input is treated as
// one logical global vector built on every rank, wrapped into a tensor-like
// abstraction (mirroring dacpp::Vector), and distributed from the root with the
// same block-cyclic layout the DAC path uses. This removes the zero-copy
// "every rank slices its own replicated array" shortcut so the comparison
// reflects the same data ownership/movement, not a free local slice.

#include <CL/sycl.hpp>
#include <mpi.h>
#include <cstdio>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

namespace __baseline_test {
struct ScopedCommunicationTimer {};

inline void print_e2e_summary(int rank, double total_time) {
    double total_max = 0.0;
    MPI_Reduce(&total_time, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mandel] e2e_seconds=" << total_max
                  << std::endl;
    }
}
}  // namespace __baseline_test


#ifndef MANDEL_N
#define MANDEL_N 8192
#endif

constexpr int row_count = MANDEL_N;
constexpr int col_count = MANDEL_N;
constexpr int max_iterations = 1000;

// Host-side vector that mirrors the DACPP Vector/Tensor data model the
// translated program builds. Construction takes the source vector by value and
// reallocates + copies element-by-element, exactly like
// dacpp::Vector(std::vector) (ReconTensor.h Tensor(std::vector)). This
// reproduces the real per-rank allocation/copy cost the DAC path pays when it
// wraps complex_points into a tensor.
template <typename T>
struct HostVector {
    std::vector<T> data;

    HostVector() = default;
    explicit HostVector(std::vector<T> init) {       // by value: copy #1
        data.resize(init.size());
        for (std::size_t i = 0; i < init.size(); ++i) {
            data[i] = init[i];                        // element-wise: copy #2
        }
    }

    std::size_t size() const { return data.size(); }
    const T* ptr() const { return data.data(); }
};

// Canonical 1D block-cyclic layout, matching the DAC runtime
// (block_cyclic_global_index_1d / block_cyclic_count_1d) with block size 64.
constexpr long long kBlockSize = 64;

static long long bc_global_index(long long local_i, int r, int size,
                                 long long block) {
    const long long m = local_i / block;
    const long long o = local_i % block;
    return (m * static_cast<long long>(size) + r) * block + o;
}

static long long bc_count(long long total, int r, int size, long long block) {
    const long long nblocks = (total + block - 1) / block;
    long long count = 0;
    for (long long blk = r; blk < nblocks; blk += size) {
        const long long start = blk * block;
        count += std::min(block, total - start);
    }
    return count;
}

int main(int argc, char** argv) {
    const auto e2e_start = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long total_points =
        static_cast<long long>(row_count) * col_count;

    // Initialize the full complex-point field on every rank, exactly like the
    // DAC program's unguarded InitializeComplexPoints().
    std::vector<std::complex<float>> complex_points(
        static_cast<std::size_t>(total_points));
    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            const std::size_t index =
                static_cast<std::size_t>(i) * col_count + j;
            const float real = -1.5f + (i * (2.0f / row_count));
            const float imag = -1.0f + (j * (2.0f / col_count));
            complex_points[index] = std::complex<float>(real, imag);
        }
    }

    // Wrap the input into the tensor-like abstraction (mirrors
    // dacpp::Vector complex_points_tensor(complex_points)).
    HostVector<std::complex<float>> complex_points_tensor(complex_points);

    // Block-cyclic partition, matching the DAC distribution choice.
    const long long local_count =
        bc_count(total_points, rank, size, kBlockSize);

    std::vector<int> counts(size), displs(size);
    long long offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(bc_count(total_points, r, size, kBlockSize));
        displs[r] = static_cast<int>(offset);
        offset += counts[r];
    }

    // Distribute the input from the root in block-cyclic order. The root packs
    // each rank's block-cyclic payload (the same scattered-read pack the DAC
    // path performs), then a single MPI_Scatterv hands each rank its portion,
    // instead of every rank slicing its own replicated copy for free.
    std::vector<std::complex<float>> local_points(
        static_cast<std::size_t>(local_count));
    std::vector<std::complex<float>> send_buffer;
    if (rank == 0) {
        send_buffer.resize(static_cast<std::size_t>(total_points));
        std::size_t pos = 0;
        for (int r = 0; r < size; ++r) {
            for (long long li = 0; li < counts[r]; ++li) {
                const long long gi =
                    bc_global_index(li, r, size, kBlockSize);
                send_buffer[pos++] =
                    complex_points_tensor.ptr()[static_cast<std::size_t>(gi)];
            }
        }
    }
    MPI_Scatterv(rank == 0 ? send_buffer.data() : nullptr,
                 counts.data(), displs.data(), MPI_C_FLOAT_COMPLEX,
                 local_points.data(), static_cast<int>(local_count),
                 MPI_C_FLOAT_COMPLEX, 0, MPI_COMM_WORLD);

    std::vector<int> local_flags(static_cast<std::size_t>(local_count), 0);

    if (local_count > 0) {
        sycl_compat::queue q{sycl_compat::default_selector_v};

        sycl_compat::buffer<std::complex<float>, 1> buf_pts(
            local_points.data(),
            sycl_compat::range<1>(static_cast<std::size_t>(local_count)));
        sycl_compat::buffer<int, 1> buf_flags(
            local_flags.data(),
            sycl_compat::range<1>(static_cast<std::size_t>(local_count)));

        q.submit([&](sycl_compat::handler& h) {
            auto pts   = buf_pts.get_access<sycl_compat::access::mode::read>(h);
            auto flags = buf_flags.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<class mandel_MPI_StandardSycl_test_kernel_1>(
                sycl_compat::range<1>(static_cast<std::size_t>(local_count)),
                [=](sycl_compat::id<1> idx) {
                std::complex<float> c = pts[idx];
                std::complex<float> z(0.0f, 0.0f);
                int it = 0;

                for (int i = 0; i < max_iterations; ++i) {
                    if (sycl_compat::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.0f) {
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

    // Count local in-set points and reduce the scalar to the root, mirroring the
    // DAC path's MPI_Reduce of the post-use count.
    long long local_in_set = 0;
    for (int f : local_flags) {
        if (f == 1) ++local_in_set;
    }
    long long mandelbrot_count = 0;
    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
        MPI_Reduce(&local_in_set, &mandelbrot_count, 1, MPI_LONG_LONG, MPI_SUM,
                   0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        // std::cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
        (void)mandelbrot_count;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto e2e_end = std::chrono::steady_clock::now();
    const double e2e_seconds =
        std::chrono::duration<double>(e2e_end - e2e_start).count();
    __baseline_test::print_e2e_summary(rank, e2e_seconds);

    MPI_Finalize();
    return 0;
}
