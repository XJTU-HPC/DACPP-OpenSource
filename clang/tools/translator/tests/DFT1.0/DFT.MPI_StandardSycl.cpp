#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef DFT_N
#define DFT_N 8
#endif

class DFTNaiveKernel;

namespace {

constexpr int N = DFT_N;
constexpr double kPi = 3.141592653589793238462643383279502884;

int items_for_rank(int rank, int size) {
    const int base = N / size;
    const int rem = N % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int rank, int size) {
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

    const int local_count = items_for_rank(rank, size);
    const int local_begin = item_begin_for_rank(rank, size);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = items_for_rank(r, size);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> in_real(N, 0.0);
    std::vector<double> in_imag(N, 0.0);
    for (int n = 0; n < N; ++n) {
        in_real[n] = static_cast<double>(n);
    }

    std::vector<double> local_out_real(local_count, 0.0);
    std::vector<double> local_out_imag(local_count, 0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    if (local_count > 0) {
        sycl_compat::buffer<double, 1> in_r_buf(
            in_real.data(), sycl_compat::range<1>(in_real.size()));
        sycl_compat::buffer<double, 1> in_i_buf(
            in_imag.data(), sycl_compat::range<1>(in_imag.size()));
        sycl_compat::buffer<double, 1> out_r_buf(
            local_out_real.data(), sycl_compat::range<1>(local_out_real.size()));
        sycl_compat::buffer<double, 1> out_i_buf(
            local_out_imag.data(), sycl_compat::range<1>(local_out_imag.size()));

        q.submit([&](sycl_compat::handler& h) {
            auto in_r = in_r_buf.get_access<sycl_compat::access::mode::read>(h);
            auto in_i = in_i_buf.get_access<sycl_compat::access::mode::read>(h);
            auto out_r =
                out_r_buf.get_access<sycl_compat::access::mode::write>(h);
            auto out_i =
                out_i_buf.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<DFTNaiveKernel>(
                sycl_compat::range<1>(static_cast<std::size_t>(local_count)),
                [=](sycl_compat::id<1> idx) {
                    const int k = local_begin + static_cast<int>(idx[0]);
                    double sum_r = 0.0;
                    double sum_i = 0.0;
                    for (int n = 0; n < N; ++n) {
                        const double angle =
                            -2.0 * kPi * static_cast<double>(k) *
                            static_cast<double>(n) / static_cast<double>(N);
                        const double c = sycl_compat::cos(angle);
                        const double s = sycl_compat::sin(angle);
                        sum_r += in_r[n] * c - in_i[n] * s;
                        sum_i += in_r[n] * s + in_i[n] * c;
                    }
                    out_r[idx] = sum_r;
                    out_i[idx] = sum_i;
                });
        });
        q.wait();
    }

    std::vector<double> global_out_real(N, 0.0);
    std::vector<double> global_out_imag(N, 0.0);
    MPI_Gatherv(local_out_real.data(),
                local_count,
                MPI_DOUBLE,
                global_out_real.data(),
                counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
    MPI_Gatherv(local_out_imag.data(),
                local_count,
                MPI_DOUBLE,
                global_out_imag.data(),
                counts.data(),
                displs.data(),
                MPI_DOUBLE,
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
        std::cerr << "[MPI_StandardSycl][dft][naive] seconds=" << max_seconds
                  << std::endl;
        std::cout << "{";
        for (int k = 0; k < N; ++k) {
            std::cout << "(" << global_out_real[k] << "," << global_out_imag[k]
                      << ")";
            if (k + 1 < N) {
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
        std::cerr << "[MPI_StandardSycl][dft][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
