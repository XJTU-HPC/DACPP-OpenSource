#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef FOULA_N
#define FOULA_N 100
#endif

#ifndef FOULA_M
#define FOULA_M 5
#endif

#ifndef FOULA_R
#define FOULA_R 0.25
#endif

class FOuLaNaiveKernel;

namespace {

constexpr int n = FOULA_N;
constexpr int m = FOULA_M;
constexpr int interior = m - 1;
constexpr double r_coeff = FOULA_R;

double phi(double x) {
    return x * x * x + x;
}

double alpha(double) {
    return 0.0;
}

double beta(double t) {
    return 1.0 + std::exp(t);
}

int items_for_rank(int rank, int size) {
    const int base = interior / size;
    const int rem = interior % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int rank, int size) {
    const int base = interior / size;
    const int rem = interior % size;
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
    const int output_begin = item_begin_for_rank(rank, size);
    const int global_begin = 1 + output_begin;
    const double h = 1.0 / m;
    const double tau = 1.0 / n;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int rr = 0; rr < size; ++rr) {
        counts[rr] = items_for_rank(rr, size);
        displs[rr] = offset;
        offset += counts[rr];
    }

    std::vector<double> current(m + 1, 0.0);
    std::vector<double> next(m + 1, 0.0);
    for (int i = 0; i <= m; ++i) {
        current[i] = phi(i * h);
    }

    std::vector<double> row1_history(n + 1, 0.0);
    row1_history[0] = current[1];
    std::vector<double> local_next(local_count, 0.0);
    std::vector<double> global_next(interior, 0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int k = 0; k <= n - 1; ++k) {
        std::fill(local_next.begin(), local_next.end(), 0.0);

        if (local_count > 0) {
            sycl_compat::buffer<double, 1> current_buf(
                current.data(), sycl_compat::range<1>(current.size()));
            sycl_compat::buffer<double, 1> out_buf(
                local_next.data(), sycl_compat::range<1>(local_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto in =
                    current_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);
                h.parallel_for<FOuLaNaiveKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        const int gi = global_begin + static_cast<int>(idx[0]);
                        out[idx] = r_coeff * in[gi - 1] +
                                   (1.0 - 2.0 * r_coeff) * in[gi] +
                                   r_coeff * in[gi + 1];
                    });
            });
            q.wait();
        }

        MPI_Allgatherv(local_next.data(),
                       local_count,
                       MPI_DOUBLE,
                       global_next.data(),
                       counts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        next[0] = alpha((k + 1) * tau);
        next[m] = beta((k + 1) * tau);
        for (int i = 1; i <= m - 1; ++i) {
            next[i] = global_next[i - 1];
        }
        row1_history[k + 1] = next[1];
        std::swap(current, next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][foula][naive] seconds=" << max_seconds
                  << std::endl;
        std::cout << "{";
        for (int i = 0; i <= n; ++i) {
            std::cout << row1_history[i];
            if (i < n) {
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
        std::cerr << "[MPI_StandardSycl][foula][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
