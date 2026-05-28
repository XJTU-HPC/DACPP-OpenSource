#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef ODDEVEN_N
#define ODDEVEN_N 8
#endif

class OddEvenEvenPairKernel;
class OddEvenOddPairKernel;

namespace {

constexpr int N = ODDEVEN_N;

int items_for_rank(int total, int rank, int size) {
    const int base = total / size;
    const int rem = total % size;
    return base + (rank < rem ? 1 : 0);
}

int item_begin_for_rank(int total, int rank, int size) {
    const int base = total / size;
    const int rem = total % size;
    return rank * base + std::min(rank, rem);
}

void fill_counts(int total, int size, std::vector<int>& counts,
                 std::vector<int>& displs, int width) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = items_for_rank(total, r, size) * width;
        displs[r] = offset;
        offset += counts[r];
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> global_arr(N, 0);
    for (int i = 0; i < N; ++i) {
        global_arr[i] = N - i;
    }

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int phase = 0; phase < N; ++phase) {
        const int even_pairs = N / 2;
        const int local_even_pairs = items_for_rank(even_pairs, rank, size);
        const int even_pair_begin =
            item_begin_for_rank(even_pairs, rank, size);
        std::vector<int> local_even(static_cast<std::size_t>(local_even_pairs) *
                                        2,
                                    0);

        if (local_even_pairs > 0) {
            sycl_compat::buffer<int, 1> arr_buf(
                global_arr.data(), sycl_compat::range<1>(global_arr.size()));
            sycl_compat::buffer<int, 1> out_buf(
                local_even.data(), sycl_compat::range<1>(local_even.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto arr = arr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);
                h.parallel_for<OddEvenEvenPairKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_even_pairs)),
                    [=](sycl_compat::id<1> idx) {
                        const int pair = even_pair_begin +
                                         static_cast<int>(idx[0]);
                        const int left_i = pair * 2;
                        const int a = arr[left_i];
                        const int b = arr[left_i + 1];
                        out[static_cast<std::size_t>(idx[0]) * 2] =
                            a <= b ? a : b;
                        out[static_cast<std::size_t>(idx[0]) * 2 + 1] =
                            a <= b ? b : a;
                    });
            });
            q.wait();
        }

        std::vector<int> counts;
        std::vector<int> displs;
        fill_counts(even_pairs, size, counts, displs, 2);
        std::vector<int> after_even(static_cast<std::size_t>(even_pairs) * 2,
                                    0);
        MPI_Allgatherv(local_even.data(),
                       static_cast<int>(local_even.size()),
                       MPI_INT,
                       after_even.data(),
                       counts.data(),
                       displs.data(),
                       MPI_INT,
                       MPI_COMM_WORLD);
        for (int i = 0; i < even_pairs * 2; ++i) {
            global_arr[i] = after_even[i];
        }

        const int odd_pairs = (N - 1) / 2;
        const int local_odd_pairs = items_for_rank(odd_pairs, rank, size);
        const int odd_pair_begin =
            item_begin_for_rank(odd_pairs, rank, size);
        std::vector<int> local_odd(static_cast<std::size_t>(local_odd_pairs) *
                                       2,
                                   0);

        if (local_odd_pairs > 0) {
            sycl_compat::buffer<int, 1> arr_buf(
                global_arr.data(), sycl_compat::range<1>(global_arr.size()));
            sycl_compat::buffer<int, 1> out_buf(
                local_odd.data(), sycl_compat::range<1>(local_odd.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto arr = arr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);
                h.parallel_for<OddEvenOddPairKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_odd_pairs)),
                    [=](sycl_compat::id<1> idx) {
                        const int pair =
                            odd_pair_begin + static_cast<int>(idx[0]);
                        const int left_i = 1 + pair * 2;
                        const int a = arr[left_i];
                        const int b = arr[left_i + 1];
                        out[static_cast<std::size_t>(idx[0]) * 2] =
                            a <= b ? a : b;
                        out[static_cast<std::size_t>(idx[0]) * 2 + 1] =
                            a <= b ? b : a;
                    });
            });
            q.wait();
        }

        fill_counts(odd_pairs, size, counts, displs, 2);
        std::vector<int> odd_values(static_cast<std::size_t>(odd_pairs) * 2,
                                    0);
        MPI_Allgatherv(local_odd.data(),
                       static_cast<int>(local_odd.size()),
                       MPI_INT,
                       odd_values.data(),
                       counts.data(),
                       displs.data(),
                       MPI_INT,
                       MPI_COMM_WORLD);
        for (int p = 0; p < odd_pairs; ++p) {
            global_arr[1 + p * 2] = odd_values[static_cast<std::size_t>(p) * 2];
            global_arr[2 + p * 2] =
                odd_values[static_cast<std::size_t>(p) * 2 + 1];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][oddeven][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "{";
        for (int i = 0; i < N; ++i) {
            std::cout << global_arr[i];
            if (i + 1 < N) {
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
        std::cerr << "[MPI_StandardSycl][oddeven][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
