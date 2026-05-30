#include <CL/sycl.hpp>
#include <mpi.h>
#include <cstdio>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>



namespace __baseline_test {
inline double& communication_time_accumulator() {
    static double value = 0.0;
    return value;
}

struct ScopedCommunicationTimer {
    double start_time;

    ScopedCommunicationTimer() : start_time(MPI_Wtime()) {}

    ~ScopedCommunicationTimer() {
        communication_time_accumulator() += MPI_Wtime() - start_time;
    }
};

inline void reset_communication_time() {
    communication_time_accumulator() = 0.0;
}

inline double communication_time() {
    return communication_time_accumulator();
}

inline void print_summary(const char* label, int rank, int size, double total_time) {
    double communication = communication_time();
    double computation = total_time - communication;
    if (computation < 0.0) {
        computation = 0.0;
    }

    double total_max = 0.0;
    double total_sum = 0.0;
    double communication_max = 0.0;
    double communication_sum = 0.0;
    double computation_max = 0.0;
    double computation_sum = 0.0;

    MPI_Reduce(&total_time, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication, &communication_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication, &communication_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&computation, &computation_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&computation, &computation_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf(
            "[MPI TEST] %s | ranks=%d | total_max=%.6f s | total_avg=%.6f s | communication_max=%.6f s | communication_avg=%.6f s | computation_max=%.6f s | computation_avg=%.6f s\n",
            label,
            size,
            total_max,
            total_sum / static_cast<double>(size),
            communication_max,
            communication_sum / static_cast<double>(size),
            computation_max,
            computation_sum / static_cast<double>(size));
    }
}
}


#ifndef DFT_N
#define DFT_N 1048576
#endif

constexpr int N = DFT_N;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    std::vector<double> in_real(N), in_imag(N);
    for (int n = 0; n < N; ++n) {
        in_real[n] = static_cast<double>(n);
        in_imag[n] = 0.0;
    }


    const int base = N / size;
    const int rem  = N % size;
    const int local_k_count = base + (rank < rem ? 1 : 0);
    const int k_begin = rank * base + std::min(rank, rem);

    std::vector<double> local_out_real(local_k_count, 0.0);
    std::vector<double> local_out_imag(local_k_count, 0.0);

        __baseline_test::reset_communication_time();
    MPI_Barrier(MPI_COMM_WORLD);
    const double __baseline_test_total_start = MPI_Wtime();
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

            h.parallel_for<class DFT_MPI_StandardSycl_test_kernel_1>(sycl::range<1>(local_k_count), [=](sycl::id<1> idx) {
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


    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (N / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> global_out_real(N), global_out_imag(N);
    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Gatherv(local_out_real.data(), local_k_count, MPI_DOUBLE,
                global_out_real.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Gatherv(local_out_imag.data(), local_k_count, MPI_DOUBLE,
                global_out_imag.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

        MPI_Barrier(MPI_COMM_WORLD);
    const double __baseline_test_total_time = MPI_Wtime() - __baseline_test_total_start;
    __baseline_test::print_summary("DFT.MPI_StandardSycl_test.cpp", rank, size, __baseline_test_total_time);
if (rank == 0) {






    }

    MPI_Finalize();
    return 0;
}
