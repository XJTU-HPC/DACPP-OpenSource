



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


const double A = 1.0;
const double D = 0.1;
const double dx = 0.1;
const double dt = 0.01;

#ifndef MDP_N
#define MDP_N 1000000
#endif

#ifndef MDP_T
#define MDP_T 20000
#endif

constexpr int N = MDP_N;
constexpr int T = MDP_T;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    const int interior = N - 2;
    if (size > interior) {
        if (rank == 0) std::cerr << "MPI size too large\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int base = interior / size;
    const int rem  = interior % size;
    const int local_count = base + (rank < rem ? 1 : 0);
    const int local_begin = 1 + rank * base + std::min(rank, rem);

    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;


    const int local_ext = local_count + 2;
    std::vector<double> p(local_ext, 0.0);
    std::vector<double> new_p(local_ext, 0.0);


    for (int li = 0; li < local_count; ++li) {
        const int gi = local_begin + li;
        double x = gi * dx;
        p[li + 1] = std::exp(-std::pow(x - 5.0, 2) / 2.0);
    }

    sycl::queue q{sycl::default_selector_v};

    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Barrier(MPI_COMM_WORLD);
    }
        __baseline_test::reset_communication_time();
    const double __baseline_test_total_start = MPI_Wtime();
const double t0 = MPI_Wtime();

    for (int t = 0; t < T; ++t) {

        double send_left  = p[1];
        double send_right = p[local_count];
        double recv_left  = 0.0;
        double recv_right = 0.0;

        {
            __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
        MPI_Sendrecv(&send_left,  1, MPI_DOUBLE, prev, 0,
                     &recv_right, 1, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        {
            __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
        MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, next, 1,
                     &recv_left,  1, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (prev != MPI_PROC_NULL) p[0] = recv_left;
        if (next != MPI_PROC_NULL) p[local_count + 1] = recv_right;


        {
            sycl::buffer<double, 1> buf_p(p.data(), sycl::range<1>(local_ext));
            sycl::buffer<double, 1> buf_np(new_p.data(), sycl::range<1>(local_ext));

            q.submit([&](sycl::handler& h) {
                auto acc_p  = buf_p.get_access<sycl::access::mode::read>(h);
                auto acc_np = buf_np.get_access<sycl::access::mode::write>(h);

                h.parallel_for<class mdp_MPI_StandardSycl_test_kernel_1>(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                    const int li = static_cast<int>(idx[0]) + 1;
                    double diffusion = D * (acc_p[li + 1] - 2 * acc_p[li] + acc_p[li - 1]) / (dx * dx);
                    double drift = -A * (acc_p[li + 1] - acc_p[li - 1]) / (2 * dx);
                    acc_np[li] = acc_p[li] + dt * (diffusion + drift);
                });
            });
            q.wait();
        }

        std::swap(p, new_p);
    }

    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Barrier(MPI_COMM_WORLD);
    }
    const double elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }


    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (interior / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> local_out(p.begin() + 1, p.begin() + 1 + local_count);

    std::vector<double> global_p;
    if (rank == 0) global_p.resize(N, 0.0);

    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Gatherv(local_out.data(), local_count, MPI_DOUBLE,
                rank == 0 ? global_p.data() + 1 : nullptr,
                counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

        MPI_Barrier(MPI_COMM_WORLD);
    const double __baseline_test_total_time = MPI_Wtime() - __baseline_test_total_start;
    __baseline_test::print_summary("mdp.MPI_StandardSycl_test.cpp", rank, size, __baseline_test_total_time);
if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mdp] seconds=" << max_elapsed << std::endl;


    }

    MPI_Finalize();
    return 0;
}
