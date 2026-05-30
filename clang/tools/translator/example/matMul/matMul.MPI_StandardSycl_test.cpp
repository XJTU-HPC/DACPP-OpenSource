








#include <CL/sycl.hpp>
#include <mpi.h>
#include <cstdio>

#include <algorithm>
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

template <typename AccessA, typename AccessB, typename AccessC>
struct MatMulSunwayKernel {
    AccessA a;
    AccessB b;
    AccessC c;
    int kernel_k;
    int kernel_n;

    MatMulSunwayKernel(AccessA a_acc, AccessB b_acc, AccessC c_acc, int k_dim, int n_dim)
        : a(a_acc), b(b_acc), c(c_acc), kernel_k(k_dim), kernel_n(n_dim) {}

    void operator()(sycl::id<2> idx) const {
        const int local_i = static_cast<int>(idx[0]);
        const int j = static_cast<int>(idx[1]);
        int sum = 0;
        for (int k = 0; k < kernel_k; ++k) {
            sum += a[local_i * kernel_k + k] * b[k * kernel_n + j];
        }
        c[local_i * kernel_n + j] = sum;
    }
};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int M = 8192;
    const int K = 8192;
    const int N = 8192;


    std::vector<int> dataA(M * K, 1);








    std::vector<int> dataB(K * N, 1);









    const int base = M / size;
    const int rem  = M % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_begin = rank * base + std::min(rank, rem);
    const int local_count = local_rows * N;

        __baseline_test::reset_communication_time();
    MPI_Barrier(MPI_COMM_WORLD);
    const double __baseline_test_total_start = MPI_Wtime();

    std::vector<int> counts_A(size), displs_A(size);
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        int rb = r * base + std::min(r, rem);
        counts_A[r] = lr * K;
        displs_A[r] = rb * K;
    }
    std::vector<int> localA(local_rows * K);
    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Scatterv(rank == 0 ? dataA.data() : nullptr,
                 counts_A.data(), displs_A.data(), MPI_INT,
                 localA.data(), local_rows * K, MPI_INT,
                 0, MPI_COMM_WORLD);
    }


    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Bcast(dataB.data(), K * N, MPI_INT, 0, MPI_COMM_WORLD);
    }


    std::vector<int> local_result(local_count, 0);

    if (local_rows > 0) {
        sycl::queue q{sycl::default_selector_v};
        const int kernel_local_rows = local_rows;
        const int kernel_k = K;
        const int kernel_n = N;

        sycl::buffer<int, 1> bufA(localA.data(), sycl::range<1>(local_rows * K));
        sycl::buffer<int, 1> bufB(dataB.data(), sycl::range<1>(K * N));
        sycl::buffer<int, 1> bufC(local_result.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto a = bufA.get_access<sycl::access::mode::read>(h);
            auto b = bufB.get_access<sycl::access::mode::read>(h);
            auto c = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class matMul_MPI_StandardSycl_test_kernel_1>(
                sycl::range<2>(kernel_local_rows, kernel_n),
                MatMulSunwayKernel<decltype(a), decltype(b), decltype(c)>(
                    a, b, c, kernel_k, kernel_n));
        });
        q.wait();
    }


    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int lr = (M / size) + (r < rem ? 1 : 0);
        counts[r] = lr * N;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_result;
    if (rank == 0) global_result.resize(M * N);

    {
        __baseline_test::ScopedCommunicationTimer __baseline_test_comm_timer;
    MPI_Gatherv(local_result.data(), local_count, MPI_INT,
                rank == 0 ? global_result.data() : nullptr,
                counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
    }

        MPI_Barrier(MPI_COMM_WORLD);
    const double __baseline_test_total_time = MPI_Wtime() - __baseline_test_total_start;
    __baseline_test::print_summary("matMul.MPI_StandardSycl_test.cpp", rank, size, __baseline_test_total_time);
if (rank == 0) {











    }

    MPI_Finalize();
    return 0;
}
