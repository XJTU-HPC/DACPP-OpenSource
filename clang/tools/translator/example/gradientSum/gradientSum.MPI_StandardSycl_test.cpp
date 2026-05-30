#include <sycl/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = sycl;

#ifndef GRADIENT_NUM_NEURONS
#define GRADIENT_NUM_NEURONS 8192
#endif

#ifndef GRADIENT_INPUT_SIZE
#define GRADIENT_INPUT_SIZE 8192
#endif

class GradientSumNaiveKernel;

namespace {

constexpr int NUM_NEURONS = GRADIENT_NUM_NEURONS;
constexpr int INPUT_SIZE = GRADIENT_INPUT_SIZE;

int rows_for_rank(int rank, int size) {
    const int base = NUM_NEURONS / size;
    const int rem = NUM_NEURONS % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = NUM_NEURONS / size;
    const int rem = NUM_NEURONS % size;
    return rank * base + std::min(rank, rem);
}

}

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local_count = rows_for_rank(rank, size);
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<int> payload_counts(size, 0);
    std::vector<int> payload_displs(size, 0);
    int offset = 0;
    int payload_offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size);
        displs[r] = offset;
        payload_counts[r] = counts[r] * INPUT_SIZE;
        payload_displs[r] = payload_offset;
        offset += counts[r];
        payload_offset += payload_counts[r];
    }

    std::vector<float> global_grads;
    if (rank == 0) {
        global_grads.resize(static_cast<std::size_t>(NUM_NEURONS) * INPUT_SIZE);
        for (int i = 0; i < NUM_NEURONS; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                global_grads[static_cast<std::size_t>(i) * INPUT_SIZE + j] =
                    static_cast<float>(i + j);
            }
        }
    }

    std::vector<float> local_grads(static_cast<std::size_t>(local_count) *
                                   INPUT_SIZE);
    std::vector<float> local_sum(local_count, 0.0f);
    std::vector<float> global_sum;
    if (rank == 0) {
        global_sum.resize(NUM_NEURONS, 0.0f);
    }

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Scatterv(rank == 0 ? global_grads.data() : nullptr,
                 payload_counts.data(),
                 payload_displs.data(),
                 MPI_FLOAT,
                 local_grads.data(),
                 static_cast<int>(local_grads.size()),
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    if (local_count > 0) {
        sycl_compat::buffer<float, 1> grad_buf(
            local_grads.data(), sycl_compat::range<1>(local_grads.size()));
        sycl_compat::buffer<float, 1> sum_buf(
            local_sum.data(), sycl_compat::range<1>(local_sum.size()));

        q.submit([&](sycl_compat::handler& h) {
            auto grads =
                grad_buf.get_access<sycl_compat::access::mode::read>(h);
            auto sums =
                sum_buf.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<GradientSumNaiveKernel>(
                sycl_compat::range<1>(static_cast<std::size_t>(local_count)),
                [=](sycl_compat::id<1> idx) {
                    int sum = 0;
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        sum += grads[static_cast<std::size_t>(idx[0]) *
                                         INPUT_SIZE + j];
                    }
                    sums[idx] = sum;
                });
        });
        q.wait();
    }
    MPI_Gatherv(local_sum.data(),
                local_count,
                MPI_FLOAT,
                rank == 0 ? global_sum.data() : nullptr,
                counts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const int print_count = std::min(5, NUM_NEURONS);
        std::cout << "First 5 neuron gradient sums:\n";
        for (int i = 0; i < print_count; ++i) {
            std::cout << global_sum[i] << " ";
        }
        std::cout << std::endl;
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
        std::cerr << "[MPI_StandardSycl][gradientSum][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
