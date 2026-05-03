#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
namespace sycl = cl::sycl;

const double dt = 0.1;
const double T = 5.0;
const size_t numIsotopes = 10;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t steps = static_cast<size_t>(T / dt);

    const size_t base_count = numIsotopes / static_cast<size_t>(size);
    const size_t remainder = numIsotopes % static_cast<size_t>(size);
    const size_t local_count =
        base_count + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_begin =
        static_cast<size_t>(rank) * base_count +
        std::min(static_cast<size_t>(rank), remainder);

    vector<double> local_lambdas(local_count);
    vector<double> local_N0s(local_count, 1000.0);
    vector<double> local_A(local_count, 0.0);

    for (size_t i = 0; i < local_count; ++i) {
        const size_t global_i = global_begin + i;
        local_lambdas[i] = 0.01 + 0.01 * global_i;
    }

    sycl::queue q{sycl::default_selector_v};

    sycl::buffer<double, 1> n0s_buf(local_N0s.data(), sycl::range<1>(local_count));
    sycl::buffer<double, 1> lambdas_buf(local_lambdas.data(), sycl::range<1>(local_count));
    sycl::buffer<double, 1> local_a_buf(local_A.data(), sycl::range<1>(local_count));

    vector<int> recv_counts;
    vector<int> recv_displs;
    vector<double> gathered_A;
    vector<double> A;

    if (rank == 0) {
        recv_counts.resize(static_cast<size_t>(size), 0);
        recv_displs.resize(static_cast<size_t>(size), 0);
        int displacement = 0;
        for (int r = 0; r < size; ++r) {
            const size_t r_count =
                base_count + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recv_counts[static_cast<size_t>(r)] = static_cast<int>(r_count);
            recv_displs[static_cast<size_t>(r)] = displacement;
            displacement += static_cast<int>(r_count);
        }

        gathered_A.assign(numIsotopes, 0.0);
        A.assign(steps * numIsotopes, 0.0);
    }

    double t = 0.0;
    size_t step = 0;

    while (t <= T) {
        const double current_t = t;

        q.submit([&](sycl::handler& h) {
            auto n0s_acc = n0s_buf.get_access<sycl::access::mode::read>(h);
            auto lambdas_acc = lambdas_buf.get_access<sycl::access::mode::read>(h);
            auto local_a_acc =
                local_a_buf.get_access<sycl::access::mode::discard_write>(h);

            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                local_a_acc[i] =
                    n0s_acc[i] * sycl::exp(-lambdas_acc[i] * current_t);
            });
        }).wait();

        {
            auto local_a_acc = local_a_buf.get_access<sycl::access::mode::read>();
            for (size_t i = 0; i < local_count; ++i) {
                local_A[i] = local_a_acc[i];
            }
        }

        MPI_Gatherv(local_A.data(),
                    static_cast<int>(local_count),
                    MPI_DOUBLE,
                    rank == 0 ? gathered_A.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? recv_displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        const size_t row = static_cast<size_t>(10.0 * current_t);
        if (rank == 0 && row < steps) {
            for (size_t i = 0; i < numIsotopes; ++i) {
                A[row * numIsotopes + i] = gathered_A[i];
            }
        }

        t += dt;
        ++step;
        if (step > steps) {
            break;
        }
    }

    if (rank == 0) {
        cout << "{";
        for (size_t i = 0; i < numIsotopes; ++i) {
            cout << A[1 * numIsotopes + i];
            if (i + 1 < numIsotopes) {
                cout << ", ";
            }
        }
        cout << "}" << endl;
    }

    MPI_Finalize();
    return 0;
}
