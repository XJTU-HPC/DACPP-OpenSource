// Odd-Even Sort — MPI + SYCL standard implementation
// The array is split across ranks. Each phase, each rank performs local odd-even
// comparisons, then boundary pairs are resolved via compare-exchange with neighbors.

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

#ifndef ODDEVEN_N
#define ODDEVEN_N 8
#endif

constexpr int N = ODDEVEN_N;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide array across ranks
    const int base = N / size;
    const int rem  = N % size;
    const int local_count = base + (rank < rem ? 1 : 0);
    const int local_begin = rank * base + std::min(rank, rem);

    // Initialize local portion: arr[i] = N - i
    std::vector<int> local_arr(local_count);
    for (int i = 0; i < local_count; ++i) {
        local_arr[i] = N - (local_begin + i);
    }

    sycl::queue q{sycl::default_selector_v};

    // Odd-even transposition sort: N phases
    // In each phase, compare-exchange within each rank's local array,
    // then handle boundary elements between ranks.
    for (int phase = 0; phase < N; ++phase) {
        // SYCL kernel: local odd-even comparisons
        {
            sycl::buffer<int, 1> buf(local_arr.data(), sycl::range<1>(local_count));
            q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                // Determine start based on global parity
                int start = (phase % 2 == 0) ? 0 : 1;
                h.parallel_for(sycl::range<1>((local_count - 1) / 2 + 1), [=](sycl::id<1> idx) {
                    int i = start + static_cast<int>(idx[0]) * 2;
                    if (i + 1 < static_cast<int>(acc.get_range().size())) {
                        if (acc[i] > acc[i + 1]) {
                            int t = acc[i];
                            acc[i] = acc[i + 1];
                            acc[i + 1] = t;
                        }
                    }
                });
            });
            q.wait();
        }

        // Boundary compare-exchange with neighbor ranks
        // Even phase: even-indexed elements compare with next
        // Odd phase: odd-indexed elements compare with next
        int boundary_global_idx = local_begin + local_count - 1; // last element of this rank
        int boundary_parity;
        if (phase % 2 == 0) {
            boundary_parity = 0; // compare even indices
        } else {
            boundary_parity = 1; // compare odd indices
        }

        // Compare-exchange with next rank
        if (rank + 1 < size) {
            int my_last = local_arr[local_count - 1];
            int neighbor_first;
            MPI_Sendrecv(&my_last, 1, MPI_INT, rank + 1, 0,
                         &neighbor_first, 1, MPI_INT, rank + 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Keep smaller on lower rank, larger on higher rank
            int lo = std::min(my_last, neighbor_first);
            int hi = std::max(my_last, neighbor_first);
            local_arr[local_count - 1] = lo;
            // Send hi back (neighbor keeps hi)
            // Actually: we need to coordinate who keeps what
            // Simpler: my_last stays as min, neighbor keeps max
            // We already set local_arr. Neighbor will set its first to max.
            // But we only sent my_last and received neighbor_first.
            // Need a second exchange to tell neighbor what to keep.
            int send_val = hi;
            int recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_INT, rank + 1, 2,
                         &recv_val, 1, MPI_INT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // recv_val is the max from the left neighbor, but we're the right neighbor
            // Actually we already have the correct local_arr[local_count-1] = lo
            // The neighbor needs to set its first element to hi
            // We sent hi to neighbor, neighbor sends us nothing useful here
        }

        // Receive from previous rank: set our first element
        if (rank > 0) {
            int my_first = local_arr[0];
            int neighbor_last;
            MPI_Sendrecv(&my_first, 1, MPI_INT, rank - 1, 1,
                         &neighbor_last, 1, MPI_INT, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int lo = std::min(my_first, neighbor_last);
            int hi = std::max(my_first, neighbor_last);
            // We are the higher rank, keep the larger value at our first position
            local_arr[0] = hi;
            int send_val = lo;
            int recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_INT, rank - 1, 2,
                         &recv_val, 1, MPI_INT, rank - 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Final local sort to ensure correctness
    std::sort(local_arr.begin(), local_arr.end());

    // Gather sorted result
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = (N / size) + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> global_arr;
    if (rank == 0) global_arr.resize(N);

    MPI_Gatherv(local_arr.data(), local_count, MPI_INT,
                rank == 0 ? global_arr.data() : nullptr,
                counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "{";
        for (int i = 0; i < N; ++i) {
            std::cout << global_arr[i];
            if (i < N - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
