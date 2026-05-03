#ifndef DACPP_MPI_PROFILE_H
#define DACPP_MPI_PROFILE_H

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>

namespace dacpp {
namespace mpi {

struct CollectPositionsProfile {
    std::atomic<long long> calls{0};
    std::atomic<long long> nanos{0};
};

inline bool profilingEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DACPP_MPI_PROFILE");
        return value && std::string(value) != "0";
    }();
    return enabled;
}

inline CollectPositionsProfile& getCollectPositionsProfile() {
    static CollectPositionsProfile profile;
    return profile;
}

inline void resetCollectPositionsProfile() {
    auto& profile = getCollectPositionsProfile();
    profile.calls.store(0, std::memory_order_relaxed);
    profile.nanos.store(0, std::memory_order_relaxed);
}

inline void recordCollectPositionsSample(long long nanos) {
    if (!profilingEnabled()) {
        return;
    }
    auto& profile = getCollectPositionsProfile();
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    profile.nanos.fetch_add(nanos, std::memory_order_relaxed);
}

inline void reportCollectPositionsProfile(const char* label,
                                          MPI_Comm comm = MPI_COMM_WORLD) {
    if (!profilingEnabled()) {
        return;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto& profile = getCollectPositionsProfile();
    const long long local_calls = profile.calls.load(std::memory_order_relaxed);
    const long long local_nanos = profile.nanos.load(std::memory_order_relaxed);

    long long total_calls = 0;
    long long total_nanos = 0;
    MPI_Reduce(&local_calls, &total_calls, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_nanos, &total_nanos, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    std::vector<long long> all_nanos;
    if (rank == 0) {
        all_nanos.resize(size);
    }
    MPI_Gather(&local_nanos, 1, MPI_LONG_LONG,
               rank == 0 ? all_nanos.data() : nullptr,
               1, MPI_LONG_LONG, 0, comm);

    if (rank != 0) {
        return;
    }

    int max_rank = 0;
    long long max_nanos = size > 0 ? all_nanos[0] : 0;
    for (int idx = 1; idx < size; ++idx) {
        if (all_nanos[idx] > max_nanos) {
            max_nanos = all_nanos[idx];
            max_rank = idx;
        }
    }

    std::fprintf(stderr,
                 "[DACPP][PROFILE][%s] collect_positions_for_item total_calls(sum): %lld\n",
                 label ? label : "wrapper",
                 total_calls);
    std::fprintf(stderr,
                 "[DACPP][PROFILE][%s] collect_positions_for_item total_ms(sum): %.3f\n",
                 label ? label : "wrapper",
                 static_cast<double>(total_nanos) / 1.0e6);
    std::fprintf(stderr,
                 "[DACPP][PROFILE][%s] collect_positions_for_item max_rank_ms: %.3f (rank=%d)\n",
                 label ? label : "wrapper",
                 static_cast<double>(max_nanos) / 1.0e6,
                 max_rank);
}

}  // namespace mpi
}  // namespace dacpp

#endif
