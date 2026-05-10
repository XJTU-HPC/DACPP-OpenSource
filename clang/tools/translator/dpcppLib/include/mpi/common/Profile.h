#ifndef DACPP_MPI_PROFILE_H
#define DACPP_MPI_PROFILE_H

#include <array>
#include <atomic>
#include <chrono>
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

enum class ProfileSegment : int {
    Init = 0,
    Scatter,
    Pack,
    Kernel,
    Halo,
    Gather,
    Bcast,
    Materialize,
    FinalSync,
    Count
};

inline constexpr int profileSegmentCount() {
    return static_cast<int>(ProfileSegment::Count);
}

inline const char* profileSegmentName(ProfileSegment segment) {
    switch (segment) {
    case ProfileSegment::Init:
        return "init";
    case ProfileSegment::Scatter:
        return "scatter";
    case ProfileSegment::Pack:
        return "pack";
    case ProfileSegment::Kernel:
        return "kernel";
    case ProfileSegment::Halo:
        return "halo";
    case ProfileSegment::Gather:
        return "gather";
    case ProfileSegment::Bcast:
        return "bcast";
    case ProfileSegment::Materialize:
        return "materialize";
    case ProfileSegment::FinalSync:
        return "final_sync";
    case ProfileSegment::Count:
        break;
    }
    return "unknown";
}

struct SegmentedProfile {
    std::array<long long, profileSegmentCount()> calls{};
    std::array<long long, profileSegmentCount()> nanos{};
};

inline bool profilingEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DACPP_MPI_PROFILE");
        return value && std::string(value) != "0";
    }();
    return enabled;
}

inline std::chrono::steady_clock::time_point profileSegmentStart() {
    return profilingEnabled() ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
}

inline void addProfileSegmentNanos(SegmentedProfile& profile,
                                   ProfileSegment segment,
                                   long long nanos) {
    if (!profilingEnabled()) {
        return;
    }
    const int idx = static_cast<int>(segment);
    if (idx < 0 || idx >= profileSegmentCount()) {
        return;
    }
    profile.calls[static_cast<std::size_t>(idx)] += 1;
    profile.nanos[static_cast<std::size_t>(idx)] += nanos;
}

inline void recordProfileSegment(
    SegmentedProfile& profile,
    ProfileSegment segment,
    std::chrono::steady_clock::time_point start) {
    if (!profilingEnabled()) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const long long nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                end - start)
                                .count();
    addProfileSegmentNanos(profile, segment, nanos);
}

inline void reportSegmentedProfile(const char* label,
                                   const SegmentedProfile& profile,
                                   MPI_Comm comm = MPI_COMM_WORLD) {
    if (!profilingEnabled()) {
        return;
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::array<long long, profileSegmentCount()> total_calls{};
    std::array<long long, profileSegmentCount()> sum_nanos{};
    std::array<long long, profileSegmentCount()> max_nanos{};

    MPI_Reduce(profile.calls.data(), total_calls.data(), profileSegmentCount(),
               MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(profile.nanos.data(), sum_nanos.data(), profileSegmentCount(),
               MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(profile.nanos.data(), max_nanos.data(), profileSegmentCount(),
               MPI_LONG_LONG, MPI_MAX, 0, comm);

    if (rank != 0) {
        return;
    }

    const char* profile_label = label ? label : "wrapper";
    for (int idx = 0; idx < profileSegmentCount(); ++idx) {
        if (total_calls[static_cast<std::size_t>(idx)] == 0) {
            continue;
        }
        const auto segment = static_cast<ProfileSegment>(idx);
        std::fprintf(
            stderr,
            "DACPP_MPI_PROFILE\t%s\t%s\tcalls=%lld\tmax_ms=%.6f\tsum_ms=%.6f\n",
            profile_label,
            profileSegmentName(segment),
            total_calls[static_cast<std::size_t>(idx)],
            static_cast<double>(max_nanos[static_cast<std::size_t>(idx)]) /
                1.0e6,
            static_cast<double>(sum_nanos[static_cast<std::size_t>(idx)]) /
                1.0e6);
    }
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
