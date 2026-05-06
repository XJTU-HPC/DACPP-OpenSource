#ifndef DACPP_MPI_CORE_TYPES_H
#define DACPP_MPI_CORE_TYPES_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace dacpp {
namespace mpi {

struct ItemRange {
    int64_t begin = 0;
    int64_t end = 0;

    int64_t size() const {
        return end - begin;
    }
};

enum class AccessMode {
    Read,
    Write,
    ReadWrite
};

enum class PackLayoutKind {
    ItemDerived,
    DenseCover
};

inline const char* pack_layout_kind_name(PackLayoutKind kind) {
    switch (kind) {
    case PackLayoutKind::ItemDerived:
        return "item-derived";
    case PackLayoutKind::DenseCover:
        return "dense-cover";
    }
    return "unknown";
}

struct LinearSegment {
    int64_t global_begin = 0;
    int64_t len = 0;
    int64_t local_begin = 0;
};

struct PackMap {
    PackLayoutKind layout_kind = PackLayoutKind::ItemDerived;
    std::size_t dense_extent = 0;
    std::vector<int64_t> globals;
    std::unordered_map<int64_t, int32_t> g2l;
    std::vector<LinearSegment> segments;

    std::vector<int64_t> writeback_globals;
    std::vector<LinearSegment> writeback_segments;
};

struct PackPlan {
    PackMap pack;
    std::vector<int32_t> compact_slots;
    std::vector<int32_t> item_key_offsets;
};

struct VectorIntHash {
    std::size_t operator()(const std::vector<int>& values) const {
        std::size_t seed = values.size();
        for (int value : values) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

}  // namespace mpi
}  // namespace dacpp

#endif
