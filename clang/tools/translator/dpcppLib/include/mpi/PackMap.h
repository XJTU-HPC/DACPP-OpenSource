#ifndef DACPP_MPI_PACK_MAP_H
#define DACPP_MPI_PACK_MAP_H

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoreTypes.h"

namespace dacpp {
namespace mpi {

inline std::vector<LinearSegment> build_segments(const std::vector<int64_t>& globals) {
    std::vector<LinearSegment> segments;
    if (globals.empty()) {
        return segments;
    }

    LinearSegment cur{globals[0], 1, 0};
    for (std::size_t idx = 1; idx < globals.size(); ++idx) {
        if (globals[idx] == globals[idx - 1] + 1) {
            ++cur.len;
            continue;
        }
        segments.push_back(cur);
        cur = LinearSegment{globals[idx], 1, static_cast<int64_t>(idx)};
    }
    segments.push_back(cur);
    return segments;
}

inline PackMap make_pack_map_from_globals(
    std::vector<int64_t> globals,
    PackLayoutKind layout_kind = PackLayoutKind::ItemDerived,
    std::size_t dense_extent = 0) {
    std::sort(globals.begin(), globals.end());
    globals.erase(std::unique(globals.begin(), globals.end()), globals.end());

    PackMap pack;
    pack.layout_kind = layout_kind;
    pack.dense_extent = dense_extent;
    pack.globals = std::move(globals);
    pack.segments = build_segments(pack.globals);

    const bool fragmented =
        pack.segments.size() > 64 &&
        pack.segments.size() * 4 > pack.globals.size();
    if (fragmented) {
        pack.g2l.reserve(pack.globals.size());
        for (std::size_t idx = 0; idx < pack.globals.size(); ++idx) {
            pack.g2l.emplace(pack.globals[idx], static_cast<int32_t>(idx));
        }
    }
    return pack;
}

inline int32_t try_lookup_local_slot(const PackMap& pack, int64_t global_idx) {
    if (pack.layout_kind == PackLayoutKind::DenseCover &&
        global_idx >= 0 &&
        static_cast<std::size_t>(global_idx) < pack.dense_extent) {
        return static_cast<int32_t>(global_idx);
    }

    if (!pack.g2l.empty()) {
        const auto it = pack.g2l.find(global_idx);
        if (it != pack.g2l.end()) {
            return it->second;
        }
    }

    auto it = std::upper_bound(
        pack.segments.begin(),
        pack.segments.end(),
        global_idx,
        [](int64_t value, const LinearSegment& segment) {
            return value < segment.global_begin;
        });
    if (it != pack.segments.begin()) {
        --it;
        if (global_idx >= it->global_begin &&
            global_idx < it->global_begin + it->len) {
            return static_cast<int32_t>(
                it->local_begin + (global_idx - it->global_begin));
        }
    }

    return -1;
}

inline int32_t lookup_local_slot_or_throw(const PackMap& pack,
                                          int64_t global_idx,
                                          const char* where) {
    const int32_t slot = try_lookup_local_slot(pack, global_idx);
    if (slot >= 0) {
        return slot;
    }

    std::string message = where ? where : "lookup_local_slot_or_throw";
    message += ": global ";
    message += std::to_string(global_idx);
    message += " not found in pack layout mismatch (kind=";
    message += pack_layout_kind_name(pack.layout_kind);
    message += ")";
    throw std::runtime_error(message);
}

inline std::vector<int32_t> build_dense_global_to_local_lut(
    const PackMap& pack,
    std::size_t dense_size,
    const char* where) {
    if (pack.layout_kind == PackLayoutKind::DenseCover &&
        pack.dense_extent != 0 && pack.dense_extent != dense_size) {
        std::string message = where ? where : "build_dense_global_to_local_lut";
        message += ": dense extent mismatch, pack dense_extent=";
        message += std::to_string(pack.dense_extent);
        message += ", requested=";
        message += std::to_string(dense_size);
        throw std::runtime_error(message);
    }

    std::vector<int32_t> lut(dense_size, -1);
    for (std::size_t local_idx = 0; local_idx < pack.globals.size(); ++local_idx) {
        const int64_t global_idx = pack.globals[local_idx];
        if (global_idx < 0 ||
            static_cast<std::size_t>(global_idx) >= dense_size) {
            std::string message =
                where ? where : "build_dense_global_to_local_lut";
            message += ": global ";
            message += std::to_string(global_idx);
            message += " outside dense extent ";
            message += std::to_string(dense_size);
            message += " - pack layout mismatch";
            throw std::runtime_error(message);
        }
        lut[static_cast<std::size_t>(global_idx)] =
            static_cast<int32_t>(local_idx);
    }
    return lut;
}

}  // namespace mpi
}  // namespace dacpp

#endif
