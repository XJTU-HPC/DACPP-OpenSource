#ifndef DACPP_MPI_HALO_H
#define DACPP_MPI_HALO_H

#include "Pack.h"

namespace dacpp {
namespace mpi {

// Item-reach analysis for fast neighbor filtering
// ---------------------------------------------------------------------------

/// Result of item-reach analysis for an AccessPattern.
struct ItemReachResult {
    /// Maximum number of items away that the pattern can reach.
    /// -1 means "unbounded" (e.g. broadcast patterns → fall back to full
    /// enumeration).
    int64_t max_item_reach = -1;
};

/// Compute the maximum item-space reach of an access pattern.
///
/// For stencil patterns, this determines how far in item-space a rank's
/// data dependencies extend.  Combined with items-per-rank, this yields a
/// tight rank window so that computeParamHalo() only checks plausible
/// neighbors instead of all P ranks.
///
/// Returns -1 (unbounded) for broadcast/all-to-all patterns where every
/// rank may be a neighbor.
inline ItemReachResult computeItemReach(const AccessPattern& pattern) {
    const auto bind_splits =
        pattern.bind_split_sizes.empty()
            ? init_bind_split_sizes(pattern)
            : pattern.bind_split_sizes;

    if (bind_splits.empty()) {
        return {-1};
    }

    // Per-bind-id reach in bind-index space.
    const int max_bind =
        *std::max_element(pattern.bind_set_id.begin(),
                          pattern.bind_set_id.end());
    if (max_bind < 0) {
        // No binds at all — reach is 0.
        return {0};
    }

    std::vector<int64_t> per_bind_reach(
        static_cast<std::size_t>(max_bind + 1), 0);

    for (int op_idx = 0; op_idx < pattern.param_ops.size; ++op_idx) {
        const Dac_Op& op = pattern.param_ops[op_idx];
        if (op.dimId < 0 || op.dimId >= pattern.data_info.dim) {
            continue;
        }
        if (op_idx >= static_cast<int>(pattern.bind_set_id.size())) {
            continue;
        }
        const int bind_id = pattern.bind_set_id[op_idx];
        if (bind_id < 0) {
            continue;
        }

        const bool is_index =
            op_idx < static_cast<int>(pattern.is_index_op.size()) &&
            pattern.is_index_op[op_idx];

        // Detect broadcast: IndexSplit whose split_size covers the full
        // dimension length means every item accesses every position in
        // that dimension → unbounded reach.
        if (is_index) {
            const int64_t split_sz =
                bind_splits[static_cast<std::size_t>(bind_id)];
            const int dim_len =
                op.dimId < static_cast<int>(pattern.data_info.dim)
                    ? pattern.data_info.dimLength[op.dimId]
                    : 0;
            if (split_sz >= dim_len && dim_len > 1) {
                return {-1};
            }
            // Index op: each item accesses exactly one position.
            // No inter-item reach contribution.
            continue;
        }

        // RegularSlice: partition extends by op.size positions with stride
        // op.stride.  Two items with bind indices b1, b2 access overlapping
        // positions iff |b1 - b2| * stride < size, i.e.
        // |b1 - b2| < ceil(size / stride).
        const int64_t reach_in_bind =
            (op.size + op.stride - 1) / op.stride - 1;
        auto& slot = per_bind_reach[static_cast<std::size_t>(bind_id)];
        slot = std::max(slot, reach_in_bind);
    }

    // Convert per-bind-index reach to item-space reach.
    // decode_item_id uses cumulative divisors: for bind dimension B,
    // the item-space stride is product(bind_splits[B+1:]).
    int64_t item_stride = 1;
    for (int B = max_bind; B >= 0; --B) {
        const int64_t split_sz =
            bind_splits[static_cast<std::size_t>(B)] <= 0
                ? 1
                : bind_splits[static_cast<std::size_t>(B)];
        per_bind_reach[static_cast<std::size_t>(B)] *= item_stride;
        item_stride *= split_sz;
    }

    int64_t max_reach = 0;
    for (int64_t r : per_bind_reach) {
        max_reach = std::max(max_reach, r);
    }
    return {max_reach};
}

/// Rank window [lo, hi] (inclusive) for neighbor enumeration.
/// lo < 0 means "unbounded" — fall back to full enumeration.
struct RankWindow {
    int lo;
    int hi;
};

/// Convert an item-space reach into a rank window.
inline RankWindow computeRankWindow(int64_t item_reach,
                                    int64_t total_items,
                                    int mpi_rank,
                                    int mpi_size) {
    if (item_reach < 0) {
        return {-1, -1};  // unbounded
    }
    if (mpi_size <= 1) {
        return {0, 0};
    }

    const int64_t min_items = total_items / mpi_size;
    if (min_items <= 0) {
        // More ranks than items — degenerate, fall back.
        return {-1, -1};
    }

    // Ceiling-division to get rank reach, plus 1 to account for the
    // uneven distribution in get_rank_item_range().
    const int64_t rank_reach =
        (item_reach + min_items - 1) / min_items + 1;

    return {
        static_cast<int>(
            std::max<int64_t>(0, static_cast<int64_t>(mpi_rank) - rank_reach)),
        static_cast<int>(std::min<int64_t>(
            mpi_size - 1, static_cast<int64_t>(mpi_rank) + rank_reach))};
}

// ---------------------------------------------------------------------------
// Halo exchange support for MPI stencil optimization
// ---------------------------------------------------------------------------

/// Information for a single neighbor direction, scoped to one parameter.
struct HaloRegion {
    int neighbor_rank = -1;
    /// Global positions I need to receive from this neighbor.
    std::vector<int64_t> recv_globals;
    /// Corresponding slots in the local array (via g2l).
    std::vector<int32_t> recv_local_slots;
    /// Global positions I need to send to this neighbor.
    std::vector<int64_t> send_globals;
    /// Corresponding slots in the local array.
    std::vector<int32_t> send_local_slots;
};

/// Complete halo information for one parameter (up to 2 neighbors).
struct ParamHalo {
    std::vector<HaloRegion> regions;
};

/// Compute halo regions for a single parameter on the current rank.
/// The caller provides the parameter's AccessPattern, its effective mode,
/// the current rank's item range, and the actual PackMap (ctx.pack_*)
/// that defines the real local data layout.
///
/// IMPORTANT: The slot mapping uses ctx_pack.g2l directly, ensuring
/// consistency with exchangeHalo() which reads/writes ctx.local_*.
/// This avoids the bug where a separately-built PackMap has different
/// slot numbering than ctx.pack_* (especially when sibling dense-cover
/// packs expand ctx.pack_* to a full global dense layout).
///
/// Neighbor strategy: data-driven discovery over all ranks.
/// For each rank, the function checks actual global position overlap
/// (recv = my_input ∩ neighbor_output, send = my_output ∩ neighbor_input).
/// This correctly handles multi-dimensional binds, non-unit strides,
/// full-dimension broadcasts, and large stencil offsets.
///
/// Algorithm:
///   For each neighbor (rank-1, rank+1):
///     recv = my_input_globals ∩ neighbor_output_globals
///     send = my_output_globals ∩ neighbor_input_globals
inline ParamHalo computeParamHalo(
    const AccessPattern& pattern,
    AccessMode mode,
    ItemRange my_range,
    int64_t total_items,
    int mpi_rank,
    int mpi_size,
    const PackMap& ctx_pack)
{
    ParamHalo halo;

    // Build my own input/output position sets.
    // For "output" we use the same collect_positions_for_item — the output
    // positions of an item are the positions its partition covers.
    std::vector<int64_t> my_input_globals;
    std::vector<int64_t> my_output_globals;
    {
        std::unordered_set<int64_t> seen_in, seen_out;
        for (int64_t item = my_range.begin; item < my_range.end; ++item) {
            auto positions = collect_positions_for_item(item, pattern);
            for (auto g : positions) {
                if (seen_in.insert(g).second) {
                    my_input_globals.push_back(g);
                }
                // For output, all partition positions are potential outputs.
                // A more precise analysis could restrict to write-mode dims,
                // but using the full partition is safe for correctness.
                if (seen_out.insert(g).second) {
                    my_output_globals.push_back(g);
                }
            }
        }
        std::sort(my_input_globals.begin(), my_input_globals.end());
        std::sort(my_output_globals.begin(), my_output_globals.end());
    }

    // Data-driven neighbor discovery: instead of assuming only rank±1 are
    // neighbors, iterate over candidate ranks and check for actual data
    // overlap.  A rank is a halo neighbor iff its output globals intersect
    // with my input globals (recv direction) or its input globals intersect
    // with my output globals (send direction).  This correctly handles:
    //   - multi-dimensional binds with non-adjacent spatial neighbors
    //   - non-unit strides (e.g. interleaved access patterns)
    //   - full-dimension broadcast access ([{}] patterns → all-to-all)
    //   - large stencil offsets that span multiple rank boundaries
    //
    // Optimization: for stencil patterns the item-space reach is bounded,
    // so we compute a rank window and only enumerate nearby ranks rather
    // than all P ranks.  Unbounded patterns (broadcast) fall back to full
    // enumeration.
    const auto reach_result = computeItemReach(pattern);
    const auto win = computeRankWindow(
        reach_result.max_item_reach, total_items, mpi_rank, mpi_size);
    const int nb_lo = (win.lo < 0) ? 0 : win.lo;
    const int nb_hi = (win.hi < 0) ? (mpi_size - 1) : win.hi;

    for (int nb_rank = nb_lo; nb_rank <= nb_hi; ++nb_rank) {
        if (nb_rank == mpi_rank) continue;

        HaloRegion region;
        region.neighbor_rank = nb_rank;

        ItemRange nb_range = get_rank_item_range(total_items, nb_rank, mpi_size);

        // Collect neighbor's input and output positions.
        std::vector<int64_t> nb_input_globals;
        std::vector<int64_t> nb_output_globals;
        {
            std::unordered_set<int64_t> seen_in, seen_out;
            for (int64_t item = nb_range.begin; item < nb_range.end; ++item) {
                auto positions = collect_positions_for_item(item, pattern);
                for (auto g : positions) {
                    if (seen_in.insert(g).second) {
                        nb_input_globals.push_back(g);
                    }
                    if (seen_out.insert(g).second) {
                        nb_output_globals.push_back(g);
                    }
                }
            }
            std::sort(nb_input_globals.begin(), nb_input_globals.end());
            std::sort(nb_output_globals.begin(), nb_output_globals.end());
        }

        // Use ctx_pack.g2l directly for local slot lookup.
        // This ensures slot numbering matches the actual layout of ctx.local_*,
        // which is critical when sibling dense-cover packs expand ctx.pack_*
        // to a full global dense layout.

        // Recv: positions I read that the neighbor produces (writes).
        // recv = my_input_globals ∩ nb_output_globals
        if (mode != AccessMode::Write) {
            std::vector<int64_t> recv_globals;
            std::set_intersection(
                my_input_globals.begin(), my_input_globals.end(),
                nb_output_globals.begin(), nb_output_globals.end(),
                std::back_inserter(recv_globals));
            for (auto g : recv_globals) {
                region.recv_globals.push_back(g);
                region.recv_local_slots.push_back(
                    lookup_local_slot_or_throw(
                        ctx_pack, g, "computeParamHalo(recv)"));
            }
        }

        // Send: positions I produce that the neighbor reads.
        // send = my_output_globals ∩ nb_input_globals
        // We use ctx_pack.g2l (not an input-only pack) so that WRITE-mode
        // globals that exist in the output but not in a standalone input pack
        // are still correctly mapped to local slots.
        if (mode != AccessMode::Read) {
            std::vector<int64_t> send_globals;
            std::set_intersection(
                my_output_globals.begin(), my_output_globals.end(),
                nb_input_globals.begin(), nb_input_globals.end(),
                std::back_inserter(send_globals));
            for (auto g : send_globals) {
                region.send_globals.push_back(g);
                region.send_local_slots.push_back(
                    lookup_local_slot_or_throw(
                        ctx_pack, g, "computeParamHalo(send)"));
            }
        }

        if (!region.recv_globals.empty() || !region.send_globals.empty()) {
            halo.regions.push_back(std::move(region));
        }
    }

    return halo;
}

/// Perform halo exchange for a single parameter, iterating over neighbors.
/// Uses non-blocking send/recv to avoid deadlock.
template <typename T>
inline void exchangeHalo(std::vector<T>& local_data,
                         const ParamHalo& halo,
                         void* mpi_type_ptr) {
    // mpi_type_ptr is a pointer to the MPI_Datatype; we cast it back.
    MPI_Datatype mpi_type = *static_cast<MPI_Datatype*>(mpi_type_ptr);

    for (const auto& region : halo.regions) {
        if (region.neighbor_rank < 0) continue;

        // Pack send buffer.
        std::vector<T> send_buf;
        send_buf.reserve(region.send_local_slots.size());
        for (int32_t slot : region.send_local_slots) {
            send_buf.push_back(local_data[static_cast<std::size_t>(slot)]);
        }

        // Prepare receive buffer.
        std::vector<T> recv_buf(region.recv_local_slots.size());

        MPI_Request reqs[2];
        int req_count = 0;

        if (!send_buf.empty()) {
            MPI_Isend(send_buf.data(),
                      static_cast<int>(send_buf.size()),
                      mpi_type,
                      region.neighbor_rank,
                      /*tag=*/0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (!recv_buf.empty()) {
            MPI_Irecv(recv_buf.data(),
                      static_cast<int>(recv_buf.size()),
                      mpi_type,
                      region.neighbor_rank,
                      /*tag=*/0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Unpack receive buffer into local data.
        for (std::size_t i = 0; i < region.recv_local_slots.size(); ++i) {
            local_data[static_cast<std::size_t>(region.recv_local_slots[i])] = recv_buf[i];
        }
    }
}

}  // namespace mpi
}  // namespace dacpp

#endif
