#include <string>

#include "StencilCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {
namespace detail {

std::string waveRouteFastPathExpr(int paramIdx,
                                  const std::string& routeIdxExpr) {
    return "ctx.wave.route_fast_paths_by_param[" + std::to_string(paramIdx) +
           "][" + routeIdxExpr + "]";
}

std::string waveReadTransitionFastPathExpr(const std::string& idxExpr) {
    return "ctx.wave.read_cache_transition_fast_paths[" + idxExpr + "]";
}

WaveSpecializationCodegenConfig buildWaveSpecializationCodegenConfig(
    Shell* shell,
    Calc* calc,
    const mpi_rewriter::DistributedStencilSitePlan& distributedSitePlan) {
    WaveSpecializationCodegenConfig config;
    config.enable_span_pairs =
        shell->getName() == "waveEqShell" && calc->getName() == "waveEq";
    config.enable_direct_kernel =
        config.enable_span_pairs &&
        distributedSitePlan.supported &&
        !distributedSitePlan.hasRootBridge &&
        distributedSitePlan.followupMappings.size() == 1 &&
        distributedSitePlan.readCacheTransitions.size() == 1 &&
        shell->getNumShellParams() == 3;
    return config;
}

void appendWaveContextFields(std::string& code) {
    code += "    dacpp::mpi::WaveSpecializationState wave;\n";
}

void appendWaveInitFlags(std::string& code,
                         const WaveSpecializationCodegenConfig& waveConfig,
                         int shellParamCount,
                         std::size_t transitionCount) {
    code += "    ctx.wave.use_span_pairs = " +
            std::string(waveConfig.enable_span_pairs ? "true" : "false") +
            ";\n";
    code += "    ctx.wave.use_direct_kernel = " +
            std::string(waveConfig.enable_direct_kernel ? "true" : "false") +
            ";\n";
    code += "    ctx.wave.route_fast_paths_by_param.clear();\n";
    code += "    ctx.wave.route_fast_paths_by_param.resize(" +
            std::to_string(shellParamCount) + ");\n";
    code += "    ctx.wave.read_cache_transition_fast_paths.clear();\n";
    code += "    ctx.wave.read_cache_transition_fast_paths.resize(" +
            std::to_string(transitionCount) + ");\n";
    code += "    ctx.wave.direct_kernel.slots.clear();\n";
    code += "    ctx.wave.direct_kernel.slots_buffer.reset();\n";
    code += "    ctx.wave.direct_kernel.next_stale_slots.clear();\n";
    code += "    ctx.wave.direct_kernel.can_sparse_clear = false;\n";
}

void appendWaveDirectKernelMetadataInit(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig) {
    if (!waveConfig.enable_direct_kernel) {
        return;
    }
    code += "    if (ctx.wave.use_direct_kernel) {\n";
    code += "        const std::size_t __dacpp_wave_items = static_cast<std::size_t>(ctx.local_item_count);\n";
    code += "        if (ctx.plan_cur.item_key_offsets.size() != __dacpp_wave_items || ctx.plan_prev.item_key_offsets.size() != __dacpp_wave_items || ctx.plan_next.item_key_offsets.size() != __dacpp_wave_items) {\n";
    code += "            ctx.wave.use_direct_kernel = false;\n";
    code += "        } else {\n";
    code += "            ctx.wave.direct_kernel.slots.resize(__dacpp_wave_items * 7);\n";
    code += "            std::vector<unsigned char> __dacpp_wave_written(ctx.plan_next.pack.globals.size(), static_cast<unsigned char>(0));\n";
    code += "            bool __dacpp_wave_slots_valid = true;\n";
    code += "            for (std::size_t __dacpp_item = 0; __dacpp_item < __dacpp_wave_items; ++__dacpp_item) {\n";
    code += "                const int32_t __dacpp_cur_base = ctx.plan_cur.item_key_offsets[__dacpp_item];\n";
    code += "                const int32_t __dacpp_prev_base = ctx.plan_prev.item_key_offsets[__dacpp_item];\n";
    code += "                const int32_t __dacpp_next_base = ctx.plan_next.item_key_offsets[__dacpp_item];\n";
    code += "                if (__dacpp_cur_base < 0 || __dacpp_prev_base < 0 || __dacpp_next_base < 0 || static_cast<std::size_t>(__dacpp_cur_base + 7) >= ctx.plan_cur.compact_slots.size() || static_cast<std::size_t>(__dacpp_prev_base) >= ctx.plan_prev.compact_slots.size() || static_cast<std::size_t>(__dacpp_next_base) >= ctx.plan_next.compact_slots.size()) {\n";
    code += "                    ctx.wave.use_direct_kernel = false;\n";
    code += "                    __dacpp_wave_slots_valid = false;\n";
    code += "                    break;\n";
    code += "                }\n";
    code += "                const std::size_t __dacpp_slot_base = __dacpp_item * 7;\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 0] = ctx.plan_cur.compact_slots[static_cast<std::size_t>(__dacpp_cur_base + 4)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 1] = ctx.plan_cur.compact_slots[static_cast<std::size_t>(__dacpp_cur_base + 1)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 2] = ctx.plan_cur.compact_slots[static_cast<std::size_t>(__dacpp_cur_base + 7)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 3] = ctx.plan_cur.compact_slots[static_cast<std::size_t>(__dacpp_cur_base + 3)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 4] = ctx.plan_cur.compact_slots[static_cast<std::size_t>(__dacpp_cur_base + 5)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 5] = ctx.plan_prev.compact_slots[static_cast<std::size_t>(__dacpp_prev_base)];\n";
    code += "                ctx.wave.direct_kernel.slots[__dacpp_slot_base + 6] = ctx.plan_next.compact_slots[static_cast<std::size_t>(__dacpp_next_base)];\n";
    code += "                const int32_t __dacpp_center_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 0];\n";
    code += "                const int32_t __dacpp_up_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 1];\n";
    code += "                const int32_t __dacpp_down_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 2];\n";
    code += "                const int32_t __dacpp_left_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 3];\n";
    code += "                const int32_t __dacpp_right_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 4];\n";
    code += "                const int32_t __dacpp_prev_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 5];\n";
    code += "                const int32_t __dacpp_next_slot = ctx.wave.direct_kernel.slots[__dacpp_slot_base + 6];\n";
    code += "                const bool __dacpp_cur_slot_valid = __dacpp_center_slot >= 0 && __dacpp_up_slot >= 0 && __dacpp_down_slot >= 0 && __dacpp_left_slot >= 0 && __dacpp_right_slot >= 0 && static_cast<std::size_t>(__dacpp_center_slot) < ctx.plan_cur.pack.globals.size() && static_cast<std::size_t>(__dacpp_up_slot) < ctx.plan_cur.pack.globals.size() && static_cast<std::size_t>(__dacpp_down_slot) < ctx.plan_cur.pack.globals.size() && static_cast<std::size_t>(__dacpp_left_slot) < ctx.plan_cur.pack.globals.size() && static_cast<std::size_t>(__dacpp_right_slot) < ctx.plan_cur.pack.globals.size();\n";
    code += "                const bool __dacpp_prev_slot_valid = __dacpp_prev_slot >= 0 && static_cast<std::size_t>(__dacpp_prev_slot) < ctx.plan_prev.pack.globals.size();\n";
    code += "                const bool __dacpp_next_slot_valid = __dacpp_next_slot >= 0 && static_cast<std::size_t>(__dacpp_next_slot) < ctx.plan_next.pack.globals.size();\n";
    code += "                if (!__dacpp_cur_slot_valid || !__dacpp_prev_slot_valid || !__dacpp_next_slot_valid) {\n";
    code += "                    ctx.wave.use_direct_kernel = false;\n";
    code += "                    __dacpp_wave_slots_valid = false;\n";
    code += "                    break;\n";
    code += "                }\n";
    code += "                __dacpp_wave_written[static_cast<std::size_t>(__dacpp_next_slot)] = static_cast<unsigned char>(1);\n";
    code += "            }\n";
    code += "            if (ctx.wave.use_direct_kernel && __dacpp_wave_slots_valid) {\n";
    code += "                ctx.wave.direct_kernel.can_sparse_clear = true;\n";
    code += "                ctx.wave.direct_kernel.next_stale_slots.clear();\n";
    code += "                ctx.wave.direct_kernel.next_stale_slots.reserve(ctx.plan_next.pack.globals.size());\n";
    code += "                for (std::size_t __dacpp_slot = 0; __dacpp_slot < __dacpp_wave_written.size(); ++__dacpp_slot) {\n";
    code += "                    if (__dacpp_wave_written[__dacpp_slot] == static_cast<unsigned char>(0)) {\n";
    code += "                        ctx.wave.direct_kernel.next_stale_slots.push_back(static_cast<int32_t>(__dacpp_slot));\n";
    code += "                    }\n";
    code += "                }\n";
    code += "                if (ctx.wave.direct_kernel.slots.empty()) {\n";
    code += "                    ctx.wave.direct_kernel.slots_buffer = std::make_unique<sycl::buffer<int32_t, 1>>(sycl::range<1>(0));\n";
    code += "                } else {\n";
    code += "                    ctx.wave.direct_kernel.slots_buffer = std::make_unique<sycl::buffer<int32_t, 1>>(ctx.wave.direct_kernel.slots.data(), sycl::range<1>(ctx.wave.direct_kernel.slots.size()));\n";
    code += "                }\n";
    code += "            } else {\n";
    code += "                ctx.wave.direct_kernel.can_sparse_clear = false;\n";
    code += "                ctx.wave.direct_kernel.next_stale_slots.clear();\n";
    code += "                ctx.wave.direct_kernel.slots_buffer.reset();\n";
    code += "            }\n";
    code += "        }\n";
    code += "    }\n";
}

void appendWaveRouteFastPathInit(std::string& code,
                                 const WaveSpecializationCodegenConfig& waveConfig,
                                 const std::string& routeStateExpr,
                                 const std::string& distExpr,
                                 const std::string& routeIdxExpr) {
    if (!waveConfig.enable_span_pairs) {
        return;
    }
    code += "    if (ctx.wave.use_span_pairs) {\n";
    code += "        std::string __dacpp_wave_span_reason;\n";
    code += "        " + routeStateExpr +
            ".use_span_pairs = dacpp::mpi::build_local_span_pairs_from_slots(" +
            distExpr + ".local_write_slots, " + distExpr +
            ".local_target_slots_by_route[" + routeIdxExpr + "], " +
            routeStateExpr + ".local_write_spans, " + routeStateExpr +
            ".local_target_spans, &__dacpp_wave_span_reason);\n";
    code += "        if (" + routeStateExpr + ".use_span_pairs) {\n";
    code += "            std::string __dacpp_wave_row_reason;\n";
    code += "            " + routeStateExpr +
            ".use_row_copy_blocks = dacpp::mpi::build_contiguous_row_copy_blocks_from_span_pairs(" +
            routeStateExpr + ".local_write_spans, " + routeStateExpr +
            ".local_target_spans, " + routeStateExpr +
            ".local_row_copy_blocks, &__dacpp_wave_row_reason);\n";
    code += "        }\n";
    code += "    }\n";
}

void appendWaveReadTransitionFastPathInit(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig,
    const std::string& idx,
    const std::string& fastPathExpr) {
    if (!waveConfig.enable_span_pairs) {
        return;
    }
    code += "    if (ctx.wave.use_span_pairs) {\n";
    code += "        std::string __dacpp_span_reason;\n";
    code += "        " + fastPathExpr +
            ".use_span_pairs = dacpp::mpi::build_span_pairs_from_slots(ctx.read_cache_transition_source_slots_" +
            idx + ", ctx.read_cache_transition_target_slots_" + idx +
            ", " + fastPathExpr + ".source_spans, " + fastPathExpr +
            ".target_spans, &__dacpp_span_reason);\n";
    code += "        if (" + fastPathExpr + ".use_span_pairs) {\n";
    code += "            std::string __dacpp_row_reason;\n";
    code += "            " + fastPathExpr +
            ".use_row_copy_blocks = dacpp::mpi::build_contiguous_row_copy_blocks_from_span_pairs(" +
            fastPathExpr + ".source_spans, " + fastPathExpr +
            ".target_spans, " + fastPathExpr +
            ".row_copy_blocks, &__dacpp_row_reason);\n";
    code += "        }\n";
    code += "    }\n";
}

bool appendWaveDistributedWriteReset(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig,
    const std::string& calcName,
    const std::string& elemType) {
    if (!(waveConfig.enable_direct_kernel && calcName == "next")) {
        return false;
    }
    code += "    if (!ctx.wave.use_direct_kernel || !ctx.wave.direct_kernel.can_sparse_clear) {\n";
    code += "        local_" + calcName + ".assign(ctx.plan_" + calcName +
            ".pack.globals.size(), " + elemType + "{});\n";
    code += "    } else if (!ctx.wave.direct_kernel.next_stale_slots.empty()) {\n";
    code += "        for (int32_t __dacpp_stale_slot : ctx.wave.direct_kernel.next_stale_slots) {\n";
    code += "            local_" + calcName +
            "[static_cast<std::size_t>(__dacpp_stale_slot)] = " + elemType +
            "{};\n";
    code += "        }\n";
    code += "    }\n";
    return true;
}

void appendWaveDistributedKernelDispatchHead(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig) {
    if (!waveConfig.enable_direct_kernel) {
        code += "        if (ctx.use_contiguous_kernel_views) {\n";
        return;
    }
    code += "        if (ctx.wave.use_direct_kernel) {\n";
    code += "            const double __dacpp_wave_dt = 0.5f * std::fmin(dx, dy) / c;\n";
    code += "            const double __dacpp_wave_coeff = (c * c) * __dacpp_wave_dt * __dacpp_wave_dt;\n";
    code += "            const double __dacpp_wave_inv_dx2 = 1.0 / (dx * dx);\n";
    code += "            const double __dacpp_wave_inv_dy2 = 1.0 / (dy * dy);\n";
    code += "            {\n";
    code += "                sycl::buffer<double, 1> buffer_cur(local_cur.data(), sycl::range<1>(local_cur.size()));\n";
    code += "                sycl::buffer<double, 1> buffer_prev(local_prev.data(), sycl::range<1>(local_prev.size()));\n";
    code += "                sycl::buffer<double, 1> buffer_next(local_next.data(), sycl::range<1>(local_next.size()));\n";
    code += "                q.submit([&](sycl::handler& h) {\n";
    code += "                    auto acc_cur = buffer_cur.get_access<sycl::access::mode::read>(h);\n";
    code += "                    auto acc_prev = buffer_prev.get_access<sycl::access::mode::read>(h);\n";
    code += "                    auto acc_next = buffer_next.get_access<sycl::access::mode::read_write>(h);\n";
    code += "                    auto acc_wave_slots = ctx.wave.direct_kernel.slots_buffer->get_access<sycl::access::mode::read>(h);\n";
    code += "                    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                        const std::size_t item_linear = idx[0];\n";
    code += "                        const std::size_t slot_base = item_linear * 7;\n";
    code += "                        const int32_t center_slot = acc_wave_slots[slot_base + 0];\n";
    code += "                        const int32_t up_slot = acc_wave_slots[slot_base + 1];\n";
    code += "                        const int32_t down_slot = acc_wave_slots[slot_base + 2];\n";
    code += "                        const int32_t left_slot = acc_wave_slots[slot_base + 3];\n";
    code += "                        const int32_t right_slot = acc_wave_slots[slot_base + 4];\n";
    code += "                        const int32_t prev_slot = acc_wave_slots[slot_base + 5];\n";
    code += "                        const int32_t next_slot = acc_wave_slots[slot_base + 6];\n";
    code += "                        const double center = acc_cur[static_cast<std::size_t>(center_slot)];\n";
    code += "                        const double up = acc_cur[static_cast<std::size_t>(up_slot)];\n";
    code += "                        const double down = acc_cur[static_cast<std::size_t>(down_slot)];\n";
    code += "                        const double left = acc_cur[static_cast<std::size_t>(left_slot)];\n";
    code += "                        const double right = acc_cur[static_cast<std::size_t>(right_slot)];\n";
    code += "                        const double prev = acc_prev[static_cast<std::size_t>(prev_slot)];\n";
    code += "                        const double u_xx = (down - 2.0 * center + up) * __dacpp_wave_inv_dx2;\n";
    code += "                        const double u_yy = (right - 2.0 * center + left) * __dacpp_wave_inv_dy2;\n";
    code += "                        acc_next[static_cast<std::size_t>(next_slot)] = 2.0 * center - prev + __dacpp_wave_coeff * (u_xx + u_yy);\n";
    code += "                    });\n";
    code += "                });\n";
    code += "                q.wait();\n";
    code += "            }\n";
    code += "        } else if (ctx.use_contiguous_kernel_views) {\n";
}

void appendWaveReadTransitionRun(std::string& code,
                                 const WaveSpecializationCodegenConfig& waveConfig,
                                 const std::string& idx,
                                 const std::string& fastPathExpr,
                                 const std::string& writerCalcName,
                                 const std::string& readerCalcName) {
    if (waveConfig.enable_span_pairs) {
        code += "    if (ctx.wave.use_span_pairs && " + fastPathExpr +
                ".use_row_copy_blocks) {\n";
        code += "        dacpp::mpi::scatter_values_by_row_copy_blocks_into(local_" +
                writerCalcName + ", " + fastPathExpr + ".row_copy_blocks, local_" +
                readerCalcName + ");\n";
        code += "    } else if (ctx.wave.use_span_pairs && " + fastPathExpr +
                ".use_span_pairs) {\n";
        code += "        dacpp::mpi::scatter_values_by_span_pairs_into(local_" +
                writerCalcName + ", " + fastPathExpr + ".source_spans, " +
                fastPathExpr + ".target_spans, local_" + readerCalcName +
                ");\n";
        code += "    } else {\n";
    }
    code += "        dacpp::mpi::scatter_values_by_slots_parallel_into(local_" +
            writerCalcName + ", ctx.read_cache_transition_source_slots_" + idx +
            ", ctx.read_cache_transition_target_slots_" + idx + ", local_" +
            readerCalcName + ", q);\n";
    if (waveConfig.enable_span_pairs) {
        code += "    }\n";
    }
}

void appendWavePublishRun(std::string& code,
                          const WaveSpecializationCodegenConfig& waveConfig,
                          bool useDistributedReaderMaterialize,
                          const std::string& calcName,
                          const std::string& routeStateExpr,
                          const std::string& routeIdxExpr,
                          const std::string& targetCacheExpr) {
    const std::string distExpr = "ctx.dist_" + calcName;
    if (useDistributedReaderMaterialize) {
        if (waveConfig.enable_span_pairs) {
            code += "    if (ctx.wave.use_span_pairs && " + routeStateExpr +
                    ".use_row_copy_blocks) {\n";
            code += "        dacpp::mpi::publish_local_writes_with_span_pairs_or_exchange_cache_only(local_" +
                    calcName + ", " + distExpr + ".local_write_slots, " +
                    distExpr + ".local_target_slots_by_route[" + routeIdxExpr +
                    "], " + routeStateExpr + ".local_write_spans, " +
                    routeStateExpr + ".local_target_spans, " + routeStateExpr +
                    ".local_row_copy_blocks, " + targetCacheExpr + ", " + distExpr +
                    ".exchange_plans_by_route[" + routeIdxExpr + "], " +
                    distExpr + ".halo_plans_by_route[" + routeIdxExpr + "], " +
                    distExpr + ".halo_runtimes_by_route[" + routeIdxExpr +
                    "], q);\n";
            code += "    } else if (ctx.wave.use_span_pairs && " +
                    routeStateExpr + ".use_span_pairs) {\n";
            code += "        dacpp::mpi::publish_local_writes_with_span_pairs_or_exchange_cache_only(local_" +
                    calcName + ", " + distExpr + ".local_write_slots, " +
                    distExpr + ".local_target_slots_by_route[" + routeIdxExpr +
                    "], " + routeStateExpr + ".local_write_spans, " +
                    routeStateExpr + ".local_target_spans, " + routeStateExpr +
                    ".local_row_copy_blocks, " + targetCacheExpr + ", " + distExpr +
                    ".exchange_plans_by_route[" + routeIdxExpr + "], " +
                    distExpr + ".halo_plans_by_route[" + routeIdxExpr + "], " +
                    distExpr + ".halo_runtimes_by_route[" + routeIdxExpr +
                    "], q);\n";
            code += "    } else {\n";
        }
        code += "        dacpp::mpi::publish_local_writes_with_halo_or_exchange_cache_only(local_" +
                calcName + ", " + distExpr + ".local_write_slots, " +
                distExpr + ".local_target_slots_by_route[" + routeIdxExpr +
                "], " + targetCacheExpr + ", " + distExpr +
                ".exchange_plans_by_route[" + routeIdxExpr + "], " + distExpr +
                ".halo_plans_by_route[" + routeIdxExpr + "], q);\n";
        if (waveConfig.enable_span_pairs) {
            code += "    }\n";
        }
        return;
    }

    code += "    dacpp::mpi::publish_local_writes_with_halo_or_exchange(local_" +
            calcName + ", " + distExpr + ".local_write_slots, " + distExpr +
            ".local_target_slots_by_route[" + routeIdxExpr + "], " +
            targetCacheExpr + ", " + distExpr + ".local_write_values, " +
            distExpr + ".exchange_plans_by_route[" + routeIdxExpr + "], " +
            distExpr + ".halo_plans_by_route[" + routeIdxExpr + "], q);\n";
}

}  // namespace detail
}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
