#include <set>
#include <string>
#include <vector>

#include "StencilCodegen_Internal.h"
#include "llvm/Support/raw_ostream.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

using namespace detail;

std::string wrapperName(Shell* shell, Calc* calc) {
    return shell->getName() + "_" + calc->getName();
}

std::string contextTypeName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_ctx_" + wrapperName(shell, calc);
}

std::string initFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_init_" + wrapperName(shell, calc);
}

std::string runFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_run_" + wrapperName(shell, calc);
}

std::string materializeFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_materialize_" + wrapperName(shell, calc);
}

std::string buildShellSignature(Shell* shell) {
    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + param->getName();
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }
    return signature;
}

std::string buildStencilWrapperCode(DacppFile* dacppFile,
                                    Shell* shell,
                                    Calc* calc,
                                    const clang::BinaryOperator* dacExpr) {
    const std::string wrapper = wrapperName(shell, calc);
    const std::string ctxType = contextTypeName(shell, calc);
    const std::string initName = initFunctionName(shell, calc);
    const std::string runName = runFunctionName(shell, calc);
    const std::string materializeName = materializeFunctionName(shell, calc);
    const std::string shellSignature = buildShellSignature(shell);
    const std::string shellArgNames = buildParamNameList(shell);
    const auto effectiveModes = mpi_rewriter::inferEffectiveParamModes(shell, calc);
    const auto transportModes = mpi_rewriter::inferPhaseCTransportParamModes(shell, calc);
    const auto splitMeta = mpi_rewriter::collectSplitBindMeta(shell);
    const auto distributedSitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    const WaveSpecializationCodegenConfig waveCodegenConfig =
        buildWaveSpecializationCodegenConfig(shell, calc, distributedSitePlan);
    const bool useDistributedReaderMaterialize =
        distributedSitePlan.supported &&
        !distributedSitePlan.hasRootBridge &&
        !distributedSitePlan.boundaryLocalUpdates.empty() &&
        distributedSitePlan.followupMappings.size() == 1;
    std::set<int> distributedMaterializeReaderParams;
    if (useDistributedReaderMaterialize) {
        for (const auto& mapping : distributedSitePlan.followupMappings) {
            if (mapping.readerParamIndex >= 0) {
                distributedMaterializeReaderParams.insert(mapping.readerParamIndex);
            }
        }
    }
    std::vector<bool> fallbackInputCacheable(
        static_cast<std::size_t>(shell->getNumShellParams()), false);
    std::vector<int> fallbackInputRefreshSource(
        static_cast<std::size_t>(shell->getNumShellParams()), -1);
    std::vector<std::string> actualTensorNames(
        static_cast<std::size_t>(shell->getNumShellParams()));
    std::vector<bool> aliasesAnyOtherParam(
        static_cast<std::size_t>(shell->getNumShellParams()), false);
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        actualTensorNames[static_cast<std::size_t>(paramIdx)] =
            resolveActualTensorName(shell->getParam(paramIdx)->getName(), dacExpr);
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& actualTensorName =
            actualTensorNames[static_cast<std::size_t>(paramIdx)];
        bool aliasesMutableParam = false;
        for (int otherIdx = 0; otherIdx < shell->getNumShellParams(); ++otherIdx) {
            if (otherIdx == paramIdx) {
                continue;
            }
            if (actualTensorNames[static_cast<std::size_t>(otherIdx)] ==
                    actualTensorName &&
                transportModes[otherIdx] != IOTYPE::READ) {
                aliasesMutableParam = true;
            }
            if (actualTensorNames[static_cast<std::size_t>(otherIdx)] ==
                actualTensorName) {
                aliasesAnyOtherParam[static_cast<std::size_t>(paramIdx)] = true;
            }
        }
        fallbackInputCacheable[static_cast<std::size_t>(paramIdx)] =
            !aliasesMutableParam &&
            !distributedSitePlan.supported &&
            isFallbackInputCacheCandidate(
                dacppFile, dacExpr, actualTensorName, transportModes[paramIdx]);
    }
    if (!distributedSitePlan.supported) {
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                continue;
            }
            const int sourceIdx = findLoopCarriedInputSourceParam(
                dacppFile, shell, calc, dacExpr, paramIdx, actualTensorNames,
                transportModes, aliasesAnyOtherParam);
            if (sourceIdx >= 0) {
                fallbackInputCacheable[static_cast<std::size_t>(paramIdx)] = true;
                fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)] =
                    sourceIdx;
            }
        }
    }

    if (distributedSitePlan.supported) {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange enabled";
        if (distributedSitePlan.hasRootBridge && !useDistributedReaderMaterialize) {
            llvm::outs() << " (root-bridge)";
        } else if (distributedSitePlan.hasRootBridge) {
            llvm::outs() << " (distributed-boundary)";
        }
        llvm::outs() << "\n";
    } else {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange disabled: "
                     << distributedSitePlan.disableReason << "\n";
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
            llvm::outs() << "[DACPP][MPI][FallbackCache] input "
                         << shell->getParam(paramIdx)->getName()
                         << " cached in init";
            const int sourceIdx =
                fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
            if (sourceIdx >= 0) {
                llvm::outs() << " refresh-from "
                             << shell->getParam(sourceIdx)->getName();
            }
            llvm::outs() << "\n";
        }
    }

    std::string code;
    code += "struct " + ctxType + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    bool use_partial_exchange = false;\n";
    code += "    bool use_contiguous_kernel_views = false;\n";
    code += "    std::string partial_exchange_disable_reason;\n";
    code += "    std::unique_ptr<sycl::queue> q;\n";
        code += "    std::vector<int64_t> binding_split_sizes;\n";
        code += "    int64_t total_items = 1;\n";
        code += "    dacpp::mpi::ItemRange item_range{};\n";
        code += "    int64_t local_item_count = 0;\n";
        appendWaveContextFields(code);
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string elemType = calc->getParam(paramIdx)->getBasicType();
        code += "    dacpp::mpi::AccessPattern pattern_" + calcName + ";\n";
        code += "    dacpp::mpi::PackPlan plan_" + calcName + ";\n";
        code += "    std::vector<" + elemType + "> local_" + calcName + ";\n";
        code += "    dacpp::mpi::DistributedTensorState<" + elemType + "> dist_" +
                calcName + ";\n";
        if (transportModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::GatheredIndexLayout input_layout_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> sendbuf_" + calcName + ";\n";
        }
        if (transportModes[paramIdx] != IOTYPE::READ) {
            code += "    dacpp::mpi::GatheredIndexLayout output_layout_" + calcName + ";\n";
            code += "    std::vector<int32_t> writeback_slots_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> writeback_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_recv_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_out_" + calcName + ";\n";
        }
    }
    if (useDistributedReaderMaterialize) {
        for (std::size_t updateIdx = 0;
             updateIdx < distributedSitePlan.boundaryLocalUpdates.size();
             ++updateIdx) {
            code += "    std::vector<int32_t> boundary_local_target_slots_" +
                    std::to_string(updateIdx) + ";\n";
            code += "    std::vector<int32_t> boundary_local_source_slots_" +
                    std::to_string(updateIdx) + ";\n";
        }
    }
    if (!distributedSitePlan.readCacheTransitions.empty()) {
        for (std::size_t transitionIdx = 0;
             transitionIdx < distributedSitePlan.readCacheTransitions.size();
             ++transitionIdx) {
            code += "    std::vector<int32_t> read_cache_transition_target_slots_" +
                    std::to_string(transitionIdx) + ";\n";
            code += "    std::vector<int32_t> read_cache_transition_source_slots_" +
                    std::to_string(transitionIdx) + ";\n";
        }
    }
    code += "};\n\n";

    code += "void " + initName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = std::make_unique<sycl::queue>(sycl::default_selector_v);\n";
    code += "    ctx.binding_split_sizes.clear();\n";
    code += "    ctx.total_items = 1;\n";
    appendPatternInitForContext(code, shell, calc, splitMeta, transportModes,
                                effectiveModes, "ctx");
    code += "    if (ctx.binding_split_sizes.empty()) ctx.binding_split_sizes.push_back(1);\n";
    code += "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code += "    ctx.item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = ctx.item_range.size();\n";
    appendWaveInitFlags(code, waveCodegenConfig, shell->getNumShellParams(),
                        distributedSitePlan.readCacheTransitions.size());
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string actualTensorName =
            resolveActualTensorName(shell->getParam(paramIdx)->getName(), dacExpr);
        code += "    ctx.pattern_" + calcName + ".bind_split_sizes = ctx.binding_split_sizes;\n";
        code += "    ctx.plan_" + calcName + " = " +
                mpi_rewriter::buildPackPlanBuilderExpr(transportModes[paramIdx],
                                                       "ctx.item_range",
                                                       "ctx.pattern_" + calcName) + ";\n";
        code += "    ctx.local_" + calcName + ".resize(ctx.plan_" + calcName + ".pack.globals.size());\n";
        if (transportModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_" + calcName +
                    ", ctx.plan_" + calcName + ".pack.globals, ctx.mpi_rank, ctx.mpi_size);\n";
            if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                code += "    if (ctx.mpi_rank == 0) dacpp::mpi::init_layout_byte_counts(ctx.input_layout_" +
                        calcName + ", sizeof(" + calc->getParam(paramIdx)->getBasicType() + "));\n";
            }
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                const std::string& tensorName = shell->getParam(paramIdx)->getName();
                const std::string mpiType =
                    mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType());
                const int sourceIdx =
                    fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
                if (sourceIdx >= 0) {
                    code += "    // DACPP fallback loop-carried input: " +
                            tensorName + " <- " +
                            shell->getParam(sourceIdx)->getName() + "\n";
                } else {
                    code += "    // DACPP fallback cached input: " + tensorName + "\n";
                }
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_" +
                        calcName + ", ctx.input_layout_" + calcName +
                        ".globals.data(), ctx.input_layout_" + calcName +
                        ".globals.size(), ctx.sendbuf_" + calcName + ");\n";
                code += "    }\n";
                if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_displs.data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                              calc->getParam(paramIdx)->getBasicType()) +
                            ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".displs.data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), ctx.input_layout_" + calcName + ".local_count, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
            }
        }
        if (transportModes[paramIdx] != IOTYPE::READ) {
            code += "    const auto& writeback_globals_" + calcName + " = ctx.plan_" + calcName +
                    ".pack.writeback_globals.empty() ? ctx.plan_" + calcName +
                    ".pack.globals : ctx.plan_" + calcName + ".pack.writeback_globals;\n";
            code += "    dacpp::mpi::init_gathered_index_layout(ctx.output_layout_" + calcName +
                    ", writeback_globals_" + calcName + ", ctx.mpi_rank, ctx.mpi_size);\n";
            if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                code += "    if (ctx.mpi_rank == 0) dacpp::mpi::init_layout_byte_counts(ctx.output_layout_" +
                        calcName + ", sizeof(" + calc->getParam(paramIdx)->getBasicType() + "));\n";
            }
            code += "    dacpp::mpi::build_local_slots_for_globals(ctx.plan_" + calcName +
                    ".pack, ctx.writeback_slots_" + calcName + ");\n";
            code += "    ctx.writeback_values_" + calcName +
                    ".resize(ctx.writeback_slots_" + calcName + ".size());\n";
            code += "    if (ctx.mpi_rank == 0) ctx.global_recv_values_" + calcName +
                    ".resize(ctx.output_layout_" + calcName + ".globals.size());\n";
        }
    }
    code += "    ctx.use_contiguous_kernel_views = true;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        code += "    ctx.use_contiguous_kernel_views = ctx.use_contiguous_kernel_views && dacpp::mpi::is_contiguous_kernel_pack_plan(ctx.plan_" +
                calcName + ", ctx.local_item_count, dacpp::mpi::partition_element_count(ctx.pattern_" +
                calcName + "));\n";
    }
    appendWaveDirectKernelMetadataInit(code, waveCodegenConfig);
    if (distributedSitePlan.supported) {
        code += "    ctx.use_partial_exchange = true;\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* shellWrapperParam = shell->getParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const IOTYPE mode = transportModes[paramIdx];
            const std::string& calcName = calcParam->getName();
            const std::string& tensorName = shellWrapperParam->getName();
            const std::string actualTensorName =
                resolveActualTensorName(tensorName, dacExpr);

            code += "    ctx.dist_" + calcName + ".enabled = true;\n";
            if (mode != IOTYPE::WRITE) {
                code += "    dacpp::mpi::init_all_rank_index_layout(ctx.dist_" + calcName +
                        ".read_layout, ctx.plan_" + calcName +
                        ".pack.globals, ctx.mpi_rank, ctx.mpi_size);\n";
                code += "    ctx.dist_" + calcName + ".local_cache.resize(ctx.plan_" +
                        calcName + ".pack.globals.size());\n";
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_" +
                        calcName + ", ctx.dist_" + calcName +
                        ".read_layout.globals.data(), ctx.dist_" + calcName +
                        ".read_layout.globals.size(), ctx.sendbuf_" + calcName + ");\n";
                code += "    }\n";
                if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_displs.data() : nullptr, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", ctx.dist_" + calcName + ".local_cache.data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                              calc->getParam(paramIdx)->getBasicType()) +
                            ", " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".displs.data() : nullptr, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", ctx.dist_" + calcName + ".local_cache.data(), ctx.input_layout_" + calcName + ".local_count, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", 0, MPI_COMM_WORLD);\n";
                }
                code += "    ctx.dist_" + calcName + ".seeded = true;\n";
                if (!useDistributedReaderMaterialize &&
                    distributedSitePlan.rootBridgeTensors.count(tensorName) != 0) {
                    code += "    ctx.dist_" + calcName + ".root_bridge_pack = dacpp::mpi::make_dense_cover_pack(static_cast<std::size_t>(" +
                            tensorName + ".getSize()));\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout = dacpp::mpi::AllRankIndexLayout{};\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.local_count = ctx.mpi_rank == 0 ? static_cast<int>(ctx.dist_" + calcName + ".root_bridge_pack.globals.size()) : 0;\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.counts.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.displs.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.globals = ctx.dist_" + calcName + ".root_bridge_pack.globals;\n";
                    code += "    if (ctx.mpi_size > 0) {\n";
                    code += "        ctx.dist_" + calcName + ".root_bridge_layout.counts[0] = static_cast<int>(ctx.dist_" + calcName + ".root_bridge_pack.globals.size());\n";
                    code += "    }\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_plan = dacpp::mpi::build_exchange_plan_from_layouts(ctx.dist_" + calcName + ".root_bridge_pack, ctx.dist_" + calcName + ".root_bridge_layout, ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                }
            } else {
                code += "    ctx.dist_" + calcName + ".local_cache.assign(ctx.plan_" + calcName + ".pack.globals.size(), " + calcParam->getBasicType() + "{});\n";
                code += "    ctx.dist_" + calcName + ".local_write_slots = ctx.writeback_slots_" + calcName + ";\n";
                code += "    ctx.dist_" + calcName + ".local_write_globals = writeback_globals_" + calcName + ";\n";
                code += "    ctx.dist_" + calcName + ".local_write_values.resize(ctx.dist_" + calcName + ".local_write_slots.size());\n";
                code += "    dacpp::mpi::init_all_rank_index_layout(ctx.dist_" + calcName +
                        ".write_layout, writeback_globals_" + calcName +
                        ", ctx.mpi_rank, ctx.mpi_size);\n";
                code += "    if (!dacpp::mpi::validate_unique_writers(ctx.dist_" + calcName + ".write_layout, ctx.mpi_size, &ctx.partial_exchange_disable_reason)) {\n";
                code += "        ctx.use_partial_exchange = false;\n";
                code += "    }\n";
                const auto followupMappings =
                    findFollowupMappingsForWriter(distributedSitePlan, tensorName);
                if (!followupMappings.empty()) {
                    code += "    ctx.dist_" + calcName + ".local_target_slots_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.wave.route_fast_paths_by_param[" +
                            std::to_string(paramIdx) + "].resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.dist_" + calcName + ".exchange_plans_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.dist_" + calcName + ".halo_plans_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.dist_" + calcName + ".halo_runtimes_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    for (std::size_t routeIdx = 0; routeIdx < followupMappings.size(); ++routeIdx) {
                        const auto* followupMapping = followupMappings[routeIdx];
                        const int readerIdx =
                            followupMapping ? followupMapping->readerParamIndex : -1;
                        if (readerIdx < 0) {
                            code += "    ctx.partial_exchange_disable_reason = \"phase-c route missing reader for WRITE tensor " + tensorName + "\";\n";
                            code += "    ctx.use_partial_exchange = false;\n";
                            continue;
                        }
                        const std::string& readerCalcName = calc->getParam(readerIdx)->getName();
                        if (followupMapping->rank == 2) {
                            const std::string writerCols =
                                calcName + "_cols";
                            const std::string readerCols =
                                readerCalcName + "_cols";
                            const std::string rowOffset =
                                std::to_string(followupMapping->targetRowOffset);
                            const std::string colOffset =
                                std::to_string(followupMapping->targetColOffset);
                            code += "    const int " + writerCols + " = " +
                                    shell->getParam(paramIdx)->getName() + ".getShape(1);\n";
                            code += "    const int " + readerCols + " = " +
                                    shell->getParam(readerIdx)->getName() + ".getShape(1);\n";
                            code += "    dacpp::mpi::build_target_slots_for_globals_2d_offset(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, " + writerCols + ", " + readerCols + ", " + rowOffset + ", " + colOffset + ", ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                            code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts_2d_offset(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, " + writerCols + ", " + readerCols + ", " + rowOffset + ", " + colOffset + ", ctx.mpi_rank, ctx.mpi_size);\n";
                        } else {
                            const std::string targetOffset =
                                std::to_string(followupMapping->targetOffset);
                            if (followupMapping->targetOffset != 0) {
                                code += "    dacpp::mpi::build_target_slots_for_globals_with_offset(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, " + targetOffset + ", ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                                code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts_with_target_offset(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, " + targetOffset + ", ctx.mpi_rank, ctx.mpi_size);\n";
                            } else {
                                code += "    dacpp::mpi::build_target_slots_for_globals(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                                code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                            }
                        }
                        code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "].supported) {\n";
                        code += "        ctx.partial_exchange_disable_reason = ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "].unsupported_reason;\n";
                        code += "        ctx.use_partial_exchange = false;\n";
                        code += "    }\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_halo_plan_from_exchange_plan(ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "]);\n";
                        code += "    if (ctx.dist_" + calcName + ".halo_plans_by_route[" + std::to_string(routeIdx) + "].supported) {\n";
                        code += "        dacpp::mpi::prepare_halo_exchange_runtime(ctx.dist_" + calcName + ".halo_plans_by_route[" + std::to_string(routeIdx) + "], ctx.dist_" + calcName + ".halo_runtimes_by_route[" + std::to_string(routeIdx) + "]);\n";
                        code += "    }\n";
                        if (routeIdx == 0) {
                            appendWaveRouteFastPathInit(
                                code, waveCodegenConfig,
                                waveRouteFastPathExpr(paramIdx,
                                                      std::to_string(routeIdx)),
                                "ctx.dist_" + calcName,
                                std::to_string(routeIdx));
                        }
                        llvm::outs() << "[DACPP][MPI][PhaseC] halo-exchange enabled route="
                                     << tensorName << "->"
                                     << shell->getParam(readerIdx)->getName()
                                     << "\n";
                    }
                    code += "    if (!ctx.dist_" + calcName + ".local_target_slots_by_route.empty()) ctx.dist_" + calcName + ".local_target_slots = ctx.dist_" + calcName + ".local_target_slots_by_route.front();\n";
                    code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route.empty()) ctx.dist_" + calcName + ".exchange_plan = ctx.dist_" + calcName + ".exchange_plans_by_route.front();\n";
                    code += "    if (!ctx.dist_" + calcName + ".halo_plans_by_route.empty()) ctx.dist_" + calcName + ".halo_plan = ctx.dist_" + calcName + ".halo_plans_by_route.front();\n";
                } else if (!distributedSitePlan.hasRootBridge) {
                    const int readerIdx = findDefaultReaderIndex(transportModes);
                    if (readerIdx >= 0) {
                        const std::string& readerCalcName = calc->getParam(readerIdx)->getName();
                        code += "    ctx.dist_" + calcName + ".local_target_slots_by_route.resize(1);\n";
                        code += "    ctx.wave.route_fast_paths_by_param[" +
                                std::to_string(paramIdx) + "].resize(1);\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plans_by_route.resize(1);\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route.resize(1);\n";
                        code += "    ctx.dist_" + calcName + ".halo_runtimes_by_route.resize(1);\n";
                        code += "    dacpp::mpi::build_target_slots_for_globals(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, ctx.dist_" + calcName + ".local_target_slots_by_route[0]);\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[0] = dacpp::mpi::build_exchange_plan_from_layouts(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                        code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route[0].supported) {\n";
                        code += "        ctx.partial_exchange_disable_reason = ctx.dist_" + calcName + ".exchange_plans_by_route[0].unsupported_reason;\n";
                        code += "        ctx.use_partial_exchange = false;\n";
                        code += "    }\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route[0] = dacpp::mpi::build_halo_plan_from_exchange_plan(ctx.dist_" + calcName + ".exchange_plans_by_route[0]);\n";
                        code += "    if (ctx.dist_" + calcName + ".halo_plans_by_route[0].supported) {\n";
                        code += "        dacpp::mpi::prepare_halo_exchange_runtime(ctx.dist_" + calcName + ".halo_plans_by_route[0], ctx.dist_" + calcName + ".halo_runtimes_by_route[0]);\n";
                        code += "    }\n";
                        appendWaveRouteFastPathInit(
                            code, waveCodegenConfig,
                            waveRouteFastPathExpr(paramIdx, "0"),
                            "ctx.dist_" + calcName, "0");
                        code += "    ctx.dist_" + calcName + ".local_target_slots = ctx.dist_" + calcName + ".local_target_slots_by_route.front();\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plan = ctx.dist_" + calcName + ".exchange_plans_by_route.front();\n";
                        code += "    ctx.dist_" + calcName + ".halo_plan = ctx.dist_" + calcName + ".halo_plans_by_route.front();\n";
                        llvm::outs() << "[DACPP][MPI][PhaseC] halo-exchange enabled route="
                                     << tensorName << "->"
                                     << shell->getParam(readerIdx)->getName()
                                     << "\n";
                    } else {
                        code += "    ctx.partial_exchange_disable_reason = \"phase-c could not find reader for WRITE tensor " + tensorName + "\";\n";
                        code += "    ctx.use_partial_exchange = false;\n";
                }
            }
        }
        if (useDistributedReaderMaterialize &&
            paramIdx + 1 == shell->getNumShellParams()) {
            for (std::size_t updateIdx = 0;
                 updateIdx < distributedSitePlan.boundaryLocalUpdates.size();
                 ++updateIdx) {
                const auto& update =
                    distributedSitePlan.boundaryLocalUpdates[updateIdx];
                if (update.paramIndex < 0 ||
                    update.paramIndex >= shell->getNumShellParams()) {
                    continue;
                }
                Param* calcParam = calc->getParam(update.paramIndex);
                Param* shellParam = shell->getParam(update.paramIndex);
                const std::string& calcName = calcParam->getName();
                const std::string& tensorName = shellParam->getName();
                std::string sourceCalcName = calcName;
                std::string sourceTensorName = tensorName;
                if (update.sourceParamIndex >= 0 &&
                    update.sourceParamIndex < shell->getNumShellParams()) {
                    sourceCalcName = calc->getParam(update.sourceParamIndex)->getName();
                    sourceTensorName = shell->getParam(update.sourceParamIndex)->getName();
                }
                const std::string rowExpr =
                    update.targetRowUsesLoop ? "__dacpp_idx" : update.targetRowExpr;
                const std::string colExpr =
                    update.targetColUsesLoop ? "__dacpp_idx" : update.targetColExpr;
                const std::string sourceRowExpr =
                    update.sourceRowUsesLoop ? "__dacpp_idx" : update.sourceRowExpr;
                const std::string sourceColExpr =
                    update.sourceColUsesLoop ? "__dacpp_idx" : update.sourceColExpr;
                const std::string idx = std::to_string(updateIdx);
                code += "    // DACPP boundary-local slot plan: " +
                        tensorName + "\n";
                code += "    ctx.boundary_local_target_slots_" + idx + ".clear();\n";
                code += "    ctx.boundary_local_source_slots_" + idx + ".clear();\n";
                if (update.rank == 1) {
                    code += "    {\n";
                    code += "        const int64_t __dacpp_size = static_cast<int64_t>(" +
                            tensorName + ".getSize());\n";
                    code += "        const int64_t __dacpp_target_global = " +
                            (update.targetRowUsesLoop ? "__dacpp_idx" : update.targetRowExpr) +
                            ";\n";
                    code += "        if (__dacpp_target_global >= 0 && __dacpp_target_global < __dacpp_size) {\n";
                    code += "            const int32_t __dacpp_target_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                            calcName + ".pack, __dacpp_target_global);\n";
                    code += "            if (__dacpp_target_slot >= 0) {\n";
                    if (update.constantRhs) {
                        code += "                ctx.boundary_local_target_slots_" + idx +
                                ".push_back(__dacpp_target_slot);\n";
                    } else {
                        code += "                const int64_t __dacpp_source_size = static_cast<int64_t>(" +
                                sourceTensorName + ".getSize());\n";
                        code += "                const int64_t __dacpp_source_global = " +
                                (update.sourceRowUsesLoop ? "__dacpp_idx" : update.sourceRowExpr) +
                                ";\n";
                        code += "                if (__dacpp_source_global >= 0 && __dacpp_source_global < __dacpp_source_size) {\n";
                        code += "                    const int32_t __dacpp_source_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                                sourceCalcName + ".pack, __dacpp_source_global);\n";
                        code += "                    if (__dacpp_source_slot >= 0) {\n";
                        code += "                        ctx.boundary_local_target_slots_" + idx +
                                ".push_back(__dacpp_target_slot);\n";
                        code += "                        ctx.boundary_local_source_slots_" + idx +
                                ".push_back(__dacpp_source_slot);\n";
                        code += "                    }\n";
                        code += "                }\n";
                    }
                    code += "            }\n";
                    code += "        }\n";
                    code += "    }\n";
                    continue;
                }
                code += "    {\n";
                code += "        const int64_t __dacpp_rows = static_cast<int64_t>(" +
                        tensorName + ".getShape(0));\n";
                code += "        const int64_t __dacpp_cols = static_cast<int64_t>(" +
                        tensorName + ".getShape(1));\n";
                const std::string boundaryEndExpr =
                    (update.targetRowUsesLoop || update.sourceRowUsesLoop)
                        ? "__dacpp_rows"
                        : "__dacpp_cols";
                code += "        const int64_t __dacpp_end = " +
                        boundaryEndExpr + ";\n";
                code += "        for (int64_t __dacpp_idx = 0; __dacpp_idx < __dacpp_end; ++__dacpp_idx) {\n";
                code += "            const int64_t __dacpp_target_row = " + rowExpr + ";\n";
                code += "            const int64_t __dacpp_target_col = " + colExpr + ";\n";
                code += "            if (__dacpp_target_row < 0 || __dacpp_target_col < 0 || __dacpp_target_row >= __dacpp_rows || __dacpp_target_col >= __dacpp_cols) continue;\n";
                code += "            const int64_t __dacpp_target_global = __dacpp_target_row * __dacpp_cols + __dacpp_target_col;\n";
                code += "            const int32_t __dacpp_target_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                        calcName + ".pack, __dacpp_target_global);\n";
                code += "            if (__dacpp_target_slot < 0) continue;\n";
                if (update.constantRhs) {
                    code += "            ctx.boundary_local_target_slots_" + idx +
                            ".push_back(__dacpp_target_slot);\n";
                } else {
                    code += "            const int64_t __dacpp_source_row = " + sourceRowExpr + ";\n";
                    code += "            const int64_t __dacpp_source_col = " + sourceColExpr + ";\n";
                    code += "            if (__dacpp_source_row < 0 || __dacpp_source_col < 0 || __dacpp_source_row >= __dacpp_rows || __dacpp_source_col >= __dacpp_cols) continue;\n";
                    code += "            const int64_t __dacpp_source_global = __dacpp_source_row * __dacpp_cols + __dacpp_source_col;\n";
                    code += "            const int32_t __dacpp_source_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                            calcName + ".pack, __dacpp_source_global);\n";
                    code += "            if (__dacpp_source_slot < 0) continue;\n";
                    code += "            ctx.boundary_local_target_slots_" + idx +
                            ".push_back(__dacpp_target_slot);\n";
                    code += "            ctx.boundary_local_source_slots_" + idx +
                            ".push_back(__dacpp_source_slot);\n";
                }
                code += "        }\n";
                code += "    }\n";
            }
        }
    }
    if (!distributedSitePlan.readCacheTransitions.empty()) {
        for (std::size_t transitionIdx = 0;
             transitionIdx < distributedSitePlan.readCacheTransitions.size();
             ++transitionIdx) {
                const auto& transition =
                    distributedSitePlan.readCacheTransitions[transitionIdx];
                if (transition.writerParamIndex < 0 ||
                    transition.readerParamIndex < 0 ||
                    transition.rank != 2) {
                    code += "    ctx.partial_exchange_disable_reason = \"phase-c unsupported read-cache transition\";\n";
                    code += "    ctx.use_partial_exchange = false;\n";
                    continue;
                }
                const std::string& writerCalcName =
                    calc->getParam(transition.writerParamIndex)->getName();
                const std::string& readerCalcName =
                    calc->getParam(transition.readerParamIndex)->getName();
                const std::string writerCols = writerCalcName + "_read_cache_cols_" +
                                               std::to_string(transitionIdx);
                const std::string readerCols = readerCalcName + "_read_cache_cols_" +
                                               std::to_string(transitionIdx);
                const std::string idx = std::to_string(transitionIdx);
                code += "    // DACPP read-cache state transition slot plan: " +
                        transition.writerTensor + " -> " +
                        transition.readerTensor + "\n";
                code += "    ctx.read_cache_transition_target_slots_" + idx + ".clear();\n";
                code += "    ctx.read_cache_transition_source_slots_" + idx + ".clear();\n";
                code += "    {\n";
                code += "        const int " + writerCols + " = " +
                        shell->getParam(transition.writerParamIndex)->getName() +
                        ".getShape(1);\n";
                code += "        const int " + readerCols + " = " +
                        shell->getParam(transition.readerParamIndex)->getName() +
                        ".getShape(1);\n";
                code += "        const auto& __dacpp_transition_globals = ctx.plan_" +
                        writerCalcName + ".pack.globals;\n";
                code += "        for (int64_t __dacpp_source_global : __dacpp_transition_globals) {\n";
                code += "            const int32_t __dacpp_source_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                        writerCalcName + ".pack, __dacpp_source_global);\n";
                code += "            if (__dacpp_source_slot < 0) continue;\n";
                code += "            const int64_t __dacpp_target_global = dacpp::mpi::map_2d_global_with_offset(__dacpp_source_global, " +
                        writerCols + ", " + readerCols + ", " +
                        std::to_string(transition.targetRowOffset) + ", " +
                        std::to_string(transition.targetColOffset) + ");\n";
                code += "            if (__dacpp_target_global < 0) continue;\n";
                code += "            const int32_t __dacpp_target_slot = dacpp::mpi::try_lookup_local_slot(ctx.plan_" +
                        readerCalcName + ".pack, __dacpp_target_global);\n";
                code += "            if (__dacpp_target_slot < 0) continue;\n";
                code += "            ctx.read_cache_transition_source_slots_" + idx + ".push_back(__dacpp_source_slot);\n";
                code += "            ctx.read_cache_transition_target_slots_" + idx + ".push_back(__dacpp_target_slot);\n";
                code += "        }\n";
                code += "    }\n";
                appendWaveReadTransitionFastPathInit(
                    code, waveCodegenConfig, idx,
                    waveReadTransitionFastPathExpr(idx));
            }
        }
    } else {
        code += "    ctx.use_partial_exchange = false;\n";
        code += "    ctx.partial_exchange_disable_reason = \"" +
                distributedSitePlan.disableReason + "\";\n";
    }
    code += "}\n\n";

    code += "void " + runName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    int mpi_size = ctx.mpi_size;\n";
    code += "    dacpp::mpi::resetCollectPositionsProfile();\n";
    code += "    auto dacpp_wrapper_start = std::chrono::steady_clock::now();\n";
    code += "    auto& q = *ctx.q;\n";
    code += "    const int64_t local_item_count = ctx.local_item_count;\n";
    code += "    const bool dacpp_profile_enabled = dacpp::mpi::profilingEnabled();\n";
    code += "    double dacpp_profile_input_ms = 0.0;\n";
    code += "    double dacpp_profile_dist_setup_ms = 0.0;\n";
    code += "    double dacpp_profile_kernel_ms = 0.0;\n";
    code += "    double dacpp_profile_read_transition_ms = 0.0;\n";
    code += "    double dacpp_profile_publish_ms = 0.0;\n";
    code += "    double dacpp_profile_boundary_ms = 0.0;\n";
    code += "    double dacpp_profile_root_bridge_ms = 0.0;\n";
    code += "    double dacpp_profile_writeback_ms = 0.0;\n";
    code += "    auto dacpp_profile_now = [&]() {\n";
    code += "        return dacpp_profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};\n";
    code += "    };\n";
    code += "    auto dacpp_profile_add = [&](double& bucket, std::chrono::steady_clock::time_point start) {\n";
    code += "        if (dacpp_profile_enabled) {\n";
    code += "            bucket += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();\n";
    code += "        }\n";
    code += "    };\n";
    code += "    if (!ctx.use_partial_exchange) {\n";
    code += "    auto dacpp_profile_input_start = dacpp_profile_now();\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const IOTYPE mode = transportModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string slotsName = "ctx.plan_" + calcName + ".compact_slots";
        const std::string keyOffsetsName = "ctx.plan_" + calcName + ".item_key_offsets";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

        code += "    auto& " + localName + " = ctx.local_" + calcName + ";\n";
        if (mode != IOTYPE::WRITE) {
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                const int sourceIdx =
                    fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
                if (sourceIdx >= 0) {
                    code += "    // DACPP fallback loop-carried input reused: " +
                            tensorName + "\n";
                } else {
                    code += "    // DACPP fallback cached input reused: " + tensorName + "\n";
                }
            } else {
                code += "    auto& input_layout_" + calcName + " = ctx.input_layout_" + calcName + ";\n";
                code += "    auto& global_" + calcName + " = ctx.global_" + calcName + ";\n";
                code += "    auto& sendbuf_" + calcName + " = ctx.sendbuf_" + calcName + ";\n";
                code += "    if (mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(global_" + calcName +
                        ", input_layout_" + calcName + ".globals.data(), input_layout_" + calcName +
                        ".globals.size(), sendbuf_" + calcName + ");\n";
                code += "    }\n";
                code += "    " + localName + ".resize(static_cast<std::size_t>(input_layout_" + calcName + ".local_count));\n";
                if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                    code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".byte_counts.data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".byte_displs.data() : nullptr, " + mpiType + ", " + localName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("input_layout_" + calcName + ".local_count",
                                                              calcParam->getBasicType()) +
                            ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".counts.data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".displs.data() : nullptr, " + mpiType + ", " + localName + ".data(), input_layout_" + calcName + ".local_count, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
            }
        } else {
            code += "    " + localName + ".assign(" + packName + ".globals.size(), " + calcParam->getBasicType() + "{});\n";
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" + calcName + "));\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = ctx.pattern_" + calcName + ".partition_shape[1];\n";
        }
    }

    code += "    dacpp_profile_add(dacpp_profile_input_ms, dacpp_profile_input_start);\n";
    code += "    auto dacpp_profile_kernel_start = dacpp_profile_now();\n";
    code += "    if (local_item_count > 0) {\n";
    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name +
                "(ctx.plan_" + name + ".compact_slots.data(), sycl::range<1>(ctx.plan_" + name +
                ".compact_slots.size()));\n";
        code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                "(ctx.plan_" + name + ".item_key_offsets.data(), sycl::range<1>(ctx.plan_" + name +
                ".item_key_offsets.size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                mpi_rewriter::toAccessorMode(effectiveModes[paramIdx]) + ">(h);\n";
        code += "                auto slots_acc_" + name +
                " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
        code += "                auto key_offsets_acc_" + name +
                " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                    auto* data_" + name +
                " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* slots_" + name +
                " = slots_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* key_offsets_" + name +
                " = key_offsets_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear]};\n";
        } else {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear], " + name + "_cols};\n";
        }
    }
    code += "                    " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp_profile_add(dacpp_profile_kernel_ms, dacpp_profile_kernel_start);\n";

    code += "    auto dacpp_profile_writeback_start = dacpp_profile_now();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (transportModes[paramIdx] == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_out_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());
        const mpi_rewriter::OutputSyncRequirement syncRequirement =
            mpi_rewriter::classifyOutputSyncRequirement(dacppFile, tensorName, dacExpr);
        const bool needsBcast =
            syncRequirement == mpi_rewriter::OutputSyncRequirement::DistributedFollowup
                ? true
                : mpi_rewriter::requiresBroadcast(syncRequirement);
        llvm::outs() << "[DACPP][MPI] output " << tensorName
                     << " sync="
                     << mpi_rewriter::outputSyncRequirementName(syncRequirement)
                     << "\n";

        code += "    auto& output_layout_" + calcName + " = ctx.output_layout_" + calcName + ";\n";
        code += "    auto& writeback_values_" + calcName + " = ctx.writeback_values_" + calcName + ";\n";
        code += "    auto& global_recv_values_" + calcName + " = ctx.global_recv_values_" + calcName + ";\n";
        code += "    auto& global_out_" + calcName + " = ctx.global_out_" + calcName + ";\n";
        code += "    dacpp::mpi::pack_values_by_slots_parallel_into(" + localName +
                ", ctx.writeback_slots_" + calcName + ", writeback_values_" + calcName + ");\n";
        if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr("output_layout_" + calcName + ".local_count",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName +
                    ".data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".byte_counts.data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), output_layout_" + calcName + ".local_count, " + mpiType +
                    ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? output_layout_" +
                    calcName + ".counts.data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + tensorName + ".tensor2Array(global_out_" + calcName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName +
                ", output_layout_" + calcName + ".globals, global_out_" + calcName + ");\n";
        code += "        " + tensorName + ".array2Tensor(global_out_" + calcName + ");\n";
        if (needsBcast) {
            code += "        if (!global_out_" + calcName + ".empty()) {\n";
            code += "            MPI_Bcast(global_out_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
        }
        code += "    } else ";
        if (needsBcast) {
            code += "{\n";
            code += "        global_out_" + calcName + ".resize(static_cast<std::size_t>(" +
                    tensorName + ".getSize()));\n";
            code += "        if (!global_out_" + calcName + ".empty()) {\n";
            code += "            MPI_Bcast(global_out_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
            code += "        " + tensorName + ".array2Tensor(global_out_" + calcName + ");\n";
            code += "    }\n";
        } else {
            code += "{\n";
            code += "    }\n";
        }
        for (int readerIdx = 0; readerIdx < shell->getNumShellParams(); ++readerIdx) {
            if (fallbackInputRefreshSource[static_cast<std::size_t>(readerIdx)] !=
                paramIdx) {
                continue;
            }
            const std::string& readerCalcName =
                calc->getParam(readerIdx)->getName();
            const std::string& readerTensorName =
                shell->getParam(readerIdx)->getName();
            code += "    // DACPP fallback loop-carried input refreshed: " +
                    readerTensorName + " <- " + tensorName + "\n";
            code += "    dacpp::mpi::pack_values_by_globals_parallel_range_into(global_out_" +
                    calcName + ", ctx.plan_" + readerCalcName +
                    ".pack.globals.data(), ctx.plan_" + readerCalcName +
                    ".pack.globals.size(), ctx.local_" + readerCalcName + ");\n";
        }
    }
    code += "    dacpp_profile_add(dacpp_profile_writeback_ms, dacpp_profile_writeback_start);\n";
    if (distributedSitePlan.supported) {
        code += "    } else {\n";
        code += "    auto dacpp_profile_dist_setup_start = dacpp_profile_now();\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            code += "    auto& local_" + calcName + " = ctx.dist_" + calcName +
                    ".local_cache;\n";
            if (transportModes[paramIdx] == IOTYPE::WRITE) {
                if (!appendWaveDistributedWriteReset(
                        code, waveCodegenConfig, calcName,
                        calcParam->getBasicType())) {
                    code += "    local_" + calcName + ".assign(ctx.plan_" + calcName +
                            ".pack.globals.size(), " + calcParam->getBasicType() + "{});\n";
                }
            }
            code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" +
                    calcName + "));\n";
            if (mpi_rewriter::inferViewRank(shell->getShellParam(paramIdx), calcParam) > 1) {
                code += "    const int " + calcName + "_cols = ctx.pattern_" +
                        calcName + ".partition_shape[1];\n";
            }
        }
        code += "    dacpp_profile_add(dacpp_profile_dist_setup_ms, dacpp_profile_dist_setup_start);\n";
        code += "    auto dacpp_profile_kernel_start = dacpp_profile_now();\n";
        code += "    if (local_item_count > 0) {\n";
        appendWaveDistributedKernelDispatchHead(code, waveCodegenConfig);
        code += "            {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                    "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
        }
        code += "                q.submit([&](sycl::handler& h) {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                    auto acc_" + name + " = buffer_" + name + ".get_access<" +
                    mpi_rewriter::toAccessorMode(effectiveModes[paramIdx]) + ">(h);\n";
        }
        code += "                    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
        code += "                        const int item_linear = static_cast<int>(idx[0]);\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            ShellParam* shellParam = shell->getShellParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                        auto* data_" + name +
                    " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                        const int key_offset_" + name + " = item_linear * " +
                    name + "_partition_size;\n";
            if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
                code += "                        dacpp::mpi::ContiguousView1D<" +
                        mpi_rewriter::viewElementType(calcParam, effectiveModes[paramIdx]) +
                        "> view_" + name + "{data_" + name + ", key_offset_" + name + "};\n";
            } else {
                code += "                        dacpp::mpi::ContiguousView2D<" +
                        mpi_rewriter::viewElementType(calcParam, effectiveModes[paramIdx]) +
                        "> view_" + name + "{data_" + name + ", key_offset_" + name + ", " +
                        name + "_cols};\n";
            }
        }
        code += "                        " + calc->getName() + "_mpi_local(";
        for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
            code += "view_" + calc->getParam(paramIdx)->getName();
            if (paramIdx + 1 != calc->getNumParams()) {
                code += ", ";
            }
        }
        code += ");\n";
        code += "                    });\n";
        code += "                });\n";
        code += "                q.wait();\n";
        code += "            }\n";
        code += "        } else {\n";
        code += "        {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                    "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
            code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name +
                    "(ctx.plan_" + name + ".compact_slots.data(), sycl::range<1>(ctx.plan_" + name +
                    ".compact_slots.size()));\n";
            code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                    "(ctx.plan_" + name + ".item_key_offsets.data(), sycl::range<1>(ctx.plan_" + name +
                    ".item_key_offsets.size()));\n";
        }
        code += "            q.submit([&](sycl::handler& h) {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                    mpi_rewriter::toAccessorMode(effectiveModes[paramIdx]) + ">(h);\n";
            code += "                auto slots_acc_" + name +
                    " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
            code += "                auto key_offsets_acc_" + name +
                    " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
        }
        code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
        code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            ShellParam* shellParam = shell->getShellParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                    auto* data_" + name +
                    " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                    auto* slots_" + name +
                    " = slots_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                    auto* key_offsets_" + name +
                    " = key_offsets_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                        " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                        "[item_linear]};\n";
            } else {
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                        " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                        "[item_linear], " + name + "_cols};\n";
            }
        }
        code += "                    " + calc->getName() + "_mpi_local(";
        for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
            code += "view_" + calc->getParam(paramIdx)->getName();
            if (paramIdx + 1 != calc->getNumParams()) {
                code += ", ";
            }
        }
        code += ");\n";
        code += "                });\n";
        code += "            });\n";
        code += "            q.wait();\n";
        code += "        }\n";
        code += "        }\n";
        code += "    }\n";
        code += "    dacpp_profile_add(dacpp_profile_kernel_ms, dacpp_profile_kernel_start);\n";
        if (!distributedSitePlan.readCacheTransitions.empty()) {
            code += "    auto dacpp_profile_read_transition_start = dacpp_profile_now();\n";
            for (std::size_t transitionIdx = 0;
                 transitionIdx < distributedSitePlan.readCacheTransitions.size();
                 ++transitionIdx) {
                const auto& transition =
                    distributedSitePlan.readCacheTransitions[transitionIdx];
                if (transition.writerParamIndex < 0 ||
                    transition.readerParamIndex < 0) {
                    continue;
                }
                const std::string& writerCalcName =
                    calc->getParam(transition.writerParamIndex)->getName();
                const std::string& readerCalcName =
                    calc->getParam(transition.readerParamIndex)->getName();
                const std::string idx = std::to_string(transitionIdx);
                code += "    // DACPP read-cache state transition: " +
                        transition.writerTensor + " -> " +
                        transition.readerTensor + "\n";
                appendWaveReadTransitionRun(
                    code, waveCodegenConfig, idx,
                    waveReadTransitionFastPathExpr(idx),
                    writerCalcName,
                    readerCalcName);
            }
            code += "    dacpp_profile_add(dacpp_profile_read_transition_ms, dacpp_profile_read_transition_start);\n";
        }
        code += "    auto dacpp_profile_publish_start = dacpp_profile_now();\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (transportModes[paramIdx] == IOTYPE::READ) {
                continue;
            }
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            Param* shellWrapperParam = shell->getParam(paramIdx);
            const std::string& tensorName = shellWrapperParam->getName();
            const auto followupMappings =
                findFollowupMappingsForWriter(distributedSitePlan, tensorName);
            const std::size_t routeCount =
                !followupMappings.empty()
                    ? followupMappings.size()
                    : (distributedSitePlan.hasRootBridge ? 0 : 1);
            for (std::size_t routeIdx = 0; routeIdx < routeCount; ++routeIdx) {
                const auto* followupMapping =
                    !followupMappings.empty() ? followupMappings[routeIdx] : nullptr;
                std::string targetCache = "ctx.dist_" + calcName + ".local_cache";
                if (followupMapping && followupMapping->readerParamIndex >= 0) {
                    targetCache = "ctx.dist_" +
                                  calc->getParam(followupMapping->readerParamIndex)->getName() +
                                  ".local_cache";
                }
                appendWavePublishRun(
                    code, waveCodegenConfig, useDistributedReaderMaterialize,
                    calcName,
                    waveRouteFastPathExpr(paramIdx, std::to_string(routeIdx)),
                    std::to_string(routeIdx), targetCache);
            }
        }
        code += "    dacpp_profile_add(dacpp_profile_publish_ms, dacpp_profile_publish_start);\n";
        if (useDistributedReaderMaterialize) {
            code += "    auto dacpp_profile_boundary_start = dacpp_profile_now();\n";
            for (const auto& update : distributedSitePlan.boundaryLocalUpdates) {
                if (update.paramIndex < 0 ||
                    update.paramIndex >= shell->getNumShellParams()) {
                    continue;
                }
                Param* calcParam = calc->getParam(update.paramIndex);
                Param* shellParam = shell->getParam(update.paramIndex);
                const std::string& calcName = calcParam->getName();
                const std::string& tensorName = shellParam->getName();
                std::string sourceCalcName = calcName;
                if (update.sourceParamIndex >= 0 &&
                    update.sourceParamIndex < shell->getNumShellParams()) {
                    sourceCalcName = calc->getParam(update.sourceParamIndex)->getName();
                }
                const std::string idx = std::to_string(
                    static_cast<std::size_t>(&update - distributedSitePlan.boundaryLocalUpdates.data()));
                code += "    // DACPP distributed boundary-local update: " +
                        tensorName + "\n";
                code += "    {\n";
                if (update.constantRhs) {
                    code += "        for (int32_t __dacpp_target_slot : ctx.boundary_local_target_slots_" +
                            idx + ") {\n";
                    code += "            local_" + calcName +
                            "[static_cast<std::size_t>(__dacpp_target_slot)] = static_cast<" +
                            calcParam->getBasicType() + ">(" +
                            (update.constantValue.empty() ? "0" : update.constantValue) + ");\n";
                    code += "        }\n";
                } else {
                    code += "        const std::size_t __dacpp_boundary_count = ctx.boundary_local_target_slots_" +
                            idx + ".size();\n";
                    code += "        for (std::size_t __dacpp_boundary_i = 0; __dacpp_boundary_i < __dacpp_boundary_count; ++__dacpp_boundary_i) {\n";
                    code += "            const int32_t __dacpp_target_slot = ctx.boundary_local_target_slots_" +
                            idx + "[__dacpp_boundary_i];\n";
                    code += "            const int32_t __dacpp_source_slot = ctx.boundary_local_source_slots_" +
                            idx + "[__dacpp_boundary_i];\n";
                    code += "            local_" + calcName +
                            "[static_cast<std::size_t>(__dacpp_target_slot)] = local_" +
                            sourceCalcName + "[static_cast<std::size_t>(__dacpp_source_slot)];\n";
                    code += "        }\n";
                }
                code += "    }\n";
            }
            code += "    dacpp_profile_add(dacpp_profile_boundary_ms, dacpp_profile_boundary_start);\n";
        }
        if (distributedSitePlan.hasRootBridge && !useDistributedReaderMaterialize) {
            code += "    auto dacpp_profile_root_bridge_start = dacpp_profile_now();\n";
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
                if (transportModes[paramIdx] == IOTYPE::READ) {
                    continue;
                }
                Param* calcParam = calc->getParam(paramIdx);
                Param* shellWrapperParam = shell->getParam(paramIdx);
                const std::string& calcName = calcParam->getName();
                const std::string& tensorName = shellWrapperParam->getName();
                const std::string mpiType =
                    mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

                code += "    dacpp::mpi::pack_values_by_slots_parallel_into(local_" + calcName +
                        ", ctx.writeback_slots_" + calcName + ", ctx.writeback_values_" +
                        calcName + ");\n";
                if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                    code += "    MPI_Gatherv(ctx.writeback_values_" + calcName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.output_layout_" + calcName + ".local_count",
                                                              calcParam->getBasicType()) +
                            ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" +
                            calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                            calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                            calcName + ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Gatherv(ctx.writeback_values_" + calcName +
                            ".data(), ctx.output_layout_" + calcName + ".local_count, " + mpiType +
                            ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" + calcName +
                            ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + calcName +
                            ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + calcName +
                            ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_out_" + calcName + ");\n";
                code += "        dacpp::mpi::apply_writeback_by_globals(ctx.global_recv_values_" + calcName +
                        ", ctx.output_layout_" + calcName + ".globals, ctx.global_out_" + calcName + ");\n";
                code += "        " + tensorName + ".array2Tensor(ctx.global_out_" + calcName + ");\n";
                code += "    }\n";
            }
            code += "    dacpp_profile_add(dacpp_profile_root_bridge_ms, dacpp_profile_root_bridge_start);\n";
        }
        code += "    }\n";
    } else {
        code += "    }\n";
    }

    code += "    if (dacpp::mpi::profilingEnabled()) {\n";
    code += "        auto dacpp_wrapper_end = std::chrono::steady_clock::now();\n";
    code += "        double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();\n";
    code += "        double dacpp_wrapper_max_ms = 0.0;\n";
    code += "        MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);\n";
    code += "        double dacpp_profile_local_parts[8] = {dacpp_profile_input_ms, dacpp_profile_dist_setup_ms, dacpp_profile_kernel_ms, dacpp_profile_read_transition_ms, dacpp_profile_publish_ms, dacpp_profile_boundary_ms, dacpp_profile_root_bridge_ms, dacpp_profile_writeback_ms};\n";
    code += "        double dacpp_profile_max_parts[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};\n";
    code += "        MPI_Reduce(dacpp_profile_local_parts, dacpp_profile_max_parts, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);\n";
    code += "        if (mpi_rank == 0) {\n";
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\\n\", \"" + wrapper + "\", dacpp_wrapper_max_ms);\n";
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] run_breakdown_ms(max): input=%.3f dist_setup=%.3f kernel=%.3f read_transition=%.3f publish=%.3f boundary=%.3f root_bridge=%.3f writeback=%.3f\\n\", \"" + wrapper + "\", dacpp_profile_max_parts[0], dacpp_profile_max_parts[1], dacpp_profile_max_parts[2], dacpp_profile_max_parts[3], dacpp_profile_max_parts[4], dacpp_profile_max_parts[5], dacpp_profile_max_parts[6], dacpp_profile_max_parts[7]);\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp::mpi::reportCollectPositionsProfile(\"" + wrapper + "\", MPI_COMM_WORLD);\n";
    code += "}\n\n";

    code += "void " + materializeName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    if (!ctx.use_partial_exchange) {\n";
    code += "        return;\n";
    code += "    }\n";
    if (!distributedSitePlan.supported) {
        code += "    return;\n";
    } else if (useDistributedReaderMaterialize) {
        for (int readerParamIdx : distributedMaterializeReaderParams) {
            if (readerParamIdx < 0 ||
                readerParamIdx >= shell->getNumShellParams()) {
                continue;
            }
            Param* readerCalcParam = calc->getParam(readerParamIdx);
            Param* readerShellParam = shell->getParam(readerParamIdx);
            const std::string& readerCalcName = readerCalcParam->getName();
            const std::string& readerTensorName = readerShellParam->getName();
            const std::string mpiType =
                mpi_rewriter::mpiDatatypeFor(readerCalcParam->getBasicType());
            if (mpi_rewriter::usesByteTransport(readerCalcParam->getBasicType())) {
                code += "    MPI_Gatherv(ctx.dist_" + readerCalcName + ".local_cache.data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + readerCalcName + ".local_count",
                                                          readerCalcParam->getBasicType()) +
                        ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        readerCalcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        readerCalcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        readerCalcName + ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Gatherv(ctx.dist_" + readerCalcName +
                        ".local_cache.data(), ctx.input_layout_" + readerCalcName +
                        ".local_count, " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        readerCalcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        readerCalcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        readerCalcName + ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + readerTensorName + ".tensor2Array(ctx.global_" + readerCalcName + ");\n";
            code += "        dacpp::mpi::apply_writeback_by_globals(ctx.sendbuf_" + readerCalcName +
                    ", ctx.input_layout_" + readerCalcName + ".globals, ctx.global_" +
                    readerCalcName + ");\n";
            code += "        " + readerTensorName + ".array2Tensor(ctx.global_" + readerCalcName + ");\n";
            code += "    }\n";
        }
    } else if (distributedSitePlan.hasRootBridge) {
        code += "    return;\n";
    } else if (!distributedSitePlan.followupMappings.empty()) {
        for (const auto& mapping : distributedSitePlan.followupMappings) {
            if (mapping.writerParamIndex < 0 || mapping.readerParamIndex < 0) {
                continue;
            }
            Param* writerCalcParam = calc->getParam(mapping.writerParamIndex);
            Param* readerCalcParam = calc->getParam(mapping.readerParamIndex);
            Param* readerShellParam = shell->getParam(mapping.readerParamIndex);
            const std::string& writerCalcName = writerCalcParam->getName();
            const std::string& readerCalcName = readerCalcParam->getName();
            const std::string& readerTensorName = readerShellParam->getName();
            const std::string mpiType =
                mpi_rewriter::mpiDatatypeFor(writerCalcParam->getBasicType());

            if (mpi_rewriter::usesByteTransport(writerCalcParam->getBasicType())) {
                code += "    MPI_Gatherv(ctx.dist_" + writerCalcName +
                        ".local_write_values.data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("ctx.output_layout_" + writerCalcName + ".local_count",
                                                          writerCalcParam->getBasicType()) +
                        ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" +
                        writerCalcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                        writerCalcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                        writerCalcName + ".byte_displs.data() : nullptr, " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Gatherv(ctx.dist_" + writerCalcName +
                        ".local_write_values.data(), ctx.output_layout_" + writerCalcName +
                        ".local_count, " + mpiType +
                        ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" + writerCalcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + writerCalcName +
                        ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + writerCalcName +
                        ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + readerTensorName + ".tensor2Array(ctx.global_" + readerCalcName + ");\n";
            if (mapping.rank == 2) {
                Param* writerShellParam = shell->getParam(mapping.writerParamIndex);
                code += "        const int64_t __dacpp_writer_cols = static_cast<int64_t>(" +
                        writerShellParam->getName() + ".getShape(1));\n";
                code += "        const int64_t __dacpp_reader_cols = static_cast<int64_t>(" +
                        readerTensorName + ".getShape(1));\n";
            }
            code += "        for (std::size_t __dacpp_idx = 0; __dacpp_idx < ctx.output_layout_" +
                    writerCalcName + ".globals.size() && __dacpp_idx < ctx.global_recv_values_" +
                    writerCalcName + ".size(); ++__dacpp_idx) {\n";
            if (mapping.rank == 2) {
                code += "            const int64_t __dacpp_target_global = dacpp::mpi::map_2d_global_with_offset(ctx.output_layout_" +
                        writerCalcName + ".globals[__dacpp_idx], __dacpp_writer_cols, __dacpp_reader_cols, " +
                        std::to_string(mapping.targetRowOffset) + ", " +
                        std::to_string(mapping.targetColOffset) + ");\n";
            } else {
                code += "            const int64_t __dacpp_target_global = ctx.output_layout_" +
                        writerCalcName + ".globals[__dacpp_idx] + static_cast<int64_t>(" +
                        std::to_string(mapping.targetOffset) + ");\n";
            }
            code += "            if (__dacpp_target_global >= 0 && static_cast<std::size_t>(__dacpp_target_global) < ctx.global_" +
                    readerCalcName + ".size()) {\n";
            code += "                ctx.global_" + readerCalcName + "[static_cast<std::size_t>(__dacpp_target_global)] = ctx.global_recv_values_" +
                    writerCalcName + "[__dacpp_idx];\n";
            code += "            }\n";
            code += "        }\n";
            code += "        " + readerTensorName + ".array2Tensor(ctx.global_" + readerCalcName + ");\n";
            code += "    }\n";
        }
    } else {
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (transportModes[paramIdx] == IOTYPE::WRITE) {
                continue;
            }
            Param* calcParam = calc->getParam(paramIdx);
            Param* shellWrapperParam = shell->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            const std::string& tensorName = shellWrapperParam->getName();
            const std::string mpiType =
                mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

            if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                code += "    MPI_Gatherv(ctx.dist_" + calcName + ".local_cache.data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                          calcParam->getBasicType()) +
                        ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Gatherv(ctx.dist_" + calcName + ".local_cache.data(), ctx.input_layout_" +
                        calcName + ".local_count, " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
            code += "        dacpp::mpi::apply_writeback_by_globals(ctx.sendbuf_" + calcName +
                    ", ctx.input_layout_" + calcName + ".globals, ctx.global_" + calcName + ");\n";
            code += "        " + tensorName + ".array2Tensor(ctx.global_" + calcName + ");\n";
            code += "    }\n";
        }
    }
    code += "}\n\n";

    code += "void " + wrapper + "(" + shellSignature + ") {\n";
    code += "    " + ctxType + " ctx;\n";
    code += "    " + initName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "    " + materializeName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "}\n";

    code += "\n";
    code += mpi_rewriter::buildRootCentricPostRegionHelpers(
        dacppFile, shell, calc, dacExpr, ctxType, shellSignature);

    return code;
}

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
