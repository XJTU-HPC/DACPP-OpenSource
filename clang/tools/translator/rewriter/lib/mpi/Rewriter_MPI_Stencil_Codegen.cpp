#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rewriter_MPI_Stencil_Common.h"
#include "llvm/Support/raw_ostream.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

namespace {

std::string buildParamNameList(Shell* shell) {
    std::string args;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += shell->getParam(paramIdx)->getName();
    }
    return args;
}

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return shellParamName;
    }
    const clang::CallExpr* shellCall =
        dacppTranslator::getNode<clang::CallExpr>(
            dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                             : dacExpr->getRHS());
    if (!shellCall) {
        return shellParamName;
    }
    const clang::FunctionDecl* callee = shellCall->getDirectCallee();
    if (!callee) {
        return shellParamName;
    }
    for (unsigned paramIdx = 0;
         paramIdx < callee->getNumParams() && paramIdx < shellCall->getNumArgs();
         ++paramIdx) {
        const clang::ParmVarDecl* param = callee->getParamDecl(paramIdx);
        if (!param || param->getNameAsString() != shellParamName) {
            continue;
        }
        const auto* dre = dacppTranslator::getNode<clang::DeclRefExpr>(
            const_cast<clang::Expr*>(shellCall->getArg(paramIdx)));
        if (dre && dre->getDecl()) {
            return dre->getDecl()->getNameAsString();
        }
    }
    return shellParamName;
}

void appendPatternInitForContext(std::string& code,
                                 Shell* shell,
                                 Calc* calc,
                                 const std::unordered_map<std::string, mpi_rewriter::SplitBindMeta>& splitMeta,
                                 const std::vector<IOTYPE>& paramModes,
                                 const std::string& ctxVar) {
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = ctxVar + ".pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    " + patternName + " = dacpp::mpi::AccessPattern{};\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + mpi_rewriter::toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName + ".getDim(); ++dim) " +
                patternName + ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            mpi_rewriter::SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName = "pattern_" + name + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" + std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitSize()) + ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitStride()) + ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + ") - " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitSize()) + ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitStride()) + " + 1);\n";
            }
            code += "    " + patternName + ".param_ops.push_back(" + opName + ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + ");\n";
        }

        code += "    " + patternName + ".partition_shape = dacpp::mpi::init_partition_shape(" +
                patternName + ");\n";
        code += "    " + patternName + ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" +
                patternName + ");\n";
        code += "    if (" + ctxVar + ".binding_split_sizes.size() < " + patternName +
                ".bind_split_sizes.size()) " + ctxVar + ".binding_split_sizes.resize(" + patternName +
                ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < " + patternName + ".bind_split_sizes.size(); ++bind_i) {\n";
        code += "        " + ctxVar + ".binding_split_sizes[bind_i] = std::max<int64_t>(" +
                ctxVar + ".binding_split_sizes[bind_i], " + patternName + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }
}

}  // namespace

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
    const std::string shellSignature = buildShellSignature(shell);
    const std::string shellArgNames = buildParamNameList(shell);
    const auto paramModes = mpi_rewriter::inferEffectiveParamModes(shell, calc);
    const auto splitMeta = mpi_rewriter::collectSplitBindMeta(shell);
    const auto distributedSitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);

    if (distributedSitePlan.supported) {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange enabled";
        if (distributedSitePlan.hasRootBridge) {
            llvm::outs() << " (root-bridge)";
        }
        llvm::outs() << "\n";
    } else {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange disabled: "
                     << distributedSitePlan.disableReason << "\n";
    }

    std::string code;
    code += "struct " + ctxType + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    bool use_partial_exchange = false;\n";
    code += "    std::string partial_exchange_disable_reason;\n";
    code += "    std::unique_ptr<sycl::queue> q;\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += "    int64_t total_items = 1;\n";
    code += "    dacpp::mpi::ItemRange item_range{};\n";
    code += "    int64_t local_item_count = 0;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string elemType = calc->getParam(paramIdx)->getBasicType();
        code += "    dacpp::mpi::AccessPattern pattern_" + calcName + ";\n";
        code += "    dacpp::mpi::PackPlan plan_" + calcName + ";\n";
        code += "    std::vector<" + elemType + "> local_" + calcName + ";\n";
        code += "    dacpp::mpi::DistributedTensorState<" + elemType + "> dist_" +
                calcName + ";\n";
        if (paramModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::GatheredIndexLayout input_layout_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> sendbuf_" + calcName + ";\n";
        }
        if (paramModes[paramIdx] != IOTYPE::READ) {
            code += "    dacpp::mpi::GatheredIndexLayout output_layout_" + calcName + ";\n";
            code += "    std::vector<int32_t> writeback_slots_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> writeback_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_recv_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_out_" + calcName + ";\n";
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
    appendPatternInitForContext(code, shell, calc, splitMeta, paramModes, "ctx");
    code += "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code += "    ctx.item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = ctx.item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string actualTensorName =
            resolveActualTensorName(shell->getParam(paramIdx)->getName(), dacExpr);
        code += "    ctx.pattern_" + calcName + ".bind_split_sizes = ctx.binding_split_sizes;\n";
        code += "    ctx.plan_" + calcName + " = " +
                mpi_rewriter::buildPackPlanBuilderExpr(paramModes[paramIdx],
                                                       "ctx.item_range",
                                                       "ctx.pattern_" + calcName) + ";\n";
        code += "    ctx.local_" + calcName + ".resize(ctx.plan_" + calcName + ".pack.globals.size());\n";
        if (paramModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_" + calcName +
                    ", ctx.plan_" + calcName + ".pack.globals, ctx.mpi_rank, ctx.mpi_size);\n";
            if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                code += "    if (ctx.mpi_rank == 0) dacpp::mpi::init_layout_byte_counts(ctx.input_layout_" +
                        calcName + ", sizeof(" + calc->getParam(paramIdx)->getBasicType() + "));\n";
            }
        }
        if (paramModes[paramIdx] != IOTYPE::READ) {
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
    if (distributedSitePlan.supported) {
        code += "    ctx.use_partial_exchange = true;\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* shellWrapperParam = shell->getParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const IOTYPE mode = paramModes[paramIdx];
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
                if (distributedSitePlan.rootBridgeTensors.count(tensorName) != 0) {
                    code += "    ctx.dist_" + calcName + ".root_bridge_pack = dacpp::mpi::make_dense_cover_pack(static_cast<std::size_t>(" +
                            tensorName + ".getSize()));\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout = dacpp::mpi::AllRankIndexLayout{};\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.local_count = ctx.mpi_rank == 0 ? static_cast<int>(ctx.dist_" + calcName + ".root_bridge_pack.globals.size()) : 0;\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.counts.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.displs.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.globals = ctx.dist_" + calcName + ".root_bridge_pack.globals;\n";
                    code += "    if (ctx.mpi_size > 0) {\n";
                    code += "        ctx.dist_" + calcName + ".root_bridge_layout.counts[0] = ctx.dist_" + calcName + ".root_bridge_layout.local_count;\n";
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
                for (int readerIdx = 0; readerIdx < shell->getNumShellParams(); ++readerIdx) {
                    if (paramModes[readerIdx] == IOTYPE::WRITE) {
                        continue;
                    }
                    const std::string& readerCalcName = calc->getParam(readerIdx)->getName();
                    code += "    dacpp::mpi::build_target_slots_for_globals(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, ctx.dist_" + calcName + ".local_target_slots);\n";
                    code += "    ctx.dist_" + calcName + ".exchange_plan = dacpp::mpi::build_exchange_plan_from_layouts(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                    code += "    if (!ctx.dist_" + calcName + ".exchange_plan.supported) {\n";
                        code += "        ctx.partial_exchange_disable_reason = ctx.dist_" + calcName + ".exchange_plan.unsupported_reason;\n";
                    code += "        ctx.use_partial_exchange = false;\n";
                    code += "    }\n";
                    break;
                }
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
    code += "    if (!ctx.use_partial_exchange) {\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const IOTYPE mode = paramModes[paramIdx];
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
        } else {
            code += "    " + localName + ".assign(" + packName + ".globals.size(), " + calcParam->getBasicType() + "{});\n";
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" + calcName + "));\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = ctx.pattern_" + calcName + ".partition_shape[1];\n";
        }
    }

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
                mpi_rewriter::toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
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
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear]};\n";
        } else {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
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

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (paramModes[paramIdx] == IOTYPE::READ) {
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
    }
    if (distributedSitePlan.supported) {
        code += "    } else {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            code += "    auto& local_" + calcName + " = ctx.dist_" + calcName +
                    ".local_cache;\n";
            if (paramModes[paramIdx] == IOTYPE::WRITE) {
                code += "    local_" + calcName + ".assign(ctx.plan_" + calcName +
                        ".pack.globals.size(), " + calcParam->getBasicType() + "{});\n";
            }
        }
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
                    mpi_rewriter::toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
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
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                        " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                        "[item_linear]};\n";
            } else {
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
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
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (paramModes[paramIdx] == IOTYPE::READ) {
                continue;
            }
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            code += "    dacpp::mpi::publish_local_writes_with_exchange(local_" + calcName +
                    ", ctx.dist_" + calcName + ".local_write_slots, ctx.dist_" + calcName +
                    ".local_target_slots, ctx.dist_" + calcName +
                    ".local_cache, ctx.dist_" + calcName + ".local_write_values, ctx.dist_" +
                    calcName + ".exchange_plan);\n";
        }
        if (distributedSitePlan.hasRootBridge) {
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
                if (paramModes[paramIdx] == IOTYPE::READ) {
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
    code += "        if (mpi_rank == 0) {\n";
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\\n\", \"" + wrapper + "\", dacpp_wrapper_max_ms);\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp::mpi::reportCollectPositionsProfile(\"" + wrapper + "\", MPI_COMM_WORLD);\n";
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
    code += "}\n";

    code += "\n";
    code += mpi_rewriter::buildRootCentricPostRegionHelpers(
        dacppFile, shell, calc, dacExpr, ctxType, shellSignature);

    return code;
}

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
