#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

std::vector<bool> collectSiblingWrittenParams(DacppFile* dacppFile,
                                              Shell* shell) {
    std::vector<bool> written(
        static_cast<std::size_t>(shell ? shell->getNumShellParams() : 0), false);
    if (!dacppFile || !shell) {
        return written;
    }
    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || !plan.dacExpr) {
        return written;
    }

    const auto* shellCall = dacppTranslator::getNode<clang::CallExpr>(
        dacppTranslator::Expression::shellLHS_p(plan.dacExpr)
            ? plan.dacExpr->getLHS()
            : plan.dacExpr->getRHS());
    std::unordered_map<const clang::ValueDecl*, int> argDeclIndices;
    if (shellCall) {
        for (int paramIdx = 0;
             paramIdx < std::min<int>(shell->getNumShellParams(),
                                      static_cast<int>(shellCall->getNumArgs()));
             ++paramIdx) {
            const clang::Expr* arg = shellCall->getArg(static_cast<unsigned>(paramIdx));
            if (const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                    arg ? arg->IgnoreParenImpCasts() : nullptr)) {
                argDeclIndices.emplace(declRef->getDecl(), paramIdx);
            }
        }
    }

    for (const clang::Stmt* siblingStmt : plan.siblingStmts) {
        const auto summary = summarizeStmtAccess(
            siblingStmt, argDeclIndices, shell->getNumShellParams());
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (summary[static_cast<std::size_t>(paramIdx)].writes) {
                written[static_cast<std::size_t>(paramIdx)] = true;
            }
        }
    }
    return written;
}

}  // namespace

std::vector<std::string> buildMPIRegionSiblingLookupInitCode(
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated) {
    std::vector<std::string> snippets;
    if (!shell || !calc) {
        return snippets;
    }

    snippets.reserve(static_cast<std::size_t>(shell->getNumShellParams()));
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        std::string code;
        code += "    std::size_t __dacpp_total_dense_" + name + " = 1;\n";
        code += "    for (int __dacpp_dim : ctx.pattern_" + name +
                ".data_info.dimLength) {\n";
        code += "        __dacpp_total_dense_" + name +
                " *= static_cast<std::size_t>(std::max(0, __dacpp_dim));\n";
        code += "    }\n";
        code += "    ctx.global_to_local_" + name + ".assign(\n";
        code += "        __dacpp_total_dense_" + name + ", -1);\n";
        code += "    for (const auto& __dacpp_g2l_entry : ctx.pack_" + name +
                ".g2l) {\n";
        code += "        if (__dacpp_g2l_entry.first >= 0 &&\n";
        code += "            static_cast<std::size_t>(__dacpp_g2l_entry.first) < ctx.global_to_local_" +
                name + ".size()) {\n";
        code += "            ctx.global_to_local_" + name +
                "[static_cast<std::size_t>(__dacpp_g2l_entry.first)] =\n";
        code += "                __dacpp_g2l_entry.second;\n";
        code += "        }\n";
        code += "    }\n";
        code += "    if (!ctx.global_to_local_" + name + ".empty()) {\n";
        code += "        ctx.global_to_local_buf_" + name +
                " = std::make_unique<sycl::buffer<int32_t, 1>>(\n";
        code += "            ctx.global_to_local_" + name +
                ".data(), sycl::range<1>(ctx.global_to_local_" + name +
                ".size()));\n";
        code += "    }\n";
        snippets.push_back(std::move(code));
    }

    (void)generated;
    return snippets;
}

std::string buildMPIRegionInitCode(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated,
    const std::vector<IOTYPE>& storageModes,
    const MPIRegionTransferPolicy& transferPolicy,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
    const std::string& shellParamSig) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "void " + generated.initName + "(" + generated.ctxTypeName +
            "& ctx";
    if (!shellParamSig.empty()) {
        code += ", " + shellParamSig;
    }
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = sycl::queue(sycl::default_selector_v);\n";
    code += "    ctx.binding_split_sizes.clear();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "ctx.pattern_" + name;
        const IOTYPE mode = storageModes[paramIdx];

        code += "    ctx.pattern_" + name + " = dacpp::mpi::AccessPattern();\n";
        code += "    ctx.pattern_" + name + ".param_id = " +
                std::to_string(paramIdx) + ";\n";
        code += "    ctx.pattern_" + name + ".name = \"" + name + "\";\n";
        code += "    ctx.pattern_" + name + ".mode = " + toPlannerMode(mode) +
                ";\n";
        code += "    ctx.pattern_" + name + ".data_info.dim = " + tensorName +
                ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) ctx.pattern_" + name +
                ".data_info.dimLength.push_back(" + tensorName +
                ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName =
                "__dacpp_op_" + name + "_" + std::to_string(splitIdx);

            code += "    { Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" +
                    std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1
                                           : static_cast<RegularSplit*>(split)
                                                 ->getSplitSize()) +
                    ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1
                                           : static_cast<RegularSplit*>(split)
                                                 ->getSplitStride()) +
                    ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        ") - " +
                        std::to_string(
                            static_cast<RegularSplit*>(split)->getSplitSize()) +
                        ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)
                                           ->getSplitStride()) +
                        " + 1);\n";
            }
            code += "    ctx.pattern_" + name + ".param_ops.push_back(" + opName +
                    ");\n";
            code += "    ctx.pattern_" + name + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    ctx.pattern_" + name + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    ctx.pattern_" + name + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + "); }\n";
        }

        code += "    ctx.pattern_" + name +
                ".partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_" +
                name + ");\n";
        code += "    ctx.pattern_" + name +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_" +
                name + ");\n";
        code += "    if (ctx.binding_split_sizes.size() < ctx.pattern_" + name +
                ".bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_" +
                name + ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < ctx.pattern_" + name +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code +=
            "        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_" +
            name + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }

    code += "    ctx.total_items = 1;\n";
    code +=
        "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code +=
        "    auto item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "    ctx.pattern_" + name +
                ".bind_split_sizes = ctx.binding_split_sizes;\n";
    }
    const std::vector<bool> siblingWrittenParams =
        collectSiblingWrittenParams(dacppFile, shell);

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const IOTYPE mode = storageModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "ctx.pattern_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const bool needsSiblingDenseCover =
            siblingWrittenParams[static_cast<std::size_t>(paramIdx)];

        code += "    auto __dacpp_base_pack_" + calcName + " = " +
                buildPackBuilderExpr(mode, patternName) + ";\n";
        if (needsSiblingDenseCover) {
            code += "    {\n";
            code += "        std::size_t __dacpp_dense_count_" + calcName +
                    " = 1;\n";
            code += "        for (int __dacpp_dim : " + patternName +
                    ".data_info.dimLength) {\n";
            code += "            __dacpp_dense_count_" + calcName +
                    " *= static_cast<std::size_t>(std::max(0, __dacpp_dim));\n";
            code += "        }\n";
            code += "        std::vector<int64_t> __dacpp_all_globals_" + calcName +
                    "(__dacpp_dense_count_" + calcName + ");\n";
            code += "        for (std::size_t __dacpp_g = 0; __dacpp_g < __dacpp_dense_count_" +
                    calcName + "; ++__dacpp_g) {\n";
            code += "            __dacpp_all_globals_" + calcName +
                    "[__dacpp_g] = static_cast<int64_t>(__dacpp_g);\n";
            code += "        }\n";
            code += "        ctx.pack_" + calcName +
                    " = dacpp::mpi::make_pack_map_from_globals(std::move(__dacpp_all_globals_" +
                    calcName + "));\n";
            code += "        ctx.pack_" + calcName +
                    ".writeback_globals = ctx.pack_" + calcName + ".globals;\n";
            code += "    }\n";
        } else {
            code += "    ctx.pack_" + calcName + " = std::move(__dacpp_base_pack_" +
                    calcName + ");\n";
        }
        code += "    ctx.slots_" + calcName +
                " = dacpp::mpi::build_item_slots(item_range, " + patternName +
                ", ctx.pack_" + calcName + ");\n";
        code += "    ctx.local_" + calcName +
                ".resize(ctx.pack_" + calcName + ".globals.size());\n";

        if (transferPolicy.needsInitScatter[static_cast<std::size_t>(paramIdx)]) {
            if (needsSiblingDenseCover) {
                const std::string payloadDenseCount = mpiPayloadCountExpr(
                    "static_cast<int>(ctx.local_" + calcName + ".size())",
                    calcParam->getBasicType());
                code += "    {\n";
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        std::vector<" + calcParam->getBasicType() +
                        "> __dacpp_global_" + calcName + ";\n";
                code += "        " + tensorName + ".tensor2Array(__dacpp_global_" +
                        calcName + ");\n";
                code += "        if (__dacpp_global_" + calcName + ".size() == ctx.local_" +
                        calcName + ".size()) {\n";
                code += "            std::copy(__dacpp_global_" + calcName +
                        ".begin(), __dacpp_global_" + calcName +
                        ".end(), ctx.local_" + calcName + ".begin());\n";
                code += "        }\n";
                code += "    }\n";
                code += "    if (!ctx.local_" + calcName + ".empty()) {\n";
                code += "        MPI_Bcast(ctx.local_" + calcName + ".data(), " +
                        payloadDenseCount + ", " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
                code += "    }\n";
                code += "    }\n";
            } else {
            const std::string payloadRecvCount = mpiPayloadCountExpr(
                "recv_count_" + calcName, calcParam->getBasicType());
            code += "    {\n";
            code += "    int recv_count_" + calcName + " = 0;\n";
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    std::vector<" + calcParam->getBasicType() +
                    "> sendbuf_" + calcName + ";\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        sendcounts_" + calcName +
                    ".resize(ctx.mpi_size);\n";
            code += "        displs_" + calcName + ".resize(ctx.mpi_size);\n";
            code += "        int current_displ = 0;\n";
            code += "        std::vector<" + calcParam->getBasicType() +
                    "> global_" + calcName + ";\n";
            code += "        " + tensorName + ".tensor2Array(global_" + calcName +
                    ");\n";
            code += "        for (int r = 0; r < ctx.mpi_size; ++r) {\n";
            code +=
                "            auto r_range = dacpp::mpi::get_rank_item_range(ctx.total_items, r, ctx.mpi_size);\n";
            code += "            auto r_pack = " +
                    buildRemotePackBuilderExpr(mode, "r_range", patternName) +
                    ";\n";
            code += "            auto r_values = dacpp::mpi::pack_values_by_globals(global_" +
                    calcName + ", r_pack.globals);\n";
            code += "            int r_count = static_cast<int>(r_values.size());\n";
            code += "            sendcounts_" + calcName + "[r] = r_count;\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += r_count;\n";
            code += "            sendbuf_" + calcName +
                    ".insert(sendbuf_" + calcName +
                    ".end(), r_values.begin(), r_values.end());\n";
            code += "        }\n";
            code += "    }\n";
            code += "    MPI_Scatter(ctx.mpi_rank == 0 ? sendcounts_" +
                    calcName +
                    ".data() : nullptr, 1, MPI_INT, &recv_count_" + calcName +
                    ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    ctx.local_" + calcName +
                    ".resize(recv_count_" + calcName + ");\n";
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    { std::vector<int> sc_bytes = sendcounts_" +
                        calcName + ";\n";
                code += "      std::vector<int> ds_bytes = displs_" + calcName +
                        ";\n";
                code += "      if (ctx.mpi_rank == 0) { for (int r = 0; r < ctx.mpi_size; ++r) { sc_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); ds_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); } }\n";
                code += "      MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? sc_bytes.data() : nullptr, ctx.mpi_rank == 0 ? ds_bytes.data() : nullptr, " +
                        mpiType + ", ctx.local_" + calcName + ".data(), " +
                        payloadRecvCount + ", " + mpiType +
                        ", 0, MPI_COMM_WORLD); }\n";
            } else {
                code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? sendcounts_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? displs_" +
                        calcName + ".data() : nullptr, " + mpiType +
                        ", ctx.local_" + calcName + ".data(), " +
                        payloadRecvCount + ", " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    }\n";
            }
        }

        code += "    ctx." + calcName +
                "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" +
                calcName + "));\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    ctx." + calcName + "_cols = ctx.pattern_" + calcName +
                    ".partition_shape[1];\n";
        }

        code += "    if (ctx.local_item_count > 0) {\n";
        code += "        ctx.buf_" + calcName +
                " = std::make_unique<sycl::buffer<" +
                calcParam->getBasicType() + ", 1>>(ctx.local_" + calcName +
                ".data(), sycl::range<1>(ctx.local_" + calcName + ".size()));\n";
        code += "        ctx.slots_buf_" + calcName +
                " = std::make_unique<sycl::buffer<int32_t, 1>>(ctx.slots_" +
                calcName + ".data(), sycl::range<1>(ctx.slots_" + calcName +
                ".size()));\n";
        code += "    }\n";
    }

    for (const auto& snippet :
         buildMPIRegionSiblingLookupInitCode(shell, calc, generated)) {
        code += snippet;
    }

    code += "    ctx.has_halo = (ctx.mpi_size > 1 && ctx.local_item_count > 0);\n";
    code += "    if (ctx.has_halo) {\n";
    code +=
        "        auto my_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "        ctx.halo_" + name +
                " = dacpp::mpi::computeParamHalo(\n";
        code += "            ctx.pattern_" + name + ", ctx.pattern_" + name +
                ".mode,\n";
        code += "            my_range, ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    }
    code += "    }\n";

    // Broadcast non-shell captured variables from rank 0
    if (dacppFile) {
        const auto& plan = dacppFile->getBufferRegionPlan();
        for (const auto& var : plan.capturedNonShellVars) {
            const std::string& varName = var.first;
            const std::string& varType = var.second;
            const std::string mpiType = mpiDatatypeFor(varType);
            // Copy the host-side value into ctx on rank 0, then broadcast
            code += "    {\n";
            code += "        if (ctx.mpi_rank == 0) ctx." + varName +
                    " = " + varName + ";\n";
            code += "        MPI_Bcast(&ctx." + varName + ", 1, " + mpiType +
                    ", 0, MPI_COMM_WORLD);\n";
            code += "    }\n";
        }
    }

    code += "}\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
