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
        static_cast<std::size_t>(shell ? shell->getNumShellParams() : 0),
        false);
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
            const clang::Expr* arg =
                shellCall->getArg(static_cast<unsigned>(paramIdx));
            if (const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                    arg ? arg->IgnoreParenImpCasts() : nullptr)) {
                argDeclIndices.emplace(declRef->getDecl(), paramIdx);
            }
        }
    }

    for (const clang::Stmt* siblingStmt : plan.siblingStmts) {
        const auto summary = summarizeStmtAccess(
            siblingStmt, argDeclIndices, shell->getNumShellParams());
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams();
             ++paramIdx) {
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
        code += "    dacpp::mpi::init_region_lookup(ctx.state_" + name +
                ", \"buildMPIRegionSiblingLookupInitCode\");\n";
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

    code += "void " + generated.initName + "(" + generated.ctxTypeName + "& ctx";
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
        const std::string stateName = "ctx.state_" + name;
        const std::string patternName = stateName + ".pattern";
        const IOTYPE mode = storageModes[paramIdx];

        code += "    " + patternName + " = dacpp::mpi::AccessPattern();\n";
        code += "    " + patternName + ".param_id = " +
                std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName +
                ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) " + patternName +
                ".data_info.dimLength.push_back(" + tensorName +
                ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit();
             ++splitIdx) {
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
            code += "    " + patternName + ".param_ops.push_back(" + opName +
                    ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + "); }\n";
        }

        code += "    " + patternName +
                ".partition_shape = dacpp::mpi::init_partition_shape(" +
                patternName + ");\n";
        code += "    " + patternName +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" +
                patternName + ");\n";
        code += "    if (ctx.binding_split_sizes.size() < " + patternName +
                ".bind_split_sizes.size()) ctx.binding_split_sizes.resize(" +
                patternName + ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < " + patternName +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code +=
            "        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], " +
            patternName + ".bind_split_sizes[bind_i]);\n";
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
        code += "    ctx.state_" + name +
                ".pattern.bind_split_sizes = ctx.binding_split_sizes;\n";
    }

    const std::vector<bool> siblingWrittenParams =
        collectSiblingWrittenParams(dacppFile, shell);

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string stateName = "ctx.state_" + calcName;
        const std::string patternName = stateName + ".pattern";
        const bool needsSiblingDenseCover =
            siblingWrittenParams[static_cast<std::size_t>(paramIdx)];

        code += "    " + stateName + ".runtime_pack = dacpp::mpi::PackMap();\n";
        if (needsSiblingDenseCover) {
            code += "    {\n";
            code += "        std::size_t __dacpp_dense_count_" + calcName +
                    " = 1;\n";
            code += "        for (int __dacpp_dim : " + patternName +
                    ".data_info.dimLength) {\n";
            code += "            __dacpp_dense_count_" + calcName +
                    " *= static_cast<std::size_t>(std::max(0, __dacpp_dim));\n";
            code += "        }\n";
            code += "        " + stateName +
                    ".runtime_pack = dacpp::mpi::make_dense_cover_pack(__dacpp_dense_count_" +
                    calcName + ");\n";
            code += "    }\n";
        }
        code += "    dacpp::mpi::init_region_param_storage(" + tensorName +
                ", " + stateName +
                ", item_range, ctx.mpi_rank, ctx.mpi_size, " +
                std::string(
                    transferPolicy.needsInitScatter[static_cast<std::size_t>(
                        paramIdx)]
                        ? "true"
                        : "false") +
                ");\n";
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
        code += "        ctx.state_" + name +
                ".halo = dacpp::mpi::computeParamHalo(\n";
        code += "            ctx.state_" + name + ".pattern, ctx.state_" + name +
                ".pattern.mode,\n";
        code +=
            "            my_range, ctx.total_items, ctx.mpi_rank, ctx.mpi_size,\n";
        code += "            ctx.state_" + name + ".runtime_pack);\n";
    }
    code += "    }\n";

    if (dacppFile) {
        const auto& plan = dacppFile->getBufferRegionPlan();
        for (const auto& var : plan.capturedNonShellVars) {
            const std::string& varName = var.first;
            const std::string& varType = var.second;
            const std::string mpiType = mpiDatatypeFor(varType);
            code += "    {\n";
            code += "        if (ctx.mpi_rank == 0) ctx." + varName + " = " +
                    varName + ";\n";
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
