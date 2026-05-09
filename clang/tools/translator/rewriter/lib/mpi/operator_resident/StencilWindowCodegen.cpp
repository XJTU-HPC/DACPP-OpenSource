#include <string>
#include <vector>

#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

const ParamAccessPlan* findStencilReader(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::StencilWindow) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* findStencilWriter(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes) {
            return &param;
        }
    }
    return nullptr;
}

std::vector<const ParamAccessPlan*> findStencilDirectReaders(
    const ShellPartitionPlan& plan) {
    std::vector<const ParamAccessPlan*> readers;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            readers.push_back(&param);
        }
    }
    return readers;
}

bool isScalarOnlyStencilReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::ReplicatedScalar &&
           param.reads &&
           !param.writes;
}

bool supportsStencilWindow2DParams(const ShellPartitionPlan& plan,
                                   bool allowDirectReaders) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::StencilWindow) {
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            continue;
        }
        if (allowDirectReaders &&
            param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            continue;
        }
        if (isScalarOnlyStencilReader(param)) {
            return false;
        }
        return false;
    }
    return true;
}

std::string checkedMulExpr(const std::string& lhs,
                           const std::string& rhs,
                           const std::string& what) {
    return "dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(" +
           lhs + "), static_cast<int64_t>(" + rhs + "), \"" + what + "\")";
}

std::string checkedMpiCountExpr(const std::string& expr,
                                const std::string& what) {
    return "dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(" +
           expr + "), \"" + what + "\")";
}

std::string checkedMpiProductCountExpr(const std::string& lhs,
                                       const std::string& rhs,
                                       const std::string& what) {
    return checkedMpiCountExpr(checkedMulExpr(lhs, rhs, what), what);
}

std::string checkedMpiPayloadCountExpr(const std::string& elemCountExpr,
                                       const std::string& type,
                                       const std::string& what) {
    if (usesByteTransport(type)) {
        return checkedMpiCountExpr(
            checkedMulExpr(elemCountExpr, "sizeof(" + type + ")", what),
            what);
    }
    return checkedMpiCountExpr(elemCountExpr, what);
}

std::string checkedDense2DPayloadCountExpr(const std::string& rowsExpr,
                                           const std::string& colsExpr,
                                           const std::string& type,
                                           const std::string& what) {
    const std::string elemCount =
        checkedMulExpr(rowsExpr, colsExpr, what);
    if (usesByteTransport(type)) {
        return checkedMpiCountExpr(
            checkedMulExpr(elemCount, "sizeof(" + type + ")", what), what);
    }
    return checkedMpiCountExpr(elemCount, what);
}

const ParamAccessPlan* findParamByIndex(const ShellPartitionPlan& plan,
                                        int paramIndex) {
    for (const auto& param : plan.params) {
        if (param.paramIndex == paramIndex) {
            return &param;
        }
    }
    return nullptr;
}

DistributedStencilSitePlan stencilSitePlanFor(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    if (!dacppFile || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.exprNode.dacExpr) {
        return {};
    }
    return analyzeDistributedStencilSite(dacppFile, plan.exprNode.shell,
                                        plan.exprNode.calc,
                                        plan.exprNode.dacExpr);
}

std::vector<DistributedFollowupMapping> followupsForWriter(
    const DistributedStencilSitePlan& sitePlan,
    const ParamAccessPlan& writer) {
    std::vector<DistributedFollowupMapping> result;
    if (!sitePlan.supported || sitePlan.hasRootBridge) {
        return result;
    }
    for (const auto& mapping : sitePlan.followupMappings) {
        if (mapping.writerParamIndex == writer.paramIndex ||
            mapping.writerTensor == writer.actualTensorName ||
            mapping.writerTensor == writer.shellParamName) {
            result.push_back(mapping);
        }
    }
    return result;
}

void emitReadCacheTransitions2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<ReadCacheStateTransition>& transitions) {
    for (const auto& transition : transitions) {
        if (transition.rank != 2) {
            continue;
        }
        const ParamAccessPlan* source =
            findParamByIndex(plan, transition.writerParamIndex);
        const ParamAccessPlan* target =
            findParamByIndex(plan, transition.readerParamIndex);
        if (!source || !target) {
            continue;
        }
        const std::string targetType = elemType(plan, *target);
        const std::string targetMpiType = mpiDatatypeFor(targetType);
        const std::string transitionGlobal =
            "__or_transition_global_" + target->calcParamName;
        code += "    {\n";
        code += "        std::vector<" + targetType + "> " +
                transitionGlobal + ";\n";
        code += "        if (mpi_rank == 0) {\n";
        code += "            " + paramVarName(*target) +
                ".tensor2Array(" + transitionGlobal + ");\n";
        code += "            const int64_t __or_transition_source_cols_" +
                source->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*source) + ".getShape(1));\n";
        code += "            const int64_t __or_transition_target_cols_" +
                target->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*target) + ".getShape(1));\n";
        code += "            for (std::size_t __or_idx = 0; __or_idx < __or_global_" +
                source->calcParamName + ".size(); ++__or_idx) {\n";
        code += "                const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), __or_transition_source_cols_" +
                source->calcParamName + ", __or_transition_target_cols_" +
                target->calcParamName + ", " +
                std::to_string(transition.targetRowOffset) + ", " +
                std::to_string(transition.targetColOffset) + ");\n";
        code += "                if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < " +
                transitionGlobal + ".size()) {\n";
        code += "                    " + transitionGlobal +
                "[static_cast<std::size_t>(__or_target)] = static_cast<" +
                targetType + ">(__or_global_" + source->calcParamName +
                "[__or_idx]);\n";
        code += "                }\n";
        code += "            }\n";
        code += "        } else {\n";
        code += "            " + checkedMpiPayloadCountExpr(
                paramVarName(*target) + ".getSize()", targetType,
                "[DACPP][MPI][OR] StencilWindow2D read-cache broadcast count exceeds MPI int range") +
                ";\n";
        code += "            " + transitionGlobal +
                ".resize(static_cast<std::size_t>(" + paramVarName(*target) +
                ".getSize()));\n";
        code += "        }\n";
        code += "        if (!" + transitionGlobal + ".empty()) {\n";
        code += "            MPI_Bcast(" + transitionGlobal + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(*target) + ".getSize()", targetType,
                    "[DACPP][MPI][OR] StencilWindow2D read-cache broadcast count") +
                ", " + targetMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "        " + paramVarName(*target) + ".array2Tensor(" +
                transitionGlobal + ");\n";
        code += "    }\n";
    }
}

void emitBoundaryLocalMaterialize2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<BoundaryLocalUpdate>& updates,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::string& materializedWriterName,
    const std::string& readerGlobalName,
    const std::string& readerRowsExpr,
    const std::string& readerColsExpr,
    const std::string& outputRowsExpr,
    const std::string& outputColsExpr) {
    for (const auto& update : updates) {
        if (update.rank != 2 || update.paramIndex != reader.paramIndex) {
            continue;
        }
        const std::string loopLower =
            update.loopLowerExpr.empty() ? "0" : update.loopLowerExpr;
        const std::string loopUpper =
            update.loopUpperExpr.empty()
                ? ((update.targetRowUsesLoop || update.sourceRowUsesLoop)
                       ? readerRowsExpr
                       : readerColsExpr)
                : update.loopUpperExpr;
        const std::string loopCmp = update.loopUpperInclusive ? "<=" : "<";
        code += "            {\n";
        code += "                const int64_t __or_boundary_begin = static_cast<int64_t>(" +
                loopLower + ");\n";
        code += "                const int64_t __or_boundary_end = static_cast<int64_t>(" +
                loopUpper + ");\n";
        code += "                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx " +
                loopCmp +
                " __or_boundary_end; ++__or_boundary_idx) {\n";
        code += "                    const int64_t __or_boundary_target_row = " +
                (update.targetRowUsesLoop ? "__or_boundary_idx"
                                          : update.targetRowExpr) +
                ";\n";
        code += "                    const int64_t __or_boundary_target_col = " +
                (update.targetColUsesLoop ? "__or_boundary_idx"
                                          : update.targetColExpr) +
                ";\n";
        code += "                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= " +
                readerRowsExpr + " || __or_boundary_target_col >= " +
                readerColsExpr + ") continue;\n";
        code += "                    const int64_t __or_boundary_target = __or_boundary_target_row * " +
                readerColsExpr + " + __or_boundary_target_col;\n";
        code += "                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= " +
                readerGlobalName + ".size()) continue;\n";
        if (update.constantRhs) {
            code += "                    " + readerGlobalName +
                    "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                    elemType(plan, reader) + ">(" +
                    (update.constantValue.empty() ? "0"
                                                  : update.constantValue) +
                    ");\n";
        } else {
            code += "                    const int64_t __or_boundary_source_row = " +
                    (update.sourceRowUsesLoop ? "__or_boundary_idx"
                                              : update.sourceRowExpr) +
                    ";\n";
            code += "                    const int64_t __or_boundary_source_col = " +
                    (update.sourceColUsesLoop ? "__or_boundary_idx"
                                              : update.sourceColExpr) +
                    ";\n";
            if (update.sourceParamIndex == writer.paramIndex) {
                code += "                    if (__or_boundary_source_row < 0 || __or_boundary_source_col < 0 || __or_boundary_source_row >= " +
                        outputRowsExpr + " || __or_boundary_source_col >= " +
                        outputColsExpr + ") continue;\n";
                code += "                    const int64_t __or_boundary_source = __or_boundary_source_row * " +
                        outputColsExpr + " + __or_boundary_source_col;\n";
                code += "                    if (__or_boundary_source >= 0 && static_cast<std::size_t>(__or_boundary_source) < " +
                        materializedWriterName + ".size()) {\n";
                code += "                        " + readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                        elemType(plan, reader) + ">(" +
                        materializedWriterName +
                        "[static_cast<std::size_t>(__or_boundary_source)]);\n";
                code += "                    }\n";
            } else {
                code += "                    if (__or_boundary_source_row < 0 || __or_boundary_source_col < 0 || __or_boundary_source_row >= " +
                        readerRowsExpr + " || __or_boundary_source_col >= " +
                        readerColsExpr + ") continue;\n";
                code += "                    const int64_t __or_boundary_source = __or_boundary_source_row * " +
                        readerColsExpr + " + __or_boundary_source_col;\n";
                code += "                    if (__or_boundary_source >= 0 && static_cast<std::size_t>(__or_boundary_source) < " +
                        readerGlobalName + ".size()) {\n";
                code += "                        " + readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_target)] = " +
                        readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_source)];\n";
                code += "                    }\n";
            }
        }
        code += "                }\n";
        code += "            }\n";
    }
}

void emitFollowupMaterialize2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& writer,
    const std::vector<DistributedFollowupMapping>& followups,
    const std::vector<BoundaryLocalUpdate>& boundaryUpdates) {
    if (followups.empty()) {
        return;
    }
    const std::string materialized =
        "__or_materialized_" + writer.calcParamName;
    for (const auto& mapping : followups) {
        if (mapping.rank != 2) {
            continue;
        }
        const ParamAccessPlan* reader =
            findParamByIndex(plan, mapping.readerParamIndex);
        if (!reader) {
            continue;
        }
        const std::string readerType = elemType(plan, *reader);
        const std::string readerMpiType = mpiDatatypeFor(readerType);
        const std::string followupGlobal =
            "__or_followup_global_" + reader->calcParamName;
        code += "    {\n";
        code += "        std::vector<" + readerType + "> " + followupGlobal +
                ";\n";
        code += "        if (mpi_rank == 0) {\n";
        code += "            " + paramVarName(*reader) +
                ".tensor2Array(" + followupGlobal + ");\n";
        code += "            const int64_t __or_followup_reader_cols_" +
                reader->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*reader) + ".getShape(1));\n";
        code += "            for (std::size_t __or_idx = 0; __or_idx < " +
                materialized + ".size(); ++__or_idx) {\n";
        code += "                const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), __or_output_cols, __or_followup_reader_cols_" +
                reader->calcParamName + ", " +
                std::to_string(mapping.targetRowOffset) + ", " +
                std::to_string(mapping.targetColOffset) + ");\n";
        code += "                if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < " +
                followupGlobal + ".size()) {\n";
        code += "                    " + followupGlobal +
                "[static_cast<std::size_t>(__or_target)] = static_cast<" +
                readerType + ">(" + materialized + "[__or_idx]);\n";
        code += "                }\n";
        code += "            }\n";
        emitBoundaryLocalMaterialize2D(
            code, plan, boundaryUpdates, *reader, writer, materialized,
            followupGlobal, paramVarName(*reader) + ".getShape(0)",
            "__or_followup_reader_cols_" + reader->calcParamName,
            "__or_output_rows", "__or_output_cols");
        code += "        } else {\n";
        code += "            " + checkedMpiPayloadCountExpr(
                paramVarName(*reader) + ".getSize()", readerType,
                "[DACPP][MPI][OR] StencilWindow2D followup broadcast count exceeds MPI int range") +
                ";\n";
        code += "            " + followupGlobal +
                ".resize(static_cast<std::size_t>(" + paramVarName(*reader) +
                ".getSize()));\n";
        code += "        }\n";
        code += "        if (!" + followupGlobal + ".empty()) {\n";
        code += "            MPI_Bcast(" + followupGlobal + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(*reader) + ".getSize()", readerType,
                    "[DACPP][MPI][OR] StencilWindow2D followup broadcast count") +
                ", " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "        " + paramVarName(*reader) + ".array2Tensor(" +
                followupGlobal + ");\n";
        code += "    }\n";
    }
}

void emitLoopLoweredContextType2D(
    std::string& code,
    const std::string& ctxName,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::vector<const ParamAccessPlan*>& directReaders) {
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t __or_input_rows = 0;\n";
    code += "    int64_t __or_input_cols = 0;\n";
    code += "    int64_t __or_output_rows = 0;\n";
    code += "    int64_t __or_output_cols = 0;\n";
    code += "    int64_t __or_local_output_rows = 0;\n";
    code += "    int64_t __or_output_row_begin = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_output_row_range{};\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    sycl::queue q{sycl::default_selector_v};\n";
    code += "    std::vector<" + elemType(plan, reader) + "> " +
            globalName(reader) + ";\n";
    for (const auto* directReader : directReaders) {
        code += "    std::vector<" + elemType(plan, *directReader) + "> " +
                globalName(*directReader) + ";\n";
    }
    code += "    std::vector<" + elemType(plan, writer) + "> " +
            localName(writer) + ";\n";
    code += "};\n";
}

void emitLoopLoweredInitFunction2D(
    std::string& code,
    const std::string& ctxName,
    const std::string& initName,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::vector<const ParamAccessPlan*>& directReaders) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.__or_input_rows = " + paramVarName(reader) +
            ".getShape(0);\n";
    code += "    ctx.__or_input_cols = " + paramVarName(reader) +
            ".getShape(1);\n";
    code += "    ctx.__or_output_rows = " + paramVarName(writer) +
            ".getShape(0);\n";
    code += "    ctx.__or_output_cols = " + paramVarName(writer) +
            ".getShape(1);\n";
    code += "    ctx.__or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_rows, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.__or_local_output_rows = ctx.__or_output_row_range.count;\n";
    code += "    ctx.__or_output_row_begin = ctx.__or_output_row_range.begin;\n";
    code += "    ctx.__or_local_item_count = " +
            checkedMulExpr(
                "ctx.__or_local_output_rows", "ctx.__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D local item count overflow") +
            ";\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_output_rows, ctx.mpi_size, ctx.__or_row_counts, ctx.__or_row_displs);\n";
    code += "    ctx.__or_counts.resize(static_cast<std::size_t>(ctx.mpi_size));\n";
    code += "    ctx.__or_displs.resize(static_cast<std::size_t>(ctx.mpi_size));\n";
    code += "    for (int r = 0; r < ctx.mpi_size; ++r) {\n";
    code += "        ctx.__or_counts[static_cast<std::size_t>(r)] = " +
            checkedMpiProductCountExpr(
                "ctx.__or_row_counts[static_cast<std::size_t>(r)]",
                "ctx.__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D row-block gather count exceeds MPI int range") +
            ";\n";
    code += "        ctx.__or_displs[static_cast<std::size_t>(r)] = " +
            checkedMpiProductCountExpr(
                "ctx.__or_row_displs[static_cast<std::size_t>(r)]",
                "ctx.__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D row-block gather displacement exceeds MPI int range") +
            ";\n";
    code += "    }\n";
    code += "    const int64_t __or_reader_dense_count = " +
            checkedMulExpr(
                "ctx.__or_input_rows", "ctx.__or_input_cols",
                "[DACPP][MPI][OR] StencilWindow2D reader buffer size overflow") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "__or_reader_dense_count",
            "[DACPP][MPI][OR] StencilWindow2D reader buffer size exceeds MPI int range") +
            ";\n";
    code += "    ctx." + globalName(reader) +
            ".resize(static_cast<std::size_t>(__or_reader_dense_count));\n";
    for (const auto* directReader : directReaders) {
        code += "    const int64_t __or_direct_rows_" +
                directReader->calcParamName + " = " +
                paramVarName(*directReader) + ".getShape(0);\n";
        code += "    const int64_t __or_direct_cols_" +
                directReader->calcParamName + " = " +
                paramVarName(*directReader) + ".getShape(1);\n";
        code += "    if (__or_direct_rows_" + directReader->calcParamName +
                " != ctx.__or_output_rows || __or_direct_cols_" +
                directReader->calcParamName + " != ctx.__or_output_cols) {\n";
        code += "        if (ctx.mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] stencil direct reader " +
                directReader->shellParamName +
                " shape mismatch with output\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 4);\n";
        code += "    }\n";
        code += "    const int64_t __or_direct_dense_count_" +
                directReader->calcParamName + " = " +
                checkedMulExpr(
                    "ctx.__or_output_rows", "ctx.__or_output_cols",
                    "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size overflow") +
                ";\n";
        code += "    " + checkedMpiCountExpr(
                "__or_direct_dense_count_" + directReader->calcParamName,
                "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size exceeds MPI int range") +
                ";\n";
        code += "    ctx." + globalName(*directReader) +
                ".resize(static_cast<std::size_t>(__or_direct_dense_count_" +
                directReader->calcParamName + "));\n";
    }
    code += "    " + checkedMpiCountExpr(
            "ctx.__or_local_item_count",
            "[DACPP][MPI][OR] StencilWindow2D local writer size exceeds MPI int range") +
            ";\n";
    code += "    ctx." + localName(writer) +
            ".assign(static_cast<std::size_t>(ctx.__or_local_item_count), " +
            elemType(plan, writer) + "{});\n";
    if (plan.orLoopLower.hoistReaderSync) {
        code += "    " + checkedMpiCountExpr(
                "__or_reader_dense_count",
                "[DACPP][MPI][OR] StencilWindow2D reader broadcast count exceeds MPI int range") +
                ";\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) + ".tensor2Array(ctx." +
                globalName(reader) + ");\n";
        code += "    }\n";
        if (usesByte(plan, reader)) {
            code += "    MPI_Bcast(ctx." + globalName(reader) +
                    ".data(), " + checkedDense2DPayloadCountExpr(
                        "ctx.__or_input_rows", "ctx.__or_input_cols",
                        readerType,
                        "[DACPP][MPI][OR] StencilWindow2D reader broadcast count exceeds MPI int range") +
                    ", MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(ctx." + globalName(reader) +
                    ".data(), " + checkedDense2DPayloadCountExpr(
                        "ctx.__or_input_rows", "ctx.__or_input_cols",
                        readerType,
                        "[DACPP][MPI][OR] StencilWindow2D reader broadcast count exceeds MPI int range") +
                    ", " +
                    readerMpiType + ", 0, MPI_COMM_WORLD);\n";
        }
    }
    code += "}\n";
}

void emitLoopLoweredRunFunction2D(
    std::string& code,
    const std::string& ctxName,
    const std::string& runName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::vector<const ParamAccessPlan*>& directReaders) {
    const std::string readerType = elemType(plan, reader);
    const std::string writerType = elemType(plan, writer);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string calcName = plan.exprNode.calc->getName();
    const DistributedStencilSitePlan sitePlan =
        stencilSitePlanFor(dacppFile, plan);
    const auto followups = followupsForWriter(sitePlan, writer);

    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto& q = ctx.q;\n";
    code += "    auto& __or_counts = ctx.__or_counts;\n";
    code += "    auto& __or_displs = ctx.__or_displs;\n";
    code += "    const int64_t __or_input_rows = ctx.__or_input_rows;\n";
    code += "    const int64_t __or_input_cols = ctx.__or_input_cols;\n";
    code += "    const int64_t __or_output_rows = ctx.__or_output_rows;\n";
    code += "    const int64_t __or_output_cols = ctx.__or_output_cols;\n";
    code += "    const int64_t __or_local_output_rows = ctx.__or_local_output_rows;\n";
    code += "    const int64_t __or_output_row_begin = ctx.__or_output_row_begin;\n";
    code += "    const int64_t __or_local_item_count = ctx.__or_local_item_count;\n";
    code += "    auto& " + globalName(reader) + " = ctx." +
            globalName(reader) + ";\n";
    for (const auto* directReader : directReaders) {
        code += "    auto& " + globalName(*directReader) + " = ctx." +
                globalName(*directReader) + ";\n";
    }
    code += "    auto& " + localName(writer) + " = ctx." +
            localName(writer) + ";\n";
    if (!plan.orLoopLower.hoistReaderSync) {
        code += "    const int64_t __or_reader_dense_count = " +
                checkedMulExpr(
                    "__or_input_rows", "__or_input_cols",
                    "[DACPP][MPI][OR] StencilWindow2D reader buffer size overflow") +
                ";\n";
        code += "    " + checkedMpiCountExpr(
                "__or_reader_dense_count",
                "[DACPP][MPI][OR] StencilWindow2D reader buffer size exceeds MPI int range") +
                ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) + ".tensor2Array(" +
                globalName(reader) + ");\n";
        code += "    }\n";
        code += "    " + globalName(reader) +
                ".resize(static_cast<std::size_t>(__or_reader_dense_count));\n";
        code += "    MPI_Bcast(" + globalName(reader) + ".data(), " +
                checkedDense2DPayloadCountExpr(
                    "__or_input_rows", "__or_input_cols", readerType,
                    "[DACPP][MPI][OR] StencilWindow2D reader broadcast count exceeds MPI int range") +
                ", " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        const std::string directMpiType = mpiDatatypeFor(directType);
        code += "    const int64_t __or_direct_dense_count_" +
                directReader->calcParamName + " = " +
                checkedMulExpr(
                    "__or_output_rows", "__or_output_cols",
                    "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size overflow") +
                ";\n";
        code += "    " + checkedMpiCountExpr(
                "__or_direct_dense_count_" + directReader->calcParamName,
                "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size exceeds MPI int range") +
                ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(*directReader) + ".tensor2Array(" +
                globalName(*directReader) + ");\n";
        code += "    }\n";
        code += "    " + globalName(*directReader) +
                ".resize(static_cast<std::size_t>(__or_direct_dense_count_" +
                directReader->calcParamName + "));\n";
        code += "    MPI_Bcast(" + globalName(*directReader) + ".data(), " +
                checkedDense2DPayloadCountExpr(
                    "__or_output_rows", "__or_output_cols", directType,
                    "[DACPP][MPI][OR] StencilWindow2D direct-reader broadcast count exceeds MPI int range") +
                ", " + directMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType +
            ", 1> __or_reader_buf(" + globalName(reader) +
            ".data(), sycl::range<1>(" + globalName(reader) +
            ".size()));\n";
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        code += "        sycl::buffer<" + directType + ", 1> __or_buffer_" +
                directReader->calcParamName + "(" +
                globalName(*directReader) +
                ".data(), sycl::range<1>(" +
                globalName(*directReader) + ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(" +
            localName(writer) + ".data(), sycl::range<1>(" +
            localName(writer) + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    for (const auto* directReader : directReaders) {
        code += "            auto __or_acc_" + directReader->calcParamName +
                " = __or_buffer_" + directReader->calcParamName +
                ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int local_row = item_linear / static_cast<int>(__or_output_cols);\n";
    code += "                const int local_col = item_linear % static_cast<int>(__or_output_cols);\n";
    code += "                const int output_row = static_cast<int>(__or_output_row_begin) + local_row;\n";
    code += "                const int output_col = local_col;\n";
    code += "                const int input_row = output_row;\n";
    code += "                const int input_col = output_col;\n";
    code += "                const int global_output_linear = output_row * static_cast<int>(__or_output_cols) + output_col;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto* directReader : directReaders) {
        code += "                auto* __or_data_" +
                directReader->calcParamName + " = __or_acc_" +
                directReader->calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ContiguousView2D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, input_row * static_cast<int>(__or_input_cols) + input_col, static_cast<int>(__or_input_cols)};\n";
            continue;
        }
        if (param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            code += "                dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_data_" + param.calcParamName +
                    ", global_output_linear};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += "                dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, item_linear};\n";
            continue;
        }
    }
    code += "                " + calcName + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "        q.wait();\n";
    code += "    }\n";
    code += "    auto& __or_resident_out_" + writer.calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" +
            writerType + ">(" + paramVarName(writer) + ", " +
            localName(writer) + ".size());\n";
    code += "    __or_resident_out_" + writer.calcParamName + " = " +
            localName(writer) + ";\n";
    code += "    const int64_t __or_materialized_output_items = "
            "dacpp::mpi::operator_resident::checked_mul_int64_or_abort("
            "static_cast<int64_t>(__or_output_rows), "
            "static_cast<int64_t>(__or_output_cols), "
            "\"[DACPP][MPI][OR] materialized output size exceeds MPI int range\");\n";
    emitGatherMaterializeFromLocalBuffer(
        code, plan, writer, localName(writer),
        "__or_materialized_output_items");
    emitReadCacheTransitions2D(code, plan, sitePlan.readCacheTransitions);
    emitFollowupMaterialize2D(code, plan, writer, followups,
                              sitePlan.boundaryLocalUpdates);
    code += "}\n";
}

void emitLoopLoweredMaterializeFunction2D(std::string& code,
                                          const std::string& ctxName,
                                          const std::string& materializeName,
                                          const ShellPartitionPlan& plan) {
    code += "void " + materializeName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    (void)ctx;\n";
    for (const auto& param : plan.params) {
        code += "    (void)" + paramVarName(param) + ";\n";
    }
    code += "}\n";
}

void emitResidentHaloBoundaryRefresh2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<BoundaryLocalUpdate>& updates,
    const ParamAccessPlan& reader,
    const std::string& localReaderName,
    const std::string& localRowBeginExpr,
    const std::string& localRowCountExpr,
    const std::string& readerRowsExpr,
    const std::string& readerColsExpr) {
    for (const auto& update : updates) {
        if (update.rank != 2 || update.paramIndex != reader.paramIndex) {
            continue;
        }
        if (!update.constantRhs &&
            update.sourceParamIndex != reader.paramIndex) {
            continue;
        }
        const std::string loopLower =
            update.loopLowerExpr.empty() ? "0" : update.loopLowerExpr;
        const std::string loopUpper =
            update.loopUpperExpr.empty()
                ? (update.targetRowUsesLoop ? readerRowsExpr : readerColsExpr)
                : update.loopUpperExpr;
        const std::string loopCmp = update.loopUpperInclusive ? "<=" : "<";
        code += "    {\n";
        code += "        const int64_t __or_boundary_begin = static_cast<int64_t>(" +
                loopLower + ");\n";
        code += "        const int64_t __or_boundary_end = static_cast<int64_t>(" +
                loopUpper + ");\n";
        code += "        for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx " +
                loopCmp +
                " __or_boundary_end; ++__or_boundary_idx) {\n";
        code += "            const int64_t __or_boundary_target_row = " +
                (update.targetRowUsesLoop ? "__or_boundary_idx"
                                          : update.targetRowExpr) +
                ";\n";
        code += "            const int64_t __or_boundary_target_col = " +
                (update.targetColUsesLoop ? "__or_boundary_idx"
                                          : update.targetColExpr) +
                ";\n";
        code += "            const int64_t __or_boundary_source_row = " +
                (update.sourceRowUsesLoop ? "__or_boundary_idx"
                                          : update.sourceRowExpr) +
                ";\n";
        code += "            const int64_t __or_boundary_source_col = " +
                (update.sourceColUsesLoop ? "__or_boundary_idx"
                                          : update.sourceColExpr) +
                ";\n";
        code += "            if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= " +
                readerRowsExpr + " || __or_boundary_target_col >= " +
                readerColsExpr + ") continue;\n";
        code += "            if (__or_boundary_target_row < " +
                localRowBeginExpr + " || __or_boundary_target_row >= " +
                localRowBeginExpr + " + " + localRowCountExpr +
                ") continue;\n";
        code += "            const int64_t __or_boundary_target_local = (__or_boundary_target_row - " +
                localRowBeginExpr + ") * " + readerColsExpr +
                " + __or_boundary_target_col;\n";
        if (update.constantRhs) {
            code += "            if (__or_boundary_target_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= " +
                    localReaderName + ".size()) continue;\n";
            code += "            " + localReaderName +
                    "[static_cast<std::size_t>(__or_boundary_target_local)] = static_cast<" +
                    elemType(plan, reader) +
                    ">(" +
                    (update.constantValue.empty() ? "0"
                                                  : update.constantValue) +
                    ");\n";
        } else {
            code += "            if (__or_boundary_source_row < 0 || __or_boundary_source_col < 0 || __or_boundary_source_row >= " +
                    readerRowsExpr + " || __or_boundary_source_col >= " +
                    readerColsExpr + ") continue;\n";
            code += "            if (__or_boundary_source_row < " +
                    localRowBeginExpr + " || __or_boundary_source_row >= " +
                    localRowBeginExpr + " + " + localRowCountExpr +
                    ") continue;\n";
            code += "            const int64_t __or_boundary_source_local = (__or_boundary_source_row - " +
                    localRowBeginExpr + ") * " + readerColsExpr +
                    " + __or_boundary_source_col;\n";
            code += "            if (__or_boundary_target_local < 0 || __or_boundary_source_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= " +
                    localReaderName + ".size() || static_cast<std::size_t>(__or_boundary_source_local) >= " +
                    localReaderName + ".size()) continue;\n";
            code += "            " + localReaderName +
                    "[static_cast<std::size_t>(__or_boundary_target_local)] = " +
                    localReaderName +
                    "[static_cast<std::size_t>(__or_boundary_source_local)];\n";
        }
        code += "        }\n";
        code += "    }\n";
    }
}

void emitResidentHaloContextType2D(std::string& code,
                                   const std::string& ctxName,
                                   const ShellPartitionPlan& plan,
                                   const ParamAccessPlan& reader,
                                   const ParamAccessPlan& writer,
                                   const ParamAccessPlan* directReader) {
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t __or_input_rows = 0;\n";
    code += "    int64_t __or_input_cols = 0;\n";
    code += "    int64_t __or_output_rows = 0;\n";
    code += "    int64_t __or_output_cols = 0;\n";
    code += "    int64_t __or_local_output_rows = 0;\n";
    code += "    int64_t __or_output_row_begin = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    int __or_window_rows = " +
            std::to_string(plan.orLoopLower.stencilResidentHalo.windowRows) +
            ";\n";
    code += "    int __or_window_cols = " +
            std::to_string(plan.orLoopLower.stencilResidentHalo.windowCols) +
            ";\n";
    code += "    int __or_followup_row_offset = " +
            std::to_string(
                plan.orLoopLower.stencilResidentHalo.followupTargetRowOffset) +
            ";\n";
    code += "    int __or_followup_col_offset = " +
            std::to_string(
                plan.orLoopLower.stencilResidentHalo.followupTargetColOffset) +
            ";\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_output_row_range{};\n";
    code += "    dacpp::mpi::operator_resident::ResidentHalo2DRowLayout __or_halo_layout{};\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    sycl::queue q{sycl::default_selector_v};\n";
    code += "    std::vector<" + elemType(plan, reader) + "> " +
            localName(reader) + ";\n";
    if (directReader) {
        code += "    std::vector<" + elemType(plan, *directReader) + "> " +
                localName(*directReader) + ";\n";
    }
    code += "    std::vector<" + elemType(plan, writer) + "> " +
            localName(writer) + ";\n";
    code += "};\n";
}

void emitResidentHaloInitFunction2D(std::string& code,
                                    const std::string& ctxName,
                                    const std::string& initName,
                                    const ShellPartitionPlan& plan,
                                    const ParamAccessPlan& reader,
                                    const ParamAccessPlan& writer,
                                    const ParamAccessPlan* directReader) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string directReaderType =
        directReader ? elemType(plan, *directReader) : std::string();
    const std::string directReaderMpiType =
        directReader ? mpiDatatypeFor(directReaderType) : std::string();
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.__or_input_rows = " + paramVarName(reader) +
            ".getShape(0);\n";
    code += "    ctx.__or_input_cols = " + paramVarName(reader) +
            ".getShape(1);\n";
    code += "    ctx.__or_output_rows = " + paramVarName(writer) +
            ".getShape(0);\n";
    code += "    ctx.__or_output_cols = " + paramVarName(writer) +
            ".getShape(1);\n";
    code += "    ctx.__or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_rows, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.__or_local_output_rows = ctx.__or_output_row_range.count;\n";
    code += "    ctx.__or_output_row_begin = ctx.__or_output_row_range.begin;\n";
    code += "    ctx.__or_local_item_count = " +
            checkedMulExpr(
                "ctx.__or_local_output_rows", "ctx.__or_output_cols",
                "[DACPP][MPI][OR] resident halo 2D local item count overflow") +
            ";\n";
    code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout(ctx.__or_output_rows, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_window_rows);\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_output_rows, ctx.mpi_size, ctx.__or_row_counts, ctx.__or_row_displs);\n";
    code += "    ctx.__or_counts.resize(static_cast<std::size_t>(ctx.mpi_size));\n";
    code += "    ctx.__or_displs.resize(static_cast<std::size_t>(ctx.mpi_size));\n";
    code += "    for (int r = 0; r < ctx.mpi_size; ++r) {\n";
    code += "        ctx.__or_counts[static_cast<std::size_t>(r)] = " +
            checkedMpiProductCountExpr(
                "ctx.__or_row_counts[static_cast<std::size_t>(r)]",
                "ctx.__or_output_cols",
                "[DACPP][MPI][OR] resident halo 2D gather count exceeds MPI int range") +
            ";\n";
    code += "        ctx.__or_displs[static_cast<std::size_t>(r)] = " +
            checkedMpiProductCountExpr(
                "ctx.__or_row_displs[static_cast<std::size_t>(r)]",
                "ctx.__or_output_cols",
                "[DACPP][MPI][OR] resident halo 2D gather displacement exceeds MPI int range") +
            ";\n";
    code += "    }\n";
    code += "    " + checkedMpiCountExpr(
            "ctx.__or_halo_layout.local_size",
            "[DACPP][MPI][OR] resident halo 2D local reader size exceeds MPI int range") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "ctx.__or_local_item_count",
            "[DACPP][MPI][OR] resident halo 2D local writer size exceeds MPI int range") +
            ";\n";
    code += "    ctx." + localName(reader) +
            ".assign(static_cast<std::size_t>(ctx.__or_halo_layout.local_size), " +
            readerType + "{});\n";
    if (directReader) {
        code += "    ctx." + localName(*directReader) +
                ".assign(static_cast<std::size_t>(ctx.__or_local_item_count), " +
                directReaderType + "{});\n";
    }
    code += "    ctx." + localName(writer) +
            ".assign(static_cast<std::size_t>(ctx.__or_local_item_count), " +
            elemType(plan, writer) + "{});\n";
    code += "    const int64_t __or_initial_reader_dense_count = " +
            checkedMulExpr(
                "ctx.__or_input_rows", "ctx.__or_input_cols",
                "[DACPP][MPI][OR] resident halo 2D initial reader size overflow") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "__or_initial_reader_dense_count",
            "[DACPP][MPI][OR] resident halo 2D initial reader size exceeds MPI int range") +
            ";\n";
    code += "    std::vector<" + readerType + "> __or_initial_global_" +
            reader.calcParamName + ";\n";
    code += "    if (ctx.mpi_rank == 0) {\n";
    code += "        " + paramVarName(reader) +
            ".tensor2Array(__or_initial_global_" + reader.calcParamName +
            ");\n";
    code += "    }\n";
    code += "    dacpp::mpi::operator_resident::scatter_window_2d_rows(__or_initial_global_" +
            reader.calcParamName + ", ctx." + localName(reader) +
            ", ctx.__or_output_rows, ctx.__or_input_cols, ctx.__or_window_rows, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
            readerMpiType + ");\n";
    if (directReader) {
        code += "    const int64_t __or_direct_rows_" +
                directReader->calcParamName + " = " +
                paramVarName(*directReader) + ".getShape(0);\n";
        code += "    const int64_t __or_direct_cols_" +
                directReader->calcParamName + " = " +
                paramVarName(*directReader) + ".getShape(1);\n";
        code += "    if (__or_direct_rows_" + directReader->calcParamName +
                " != ctx.__or_output_rows || __or_direct_cols_" +
                directReader->calcParamName + " != ctx.__or_output_cols) {\n";
        code += "        if (ctx.mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] resident halo direct reader " +
                directReader->shellParamName +
                " shape mismatch with output\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 4);\n";
        code += "    }\n";
        code += "    std::vector<" + directReaderType + "> __or_initial_global_" +
                directReader->calcParamName + ";\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(*directReader) +
                ".tensor2Array(__or_initial_global_" +
                directReader->calcParamName + ");\n";
        code += "    }\n";
        code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_initial_global_" +
                directReader->calcParamName +
                ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                directReaderMpiType + ", ctx." + localName(*directReader) +
                ".data(), " + checkedMpiCountExpr(
                    "ctx.__or_local_item_count",
                    "[DACPP][MPI][OR] resident halo direct reader scatter count exceeds MPI int range") +
                ", " + directReaderMpiType +
                ", 0, MPI_COMM_WORLD);\n";
        code += "    auto& __or_resident_in_" + directReader->calcParamName +
                " = dacpp::mpi::operator_resident::ensure_resident<" +
                directReaderType + ">(" + paramVarName(*directReader) +
                ", ctx." + localName(*directReader) + ".size());\n";
        code += "    __or_resident_in_" + directReader->calcParamName +
                " = ctx." + localName(*directReader) + ";\n";
    }
    code += "}\n";
}

void emitResidentHaloRunFunction2D(std::string& code,
                                   const std::string& ctxName,
                                   const std::string& runName,
                                   DacppFile* dacppFile,
                                   const ShellPartitionPlan& plan,
                                   const ParamAccessPlan& reader,
                                   const ParamAccessPlan& writer,
                                   const ParamAccessPlan* directReader) {
    const std::string readerType = elemType(plan, reader);
    const std::string writerType = elemType(plan, writer);
    const std::string writerMpiType = mpiDatatypeFor(writerType);
    const std::string directReaderType =
        directReader ? elemType(plan, *directReader) : std::string();
    const std::string calcName = plan.exprNode.calc->getName();
    const DistributedStencilSitePlan sitePlan =
        stencilSitePlanFor(dacppFile, plan);
    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto& q = ctx.q;\n";
    code += "    const int64_t __or_input_rows = ctx.__or_input_rows;\n";
    code += "    const int64_t __or_input_cols = ctx.__or_input_cols;\n";
    code += "    const int64_t __or_output_rows = ctx.__or_output_rows;\n";
    code += "    const int64_t __or_output_cols = ctx.__or_output_cols;\n";
    code += "    const int64_t __or_local_output_rows = ctx.__or_local_output_rows;\n";
    code += "    const int64_t __or_local_item_count = ctx.__or_local_item_count;\n";
    code += "    const int64_t __or_local_reader_rows = ctx.__or_halo_layout.local_row_count;\n";
    code += "    const int64_t __or_local_reader_row_begin = ctx.__or_halo_layout.global_row_begin;\n";
    code += "    auto& " + localName(reader) + " = ctx." +
            localName(reader) + ";\n";
    if (directReader) {
        code += "    auto& " + localName(*directReader) + " = ctx." +
                localName(*directReader) + ";\n";
    }
    code += "    auto& " + localName(writer) + " = ctx." +
            localName(writer) + ";\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(" +
            localName(reader) + ".data(), sycl::range<1>(" +
            localName(reader) + ".size()));\n";
    if (directReader) {
        code += "        sycl::buffer<" + directReaderType +
                ", 1> __or_direct_reader_buf(" + localName(*directReader) +
                ".data(), sycl::range<1>(" + localName(*directReader) +
                ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(" +
            localName(writer) + ".data(), sycl::range<1>(" +
            localName(writer) + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    if (directReader) {
        code += "            auto __or_direct_reader_acc = __or_direct_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int local_row = item_linear / static_cast<int>(__or_output_cols);\n";
    code += "                const int local_col = item_linear % static_cast<int>(__or_output_cols);\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    if (directReader) {
        code += "                auto* __or_direct_reader_data = __or_direct_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ResidentHaloView2D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, local_row * static_cast<int>(__or_input_cols) + local_col, static_cast<int>(__or_input_cols)};\n";
            continue;
        }
        if (directReader &&
            param.paramIndex == directReader->paramIndex) {
            code += "                dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_direct_reader_data, item_linear};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += "                dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, item_linear};\n";
            continue;
        }
    }
    code += "                " + calcName + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "        q.wait();\n";
    code += "    }\n";
    if (directReader) {
        code += "    dacpp::mpi::operator_resident::apply_read_cache_transition_2d(" +
                localName(*directReader) + ", " + localName(reader) +
                ", __or_local_output_rows, __or_output_cols, __or_input_cols, " +
                std::to_string(
                    plan.orLoopLower.stencilResidentHalo.readCacheTargetRowOffset) +
                ", " +
                std::to_string(
                    plan.orLoopLower.stencilResidentHalo.readCacheTargetColOffset) +
                ");\n";
        code += "    auto& __or_resident_in_" + directReader->calcParamName +
                " = dacpp::mpi::operator_resident::ensure_resident<" +
                directReaderType + ">(" + paramVarName(*directReader) +
                ", " + localName(*directReader) + ".size());\n";
        code += "    __or_resident_in_" + directReader->calcParamName +
                " = " + localName(*directReader) + ";\n";
    }
    code += "    dacpp::mpi::operator_resident::apply_followup_2d(" +
            localName(reader) + ", " + localName(writer) +
            ", __or_local_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset);\n";
    code += "    dacpp::mpi::operator_resident::exchange_halo_2d_rows(" +
            localName(reader) + ", " + localName(writer) +
            ", ctx.__or_halo_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, mpi_rank, ctx.mpi_size, " +
            writerMpiType + ");\n";
    emitResidentHaloBoundaryRefresh2D(
        code, plan, sitePlan.boundaryLocalUpdates, reader, localName(reader),
        "__or_local_reader_row_begin", "__or_local_reader_rows",
        "__or_input_rows", "__or_input_cols");
    code += "    auto& __or_resident_out_" + writer.calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" +
            writerType + ">(" + paramVarName(writer) + ", " +
            localName(writer) + ".size());\n";
    code += "    __or_resident_out_" + writer.calcParamName + " = " +
            localName(writer) + ";\n";
    code += "}\n";
}

void emitResidentHaloMaterializeFunction2D(
    std::string& code,
    const std::string& ctxName,
    const std::string& materializeName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const ParamAccessPlan* directReader) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string writerType = elemType(plan, writer);
    const std::string writerMpiType = mpiDatatypeFor(writerType);
    const std::string directReaderType =
        directReader ? elemType(plan, *directReader) : std::string();
    const std::string directReaderMpiType =
        directReader ? mpiDatatypeFor(directReaderType) : std::string();
    const DistributedStencilSitePlan sitePlan =
        stencilSitePlanFor(dacppFile, plan);
    const auto& halo = plan.orLoopLower.stencilResidentHalo;
    code += "void " + materializeName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    std::vector<" + writerType + "> __or_materialized_" +
            writer.calcParamName + ";\n";
    code += "    const int64_t __or_materialized_writer_count = " +
            checkedMulExpr(
                "ctx.__or_output_rows", "ctx.__or_output_cols",
                "[DACPP][MPI][OR] resident halo 2D materialized output size overflow") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "__or_materialized_writer_count",
            "[DACPP][MPI][OR] resident halo 2D materialized output size exceeds MPI int range") +
            ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        __or_materialized_" + writer.calcParamName +
            ".resize(static_cast<std::size_t>(__or_materialized_writer_count));\n";
    code += "    }\n";
    code += "    MPI_Gatherv(ctx." + localName(writer) +
            ".data(), " +
            checkedMpiCountExpr(
                "ctx.__or_local_item_count",
                "[DACPP][MPI][OR] resident halo 2D materialize sendcount exceeds MPI int range") +
            ", " +
            writerMpiType + ", mpi_rank == 0 ? __or_materialized_" +
            writer.calcParamName +
            ".data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
            writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(writer) + ".array2Tensor(__or_materialized_" +
            writer.calcParamName + ");\n";
    code += "    }\n";
    if (directReader) {
        code += "    std::vector<" + directReaderType + "> __or_materialized_" +
                directReader->calcParamName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        __or_materialized_" + directReader->calcParamName +
                ".resize(static_cast<std::size_t>(__or_materialized_writer_count));\n";
        code += "    }\n";
        code += "    MPI_Gatherv(ctx." + localName(*directReader) +
                ".data(), " + checkedMpiCountExpr(
                    "ctx.__or_local_item_count",
                    "[DACPP][MPI][OR] resident halo 2D direct-reader materialize sendcount exceeds MPI int range") +
                ", " + directReaderMpiType +
                ", mpi_rank == 0 ? __or_materialized_" +
                directReader->calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                directReaderMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(*directReader) +
                ".array2Tensor(__or_materialized_" +
                directReader->calcParamName + ");\n";
        code += "    }\n";
    }
    code += "    if (mpi_rank == 0) {\n";
    code += "        std::vector<" + readerType + "> __or_materialized_" +
            reader.calcParamName + ";\n";
    code += "        " + paramVarName(reader) + ".tensor2Array(__or_materialized_" +
            reader.calcParamName + ");\n";
    code += "        const int64_t __or_followup_reader_cols_" +
            reader.calcParamName + " = static_cast<int64_t>(" +
            paramVarName(reader) + ".getShape(1));\n";
    code += "        for (std::size_t __or_idx = 0; __or_idx < __or_materialized_" +
            writer.calcParamName + ".size(); ++__or_idx) {\n";
    code += "            const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), ctx.__or_output_cols, __or_followup_reader_cols_" +
            reader.calcParamName + ", " +
            std::to_string(halo.followupTargetRowOffset) + ", " +
            std::to_string(halo.followupTargetColOffset) + ");\n";
    code += "            if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < __or_materialized_" +
            reader.calcParamName + ".size()) {\n";
    code += "                __or_materialized_" + reader.calcParamName +
            "[static_cast<std::size_t>(__or_target)] = static_cast<" +
            readerType + ">(__or_materialized_" + writer.calcParamName +
            "[__or_idx]);\n";
    code += "            }\n";
    code += "        }\n";
    emitBoundaryLocalMaterialize2D(
        code, plan, sitePlan.boundaryLocalUpdates, reader, writer,
        "__or_materialized_" + writer.calcParamName,
        "__or_materialized_" + reader.calcParamName,
        paramVarName(reader) + ".getShape(0)",
        "__or_followup_reader_cols_" + reader.calcParamName,
        "ctx.__or_output_rows", "ctx.__or_output_cols");
    code += "        " + paramVarName(reader) + ".array2Tensor(__or_materialized_" +
            reader.calcParamName + ");\n";
    code += "    }\n";
    for (const auto& param : plan.params) {
        if (param.paramIndex != reader.paramIndex &&
            param.paramIndex != writer.paramIndex) {
            code += "    (void)" + paramVarName(param) + ";\n";
        }
    }
    code += "}\n";
}

}  // namespace

std::string buildStencilWindow2DWrapperCode(
    const std::string& wrapperName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    const auto directReaders = findStencilDirectReaders(plan);
    if (!reader || !writer ||
        !supportsStencilWindow2DParams(plan, true)) {
        return {};
    }

    const std::string readerType = elemType(plan, *reader);
    const std::string writerType = elemType(plan, *writer);
    const std::string readerMpiType = mpi_rewriter::mpiDatatypeFor(readerType);
    const std::string writerMpiType = mpi_rewriter::mpiDatatypeFor(writerType);
    const std::string readerArg = paramVarName(*reader);
    const std::string writerArg = paramVarName(*writer);
    const std::string calcName = plan.exprNode.calc->getName();
    const DistributedStencilSitePlan sitePlan =
        stencilSitePlanFor(dacppFile, plan);
    const auto followups = followupsForWriter(sitePlan, *writer);

    std::string code;
    code += "void " + wrapperName + "(" + wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    const int64_t __or_input_rows = " + readerArg + ".getShape(0);\n";
    code += "    const int64_t __or_input_cols = " + readerArg + ".getShape(1);\n";
    code += "    const int64_t __or_output_rows = " + writerArg + ".getShape(0);\n";
    code += "    const int64_t __or_output_cols = " + writerArg + ".getShape(1);\n";
    code += "    const auto __or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_rows, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_output_rows = __or_output_row_range.count;\n";
    code += "    const int64_t __or_output_row_begin = __or_output_row_range.begin;\n";
    code += "    const int64_t __or_local_item_count = " +
            checkedMulExpr(
                "__or_local_output_rows", "__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D local item count overflow") +
            ";\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_output_rows, mpi_size, __or_row_counts, __or_row_displs);\n";
    code += "    std::vector<int> __or_counts(mpi_size);\n";
    code += "    std::vector<int> __or_displs(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_counts[r] = " +
            checkedMpiProductCountExpr(
                "__or_row_counts[r]", "__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D row-block gather count exceeds MPI int range") +
            ";\n";
    code += "        __or_displs[r] = " +
            checkedMpiProductCountExpr(
                "__or_row_displs[r]", "__or_output_cols",
                "[DACPP][MPI][OR] StencilWindow2D row-block gather displacement exceeds MPI int range") +
            ";\n";
    code += "    }\n";
    code += "    const int64_t __or_reader_dense_count = " +
            checkedMulExpr(
                "__or_input_rows", "__or_input_cols",
                "[DACPP][MPI][OR] StencilWindow2D reader buffer size overflow") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "__or_reader_dense_count",
            "[DACPP][MPI][OR] StencilWindow2D reader buffer size exceeds MPI int range") +
            ";\n";
    code += "    std::vector<" + readerType + "> __or_global_" + reader->calcParamName + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + readerArg + ".tensor2Array(__or_global_" + reader->calcParamName + ");\n";
    code += "    }\n";
    code += "    __or_global_" + reader->calcParamName + ".resize(static_cast<std::size_t>(__or_reader_dense_count));\n";
    code += "    MPI_Bcast(__or_global_" + reader->calcParamName + ".data(), " +
            checkedDense2DPayloadCountExpr(
                "__or_input_rows", "__or_input_cols", readerType,
                "[DACPP][MPI][OR] StencilWindow2D reader broadcast count exceeds MPI int range") +
            ", " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        const std::string directMpiType =
            mpi_rewriter::mpiDatatypeFor(directType);
        const std::string directArg = paramVarName(*directReader);
        code += "    const int64_t __or_direct_rows_" + directReader->calcParamName +
                " = " + directArg + ".getShape(0);\n";
        code += "    const int64_t __or_direct_cols_" + directReader->calcParamName +
                " = " + directArg + ".getShape(1);\n";
        code += "    if (__or_direct_rows_" + directReader->calcParamName +
                " != __or_output_rows || __or_direct_cols_" +
                directReader->calcParamName + " != __or_output_cols) {\n";
        code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] stencil direct reader " +
                directReader->shellParamName +
                " shape mismatch with output\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 4);\n";
        code += "    }\n";
        code += "    const int64_t __or_direct_dense_count_" +
                directReader->calcParamName + " = " +
                checkedMulExpr(
                    "__or_output_rows", "__or_output_cols",
                    "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size overflow") +
                ";\n";
        code += "    " + checkedMpiCountExpr(
                "__or_direct_dense_count_" + directReader->calcParamName,
                "[DACPP][MPI][OR] StencilWindow2D direct-reader buffer size exceeds MPI int range") +
                ";\n";
        code += "    std::vector<" + directType + "> __or_global_" +
                directReader->calcParamName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + directArg + ".tensor2Array(__or_global_" +
                directReader->calcParamName + ");\n";
        code += "    }\n";
        code += "    __or_global_" + directReader->calcParamName +
                ".resize(static_cast<std::size_t>(__or_direct_dense_count_" +
                directReader->calcParamName + "));\n";
        code += "    MPI_Bcast(__or_global_" + directReader->calcParamName +
                ".data(), " +
                checkedDense2DPayloadCountExpr(
                    "__or_output_rows", "__or_output_cols", directType,
                    "[DACPP][MPI][OR] StencilWindow2D direct-reader broadcast count exceeds MPI int range") +
                ", " + directMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    " + checkedMpiCountExpr(
            "__or_local_item_count",
            "[DACPP][MPI][OR] StencilWindow2D local writer size exceeds MPI int range") +
            ";\n";
    code += "    std::vector<" + writerType + "> __or_local_" + writer->calcParamName + "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(__or_global_" + reader->calcParamName + ".data(), sycl::range<1>(__or_global_" + reader->calcParamName + ".size()));\n";
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        code += "        sycl::buffer<" + directType + ", 1> __or_buffer_" +
                directReader->calcParamName + "(__or_global_" +
                directReader->calcParamName + ".data(), sycl::range<1>(__or_global_" +
                directReader->calcParamName + ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(__or_local_" + writer->calcParamName + ".data(), sycl::range<1>(__or_local_" + writer->calcParamName + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    for (const auto* directReader : directReaders) {
        code += "            auto __or_acc_" + directReader->calcParamName +
                " = __or_buffer_" + directReader->calcParamName +
                ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int local_row = item_linear / static_cast<int>(__or_output_cols);\n";
    code += "                const int local_col = item_linear % static_cast<int>(__or_output_cols);\n";
    code += "                const int output_row = static_cast<int>(__or_output_row_begin) + local_row;\n";
    code += "                const int output_col = local_col;\n";
    code += "                const int input_row = output_row;\n";
    code += "                const int input_col = output_col;\n";
    code += "                const int global_output_linear = output_row * static_cast<int>(__or_output_cols) + output_col;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto* directReader : directReaders) {
        code += "                auto* __or_data_" + directReader->calcParamName +
                " = __or_acc_" + directReader->calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ContiguousView2D<const " + paramType + "> view_" + param.calcParamName + "{__or_reader_data, input_row * static_cast<int>(__or_input_cols) + input_col, static_cast<int>(__or_input_cols)};\n";
            continue;
        }
        if (param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            code += "                dacpp::mpi::ContiguousView1D<const " + paramType + "> view_" + param.calcParamName + "{__or_data_" + param.calcParamName + ", global_output_linear};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += "                dacpp::mpi::ContiguousView1D<" + paramType + "> view_" + param.calcParamName + "{__or_writer_data, item_linear};\n";
            continue;
        }
    }
    code += "                " + calcName + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "        q.wait();\n";
    code += "    }\n";
    code += "    auto& __or_resident_out_" + writer->calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" + writerType +
            ">(" + writerArg + ", __or_local_" + writer->calcParamName +
            ".size());\n";
    code += "    __or_resident_out_" + writer->calcParamName + " = __or_local_" +
            writer->calcParamName + ";\n";
    code += "    const int64_t __or_materialized_output_items = "
            "dacpp::mpi::operator_resident::checked_mul_int64_or_abort("
            "static_cast<int64_t>(__or_output_rows), "
            "static_cast<int64_t>(__or_output_cols), "
            "\"[DACPP][MPI][OR] materialized output size exceeds MPI int range\");\n";
    emitGatherMaterializeFromLocalBuffer(
        code, plan, *writer, "__or_local_" + writer->calcParamName,
        "__or_materialized_output_items");
    emitReadCacheTransitions2D(code, plan, sitePlan.readCacheTransitions);
    emitFollowupMaterialize2D(code, plan, *writer, followups,
                              sitePlan.boundaryLocalUpdates);
    code += "}\n";
    return code;
}

std::string buildLoopLoweredStencil2DFullSyncFamilyCode(
    const std::string& baseName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    const auto directReaders = findStencilDirectReaders(plan);
    if (!reader || !writer || !plan.exprNode.shell || !plan.exprNode.calc ||
        !supportsStencilWindow2DParams(plan, true)) {
        return {};
    }

    Shell* shell = plan.exprNode.shell;
    Calc* calc = plan.exprNode.calc;
    const int exprIndex = plan.exprIndex;
    const std::string ctxName =
        operatorResidentContextTypeName(shell, calc, exprIndex);
    const std::string initName =
        operatorResidentInitFunctionName(shell, calc, exprIndex);
    const std::string runName =
        operatorResidentRunFunctionName(shell, calc, exprIndex);
    const std::string materializeName =
        operatorResidentMaterializeFunctionName(shell, calc, exprIndex);

    std::string code;
    emitLoopLoweredContextType2D(code, ctxName, plan, *reader, *writer,
                                 directReaders);
    emitLoopLoweredInitFunction2D(code, ctxName, initName, plan, *reader,
                                  *writer, directReaders);
    emitLoopLoweredRunFunction2D(code, ctxName, runName, dacppFile, plan,
                                 *reader, *writer, directReaders);
    emitLoopLoweredMaterializeFunction2D(code, ctxName, materializeName, plan);
    code += "void " + baseName + "(" + wrapperSignature(plan) + ") {\n";
    code += "    " + ctxName + " ctx;\n";
    code += "    " + initName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + materializeName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "}\n";
    return code;
}

std::string buildLoopLoweredStencil2DResidentHaloFamilyCode(
    const std::string& baseName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    const auto directReaders = findStencilDirectReaders(plan);
    const ParamAccessPlan* directReader =
        directReaders.size() == 1 ? directReaders.front() : nullptr;
    if (!reader || !writer || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.orLoopLower.stencilResidentHalo.enabled ||
        directReaders.size() > 1 ||
        (plan.orLoopLower.stencilResidentHalo.hasDirectReader !=
         (directReader != nullptr)) ||
        !supportsStencilWindow2DParams(plan, directReader != nullptr)) {
        return {};
    }

    Shell* shell = plan.exprNode.shell;
    Calc* calc = plan.exprNode.calc;
    const int exprIndex = plan.exprIndex;
    const std::string ctxName =
        operatorResidentContextTypeName(shell, calc, exprIndex);
    const std::string initName =
        operatorResidentInitFunctionName(shell, calc, exprIndex);
    const std::string runName =
        operatorResidentRunFunctionName(shell, calc, exprIndex);
    const std::string materializeName =
        operatorResidentMaterializeFunctionName(shell, calc, exprIndex);

    std::string code;
    emitResidentHaloContextType2D(code, ctxName, plan, *reader, *writer,
                                  directReader);
    emitResidentHaloInitFunction2D(code, ctxName, initName, plan, *reader,
                                   *writer, directReader);
    emitResidentHaloRunFunction2D(code, ctxName, runName, dacppFile, plan,
                                  *reader, *writer, directReader);
    emitResidentHaloMaterializeFunction2D(code, ctxName, materializeName,
                                          dacppFile, plan, *reader, *writer,
                                          directReader);
    code += "void " + baseName + "(" + wrapperSignature(plan) + ") {\n";
    code += "    " + ctxName + " ctx;\n";
    code += "    " + initName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + materializeName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "}\n";
    return code;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
