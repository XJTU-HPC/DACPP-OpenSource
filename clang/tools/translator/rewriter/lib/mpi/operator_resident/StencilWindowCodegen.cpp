#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"
#include "ShellPartitionAnalysis_Internal.h"

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

bool hasFullPostUse(const ParamAccessPlan& param) {
    return param.postUseSync.kind == PostUseSyncKind::FullTensor ||
           param.broadcastMaterializedOutput;
}

bool hasBoundedPostUse(const ParamAccessPlan& param) {
    return param.postUseSync.kind == PostUseSyncKind::BoundedIndexedRootRead &&
           !param.postUseSync.boundedIndices.empty();
}

std::string compactExprText(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(),
                              [](unsigned char c) {
                                  return std::isspace(c) != 0;
                              }),
               text.end());
    return text;
}

bool exprIsStaticIndex(const std::string& exprText,
                       int64_t value,
                       int64_t extent) {
    const std::string compact = compactExprText(exprText);
    if (compact == std::to_string(value)) {
        return true;
    }
    if (extent > 0 && value == extent - 1 &&
        (compact == std::to_string(extent - 1) ||
         (compact.size() >= 2 &&
          compact.compare(compact.size() - 2, 2, "-1") == 0))) {
        return true;
    }
    return false;
}

bool updateLoopCoversIndex(const BoundaryLocalUpdate& update,
                           int64_t index,
                           int64_t extent) {
    const std::string lower = compactExprText(update.loopLowerExpr);
    const std::string upper = compactExprText(update.loopUpperExpr);
    if (lower != "0" || !update.loopUpperInclusive || index < 0) {
        return false;
    }
    if (extent > 0 && index >= extent) {
        return false;
    }
    return (extent > 0 &&
            (upper == std::to_string(extent - 1) ||
             (upper.size() >= 2 &&
              upper.compare(upper.size() - 2, 2, "-1") == 0))) ||
           (!upper.empty() &&
            upper.compare(upper.size() >= 2 ? upper.size() - 2 : 0,
                          upper.size() >= 2 ? 2 : upper.size(),
                          "-1") == 0);
}

bool constantBoundaryValueForIndex(const ShellPartitionPlan& plan,
                                   const ParamAccessPlan& param,
                                   const std::vector<BoundaryLocalUpdate>& updates,
                                   const PostUseBoundedIndex& bounded,
                                   std::string& constantValue) {
    if (bounded.indices.size() != 2) {
        return false;
    }
    const int64_t targetRow = bounded.indices[0];
    const int64_t targetCol = bounded.indices[1];
    const int64_t rows = operator_resident::shapeValueFor(
        plan.exprNode.shell, param.paramIndex, 0);
    const int64_t cols = operator_resident::shapeValueFor(
        plan.exprNode.shell, param.paramIndex, 1);
    for (const auto& update : updates) {
        if (update.rank != 2 || update.paramIndex != param.paramIndex ||
            !update.constantRhs) {
            continue;
        }
        const bool rowMatches =
            update.targetRowUsesLoop
                ? updateLoopCoversIndex(update, targetRow, rows)
                : exprIsStaticIndex(update.targetRowExpr, targetRow, rows);
        const bool colMatches =
            update.targetColUsesLoop
                ? updateLoopCoversIndex(update, targetCol, cols)
                : exprIsStaticIndex(update.targetColExpr, targetCol, cols);
        if (rowMatches && colMatches) {
            constantValue = update.constantValue.empty() ? "0" : update.constantValue;
            return true;
        }
    }
    return false;
}

void emitBoundedReviseValues(std::string& code,
                             const ParamAccessPlan& param,
                             const std::string& valuesName,
                             const std::string& type,
                             const std::string& indent) {
    (void)type;
    for (std::size_t idx = 0; idx < param.postUseSync.boundedIndices.size();
         ++idx) {
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        code += indent + paramVarName(param) + ".reviseValue(" + valuesName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")], std::vector<int>{";
        for (std::size_t dim = 0; dim < bounded.indices.size(); ++dim) {
            if (dim != 0) {
                code += ", ";
            }
            code += std::to_string(bounded.indices[dim]);
        }
        code += "});\n";
    }
}

void emitResidentHaloBoundedIndexedRootRead2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& param,
    const std::string& localBufferName,
    const std::string& rowCountExpr,
    const std::string& ownerRowTotalExpr,
    const std::string& ownerWindowRowsExpr,
    const std::string& logicalColsExpr,
    const std::string& localStrideExpr,
    const std::string& rowOffsetExpr,
    const std::string& colOffsetExpr,
    const std::string& mpiRankExpr,
    const std::string& mpiSizeExpr,
    const std::string& profileName,
    const std::vector<BoundaryLocalUpdate>& boundaryUpdates,
    int tagBase) {
    if (!hasBoundedPostUse(param)) {
        return;
    }
    const std::string type = elemType(plan, param);
    const std::string mpiType = usesByteTransport(type) ? "MPI_BYTE"
                                                        : mpiDatatypeFor(type);
    const std::string valueMpiCount =
        usesByteTransport(type) ? "static_cast<int>(sizeof(" + type + "))"
                                : "1";
    const bool hasProfile = !profileName.empty();
    const bool spatialOwned2D = ownerWindowRowsExpr == "spatial-2d";
    const bool spatialWindow2D = ownerWindowRowsExpr == "spatial-2d-window";
    const bool spatial2D = spatialOwned2D || spatialWindow2D;
    const std::string localRowBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_row_begin"
                  : "ctx.__or_halo_layout.global_row_begin";
    const std::string localColBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_col_begin" : "0";
    const std::string ownerRowBeginExpr =
        spatialOwned2D ? "ctx.__or_spatial_layout.owned_rows.begin"
                       : localRowBeginExpr;
    const std::string ownerColBeginExpr =
        spatialOwned2D ? "ctx.__or_spatial_layout.owned_cols.begin"
                       : localColBeginExpr;
    const std::string ownerColCountExpr =
        spatialOwned2D ? "ctx.__or_spatial_layout.owned_cols.count"
                       : spatialWindow2D
                             ? "ctx.__or_spatial_layout.local_col_count"
                             : logicalColsExpr;
    if (hasProfile) {
        code += "    auto dacpp_profile_bounded_sync_start_" +
                param.calcParamName + " = dacpp::mpi::profileSegmentStart();\n";
    }
    code += "    {\n";
    code += "        std::vector<" + type + "> __or_bounded_values_" +
            param.calcParamName + "(static_cast<std::size_t>(" +
            std::to_string(param.postUseSync.boundedIndices.size()) + "), " +
            type + "{});\n";
    std::vector<bool> directRootConstants(
        param.postUseSync.boundedIndices.size(), false);
    for (std::size_t idx = 0; idx < param.postUseSync.boundedIndices.size();
         ++idx) {
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        if (bounded.indices.size() != 2) {
            continue;
        }
        std::string constantValue;
        if (constantBoundaryValueForIndex(plan, param, boundaryUpdates, bounded,
                                          constantValue)) {
            directRootConstants[idx] = true;
            code += "        if (" + mpiRankExpr + " == 0) {\n";
            code += "            __or_bounded_values_" + param.calcParamName +
                    "[static_cast<std::size_t>(" + std::to_string(idx) +
                    ")] = static_cast<" + type + ">(" + constantValue + ");\n";
            code += "        }\n";
            continue;
        }
        code += "        {\n";
        code += "            const int64_t __or_target_row = " +
                std::to_string(bounded.indices[0]) + ";\n";
        code += "            const int64_t __or_target_col = " +
                std::to_string(bounded.indices[1]) + ";\n";
        code += "            if (__or_target_row >= " + ownerRowBeginExpr +
                " && __or_target_row < " + ownerRowBeginExpr + " + " +
                rowCountExpr + " && __or_target_col >= " +
                ownerColBeginExpr + " && __or_target_col < " +
                ownerColBeginExpr + " + " + ownerColCountExpr + ") {\n";
        code += "                const int64_t __or_local_row = __or_target_row + " +
                rowOffsetExpr + " - " + localRowBeginExpr + ";\n";
        code += "                const int64_t __or_local_col = __or_target_col + " +
                colOffsetExpr + " - " + localColBeginExpr + ";\n";
        code += "                const int64_t __or_local_linear = __or_local_row * " +
                localStrideExpr + " + __or_local_col;\n";
        code += "                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= " +
                localBufferName + ".size()) {\n";
        code += "                    if (" + mpiRankExpr +
                " == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] resident halo bounded indexed read local index out of range\\n\");\n";
        code += "                    MPI_Abort(MPI_COMM_WORLD, 6);\n";
        code += "                }\n";
        code += "                __or_bounded_values_" + param.calcParamName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")] = " + localBufferName +
                "[static_cast<std::size_t>(__or_local_linear)];\n";
        code += "            }\n";
        code += "        }\n";
    }
    for (std::size_t idx = 0; idx < param.postUseSync.boundedIndices.size();
         ++idx) {
        if (directRootConstants[idx]) {
            continue;
        }
        code += "        {\n";
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        code += "            const int64_t __or_target_row = " +
                std::to_string(bounded.indices[0]) + ";\n";
        code += "            const int64_t __or_target_col = " +
                std::to_string(bounded.indices[1]) + ";\n";
        code += "            int __or_owner_rank = -1;\n";
        code += "            if (__or_target_col >= 0 && __or_target_col < " +
                logicalColsExpr + ") {\n";
        if (spatialOwned2D) {
            code += "                __or_owner_rank = dacpp::mpi::operator_resident::spatial_2d_owner_rank(__or_target_row, __or_target_col, " +
                    ownerRowTotalExpr + ", " + logicalColsExpr + ", " +
                    mpiSizeExpr + ");\n";
        } else if (spatialWindow2D) {
            code += "                for (int __or_owner_candidate = 0; __or_owner_candidate < " +
                    mpiSizeExpr + "; ++__or_owner_candidate) {\n";
            code += "                    const auto __or_owner_layout = dacpp::mpi::operator_resident::resident_halo_2d_spatial_layout(ctx.__or_output_rows, ctx.__or_output_cols, ctx.__or_input_rows, ctx.__or_input_cols, __or_owner_candidate, " +
                    mpiSizeExpr + ", ctx.__or_spatial_halo_width);\n";
            code += "                    if (__or_target_row >= __or_owner_layout.global_row_begin && __or_target_row < __or_owner_layout.global_row_begin + __or_owner_layout.local_row_count && __or_target_col >= __or_owner_layout.global_col_begin && __or_target_col < __or_owner_layout.global_col_begin + __or_owner_layout.local_col_count) {\n";
            code += "                        __or_owner_rank = __or_owner_candidate;\n";
            code += "                        break;\n";
            code += "                    }\n";
            code += "                }\n";
        } else {
            code += "                for (int __or_owner_candidate = 0; __or_owner_candidate < " +
                    mpiSizeExpr + "; ++__or_owner_candidate) {\n";
            if (ownerWindowRowsExpr.empty()) {
            code += "                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(" +
                    ownerRowTotalExpr + ", __or_owner_candidate, " +
                    mpiSizeExpr + ");\n";
            code += "                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {\n";
            code += "                        __or_owner_rank = __or_owner_candidate;\n";
            code += "                        break;\n";
            code += "                    }\n";
            } else {
            code += "                    const auto __or_owner_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout(" +
                    ownerRowTotalExpr + ", " + logicalColsExpr +
                    ", __or_owner_candidate, " + mpiSizeExpr + ", " +
                    ownerWindowRowsExpr + ");\n";
            code += "                    if (__or_target_row >= __or_owner_layout.global_row_begin && __or_target_row < __or_owner_layout.global_row_begin + __or_owner_layout.local_row_count) {\n";
            code += "                        __or_owner_rank = __or_owner_candidate;\n";
            code += "                    }\n";
            }
            code += "                }\n";
        }
        code += "            }\n";
        code += "            if (__or_owner_rank < 0) {\n";
        code += "                if (" + mpiRankExpr +
                " == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] resident halo bounded indexed read has no owner\\n\");\n";
        code += "                MPI_Abort(MPI_COMM_WORLD, 6);\n";
        code += "            }\n";
        code += "            if (" + mpiRankExpr +
                " == __or_owner_rank && __or_owner_rank != 0) {\n";
        code += "                MPI_Send(&__or_bounded_values_" + param.calcParamName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")], " + valueMpiCount + ", " + mpiType + ", 0, " +
                std::to_string(tagBase + static_cast<int>(idx)) +
                ", MPI_COMM_WORLD);\n";
        code += "            } else if (" + mpiRankExpr +
                " == 0 && __or_owner_rank != 0) {\n";
        code += "                MPI_Recv(&__or_bounded_values_" + param.calcParamName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")], " + valueMpiCount + ", " + mpiType + ", __or_owner_rank, " +
                std::to_string(tagBase + static_cast<int>(idx)) +
                ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "            }\n";
        code += "        }\n";
    }
    code += "        if (" + mpiRankExpr + " == 0) {\n";
    emitBoundedReviseValues(code, param,
                            "__or_bounded_values_" + param.calcParamName,
                            type,
                            "            ");
    code += "        }\n";
    code += "    }\n";
    if (hasProfile) {
        code += "    dacpp::mpi::recordProfileSegment(" + profileName +
                ", dacpp::mpi::ProfileSegment::FinalSync, "
                "dacpp_profile_bounded_sync_start_" +
                param.calcParamName + ");\n";
    }
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
    code += "    int64_t __or_local_output_cols = 0;\n";
    code += "    int64_t __or_output_row_begin = 0;\n";
    code += "    int64_t __or_output_col_begin = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_output_row_range{};\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::SegmentedProfile __or_profile;\n";
    code += "    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();\n";
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
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
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
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    if (plan.orLoopLower.hoistReaderSync) {
        code += "    auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
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
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
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
        code += "    auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
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
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
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
        code += "    auto dacpp_profile_bcast_start_" +
                directReader->calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
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
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_" +
                directReader->calcParamName + ");\n";
    }
    code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
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
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
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
        "__or_materialized_output_items", "ctx.__or_profile");
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
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    (void)ctx;\n";
    for (const auto& param : plan.params) {
        code += "    (void)" + paramVarName(param) + ";\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
            "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
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
    const std::string& localColBeginExpr,
    const std::string& localColCountExpr,
    const std::string& localStrideExpr,
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
        code += "            if (__or_boundary_target_col < " +
                localColBeginExpr + " || __or_boundary_target_col >= " +
                localColBeginExpr + " + " + localColCountExpr +
                ") continue;\n";
        code += "            const int64_t __or_boundary_target_local = (__or_boundary_target_row - " +
                localRowBeginExpr + ") * " + localStrideExpr +
                " + (__or_boundary_target_col - " + localColBeginExpr +
                ");\n";
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
            code += "            if (__or_boundary_source_col < " +
                    localColBeginExpr + " || __or_boundary_source_col >= " +
                    localColBeginExpr + " + " + localColCountExpr +
                    ") continue;\n";
            code += "            const int64_t __or_boundary_source_local = (__or_boundary_source_row - " +
                    localRowBeginExpr + ") * " + localStrideExpr +
                    " + (__or_boundary_source_col - " + localColBeginExpr +
                    ");\n";
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
    code += "    int64_t __or_local_output_cols = 0;\n";
    code += "    int64_t __or_output_row_begin = 0;\n";
    code += "    int64_t __or_output_col_begin = 0;\n";
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
    if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
        code += "    int __or_temporal_block_size = " +
                std::to_string(
                    plan.orLoopLower.stencilResidentHalo.temporalBlockSize) +
                ";\n";
    }
    if (plan.orLoopLower.stencilResidentHalo.spatial2DEnabled) {
        code += "    int __or_spatial_halo_width = " +
                std::to_string(
                    plan.orLoopLower.stencilResidentHalo.spatial2DHaloWidth) +
                ";\n";
    }
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_output_row_range{};\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_output_col_range{};\n";
    code += "    dacpp::mpi::operator_resident::ResidentHalo2DRowLayout __or_halo_layout{};\n";
    code += "    dacpp::mpi::operator_resident::ResidentHalo2DSpatialLayout __or_spatial_layout{};\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::SegmentedProfile __or_profile;\n";
    code += "    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();\n";
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
    const bool spatial2D =
        plan.orLoopLower.stencilResidentHalo.spatial2DEnabled;
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
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
    if (spatial2D) {
        code += "    ctx.__or_spatial_layout = dacpp::mpi::operator_resident::resident_halo_2d_spatial_layout(ctx.__or_output_rows, ctx.__or_output_cols, ctx.__or_input_rows, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_spatial_halo_width);\n";
        code += "    ctx.__or_output_row_range = ctx.__or_spatial_layout.owned_rows;\n";
        code += "    ctx.__or_output_col_range = ctx.__or_spatial_layout.owned_cols;\n";
        code += "    ctx.__or_local_output_rows = ctx.__or_output_row_range.count;\n";
        code += "    ctx.__or_local_output_cols = ctx.__or_output_col_range.count;\n";
        code += "    ctx.__or_output_row_begin = ctx.__or_output_row_range.begin;\n";
        code += "    ctx.__or_output_col_begin = ctx.__or_output_col_range.begin;\n";
    } else {
        code += "    ctx.__or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_rows, ctx.mpi_rank, ctx.mpi_size);\n";
        code += "    ctx.__or_output_col_range = {0, ctx.__or_output_cols};\n";
        code += "    ctx.__or_local_output_rows = ctx.__or_output_row_range.count;\n";
        code += "    ctx.__or_local_output_cols = ctx.__or_output_cols;\n";
        code += "    ctx.__or_output_row_begin = ctx.__or_output_row_range.begin;\n";
        code += "    ctx.__or_output_col_begin = 0;\n";
    }
    code += "    ctx.__or_local_item_count = " +
            checkedMulExpr(
                "ctx.__or_local_output_rows", "ctx.__or_local_output_cols",
                "[DACPP][MPI][OR] resident halo 2D local item count overflow") +
            ";\n";
    if (spatial2D) {
        code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout(0, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_window_rows);\n";
    } else if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
        code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout_temporal(ctx.__or_output_rows, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_temporal_block_size);\n";
    } else {
        code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout(ctx.__or_output_rows, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_window_rows);\n";
    }
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
            spatial2D ? "ctx.__or_spatial_layout.local_size"
                      : "ctx.__or_halo_layout.local_size",
            spatial2D
                ? "[DACPP][MPI][OR] spatial resident halo 2D local reader size exceeds MPI int range"
                : "[DACPP][MPI][OR] resident halo 2D local reader size exceeds MPI int range") +
            ";\n";
    code += "    " + checkedMpiCountExpr(
            "ctx.__or_local_item_count",
            "[DACPP][MPI][OR] resident halo 2D local writer size exceeds MPI int range") +
            ";\n";
    code += "    const int64_t __or_local_reader_size = " +
            std::string(spatial2D ? "ctx.__or_spatial_layout.local_size"
                                  : "ctx.__or_halo_layout.local_size") +
            ";\n";
    code += "    ctx." + localName(reader) +
            ".assign(static_cast<std::size_t>(__or_local_reader_size), " +
            readerType + "{});\n";
    if (directReader) {
        code += "    ctx." + localName(*directReader) +
                ".assign(static_cast<std::size_t>(__or_local_reader_size), " +
                directReaderType + "{});\n";
    }
    code += "    ctx." + localName(writer) +
            ".assign(static_cast<std::size_t>(__or_local_reader_size), " +
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
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    code += "    auto dacpp_profile_scatter_start = dacpp::mpi::profileSegmentStart();\n";
    if (reader.constantInit.supported) {
        code += "    std::fill(ctx." + localName(reader) + ".begin(), ctx." +
                localName(reader) + ".end(), " +
                reader.constantInit.valueExpr + ");\n";
        code += "    // Constant-initialized resident halo reader " +
                reader.calcParamName +
                " is filled locally; skip root tensor2Array/scatter_window_2d_rows.\n";
    } else {
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) +
                ".tensor2Array(__or_initial_global_" + reader.calcParamName +
                ");\n";
        code += "    }\n";
        if (spatial2D) {
            code += "    dacpp::mpi::operator_resident::scatter_window_2d_spatial(__or_initial_global_" +
                    reader.calcParamName + ", ctx." + localName(reader) +
                    ", ctx.__or_output_rows, ctx.__or_output_cols, ctx.__or_input_rows, ctx.__or_input_cols, ctx.__or_spatial_halo_width, ctx.__or_spatial_layout, ctx.mpi_rank, ctx.mpi_size, " +
                    readerMpiType + ");\n";
        } else if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
            code += "    dacpp::mpi::operator_resident::scatter_window_2d_rows_temporal(__or_initial_global_" +
                    reader.calcParamName + ", ctx." + localName(reader) +
                    ", ctx.__or_output_rows, ctx.__or_input_cols, ctx.__or_temporal_block_size, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
                    readerMpiType + ");\n";
        } else {
            code += "    dacpp::mpi::operator_resident::scatter_window_2d_rows(__or_initial_global_" +
                    reader.calcParamName + ", ctx." + localName(reader) +
                    ", ctx.__or_output_rows, ctx.__or_input_cols, ctx.__or_window_rows, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
                    readerMpiType + ");\n";
        }
    }
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
        if (directReader->constantInit.supported) {
            if (spatial2D ||
                plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
                code += "    std::fill(ctx." + localName(*directReader) +
                        ".begin(), ctx." + localName(*directReader) +
                        ".end(), " + directReader->constantInit.valueExpr +
                        ");\n";
            } else {
                code += "    std::fill(ctx." + localName(*directReader) +
                        ".begin(), ctx." + localName(*directReader) +
                        ".end(), " + directReaderType + "{});\n";
                code += "    std::vector<" + directReaderType +
                        "> __or_initial_owned_" + directReader->calcParamName +
                        "(static_cast<std::size_t>(ctx.__or_local_item_count), " +
                        directReader->constantInit.valueExpr + ");\n";
                code += "    dacpp::mpi::operator_resident::write_owned_slice_2d_rows(ctx." +
                        localName(*directReader) + ", __or_initial_owned_" +
                        directReader->calcParamName +
                        ", ctx.__or_halo_layout, ctx.__or_output_cols, ctx.__or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset);\n";
            }
            code += "    // Constant-initialized resident direct reader " +
                    directReader->calcParamName +
                    " is filled locally; skip root tensor2Array/MPI_Scatterv.\n";
        } else {
            code += "    std::vector<" + directReaderType + "> __or_initial_global_" +
                    directReader->calcParamName + ";\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + paramVarName(*directReader) +
                    ".tensor2Array(__or_initial_global_" +
                    directReader->calcParamName + ");\n";
            code += "    }\n";
            if (spatial2D ||
                plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
                code += "    std::vector<" + directReaderType +
                        "> __or_initial_global_window_" +
                        directReader->calcParamName + ";\n";
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        __or_initial_global_window_" +
                        directReader->calcParamName +
                        ".assign(static_cast<std::size_t>(__or_initial_reader_dense_count), " +
                        directReaderType + "{});\n";
                code += "        for (int64_t __or_direct_row = 0; __or_direct_row < ctx.__or_output_rows; ++__or_direct_row) {\n";
                code += "            for (int64_t __or_direct_col = 0; __or_direct_col < ctx.__or_output_cols; ++__or_direct_col) {\n";
                code += "                const int64_t __or_direct_src = __or_direct_row * ctx.__or_output_cols + __or_direct_col;\n";
                code += "                const int64_t __or_direct_dst = (__or_direct_row + ctx.__or_followup_row_offset) * ctx.__or_input_cols + __or_direct_col + ctx.__or_followup_col_offset;\n";
                code += "                if (__or_direct_src >= 0 && __or_direct_dst >= 0 && static_cast<std::size_t>(__or_direct_src) < __or_initial_global_" +
                        directReader->calcParamName +
                        ".size() && static_cast<std::size_t>(__or_direct_dst) < __or_initial_global_window_" +
                        directReader->calcParamName + ".size()) {\n";
                code += "                    __or_initial_global_window_" +
                        directReader->calcParamName +
                        "[static_cast<std::size_t>(__or_direct_dst)] = __or_initial_global_" +
                        directReader->calcParamName +
                        "[static_cast<std::size_t>(__or_direct_src)];\n";
                code += "                }\n";
                code += "            }\n";
                code += "        }\n";
                code += "    }\n";
                if (spatial2D) {
                    code += "    dacpp::mpi::operator_resident::scatter_window_2d_spatial(__or_initial_global_window_" +
                            directReader->calcParamName + ", ctx." +
                            localName(*directReader) +
                            ", ctx.__or_output_rows, ctx.__or_output_cols, ctx.__or_input_rows, ctx.__or_input_cols, ctx.__or_spatial_halo_width, ctx.__or_spatial_layout, ctx.mpi_rank, ctx.mpi_size, " +
                            directReaderMpiType + ");\n";
                } else {
                    code += "    dacpp::mpi::operator_resident::scatter_window_2d_rows_temporal(__or_initial_global_window_" +
                            directReader->calcParamName + ", ctx." +
                            localName(*directReader) +
                            ", ctx.__or_output_rows, ctx.__or_input_cols, ctx.__or_temporal_block_size, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
                            directReaderMpiType + ");\n";
                }
                code += "    // Direct-reader recurrence state " +
                        directReader->calcParamName +
                        " is embedded in the widened resident halo layout for k=2 replay.\n";
            } else {
            code += "    std::vector<" + directReaderType + "> __or_initial_owned_" +
                    directReader->calcParamName +
                    "(static_cast<std::size_t>(ctx.__or_local_item_count), " +
                    directReaderType + "{});\n";
            code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_initial_global_" +
                    directReader->calcParamName +
                    ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                    directReaderMpiType + ", __or_initial_owned_" +
                    directReader->calcParamName +
                    ".data(), " + checkedMpiCountExpr(
                        "ctx.__or_local_item_count",
                        "[DACPP][MPI][OR] resident halo direct reader scatter count exceeds MPI int range") +
                    ", " + directReaderMpiType +
                    ", 0, MPI_COMM_WORLD);\n";
            code += "    dacpp::mpi::operator_resident::write_owned_slice_2d_rows(ctx." +
                    localName(*directReader) + ", __or_initial_owned_" +
                    directReader->calcParamName +
                    ", ctx.__or_halo_layout, ctx.__or_output_cols, ctx.__or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset);\n";
            }
        }
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start);\n";
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
    const int temporalBlockSize =
        plan.orLoopLower.stencilResidentHalo.temporalBlockSize;
    const bool temporalBlocked = temporalBlockSize > 1;
    const bool spatial2D =
        plan.orLoopLower.stencilResidentHalo.spatial2DEnabled;
    const std::string localReaderRowsExpr =
        spatial2D ? "ctx.__or_spatial_layout.local_row_count"
                  : "ctx.__or_halo_layout.local_row_count";
    const std::string localReaderColsExpr =
        spatial2D ? "ctx.__or_spatial_layout.local_col_count"
                  : "__or_input_cols";
    const std::string localReaderRowBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_row_begin"
                  : "ctx.__or_halo_layout.global_row_begin";
    const std::string localReaderColBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_col_begin" : "0";
    const std::string ownedRowOffsetExpr =
        spatial2D ? "ctx.__or_spatial_layout.owned_row_offset"
                  : "ctx.__or_halo_layout.owned_row_offset";
    const std::string ownedColOffsetExpr =
        spatial2D ? "ctx.__or_spatial_layout.owned_col_offset" : "0";
    const std::string readerRowOffsetExpr =
        spatial2D ? "(" + ownedRowOffsetExpr + " - __or_writer_row_offset)"
                  : "0";
    const std::string readerColOffsetExpr =
        spatial2D ? "(" + ownedColOffsetExpr + " - __or_writer_col_offset)"
                  : "0";
    const std::string writerColOffsetExpr =
        spatial2D ? ownedColOffsetExpr : "__or_writer_col_offset";
    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto& q = ctx.q;\n";
    code += "    const int64_t __or_input_rows = ctx.__or_input_rows;\n";
    code += "    const int64_t __or_input_cols = ctx.__or_input_cols;\n";
    code += "    const int64_t __or_output_rows = ctx.__or_output_rows;\n";
    code += "    const int64_t __or_output_cols = ctx.__or_output_cols;\n";
    code += "    const int64_t __or_local_output_rows = ctx.__or_local_output_rows;\n";
    code += "    const int64_t __or_local_output_cols = ctx.__or_local_output_cols;\n";
    code += "    const int64_t __or_local_item_count = ctx.__or_local_item_count;\n";
    code += "    const int64_t __or_local_reader_rows = " + localReaderRowsExpr + ";\n";
    code += "    const int64_t __or_local_reader_cols = " + localReaderColsExpr + ";\n";
    code += "    const int64_t __or_local_reader_row_begin = " + localReaderRowBeginExpr + ";\n";
    code += "    const int64_t __or_local_reader_col_begin = " + localReaderColBeginExpr + ";\n";
    code += "    const int __or_writer_row_offset = ctx.__or_followup_row_offset;\n";
    code += "    const int __or_writer_col_offset = ctx.__or_followup_col_offset;\n";
    code += "    int64_t __or_kernel_item_count = __or_local_item_count;\n";
    if (temporalBlocked && !spatial2D) {
        code += "    const int __or_temporal_block_size = ctx.__or_temporal_block_size;\n";
        code += "    const bool __or_temporal_block_safe = __or_output_rows >= static_cast<int64_t>(ctx.mpi_size) * static_cast<int64_t>(__or_temporal_block_size);\n";
        code += "    const int __or_runtime_temporal_block_size = __or_temporal_block_safe ? __or_temporal_block_size : 1;\n";
    }
    code += "    auto& " + localName(reader) + " = ctx." +
            localName(reader) + ";\n";
    if (directReader) {
        code += "    auto& " + localName(*directReader) + " = ctx." +
                localName(*directReader) + ";\n";
    }
    code += "    auto& " + localName(writer) + " = ctx." +
            localName(writer) + ";\n";
    if (temporalBlocked) {
        code += "    int64_t __or_time_step = 0;\n";
        code += "    const int64_t __or_time_limit = static_cast<int64_t>(" +
                plan.orLoopLower.stencilResidentHalo.temporalLoopLimitExpr +
                ");\n";
        code += "    const int64_t __or_time_end = __or_time_limit" +
                std::string(plan.orLoopLower.stencilResidentHalo
                                .temporalLoopLimitInclusive
                                ? " + 1"
                                : "") +
                ";\n";
        code += "    while (__or_time_step < __or_time_end) {\n";
        if (spatial2D) {
            code += "    const int __or_inner_steps = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(ctx.__or_temporal_block_size), __or_time_end - __or_time_step));\n";
            code += "    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
            code += "    dacpp::mpi::operator_resident::exchange_halo_2d_spatial_inplace(" +
                    localName(reader) +
                    ", ctx.__or_spatial_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, ctx.__or_spatial_halo_width, mpi_rank, ctx.mpi_size, " +
                    writerMpiType + ");\n";
            code += "    if (__or_time_step > 0) {\n";
            emitResidentHaloBoundaryRefresh2D(
                code, plan, sitePlan.boundaryLocalUpdates, reader,
                localName(reader), "__or_local_reader_row_begin",
                "__or_local_reader_rows", "__or_local_reader_col_begin",
                "__or_local_reader_cols", "__or_local_reader_cols",
                "__or_input_rows", "__or_input_cols");
            code += "    }\n";
            code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
        } else {
            code += "    const int __or_inner_steps = static_cast<int>(std::min<int64_t>(__or_runtime_temporal_block_size, __or_time_end - __or_time_step));\n";
            code += "    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
            code += "    dacpp::mpi::operator_resident::exchange_halo_2d_rows_temporal_inplace(" +
                    localName(reader) +
                    ", ctx.__or_halo_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, __or_temporal_block_size, mpi_rank, ctx.mpi_size, " +
                    writerMpiType + ");\n";
            code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
        }
    }
    auto emitOneKernel = [&](const std::string& indent,
                             const std::string& readerBufferName,
                             const std::string& writerBufferName,
                             const std::string& directReaderBufferName,
                             const std::string& readerRowOffsetExpr,
                             const std::string& writerRowOffsetExpr,
                             bool recordKernelProfile) {
        if (recordKernelProfile) {
            code += indent +
                    "auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
        }
        code += indent + "if (__or_kernel_item_count > 0) {\n";
        code += indent + "    sycl::buffer<" + readerType +
                ", 1> __or_reader_buf(" + readerBufferName +
                ".data(), sycl::range<1>(" + readerBufferName +
                ".size()));\n";
    if (directReader) {
            code += indent + "    sycl::buffer<" + directReaderType +
                    ", 1> __or_direct_reader_buf(" +
                    directReaderBufferName + ".data(), sycl::range<1>(" +
                    directReaderBufferName + ".size()));\n";
    }
        code += indent + "    sycl::buffer<" + writerType +
                ", 1> __or_writer_buf(" + writerBufferName +
                ".data(), sycl::range<1>(" + writerBufferName +
                ".size()));\n";
        code += indent + "    q.submit([&](sycl::handler& h) {\n";
        code += indent + "        auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    if (directReader) {
            code += indent + "        auto __or_direct_reader_acc = __or_direct_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    }
        code += indent + "        auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
        code += indent + "        h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_kernel_item_count)), [=](sycl::id<1> idx) {\n";
        code += indent + "            const int item_linear = static_cast<int>(idx[0]);\n";
        code += indent + "            const int local_row = item_linear / static_cast<int>(__or_local_output_cols);\n";
        code += indent + "            const int local_col = item_linear % static_cast<int>(__or_local_output_cols);\n";
        code += indent + "            auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    if (directReader) {
            code += indent + "            auto* __or_direct_reader_data = __or_direct_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
        code += indent + "            auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
                code += indent + "            dacpp::mpi::ResidentHaloView2D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, static_cast<int>((local_row + " +
                    readerRowOffsetExpr +
                    ") * static_cast<int>(__or_local_reader_cols) + local_col + static_cast<int>(" +
                    readerColOffsetExpr + ")), static_cast<int>(__or_local_reader_cols)};\n";
            continue;
        }
        if (directReader &&
            param.paramIndex == directReader->paramIndex) {
                code += indent + "            dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_direct_reader_data, static_cast<int>((local_row + " +
                    writerRowOffsetExpr +
                    ") * static_cast<int>(__or_local_reader_cols) + local_col + static_cast<int>(" +
                    writerColOffsetExpr + "))};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
                code += indent + "            dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, static_cast<int>((local_row + " +
                    writerRowOffsetExpr +
                    ") * static_cast<int>(__or_local_reader_cols) + local_col + static_cast<int>(" +
                    writerColOffsetExpr + "))};\n";
            continue;
        }
    }
        code += indent + "            " + calcName + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
        code += ");\n";
        code += indent + "        });\n";
        code += indent + "    });\n";
        code += indent + "    q.wait();\n";
        code += indent + "}\n";
        if (recordKernelProfile) {
            code += indent +
                    "dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
        }
    };
    auto emitSpatialKernel = [&](const std::string& indent,
                                 const std::string& readerBufferName,
                                 const std::string& writerBufferName,
                                 const std::string& directReaderBufferName) {
        code += indent + "if (__or_kernel_item_count > 0) {\n";
        code += indent + "    sycl::buffer<" + readerType +
                ", 1> __or_reader_buf(" + readerBufferName +
                ".data(), sycl::range<1>(" + readerBufferName +
                ".size()));\n";
        if (directReader) {
            code += indent + "    sycl::buffer<" + directReaderType +
                    ", 1> __or_direct_reader_buf(" +
                    directReaderBufferName + ".data(), sycl::range<1>(" +
                    directReaderBufferName + ".size()));\n";
        }
        code += indent + "    sycl::buffer<" + writerType +
                ", 1> __or_writer_buf(" + writerBufferName +
                ".data(), sycl::range<1>(" + writerBufferName +
                ".size()));\n";
        code += indent + "    q.submit([&](sycl::handler& h) {\n";
        code += indent + "        auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
        if (directReader) {
            code += indent + "        auto __or_direct_reader_acc = __or_direct_reader_buf.get_access<sycl::access::mode::read>(h);\n";
        }
        code += indent + "        auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
        code += indent + "        h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_kernel_item_count)), [=](sycl::id<1> idx) {\n";
        code += indent + "            const int item_linear = static_cast<int>(idx[0]);\n";
        code += indent + "            const int local_row = static_cast<int>(__or_compute_row_begin) + item_linear / static_cast<int>(__or_compute_cols);\n";
        code += indent + "            const int local_col = static_cast<int>(__or_compute_col_begin) + item_linear % static_cast<int>(__or_compute_cols);\n";
        code += indent + "            auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (directReader) {
            code += indent + "            auto* __or_direct_reader_data = __or_direct_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        }
        code += indent + "            auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        for (const auto& param : plan.params) {
            const std::string paramType = elemType(plan, param);
            if (param.access == ParamAccessKind::StencilWindow) {
                code += indent + "            dacpp::mpi::ResidentHaloView2D<const " +
                        paramType + "> view_" + param.calcParamName +
                        "{__or_reader_data, static_cast<int>((local_row - 1) * static_cast<int>(__or_local_reader_cols) + local_col - 1), static_cast<int>(__or_local_reader_cols)};\n";
                continue;
            }
            if (directReader &&
                param.paramIndex == directReader->paramIndex) {
                code += indent + "            dacpp::mpi::ContiguousView1D<const " +
                        paramType + "> view_" + param.calcParamName +
                        "{__or_direct_reader_data, static_cast<int>(local_row * static_cast<int>(__or_local_reader_cols) + local_col)};\n";
                continue;
            }
            if (param.access == ParamAccessKind::OutputDirect &&
                param.writes &&
                !param.reads) {
                code += indent + "            dacpp::mpi::ContiguousView1D<" +
                        paramType + "> view_" + param.calcParamName +
                        "{__or_writer_data, static_cast<int>(local_row * static_cast<int>(__or_local_reader_cols) + local_col)};\n";
                continue;
            }
        }
        code += indent + "            " + calcName + "_mpi_local(";
        for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
             ++paramIdx) {
            if (paramIdx != 0) {
                code += ", ";
            }
            code += "view_" +
                    plan.exprNode.calc->getParam(paramIdx)->getName();
        }
        code += ");\n";
        code += indent + "        });\n";
        code += indent + "    });\n";
        code += indent + "    q.wait();\n";
        code += indent + "}\n";
    };
    if (temporalBlocked && spatial2D) {
        code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    for (int __or_block_step = 0; __or_block_step < __or_inner_steps; ++__or_block_step) {\n";
        code += "        const int64_t __or_spatial_remaining_halo = __or_inner_steps - __or_block_step - 1;\n";
        code += "        const int64_t __or_compute_row_begin = std::max<int64_t>(1, ctx.__or_spatial_layout.owned_row_offset - __or_spatial_remaining_halo);\n";
        code += "        const int64_t __or_compute_row_end = std::min<int64_t>(__or_local_reader_rows - 1, ctx.__or_spatial_layout.owned_row_offset + __or_local_output_rows + __or_spatial_remaining_halo);\n";
        code += "        const int64_t __or_compute_col_begin = std::max<int64_t>(1, ctx.__or_spatial_layout.owned_col_offset - __or_spatial_remaining_halo);\n";
        code += "        const int64_t __or_compute_col_end = std::min<int64_t>(__or_local_reader_cols - 1, ctx.__or_spatial_layout.owned_col_offset + __or_local_output_cols + __or_spatial_remaining_halo);\n";
        code += "        const int64_t __or_compute_rows = std::max<int64_t>(0, __or_compute_row_end - __or_compute_row_begin);\n";
        code += "        const int64_t __or_compute_cols = std::max<int64_t>(0, __or_compute_col_end - __or_compute_col_begin);\n";
        code += "        __or_kernel_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_compute_rows, __or_compute_cols, \"[DACPP][MPI][OR] spatial temporal resident halo 2D kernel item count overflow\");\n";
        emitSpatialKernel("        ", localName(reader), localName(writer),
                          directReader ? localName(*directReader) : "");
        if (directReader) {
            code += "        std::swap(" + localName(*directReader) + ", " +
                    localName(reader) + ");\n";
        }
        code += "        std::swap(" + localName(reader) + ", " +
                localName(writer) + ");\n";
        emitResidentHaloBoundaryRefresh2D(
            code, plan, sitePlan.boundaryLocalUpdates, reader,
            localName(reader), "__or_local_reader_row_begin",
            "__or_local_reader_rows", "__or_local_reader_col_begin",
            "__or_local_reader_cols", "__or_local_reader_cols",
            "__or_input_rows", "__or_input_cols");
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
        code += "    __or_time_step += __or_inner_steps;\n";
        code += "    }\n";
        code += "    // P11 spatial temporal-block=2 leaves the latest state in the resident reader buffer after spatial role rotation.\n";
        code += "}\n";
        return;
    }
    if (temporalBlocked) {
        code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    for (int __or_block_step = 0; __or_block_step < __or_inner_steps; ++__or_block_step) {\n";
        code += "        const int64_t __or_compute_row_begin = std::max<int64_t>(1, ctx.__or_halo_layout.owned_row_offset + 1 - (__or_inner_steps - __or_block_step - 1));\n";
        code += "        const int64_t __or_compute_row_end = std::min<int64_t>(__or_local_reader_rows - 1, ctx.__or_halo_layout.owned_row_offset + 1 + __or_local_output_rows + (__or_inner_steps - __or_block_step - 1));\n";
        code += "        const int64_t __or_compute_rows = std::max<int64_t>(0, __or_compute_row_end - __or_compute_row_begin);\n";
        code += "        __or_kernel_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_compute_rows, __or_local_output_cols, \"[DACPP][MPI][OR] temporal resident halo 2D kernel item count overflow\");\n";
        emitOneKernel("        ", localName(reader), localName(writer),
                      directReader ? localName(*directReader) : "",
                      "__or_compute_row_begin - 1",
                      "__or_compute_row_begin",
                      false);
        if (directReader) {
            code += "        std::swap(" + localName(*directReader) + ", " +
                    localName(reader) + ");\n";
        }
        code += "        std::swap(" + localName(reader) + ", " +
                localName(writer) + ");\n";
        emitResidentHaloBoundaryRefresh2D(
            code, plan, sitePlan.boundaryLocalUpdates, reader,
            localName(reader), "__or_local_reader_row_begin",
            "__or_local_reader_rows", "__or_local_reader_col_begin",
            "__or_local_reader_cols", "__or_local_reader_cols",
            "__or_input_rows", "__or_input_cols");
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
        code += "    __or_time_step += __or_inner_steps;\n";
        code += "    }\n";
        code += "    // P4.6 temporal-block=2 leaves the latest state in the resident reader buffer after the block loop.\n";
        code += "}\n";
        return;
    }
    code += "    __or_kernel_item_count = __or_local_item_count;\n";
    emitOneKernel("    ", localName(reader), localName(writer),
                  directReader ? localName(*directReader) : "",
                  readerRowOffsetExpr,
                  spatial2D ? ownedRowOffsetExpr : "__or_writer_row_offset",
                  true);
    code += "    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
    if (spatial2D) {
        code += "    dacpp::mpi::operator_resident::exchange_halo_2d_spatial_inplace(" +
                localName(writer) +
                ", ctx.__or_spatial_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, ctx.__or_spatial_halo_width, mpi_rank, ctx.mpi_size, " +
                writerMpiType + ");\n";
    } else {
        code += "    dacpp::mpi::operator_resident::exchange_halo_2d_rows_inplace(" +
                localName(writer) +
                ", ctx.__or_halo_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, mpi_rank, ctx.mpi_size, " +
                writerMpiType + ");\n";
    }
    emitResidentHaloBoundaryRefresh2D(
        code, plan, sitePlan.boundaryLocalUpdates, reader, localName(writer),
        "__or_local_reader_row_begin", "__or_local_reader_rows",
        "__or_local_reader_col_begin", "__or_local_reader_cols",
        "__or_local_reader_cols", "__or_input_rows", "__or_input_cols");
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
    if (directReader) {
        code += "    ctx." + localName(*directReader) + ".swap(ctx." +
                localName(reader) + ");\n";
    }
    code += "    ctx." + localName(reader) + ".swap(ctx." +
            localName(writer) + ");\n";
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
    const bool writerFull = hasFullPostUse(writer);
    const bool writerBounded = hasBoundedPostUse(writer);
    const bool directReaderFull = directReader && hasFullPostUse(*directReader);
    const bool directReaderBounded =
        directReader && hasBoundedPostUse(*directReader);
    const bool readerFull = hasFullPostUse(reader);
    const bool readerBounded = hasBoundedPostUse(reader);
    const bool needsWriterGlobal = writerFull || readerFull;
    const bool temporalBlocked = halo.temporalBlockSize > 1;
    const bool spatial2D = halo.spatial2DEnabled;
    const std::string ownedSliceFn =
        spatial2D
            ? "owned_slice_2d_spatial"
            : temporalBlocked
            ? "owned_slice_2d_rows_temporal"
            : "owned_slice_2d_rows";
    const std::string layoutExpr =
        spatial2D ? "ctx.__or_spatial_layout" : "ctx.__or_halo_layout";
    const std::string writerRowOffset =
        temporalBlocked
            ? std::to_string(halo.followupTargetRowOffset)
            : std::to_string(halo.followupTargetRowOffset);
    const std::string writerColOffset =
        std::to_string(halo.followupTargetColOffset);
    const std::string ownedRowCountExpr =
        spatial2D ? "ctx.__or_spatial_layout.owned_rows.count"
                  : temporalBlocked ? "ctx.__or_halo_layout.local_row_count"
                                    : "ctx.__or_output_row_range.count";
    const std::string localReaderRowCountExpr =
        spatial2D ? "ctx.__or_spatial_layout.local_row_count"
                  : "ctx.__or_halo_layout.local_row_count";
    const std::string localReaderColCountExpr =
        spatial2D ? "ctx.__or_spatial_layout.local_col_count"
                  : "ctx.__or_input_cols";
    const std::string localReaderRowBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_row_begin"
                  : "ctx.__or_halo_layout.global_row_begin";
    const std::string localReaderColBeginExpr =
        spatial2D ? "ctx.__or_spatial_layout.global_col_begin" : "0";
    const std::string readerBoundedWindowRows =
        temporalBlocked ? "ctx.__or_temporal_block_size"
                        : "ctx.__or_window_rows";
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
    if (needsWriterGlobal) {
        code += "    auto dacpp_profile_gather_start_writer = dacpp::mpi::profileSegmentStart();\n";
        code += "    const auto __or_owned_" + writer.calcParamName +
                " = dacpp::mpi::operator_resident::" + ownedSliceFn + "(ctx." +
                localName(reader) +
                ", " + layoutExpr + ", " +
                (spatial2D ? "ctx.__or_output_rows, ctx.__or_output_cols, "
                           : "ctx.__or_output_cols, ctx.__or_input_cols, ") +
                writerRowOffset + ", " + writerColOffset + ");\n";
    }
    code += "    " + checkedMpiCountExpr(
            "__or_materialized_writer_count",
            "[DACPP][MPI][OR] resident halo 2D materialized output size exceeds MPI int range") +
            ";\n";
    if (needsWriterGlobal && spatial2D) {
    code += "    dacpp::mpi::operator_resident::gather_spatial_owned_to_root(__or_owned_" +
            writer.calcParamName +
            ", __or_materialized_" + writer.calcParamName +
            ", ctx.__or_output_rows, ctx.__or_output_cols, mpi_rank, ctx.mpi_size, " +
            writerMpiType + ");\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_writer);\n";
    } else if (needsWriterGlobal) {
    code += "    if (mpi_rank == 0) {\n";
    code += "        __or_materialized_" + writer.calcParamName +
            ".resize(static_cast<std::size_t>(__or_materialized_writer_count));\n";
    code += "    }\n";
    code += "    MPI_Gatherv(__or_owned_" + writer.calcParamName +
            ".data(), " +
            checkedMpiCountExpr(
                "ctx.__or_local_item_count",
                "[DACPP][MPI][OR] resident halo 2D materialize sendcount exceeds MPI int range") +
            ", " +
            writerMpiType + ", mpi_rank == 0 ? __or_materialized_" +
            writer.calcParamName +
            ".data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
            writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_writer);\n";
    }
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    if (writerFull) {
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(writer) + ".array2Tensor(__or_materialized_" +
                writer.calcParamName + ");\n";
        code += "    }\n";
    } else if (writerBounded) {
        emitResidentHaloBoundedIndexedRootRead2D(
            code, plan, writer, "ctx." + localName(reader),
            ownedRowCountExpr, "ctx.__or_output_rows",
            spatial2D ? "spatial-2d" : "",
            "ctx.__or_output_cols", "ctx.__or_input_cols",
            writerRowOffset,
            writerColOffset, "mpi_rank",
            "ctx.mpi_size", "", sitePlan.boundaryLocalUpdates, 4710);
    } else {
        code += "    // No host post-use for " + writer.calcParamName +
                "; skip full resident halo writer materialization.\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    if (directReader && directReaderFull) {
        code += "    auto dacpp_profile_gather_start_" +
                directReader->calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "    const auto __or_owned_" + directReader->calcParamName +
                " = dacpp::mpi::operator_resident::" + ownedSliceFn + "(ctx." +
                localName(*directReader) +
                ", " + layoutExpr + ", " +
                (spatial2D ? "ctx.__or_output_rows, ctx.__or_output_cols, "
                           : "ctx.__or_output_cols, ctx.__or_input_cols, ") +
                writerRowOffset + ", " + writerColOffset + ");\n";
        code += "    std::vector<" + directReaderType + "> __or_materialized_" +
                directReader->calcParamName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        __or_materialized_" + directReader->calcParamName +
                ".resize(static_cast<std::size_t>(__or_materialized_writer_count));\n";
        code += "    }\n";
        if (spatial2D) {
        code += "    dacpp::mpi::operator_resident::gather_spatial_owned_to_root(__or_owned_" +
                directReader->calcParamName +
                ", __or_materialized_" + directReader->calcParamName +
                ", ctx.__or_output_rows, ctx.__or_output_cols, mpi_rank, ctx.mpi_size, " +
                directReaderMpiType + ");\n";
        } else {
        code += "    MPI_Gatherv(__or_owned_" + directReader->calcParamName +
                ".data(), " + checkedMpiCountExpr(
                    "ctx.__or_local_item_count",
                    "[DACPP][MPI][OR] resident halo 2D direct-reader materialize sendcount exceeds MPI int range") +
                ", " + directReaderMpiType +
                ", mpi_rank == 0 ? __or_materialized_" +
                directReader->calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                directReaderMpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_" +
                directReader->calcParamName + ");\n";
        code += "    dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(*directReader) +
                ".array2Tensor(__or_materialized_" +
                directReader->calcParamName + ");\n";
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    } else if (directReader && directReaderBounded) {
        emitResidentHaloBoundedIndexedRootRead2D(
            code, plan, *directReader, "ctx." + localName(*directReader),
            ownedRowCountExpr, "ctx.__or_output_rows",
            spatial2D ? "spatial-2d" : "",
            "ctx.__or_output_cols", "ctx.__or_input_cols",
            writerRowOffset,
            writerColOffset, "mpi_rank",
            "ctx.mpi_size", "ctx.__or_profile",
            sitePlan.boundaryLocalUpdates, 4810);
    } else if (directReader) {
        code += "    // No host post-use for " + directReader->calcParamName +
                "; skip full resident halo direct-reader materialization.\n";
    }
    code += "    dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    if (readerFull) {
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
    } else if (readerBounded) {
        emitResidentHaloBoundedIndexedRootRead2D(
            code, plan, reader, "ctx." + localName(reader),
            localReaderRowCountExpr, "ctx.__or_output_rows",
            spatial2D ? "spatial-2d-window" : readerBoundedWindowRows,
            "ctx.__or_input_cols",
            localReaderColCountExpr, "0", "0", "mpi_rank", "ctx.mpi_size",
            "", sitePlan.boundaryLocalUpdates, 4910);
    } else {
        code += "    // No host post-use for " + reader.calcParamName +
                "; skip full resident halo reader materialization.\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    for (const auto& param : plan.params) {
        if (param.paramIndex != reader.paramIndex &&
            param.paramIndex != writer.paramIndex) {
            code += "    (void)" + paramVarName(param) + ";\n";
        }
    }
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
            "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
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
    code += "    auto& q = dacpp::mpi::operator_resident::default_queue();\n";
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
