#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"
#include "llvm/ADT/StringSwitch.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

void emitOutputBuffer(std::string& code,
                      const ShellPartitionPlan& plan,
                      const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    code += "    std::vector<" + type + "> " + localName(param) +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
    if (param.outputInit.skipInitialSync) {
        code += "    std::fill(" + localName(param) + ".begin(), " +
                localName(param) + ".end(), " +
                (param.outputInit.valueExpr.empty() ? type + "{}"
                                                    : param.outputInit.valueExpr) +
                ");\n";
        code += "    // Output-direct no-read fast path for " +
                param.calcParamName +
                " initializes local output and skips root pack/scatter.\n";
    }
}

void emitAbortIfMpiCountTooLarge(std::string& code,
                                 const std::string& countExpr,
                                 const std::string& message) {
    code += "    if ((" + countExpr + ") > static_cast<int64_t>(2147483647)) {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"" + message +
            "\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 5);\n";
    code += "    }\n";
}

void emitAbortIfRuntimeCondition(std::string& code,
                                 const std::string& conditionExpr,
                                 const std::string& message,
                                 int abortCode) {
    code += "    if (" + conditionExpr + ") {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"" + message +
            "\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, " + std::to_string(abortCode) +
            ");\n";
    code += "    }\n";
}

bool isBuiltinIntegerType(const std::string& type) {
    return llvm::StringSwitch<bool>(type)
        .Cases("bool", "char", "signed char", "unsigned char", true)
        .Cases("short", "short int", "unsigned short", "unsigned short int", true)
        .Cases("int", "unsigned", "unsigned int", true)
        .Cases("long", "unsigned long", true)
        .Cases("long long", "unsigned long long", true)
        .Default(false);
}

std::string localLinearIndexExprForBoundedIndex(const ShellPartitionPlan& plan,
                                                const PostUseBoundedIndex& index) {
    if ((plan.signature.layout == LocalLayoutKind::Contiguous1D ||
         plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor) &&
        index.indices.size() == 1) {
        if (usesBlockCyclic1D(plan)) {
            return "dacpp::mpi::operator_resident::block_cyclic_local_index_1d(" +
                   std::to_string(index.indices[0]) +
                   ", mpi_size, __or_block_cyclic_block_size)";
        }
        return "(" + std::to_string(index.indices[0]) + " - __or_range.begin)";
    }
    if (plan.signature.layout == LocalLayoutKind::RowBlock2D &&
        index.indices.size() == 2) {
        return "((" + std::to_string(index.indices[0]) +
               " - __or_row_range.begin) * __or_cols + " +
               std::to_string(index.indices[1]) + ")";
    }
    if (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
        plan.signature.bindOrder.size() == 2 && index.indices.size() == 2) {
        return "((" + std::to_string(index.indices[0]) +
               " - __or_row_range.begin) * __or_cols + " +
               std::to_string(index.indices[1]) + ")";
    }
    return "";
}

std::string ownerPredicateForBoundedIndex(const ShellPartitionPlan& plan,
                                          const PostUseBoundedIndex& index) {
    if ((plan.signature.layout == LocalLayoutKind::Contiguous1D ||
         plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor) &&
        index.indices.size() == 1) {
        if (usesBlockCyclic1D(plan)) {
            return "(dacpp::mpi::operator_resident::block_cyclic_owner_1d(__or_target_index, mpi_size, __or_block_cyclic_block_size) == mpi_rank)";
        }
        return "(__or_target_index >= __or_range.begin && __or_target_index < __or_range.begin + __or_range.count)";
    }
    if (plan.signature.layout == LocalLayoutKind::RowBlock2D &&
        index.indices.size() == 2) {
        return "(__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)";
    }
    if (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
        plan.signature.bindOrder.size() == 2 && index.indices.size() == 2) {
        return "(__or_target_row >= __or_row_range.begin && __or_target_row < __or_row_range.begin + __or_row_range.count && __or_target_col >= 0 && __or_target_col < __or_cols)";
    }
    return "";
}

std::string ownerTotalExprForBoundedIndex(const ShellPartitionPlan& plan,
                                          const PostUseBoundedIndex& index) {
    if ((plan.signature.layout == LocalLayoutKind::Contiguous1D ||
         plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor) &&
        index.indices.size() == 1) {
        return "__or_total_items";
    }
    if (plan.signature.layout == LocalLayoutKind::RowBlock2D &&
        index.indices.size() == 2) {
        return "__or_rows";
    }
    if (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
        plan.signature.bindOrder.size() == 2 && index.indices.size() == 2) {
        return "__or_rows";
    }
    return "";
}

void emitBoundedIndexedRootReadSync(std::string& code,
                                    const ShellPartitionPlan& plan,
                                    const ParamAccessPlan& param,
                                    const std::string& localBufferName,
                                    const std::string& profileName) {
    if (param.postUseSync.boundedIndices.empty()) {
        return;
    }
    const std::string type = elemType(plan, param);
    const std::string mpiType = usesByteTransport(type) ? "MPI_BYTE"
                                                        : mpiDatatypeFor(type);
    const std::string valueMpiCount =
        usesByteTransport(type) ? "static_cast<int>(sizeof(" + type + "))"
                                : "1";
    const bool hasProfile = !profileName.empty();
    if (hasProfile) {
        code += "    auto dacpp_profile_bounded_sync_start_" +
                param.calcParamName + " = dacpp::mpi::profileSegmentStart();\n";
    }
    code += "    {\n";
    code += "        std::vector<" + type + "> __or_bounded_values_" +
            param.calcParamName + "(static_cast<std::size_t>(" +
            std::to_string(param.postUseSync.boundedIndices.size()) + "), " +
            type + "{});\n";
    for (std::size_t idx = 0; idx < param.postUseSync.boundedIndices.size();
         ++idx) {
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        const std::string localIndex =
            localLinearIndexExprForBoundedIndex(plan, bounded);
        const std::string ownerPredicate =
            ownerPredicateForBoundedIndex(plan, bounded);
        if (localIndex.empty() || ownerPredicate.empty()) {
            continue;
        }
        code += "        {\n";
        if (bounded.indices.size() == 1) {
            code += "            const int64_t __or_target_index = " +
                    std::to_string(bounded.indices[0]) + ";\n";
        } else {
            code += "            const int64_t __or_target_row = " +
                    std::to_string(bounded.indices[0]) + ";\n";
            code += "            const int64_t __or_target_col = " +
                    std::to_string(bounded.indices[1]) + ";\n";
        }
        code += "            if (" + ownerPredicate + ") {\n";
        code += "                const int64_t __or_local_linear = " +
                localIndex + ";\n";
        code += "                if (__or_local_linear < 0 || static_cast<std::size_t>(__or_local_linear) >= " +
                localBufferName + ".size()) {\n";
        code += "                    if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] bounded indexed read local index out of range\\n\");\n";
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
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        const std::string ownerTotal =
            ownerTotalExprForBoundedIndex(plan, bounded);
        if (ownerTotal.empty()) {
            continue;
        }
        code += "        {\n";
        if (bounded.indices.size() == 1) {
            code += "            const int64_t __or_target_index = " +
                    std::to_string(bounded.indices[0]) + ";\n";
            code += "            int __or_owner_rank = -1;\n";
            if (usesBlockCyclic1D(plan)) {
                code += "            if (__or_target_index >= 0 && __or_target_index < " +
                        ownerTotal + ") {\n";
                code += "                __or_owner_rank = dacpp::mpi::operator_resident::block_cyclic_owner_1d(__or_target_index, mpi_size, __or_block_cyclic_block_size);\n";
                code += "            }\n";
            } else {
                code += "            for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {\n";
                code += "                const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(" +
                        ownerTotal + ", __or_owner_candidate, mpi_size);\n";
                code += "                if (__or_target_index >= __or_owner_range.begin && __or_target_index < __or_owner_range.begin + __or_owner_range.count) {\n";
                code += "                    __or_owner_rank = __or_owner_candidate;\n";
                code += "                    break;\n";
                code += "                }\n";
                code += "            }\n";
            }
        } else {
            code += "            const int64_t __or_target_row = " +
                    std::to_string(bounded.indices[0]) + ";\n";
            code += "            const int64_t __or_target_col = " +
                    std::to_string(bounded.indices[1]) + ";\n";
            code += "            int __or_owner_rank = -1;\n";
            code += "            if (__or_target_col >= 0 && __or_target_col < __or_cols) {\n";
            code += "                for (int __or_owner_candidate = 0; __or_owner_candidate < mpi_size; ++__or_owner_candidate) {\n";
            code += "                    const auto __or_owner_range = dacpp::mpi::operator_resident::rank_range_1d(" +
                    ownerTotal + ", __or_owner_candidate, mpi_size);\n";
            code += "                    if (__or_target_row >= __or_owner_range.begin && __or_target_row < __or_owner_range.begin + __or_owner_range.count) {\n";
            code += "                        __or_owner_rank = __or_owner_candidate;\n";
            code += "                        break;\n";
            code += "                    }\n";
            code += "                }\n";
            code += "            }\n";
        }
        code += "            if (__or_owner_rank < 0) {\n";
        code += "                if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] bounded indexed read has no owner\\n\");\n";
        code += "                MPI_Abort(MPI_COMM_WORLD, 6);\n";
        code += "            }\n";
        code += "            if (mpi_rank == __or_owner_rank && __or_owner_rank != 0) {\n";
        code += "                MPI_Send(&__or_bounded_values_" + param.calcParamName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")], " + valueMpiCount + ", " + mpiType +
                ", 0, 4701, MPI_COMM_WORLD);\n";
        code += "            } else if (mpi_rank == 0 && __or_owner_rank != 0) {\n";
        code += "                MPI_Recv(&__or_bounded_values_" + param.calcParamName +
                "[static_cast<std::size_t>(" + std::to_string(idx) +
                ")], " + valueMpiCount + ", " + mpiType +
                ", __or_owner_rank, 4701, MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "            }\n";
        code += "        }\n";
    }
    code += "        if (mpi_rank == 0) {\n";
    for (std::size_t idx = 0; idx < param.postUseSync.boundedIndices.size();
         ++idx) {
        const auto& bounded = param.postUseSync.boundedIndices[idx];
        code += "            " + paramVarName(param) + ".reviseValue(__or_bounded_values_" +
                param.calcParamName + "[static_cast<std::size_t>(" +
                std::to_string(idx) + ")], std::vector<int>{";
        for (std::size_t dim = 0; dim < bounded.indices.size(); ++dim) {
            if (dim != 0) {
                code += ", ";
            }
            code += std::to_string(bounded.indices[dim]);
        }
        code += "});\n";
    }
    code += "        }\n";
    code += "    }\n";
    if (hasProfile) {
        code += "    dacpp::mpi::recordProfileSegment(" + profileName +
                ", dacpp::mpi::ProfileSegment::FinalSync, "
                "dacpp_profile_bounded_sync_start_" +
                param.calcParamName + ");\n";
    }
}

} // namespace

void emitScalarBroadcast(std::string& code,
                         const ShellPartitionPlan& plan,
                         const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string scalar = scalarName(param);
    const std::string local = localName(param);
    code += "    if (" + paramVarName(param) + ".getSize() != 1) {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] scalar parameter " +
            param.shellParamName + " expected size 1\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 2);\n";
    code += "    }\n";
    code += "    " + type + " " + scalar + "{};\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        std::vector<" + type + "> __or_scalar_vec_" +
            param.calcParamName + ";\n";
    code += "        " + paramVarName(param) + ".tensor2Array(__or_scalar_vec_" +
            param.calcParamName + ");\n";
    code += "        if (!__or_scalar_vec_" + param.calcParamName +
            ".empty()) " + scalar + " = __or_scalar_vec_" +
            param.calcParamName + "[0];\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        code += "    MPI_Bcast(&" + scalar + ", static_cast<int>(sizeof(" +
                type + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Bcast(&" + scalar + ", 1, " +
                mpiDatatypeFor(type) + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    std::vector<" + type + "> " + local + "(1, " + scalar +
            ");\n";
}

void emitReplicatedFullTensorBroadcast(std::string& code,
                                       const ShellPartitionPlan& plan,
                                       const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string local = localName(param);
    code += "    const int64_t __or_full_count_" + param.calcParamName +
            " = " + paramVarName(param) + ".getSize();\n";
    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_full_count_" +
            param.calcParamName + "));\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        auto dacpp_profile_pack_start_" + param.calcParamName +
            " = dacpp::mpi::profileSegmentStart();\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + local +
            ");\n";
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
            "dacpp::mpi::ProfileSegment::Pack, "
            "dacpp_profile_pack_start_" +
            param.calcParamName + ");\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        code += "    MPI_Bcast(" + local +
                ".data(), static_cast<int>(__or_full_count_" +
                param.calcParamName + " * sizeof(" + type +
                ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Bcast(" + local +
                ".data(), static_cast<int>(__or_full_count_" +
                param.calcParamName + "), " + mpiDatatypeFor(type) +
                ", 0, MPI_COMM_WORLD);\n";
    }
}

void emitRowPartitionFullRowScatter(std::string& code,
                                    const ShellPartitionPlan& plan,
                                    const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string local = localName(param);
    const std::string global = globalName(param);
    const std::string payloadLen = "__or_payload_len_" + param.calcParamName;

    const int voidDim = param.voidDims.empty() ? 0 : param.voidDims[0];
    const int indexDim = param.indexDim;
    const bool effectiveIndexedRowFullCols =
        param.payloadDirection == PayloadDirection::IndexedRowFullCols;
    const bool effectiveIndexedColFullRows =
        param.payloadDirection == PayloadDirection::IndexedColFullRows;
    const int effectiveIndexDim =
        effectiveIndexedRowFullCols ? 0 : (effectiveIndexedColFullRows ? 1 : indexDim);
    const int effectivePayloadDim =
        effectiveIndexedRowFullCols ? 1 : (effectiveIndexedColFullRows ? 0 : voidDim);
    int indexedBindPos = 0;
    if (!param.bindOrder.empty()) {
        for (std::size_t idx = 0; idx < plan.signature.bindOrder.size(); ++idx) {
            if (plan.signature.bindOrder[idx] == param.bindOrder[0]) {
                indexedBindPos = static_cast<int>(idx);
                break;
            }
        }
    }
    const bool broadcastIndexedPayload =
        plan.signature.bindOrder.size() == 2 && indexedBindPos != 0;
    const bool canUseDirectRowMajorFullRowPayload =
        !usesByte(plan, param) && !broadcastIndexedPayload &&
        effectiveIndexedRowFullCols;

    code += "    const int64_t " + payloadLen + " = " + paramVarName(param) +
            ".getShape(" + std::to_string(effectivePayloadDim) + ");\n";
    code += "    const int64_t __or_payload_index_extent_" +
            param.calcParamName + " = " + paramVarName(param) +
            ".getShape(" + std::to_string(effectiveIndexDim) + ");\n";
    std::string uniqueCount;
    std::string uniqueBegin;
    std::string countVar;
    std::string displVar;
    if (plan.signature.bindOrder.size() == 1) {
        uniqueCount = "__or_local_item_count";
        uniqueBegin = "__or_range.begin";
        countVar = "__or_counts";
        displVar = "__or_displs";
    } else if (indexedBindPos == 0) {
        uniqueCount = "__or_row_range.count";
        uniqueBegin = "__or_row_range.begin";
        countVar = "__or_row_counts";
        displVar = "__or_row_displs";
    } else {
        uniqueCount = "__or_cols";
        uniqueBegin = "0";
        countVar = "__or_full_col_counts_" + param.calcParamName;
        displVar = "__or_full_col_displs_" + param.calcParamName;
        code += "    std::vector<int> " + countVar + "(mpi_size, 0);\n";
        code += "    std::vector<int> " + displVar + "(mpi_size, 0);\n";
        code += "    " + countVar + "[0] = static_cast<int>(__or_cols);\n";
    }
    code += "    const int64_t __or_payload_unique_count_" +
            param.calcParamName + " = " + uniqueCount + ";\n";
    code += "    const int64_t __or_payload_index_begin_" +
            param.calcParamName + " = " + uniqueBegin + ";\n";
    emitAbortIfRuntimeCondition(
        code,
        paramVarName(param) + ".getDim() != 2 || " + payloadLen +
            " < 0 || __or_payload_unique_count_" + param.calcParamName +
            " < 0 || __or_payload_index_begin_" + param.calcParamName +
            " < 0 || (__or_payload_unique_count_" + param.calcParamName +
            " > 0 && __or_payload_index_begin_" + param.calcParamName +
            " + __or_payload_unique_count_" + param.calcParamName +
            " > __or_payload_index_extent_" + param.calcParamName + ")",
        "[DACPP][MPI][OR] RowPartitionFullRow input payload/index range out of bounds",
        5);
    emitAbortIfMpiCountTooLarge(
        code,
        "__or_payload_unique_count_" + param.calcParamName + " * " + payloadLen,
        "[DACPP][MPI][OR] RowPartitionFullRow input payload exceeds MPI int count");
    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_payload_unique_count_" +
            param.calcParamName + " * " +
            payloadLen + "));\n";
    if (param.constantInit.supported) {
        if (param.constantInit.indexExpr) {
            const std::string indexName =
                param.constantInit.globalIndexName.empty()
                    ? "__or_global_index"
                    : param.constantInit.globalIndexName;
            code += "    for (int64_t __or_payload_i = 0; __or_payload_i < __or_payload_unique_count_" +
                    param.calcParamName + "; ++__or_payload_i) {\n";
            code += "        const int64_t " + indexName +
                    " = __or_payload_index_begin_" + param.calcParamName +
                    " + __or_payload_i;\n";
            code += "        for (int64_t __or_payload_j = 0; __or_payload_j < " +
                    payloadLen + "; ++__or_payload_j) {\n";
            code += "            " + local +
                    "[static_cast<std::size_t>(__or_payload_i * " +
                    payloadLen + " + __or_payload_j)] = " +
                    param.constantInit.valueExpr + ";\n";
            code += "        }\n";
            code += "    }\n";
            code += "    // Index-generated RowPartitionFullRow input " +
                    param.calcParamName +
                    " is filled locally; skip root pack/scatter.\n";
        } else {
            code += "    std::fill(" + local + ".begin(), " + local +
                    ".end(), " + param.constantInit.valueExpr + ");\n";
            code += "    // Constant-initialized RowPartitionFullRow input " +
                    param.calcParamName +
                    " is filled locally; skip root pack/scatter.\n";
        }
        return;
    }
    code += "    std::vector<int> __or_payload_counts_" + param.calcParamName +
            "(mpi_size);\n";
    code += "    std::vector<int> __or_payload_displs_" + param.calcParamName +
            "(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_payload_counts_" + param.calcParamName +
            "[r] = static_cast<int>(static_cast<int64_t>(" + countVar +
            "[r]) * " + payloadLen + ");\n";
    code += "        __or_payload_displs_" + param.calcParamName +
            "[r] = static_cast<int>(static_cast<int64_t>(" + displVar +
            "[r]) * " + payloadLen + ");\n";
    code += "    }\n";
    code += "    std::vector<" + type + "> __or_sendbuf_" +
            param.calcParamName + ";\n";
    if (canUseDirectRowMajorFullRowPayload) {
        code += "    " + type + "* __or_payload_send_ptr_" +
                param.calcParamName + " = nullptr;\n";
        code += "    bool __or_payload_direct_" + param.calcParamName +
                " = false;\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        const int64_t __or_input_rows_" +
                param.calcParamName + " = " + paramVarName(param) +
                ".getShape(0);\n";
        code += "        const int64_t __or_input_cols_" +
                param.calcParamName + " = " + paramVarName(param) +
                ".getShape(1);\n";
        code += "        __or_payload_direct_" + param.calcParamName +
                " = (" + paramVarName(param) + ".getDim() == 2 && " +
                paramVarName(param) + ".getOffset() >= 0 && " +
                paramVarName(param) + ".getStride(1) == 1 && " +
                paramVarName(param) + ".getStride(0) == __or_input_cols_" +
                param.calcParamName + " && " + payloadLen +
                " == __or_input_cols_" + param.calcParamName +
                " && __or_input_rows_" + param.calcParamName +
                " >= " + displVar + "[mpi_size - 1] + " + countVar +
                "[mpi_size - 1]);\n";
        code += "        if (__or_payload_direct_" + param.calcParamName +
                ") {\n";
        code += "            __or_payload_send_ptr_" + param.calcParamName +
                " = " + paramVarName(param) + ".getDataPtr().get() + " +
                paramVarName(param) + ".getOffset();\n";
        code += "        }\n";
        code += "    }\n";
    }
    code += "    if (mpi_rank == 0) {\n";
    if (canUseDirectRowMajorFullRowPayload) {
        code += "        if (!__or_payload_direct_" + param.calcParamName +
                ") {\n";
    }
    code += "        auto dacpp_profile_pack_start_" + param.calcParamName +
            " = dacpp::mpi::profileSegmentStart();\n";
    code += "        std::vector<" + type + "> " + global + ";\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + global +
            ");\n";
    code += "        __or_sendbuf_" + param.calcParamName +
            ".resize(static_cast<std::size_t>(";
    if (plan.signature.bindOrder.size() == 1) {
        code += "__or_total_items";
    } else if (indexedBindPos == 0) {
        code += "__or_rows";
    } else {
        code += "__or_cols";
    }
    code += " * " +
            payloadLen + "));\n";
    code += "        const int64_t __or_input_cols_" + param.calcParamName +
            " = " + paramVarName(param) + ".getShape(1);\n";
    code += "        const int64_t __or_input_rows_" + param.calcParamName +
            " = " + paramVarName(param) + ".getShape(0);\n";
    code += "        if ((" + std::string(effectiveIndexedRowFullCols ? "true" : "false") +
            " && " + payloadLen + " > __or_input_cols_" +
            param.calcParamName + ") || (" +
            std::string(effectiveIndexedColFullRows ? "true" : "false") + " && " +
            payloadLen + " > __or_input_rows_" + param.calcParamName +
            ")) {\n";
    code += "            std::fprintf(stderr, \"[DACPP][MPI][OR] RowPartitionFullRow payload length exceeds input dimension\\n\");\n";
    code += "            MPI_Abort(MPI_COMM_WORLD, 5);\n";
    code += "        }\n";
    code += "        for (int r = 0; r < mpi_size; ++r) {\n";
    code += "            for (int64_t local_i = 0; local_i < " + countVar +
            "[r]; ++local_i) {\n";
    code += "                const int64_t indexed_value = static_cast<int64_t>(" +
            displVar + "[r]) + local_i;\n";
    code += "                const int64_t dst_base = (__or_payload_displs_" +
            param.calcParamName + "[r] + local_i * " +
            payloadLen + ");\n";
    code += "                for (int64_t payload_i = 0; payload_i < " +
            payloadLen + "; ++payload_i) {\n";
    if (effectiveIndexedRowFullCols) {
        code += "                    const int64_t src_linear = indexed_value * __or_input_cols_" +
                param.calcParamName + " + payload_i;\n";
    } else {
        code += "                    const int64_t src_linear = payload_i * __or_input_cols_" +
                param.calcParamName + " + indexed_value;\n";
    }
    code += "                    __or_sendbuf_" + param.calcParamName +
            "[static_cast<std::size_t>(dst_base + payload_i)] = " + global +
            "[static_cast<std::size_t>(src_linear)];\n";
    code += "                }\n";
    code += "            }\n";
    code += "        }\n";
    if (canUseDirectRowMajorFullRowPayload) {
        code += "        __or_payload_send_ptr_" + param.calcParamName +
                " = __or_sendbuf_" + param.calcParamName + ".data();\n";
    }
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
            "dacpp::mpi::ProfileSegment::Pack, "
            "dacpp_profile_pack_start_" +
            param.calcParamName + ");\n";
    if (canUseDirectRowMajorFullRowPayload) {
        code += "        }\n";
    }
    code += "    }\n";
    if (broadcastIndexedPayload) {
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + local + ".swap(__or_sendbuf_" +
                param.calcParamName + ");\n";
        code += "    }\n";
        if (usesByte(plan, param)) {
            emitAbortIfMpiCountTooLarge(
                code,
                "__or_payload_unique_count_" + param.calcParamName + " * " +
                    payloadLen + " * static_cast<int64_t>(sizeof(" + type +
                    "))",
                "[DACPP][MPI][OR] RowPartitionFullRow input byte payload exceeds MPI int count");
            code += "    MPI_Bcast(" + local +
                    ".data(), static_cast<int>(__or_payload_unique_count_" +
                    param.calcParamName + " * " + payloadLen +
                    " * sizeof(" + type +
                    ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(" + local +
                    ".data(), static_cast<int>(__or_payload_unique_count_" +
                    param.calcParamName + " * " + payloadLen + "), " +
                    mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
    } else if (usesByte(plan, param)) {
        emitAbortIfMpiCountTooLarge(
            code,
            "__or_payload_unique_count_" + param.calcParamName + " * " +
                payloadLen + " * static_cast<int64_t>(sizeof(" + type + "))",
            "[DACPP][MPI][OR] RowPartitionFullRow input byte payload exceeds MPI int count");
        code += "    std::vector<int> __or_payload_counts_bytes_" +
                param.calcParamName + ";\n";
        code += "    std::vector<int> __or_payload_displs_bytes_" +
                param.calcParamName + ";\n";
        code += "    dacpp::mpi::operator_resident::byte_counts_displs(__or_payload_counts_" +
                param.calcParamName + ", __or_payload_displs_" +
                param.calcParamName + ", sizeof(" + type +
                "), __or_payload_counts_bytes_" + param.calcParamName +
                ", __or_payload_displs_bytes_" + param.calcParamName +
                ");\n";
        code += "    auto dacpp_profile_scatter_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "    MPI_Scatterv(mpi_rank == 0 ? __or_sendbuf_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_counts_bytes_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_displs_bytes_" +
                param.calcParamName + ".data() : nullptr, MPI_BYTE, " +
                local + ".data(), static_cast<int>(__or_payload_unique_count_" +
                param.calcParamName + " * " + payloadLen + " * sizeof(" + type +
                ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Scatter, "
                "dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
    } else {
        code += "    auto dacpp_profile_scatter_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        const std::string sendData =
            canUseDirectRowMajorFullRowPayload
                ? "__or_payload_send_ptr_" + param.calcParamName
                : "__or_sendbuf_" + param.calcParamName + ".data()";
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + sendData +
                " : nullptr, mpi_rank == 0 ? __or_payload_counts_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_displs_" +
                param.calcParamName + ".data() : nullptr, " + mpiType +
                ", " + local +
                ".data(), static_cast<int>(__or_payload_unique_count_" +
                param.calcParamName + " * " +
                payloadLen + "), " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Scatter, "
                "dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
    }
}

void emitParamLocalStorage(std::string& code,
                           const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            emitScalarBroadcast(code, plan, param);
        } else if (param.access == ParamAccessKind::ReplicatedFullTensor) {
            emitReplicatedFullTensorBroadcast(code, plan, param);
        } else if (param.access == ParamAccessKind::RowPartitionFullRow) {
            emitRowPartitionFullRowScatter(code, plan, param);
        } else if (param.writes && param.reads &&
                   !param.outputInit.skipInitialSync) {
            emitResidentOrScatter(code, plan, param);
        } else if (param.writes) {
            emitOutputBuffer(code, plan, param);
        } else {
            emitResidentOrScatter(code, plan, param);
        }
    }
}

void emitResidencyAndMaterialization(std::string& code,
                                     const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
            continue;
        }
        const std::string type = elemType(plan, param);
        if (param.postUseReductionCountEqOne && isBuiltinIntegerType(type)) {
            code += "    int64_t __or_local_reduction_count_" +
                    param.calcParamName + " = 0;\n";
            code += "    for (const auto& __or_value : " + localName(param) +
                    ") {\n";
            code += "        if (__or_value == static_cast<" + type +
                    ">(1)) {\n";
            code += "            ++__or_local_reduction_count_" +
                    param.calcParamName + ";\n";
            code += "        }\n";
            code += "    }\n";
            code += "    int64_t __or_global_reduction_count_" +
                    param.calcParamName + " = 0;\n";
            code += "    auto dacpp_profile_reduce_start_" +
                    param.calcParamName +
                    " = dacpp::mpi::profileSegmentStart();\n";
            code += "    MPI_Reduce(&__or_local_reduction_count_" +
                    param.calcParamName + ", &__or_global_reduction_count_" +
                    param.calcParamName +
                    ", 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);\n";
            code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::FinalSync, dacpp_profile_reduce_start_" +
                    param.calcParamName + ");\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        " + param.postUseReductionScalarName +
                    " = static_cast<decltype(" +
                    param.postUseReductionScalarName +
                    ")>(__or_global_reduction_count_" +
                    param.calcParamName + ");\n";
            code += "    }\n";
            code += "    // Post-use reduction for " + param.calcParamName +
                    " replaces full output materialization.\n";
            if (!param.retainResidentAfterWrite) {
                continue;
            }
        }
        if (param.postUseSync.kind == PostUseSyncKind::BoundedIndexedRootRead) {
            emitBoundedIndexedRootReadSync(code, plan, param, localName(param),
                                           "dacpp_profile");
            if (!param.retainResidentAfterWrite) {
                continue;
            }
        } else if (param.postUseSync.kind == PostUseSyncKind::None &&
                   !param.retainResidentAfterWrite) {
            code += "    // No host post-use for " + param.calcParamName +
                    "; skip full output materialization.\n";
            continue;
        }
        if (param.materializeAfterWrite) {
            emitGatherMaterialize(code, plan, param, "dacpp_profile");
        }
        if (param.materializeAfterWrite && !param.retainResidentAfterWrite) {
            code += "    // No downstream resident reader for " +
                    param.calcParamName +
                    "; host materialization above preserves visibility.\n";
            continue;
        }
        if (param.reads) {
            code += "    auto& __or_resident_out_" + param.calcParamName +
                    " = dacpp::mpi::operator_resident::ensure_resident<" +
                    type + ">(" + paramVarName(param) + ", " +
                    localName(param) + ".size());\n";
            code += "    __or_resident_out_" + param.calcParamName + " = " +
                    localName(param) + ";\n";
        } else {
            code += "    auto& __or_resident_out_" + param.calcParamName +
                    " = dacpp::mpi::operator_resident::replace_resident<" +
                    type + ">(" + paramVarName(param) + ", std::move(" +
                    localName(param) + "));\n";
            code += "    (void)__or_resident_out_" + param.calcParamName +
                    ";\n";
        }
    }
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
