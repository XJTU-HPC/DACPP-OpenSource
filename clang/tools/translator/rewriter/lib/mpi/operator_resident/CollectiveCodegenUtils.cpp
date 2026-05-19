#include <cstring>

#include "OperatorResidentCodegen_Internal.h"
#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

std::string checkedMpiCountExpr(const std::string& expr,
                                const std::string& message) {
    return "dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(" +
           expr + "), \"" + message + "\")";
}

std::string checkedMpiPayloadCountExpr(const std::string& elemCountExpr,
                                       const std::string& type,
                                       const std::string& message) {
    if (usesByteTransport(type)) {
        return checkedMpiCountExpr(
            "dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(" +
                elemCountExpr + "), static_cast<int64_t>(sizeof(" + type +
                ")), \"" + message + "\")",
            message);
    }
    return checkedMpiCountExpr(elemCountExpr, message);
}

}  // namespace

bool usesByte(const ShellPartitionPlan& plan, const ParamAccessPlan& param) {
    return usesByteTransport(elemType(plan, param));
}

void emitByteCounts(std::string& code,
                    const std::string& suffix,
                    const std::string& type) {
    code += "    std::vector<int> __or_counts_bytes_" + suffix + ";\n";
    code += "    std::vector<int> __or_displs_bytes_" + suffix + ";\n";
    code += "    dacpp::mpi::operator_resident::byte_counts_displs(__or_counts, __or_displs, sizeof(" +
            type + "), __or_counts_bytes_" + suffix + ", __or_displs_bytes_" +
            suffix + ");\n";
}

void emitScatter(std::string& code,
                 const ShellPartitionPlan& plan,
                 const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string local = localName(param);
    const std::string global = globalName(param);
    const bool canUseDirectTensorScatter =
        !usesByte(plan, param) &&
        (plan.signature.layout == LocalLayoutKind::Contiguous1D ||
         (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
          plan.signature.bindOrder.size() == 1)) &&
        param.tensorDims.size() == 1 && param.tensorDims[0] == 0;
    const bool canUseRowBlockDirectScatter =
        plan.signature.layout == LocalLayoutKind::RowBlock2D &&
        param.tensorDims.size() == 2 && param.tensorDims[0] == 0 &&
        param.tensorDims[1] == 1;

    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
    if (param.constantInit.supported) {
        code += "    std::fill(" + local + ".begin(), " + local +
                ".end(), " + param.constantInit.valueExpr + ");\n";
        code += "    // Constant-initialized input " + param.calcParamName +
                " is filled locally; skip root pack/scatter.\n";
        return;
    }
    code += "    std::vector<" + type + "> " + global + ";\n";
    if (canUseDirectTensorScatter) {
        code += "    " + type + "* __or_scatter_src_" +
                param.calcParamName + " = nullptr;\n";
        code += "    bool __or_scatter_direct_" + param.calcParamName +
                " = false;\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        const int64_t __or_direct_offset_" +
                param.calcParamName + " = " + paramVarName(param) +
                ".getOffset();\n";
        code += "        const int64_t __or_direct_stride_" +
                param.calcParamName + " = " + paramVarName(param) +
                ".getStride(0);\n";
        code += "        __or_scatter_direct_" + param.calcParamName +
                " = (__or_direct_offset_" + param.calcParamName +
                " >= 0 && __or_direct_stride_" + param.calcParamName +
                " == 1 && " + paramVarName(param) +
                ".getSize() >= __or_total_items && __or_direct_offset_" +
                param.calcParamName + " + __or_total_items <= " +
                paramVarName(param) + ".getSize());\n";
        code += "        if (__or_scatter_direct_" + param.calcParamName +
                ") {\n";
        code += "            __or_scatter_src_" + param.calcParamName +
                " = " + paramVarName(param) +
                ".getDataPtr().get() + __or_direct_offset_" +
                param.calcParamName + ";\n";
        code += "        }\n";
        code += "    }\n";
    }
    if (canUseRowBlockDirectScatter) {
        code += "    " + type + "* __or_rowblock_scatter_src_" +
                param.calcParamName + " = nullptr;\n";
        code += "    int __or_rowblock_direct_" + param.calcParamName +
                " = 0;\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        const int64_t __or_rb_offset_" +
                param.calcParamName + " = " + paramVarName(param) +
                ".getOffset();\n";
        code += "        int64_t __or_rb_rows_" + param.calcParamName +
                " = -1;\n";
        code += "        int64_t __or_rb_cols_" + param.calcParamName +
                " = -1;\n";
        code += "        if (" + paramVarName(param) +
                ".getDim() == 2) {\n";
        code += "            __or_rb_rows_" + param.calcParamName +
                " = " + paramVarName(param) + ".getShape(0);\n";
        code += "            __or_rb_cols_" + param.calcParamName +
                " = " + paramVarName(param) + ".getShape(1);\n";
        code += "        }\n";
        code += "        const int64_t __or_rb_size_" + param.calcParamName +
                " = " + paramVarName(param) + ".getSize();\n";
        code += "        const int64_t __or_rb_last_begin_" +
                param.calcParamName +
                " = mpi_size > 0 ? static_cast<int64_t>(__or_row_displs[mpi_size - 1]) : 0;\n";
        code += "        const int64_t __or_rb_last_count_" +
                param.calcParamName +
                " = mpi_size > 0 ? static_cast<int64_t>(__or_row_counts[mpi_size - 1]) : 0;\n";
        code += "        bool __or_rb_ranges_ok_" + param.calcParamName +
                " = mpi_size >= 0;\n";
        code += "        for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {\n";
        code += "            const int64_t __or_r_begin = static_cast<int64_t>(__or_row_displs[__or_r]);\n";
        code += "            const int64_t __or_r_count = static_cast<int64_t>(__or_row_counts[__or_r]);\n";
        code += "            if (__or_r_begin < 0 || __or_r_count < 0 || __or_r_begin > __or_rows || __or_r_count > __or_rows - __or_r_begin) {\n";
        code += "                __or_rb_ranges_ok_" + param.calcParamName +
                " = false;\n";
        code += "                break;\n";
        code += "            }\n";
        code += "        }\n";
        code += "        __or_rowblock_direct_" + param.calcParamName +
                " = (" + paramVarName(param) + ".getDim() == 2 && "
                "__or_rb_offset_" + param.calcParamName + " >= 0 && "
                "__or_rb_rows_" + param.calcParamName + " == __or_rows && "
                "__or_rb_cols_" + param.calcParamName + " == __or_cols && "
                "__or_rb_rows_" + param.calcParamName + " >= 0 && "
                "__or_rb_cols_" + param.calcParamName + " >= 0 && " +
                paramVarName(param) + ".getStride(1) == 1 && " +
                paramVarName(param) + ".getStride(0) == __or_cols && "
                "__or_rb_size_" + param.calcParamName +
                " >= 0 && __or_rb_offset_" + param.calcParamName +
                " <= __or_rb_size_" + param.calcParamName +
                " && __or_total_items <= __or_rb_size_" +
                param.calcParamName + " - __or_rb_offset_" +
                param.calcParamName + " && __or_rb_last_begin_" +
                param.calcParamName + " >= 0 && __or_rb_last_count_" +
                param.calcParamName + " >= 0 && __or_rb_last_begin_" +
                param.calcParamName + " + __or_rb_last_count_" +
                param.calcParamName + " <= __or_rows && "
                "__or_rb_ranges_ok_" + param.calcParamName +
                ") ? 1 : 0;\n";
        code += "        if (__or_rowblock_direct_" +
                param.calcParamName + ") {\n";
        code += "            __or_rowblock_scatter_src_" + param.calcParamName +
                " = " + paramVarName(param) +
                ".getDataPtr().get() + __or_rb_offset_" +
                param.calcParamName + ";\n";
        code += "        }\n";
        code += "    }\n";
        code += "    MPI_Bcast(&__or_rowblock_direct_" +
                param.calcParamName +
                ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
    }
    code += "    if (mpi_rank == 0) {\n";
    if (canUseDirectTensorScatter || canUseRowBlockDirectScatter) {
        code += "        if (";
        if (canUseDirectTensorScatter) {
            code += "!__or_scatter_direct_" + param.calcParamName;
        }
        if (canUseDirectTensorScatter && canUseRowBlockDirectScatter) {
            code += " && ";
        }
        if (canUseRowBlockDirectScatter) {
            code += "!__or_rowblock_direct_" + param.calcParamName;
        }
        code += ") {\n";
    }
    code += "        auto dacpp_profile_pack_start_" + param.calcParamName +
            " = dacpp::mpi::profileSegmentStart();\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + global +
            ");\n";
    if (canUseDirectTensorScatter) {
        code += "        __or_scatter_src_" + param.calcParamName +
                " = " + global + ".data();\n";
    }
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
            "dacpp::mpi::ProfileSegment::Pack, "
            "dacpp_profile_pack_start_" +
            param.calcParamName + ");\n";
    if (canUseDirectTensorScatter || canUseRowBlockDirectScatter) {
        code += "        }\n";
    }
    code += "    }\n";
    if (canUseRowBlockDirectScatter) {
        if (usesByte(plan, param)) {
            emitByteCounts(code, param.calcParamName, type);
        }
        code += "    if (__or_rowblock_direct_" + param.calcParamName +
                ") {\n";
        code += "        auto dacpp_profile_scatter_start_" +
                param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        if (usesByte(plan, param)) {
            code += "        MPI_Scatterv(mpi_rank == 0 ? __or_rowblock_scatter_src_" +
                    param.calcParamName +
                    " : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                    param.calcParamName +
                    ".data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                    param.calcParamName +
                    ".data() : nullptr, MPI_BYTE, " + local +
                    ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                    type + ")), \"[DACPP][MPI][OR] row-block direct scatter byte count exceeds MPI int range\"), \"[DACPP][MPI][OR] row-block direct scatter byte count exceeds MPI int range\"), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "        MPI_Scatterv(mpi_rank == 0 ? __or_rowblock_scatter_src_" +
                    param.calcParamName +
                    " : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                    mpiType + ", " + local +
                    ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), \"[DACPP][MPI][OR] row-block direct scatter count exceeds MPI int range\"), " +
                    mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Scatter, "
                "dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
        code += "    } else {\n";
        if (usesByte(plan, param)) {
            code += "        auto dacpp_profile_scatter_start_" +
                    param.calcParamName +
                    " = dacpp::mpi::profileSegmentStart();\n";
            code += "        MPI_Scatterv(mpi_rank == 0 ? " + global +
                    ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                    param.calcParamName +
                    ".data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                    param.calcParamName +
                    ".data() : nullptr, MPI_BYTE, " + local +
                    ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                    type + ")), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
            code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
                    "dacpp::mpi::ProfileSegment::Scatter, "
                    "dacpp_profile_scatter_start_" +
                    param.calcParamName + ");\n";
        } else {
            code += "        if (mpi_rank != 0) {\n";
            code += "            " + global +
                    ".resize(static_cast<std::size_t>(__or_total_items));\n";
            code += "        }\n";
            code += "        auto dacpp_profile_bcast_start_" +
                    param.calcParamName +
                    " = dacpp::mpi::profileSegmentStart();\n";
            code += "        MPI_Bcast(" + global + ".data(), " +
                    checkedMpiPayloadCountExpr(
                        "__or_total_items", type,
                        "[DACPP][MPI][OR] row-block broadcast count exceeds MPI int range") +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, "
                    "dacpp::mpi::ProfileSegment::Bcast, "
                    "dacpp_profile_bcast_start_" +
                    param.calcParamName + ");\n";
            code += "        const int64_t __or_rowblock_offset_" +
                    param.calcParamName +
                    " = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_row_range.begin, __or_cols, \"[DACPP][MPI][OR] row-block local offset overflow\");\n";
            code += "        const int64_t __or_rowblock_local_bytes_" +
                    param.calcParamName +
                    " = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                    type +
                    ")), \"[DACPP][MPI][OR] row-block local byte copy size overflow\");\n";
            code += "        std::memcpy(" + local +
                    ".data(), " + global +
                    ".data() + __or_rowblock_offset_" +
                    param.calcParamName +
                    ", static_cast<std::size_t>(__or_rowblock_local_bytes_" +
                    param.calcParamName + "));\n";
        }
        code += "    }\n";
        return;
    }
    if (!usesByte(plan, param) &&
        plan.signature.layout == LocalLayoutKind::RowBlock2D) {
        code += "    if (mpi_rank != 0) {\n";
        code += "        " + global +
                ".resize(static_cast<std::size_t>(__or_total_items));\n";
        code += "    }\n";
        code += "    auto dacpp_profile_bcast_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "    MPI_Bcast(" + global + ".data(), " +
                checkedMpiPayloadCountExpr(
                    "__or_total_items", type,
                    "[DACPP][MPI][OR] row-block broadcast count exceeds MPI int range") +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Bcast, "
                "dacpp_profile_bcast_start_" +
                param.calcParamName + ");\n";
        code += "    const int64_t __or_rowblock_offset_" + param.calcParamName +
                " = __or_row_range.begin * __or_cols;\n";
        code += "    std::memcpy(" + local +
                ".data(), " + global + ".data() + __or_rowblock_offset_" +
                param.calcParamName +
                ", static_cast<std::size_t>(__or_local_item_count) * sizeof(" +
                type + "));\n";
        return;
    }
    if (usesByte(plan, param)) {
        emitByteCounts(code, param.calcParamName, type);
        code += "    auto dacpp_profile_scatter_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                ".data() : nullptr, " + mpiType + ", " + local +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                type + ")), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Scatter, "
                "dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
    } else {
        code += "    auto dacpp_profile_scatter_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        const std::string sendData =
            canUseDirectTensorScatter
                ? "__or_scatter_src_" + param.calcParamName
                : global + ".data()";
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + sendData +
                " : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                mpiType + ", " + local +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), \"[DACPP][MPI][OR] scatter count exceeds MPI int range\"), " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, "
                "dacpp::mpi::ProfileSegment::Scatter, "
                "dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
    }
}

void emitResidentOrScatter(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param) {
    if (!param.readFromResident) {
        emitScatter(code, plan, param);
        return;
    }

    const std::string type = elemType(plan, param);
    const std::string local = localName(param);
    const std::string resident = "__or_resident_" + param.calcParamName;
    if (param.reads && !param.writes) {
        code += "    std::vector<" + type + ">* " + resident +
                " = dacpp::mpi::operator_resident::find_resident<" + type +
                ">(" + paramVarName(param) + ");\n";
        code += "    if (!" + resident + ") {\n";
        code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] missing resident tensor " +
                param.shellParamName + "\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 3);\n";
        code += "    }\n";
        code += "    auto& " + local + " = *" + resident + ";\n";
        return;
    }

    code += "    std::vector<" + type + "> " + local + ";\n";
    code += "    if (auto* __or_resident_" + param.calcParamName +
            " = dacpp::mpi::operator_resident::find_resident<" + type +
            ">(" + paramVarName(param) + ")) {\n";
    code += "        " + local + " = *__or_resident_" + param.calcParamName +
            ";\n";
    code += "    } else {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] missing resident tensor " +
            param.shellParamName + "\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 3);\n";
    code += "    }\n";
}

void emitGatherMaterialize(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param,
                           const std::string& profileName) {
    emitGatherMaterializeFromLocalBuffer(code, plan, param, localName(param),
                                         "__or_total_items", profileName);
}

void emitGatherMaterializeFromLocalBuffer(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& param,
    const std::string& localBufferName,
    const std::string& totalItemCountExpr,
    const std::string& profileName) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string global = "__or_materialized_" + param.calcParamName;
    code += "    std::vector<" + type + "> " + global + ";\n";
    code += "    " + checkedMpiCountExpr(
            totalItemCountExpr,
            "[DACPP][MPI][OR] materialized output size exceeds MPI int range") +
            ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + global + ".resize(static_cast<std::size_t>(" +
            totalItemCountExpr + "));\n";
    code += "    }\n";
    const bool hasProfile = !profileName.empty();
    if (hasProfile) {
        code += "    auto dacpp_profile_gather_start_" + param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
    }
    if (usesByte(plan, param)) {
        emitByteCounts(code, param.calcParamName + "_gather", type);
        code += "    MPI_Gatherv(" + localBufferName +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                type + ")), \"[DACPP][MPI][OR] gather byte count exceeds MPI int range\"), \"[DACPP][MPI][OR] gather byte count exceeds MPI int range\"), " + mpiType + ", mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, " + mpiType +
                ", 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Gatherv(" + localBufferName +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), \"[DACPP][MPI][OR] gather count exceeds MPI int range\"), " +
                mpiType + ", mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    if (hasProfile) {
        code += "    dacpp::mpi::recordProfileSegment(" + profileName +
                ", dacpp::mpi::ProfileSegment::Gather, "
                "dacpp_profile_gather_start_" +
                param.calcParamName + ");\n";
        code += "    auto dacpp_profile_materialize_start_" +
                param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
    }
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".array2Tensor(" + global +
            ");\n";
    if (param.broadcastMaterializedOutput) {
        code += "        " + checkedMpiPayloadCountExpr(
                paramVarName(param) + ".getSize()", type,
                "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ";\n";
        code += "        if (!" + global + ".empty()) {\n";
        if (hasProfile) {
            code += "            auto dacpp_profile_bcast_start_" +
                    param.calcParamName +
                    " = dacpp::mpi::profileSegmentStart();\n";
        }
        code += "            MPI_Bcast(" + global + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(param) + ".getSize()", type,
                    "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        if (hasProfile) {
            code += "            dacpp::mpi::recordProfileSegment(" +
                    profileName +
                    ", dacpp::mpi::ProfileSegment::Bcast, "
                    "dacpp_profile_bcast_start_" +
                    param.calcParamName + ");\n";
        }
        code += "        }\n";
    }
    if (param.broadcastMaterializedOutput) {
    code += "    } else {\n";
        code += "        " + checkedMpiPayloadCountExpr(
                paramVarName(param) + ".getSize()", type,
                "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ";\n";
        code += "        " + global + ".resize(static_cast<std::size_t>(" +
                paramVarName(param) + ".getSize()));\n";
        code += "        if (!" + global + ".empty()) {\n";
        if (hasProfile) {
            code += "            auto dacpp_profile_bcast_start_" +
                    param.calcParamName +
                    " = dacpp::mpi::profileSegmentStart();\n";
        }
        code += "            MPI_Bcast(" + global + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(param) + ".getSize()", type,
                    "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        if (hasProfile) {
            code += "            dacpp::mpi::recordProfileSegment(" +
                    profileName +
                    ", dacpp::mpi::ProfileSegment::Bcast, "
                    "dacpp_profile_bcast_start_" +
                    param.calcParamName + ");\n";
        }
        code += "        }\n";
        code += "        " + paramVarName(param) + ".array2Tensor(" + global +
                ");\n";
        code += "    }\n";
    } else {
        code += "    }\n";
    }
    if (hasProfile) {
        code += "    dacpp::mpi::recordProfileSegment(" + profileName +
                ", dacpp::mpi::ProfileSegment::Materialize, "
                "dacpp_profile_materialize_start_" +
                param.calcParamName + ");\n";
    }
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
