#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

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

    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    std::vector<" + type + "> " + global + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + global +
            ");\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        emitByteCounts(code, param.calcParamName, type);
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                ".data() : nullptr, " + mpiType + ", " + local +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_local_item_count), static_cast<int64_t>(sizeof(" +
                type + ")), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), \"[DACPP][MPI][OR] scatter byte count exceeds MPI int range\"), " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                mpiType + ", " + local +
                ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), \"[DACPP][MPI][OR] scatter count exceeds MPI int range\"), " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
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
                           const ParamAccessPlan& param) {
    emitGatherMaterializeFromLocalBuffer(code, plan, param, localName(param),
                                         "__or_total_items");
}

void emitGatherMaterializeFromLocalBuffer(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& param,
    const std::string& localBufferName,
    const std::string& totalItemCountExpr) {
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
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".array2Tensor(" + global +
            ");\n";
    if (param.broadcastMaterializedOutput) {
        code += "        " + checkedMpiPayloadCountExpr(
                paramVarName(param) + ".getSize()", type,
                "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ";\n";
        code += "        if (!" + global + ".empty()) {\n";
        code += "            MPI_Bcast(" + global + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(param) + ".getSize()", type,
                    "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
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
        code += "            MPI_Bcast(" + global + ".data(), " +
                checkedMpiPayloadCountExpr(
                    paramVarName(param) + ".getSize()", type,
                    "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range") +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "        " + paramVarName(param) + ".array2Tensor(" + global +
                ");\n";
        code += "    }\n";
    } else {
        code += "    }\n";
    }
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
