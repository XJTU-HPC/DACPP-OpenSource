#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

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
    code += "        " + paramVarName(param) + ".tensor2Array(" + local +
            ");\n";
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
    int indexedBindPos = 0;
    if (!param.bindOrder.empty()) {
        for (std::size_t idx = 0; idx < plan.signature.bindOrder.size(); ++idx) {
            if (plan.signature.bindOrder[idx] == param.bindOrder[0]) {
                indexedBindPos = static_cast<int>(idx);
                break;
            }
        }
    }

    code += "    const int64_t " + payloadLen + " = " + paramVarName(param) +
            ".getShape(" + std::to_string(voidDim) + ");\n";
    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_local_item_count * " +
            payloadLen + "));\n";
    code += "    std::vector<int> __or_payload_counts_" + param.calcParamName +
            "(mpi_size);\n";
    code += "    std::vector<int> __or_payload_displs_" + param.calcParamName +
            "(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_payload_counts_" + param.calcParamName +
            "[r] = static_cast<int>(__or_counts[r] * " + payloadLen + ");\n";
    code += "        __or_payload_displs_" + param.calcParamName +
            "[r] = static_cast<int>(__or_displs[r] * " + payloadLen + ");\n";
    code += "    }\n";
    code += "    std::vector<" + type + "> __or_sendbuf_" +
            param.calcParamName + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        std::vector<" + type + "> " + global + ";\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + global +
            ");\n";
    code += "        __or_sendbuf_" + param.calcParamName +
            ".resize(static_cast<std::size_t>(__or_total_items * " +
            payloadLen + "));\n";
    code += "        const int64_t __or_input_cols_" + param.calcParamName +
            " = " + paramVarName(param) + ".getShape(1);\n";
    code += "        for (int r = 0; r < mpi_size; ++r) {\n";
    code += "            for (int64_t local_i = 0; local_i < __or_counts[r]; ++local_i) {\n";
    code += "                const int64_t item_global = static_cast<int64_t>(__or_displs[r]) + local_i;\n";
    if (plan.signature.bindOrder.size() == 1) {
        code += "                const int64_t indexed_value = item_global;\n";
    } else if (indexedBindPos == 0) {
        code += "                const int64_t indexed_value = item_global / __or_cols;\n";
    } else {
        code += "                const int64_t indexed_value = item_global % __or_cols;\n";
    }
    code += "                const int64_t dst_base = (__or_payload_displs_" +
            param.calcParamName + "[r] + local_i * " +
            payloadLen + ");\n";
    code += "                for (int64_t payload_i = 0; payload_i < " +
            payloadLen + "; ++payload_i) {\n";
    if (indexDim == 0 && voidDim == 1) {
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
    code += "    }\n";
    if (usesByte(plan, param)) {
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
        code += "    MPI_Scatterv(mpi_rank == 0 ? __or_sendbuf_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_counts_bytes_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_displs_bytes_" +
                param.calcParamName + ".data() : nullptr, MPI_BYTE, " +
                local + ".data(), static_cast<int>(__or_local_item_count * " +
                payloadLen + " * sizeof(" + type +
                ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Scatterv(mpi_rank == 0 ? __or_sendbuf_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_counts_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_payload_displs_" +
                param.calcParamName + ".data() : nullptr, " + mpiType +
                ", " + local +
                ".data(), static_cast<int>(__or_local_item_count * " +
                payloadLen + "), " + mpiType + ", 0, MPI_COMM_WORLD);\n";
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
        } else if (param.writes && param.reads) {
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
        code += "    auto& __or_resident_out_" + param.calcParamName +
                " = dacpp::mpi::operator_resident::ensure_resident<" + type +
                ">(" + paramVarName(param) + ", " + localName(param) +
                ".size());\n";
        code += "    __or_resident_out_" + param.calcParamName + " = " +
                localName(param) + ";\n";
        if (param.materializeAfterWrite) {
            emitGatherMaterialize(code, plan, param);
        }
    }
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
