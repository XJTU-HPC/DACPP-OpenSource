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

void emitParamLocalStorage(std::string& code,
                           const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            emitScalarBroadcast(code, plan, param);
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
