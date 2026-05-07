#include <string>

#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string operatorResidentWrapperName(Shell* shell,
                                        Calc* calc,
                                        int exprIndex) {
    return "__dacpp_mpi_or_" + shell->getName() + "_" + calc->getName() +
           "_" + std::to_string(exprIndex);
}

std::string buildOperatorResidentWrapperCode(
    DacppFile*,
    const OperatorResidentChainPlan&,
    const ShellPartitionPlan& exprPlan) {
    Shell* shell = exprPlan.exprNode.shell;
    Calc* calc = exprPlan.exprNode.calc;
    const std::string wrapper =
        operatorResidentWrapperName(shell, calc, exprPlan.exprIndex);

    std::string code;
    code += "void " + wrapper + "(" +
            operator_resident::wrapperSignature(exprPlan) + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    operator_resident::emitPartitionCode(code, exprPlan);
    operator_resident::emitParamLocalStorage(code, exprPlan);
    operator_resident::emitKernel(code, exprPlan);
    operator_resident::emitResidencyAndMaterialization(code, exprPlan);
    code += "}\n";
    return code;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
