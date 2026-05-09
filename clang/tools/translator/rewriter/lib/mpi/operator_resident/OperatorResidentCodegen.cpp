#include <string>

#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {

bool isShellDerivedStencilLayout(LocalLayoutKind layout) {
    return layout == LocalLayoutKind::StencilWindow1D ||
           layout == LocalLayoutKind::StencilWindow2D;
}

bool isLoopLoweredOperatorResidentPlan(const ShellPartitionPlan& plan) {
    return plan.loopLowerCandidate ||
           plan.orLoopLower.kind == OrLoopLowerKind::StencilFullSync ||
           plan.orLoopLower.kind == OrLoopLowerKind::StencilResidentHalo;
}

std::string operatorResidentWrapperName(Shell* shell,
                                        Calc* calc,
                                        int exprIndex) {
    return "__dacpp_mpi_or_" + shell->getName() + "_" + calc->getName() +
           "_" + std::to_string(exprIndex);
}

std::string operatorResidentContextTypeName(Shell* shell,
                                            Calc* calc,
                                            int exprIndex) {
    return operatorResidentWrapperName(shell, calc, exprIndex) + "_ctx";
}

std::string operatorResidentInitFunctionName(Shell* shell,
                                             Calc* calc,
                                             int exprIndex) {
    return operatorResidentWrapperName(shell, calc, exprIndex) + "_init";
}

std::string operatorResidentRunFunctionName(Shell* shell,
                                            Calc* calc,
                                            int exprIndex) {
    return operatorResidentWrapperName(shell, calc, exprIndex) + "_run";
}

std::string operatorResidentMaterializeFunctionName(Shell* shell,
                                                    Calc* calc,
                                                    int exprIndex) {
    return operatorResidentWrapperName(shell, calc, exprIndex) +
           "_materialize";
}

std::string buildOperatorResidentWrapperCode(
    DacppFile* dacppFile,
    const OperatorResidentChainPlan&,
    const ShellPartitionPlan& exprPlan) {
    Shell* shell = exprPlan.exprNode.shell;
    Calc* calc = exprPlan.exprNode.calc;
    const std::string wrapper =
        operatorResidentWrapperName(shell, calc, exprPlan.exprIndex);

    if (exprPlan.orLoopLower.kind == OrLoopLowerKind::StencilFullSync &&
        exprPlan.signature.layout == LocalLayoutKind::StencilWindow1D) {
        return operator_resident::buildLoopLoweredStencil1DFullSyncFamilyCode(
            wrapper, dacppFile, exprPlan);
    }
    if (exprPlan.orLoopLower.kind == OrLoopLowerKind::StencilResidentHalo &&
        exprPlan.signature.layout == LocalLayoutKind::StencilWindow1D) {
        return operator_resident::
            buildLoopLoweredStencil1DResidentHaloFamilyCode(
                wrapper, dacppFile, exprPlan);
    }
    if (exprPlan.orLoopLower.kind == OrLoopLowerKind::StencilFullSync &&
        exprPlan.signature.layout == LocalLayoutKind::StencilWindow2D) {
        return operator_resident::buildLoopLoweredStencil2DFullSyncFamilyCode(
            wrapper, dacppFile, exprPlan);
    }
    if (isShellDerivedStencilLayout(exprPlan.signature.layout)) {
        if (exprPlan.signature.layout == LocalLayoutKind::StencilWindow1D) {
            return operator_resident::buildStencilWindow1DWrapperCode(
                wrapper, dacppFile, exprPlan);
        }
        return operator_resident::buildStencilWindow2DWrapperCode(
            wrapper, dacppFile, exprPlan);
    }
    if (exprPlan.loopLowerCandidate &&
        exprPlan.signature.layout == LocalLayoutKind::Contiguous1D) {
        return operator_resident::buildLoopLoweredDirect1DFamilyCode(
            wrapper, dacppFile, exprPlan);
    }

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
