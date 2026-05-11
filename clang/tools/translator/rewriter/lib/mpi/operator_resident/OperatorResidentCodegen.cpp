#include <string>

#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"
#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

bool isShellDerivedStencilLayout(LocalLayoutKind layout) {
    return layout == LocalLayoutKind::StencilWindow1D ||
           layout == LocalLayoutKind::StencilWindow2D;
}

bool isLoopLoweredOperatorResidentPlan(const ShellPartitionPlan& plan) {
    return plan.loopLowerCandidate ||
           plan.orLoopLower.kind == OrLoopLowerKind::StencilFullSync ||
           plan.orLoopLower.kind == OrLoopLowerKind::StencilResidentHalo ||
           plan.orLoopLower.kind ==
               OrLoopLowerKind::FixedBlockPhaseExchange ||
           plan.orLoopLower.kind ==
               OrLoopLowerKind::FixedBlockPhaseExchangeFollower;
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
    if (exprPlan.orLoopLower.kind == OrLoopLowerKind::StencilResidentHalo &&
        exprPlan.signature.layout == LocalLayoutKind::StencilWindow2D) {
        return operator_resident::
            buildLoopLoweredStencil2DResidentHaloFamilyCode(
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
    if (exprPlan.orLoopLower.kind ==
            OrLoopLowerKind::FixedBlockPhaseExchange &&
        exprPlan.signature.layout == LocalLayoutKind::FixedBlock) {
        return operator_resident::
            buildLoopLoweredFixedBlockPhaseExchangeFamilyCode(
                wrapper, dacppFile, exprPlan);
    }
    if (exprPlan.orLoopLower.kind ==
            OrLoopLowerKind::FixedBlockPhaseExchangeFollower) {
        // The follower DAC expression is removed at rewrite time, so its
        // wrapper body is intentionally empty. Emitting a stub keeps the
        // generated wrapper-name table consistent for any leftover
        // references but the function is never called.
        std::string stub;
        stub += "void " + wrapper + "(" +
                operator_resident::wrapperSignature(exprPlan) + ") {\n";
        for (const auto& param : exprPlan.params) {
            stub += "    (void)" + operator_resident::paramVarName(param) +
                    ";\n";
        }
        stub += "}\n";
        return stub;
    }
    if (exprPlan.signature.layout == LocalLayoutKind::FixedBlock) {
        return operator_resident::buildFixedBlockWrapperCode(
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
    code += "    dacpp::mpi::SegmentedProfile dacpp_profile;\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += mpi_rewriter::profileSegmentStartCode("dacpp_profile_init_start");
    operator_resident::emitPartitionCode(code, exprPlan);
    code += mpi_rewriter::profileRecordCode("dacpp_profile", "Init",
                                            "dacpp_profile_init_start");
    operator_resident::emitParamLocalStorage(code, exprPlan);
    code += mpi_rewriter::profileSegmentStartCode(
        "dacpp_profile_kernel_start");
    operator_resident::emitKernel(code, exprPlan);
    code += mpi_rewriter::profileRecordCode("dacpp_profile", "Kernel",
                                            "dacpp_profile_kernel_start");
    operator_resident::emitResidencyAndMaterialization(code, exprPlan);
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + wrapper +
            "\", dacpp_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
    return code;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
