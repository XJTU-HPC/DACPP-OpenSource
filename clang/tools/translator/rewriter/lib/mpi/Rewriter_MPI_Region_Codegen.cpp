#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

MPIRegionGeneratedCode buildMPIRegionNames(Expression* expr) {
    MPIRegionGeneratedCode generated;
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return generated;
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const std::string baseName = shell->getName() + "_" + calc->getName();
    generated.ctxTypeName = "__dacpp_mpi_ctx_" + baseName;
    generated.ctxVarName = generated.ctxTypeName + "_0";
    generated.initName = "__dacpp_mpi_init_" + baseName;
    generated.submitName = "__dacpp_mpi_submit_" + baseName;
    generated.haloName = "__dacpp_mpi_halo_" + baseName;
    generated.syncName = "__dacpp_mpi_sync_" + baseName;
    return generated;
}

}  // namespace

std::string buildShellParamSignature(Shell* shell) {
    if (!shell) {
        return "";
    }

    std::string shellParamSig;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' &&
            paramType.back() != '*') {
            paramType += "&";
        }
        if (!shellParamSig.empty()) {
            shellParamSig += ", ";
        }
        shellParamSig += paramType + " " + param->getName();
    }
    return shellParamSig;
}

std::string buildMPIRegionCodegen(
    DacppFile* dacppFile,
    Expression* expr,
    const MPIRegionTransferPolicy& transferPolicy) {
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return "";
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const auto generated = buildMPIRegionNames(expr);
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto storageModes =
        inferMPIRegionStorageModes(dacppFile, expr, paramModes);
    const auto splitMeta = collectSplitBindMeta(shell);
    const std::string shellParamSig = buildShellParamSignature(shell);

    std::string code;

    code += buildMPIRegionCtxCode(dacppFile, shell, calc, generated);
    code += buildMPIRegionInitCode(dacppFile, shell, calc, generated,
                                   storageModes, transferPolicy,
                                   splitMeta, shellParamSig);
    code += buildMPIRegionSubmitCode(shell, calc, generated, paramModes);
    code += buildMPIRegionHaloCode(shell, calc, generated, paramModes);
    code += buildMPIRegionSyncCode(dacppFile, shell, calc, generated,
                                   storageModes, transferPolicy,
                                   shellParamSig);

    return code;
}

MPIRegionGeneratedCode buildMPIRegionCode(DacppFile* dacppFile,
                                          Expression* expr) {
    MPIRegionGeneratedCode generated = buildMPIRegionNames(expr);
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return generated;
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto transferPolicy =
        analyzeMPIRegionTransferPolicy(dacppFile, expr, paramModes);
    generated.definitions =
        buildMPIRegionCodegen(dacppFile, expr, transferPolicy);
    generated.definitions += buildMPIRegionSiblingCode(dacppFile, expr, generated);
    return generated;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
