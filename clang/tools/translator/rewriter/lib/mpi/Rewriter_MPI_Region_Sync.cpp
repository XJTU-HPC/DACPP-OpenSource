#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildMPIRegionSyncCode(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated,
    const std::vector<IOTYPE>& storageModes,
    const MPIRegionTransferPolicy& transferPolicy,
    const std::string& shellParamSig,
    const clang::BinaryOperator* dacExpr) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "void " + generated.syncName + "(" + generated.ctxTypeName +
            "& ctx";
    if (!shellParamSig.empty()) {
        code += ", " + shellParamSig;
    }
    code += ") {\n";
    code += "    ctx.q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (!transferPolicy.needsSyncGather[static_cast<std::size_t>(paramIdx)]) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const bool needsBcast = tensorNeedsBroadcast(dacppFile, tensorName, dacExpr);

        code += "    {\n";
        code += "        dacpp::mpi::writeback_region_output(" + tensorName +
                ", ctx.state_" + calcName + ", " +
                std::string(needsBcast ? "true" : "false") +
                ", ctx.mpi_rank, ctx.mpi_size, " + mpiType + ");\n";
        code += "    }\n";
    }
    code += "}\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
