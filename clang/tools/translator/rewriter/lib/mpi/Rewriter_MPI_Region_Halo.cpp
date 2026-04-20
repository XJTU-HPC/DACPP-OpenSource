#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildMPIRegionHaloCode(Shell* shell,
                                   Calc* calc,
                                   const MPIRegionGeneratedCode& generated,
                                   const std::vector<IOTYPE>& paramModes) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "void " + generated.haloName + "(" + generated.ctxTypeName +
            "& ctx) {\n";
    code += "    ctx.q.wait();\n";
    code += "    if (!ctx.has_halo) return;\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const IOTYPE mode = paramModes[paramIdx];
        if (mode == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        const std::string mpiType = mpiDatatypeFor(baseType);

        code += "    {\n";
        code += "        MPI_Datatype mpi_dt_" + name + " = " + mpiType + ";\n";
        code += "        dacpp::mpi::exchange_halo_buffered(ctx.state_" + name +
                ", &mpi_dt_" + name + ");\n";
        code += "    }\n";
    }
    code += "}\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
