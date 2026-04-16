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
        code += "        sycl::host_accessor ha_" + name + "(*ctx.buf_" + name +
                ", sycl::read_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name +
                ".size(); ++i)\n";
        code += "            ctx.local_" + name + "[i] = ha_" + name + "[i];\n";
        code += "        MPI_Datatype mpi_dt_" + name + " = " + mpiType + ";\n";
        code += "        dacpp::mpi::exchangeHalo(ctx.local_" + name +
                ", ctx.halo_" + name + ", &mpi_dt_" + name + ");\n";
        code += "        sycl::host_accessor ha_w_" + name + "(*ctx.buf_" +
                name + ", sycl::write_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name +
                ".size(); ++i)\n";
        code += "            ha_w_" + name + "[i] = ctx.local_" + name +
                "[i];\n";
        code += "    }\n";
    }
    code += "}\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
