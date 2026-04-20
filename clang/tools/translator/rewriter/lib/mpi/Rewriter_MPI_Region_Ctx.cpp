#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildMPIRegionCtxCode(DacppFile* dacppFile,
                                  Shell* shell,
                                  Calc* calc,
                                  const MPIRegionGeneratedCode& generated) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "struct " + generated.ctxTypeName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    sycl::queue q{};\n";
    code += "    int64_t total_items = 1;\n";
    code += "    int64_t local_item_count = 0;\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += "    bool has_halo = false;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        code += "    dacpp::mpi::RegionParamState<" + baseType +
                "> state_" + name + ";\n";
    }

    // Add non-shell captured variable fields
    if (dacppFile) {
        const auto& plan = dacppFile->getBufferRegionPlan();
        for (const auto& var : plan.capturedNonShellVars) {
            code += "    " + var.second + " " + var.first + ";\n";
        }
    }

    code += "};\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
