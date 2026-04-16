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
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        code += "    dacpp::mpi::AccessPattern pattern_" + name + ";\n";
        code += "    dacpp::mpi::PackMap pack_" + name + ";\n";
        code += "    std::vector<int32_t> slots_" + name + ";\n";
        code += "    std::vector<" + baseType + "> local_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<" + baseType +
                ", 1>> buf_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<int32_t, 1>> slots_buf_" +
                name + ";\n";
        code += "    int " + name + "_partition_size = 0;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    int " + name + "_cols = 0;\n";
        }
        code += "    dacpp::mpi::ParamHalo halo_" + name + ";\n";
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
