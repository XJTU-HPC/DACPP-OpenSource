#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildMPIRegionSubmitCode(Shell* shell,
                                     Calc* calc,
                                     const MPIRegionGeneratedCode& generated,
                                     const std::vector<IOTYPE>& paramModes) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "void " + generated.submitName + "(" + generated.ctxTypeName +
            "& ctx) {\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    code += "    sycl::queue& q = ctx.q;\n";
    code +=
        "    const std::size_t local_item_count = static_cast<std::size_t>(ctx.local_item_count);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "    const int " + name + "_partition_size = ctx." + name +
                "_partition_size;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + name + "_cols = ctx." + name +
                    "_cols;\n";
        }
    }
    code += "    {\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            auto acc_" + name + " = ctx.buf_" + name +
                "->get_access<" + toAccessorMode(paramModes[paramIdx]) +
                ">(h);\n";
        code += "            auto slots_acc_" + name +
                " = ctx.slots_buf_" + name +
                "->get_access<sycl::access::mode::read>(h);\n";
    }
    code +=
        "            h.parallel_for(sycl::range<1>(local_item_count), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto* data_" + name + " = acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                auto* slots_" + name + " = slots_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (inferViewRank(shellParam, calcParam) <= 1) {
            code += "                " +
                    toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name +
                    ", item_linear * " + name + "_partition_size};\n";
        } else {
            code += "                " +
                    toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name +
                    ", item_linear * " + name + "_partition_size, " + name +
                    "_cols};\n";
        }
    }
    code += "                " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "    }\n";
    code += "}\n\n";

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
