#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

void emitKernel(std::string& code, const ShellPartitionPlan& plan) {
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        {\n";
    for (const auto& param : plan.params) {
        const std::string type = elemType(plan, param);
        const std::string name = param.calcParamName;
        code += "            sycl::buffer<" + type + ", 1> __or_buffer_" +
                name + "(" + localName(param) +
                ".data(), sycl::range<1>(" + localName(param) +
                ".size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (const auto& param : plan.params) {
        const std::string mode =
            param.reads && !param.writes ? "sycl::access::mode::read"
                                         : "sycl::access::mode::read_write";
        code += "                auto __or_acc_" + param.calcParamName +
                " = __or_buffer_" + param.calcParamName +
                ".get_access<" + mode + ">(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (const auto& param : plan.params) {
        code += "                    auto* __or_data_" + param.calcParamName +
                " = __or_acc_" + param.calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        const std::string offset =
            param.access == ParamAccessKind::ReplicatedScalar ? "0"
                                                              : "item_linear";
        code += "                    " + viewType(plan, param) + " view_" +
                param.calcParamName + "{__or_data_" + param.calcParamName +
                ", " + offset + "};\n";
    }
    code += "                    " + plan.exprNode.calc->getName() +
            "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
