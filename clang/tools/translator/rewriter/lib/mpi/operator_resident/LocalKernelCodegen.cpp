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
        std::string offset = "item_linear";
        if (param.access == ParamAccessKind::ReplicatedScalar ||
            param.access == ParamAccessKind::ReplicatedFullTensor) {
            offset = "0";
        } else if (param.access == ParamAccessKind::RowPartitionFullRow) {
            int indexedBindPos = 0;
            if (!param.bindOrder.empty()) {
                for (std::size_t idx = 0;
                     idx < plan.signature.bindOrder.size(); ++idx) {
                    if (plan.signature.bindOrder[idx] == param.bindOrder[0]) {
                        indexedBindPos = static_cast<int>(idx);
                        break;
                    }
                }
            }
            std::string localIndex;
            if (plan.signature.bindOrder.size() == 1) {
                localIndex = "item_linear";
            } else if (indexedBindPos == 0) {
                localIndex = "item_linear / static_cast<int>(__or_cols)";
            } else {
                localIndex = "item_linear % static_cast<int>(__or_cols)";
            }
            offset = "static_cast<int>((" + localIndex +
                     ") * __or_payload_len_" + param.calcParamName + ")";
        }
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

const ParamAccessPlan* fusedDirectReader(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped && param.reads &&
            !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* fusedOutputWriter(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes &&
            !param.reads) {
            return &param;
        }
    }
    return nullptr;
}

void emitFusedPointwiseRowBlock2DKernel(std::string& code,
                                        const OperatorResidentChainPlan& chain) {
    if (!chain.fusePointwiseRowBlock2D || chain.exprPlans.size() < 2) {
        return;
    }
    const ShellPartitionPlan& first = chain.exprPlans[0];
    const ShellPartitionPlan& last = chain.exprPlans.back();
    const ParamAccessPlan* firstReader = fusedDirectReader(first);
    const ParamAccessPlan* firstWriter = fusedOutputWriter(first);
    const ParamAccessPlan* lastWriter = fusedOutputWriter(last);
    if (!firstReader || !firstWriter || !lastWriter) {
        return;
    }
    const std::string firstReaderType = elemType(first, *firstReader);
    const std::string intermediateType = elemType(first, *firstWriter);
    const std::string finalType = elemType(last, *lastWriter);
    emitResidentOrScatter(code, first, *firstReader);
    if (firstReader->constantInit.supported) {
        code += "    // Fused RowBlock2D input " + firstReader->calcParamName +
                " is generated locally; skip root pack/scatter.\n";
    }
    code += "    std::vector<" + finalType + "> " + localName(*lastWriter) +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        {\n";
    code += "            sycl::buffer<" + firstReaderType + ", 1> __or_buffer_" +
            firstReader->calcParamName + "(" + localName(*firstReader) +
            ".data(), sycl::range<1>(" + localName(*firstReader) +
            ".size()));\n";
    code += "            sycl::buffer<" + finalType + ", 1> __or_buffer_" +
            lastWriter->calcParamName + "(" + localName(*lastWriter) +
            ".data(), sycl::range<1>(" + localName(*lastWriter) +
            ".size()));\n";
    code += "            q.submit([&](sycl::handler& h) {\n";
    code += "                auto __or_acc_" + firstReader->calcParamName +
            " = __or_buffer_" + firstReader->calcParamName +
            ".get_access<sycl::access::mode::read>(h);\n";
    code += "                auto __or_acc_" + lastWriter->calcParamName +
            " = __or_buffer_" + lastWriter->calcParamName +
            ".get_access<sycl::access::mode::read_write>(h);\n";
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                    auto* __or_data_" + firstReader->calcParamName +
            " = __or_acc_" + firstReader->calcParamName +
            ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                    dacpp::mpi::ContiguousView1D<const " +
            firstReaderType + "> view_" + firstReader->calcParamName +
            "{__or_data_" + firstReader->calcParamName + ", item_linear};\n";
    for (std::size_t planIdx = 0; planIdx < chain.exprPlans.size(); ++planIdx) {
        const ShellPartitionPlan& stage = chain.exprPlans[planIdx];
        const ParamAccessPlan* stageReader = fusedDirectReader(stage);
        const ParamAccessPlan* stageWriter = fusedOutputWriter(stage);
        if (!stageReader || !stageWriter) {
            return;
        }
        const std::string stageOutType = elemType(stage, *stageWriter);
        if (planIdx + 1 == chain.exprPlans.size()) {
            code += "                    auto* __or_data_" +
                    lastWriter->calcParamName + " = __or_acc_" +
                    lastWriter->calcParamName +
                    ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                    dacpp::mpi::ContiguousView1D<" +
                    stageOutType + "> view_" + stageWriter->calcParamName +
                    "{__or_data_" + lastWriter->calcParamName +
                    ", item_linear};\n";
        } else {
            code += "                    " + stageOutType + " __or_private_" +
                    stageWriter->calcParamName + "{};\n";
            code += "                    dacpp::mpi::ContiguousView1D<" +
                    stageOutType + "> view_" + stageWriter->calcParamName +
                    "{&__or_private_" + stageWriter->calcParamName +
                    ", 0};\n";
        }
        std::string inputView;
        if (planIdx == 0) {
            inputView = "view_" + firstReader->calcParamName;
        } else {
            const ParamAccessPlan* previousWriter =
                fusedOutputWriter(chain.exprPlans[planIdx - 1]);
            if (!previousWriter) {
                return;
            }
            inputView = "view_" + stageReader->calcParamName + "_read";
            code += "                    dacpp::mpi::ContiguousView1D<const " +
                    elemType(chain.exprPlans[planIdx - 1],
                             *previousWriter) +
                    "> " + inputView + "{&__or_private_" +
                    previousWriter->calcParamName + ", 0};\n";
        }
        code += "                    " + stage.exprNode.calc->getName() +
                "_mpi_local(" + inputView + ", view_" +
                stageWriter->calcParamName + ");\n";
    }
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
    code += "    // Fused RowBlock2D pointwise chain stores only the final output buffer; intermediate stays private.\n";
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
