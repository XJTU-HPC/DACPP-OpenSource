#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

std::string paramVarName(const ParamAccessPlan& param) {
    return "__or_arg" + std::to_string(param.paramIndex);
}

std::string wrapperSignature(const ShellPartitionPlan& plan) {
    std::string signature;
    Shell* shell = plan.exprNode.shell;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' &&
            paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + paramVarName(plan.params[paramIdx]);
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }
    return signature;
}

std::string fusedParamVarName(const ParamAccessPlan& param,
                              int planOrdinal) {
    return "__or_fused_arg" + std::to_string(planOrdinal) + "_" +
           std::to_string(param.paramIndex);
}

std::string fusedWrapperSignature(const OperatorResidentChainPlan& chain) {
    if (chain.exprPlans.size() < 2) {
        return "";
    }
    auto directReader = [](const ShellPartitionPlan& plan)
        -> const ParamAccessPlan* {
        for (const auto& param : plan.params) {
            if (param.access == ParamAccessKind::DirectMapped &&
                param.reads && !param.writes) {
                return &param;
            }
        }
        return nullptr;
    };
    auto outputWriter = [](const ShellPartitionPlan& plan)
        -> const ParamAccessPlan* {
        for (const auto& param : plan.params) {
            if (param.access == ParamAccessKind::OutputDirect &&
                param.writes && !param.reads) {
                return &param;
            }
        }
        return nullptr;
    };
    const ShellPartitionPlan& first = chain.exprPlans[0];
    const ShellPartitionPlan& last = chain.exprPlans.back();
    const ParamAccessPlan* firstReader = directReader(first);
    const ParamAccessPlan* lastWriter = outputWriter(last);
    if (!firstReader || !lastWriter || !first.exprNode.shell ||
        !last.exprNode.shell) {
        return "";
    }
    auto paramDecl = [](Shell* shell, const ParamAccessPlan& param) {
        Param* shellParam = shell->getParam(param.paramIndex);
        std::string paramType = shellParam->getType();
        if (!paramType.empty() && paramType.back() != '&' &&
            paramType.back() != '*') {
            paramType += "&";
        }
        return paramType + " " + paramVarName(param);
    };
    return paramDecl(first.exprNode.shell, *firstReader) + ", " +
           paramDecl(last.exprNode.shell, *lastWriter);
}

std::string elemType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param) {
    return plan.exprNode.calc->getParam(param.paramIndex)->getBasicType();
}

std::string viewType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    if (param.reads && !param.writes) {
        return "dacpp::mpi::ContiguousView1D<const " + type + ">";
    }
    return "dacpp::mpi::ContiguousView1D<" + type + ">";
}

std::string localName(const ParamAccessPlan& param) {
    return "__or_local_" + param.calcParamName;
}

std::string globalName(const ParamAccessPlan& param) {
    return "__or_global_" + param.calcParamName;
}

std::string scalarName(const ParamAccessPlan& param) {
    return "__or_scalar_" + param.calcParamName;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
