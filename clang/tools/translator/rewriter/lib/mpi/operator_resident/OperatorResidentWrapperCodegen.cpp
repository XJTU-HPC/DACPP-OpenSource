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
