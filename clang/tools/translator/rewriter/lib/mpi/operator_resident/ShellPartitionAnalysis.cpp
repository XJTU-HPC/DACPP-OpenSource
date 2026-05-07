#include <map>
#include <string>

#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

void reject(ShellPartitionPlan& plan, const std::string& reason) {
    plan.supported = false;
    plan.rejectReason = reason;
    const std::string shellName =
        plan.exprNode.shell ? plan.exprNode.shell->getName() : "<null>";
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shellName
                 << " rejected reason=" << reason << "\n";
}

} // namespace

const char* localLayoutKindName(LocalLayoutKind kind) {
    switch (kind) {
    case LocalLayoutKind::Contiguous1D:
        return "Contiguous1D";
    case LocalLayoutKind::RowBlock2D:
        return "RowBlock2D";
    case LocalLayoutKind::RowPartitionFullRow:
        return "RowPartitionFullRow";
    case LocalLayoutKind::ReplicatedScalar:
        return "ReplicatedScalar";
    case LocalLayoutKind::ReplicatedFullTensor:
        return "ReplicatedFullTensor";
    case LocalLayoutKind::StencilWindow1D:
        return "StencilWindow1D";
    case LocalLayoutKind::StencilWindow2D:
        return "StencilWindow2D";
    case LocalLayoutKind::FixedBlock:
        return "FixedBlock";
    case LocalLayoutKind::Unsupported:
        return "Unsupported";
    }
    return "Unsupported";
}

const char* paramAccessKindName(ParamAccessKind kind) {
    switch (kind) {
    case ParamAccessKind::DirectMapped:
        return "DirectMapped";
    case ParamAccessKind::OutputDirect:
        return "OutputDirect";
    case ParamAccessKind::ReplicatedScalar:
        return "ReplicatedScalar";
    case ParamAccessKind::ReplicatedFullTensor:
        return "ReplicatedFullTensor";
    case ParamAccessKind::RowPartitionFullRow:
        return "RowPartitionFullRow";
    case ParamAccessKind::StencilWindow:
        return "StencilWindow";
    case ParamAccessKind::FixedBlock:
        return "FixedBlock";
    case ParamAccessKind::Unsupported:
        return "Unsupported";
    }
    return "Unsupported";
}

ShellPartitionPlan analyzeShellPartition(const DacExprNode& node) {
    ShellPartitionPlan plan;
    plan.exprIndex = node.exprIndex;
    plan.exprNode = node;

    Shell* shell = node.shell;
    Calc* calc = node.calc;
    if (!shell || !calc) {
        reject(plan, "missing shell or calc");
        return plan;
    }
    if (shell->getNumShellParams() != calc->getNumParams() ||
        shell->getNumShellParams() != shell->getNumParams()) {
        reject(plan, "shell/calc parameter count mismatch");
        return plan;
    }

    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto splitMeta = collectSplitBindMeta(shell);

    std::map<int, BindDomain> bindDomainsById;
    std::vector<int> firstDirectOrder;
    bool sawDirectParam = false;
    bool sawWriteParam = false;
    bool sawScalarParam = false;

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        if (!shellParam || !shellWrapperParam || !calcParam) {
            reject(plan, "null parameter metadata");
            return plan;
        }

        ParamAccessPlan paramPlan;
        paramPlan.paramIndex = paramIdx;
        paramPlan.shellParamName = shellWrapperParam->getName();
        paramPlan.calcParamName = calcParam->getName();
        paramPlan.actualTensorName = shellWrapperParam->getName();
        paramPlan.reads = paramModes[paramIdx] == IOTYPE::READ ||
                          paramModes[paramIdx] == IOTYPE::READ_WRITE;
        paramPlan.writes = paramModes[paramIdx] == IOTYPE::WRITE ||
                           paramModes[paramIdx] == IOTYPE::READ_WRITE;

        if (paramModes[paramIdx] == IOTYPE::READ_WRITE) {
            reject(plan, "read_write parameter unsupported");
            return plan;
        }

        bool onlyVoid = shellParam->getNumSplit() > 0;
        bool hasIndex = false;
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (operator_resident::splitIsRegular(split)) {
                reject(plan, "contains regular split");
                return plan;
            }
            if (operator_resident::splitIsVoid(split)) {
                TensorDimMapping mapping;
                mapping.tensorName = shellWrapperParam->getName();
                mapping.shellParamIndex = paramIdx;
                mapping.tensorDim = split ? split->getDimIdx() : splitIdx;
                mapping.kind = ShellDimKind::Void;
                mapping.splitName = "void";
                plan.mappings.push_back(mapping);
                continue;
            }
            onlyVoid = false;
            if (!operator_resident::splitIsIndex(split)) {
                reject(plan, "contains unsupported split kind");
                return plan;
            }

            auto metaIt = splitMeta.find(split->getId());
            if (metaIt == splitMeta.end()) {
                reject(plan, "missing bind metadata");
                return plan;
            }
            if (metaIt->second.offset != "0" && !metaIt->second.offset.empty()) {
                reject(plan, "nonzero bind offset");
                return plan;
            }

            const int bindId = metaIt->second.bindId;
            const int dimId = split->getDimIdx();
            paramPlan.bindOrder.push_back(bindId);
            paramPlan.tensorDims.push_back(dimId);
            hasIndex = true;

            TensorDimMapping mapping;
            mapping.tensorName = shellWrapperParam->getName();
            mapping.shellParamIndex = paramIdx;
            mapping.tensorDim = dimId;
            mapping.kind = ShellDimKind::Index;
            mapping.bindId = bindId;
            mapping.splitName = split->getId();
            plan.mappings.push_back(mapping);

            auto& domain = bindDomainsById[bindId];
            if (domain.bindId == -1) {
                domain.bindId = bindId;
                domain.representative = split->getId();
                domain.offsetExpr = "0";
                domain.runtimeSizeParam = paramIdx;
                domain.dimId = dimId;
            }

            if (firstDirectOrder.empty() || !sawDirectParam) {
                firstDirectOrder.push_back(bindId);
            }
        }

        if (onlyVoid && !hasIndex) {
            if (!paramPlan.reads || paramPlan.writes) {
                reject(plan, "void output unsupported");
                return plan;
            }
            if (shellParam->getNumSplit() != 1 ||
                !operator_resident::isScalarVoidParam(shell, paramIdx) ||
                !operator_resident::calcUsesParamAsScalar(calc, paramIdx)) {
                reject(plan, "void tensor is not a 1D scalar");
                return plan;
            }
            paramPlan.access = ParamAccessKind::ReplicatedScalar;
            sawScalarParam = true;
        } else if (hasIndex) {
            if (static_cast<int>(paramPlan.bindOrder.size()) !=
                shellParam->getNumSplit()) {
                reject(plan, "direct parameter mixes index and void dims");
                return plan;
            }
            sawDirectParam = true;
            if (paramPlan.writes) {
                paramPlan.access = ParamAccessKind::OutputDirect;
                sawWriteParam = true;
            } else {
                paramPlan.access = ParamAccessKind::DirectMapped;
            }
        } else {
            reject(plan, "parameter has no supported split");
            return plan;
        }

        plan.params.push_back(paramPlan);
    }

    if (!sawDirectParam) {
        reject(plan, "no direct mapped parameter");
        return plan;
    }
    if (!sawWriteParam) {
        reject(plan, "no direct output parameter");
        return plan;
    }

    plan.signature.bindOrder =
        operator_resident::uniquePreserveOrder(firstDirectOrder);
    for (int bindId : plan.signature.bindOrder) {
        auto it = bindDomainsById.find(bindId);
        if (it == bindDomainsById.end()) {
            reject(plan, "bind domain missing from signature");
            return plan;
        }
        plan.bindDomains.push_back(it->second);
        plan.signature.bindSizes.push_back(
            operator_resident::shapeValueFor(
                shell, static_cast<int>(it->second.runtimeSizeParam),
                it->second.dimId));
    }

    std::string layoutRejectReason;
    if (!operator_resident::assignPhaseLayout(plan, sawScalarParam,
                                              layoutRejectReason)) {
        reject(plan, layoutRejectReason);
        return plan;
    }

    plan.supported = true;
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shell->getName()
                 << " layout=" << localLayoutKindName(plan.signature.layout)
                 << " accepted\n";
    for (const auto& param : plan.params) {
        llvm::outs() << "[DACPP][MPI][OR]   param=" << param.shellParamName
                     << " access=" << paramAccessKindName(param.access)
                     << " reads=" << (param.reads ? "1" : "0")
                     << " writes=" << (param.writes ? "1" : "0") << "\n";
    }
    for (const auto& mapping : plan.mappings) {
        llvm::outs() << "[DACPP][MPI][OR]   mapping tensor="
                     << mapping.tensorName << " dim=" << mapping.tensorDim
                     << " kind="
                     << operator_resident::shellDimKindName(mapping.kind)
                     << " bind=" << mapping.bindId << "\n";
    }
    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
