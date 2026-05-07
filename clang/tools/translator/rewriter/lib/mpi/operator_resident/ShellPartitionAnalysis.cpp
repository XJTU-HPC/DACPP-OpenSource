#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

class ScalarOnlyParamVisitor
    : public clang::RecursiveASTVisitor<ScalarOnlyParamVisitor> {
public:
    explicit ScalarOnlyParamVisitor(const clang::ValueDecl* target)
        : Target(target) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* declRef) {
        if (declRef && declRef->getDecl() == Target && SubscriptDepth == 0) {
            Unsupported = true;
        }
        return true;
    }

    bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* subscript) {
        if (!subscript) {
            return true;
        }
        if (baseIsTarget(subscript->getBase())) {
            SawTargetSubscript = true;
            if (!isZeroIndex(subscript->getIdx())) {
                Unsupported = true;
            }
            ++SubscriptDepth;
            TraverseStmt(subscript->getBase());
            --SubscriptDepth;
            TraverseStmt(subscript->getIdx());
            return true;
        }
        return clang::RecursiveASTVisitor<
            ScalarOnlyParamVisitor>::TraverseArraySubscriptExpr(subscript);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall) {
            return true;
        }
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 2 && baseIsTarget(opCall->getArg(0))) {
            SawTargetSubscript = true;
            if (!isZeroIndex(opCall->getArg(1))) {
                Unsupported = true;
            }
            ++SubscriptDepth;
            TraverseStmt(opCall->getArg(0));
            --SubscriptDepth;
            TraverseStmt(opCall->getArg(1));
            return true;
        }
        return clang::RecursiveASTVisitor<
            ScalarOnlyParamVisitor>::TraverseCXXOperatorCallExpr(opCall);
    }

    bool supported() const { return SawTargetSubscript && !Unsupported; }

private:
    const clang::ValueDecl* Target = nullptr;
    int SubscriptDepth = 0;
    bool SawTargetSubscript = false;
    bool Unsupported = false;

    const clang::Expr* ignore(const clang::Expr* expr) const {
        return expr ? expr->IgnoreParenImpCasts() : nullptr;
    }

    bool baseIsTarget(const clang::Expr* expr) const {
        expr = ignore(expr);
        if (const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
            return declRef->getDecl() == Target;
        }
        return false;
    }

    bool isZeroIndex(const clang::Expr* expr) const {
        expr = ignore(expr);
        if (const auto* literal = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
            return literal->getValue() == 0;
        }
        return false;
    }
};

const char* shellDimKindName(ShellDimKind kind) {
    switch (kind) {
    case ShellDimKind::Index:
        return "Index";
    case ShellDimKind::Void:
        return "Void";
    case ShellDimKind::Split:
        return "Split";
    }
    return "Unknown";
}

int64_t shapeValueFor(Shell* shell, int paramIdx, int dimIdx) {
    if (!shell || paramIdx < 0 || paramIdx >= shell->getNumParams() ||
        dimIdx < 0) {
        return -1;
    }
    Param* param = shell->getParam(paramIdx);
    if (!param || dimIdx >= param->getDim()) {
        return -1;
    }
    return param->getShape(dimIdx);
}

bool splitIsVoid(Split* split) {
    return !split || split->getId() == "void";
}

bool splitIsIndex(Split* split) {
    return split && split->type == "IndexSplit";
}

bool splitIsRegular(Split* split) {
    return split && split->type == "RegularSplit";
}

void reject(ShellPartitionPlan& plan, const std::string& reason) {
    plan.supported = false;
    plan.rejectReason = reason;
    const std::string shellName =
        plan.exprNode.shell ? plan.exprNode.shell->getName() : "<null>";
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shellName
                 << " rejected reason=" << reason << "\n";
}

bool isScalarVoidParam(Shell* shell, int paramIdx) {
    if (!shell || paramIdx < 0 || paramIdx >= shell->getNumParams()) {
        return false;
    }
    ShellParam* shellParam = shell->getShellParam(paramIdx);
    if (shellParam && shellParam->getDimension() == 1) {
        return true;
    }
    Param* param = shell->getParam(paramIdx);
    if (!param) {
        return false;
    }
    if (param->getDimension() == 1) {
        return true;
    }
    return param->getDim() == 1 && param->getShape(0) == 1;
}

bool calcUsesParamAsScalar(Calc* calc, int paramIdx) {
    if (!calc || !calc->getCalcLoc() || !calc->getCalcLoc()->getBody() ||
        paramIdx < 0 ||
        paramIdx >= static_cast<int>(calc->getCalcLoc()->getNumParams())) {
        return false;
    }
    const clang::ValueDecl* target = calc->getCalcLoc()->getParamDecl(paramIdx);
    ScalarOnlyParamVisitor visitor(target);
    visitor.TraverseStmt(calc->getCalcLoc()->getBody());
    return visitor.supported();
}

bool sameOrder(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    return lhs == rhs;
}

std::vector<int> uniquePreserveOrder(const std::vector<int>& values) {
    std::vector<int> result;
    for (int value : values) {
        if (std::find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    }
    return result;
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
            if (splitIsRegular(split)) {
                reject(plan, "contains regular split");
                return plan;
            }
            if (splitIsVoid(split)) {
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
            if (!splitIsIndex(split)) {
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
                !isScalarVoidParam(shell, paramIdx) ||
                !calcUsesParamAsScalar(calc, paramIdx)) {
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

    plan.signature.bindOrder = uniquePreserveOrder(firstDirectOrder);
    for (int bindId : plan.signature.bindOrder) {
        auto it = bindDomainsById.find(bindId);
        if (it == bindDomainsById.end()) {
            reject(plan, "bind domain missing from signature");
            return plan;
        }
        plan.bindDomains.push_back(it->second);
        plan.signature.bindSizes.push_back(
            shapeValueFor(shell, static_cast<int>(it->second.runtimeSizeParam),
                          it->second.dimId));
    }

    const std::size_t bindRank = plan.signature.bindOrder.size();
    if (bindRank == 1) {
        for (const auto& param : plan.params) {
            if (param.access == ParamAccessKind::ReplicatedScalar) {
                continue;
            }
            if (param.bindOrder.size() != 1 ||
                !sameOrder(param.bindOrder, plan.signature.bindOrder)) {
                reject(plan, "1D direct parameter bind mismatch");
                return plan;
            }
        }
        plan.signature.layout = LocalLayoutKind::Contiguous1D;
        plan.signature.linearization = "1d-linear";
    } else if (bindRank == 2 && !sawScalarParam) {
        for (const auto& param : plan.params) {
            if (param.bindOrder.size() != 2 ||
                !sameOrder(param.bindOrder, plan.signature.bindOrder) ||
                param.tensorDims.size() != 2 ||
                param.tensorDims[0] != 0 ||
                param.tensorDims[1] != 1) {
                reject(plan, "2D parameter is not row-major direct");
                return plan;
            }
        }
        plan.signature.layout = LocalLayoutKind::RowBlock2D;
        plan.signature.linearization = "2d-row-major";
    } else {
        reject(plan, "unsupported bind rank for phase 1/2");
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
                     << " kind=" << shellDimKindName(mapping.kind)
                     << " bind=" << mapping.bindId << "\n";
    }
    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
