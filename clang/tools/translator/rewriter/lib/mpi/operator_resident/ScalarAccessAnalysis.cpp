#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "DacppStructure.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

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
        if (const auto* declRef =
                llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
            return declRef->getDecl() == Target;
        }
        return false;
    }

    bool isZeroIndex(const clang::Expr* expr) const {
        expr = ignore(expr);
        if (const auto* literal =
                llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
            return literal->getValue() == 0;
        }
        return false;
    }
};

} // namespace

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

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
