#include <set>
#include <string>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Stencil_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

std::string getStmtSourceText(const clang::Stmt* stmt,
                              clang::ASTContext* context) {
    if (!stmt || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
               context->getSourceManager(),
               context->getLangOpts())
        .str();
}

bool isWordBoundary(char c) {
    return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_');
}

bool containsWord(const std::string& text, const std::string& word) {
    if (word.empty() || text.empty() || word.size() > text.size()) {
        return false;
    }
    std::size_t pos = 0;
    while ((pos = text.find(word, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || isWordBoundary(text[pos - 1]);
        const std::size_t rightIdx = pos + word.size();
        const bool rightOk = rightIdx >= text.size() || isWordBoundary(text[rightIdx]);
        if (leftOk && rightOk) {
            return true;
        }
        ++pos;
    }
    return false;
}

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr) {
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall) {
        return shellParamName;
    }

    const clang::FunctionDecl* callee = shellCall->getDirectCallee();
    if (!callee) {
        return shellParamName;
    }

    for (unsigned paramIdx = 0;
         paramIdx < callee->getNumParams() && paramIdx < shellCall->getNumArgs();
         ++paramIdx) {
        const clang::ParmVarDecl* param = callee->getParamDecl(paramIdx);
        if (!param || param->getNameAsString() != shellParamName) {
            continue;
        }

        const auto* dre = dacppTranslator::getNode<clang::DeclRefExpr>(
            const_cast<clang::Expr*>(shellCall->getArg(paramIdx)));
        if (dre && dre->getDecl()) {
            return dre->getDecl()->getNameAsString();
        }
    }
    return shellParamName;
}

bool isVectorParam(Shell* shell, Calc* calc, int paramIdx) {
    if (!shell || !calc || paramIdx < 0 ||
        paramIdx >= shell->getNumShellParams() ||
        paramIdx >= calc->getNumParams()) {
        return false;
    }
    const std::string shellType = shell->getParam(paramIdx)->getType();
    const std::string calcType = calc->getParam(paramIdx)->getType();
    return shellType.find("Vector<") != std::string::npos ||
           calcType.find("Vector<") != std::string::npos;
}

void collectRootBridgeTensors(DistributedStencilSitePlan& plan,
                              DacppFile* dacppFile,
                              Shell* shell) {
    if (!dacppFile || !shell || !dacppFile->getContext()) {
        return;
    }
    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        const std::string stmtText = getStmtSourceText(stmt, dacppFile->getContext());
        for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
            const std::string paramName = shell->getParam(paramIdx)->getName();
            if (!containsWord(stmtText, paramName)) {
                continue;
            }
            plan.rootBridgeTensors.insert(paramName);
        }
    }
}

}  // namespace

DistributedStencilSitePlan analyzeDistributedStencilSite(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    DistributedStencilSitePlan plan;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        plan.disableReason = "missing shell/calc site";
        return plan;
    }

    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    if (!regionPlan.enabled || regionPlan.dacExpr != dacExpr) {
        plan.disableReason = "phase-c requires rewriteMPIStencil loop lowering";
        return plan;
    }

    const auto paramModes = inferEffectiveParamModes(shell, calc);
    if (paramModes.size() != static_cast<std::size_t>(shell->getNumShellParams())) {
        plan.disableReason = "failed to infer effective parameter modes";
        return plan;
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (!isVectorParam(shell, calc, paramIdx)) {
            plan.disableReason = "phase-c only supports 1D dacpp::Vector tensors";
            return plan;
        }
        if (paramModes[paramIdx] == IOTYPE::READ_WRITE) {
            plan.disableReason = "phase-c does not support READ_WRITE kernel params";
            return plan;
        }
        plan.distributedTensors.insert(resolveActualTensorName(
            shell->getParam(paramIdx)->getName(), dacExpr));
    }

    const auto rootRegions =
        collectRootCentricPostRegions(dacppFile, shell, calc, dacExpr);
    if (!regionPlan.siblingStmts.empty() &&
        rootRegions.size() != regionPlan.siblingStmts.size()) {
        plan.disableReason =
            "phase-c requires all post-shell sibling statements to lower as root-centric helpers";
        return plan;
    }

    if (!rootRegions.empty()) {
        plan.hasRootBridge = true;
        collectRootBridgeTensors(plan, dacppFile, shell);
    }

    plan.supported = true;
    return plan;
}

bool tensorUsesDistributedFollowup(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const std::string& tensorName,
    const clang::BinaryOperator* dacExpr) {
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return false;
    }
    const std::string actualName = resolveActualTensorName(tensorName, dacExpr);
    return plan.distributedTensors.count(actualName) != 0;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
