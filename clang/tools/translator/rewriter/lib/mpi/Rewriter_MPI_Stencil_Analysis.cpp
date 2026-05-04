#include <set>
#include <string>
#include <vector>
#include <regex>

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Stencil_Common.h"
#include "Rewriter_MPI_PostRegion_Internal.h"

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

bool isEffectiveWriter(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& paramModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(paramModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return paramModes[paramIdx] == IOTYPE::WRITE;
        }
    }
    return false;
}

bool isEffectiveReader(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& paramModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(paramModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return paramModes[paramIdx] == IOTYPE::READ;
        }
    }
    return false;
}

bool tryCollectDistributedFollowup(DistributedStencilSitePlan& plan,
                                   DacppFile* dacppFile,
                                   Shell* shell,
                                   const std::vector<IOTYPE>& paramModes,
                                   const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }

    detail::LoopRegionInfo info;
    if (!detail::extractLoopRegionInfo(
            forStmt, dacppFile->getContext(), shell,
            dacppFile->getBufferRegionPlan(), info)) {
        return false;
    }

    if (info.writtenTensors.size() != 1 || info.readTensors.size() != 1) {
        return false;
    }

    const std::string& readerTensor = *info.writtenTensors.begin();
    const std::string& writerTensor = *info.readTensors.begin();
    if (!isEffectiveWriter(writerTensor, shell, paramModes) ||
        !isEffectiveReader(readerTensor, shell, paramModes)) {
        return false;
    }

    int targetOffset = 0;
    const std::string pattern =
        readerTensor + R"(\s*\[\s*)" + info.loopVar + R"(\s*\]\s*=\s*)" +
        writerTensor + R"(\s*\[\s*)" + info.loopVar +
        R"(\s*([+-])?\s*(\d+)?\s*\]\s*;?)";
    std::smatch match;
    if (!std::regex_match(info.bodyText, match, std::regex(pattern))) {
        return false;
    }
    if (match.size() >= 3 && match[1].matched && match[2].matched) {
        const int rhsOffset = std::stoi(match[2].str());
        targetOffset = match[1].str() == "-" ? rhsOffset : -rhsOffset;
    }

    plan.followupMappings.push_back({writerTensor, readerTensor, targetOffset});
    plan.distributedFollowupStmts.push_back(stmt);
    return true;
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

    std::vector<RootCentricPostRegion> rootRegions;
    const bool allowDistributedFollowup = regionPlan.siblingStmts.size() == 1;
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        if (allowDistributedFollowup &&
            tryCollectDistributedFollowup(plan, dacppFile, shell, paramModes, stmt)) {
            continue;
        }
        if (detail::isRootCentricRegionSupported(dacppFile, shell, stmt)) {
            rootRegions.push_back({stmt, ""});
            continue;
        }
        plan.disableReason =
            "phase-c requires post-shell statements to lower as distributed followup or root-centric helpers";
        return plan;
    }

    if (!rootRegions.empty()) {
        plan.hasRootBridge = true;
        collectRootBridgeTensors(plan, dacppFile, shell);
    }

    plan.supported = true;
    return plan;
}

std::vector<DistributedFollowupRegion> collectDistributedFollowupRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    std::vector<DistributedFollowupRegion> regions;
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return regions;
    }
    for (const clang::Stmt* stmt : plan.distributedFollowupStmts) {
        regions.push_back({stmt});
    }
    return regions;
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
