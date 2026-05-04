#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>
#include <regex>
#include <sstream>

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

int findShellParamIndex(Shell* shell, const std::string& tensorName) {
    if (!shell) {
        return -1;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return paramIdx;
        }
    }
    return -1;
}

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string stripOuterBraces(std::string text) {
    text = trim(std::move(text));
    if (text.size() >= 2 && text.front() == '{' && text.back() == '}') {
        text = text.substr(1, text.size() - 2);
    }
    return trim(std::move(text));
}

std::string stripComments(const std::string& text) {
    std::string withoutBlock;
    withoutBlock.reserve(text.size());
    bool inBlockComment = false;
    for (std::size_t idx = 0; idx < text.size(); ++idx) {
        if (!inBlockComment && idx + 1 < text.size() &&
            text[idx] == '/' && text[idx + 1] == '*') {
            inBlockComment = true;
            ++idx;
            continue;
        }
        if (inBlockComment && idx + 1 < text.size() &&
            text[idx] == '*' && text[idx + 1] == '/') {
            inBlockComment = false;
            ++idx;
            continue;
        }
        if (!inBlockComment) {
            withoutBlock.push_back(text[idx]);
        }
    }

    std::string result;
    result.reserve(withoutBlock.size());
    for (std::size_t idx = 0; idx < withoutBlock.size(); ++idx) {
        if (idx + 1 < withoutBlock.size() &&
            withoutBlock[idx] == '/' && withoutBlock[idx + 1] == '/') {
            auto nlPos = withoutBlock.find('\n', idx);
            if (nlPos != std::string::npos) {
                result.push_back('\n');
                idx = nlPos;
            } else {
                idx = withoutBlock.size();
            }
            continue;
        }
        result.push_back(withoutBlock[idx]);
    }
    return result;
}

bool isSupportedIncrement(const clang::ForStmt* forStmt) {
    const auto* inc = forStmt ? forStmt->getInc() : nullptr;
    if (!inc) {
        return false;
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(inc)) {
        return unary->isIncrementOp();
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(inc)) {
        return opCall->getOperator() == clang::OO_PlusPlus;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(inc)) {
        if (!binary->isAssignmentOp()) {
            return false;
        }
        const std::string text = binary->getOpcodeStr().str();
        return text == "+=" || text == "=";
    }
    return false;
}

struct RouteLoopInfo {
    std::string loopVar;
    std::string bodyText;
};

bool extractRouteLoopInfo(const clang::ForStmt* forStmt,
                          clang::ASTContext* context,
                          const BufferRegionPlan& plan,
                          RouteLoopInfo& info) {
    if (!forStmt || !context || !plan.capturedNonShellVars.empty()) {
        return false;
    }

    const auto& sourceManager = context->getSourceManager();
    const auto& langOpts = context->getLangOpts();

    const auto* declStmt =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* loopVarDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!loopVarDecl || !loopVarDecl->getInit()) {
        return false;
    }
    info.loopVar = loopVarDecl->getNameAsString();

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond) {
        return false;
    }
    const std::string lhsText =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getLHS()->getSourceRange()),
            sourceManager, langOpts)
            .str();
    if (trim(lhsText) != info.loopVar ||
        (cond->getOpcode() != clang::BO_LE && cond->getOpcode() != clang::BO_LT)) {
        return false;
    }
    if (!isSupportedIncrement(forStmt)) {
        return false;
    }

    info.bodyText = stripOuterBraces(
        stripComments(getStmtSourceText(forStmt->getBody(), context)));
    if (info.bodyText.empty()) {
        return false;
    }

    const std::vector<std::string> rejected = {
        "for", "while", "switch", "return", "break", "continue", "goto",
        "MPI_", "std::", "cout", "cerr", "printf"};
    for (const auto& token : rejected) {
        if (containsWord(info.bodyText, token)) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> splitSimpleAssignments(const std::string& bodyText) {
    std::vector<std::string> assignments;
    std::stringstream ss(bodyText);
    std::string piece;
    while (std::getline(ss, piece, ';')) {
        std::string trimmed = piece;
        trimmed.erase(trimmed.begin(),
                      std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
                          return !std::isspace(ch);
                      }));
        trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
                          return !std::isspace(ch);
                      }).base(),
                      trimmed.end());
        if (!trimmed.empty()) {
            assignments.push_back(trimmed);
        }
    }
    return assignments;
}

bool parseAffineVectorAccess(const std::string& expr,
                             const std::string& loopVar,
                             std::string& tensorName,
                             int& offset) {
    static const std::regex accessPattern(
        R"(^\s*([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*)\s*([+-])?\s*(\d+)?\s*\]\s*$)");
    std::smatch match;
    if (!std::regex_match(expr, match, accessPattern)) {
        return false;
    }
    if (match[2].str() != loopVar) {
        return false;
    }
    tensorName = match[1].str();
    offset = 0;
    if (match[3].matched || match[4].matched) {
        if (!match[3].matched || !match[4].matched) {
            return false;
        }
        const int value = std::stoi(match[4].str());
        offset = match[3].str() == "-" ? -value : value;
    }
    return true;
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

    RouteLoopInfo info;
    if (!extractRouteLoopInfo(
            forStmt, dacppFile->getContext(),
            dacppFile->getBufferRegionPlan(), info)) {
        return false;
    }

    const std::vector<std::string> assignments =
        splitSimpleAssignments(info.bodyText);
    if (assignments.empty()) {
        return false;
    }

    std::vector<DistributedFollowupMapping> routes;
    routes.reserve(assignments.size());
    for (const std::string& assignment : assignments) {
        const std::size_t eqPos = assignment.find('=');
        if (eqPos == std::string::npos ||
            assignment.find('=', eqPos + 1) != std::string::npos) {
            return false;
        }

        std::string readerTensor;
        std::string writerTensor;
        int readerOffset = 0;
        int writerOffset = 0;
        if (!parseAffineVectorAccess(assignment.substr(0, eqPos), info.loopVar,
                                     readerTensor, readerOffset) ||
            !parseAffineVectorAccess(assignment.substr(eqPos + 1), info.loopVar,
                                     writerTensor, writerOffset)) {
            return false;
        }
        if (!isEffectiveWriter(writerTensor, shell, paramModes) ||
            !isEffectiveReader(readerTensor, shell, paramModes)) {
            return false;
        }
        const int writerIdx = findShellParamIndex(shell, writerTensor);
        const int readerIdx = findShellParamIndex(shell, readerTensor);
        if (writerIdx < 0 || readerIdx < 0) {
            return false;
        }

        DistributedFollowupMapping route;
        route.writerTensor = writerTensor;
        route.readerTensor = readerTensor;
        route.writerParamIndex = writerIdx;
        route.readerParamIndex = readerIdx;
        route.targetOffset = readerOffset - writerOffset;
        routes.push_back(route);
    }

    plan.followupMappings.insert(plan.followupMappings.end(),
                                 routes.begin(), routes.end());
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
