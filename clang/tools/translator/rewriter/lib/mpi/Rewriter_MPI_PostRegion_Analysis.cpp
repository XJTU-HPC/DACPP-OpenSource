#include <set>
#include <string>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_PostRegion_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

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
        const bool rightOk =
            rightIdx >= text.size() || isWordBoundary(text[rightIdx]);
        if (leftOk && rightOk) {
            return true;
        }
        ++pos;
    }
    return false;
}

bool isVectorParam(Param* param) {
    return param && param->getType().find("Vector<") != std::string::npos;
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

}  // namespace

namespace detail {

bool extractLoopRegionInfo(const clang::ForStmt* forStmt,
                           clang::ASTContext* context,
                           Shell* shell,
                           const BufferRegionPlan& plan,
                           LoopRegionInfo& info) {
    if (!forStmt || !context || !shell) {
        return false;
    }
    if (!plan.capturedNonShellVars.empty()) {
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
    info.lowerExpr =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(
                loopVarDecl->getInit()->getSourceRange()),
            sourceManager, langOpts)
            .str();

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
    if (trim(lhsText) != info.loopVar) {
        return false;
    }
    if (cond->getOpcode() == clang::BO_LE) {
        info.upperInclusive = true;
    } else if (cond->getOpcode() == clang::BO_LT) {
        info.upperInclusive = false;
    } else {
        return false;
    }
    info.upperExpr =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getRHS()->getSourceRange()),
            sourceManager, langOpts)
            .str();

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

    const std::size_t eqPos = info.bodyText.find('=');
    if (eqPos == std::string::npos ||
        info.bodyText.find('=', eqPos + 1) != std::string::npos) {
        return false;
    }
    if ((eqPos > 0 &&
         (info.bodyText[eqPos - 1] == '+' || info.bodyText[eqPos - 1] == '-' ||
          info.bodyText[eqPos - 1] == '*' || info.bodyText[eqPos - 1] == '/' ||
          info.bodyText[eqPos - 1] == '%' || info.bodyText[eqPos - 1] == '=' ||
          info.bodyText[eqPos - 1] == '!' || info.bodyText[eqPos - 1] == '<' ||
          info.bodyText[eqPos - 1] == '>')) ||
        (eqPos + 1 < info.bodyText.size() && info.bodyText[eqPos + 1] == '=')) {
        return false;
    }

    const std::string lhs = info.bodyText.substr(0, eqPos);
    const std::string rhs = info.bodyText.substr(eqPos + 1);

    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        const std::string name = shell->getParam(paramIdx)->getName();
        if (containsWord(lhs, name)) {
            if (!isVectorParam(shell->getParam(paramIdx))) {
                return false;
            }
            info.writtenTensors.insert(name);
        }
        if (containsWord(rhs, name)) {
            if (!isVectorParam(shell->getParam(paramIdx))) {
                return false;
            }
            info.readTensors.insert(name);
        }
    }

    return !info.writtenTensors.empty() &&
           (!info.readTensors.empty() ||
            info.bodyText.find('[') != std::string::npos);
}

bool isRootCentricRegionSupported(DacppFile* dacppFile,
                                  Shell* shell,
                                  const clang::Stmt* stmt) {
    if (!dacppFile || !shell || !stmt) {
        return false;
    }
    const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }
    LoopRegionInfo info;
    return extractLoopRegionInfo(
        forStmt, dacppFile->getContext(), shell,
        dacppFile->getBufferRegionPlan(), info);
}

std::string helperBaseName(Shell* shell, Calc* calc) {
    return shell->getName() + "_" + calc->getName();
}

std::string helperNameFor(Shell* shell, Calc* calc, std::size_t stmtIdx) {
    return "__dacpp_mpi_region_" + helperBaseName(shell, calc) +
           "_stmt_" + std::to_string(stmtIdx);
}

}  // namespace detail

std::vector<RootCentricPostRegion> collectRootCentricPostRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    std::vector<RootCentricPostRegion> result;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        return result;
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.dacExpr != dacExpr) {
        return result;
    }

    std::set<const clang::Stmt*> distributedFollowupStmts;
    const auto distributedRegions =
        collectDistributedFollowupRegions(dacppFile, shell, calc, dacExpr);
    for (const auto& region : distributedRegions) {
        distributedFollowupStmts.insert(region.stmt);
    }

    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        const clang::Stmt* stmt = plan.siblingStmts[stmtIdx];
        if (distributedFollowupStmts.count(stmt) != 0) {
            continue;
        }
        if (!detail::isRootCentricRegionSupported(dacppFile, shell, stmt)) {
            continue;
        }
        result.push_back({stmt, detail::helperNameFor(shell, calc, stmtIdx)});
    }
    return result;
}

std::vector<const clang::Stmt*> collectRootCentricPostRegionStmts(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr) {
    std::vector<const clang::Stmt*> result;
    if (!dacppFile || !dacExpr || !dacppFile->getContext()) {
        return result;
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.dacExpr != dacExpr) {
        return result;
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        if (!expr || expr->getDacExpr() != dacExpr) {
            continue;
        }
        Shell* shell = expr->getShell();
        for (const clang::Stmt* stmt : plan.siblingStmts) {
            if (detail::isRootCentricRegionSupported(dacppFile, shell, stmt)) {
                result.push_back(stmt);
            }
        }
        break;
    }

    return result;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
