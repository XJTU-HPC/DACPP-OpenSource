#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "clang/AST/ExprCXX.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Stencil_Analysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

namespace {

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
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

bool extractRouteLoopHeaderInfo(const clang::ForStmt* forStmt,
                                clang::ASTContext* context,
                                RouteLoopInfo& info) {
    if (!forStmt || !context) {
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
    info.loopVarDecl = loopVarDecl;

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond) {
        return false;
    }
    const std::string lhsText =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getLHS()->getSourceRange()),
            sourceManager,
            langOpts)
            .str();
    if (trim(lhsText) != info.loopVar ||
        (cond->getOpcode() != clang::BO_LE &&
         cond->getOpcode() != clang::BO_LT)) {
        return false;
    }
    if (!isSupportedIncrement(forStmt)) {
        return false;
    }

    return true;
}

const clang::Expr* ignoreTransparentExpr(const clang::Expr* expr) {
    while (expr) {
        expr = expr->IgnoreParenImpCasts();
        if (const auto* cleanup =
                llvm::dyn_cast_or_null<clang::ExprWithCleanups>(expr)) {
            expr = cleanup->getSubExpr();
            continue;
        }
        if (const auto* materialized =
                llvm::dyn_cast_or_null<clang::MaterializeTemporaryExpr>(expr)) {
            expr = materialized->getSubExpr();
            continue;
        }
        if (const auto* temporary =
                llvm::dyn_cast_or_null<clang::CXXBindTemporaryExpr>(expr)) {
            expr = temporary->getSubExpr();
            continue;
        }
        break;
    }
    return expr;
}

bool parseLoopAffineExpr(const clang::Expr* expr,
                         const clang::VarDecl* loopVarDecl,
                         const std::string& loopVar,
                         int& offset) {
    expr = ignoreTransparentExpr(expr);
    if (!expr || !loopVarDecl) {
        return false;
    }

    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (dre->getDecl() == loopVarDecl) {
            offset = 0;
            return true;
        }
        return false;
    }

    const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(expr);
    if (!binary ||
        (binary->getOpcode() != clang::BO_Add &&
         binary->getOpcode() != clang::BO_Sub)) {
        return false;
    }

    const clang::Expr* lhs = ignoreTransparentExpr(binary->getLHS());
    const clang::Expr* rhs = ignoreTransparentExpr(binary->getRHS());
    const auto* lhsRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(lhs);
    if (!lhsRef || lhsRef->getDecl() != loopVarDecl) {
        return false;
    }

    int literal = 0;
    if (!parseIntegerLiteralExpr(rhs, literal)) {
        return false;
    }

    offset = binary->getOpcode() == clang::BO_Sub ? -literal : literal;
    (void)loopVar;
    return true;
}

bool parseLoopAffineExprForAnyLoop(const clang::Expr* expr,
                                   const RouteLoopInfo& rowInfo,
                                   const RouteLoopInfo& colInfo,
                                   std::string& loopVar,
                                   int& offset) {
    int parsedOffset = 0;
    if (parseLoopAffineExpr(expr, rowInfo.loopVarDecl, rowInfo.loopVar,
                            parsedOffset)) {
        loopVar = rowInfo.loopVar;
        offset = parsedOffset;
        return true;
    }
    if (parseLoopAffineExpr(expr, colInfo.loopVarDecl, colInfo.loopVar,
                            parsedOffset)) {
        loopVar = colInfo.loopVar;
        offset = parsedOffset;
        return true;
    }
    return false;
}

bool parseLoopOrIntegerIndexExpr(const clang::Expr* expr,
                                 const RouteLoopInfo& loopInfo,
                                 bool& usesLoop,
                                 int& offset,
                                 std::string& exprText,
                                 clang::ASTContext* context) {
    int parsedOffset = 0;
    if (parseLoopAffineExpr(expr, loopInfo.loopVarDecl, loopInfo.loopVar,
                            parsedOffset)) {
        usesLoop = true;
        offset = parsedOffset;
        exprText = loopInfo.loopVar;
        if (parsedOffset > 0) {
            exprText += " + " + std::to_string(parsedOffset);
        } else if (parsedOffset < 0) {
            exprText += " - " + std::to_string(-parsedOffset);
        }
        return true;
    }
    if (parseIntegerLiteralExpr(expr, parsedOffset)) {
        usesLoop = false;
        offset = parsedOffset;
        exprText = std::to_string(parsedOffset);
        return true;
    }
    if (context) {
        const std::string text = trim(getStmtSourceText(expr, context));
        if (!text.empty() && !containsWord(text, loopInfo.loopVar)) {
            usesLoop = false;
            offset = 0;
            exprText = text;
            return true;
        }
    }
    return false;
}

const clang::Expr* getSubscriptBaseExpr(const clang::Expr* expr,
                                        const clang::Expr*& indexExpr) {
    expr = ignoreTransparentExpr(expr);
    indexExpr = nullptr;
    if (const auto* subscript = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expr)) {
        indexExpr = subscript->getIdx();
        return subscript->getBase();
    }
    if (const auto* opCall = llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 1) {
            if (const auto* memberCall =
                    llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(opCall)) {
                indexExpr = opCall->getArg(0);
                return memberCall->getImplicitObjectArgument();
            }
            if (opCall->getNumArgs() >= 2) {
                indexExpr = opCall->getArg(1);
                return opCall->getArg(0);
            }
        }
    }
    return nullptr;
}

std::string compactExprText(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(),
                              [](unsigned char c) {
                                  return std::isspace(c) != 0;
                              }),
               text.end());
    return text;
}

std::string stripOuterParens(std::string text) {
    text = compactExprText(text);
    bool changed = true;
    while (changed && text.size() >= 2 && text.front() == '(' &&
           text.back() == ')') {
        int depth = 0;
        changed = true;
        for (std::size_t i = 0; i + 1 < text.size(); ++i) {
            if (text[i] == '(') {
                ++depth;
            } else if (text[i] == ')') {
                --depth;
                if (depth == 0) {
                    changed = false;
                    break;
                }
            }
        }
        if (changed) {
            text = text.substr(1, text.size() - 2);
        }
    }
    return text;
}

bool isOneLessExpr(const std::string& maybeHigh,
                   const std::string& maybeLow) {
    const std::string high = stripOuterParens(maybeHigh);
    const std::string low = stripOuterParens(maybeLow);
    if (high.empty() || low.empty()) {
        return false;
    }
    if (low == high + "-1") {
        return true;
    }
    const auto splitMinusLiteral = [](const std::string& text,
                                      std::string& base,
                                      int& literal) {
        const std::size_t minus = text.rfind('-');
        if (minus == std::string::npos || minus + 1 >= text.size()) {
            return false;
        }
        int value = 0;
        for (std::size_t idx = minus + 1; idx < text.size(); ++idx) {
            if (!std::isdigit(static_cast<unsigned char>(text[idx]))) {
                return false;
            }
            value = value * 10 + (text[idx] - '0');
        }
        base = text.substr(0, minus);
        literal = value;
        return !base.empty();
    };
    std::string highBase;
    std::string lowBase;
    int highLiteral = 0;
    int lowLiteral = 0;
    if (splitMinusLiteral(high, highBase, highLiteral) &&
        splitMinusLiteral(low, lowBase, lowLiteral)) {
        return highBase == lowBase && lowLiteral == highLiteral + 1;
    }
    return false;
}

bool appendSimpleAssignment(const clang::Stmt* stmt,
                            std::vector<RouteAssignment>& assignments) {
    if (const auto* expr = llvm::dyn_cast_or_null<clang::Expr>(stmt)) {
        expr = expr->IgnoreParenImpCasts();
        while (const auto* cleanup =
                   llvm::dyn_cast_or_null<clang::ExprWithCleanups>(expr)) {
            expr = cleanup->getSubExpr()->IgnoreParenImpCasts();
        }
        stmt = expr;
    }

    if (const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(stmt)) {
        if (binary->getOpcode() != clang::BO_Assign) {
            return false;
        }
        assignments.push_back({binary->getLHS(), binary->getRHS(), binary});
        return true;
    }
    if (const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->getOperator() != clang::OO_Equal ||
            opCall->getNumArgs() < 2) {
            return false;
        }
        assignments.push_back({opCall->getArg(0), opCall->getArg(1), opCall});
        return true;
    }
    return false;
}

}  // namespace

bool extractRouteLoopInfo(const clang::ForStmt* forStmt,
                          clang::ASTContext* context,
                          const BufferRegionPlan& plan,
                          RouteLoopInfo& info) {
    if (!plan.capturedNonShellVars.empty()) {
        return false;
    }
    return extractRouteLoopHeaderInfo(forStmt, context, info);
}

bool extractRouteLoopInfo2D(const clang::ForStmt* outerFor,
                            clang::ASTContext* context,
                            const BufferRegionPlan& plan,
                            RouteLoopInfo2D& info) {
    if (!outerFor || !context || !plan.capturedNonShellVars.empty()) {
        return false;
    }
    if (!extractRouteLoopHeaderInfo(outerFor, context, info.row)) {
        return false;
    }
    const clang::Stmt* body = outerFor->getBody();
    if (const auto* compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body)) {
        if (compound->size() != 1) {
            return false;
        }
        body = *compound->body_begin();
    }
    const auto* innerFor = llvm::dyn_cast_or_null<clang::ForStmt>(body);
    if (!innerFor) {
        return false;
    }
    if (!extractRouteLoopHeaderInfo(innerFor, context, info.col)) {
        return false;
    }
    if (info.row.loopVar == info.col.loopVar) {
        return false;
    }
    return true;
}

bool parseIntegerLiteralExpr(const clang::Expr* expr, int& value) {
    expr = ignoreTransparentExpr(expr);
    if (const auto* intLiteral = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
        value = static_cast<int>(intLiteral->getValue().getSExtValue());
        return true;
    }
    return false;
}

bool parseAffineVectorAccessAST(const clang::Expr* expr,
                                const RouteLoopInfo& info,
                                std::string& tensorName,
                                AffineIndex1D& index) {
    const clang::Expr* indexExpr = nullptr;
    const clang::Expr* baseExpr = getSubscriptBaseExpr(expr, indexExpr);
    if (!baseExpr || !indexExpr) {
        return false;
    }

    baseExpr = ignoreTransparentExpr(baseExpr);
    const auto* baseRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }

    int offset = 0;
    if (!parseLoopAffineExpr(indexExpr, info.loopVarDecl, info.loopVar, offset)) {
        return false;
    }

    tensorName = baseRef->getDecl()->getNameAsString();
    index.loopVar = info.loopVar;
    index.offset = offset;
    return true;
}

bool parseAffineMatrixAccessAST(const clang::Expr* expr,
                                const RouteLoopInfo2D& info,
                                std::string& tensorName,
                                AffineIndex2D& index) {
    const clang::Expr* colIndexExpr = nullptr;
    const clang::Expr* rowAccessExpr =
        getSubscriptBaseExpr(expr, colIndexExpr);
    if (!rowAccessExpr || !colIndexExpr) {
        return false;
    }

    const clang::Expr* rowIndexExpr = nullptr;
    const clang::Expr* baseExpr =
        getSubscriptBaseExpr(rowAccessExpr, rowIndexExpr);
    if (!baseExpr || !rowIndexExpr) {
        return false;
    }

    baseExpr = ignoreTransparentExpr(baseExpr);
    const auto* baseRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }

    std::string firstLoop;
    std::string secondLoop;
    int firstOffset = 0;
    int secondOffset = 0;
    if (!parseLoopAffineExprForAnyLoop(rowIndexExpr, info.row, info.col,
                                       firstLoop, firstOffset) ||
        !parseLoopAffineExprForAnyLoop(colIndexExpr, info.row, info.col,
                                       secondLoop, secondOffset)) {
        return false;
    }
    if (firstLoop == secondLoop) {
        return false;
    }

    tensorName = baseRef->getDecl()->getNameAsString();
    index.rowLoopVar = info.row.loopVar;
    index.colLoopVar = info.col.loopVar;
    if (firstLoop == info.row.loopVar && secondLoop == info.col.loopVar) {
        index.rowOffset = firstOffset;
        index.colOffset = secondOffset;
        return true;
    }
    if (firstLoop == info.col.loopVar && secondLoop == info.row.loopVar) {
        index.rowOffset = secondOffset;
        index.colOffset = firstOffset;
        return true;
    }
    return false;
}

bool parseBoundaryMatrixAccessAST(const clang::Expr* expr,
                                  const RouteLoopInfo& loopInfo,
                                  clang::ASTContext* context,
                                  std::string& tensorName,
                                  bool& rowUsesLoop,
                                  int& rowOffset,
                                  std::string& rowExprText,
                                  bool& colUsesLoop,
                                  int& colOffset,
                                  std::string& colExprText) {
    const clang::Expr* colIndexExpr = nullptr;
    const clang::Expr* rowAccessExpr =
        getSubscriptBaseExpr(expr, colIndexExpr);
    if (!rowAccessExpr || !colIndexExpr) {
        return false;
    }

    const clang::Expr* rowIndexExpr = nullptr;
    const clang::Expr* baseExpr =
        getSubscriptBaseExpr(rowAccessExpr, rowIndexExpr);
    if (!baseExpr || !rowIndexExpr) {
        return false;
    }

    baseExpr = ignoreTransparentExpr(baseExpr);
    const auto* baseRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }

    if (!parseLoopOrIntegerIndexExpr(rowIndexExpr, loopInfo,
                                     rowUsesLoop, rowOffset,
                                     rowExprText, context) ||
        !parseLoopOrIntegerIndexExpr(colIndexExpr, loopInfo,
                                     colUsesLoop, colOffset,
                                     colExprText, context)) {
        return false;
    }
    if (rowUsesLoop == colUsesLoop) {
        return false;
    }

    tensorName = baseRef->getDecl()->getNameAsString();
    return true;
}

bool parseBoundaryVectorAccessAST(const clang::Expr* expr,
                                  const RouteLoopInfo& loopInfo,
                                  clang::ASTContext* context,
                                  std::string& tensorName,
                                  bool& usesLoop,
                                  int& offset,
                                  std::string& exprText) {
    const clang::Expr* indexExpr = nullptr;
    const clang::Expr* baseExpr = getSubscriptBaseExpr(expr, indexExpr);
    if (!baseExpr || !indexExpr) {
        return false;
    }

    baseExpr = ignoreTransparentExpr(baseExpr);
    const auto* baseRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }

    if (!parseLoopOrIntegerIndexExpr(indexExpr, loopInfo, usesLoop, offset,
                                     exprText, context)) {
        return false;
    }

    tensorName = baseRef->getDecl()->getNameAsString();
    return true;
}

bool isBoundaryCopyPair(bool targetUsesLoop,
                        int targetOffset,
                        const std::string& targetExpr,
                        bool sourceUsesLoop,
                        int sourceOffset,
                        const std::string& sourceExpr) {
    if (targetUsesLoop || sourceUsesLoop) {
        return targetUsesLoop && sourceUsesLoop &&
               targetOffset == 0 && sourceOffset == 0;
    }
    if (targetExpr == "0" && sourceExpr == "1") {
        return true;
    }
    return isOneLessExpr(targetExpr, sourceExpr);
}

bool isBoundaryConstantTarget(bool targetUsesLoop,
                              int targetOffset,
                              const std::string& targetExpr,
                              bool otherDimUsesLoop) {
    if (targetUsesLoop) {
        return targetOffset == 0 && !otherDimUsesLoop;
    }
    return targetExpr == "0" ||
           stripOuterParens(targetExpr).find("-1") != std::string::npos;
}

bool collectTopLevelAssignments(const clang::Stmt* stmt,
                                std::vector<RouteAssignment>& assignments) {
    if (!stmt) {
        return false;
    }

    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        for (const clang::Stmt* child : compound->body()) {
            if (!appendSimpleAssignment(child, assignments)) {
                return false;
            }
        }
        return !assignments.empty();
    }

    return appendSimpleAssignment(stmt, assignments);
}

bool collectNestedLoopAssignments2D(const clang::ForStmt* outerFor,
                                    std::vector<RouteAssignment>& assignments) {
    if (!outerFor) {
        return false;
    }
    const clang::Stmt* body = outerFor->getBody();
    if (const auto* compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body)) {
        if (compound->size() != 1) {
            return false;
        }
        body = *compound->body_begin();
    }
    const auto* innerFor = llvm::dyn_cast_or_null<clang::ForStmt>(body);
    if (!innerFor) {
        return false;
    }
    return collectTopLevelAssignments(innerFor->getBody(), assignments);
}

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
