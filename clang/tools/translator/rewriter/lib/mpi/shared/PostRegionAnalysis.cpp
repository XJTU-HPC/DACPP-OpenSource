#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "PostRegion_Internal.h"

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

bool stmtContainsCoutExpr(const clang::Stmt* stmt) {
    if (!stmt) {
        return false;
    }
    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        return dre->getDecl() && dre->getDecl()->getNameAsString() == "cout";
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtContainsCoutExpr(child)) {
            return true;
        }
    }
    return false;
}

bool isVectorParam(Param* param) {
    return param && param->getType().find("Vector<") != std::string::npos;
}

const clang::Expr* ignoreParenCasts(const clang::Expr* expr) {
    return expr ? expr->IgnoreParenImpCasts() : nullptr;
}

const clang::Expr* unwrapConstExpr(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }
    while (true) {
        const clang::Expr* next = expr->IgnoreParenImpCasts();
        if (const auto* cleanups =
                llvm::dyn_cast<clang::ExprWithCleanups>(next)) {
            next = cleanups->getSubExpr();
        } else if (const auto* materialized =
                       llvm::dyn_cast<clang::MaterializeTemporaryExpr>(next)) {
            next = materialized->getSubExpr();
        } else if (const auto* temporary =
                       llvm::dyn_cast<clang::CXXBindTemporaryExpr>(next)) {
            next = temporary->getSubExpr();
        } else if (const auto* construct =
                       llvm::dyn_cast<clang::CXXConstructExpr>(next)) {
            if (construct->getNumArgs() == 1) {
                next = construct->getArg(0);
            } else {
                return next;
            }
        }
        if (!next || next == expr) {
            return next;
        }
        expr = next;
    }
}

std::string declRefName(const clang::Expr* expr) {
    expr = unwrapConstExpr(expr);
    const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
    if (!ref || !ref->getDecl()) {
        return "";
    }
    return ref->getDecl()->getNameAsString();
}

bool isIntegerLiteralValue(const clang::Expr* expr, int64_t expected) {
    expr = ignoreParenCasts(expr);
    const auto* literal = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr);
    if (!literal) {
        return false;
    }
    return literal->getValue().getSExtValue() == expected;
}

bool evaluateIntegerConstant(const clang::Expr* expr,
                             clang::ASTContext* context,
                             int64_t& value) {
    expr = unwrapConstExpr(expr);
    if (!expr || !context || !expr->getType()->isIntegerType()) {
        return false;
    }
    if (const auto* literal = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        value = literal->getValue().getSExtValue();
        return true;
    }
    clang::Expr::EvalResult evalResult;
    if (!expr->EvaluateAsInt(evalResult, *context) ||
        !evalResult.Val.isInt()) {
        return false;
    }
    value = evalResult.Val.getInt().getSExtValue();
    return true;
}

std::vector<const clang::Expr*> collectSubscriptIndices(
    const clang::Expr* expr,
    std::string& baseName) {
    std::vector<const clang::Expr*> reversed;
    const clang::Expr* current = unwrapConstExpr(expr);
    while (current) {
        if (const auto* opCall =
                llvm::dyn_cast<clang::CXXOperatorCallExpr>(current)) {
            if (opCall->getOperator() != clang::OO_Subscript ||
                opCall->getNumArgs() < 2) {
                break;
            }
            reversed.push_back(opCall->getArg(1));
            current = unwrapConstExpr(opCall->getArg(0));
            continue;
        }
        if (const auto* array =
                llvm::dyn_cast<clang::ArraySubscriptExpr>(current)) {
            reversed.push_back(array->getIdx());
            current = unwrapConstExpr(array->getBase());
            continue;
        }
        baseName = declRefName(current);
        break;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

bool sameIndex(const PostUseBoundedIndex& lhs,
               const PostUseBoundedIndex& rhs) {
    return lhs.indices == rhs.indices;
}

void addUniqueIndex(std::vector<PostUseBoundedIndex>& indices,
                    const PostUseBoundedIndex& index) {
    for (const auto& existing : indices) {
        if (sameIndex(existing, index)) {
            return;
        }
    }
    indices.push_back(index);
}

bool stmtRangeContains(const clang::SourceManager& sourceManager,
                       const clang::Stmt* outer,
                       const clang::Stmt* inner) {
    if (!outer || !inner ||
        outer->getSourceRange().isInvalid() ||
        inner->getSourceRange().isInvalid()) {
        return false;
    }
    const auto beforeOrEqual = [&](clang::SourceLocation lhs,
                                   clang::SourceLocation rhs) {
        return lhs == rhs ||
               sourceManager.isBeforeInTranslationUnit(lhs, rhs);
    };
    return beforeOrEqual(outer->getBeginLoc(), inner->getBeginLoc()) &&
           beforeOrEqual(inner->getEndLoc(), outer->getEndLoc());
}

bool containsDacExpr(const clang::Stmt* stmt,
                     const clang::BinaryOperator* dacExpr,
                     clang::ASTContext* context) {
    if (!stmt || !dacExpr || !context) {
        return false;
    }
    return stmtRangeContains(context->getSourceManager(), stmt, dacExpr);
}

bool isIgnoredLoweredPostStmt(const clang::Stmt* stmt,
                              const std::set<const clang::Stmt*>& ignored,
                              clang::ASTContext* context) {
    if (!stmt || !context) {
        return false;
    }
    for (const clang::Stmt* ignoredStmt : ignored) {
        if (ignoredStmt &&
            stmtRangeContains(context->getSourceManager(), stmt, ignoredStmt)) {
            return true;
        }
    }
    return false;
}

const clang::CXXMemberCallExpr* tensor2ArrayCallForStmt(
    const clang::Stmt* stmt,
    const std::string& tensorName,
    std::string& targetName) {
    targetName.clear();
    const auto* exprStmt = llvm::dyn_cast_or_null<clang::Expr>(stmt);
    exprStmt = unwrapConstExpr(exprStmt);
    const auto* call =
        llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(exprStmt);
    if (!call || call->getNumArgs() != 1) {
        return nullptr;
    }
    const clang::CXXMethodDecl* method = call->getMethodDecl();
    if (!method || method->getNameAsString() != "tensor2Array") {
        return nullptr;
    }
    if (declRefName(call->getImplicitObjectArgument()) != tensorName) {
        return nullptr;
    }
    targetName = declRefName(call->getArg(0));
    if (targetName.empty()) {
        return nullptr;
    }
    return call;
}

constexpr int64_t kMaxTensor2ArrayBoundedIndices = 4096;

bool extractSimplePrefixLoop(const clang::ForStmt* loop,
                             clang::ASTContext* context,
                             std::string& loopVarName,
                             int64_t& endExclusive) {
    loopVarName.clear();
    endExclusive = 0;
    if (!loop || !context) {
        return false;
    }

    const clang::Stmt* init = loop->getInit();
    if (const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(init)) {
        if (!declStmt->isSingleDecl()) {
            return false;
        }
        const auto* varDecl =
            llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
        int64_t initValue = 0;
        if (!varDecl || !varDecl->hasInit() ||
            !evaluateIntegerConstant(varDecl->getInit(), context, initValue) ||
            initValue != 0) {
            return false;
        }
        loopVarName = varDecl->getNameAsString();
    } else if (const auto* binary =
                   llvm::dyn_cast_or_null<clang::BinaryOperator>(init)) {
        int64_t initValue = 0;
        if (binary->getOpcode() != clang::BO_Assign ||
            !evaluateIntegerConstant(binary->getRHS(), context, initValue) ||
            initValue != 0) {
            return false;
        }
        loopVarName = declRefName(binary->getLHS());
    }
    if (loopVarName.empty()) {
        return false;
    }

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(ignoreParenCasts(loop->getCond()));
    if (!cond || declRefName(cond->getLHS()) != loopVarName) {
        return false;
    }
    int64_t bound = 0;
    if (!evaluateIntegerConstant(cond->getRHS(), context, bound) ||
        bound < 0) {
        return false;
    }
    if (cond->getOpcode() == clang::BO_LT) {
        endExclusive = bound;
    } else if (cond->getOpcode() == clang::BO_LE) {
        if (bound == std::numeric_limits<int64_t>::max()) {
            return false;
        }
        endExclusive = bound + 1;
    } else {
        return false;
    }
    if (endExclusive < 0 ||
        endExclusive > kMaxTensor2ArrayBoundedIndices) {
        return false;
    }

    const clang::Stmt* inc = loop->getInc();
    if (const auto* unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(inc)) {
        return unary->isIncrementOp() &&
               declRefName(unary->getSubExpr()) == loopVarName;
    }
    if (const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(inc)) {
        return opCall->getOperator() == clang::OO_PlusPlus &&
               opCall->getNumArgs() > 0 &&
               declRefName(opCall->getArg(0)) == loopVarName;
    }
    if (const auto* compound =
            llvm::dyn_cast_or_null<clang::CompoundAssignOperator>(inc)) {
        return compound->getOpcode() == clang::BO_AddAssign &&
               declRefName(compound->getLHS()) == loopVarName &&
               isIntegerLiteralValue(compound->getRHS(), 1);
    }
    return false;
}

class VectorFixedUseVisitor
    : public clang::RecursiveASTVisitor<VectorFixedUseVisitor> {
public:
    std::string VectorName;
    clang::ASTContext* Context = nullptr;
    int WriteDepth = 0;
    bool FullUse = false;
    std::string Reason;
    std::vector<PostUseBoundedIndex> Indices;
    std::unordered_map<std::string, std::vector<int64_t>> LoopPrefixes;

    VectorFixedUseVisitor(std::string vectorName,
                          clang::ASTContext* context)
        : VectorName(std::move(vectorName)), Context(context) {}

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary) {
            return true;
        }
        if (binary->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(binary->getLHS());
            --WriteDepth;
            if (exprContainsVectorName(binary->getLHS())) {
                recordFull("tensor2Array target vector is written");
            }
            TraverseStmt(binary->getRHS());
            return true;
        }
        return clang::RecursiveASTVisitor<
            VectorFixedUseVisitor>::TraverseBinaryOperator(binary);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* binary) {
        if (!binary) {
            return true;
        }
        ++WriteDepth;
        TraverseStmt(binary->getLHS());
        --WriteDepth;
        if (exprContainsVectorName(binary->getLHS())) {
            recordFull("tensor2Array target vector compound write");
        }
        TraverseStmt(binary->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (!unary) {
            return true;
        }
        if (unary->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(unary->getSubExpr());
            --WriteDepth;
            if (exprContainsVectorName(unary->getSubExpr())) {
                recordFull("tensor2Array target vector increment/decrement");
            }
            return true;
        }
        if (unary->getOpcode() == clang::UO_AddrOf &&
            exprContainsVectorName(unary->getSubExpr())) {
            recordFull("address-of tensor2Array target vector");
            return true;
        }
        return clang::RecursiveASTVisitor<
            VectorFixedUseVisitor>::TraverseUnaryOperator(unary);
    }

    bool TraverseForStmt(clang::ForStmt* loop) {
        if (!loop) {
            return true;
        }
        std::string loopVarName;
        int64_t endExclusive = 0;
        if (!extractSimplePrefixLoop(loop, Context, loopVarName,
                                     endExclusive)) {
            return clang::RecursiveASTVisitor<
                VectorFixedUseVisitor>::TraverseForStmt(loop);
        }
        LoopPrefixes[loopVarName].push_back(endExclusive);
        TraverseStmt(loop->getBody());
        LoopPrefixes[loopVarName].pop_back();
        if (LoopPrefixes[loopVarName].empty()) {
            LoopPrefixes.erase(loopVarName);
        }
        return true;
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall) {
            return true;
        }
        if (opCall->getOperator() == clang::OO_Subscript) {
            if (WriteDepth == 0) {
                std::string baseName;
                (void)collectSubscriptIndices(opCall, baseName);
                if (baseName == VectorName) {
                    recordVectorRead(opCall);
                    return true;
                }
            }
        }
        if (opCall->isAssignmentOp()) {
            if (opCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(opCall->getArg(0));
                --WriteDepth;
                if (exprContainsVectorName(opCall->getArg(0))) {
                    recordFull("tensor2Array target vector operator write");
                }
            }
            for (unsigned argIdx = 1; argIdx < opCall->getNumArgs(); ++argIdx) {
                TraverseStmt(opCall->getArg(argIdx));
            }
            return true;
        }
        if (opCall->getOperator() == clang::OO_PlusPlus ||
            opCall->getOperator() == clang::OO_MinusMinus) {
            if (opCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(opCall->getArg(0));
                --WriteDepth;
                if (exprContainsVectorName(opCall->getArg(0))) {
                    recordFull("tensor2Array target vector operator increment");
                }
            }
            return true;
        }
        return clang::RecursiveASTVisitor<
            VectorFixedUseVisitor>::TraverseCXXOperatorCallExpr(opCall);
    }

    bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* array) {
        if (!array) {
            return true;
        }
        if (WriteDepth == 0) {
            std::string baseName;
            (void)collectSubscriptIndices(array, baseName);
            if (baseName == VectorName) {
                recordVectorRead(array);
                return true;
            }
        }
        return clang::RecursiveASTVisitor<
            VectorFixedUseVisitor>::TraverseArraySubscriptExpr(array);
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (!dre || !dre->getDecl() ||
            dre->getDecl()->getNameAsString() != VectorName ||
            WriteDepth > 0) {
            return true;
        }
        PostUseBoundedIndex ignored;
        if (isPartOfFixedVectorIndexRead(dre, ignored)) {
            return true;
        }
        recordFull("tensor2Array target vector read is not fixed-index");
        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call) {
            return true;
        }
        if (declRefName(call->getImplicitObjectArgument()) == VectorName) {
            recordFull("member call on tensor2Array target vector");
        }
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!call) {
            return true;
        }
        for (const clang::Expr* arg : call->arguments()) {
            if (exprContainsVectorName(arg) &&
                !exprContainsOnlyConstVectorIndexReads(arg)) {
                recordFull("tensor2Array target vector passed to function");
                break;
            }
        }
        return true;
    }

private:
    bool vectorReadExprIsFixed(const clang::Expr* expr) const {
        std::string baseName;
        const std::vector<const clang::Expr*> indexExprs =
            collectSubscriptIndices(expr, baseName);
        if (baseName != VectorName || indexExprs.size() != 1) {
            return false;
        }
        int64_t value = 0;
        if (evaluateIntegerConstant(indexExprs[0], Context, value) &&
            value >= 0) {
            return true;
        }
        const std::string loopVar = declRefName(indexExprs[0]);
        const auto it = LoopPrefixes.find(loopVar);
        return it != LoopPrefixes.end() && !it->second.empty();
    }

    void recordVectorRead(const clang::Expr* expr) {
        std::string baseName;
        const std::vector<const clang::Expr*> indexExprs =
            collectSubscriptIndices(expr, baseName);
        if (baseName != VectorName || indexExprs.size() != 1) {
            recordFull("unsupported tensor2Array target indexing");
            return;
        }
        int64_t value = 0;
        if (evaluateIntegerConstant(indexExprs[0], Context, value) &&
            value >= 0) {
            PostUseBoundedIndex index;
            index.indices = {value};
            addUniqueIndex(Indices, index);
            return;
        }
        const std::string loopVar = declRefName(indexExprs[0]);
        const auto it = LoopPrefixes.find(loopVar);
        if (it == LoopPrefixes.end() || it->second.empty()) {
            recordFull("non-constant tensor2Array target index");
            return;
        }
        const int64_t endExclusive = it->second.back();
        for (int64_t idx = 0; idx < endExclusive; ++idx) {
            PostUseBoundedIndex index;
            index.indices = {idx};
            addUniqueIndex(Indices, index);
        }
    }

    void recordFull(const std::string& reason) {
        if (!FullUse) {
            Reason = reason;
        }
        FullUse = true;
    }

    bool exprContainsVectorName(const clang::Expr* expr) const {
        if (!expr) {
            return false;
        }
        std::string baseName;
        (void)collectSubscriptIndices(expr, baseName);
        if (baseName == VectorName || declRefName(expr) == VectorName) {
            return true;
        }
        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (exprContainsVectorName(childExpr)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool exprContainsOnlyConstVectorIndexReads(const clang::Expr* expr) const {
        if (!expr) {
            return true;
        }
        std::string baseName;
        (void)collectSubscriptIndices(expr, baseName);
        if (baseName == VectorName) {
            return vectorReadExprIsFixed(expr);
        }
        if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(
                ignoreParenCasts(expr))) {
            if (dre->getDecl() &&
                dre->getDecl()->getNameAsString() == VectorName) {
                PostUseBoundedIndex ignored;
                return isPartOfFixedVectorIndexRead(dre, ignored);
            }
        }
        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (!exprContainsOnlyConstVectorIndexReads(childExpr)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool isPartOfFixedVectorIndexRead(const clang::DeclRefExpr* dre,
                                      PostUseBoundedIndex& index) const {
        if (!dre || !Context) {
            return false;
        }
        clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
        for (int depth = 0; depth < 8; ++depth) {
            auto parents = Context->getParents(current);
            if (parents.empty()) {
                return false;
            }
            const clang::DynTypedNode& parent = parents[0];
            if (const auto* expr = parent.get<clang::Expr>()) {
                if (vectorReadExprIsFixed(expr)) {
                    return true;
                }
                current = parent;
                continue;
            }
            return false;
        }
        return false;
    }
};

class PostUseIndexedReadVisitor
    : public clang::RecursiveASTVisitor<PostUseIndexedReadVisitor> {
public:
    std::string TensorName;
    clang::ASTContext* Context = nullptr;
    int WriteDepth = 0;
    bool FullUse = false;
    std::string Reason;
    std::vector<PostUseBoundedIndex> Indices;
    std::unordered_map<std::string, std::vector<int64_t>> LoopPrefixes;

    PostUseIndexedReadVisitor(std::string tensorName,
                              clang::ASTContext* context)
        : TensorName(std::move(tensorName)), Context(context) {}

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary) {
            return true;
        }
        if (binary->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(binary->getLHS());
            --WriteDepth;
            TraverseStmt(binary->getRHS());
            return true;
        }
        return clang::RecursiveASTVisitor<
            PostUseIndexedReadVisitor>::TraverseBinaryOperator(binary);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* binary) {
        if (!binary) {
            return true;
        }
        ++WriteDepth;
        TraverseStmt(binary->getLHS());
        --WriteDepth;
        if (exprContainsTensorName(binary->getLHS())) {
            recordFull("compound assignment reads tensor");
        }
        TraverseStmt(binary->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (!unary) {
            return true;
        }
        if (unary->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(unary->getSubExpr());
            --WriteDepth;
            if (exprContainsTensorName(unary->getSubExpr())) {
                recordFull("increment/decrement reads tensor");
            }
            return true;
        }
        if (unary->getOpcode() == clang::UO_AddrOf &&
            exprContainsTensorName(unary->getSubExpr())) {
            recordFull("address-of tensor escape");
            return true;
        }
        return clang::RecursiveASTVisitor<
            PostUseIndexedReadVisitor>::TraverseUnaryOperator(unary);
    }

    bool TraverseForStmt(clang::ForStmt* loop) {
        if (!loop) {
            return true;
        }
        std::string loopVarName;
        int64_t endExclusive = 0;
        if (!extractSimplePrefixLoop(loop, Context, loopVarName,
                                     endExclusive)) {
            return clang::RecursiveASTVisitor<
                PostUseIndexedReadVisitor>::TraverseForStmt(loop);
        }
        LoopPrefixes[loopVarName].push_back(endExclusive);
        TraverseStmt(loop->getBody());
        LoopPrefixes[loopVarName].pop_back();
        if (LoopPrefixes[loopVarName].empty()) {
            LoopPrefixes.erase(loopVarName);
        }
        return true;
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall) {
            return true;
        }
        if (opCall->getOperator() == clang::OO_Subscript) {
            if (WriteDepth == 0) {
                std::string baseName;
                (void)collectSubscriptIndices(opCall, baseName);
                if (baseName == TensorName) {
                    if (!recordTensorRead(opCall)) {
                        recordFull("non-constant indexed read");
                    }
                    return true;
                }
            }
        }
        if (opCall->getOperator() == clang::OO_LessLess) {
            for (unsigned argIdx = 0; argIdx < opCall->getNumArgs(); ++argIdx) {
                TraverseStmt(opCall->getArg(argIdx));
            }
            return true;
        }
        if (opCall->isAssignmentOp()) {
            if (opCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(opCall->getArg(0));
                --WriteDepth;
                if (opCall->getOperator() != clang::OO_Equal &&
                    exprContainsTensorName(opCall->getArg(0))) {
                    recordFull("compound operator assignment reads tensor");
                }
            }
            for (unsigned argIdx = 1; argIdx < opCall->getNumArgs(); ++argIdx) {
                TraverseStmt(opCall->getArg(argIdx));
            }
            return true;
        }
        return clang::RecursiveASTVisitor<
            PostUseIndexedReadVisitor>::TraverseCXXOperatorCallExpr(opCall);
    }

    bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* array) {
        if (!array) {
            return true;
        }
        if (WriteDepth == 0) {
            std::string baseName;
            (void)collectSubscriptIndices(array, baseName);
            if (baseName == TensorName) {
                if (!recordTensorRead(array)) {
                    recordFull("non-constant indexed read");
                }
                return true;
            }
        }
        return clang::RecursiveASTVisitor<
            PostUseIndexedReadVisitor>::TraverseArraySubscriptExpr(array);
    }

    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* array) {
        if (WriteDepth == 0) {
            std::string baseName;
            (void)collectSubscriptIndices(array, baseName);
            if (baseName == TensorName) {
                if (!recordTensorRead(array)) {
                    recordFull("non-constant indexed read");
                }
            }
        }
        return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (!dre || !dre->getDecl() ||
            dre->getDecl()->getNameAsString() != TensorName) {
            return true;
        }
        if (WriteDepth > 0) {
            return true;
        }
        PostUseBoundedIndex ignored;
        if (isPartOfConstTensorIndexRead(dre, ignored)) {
            return true;
        }
        recordFull("tensor read is not a bounded indexed root read");
        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call) {
            return true;
        }
        const clang::Expr* object = call->getImplicitObjectArgument();
        if (declRefName(object) == TensorName) {
            recordFull("member call on tensor");
        }
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!call) {
            return true;
        }
        for (const clang::Expr* arg : call->arguments()) {
            if (exprContainsTensorName(arg) &&
                !exprContainsOnlyConstTensorIndexReads(arg)) {
                recordFull("tensor passed to function");
                break;
            }
        }
        return true;
    }

private:
    void recordFull(const std::string& reason) {
        if (!FullUse) {
            Reason = reason;
        }
        FullUse = true;
    }

    bool exprContainsTensorName(const clang::Expr* expr) const {
        if (!expr) {
            return false;
        }
        std::string baseName;
        (void)collectSubscriptIndices(expr, baseName);
        if (baseName == TensorName) {
            return true;
        }
        if (declRefName(expr) == TensorName) {
            return true;
        }
        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (exprContainsTensorName(childExpr)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool indexExprValues(const clang::Expr* expr,
                         std::vector<int64_t>& values) const {
        values.clear();
        int64_t value = 0;
        if (evaluateIntegerConstant(expr, Context, value) && value >= 0) {
            values.push_back(value);
            return true;
        }
        const std::string loopVar = declRefName(expr);
        const auto it = LoopPrefixes.find(loopVar);
        if (it == LoopPrefixes.end() || it->second.empty()) {
            return false;
        }
        const int64_t endExclusive = it->second.back();
        for (int64_t idx = 0; idx < endExclusive; ++idx) {
            values.push_back(idx);
        }
        return true;
    }

    bool recordTensorRead(const clang::Expr* expr) {
        std::string baseName;
        const std::vector<const clang::Expr*> indexExprs =
            collectSubscriptIndices(expr, baseName);
        if (baseName != TensorName ||
            (indexExprs.size() != 1 && indexExprs.size() != 2)) {
            return false;
        }

        std::vector<std::vector<int64_t>> perDimValues;
        int64_t totalExpanded = 1;
        for (const clang::Expr* indexExpr : indexExprs) {
            std::vector<int64_t> values;
            if (!indexExprValues(indexExpr, values) || values.empty()) {
                return false;
            }
            if (totalExpanded >
                kMaxTensor2ArrayBoundedIndices /
                    static_cast<int64_t>(values.size())) {
                return false;
            }
            totalExpanded *= static_cast<int64_t>(values.size());
            perDimValues.push_back(std::move(values));
        }

        if (perDimValues.size() == 1) {
            for (int64_t i : perDimValues[0]) {
                PostUseBoundedIndex index;
                index.indices = {i};
                addUniqueIndex(Indices, index);
            }
            return true;
        }

        for (int64_t i : perDimValues[0]) {
            for (int64_t j : perDimValues[1]) {
                PostUseBoundedIndex index;
                index.indices = {i, j};
                addUniqueIndex(Indices, index);
            }
        }
        return true;
    }

    bool exprContainsOnlyConstTensorIndexReads(const clang::Expr* expr) const {
        if (!expr) {
            return true;
        }

        std::string baseName;
        (void)collectSubscriptIndices(expr, baseName);
        if (baseName == TensorName) {
            return tensorReadExprIsBounded(expr);
        }

        if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(
                ignoreParenCasts(expr))) {
            if (dre->getDecl() && dre->getDecl()->getNameAsString() == TensorName) {
                PostUseBoundedIndex ignored;
                return isPartOfConstTensorIndexRead(dre, ignored);
            }
        }

        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (!exprContainsOnlyConstTensorIndexReads(childExpr)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool isPartOfConstTensorIndexRead(const clang::DeclRefExpr* dre,
                                      PostUseBoundedIndex& index) const {
        if (!dre || !Context) {
            return false;
        }
        clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
        for (int depth = 0; depth < 8; ++depth) {
            auto parents = Context->getParents(current);
            if (parents.empty()) {
                return false;
            }
            const clang::DynTypedNode& parent = parents[0];
            if (const auto* expr = parent.get<clang::Expr>()) {
                if (tensorReadExprIsBounded(expr)) {
                    return true;
                }
                current = parent;
                continue;
            }
            return false;
        }
        return false;
    }

    bool tensorReadExprIsBounded(const clang::Expr* expr) const {
        std::string baseName;
        const std::vector<const clang::Expr*> indexExprs =
            collectSubscriptIndices(expr, baseName);
        if (baseName != TensorName ||
            (indexExprs.size() != 1 && indexExprs.size() != 2)) {
            return false;
        }
        for (const clang::Expr* indexExpr : indexExprs) {
            std::vector<int64_t> values;
            if (!indexExprValues(indexExpr, values) || values.empty()) {
                return false;
            }
        }
        return true;
    }
};

bool isScalarZeroAssignment(const clang::Stmt* stmt, std::string& scalarName) {
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(stmt);
    if (!binary || binary->getOpcode() != clang::BO_Assign ||
        !isIntegerLiteralValue(binary->getRHS(), 0)) {
        return false;
    }
    scalarName = declRefName(binary->getLHS());
    return !scalarName.empty();
}

bool isScalarIncrement(const clang::Stmt* stmt, const std::string& scalarName) {
    if (const auto* unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(stmt)) {
        return unary->isIncrementOp() &&
               declRefName(unary->getSubExpr()) == scalarName;
    }
    if (const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(stmt)) {
        return opCall->getOperator() == clang::OO_PlusPlus &&
               opCall->getNumArgs() > 0 &&
               declRefName(opCall->getArg(0)) == scalarName;
    }
    if (const auto* compound =
            llvm::dyn_cast_or_null<clang::CompoundAssignOperator>(stmt)) {
        return compound->getOpcode() == clang::BO_AddAssign &&
               declRefName(compound->getLHS()) == scalarName &&
               isIntegerLiteralValue(compound->getRHS(), 1);
    }
    return false;
}

bool isTensorIndexRead(const clang::Expr* expr, const std::string& tensorName) {
    expr = ignoreParenCasts(expr);
    if (const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 1) {
            return declRefName(opCall->getArg(0)) == tensorName;
        }
    }
    if (const auto* array =
            llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expr)) {
        return declRefName(array->getBase()) == tensorName;
    }
    return false;
}

bool isTensorEqOneCondition(const clang::Expr* cond,
                            const std::string& tensorName) {
    cond = ignoreParenCasts(cond);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(cond);
    if (!binary || binary->getOpcode() != clang::BO_EQ) {
        return false;
    }
    return (isTensorIndexRead(binary->getLHS(), tensorName) &&
            isIntegerLiteralValue(binary->getRHS(), 1)) ||
           (isTensorIndexRead(binary->getRHS(), tensorName) &&
            isIntegerLiteralValue(binary->getLHS(), 1));
}

bool loopBodyIsIfTensorEqOneIncrement(const clang::Stmt* body,
                                      const std::string& tensorName,
                                      const std::string& scalarName) {
    if (const auto* compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body)) {
        if (compound->size() != 1) {
            return false;
        }
        body = *compound->body_begin();
    }
    const auto* ifStmt = llvm::dyn_cast_or_null<clang::IfStmt>(body);
    if (!ifStmt || ifStmt->getElse()) {
        return false;
    }
    return isTensorEqOneCondition(ifStmt->getCond(), tensorName) &&
           isScalarIncrement(ifStmt->getThen(), scalarName);
}

std::vector<const clang::Stmt*> collectPostDacStmts(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr) {
    std::vector<const clang::Stmt*> result;
    if (!dacppFile || !dacExpr || !dacppFile->getContext()) {
        return result;
    }

    clang::ASTContext* context = dacppFile->getContext();
    clang::DynTypedNode current = clang::DynTypedNode::create(*dacExpr);
    const clang::Stmt* containedStmt = dacExpr;
    std::set<const clang::Stmt*> appended;

    for (int depth = 0; depth < 64; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* compound = parent.get<clang::CompoundStmt>()) {
            bool sawContainingChild = false;
            for (const clang::Stmt* child : compound->body()) {
                if (!child) {
                    continue;
                }
                if (!sawContainingChild) {
                    if (child == containedStmt ||
                        containsDacExpr(child, dacExpr, context)) {
                        sawContainingChild = true;
                    }
                    continue;
                }
                if (appended.insert(child).second) {
                    result.push_back(child);
                }
            }
            containedStmt = compound;
            current = clang::DynTypedNode::create(*compound);
            continue;
        }
        if (const auto* stmt = parent.get<clang::Stmt>()) {
            containedStmt = stmt;
            current = clang::DynTypedNode::create(*stmt);
            continue;
        }
        if (parent.get<clang::FunctionDecl>()) {
            break;
        }
        break;
    }

    if (!result.empty() || !dacppFile->getMainBody()) {
        return result;
    }

    const auto* mainCompound =
        llvm::dyn_cast<clang::CompoundStmt>(dacppFile->getMainBody());
    if (!mainCompound) {
        return result;
    }
    bool sawDac = false;
    for (const clang::Stmt* stmt : mainCompound->body()) {
        if (!stmt) {
            continue;
        }
        if (!sawDac) {
            if (stmt == dacExpr || containsDacExpr(stmt, dacExpr, context)) {
                sawDac = true;
            }
            continue;
        }
        result.push_back(stmt);
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
                                  const BufferRegionPlan& plan,
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
        forStmt, dacppFile->getContext(), shell, plan, info);
}

std::string helperBaseName(Shell* shell, Calc* calc, int exprIdx) {
    return shell->getName() + "_" + calc->getName() + "_" +
           std::to_string(exprIdx);
}

std::string helperNameFor(Shell* shell,
                          Calc* calc,
                          int exprIdx,
                          std::size_t stmtIdx) {
    return "__dacpp_mpi_region_" + helperBaseName(shell, calc, exprIdx) +
           "_stmt_" + std::to_string(stmtIdx);
}

}  // namespace detail

std::vector<RootCentricPostRegion> collectRootCentricPostRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    int exprIdx,
    const clang::BinaryOperator* dacExpr) {
    std::vector<RootCentricPostRegion> result;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        return result;
    }

    BufferRegionPlan plan;
    if (!buildBufferRegionPlanForDacExpr(dacppFile, shell, dacExpr, plan) ||
        !plan.enabled || plan.dacExpr != dacExpr) {
        return result;
    }
    const DistributedStencilSitePlan sitePlan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!sitePlan.supported || !sitePlan.hasRootBridge) {
        return result;
    }

    std::set<const clang::Stmt*> distributedFollowupStmts;
    const auto distributedRegions =
        collectDistributedFollowupRegions(dacppFile, shell, calc, dacExpr);
    for (const auto& region : distributedRegions) {
        distributedFollowupStmts.insert(region.stmt);
    }
    if (!sitePlan.hasRootBridge && sitePlan.followupMappings.size() == 1) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            distributedFollowupStmts.insert(stmt);
        }
    }

    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        const clang::Stmt* stmt = plan.siblingStmts[stmtIdx];
        if (distributedFollowupStmts.count(stmt) != 0) {
            continue;
        }
        if (!detail::isRootCentricRegionSupported(dacppFile, shell, plan, stmt)) {
            if (!sitePlan.hasRootBridge) {
                continue;
            }
        }
        result.push_back(
            {stmt, detail::helperNameFor(shell, calc, exprIdx, stmtIdx)});
    }
    return result;
}

PostUseReductionPlan analyzePostUseReduction(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr,
    const std::string& tensorName) {
    PostUseReductionPlan result;
    result.tensorName = tensorName;
    if (!dacppFile || !dacExpr || tensorName.empty()) {
        result.reason = "missing input";
        return result;
    }
    const auto postStmts = collectPostDacStmts(dacppFile, dacExpr);
    const clang::Stmt* resetStmt = nullptr;
    std::string scalarName;
    const clang::ForStmt* reductionLoop = nullptr;
    for (const clang::Stmt* stmt : postStmts) {
        if (!resetStmt) {
            std::string candidateScalar;
            if (!isScalarZeroAssignment(stmt, candidateScalar)) {
                result.reason = "first post statement is not scalar reset";
                return result;
            }
            resetStmt = stmt;
            scalarName = candidateScalar;
            continue;
        }
        if (!reductionLoop) {
            reductionLoop = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
            if (!reductionLoop ||
                !loopBodyIsIfTensorEqOneIncrement(reductionLoop->getBody(),
                                                  tensorName, scalarName)) {
                result.reason = "second post statement is not eq-one count loop";
                return result;
            }
            result.supported = true;
            result.resetStmt = resetStmt;
            result.loopStmt = reductionLoop;
            result.scalarName = scalarName;
            result.compareValue = "1";
            result.reason = "if tensor[i] == 1 then scalar++";
            return result;
        }
    }

    result.reason = "post reduction statements not found";
    return result;
}

PostUseSyncPlan analyzePostUseSync(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr,
    const std::string& tensorName) {
    PostUseSyncPlan result;
    result.tensorName = tensorName;
    if (!dacppFile || !dacExpr || tensorName.empty() ||
        !dacppFile->getContext()) {
        result.kind = PostUseSyncKind::FullTensor;
        result.reason = "missing input";
        return result;
    }

    const PostUseReductionPlan reduction =
        analyzePostUseReduction(dacppFile, dacExpr, tensorName);
    if (reduction.supported) {
        result.kind = PostUseSyncKind::ScalarReductionCountEqOne;
        result.reason = reduction.reason;
        return result;
    }

    std::set<const clang::Stmt*> ignoredPostStmts;
    if (shell && calc) {
        for (const auto& region :
             collectDistributedFollowupRegions(dacppFile, shell, calc, dacExpr)) {
            ignoredPostStmts.insert(region.stmt);
        }
        const DistributedStencilSitePlan sitePlan =
            analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
        if (sitePlan.supported && !sitePlan.hasRootBridge) {
            ignoredPostStmts.insert(sitePlan.distributedFollowupStmts.begin(),
                                    sitePlan.distributedFollowupStmts.end());
            ignoredPostStmts.insert(sitePlan.boundaryLocalStmts.begin(),
                                    sitePlan.boundaryLocalStmts.end());
        }
    }

    const auto postStmts = collectPostDacStmts(dacppFile, dacExpr);
    bool sawHostUse = false;
    const clang::Stmt* tensor2ArrayStmt = nullptr;
    std::string tensor2ArrayTarget;
    for (const clang::Stmt* stmt : postStmts) {
        if (!stmt) {
            continue;
        }
        if (isIgnoredLoweredPostStmt(stmt, ignoredPostStmts,
                                     dacppFile->getContext())) {
            continue;
        }

        if (!tensor2ArrayStmt) {
            std::string targetName;
            if (tensor2ArrayCallForStmt(stmt, tensorName, targetName)) {
                tensor2ArrayStmt = stmt;
                tensor2ArrayTarget = targetName;
                sawHostUse = true;
                continue;
            }
        }

        if (tensor2ArrayStmt && !tensor2ArrayTarget.empty()) {
            VectorFixedUseVisitor vectorVisitor(tensor2ArrayTarget,
                                                dacppFile->getContext());
            vectorVisitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
            if (vectorVisitor.FullUse) {
                result.kind = PostUseSyncKind::FullTensor;
                result.reason = vectorVisitor.Reason.empty()
                                    ? "unknown tensor2Array target use"
                                    : vectorVisitor.Reason;
                return result;
            }
            if (!vectorVisitor.Indices.empty()) {
                if (!stmtContainsCoutExpr(stmt)) {
                    result.kind = PostUseSyncKind::FullTensor;
                    result.reason =
                        "tensor2Array target read is not root-observable cout";
                    return result;
                }
                for (const auto& index : vectorVisitor.Indices) {
                    addUniqueIndex(result.boundedIndices, index);
                }
                continue;
            }
        }

        PostUseIndexedReadVisitor visitor(tensorName, dacppFile->getContext());
        visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
        if (visitor.FullUse) {
            result.kind = PostUseSyncKind::FullTensor;
            result.reason = visitor.Reason.empty() ? "unknown host tensor use"
                                                   : visitor.Reason;
            return result;
        }
        if (!visitor.Indices.empty()) {
            if (!stmtContainsCoutExpr(stmt)) {
                result.kind = PostUseSyncKind::FullTensor;
                result.reason = "bounded indexed read is not root-observable cout";
                return result;
            }
            sawHostUse = true;
            for (const auto& index : visitor.Indices) {
                addUniqueIndex(result.boundedIndices, index);
            }
        }
    }

    if (!sawHostUse) {
        result.kind = PostUseSyncKind::None;
        result.reason = "no host-visible post-DAC use";
        return result;
    }
    result.kind = PostUseSyncKind::BoundedIndexedRootRead;
    result.reason = tensor2ArrayStmt ? "root-only tensor2Array bounded target read"
                                     : "root-only bounded indexed read";
    result.tensor2ArrayStmt = tensor2ArrayStmt;
    result.tensor2ArrayTargetName = tensor2ArrayTarget;
    return result;
}

std::vector<const clang::Stmt*> collectRootCentricPostRegionStmts(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr) {
    std::vector<const clang::Stmt*> result;
    if (!dacppFile || !dacExpr || !dacppFile->getContext()) {
        return result;
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        if (!expr || expr->getDacExpr() != dacExpr) {
            continue;
        }
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        BufferRegionPlan plan;
        if (!buildBufferRegionPlanForDacExpr(dacppFile, shell, dacExpr, plan) ||
            !plan.enabled || plan.dacExpr != dacExpr) {
            break;
        }
        const DistributedStencilSitePlan sitePlan =
            analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
        if (!sitePlan.supported || !sitePlan.hasRootBridge) {
            break;
        }
        for (const clang::Stmt* stmt : plan.siblingStmts) {
            if (detail::isRootCentricRegionSupported(dacppFile, shell, plan, stmt) ||
                sitePlan.hasRootBridge) {
                result.push_back(stmt);
            }
        }
        break;
    }

    return result;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
