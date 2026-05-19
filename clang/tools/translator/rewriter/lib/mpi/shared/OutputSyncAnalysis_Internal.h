#ifndef DACPP_TRANSLATOR_REWRITER_MPI_OUTPUT_ANALYSIS_INTERNAL_H
#define DACPP_TRANSLATOR_REWRITER_MPI_OUTPUT_ANALYSIS_INTERNAL_H

#include <set>
#include <string>

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

constexpr int kMaxParentTraversalDepth = 16;
constexpr int kMaxEnclosingStmtDepth = 32;
constexpr int kMaxSemicolonScanOffset = 256;

inline const clang::CallExpr* getShellCallExpr(
    const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    if (const auto* lhsCall =
            llvm::dyn_cast<clang::CallExpr>(dacExpr->getLHS()->IgnoreParenImpCasts())) {
        return lhsCall;
    }
    if (const auto* rhsCall =
            llvm::dyn_cast<clang::CallExpr>(dacExpr->getRHS()->IgnoreParenImpCasts())) {
        return rhsCall;
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    if (!shellExpr) {
        return dacppTranslator::getNode<clang::CallExpr>(
            const_cast<clang::BinaryOperator*>(dacExpr));
    }

    shellExpr = shellExpr->IgnoreParenImpCasts();
    if (const auto* directCall = llvm::dyn_cast<clang::CallExpr>(shellExpr)) {
        return directCall;
    }
    if (const auto* nestedCall = dacppTranslator::getNode<clang::CallExpr>(shellExpr)) {
        return nestedCall;
    }
    return dacppTranslator::getNode<clang::CallExpr>(
        const_cast<clang::BinaryOperator*>(dacExpr));
}

inline bool containsStmt(const clang::Stmt* root, const clang::Stmt* needle) {
    if (!root || !needle) {
        return false;
    }
    if (root == needle) {
        return true;
    }
    for (const clang::Stmt* child : root->children()) {
        if (containsStmt(child, needle)) {
            return true;
        }
    }
    return false;
}

inline std::string extractBaseDeclName(const clang::Expr* expr) {
    if (!expr) {
        return "";
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        return DRE->getDecl() ? DRE->getDecl()->getNameAsString() : "";
    }
    if (const auto* ASE = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        return extractBaseDeclName(ASE->getBase());
    }
    if (const auto* ME = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        return extractBaseDeclName(ME->getBase());
    }
    if (const auto* UO = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
        if (UO->getOpcode() == clang::UO_Deref ||
            UO->getOpcode() == clang::UO_AddrOf) {
            return extractBaseDeclName(UO->getSubExpr());
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if ((opCall->getOperator() == clang::OO_Subscript ||
             opCall->getOperator() == clang::OO_Call) &&
            opCall->getNumArgs() > 0) {
            return extractBaseDeclName(opCall->getArg(0));
        }
    }
    if (const auto* memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
        return extractBaseDeclName(memberCall->getImplicitObjectArgument());
    }

    for (const clang::Stmt* child : expr->children()) {
        if (const auto* childExpr = llvm::dyn_cast_or_null<clang::Expr>(child)) {
            std::string name = extractBaseDeclName(childExpr);
            if (!name.empty()) {
                return name;
            }
        }
    }
    return "";
}

inline std::string extractBaseNameFromSourceText(const clang::Expr* expr,
                                                 clang::ASTContext& context) {
    if (!expr) {
        return "";
    }

    std::string text = clang::Lexer::getSourceText(
                           clang::CharSourceRange::getTokenRange(
                               expr->getSourceRange()),
                           context.getSourceManager(),
                           context.getLangOpts())
                           .str();
    if (text.empty()) {
        return "";
    }

    const auto isIdentStart = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    const auto isIdentChar = [&](char c) {
        return isIdentStart(c) || (c >= '0' && c <= '9');
    };

    for (std::size_t idx = 0; idx < text.size(); ++idx) {
        if (!isIdentStart(text[idx])) {
            continue;
        }

        std::size_t end = idx + 1;
        while (end < text.size() && isIdentChar(text[end])) {
            ++end;
        }

        if (end + 1 < text.size() && text[end] == ':' && text[end + 1] == ':') {
            idx = end + 1;
            continue;
        }

        return text.substr(idx, end - idx);
    }

    return "";
}

inline std::string resolveActualTensorName(
    const std::string& shellParamName,
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
        const std::string actualName = extractBaseDeclName(shellCall->getArg(paramIdx));
        if (!actualName.empty()) {
            return actualName;
        }
        if (const auto* directCallee = shellCall->getDirectCallee()) {
            const std::string textName = extractBaseNameFromSourceText(
                shellCall->getArg(paramIdx), directCallee->getASTContext());
            if (!textName.empty()) {
                return textName;
            }
        }
        return shellParamName;
    }
    return shellParamName;
}

inline bool containsCoutExpr(const clang::Stmt* stmt) {
    if (!stmt) {
        return false;
    }
    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        return dre->getDecl() && dre->getDecl()->getNameAsString() == "cout";
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (containsCoutExpr(child)) {
            return true;
        }
    }
    return false;
}

inline bool isPrintMethodCall(const clang::CXXMemberCallExpr* call) {
    if (!call) {
        return false;
    }
    const clang::CXXMethodDecl* method = call->getMethodDecl();
    if (method && method->getNameAsString() == "print") {
        return true;
    }
    const auto* member = llvm::dyn_cast_or_null<clang::MemberExpr>(
        call->getCallee()->IgnoreParenImpCasts());
    return member && member->getMemberDecl() &&
           member->getMemberDecl()->getNameAsString() == "print";
}

inline bool isRootOnlyObservableCall(const clang::DeclRefExpr* dre,
                                     clang::ASTContext* context) {
    if (!dre || !context) {
        return false;
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* memberCall = parent.get<clang::CXXMemberCallExpr>()) {
            if (isPrintMethodCall(memberCall)) {
                return true;
            }
        }
        if (const auto* opCall = parent.get<clang::CXXOperatorCallExpr>()) {
            if (opCall->getOperator() == clang::OO_LessLess &&
                containsCoutExpr(opCall)) {
                return true;
            }
        }
        if (parent.get<clang::CompoundStmt>() ||
            parent.get<clang::ForStmt>() ||
            parent.get<clang::WhileStmt>() ||
            parent.get<clang::IfStmt>() ||
            parent.get<clang::ReturnStmt>()) {
            break;
        }
        current = parent;
    }
    return false;
}

inline std::string tensor2ArrayTargetForObjectRead(const clang::DeclRefExpr* dre,
                                                   clang::ASTContext* context) {
    if (!dre || !context) {
        return "";
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* memberCall = parent.get<clang::CXXMemberCallExpr>()) {
            const clang::CXXMethodDecl* method = memberCall->getMethodDecl();
            const std::string methodName =
                method ? method->getNameAsString() : "";
            if (methodName == "tensor2Array" &&
                memberCall->getNumArgs() >= 1 &&
                containsStmt(memberCall->getImplicitObjectArgument(), dre)) {
                return extractBaseDeclName(memberCall->getArg(0));
            }
        }
        if (parent.get<clang::CompoundStmt>() ||
            parent.get<clang::ForStmt>() ||
            parent.get<clang::WhileStmt>() ||
            parent.get<clang::IfStmt>() ||
            parent.get<clang::ReturnStmt>()) {
            break;
        }
        current = parent;
    }
    return "";
}

inline bool isTensor2ArrayOutputArgument(const clang::DeclRefExpr* dre,
                                         clang::ASTContext* context) {
    if (!dre || !context) {
        return false;
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* memberCall = parent.get<clang::CXXMemberCallExpr>()) {
            const clang::CXXMethodDecl* method = memberCall->getMethodDecl();
            const std::string methodName =
                method ? method->getNameAsString() : "";
            if (methodName == "tensor2Array") {
                for (unsigned argIdx = 0; argIdx < memberCall->getNumArgs();
                     ++argIdx) {
                    if (containsStmt(memberCall->getArg(argIdx), dre)) {
                        return true;
                    }
                }
                return false;
            }
        }
        if (parent.get<clang::CompoundStmt>() ||
            parent.get<clang::ForStmt>() ||
            parent.get<clang::WhileStmt>() ||
            parent.get<clang::IfStmt>() ||
            parent.get<clang::ReturnStmt>()) {
            break;
        }
        current = parent;
    }
    return false;
}

inline void collectBaseWrites(const clang::Stmt* stmt,
                              std::set<std::string>& writes) {
    if (!stmt) {
        return;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (binary->isAssignmentOp()) {
            const std::string name = extractBaseDeclName(binary->getLHS());
            if (!name.empty()) {
                writes.insert(name);
            }
        }
    } else if (const auto* compound =
                   llvm::dyn_cast<clang::CompoundAssignOperator>(stmt)) {
        const std::string name = extractBaseDeclName(compound->getLHS());
        if (!name.empty()) {
            writes.insert(name);
        }
    } else if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
        if (unary->isIncrementDecrementOp()) {
            const std::string name = extractBaseDeclName(unary->getSubExpr());
            if (!name.empty()) {
                writes.insert(name);
            }
        }
    } else if (const auto* opCall =
                   llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0) {
            const std::string name = extractBaseDeclName(opCall->getArg(0));
            if (!name.empty()) {
                writes.insert(name);
            }
        } else if ((opCall->getOperator() == clang::OO_PlusPlus ||
                    opCall->getOperator() == clang::OO_MinusMinus) &&
                   opCall->getNumArgs() > 0) {
            const std::string name = extractBaseDeclName(opCall->getArg(0));
            if (!name.empty()) {
                writes.insert(name);
            }
        }
    }

    for (const clang::Stmt* child : stmt->children()) {
        collectBaseWrites(child, writes);
    }
}

inline std::set<std::string> controlWriteTargetsForRead(
    const clang::DeclRefExpr* dre,
    clang::ASTContext* context) {
    std::set<std::string> writes;
    if (!dre || !context) {
        return writes;
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* ifStmt = parent.get<clang::IfStmt>()) {
            if (containsStmt(ifStmt->getCond(), dre)) {
                collectBaseWrites(ifStmt->getThen(), writes);
                collectBaseWrites(ifStmt->getElse(), writes);
                return writes;
            }
        }
        if (parent.get<clang::ForStmt>() ||
            parent.get<clang::WhileStmt>() ||
            parent.get<clang::CompoundStmt>() ||
            parent.get<clang::ReturnStmt>()) {
            break;
        }
        current = parent;
    }
    return writes;
}

inline std::string assignmentLhsBaseForRhsRead(const clang::DeclRefExpr* dre,
                                               clang::ASTContext* context) {
    if (!dre || !context) {
        return "";
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* binary = parent.get<clang::BinaryOperator>()) {
            if (binary->isAssignmentOp() && containsStmt(binary->getRHS(), dre)) {
                return extractBaseDeclName(binary->getLHS());
            }
        }
        if (const auto* opCall = parent.get<clang::CXXOperatorCallExpr>()) {
            if (opCall->isAssignmentOp() && opCall->getNumArgs() > 1) {
                for (unsigned argIdx = 1; argIdx < opCall->getNumArgs(); ++argIdx) {
                    if (containsStmt(opCall->getArg(argIdx), dre)) {
                        return extractBaseDeclName(opCall->getArg(0));
                    }
                }
            }
        }
        if (parent.get<clang::CompoundStmt>() ||
            parent.get<clang::ForStmt>() ||
            parent.get<clang::WhileStmt>() ||
            parent.get<clang::IfStmt>() ||
            parent.get<clang::ReturnStmt>()) {
            break;
        }
        current = parent;
    }
    return "";
}

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_OUTPUT_ANALYSIS_INTERNAL_H
