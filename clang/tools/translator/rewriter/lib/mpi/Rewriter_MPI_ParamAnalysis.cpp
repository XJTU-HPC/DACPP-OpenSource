#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

class ParamAccessVisitor : public clang::RecursiveASTVisitor<ParamAccessVisitor> {
public:
    const std::unordered_map<const clang::ValueDecl*, int>& ParamIndices;
    std::vector<bool> Reads;
    std::vector<bool> UpdateReads;
    std::vector<bool> Writes;

    explicit ParamAccessVisitor(
        const std::unordered_map<const clang::ValueDecl*, int>& paramIndices,
        int paramCount)
        : ParamIndices(paramIndices),
          Reads(paramCount, false),
          UpdateReads(paramCount, false),
          Writes(paramCount, false) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (!dre) {
            return true;
        }

        auto it = ParamIndices.find(dre->getDecl());
        if (it == ParamIndices.end()) {
            return true;
        }

        if (WriteDepth > 0) {
            Writes[it->second] = true;
        } else if (UpdateReadDepth > 0) {
            UpdateReads[it->second] = true;
        } else {
            Reads[it->second] = true;
        }
        return true;
    }

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

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseBinaryOperator(
            binary);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* binary) {
        if (!binary) {
            return true;
        }

        ++WriteDepth;
        TraverseStmt(binary->getLHS());
        --WriteDepth;
        ++UpdateReadDepth;
        TraverseStmt(binary->getLHS());
        --UpdateReadDepth;
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
            ++UpdateReadDepth;
            TraverseStmt(unary->getSubExpr());
            --UpdateReadDepth;
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseUnaryOperator(
            unary);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall) {
            return true;
        }

        if (opCall->isAssignmentOp()) {
            if (opCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(opCall->getArg(0));
                --WriteDepth;

                if (opCall->getOperator() != clang::OO_Equal) {
                    ++UpdateReadDepth;
                    TraverseStmt(opCall->getArg(0));
                    --UpdateReadDepth;
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

                ++UpdateReadDepth;
                TraverseStmt(opCall->getArg(0));
                --UpdateReadDepth;
            }

            for (unsigned argIdx = 1; argIdx < opCall->getNumArgs(); ++argIdx) {
                TraverseStmt(opCall->getArg(argIdx));
            }
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseCXXOperatorCallExpr(
            opCall);
    }

private:
    int WriteDepth = 0;
    int UpdateReadDepth = 0;
};

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string sourceText(const clang::Stmt* stmt, clang::ASTContext* context) {
    if (!stmt || !context) {
        return "";
    }
    return trim(clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
                    context->getSourceManager(),
                    context->getLangOpts())
                    .str());
}

const clang::DeclRefExpr* getBaseDeclRef(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        return dre;
    }
    if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        return getBaseDeclRef(member->getBase());
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            return getBaseDeclRef(opCall->getArg(0));
        }
    }
    return nullptr;
}

bool getParamSubscriptKey(const clang::Expr* expr,
                          const clang::ValueDecl* paramDecl,
                          clang::ASTContext* context,
                          std::string& key) {
    if (!expr || !paramDecl || !context) {
        return false;
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        const clang::DeclRefExpr* base = getBaseDeclRef(subscript->getBase());
        if (!base || base->getDecl() != paramDecl) {
            return false;
        }
        key = sourceText(subscript->getIdx(), context);
        return !key.empty();
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() != clang::OO_Subscript ||
            opCall->getNumArgs() < 2) {
            return false;
        }
        const clang::DeclRefExpr* base = getBaseDeclRef(opCall->getArg(0));
        if (!base || base->getDecl() != paramDecl) {
            return false;
        }
        key = sourceText(opCall->getArg(1), context);
        return !key.empty();
    }
    return false;
}

class ParamReferenceVisitor
    : public clang::RecursiveASTVisitor<ParamReferenceVisitor> {
public:
    explicit ParamReferenceVisitor(const clang::ValueDecl* paramDecl)
        : ParamDecl(paramDecl) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (dre && dre->getDecl() == ParamDecl) {
            Found = true;
        }
        return !Found;
    }

    bool found() const { return Found; }

private:
    const clang::ValueDecl* ParamDecl = nullptr;
    bool Found = false;
};

bool containsParamReference(const clang::Stmt* stmt,
                            const clang::ValueDecl* paramDecl) {
    if (!stmt || !paramDecl) {
        return false;
    }
    ParamReferenceVisitor visitor(paramDecl);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return visitor.found();
}

bool isKnownSafeValueCall(const clang::CallExpr* call) {
    if (!call) {
        return false;
    }
    const clang::FunctionDecl* callee = call->getDirectCallee();
    if (!callee) {
        return false;
    }
    const std::string qualifiedName = callee->getQualifiedNameAsString();
    return qualifiedName == "std::max" || qualifiedName == "std::min" ||
           qualifiedName == "std::fmax" || qualifiedName == "std::fmin";
}

class WriteBeforeReadVisitor
    : public clang::RecursiveASTVisitor<WriteBeforeReadVisitor> {
public:
    WriteBeforeReadVisitor(const clang::ValueDecl* paramDecl,
                           clang::ASTContext* context)
        : ParamDecl(paramDecl), Context(context) {}

    bool TraverseStmt(clang::Stmt* stmt) {
        if (!stmt || NeedsOldValue) {
            return true;
        }
        if (llvm::isa<clang::IfStmt>(stmt) ||
            llvm::isa<clang::SwitchStmt>(stmt) ||
            llvm::isa<clang::ForStmt>(stmt) ||
            llvm::isa<clang::WhileStmt>(stmt) ||
            llvm::isa<clang::DoStmt>(stmt) ||
            llvm::isa<clang::ReturnStmt>(stmt) ||
            llvm::isa<clang::CXXForRangeStmt>(stmt)) {
            NeedsOldValue = true;
            return true;
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseStmt(stmt);
    }

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary || NeedsOldValue) {
            return true;
        }
        if (binary->getOpcode() == clang::BO_Assign) {
            std::string lhsKey;
            const bool writesParam =
                getParamSubscriptKey(binary->getLHS(), ParamDecl, Context, lhsKey);
            if (!writesParam) {
                TraverseStmt(binary->getLHS());
            }
            TraverseStmt(binary->getRHS());
            if (writesParam) {
                WrittenKeys.insert(lhsKey);
            }
            return true;
        }
        if (binary->isAssignmentOp()) {
            NeedsOldValue = true;
            return true;
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseBinaryOperator(binary);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator*) {
        NeedsOldValue = true;
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (unary && unary->isIncrementDecrementOp()) {
            NeedsOldValue = true;
            return true;
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseUnaryOperator(unary);
    }

    bool TraverseCallExpr(clang::CallExpr* call) {
        if (!call || NeedsOldValue) {
            return true;
        }
        if (!isKnownSafeValueCall(call)) {
            for (const clang::Expr* arg : call->arguments()) {
                if (containsParamReference(arg, ParamDecl)) {
                    NeedsOldValue = true;
                    return true;
                }
            }
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseCallExpr(call);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall || NeedsOldValue) {
            return true;
        }
        if (opCall->getOperator() == clang::OO_Subscript) {
            std::string key;
            if (getParamSubscriptKey(opCall, ParamDecl, Context, key)) {
                if (WrittenKeys.count(key) == 0) {
                    NeedsOldValue = true;
                    return true;
                }
                if (opCall->getNumArgs() > 1) {
                    TraverseStmt(opCall->getArg(1));
                }
                return true;
            }
        }
        if (opCall->isAssignmentOp() ||
            opCall->getOperator() == clang::OO_PlusPlus ||
            opCall->getOperator() == clang::OO_MinusMinus) {
            NeedsOldValue = true;
            return true;
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseCXXOperatorCallExpr(opCall);
    }

    bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* subscript) {
        if (!subscript || NeedsOldValue) {
            return true;
        }
        std::string key;
        if (getParamSubscriptKey(subscript, ParamDecl, Context, key)) {
            if (WrittenKeys.count(key) == 0) {
                NeedsOldValue = true;
                return true;
            }
            TraverseStmt(subscript->getIdx());
            return true;
        }
        return clang::RecursiveASTVisitor<WriteBeforeReadVisitor>::TraverseArraySubscriptExpr(subscript);
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (dre && dre->getDecl() == ParamDecl) {
            NeedsOldValue = true;
        }
        return true;
    }

    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* subscript) {
        std::string key;
        if (!getParamSubscriptKey(subscript, ParamDecl, Context, key)) {
            return true;
        }
        if (WrittenKeys.count(key) == 0) {
            NeedsOldValue = true;
        }
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall || opCall->getOperator() != clang::OO_Subscript) {
            return true;
        }
        std::string key;
        if (!getParamSubscriptKey(opCall, ParamDecl, Context, key)) {
            return true;
        }
        if (WrittenKeys.count(key) == 0) {
            NeedsOldValue = true;
        }
        return true;
    }

    bool needsOldValue() const { return NeedsOldValue; }

private:
    const clang::ValueDecl* ParamDecl = nullptr;
    clang::ASTContext* Context = nullptr;
    std::set<std::string> WrittenKeys;
    bool NeedsOldValue = false;
};

bool paramNeedsOldValueBeforeKernelWrite(clang::FunctionDecl* calcLoc,
                                         int paramIdx,
                                         clang::ASTContext* context) {
    if (!calcLoc || !context || paramIdx < 0 ||
        paramIdx >= static_cast<int>(calcLoc->getNumParams())) {
        return true;
    }
    const auto* paramDecl = calcLoc->getParamDecl(paramIdx);
    if (!paramDecl || !calcLoc->getBody()) {
        return true;
    }

    WriteBeforeReadVisitor visitor(paramDecl, context);
    if (const auto* compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(calcLoc->getBody())) {
        for (const clang::Stmt* stmt : compound->body()) {
            visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
            if (visitor.needsOldValue()) {
                return true;
            }
        }
        return false;
    }

    visitor.TraverseStmt(calcLoc->getBody());
    return visitor.needsOldValue();
}

}  // namespace

std::vector<AccessSummary> summarizeStmtAccess(
    const clang::Stmt* stmt,
    const std::unordered_map<const clang::ValueDecl*, int>& paramIndices,
    int paramCount) {
    std::vector<AccessSummary> summary(static_cast<std::size_t>(paramCount));
    if (!stmt || paramIndices.empty() || paramCount <= 0) {
        return summary;
    }

    ParamAccessVisitor visitor(paramIndices, paramCount);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        summary[static_cast<std::size_t>(paramIdx)].reads =
            visitor.Reads[paramIdx] || visitor.UpdateReads[paramIdx];
        summary[static_cast<std::size_t>(paramIdx)].writes =
            visitor.Writes[paramIdx];
    }
    return summary;
}

std::vector<IOTYPE> inferEffectiveParamModes(Shell* shell, Calc* calc) {
    std::vector<IOTYPE> modes;
    modes.reserve(shell->getNumShellParams());
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        modes.push_back(shell->getShellParam(paramIdx)->getRw());
    }

    clang::FunctionDecl* calcLoc = calc->getCalcLoc();
    if (!calcLoc || !calcLoc->getBody()) {
        return modes;
    }

    std::unordered_map<const clang::ValueDecl*, int> paramIndices;
    for (int paramIdx = 0; paramIdx < calcLoc->getNumParams(); ++paramIdx) {
        paramIndices.emplace(calcLoc->getParamDecl(paramIdx), paramIdx);
    }

    ParamAccessVisitor visitor(paramIndices, calc->getNumParams());
    visitor.TraverseStmt(calcLoc->getBody());

    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        const bool reads = visitor.Reads[paramIdx];
        const bool updateReads = visitor.UpdateReads[paramIdx];
        const bool writes = visitor.Writes[paramIdx];

        if (reads && writes) {
            modes[paramIdx] = IOTYPE::READ_WRITE;
        } else if (writes && updateReads) {
            modes[paramIdx] = IOTYPE::READ_WRITE;
        } else if (writes) {
            modes[paramIdx] = IOTYPE::WRITE;
        } else if (reads || updateReads) {
            modes[paramIdx] = IOTYPE::READ;
        }
    }

    return modes;
}

std::vector<IOTYPE> inferPhaseCTransportParamModes(Shell* shell, Calc* calc) {
    std::vector<IOTYPE> modes = inferEffectiveParamModes(shell, calc);
    if (!shell || !calc) {
        return modes;
    }

    clang::FunctionDecl* calcLoc = calc->getCalcLoc();
    if (!calcLoc || !calcLoc->getASTContext().getTranslationUnitDecl()) {
        return modes;
    }
    clang::ASTContext* context = &calcLoc->getASTContext();

    for (int paramIdx = 0; paramIdx < static_cast<int>(modes.size()) &&
                           paramIdx < calc->getNumParams();
         ++paramIdx) {
        if (modes[paramIdx] != IOTYPE::READ_WRITE) {
            continue;
        }
        if (!paramNeedsOldValueBeforeKernelWrite(calcLoc, paramIdx, context)) {
            modes[paramIdx] = IOTYPE::WRITE;
        }
    }
    return modes;
}

void collectReturnStmts(const clang::Stmt* stmt,
                        std::vector<const clang::ReturnStmt*>& returns) {
    if (!stmt) {
        return;
    }
    if (const auto* returnStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        returns.push_back(returnStmt);
    }
    for (const clang::Stmt* child : stmt->children()) {
        collectReturnStmts(child, returns);
    }
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
