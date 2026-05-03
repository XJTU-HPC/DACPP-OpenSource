#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

// Maximum depth for parent-traversal lookups (statement-level analysis).
constexpr int kMaxParentTraversalDepth = 16;
// Maximum depth for finding the enclosing statement boundary.
constexpr int kMaxEnclosingStmtDepth = 32;
// Maximum byte offset scanned past a statement for the trailing semicolon.
constexpr int kMaxSemicolonScanOffset = 256;

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

bool containsStmt(const clang::Stmt* root, const clang::Stmt* needle) {
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

std::string extractBaseDeclName(const clang::Expr* expr) {
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
    if (const auto* OpCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if ((OpCall->getOperator() == clang::OO_Subscript ||
             OpCall->getOperator() == clang::OO_Call) &&
            OpCall->getNumArgs() > 0) {
            return extractBaseDeclName(OpCall->getArg(0));
        }
    }
    if (const auto* MCE = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
        return extractBaseDeclName(MCE->getImplicitObjectArgument());
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
        const std::string actualName = extractBaseDeclName(shellCall->getArg(paramIdx));
        return actualName.empty() ? shellParamName : actualName;
    }
    return shellParamName;
}

bool containsCoutExpr(const clang::Stmt* stmt) {
    if (!stmt) {
        return false;
    }
    if (const auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        return DRE->getDecl() &&
               DRE->getDecl()->getNameAsString() == "cout";
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (containsCoutExpr(child)) {
            return true;
        }
    }
    return false;
}

bool isPrintMethodCall(const clang::CXXMemberCallExpr* call) {
    if (!call) {
        return false;
    }
    const clang::CXXMethodDecl* method = call->getMethodDecl();
    if (method && method->getNameAsString() == "print") {
        return true;
    }
    const auto* member =
        llvm::dyn_cast_or_null<clang::MemberExpr>(
            call->getCallee()->IgnoreParenImpCasts());
    return member && member->getMemberDecl() &&
           member->getMemberDecl()->getNameAsString() == "print";
}

bool isRootOnlyObservableCall(const clang::DeclRefExpr* DRE,
                              clang::ASTContext* context) {
    if (!DRE || !context) {
        return false;
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*DRE);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* memberCall =
                parent.get<clang::CXXMemberCallExpr>()) {
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

std::string assignmentLhsBaseForRhsRead(const clang::DeclRefExpr* DRE,
                                        clang::ASTContext* context) {
    if (!DRE || !context) {
        return "";
    }

    clang::DynTypedNode current = clang::DynTypedNode::create(*DRE);
    for (int depth = 0; depth < kMaxParentTraversalDepth; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            break;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* BO = parent.get<clang::BinaryOperator>()) {
            if (BO->isAssignmentOp() && containsStmt(BO->getRHS(), DRE)) {
                return extractBaseDeclName(BO->getLHS());
            }
        }
        if (const auto* OpCall = parent.get<clang::CXXOperatorCallExpr>()) {
            if (OpCall->isAssignmentOp() && OpCall->getNumArgs() > 1) {
                for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                    if (containsStmt(OpCall->getArg(argIdx), DRE)) {
                        return extractBaseDeclName(OpCall->getArg(0));
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

}  // namespace

class RootOnlyPrintRewriteVisitor
    : public clang::RecursiveASTVisitor<RootOnlyPrintRewriteVisitor> {
public:
    clang::Rewriter* TheRewriter = nullptr;
    clang::ASTContext* Context = nullptr;
    std::set<const clang::Stmt*> RewrittenStmts;

    RootOnlyPrintRewriteVisitor(clang::Rewriter* rewriter,
                                clang::ASTContext* context)
        : TheRewriter(rewriter), Context(context) {}

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call || !TheRewriter) {
            return true;
        }
        if (!isPrintCall(call)) {
            return true;
        }

        rewriteOutputStatement(call);
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
        if (!call || !TheRewriter) {
            return true;
        }
        if (call->getOperator() != clang::OO_LessLess ||
            !containsCoutExpr(call)) {
            return true;
        }

        rewriteOutputStatement(call);
        return true;
    }

private:
    void rewriteOutputStatement(const clang::Stmt* output) {
        const clang::Stmt* stmt = enclosingStatement(output);
        if (!stmt || RewrittenStmts.count(stmt) != 0) {
            return;
        }
        const clang::SourceManager& SM = TheRewriter->getSourceMgr();
        if (!stmt->getBeginLoc().isValid() ||
            !SM.isWrittenInMainFile(stmt->getBeginLoc())) {
            return;
        }
        const clang::LangOptions& LO = TheRewriter->getLangOpts();
        clang::SourceLocation replacementEnd = statementEndIncludingSemi(stmt);
        if (replacementEnd.isInvalid()) {
            return;
        }
        const std::string stmtText =
            clang::Lexer::getSourceText(
                clang::CharSourceRange::getCharRange(stmt->getBeginLoc(),
                                                     replacementEnd),
                SM, LO)
                .str();
        if (stmtText.empty()) {
            return;
        }

        TheRewriter->ReplaceText(
            clang::CharSourceRange::getCharRange(stmt->getBeginLoc(),
                                                 replacementEnd),
            "if (__dacpp_mpi_is_root_rank()) {\n        " + stmtText + "\n    }");
        RewrittenStmts.insert(stmt);
    }

    bool isPrintCall(const clang::CXXMemberCallExpr* call) const {
        return isPrintMethodCall(call);
    }

    const clang::Stmt* enclosingStatement(const clang::Stmt* stmt) const {
        if (!stmt || !Context) {
            return nullptr;
        }

        clang::DynTypedNode current = clang::DynTypedNode::create(*stmt);
        for (int depth = 0; depth < kMaxEnclosingStmtDepth; ++depth) {
            auto parents = Context->getParents(current);
            if (parents.empty()) {
                break;
            }
            const clang::DynTypedNode& parent = parents[0];
            if (parent.get<clang::CompoundStmt>()) {
                if (const auto* asStmt = current.get<clang::Stmt>()) {
                    return asStmt;
                }
                break;
            }
            if (parent.get<clang::ForStmt>() ||
                parent.get<clang::WhileStmt>() ||
                parent.get<clang::IfStmt>()) {
                if (const auto* asStmt = current.get<clang::Stmt>()) {
                    return asStmt;
                }
                break;
            }
            if (parent.get<clang::ReturnStmt>()) {
                break;
            }
            current = parent;
        }
        return nullptr;
    }

    clang::SourceLocation statementEndIncludingSemi(
        const clang::Stmt* stmt) const {
        if (!stmt || !TheRewriter) {
            return clang::SourceLocation();
        }
        const clang::SourceManager& SM = TheRewriter->getSourceMgr();
        const clang::LangOptions& LO = TheRewriter->getLangOpts();
        clang::SourceLocation afterToken =
            clang::Lexer::getLocForEndOfToken(stmt->getEndLoc(), 0, SM, LO);
        if (afterToken.isInvalid()) {
            return clang::SourceLocation();
        }

        for (int offset = 0; offset < kMaxSemicolonScanOffset; ++offset) {
            clang::SourceLocation loc = afterToken.getLocWithOffset(offset);
            if (loc.isInvalid() || !SM.isWrittenInSameFile(afterToken, loc)) {
                break;
            }
            bool invalid = false;
            const char* data = SM.getCharacterData(loc, &invalid);
            if (invalid || !data) {
                break;
            }
            if (*data == ';') {
                return loc.getLocWithOffset(1);
            }
            if (*data != ' ' && *data != '\t' && *data != '\r' && *data != '\n') {
                break;
            }
        }

        return afterToken;
    }
};

class TensorUseVisitor : public clang::RecursiveASTVisitor<TensorUseVisitor> {
public:
    std::string TargetName;
    const std::vector<const clang::BinaryOperator*>& DacExprs;
    const clang::BinaryOperator* CurrentDacExpr = nullptr;
    clang::ASTContext* Context = nullptr;

    bool HasPostDacRead = false;
    int InsideDacExpr = 0;
    int RootRegionDepth = 0;
    bool SeenCurrentDacExpr = false;
    const std::set<const clang::Stmt*> RootRegionStmts;

    // Read/write tracking — mirrors ParamAccessVisitor logic.
    int WriteDepth = 0;
    int UpdateReadDepth = 0;
    bool HasReadOutsideRootRegion = false;
    bool HasReadInsideRootRegion = false;
    std::set<std::string> RootOnlyPropagationTargets;

    TensorUseVisitor(std::string name,
                     const std::vector<const clang::BinaryOperator*>& dacExprs,
                     const clang::BinaryOperator* currentDacExpr,
                     clang::ASTContext* context,
                     std::set<const clang::Stmt*> rootRegionStmts = {})
        : TargetName(std::move(name)),
          DacExprs(dacExprs),
          CurrentDacExpr(currentDacExpr),
          Context(context),
          SeenCurrentDacExpr(currentDacExpr == nullptr),
          RootRegionStmts(std::move(rootRegionStmts)) {}

    bool TraverseStmt(clang::Stmt* S) {
        if (!S) {
            return true;
        }

        const bool isCurrentDacExpr = CurrentDacExpr && S == CurrentDacExpr;
        bool isDacExpr = false;
        for (auto* expr : DacExprs) {
            if (S == expr) {
                isDacExpr = true;
                break;
            }
        }

        if (isDacExpr) {
            ++InsideDacExpr;
        }
        if (RootRegionStmts.count(S) != 0) {
            ++RootRegionDepth;
        }

        bool result = clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseStmt(S);

        if (RootRegionStmts.count(S) != 0) {
            --RootRegionDepth;
        }
        if (isDacExpr) {
            --InsideDacExpr;
        }
        if (isCurrentDacExpr) {
            SeenCurrentDacExpr = true;
        }

        return result;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (DRE->getDecl() && DRE->getDecl()->getNameAsString() == TargetName &&
            InsideDacExpr == 0 && SeenCurrentDacExpr) {
            // Pure read, or compound-assignment read (e.g. +=) → needs bcast.
            // Pure write (e.g. =) with no read → does NOT need bcast.
            if (WriteDepth == 0 || UpdateReadDepth > 0) {
                HasPostDacRead = true;
                if (RootRegionDepth > 0) {
                    HasReadInsideRootRegion = true;
                } else if (isRootOnlyObservableCall(DRE, Context)) {
                    // MPI codegen rewrites visible output statements so only
                    // rank 0 executes observable .print()/cout output.
                } else if (const std::string assignedName =
                               assignmentLhsBaseForRhsRead(DRE, Context);
                           !assignedName.empty()) {
                    RootOnlyPropagationTargets.insert(assignedName);
                } else {
                    HasReadOutsideRootRegion = true;
                }
            }
        }
        return true;
    }

    bool TraverseBinaryOperator(clang::BinaryOperator* BO) {
        if (!BO) {
            return true;
        }

        if (BO->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(BO->getLHS());
            --WriteDepth;
            TraverseStmt(BO->getRHS());
            return true;
        }

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseBinaryOperator(BO);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BO) {
        if (!BO) {
            return true;
        }

        ++WriteDepth;
        TraverseStmt(BO->getLHS());
        --WriteDepth;
        ++UpdateReadDepth;
        TraverseStmt(BO->getLHS());
        --UpdateReadDepth;
        TraverseStmt(BO->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* UO) {
        if (!UO) {
            return true;
        }

        if (UO->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(UO->getSubExpr());
            --WriteDepth;
            ++UpdateReadDepth;
            TraverseStmt(UO->getSubExpr());
            --UpdateReadDepth;
            return true;
        }

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseUnaryOperator(UO);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
        if (!OpCall) {
            return true;
        }

        if (OpCall->isAssignmentOp()) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                if (OpCall->getOperator() != clang::OO_Equal) {
                    ++UpdateReadDepth;
                    TraverseStmt(OpCall->getArg(0));
                    --UpdateReadDepth;
                }
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        if (OpCall->getOperator() == clang::OO_PlusPlus ||
            OpCall->getOperator() == clang::OO_MinusMinus) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                ++UpdateReadDepth;
                TraverseStmt(OpCall->getArg(0));
                --UpdateReadDepth;
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseCXXOperatorCallExpr(OpCall);
    }
};

bool isRootOnlyObservableAfterDac(
    DacppFile* dacppFile,
    const std::string& tensorName,
    const clang::BinaryOperator* currentDacExpr,
    const std::set<const clang::Stmt*>& rootRegionStmts,
    std::set<std::string>& visiting) {
    if (!dacppFile || tensorName.empty() || !dacppFile->getMainBody()) {
        return false;
    }
    if (!visiting.insert(tensorName).second) {
        return false;
    }

    TensorUseVisitor visitor(
        tensorName, dacppFile->dacExprs, currentDacExpr,
        dacppFile->getContext(), rootRegionStmts);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(dacppFile->getMainBody()));

    if (visitor.HasReadOutsideRootRegion) {
        visiting.erase(tensorName);
        return false;
    }
    for (const std::string& propagatedName : visitor.RootOnlyPropagationTargets) {
        if (!isRootOnlyObservableAfterDac(
                dacppFile, propagatedName, currentDacExpr,
                rootRegionStmts, visiting)) {
            visiting.erase(tensorName);
            return false;
        }
    }

    visiting.erase(tensorName);
    return true;
}

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

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (!DRE) {
            return true;
        }

        auto it = ParamIndices.find(DRE->getDecl());
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

    bool TraverseBinaryOperator(clang::BinaryOperator* BO) {
        if (!BO) {
            return true;
        }

        if (BO->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(BO->getLHS());
            --WriteDepth;
            TraverseStmt(BO->getRHS());
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseBinaryOperator(BO);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BO) {
        if (!BO) {
            return true;
        }

        ++WriteDepth;
        TraverseStmt(BO->getLHS());
        --WriteDepth;
        ++UpdateReadDepth;
        TraverseStmt(BO->getLHS());
        --UpdateReadDepth;
        TraverseStmt(BO->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* UO) {
        if (!UO) {
            return true;
        }

        if (UO->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(UO->getSubExpr());
            --WriteDepth;
            ++UpdateReadDepth;
            TraverseStmt(UO->getSubExpr());
            --UpdateReadDepth;
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseUnaryOperator(UO);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
        if (!OpCall) {
            return true;
        }

        if (OpCall->isAssignmentOp()) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                if (OpCall->getOperator() != clang::OO_Equal) {
                    ++UpdateReadDepth;
                    TraverseStmt(OpCall->getArg(0));
                    --UpdateReadDepth;
                }
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        if (OpCall->getOperator() == clang::OO_PlusPlus ||
            OpCall->getOperator() == clang::OO_MinusMinus) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                ++UpdateReadDepth;
                TraverseStmt(OpCall->getArg(0));
                --UpdateReadDepth;
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseCXXOperatorCallExpr(OpCall);
    }

private:
    int WriteDepth = 0;
    int UpdateReadDepth = 0;
};

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

const char* outputSyncRequirementName(OutputSyncRequirement requirement) {
    switch (requirement) {
    case OutputSyncRequirement::RootOnly:
        return "root-only";
    case OutputSyncRequirement::AllRanksNeeded:
        return "all-ranks-needed";
    case OutputSyncRequirement::RootCentricFollowup:
        return "root-centric-followup";
    case OutputSyncRequirement::DistributedFollowup:
        return "distributed-followup";
    }
    return "unknown";
}

bool requiresBroadcast(OutputSyncRequirement requirement) {
    return requirement == OutputSyncRequirement::AllRanksNeeded ||
           requirement == OutputSyncRequirement::DistributedFollowup;
}

OutputSyncRequirement classifyOutputSyncRequirement(
    DacppFile* dacppFile,
    const std::string& tensorName,
    const clang::BinaryOperator* currentDacExpr) {
    if (!dacppFile) {
        return OutputSyncRequirement::RootOnly;
    }

    if (!dacppFile->getMPIBroadcastOutputs()) {
        return OutputSyncRequirement::RootOnly;
    }

    if (!dacppFile->getMainBody()) {
        return OutputSyncRequirement::AllRanksNeeded;
    }

    std::vector<const clang::Stmt*> rootRegionStmtVec =
        collectRootCentricPostRegionStmts(dacppFile, currentDacExpr);
    std::set<const clang::Stmt*> rootRegionStmts(
        rootRegionStmtVec.begin(), rootRegionStmtVec.end());

    const std::string actualTensorName =
        resolveActualTensorName(tensorName, currentDacExpr);
    TensorUseVisitor visitor(
        actualTensorName, dacppFile->dacExprs, currentDacExpr,
        dacppFile->getContext(), rootRegionStmts);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(dacppFile->getMainBody()));

    if (!visitor.HasPostDacRead) {
        return OutputSyncRequirement::RootOnly;
    }
    if (!visitor.HasReadOutsideRootRegion && visitor.HasReadInsideRootRegion) {
        return OutputSyncRequirement::RootCentricFollowup;
    }
    if (!visitor.HasReadOutsideRootRegion) {
        std::set<std::string> visiting;
        bool propagatedRootOnly = true;
        for (const std::string& propagatedName :
             visitor.RootOnlyPropagationTargets) {
            if (!isRootOnlyObservableAfterDac(
                    dacppFile, propagatedName, currentDacExpr,
                    rootRegionStmts, visiting)) {
                propagatedRootOnly = false;
                break;
            }
        }
        if (propagatedRootOnly) {
            return OutputSyncRequirement::RootOnly;
        }
    }
    return OutputSyncRequirement::AllRanksNeeded;
}

bool tensorNeedsBroadcast(DacppFile* dacppFile,
                          const std::string& tensorName,
                          const clang::BinaryOperator* currentDacExpr) {
    return requiresBroadcast(
        classifyOutputSyncRequirement(dacppFile, tensorName, currentDacExpr));
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
            modes[paramIdx] = modes[paramIdx] == IOTYPE::WRITE
                                  ? IOTYPE::WRITE
                                  : IOTYPE::READ_WRITE;
        } else if (writes) {
            modes[paramIdx] = IOTYPE::WRITE;
        } else if (reads || updateReads) {
            modes[paramIdx] = IOTYPE::READ;
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

void rewritePrintCallsRootOnly(clang::Rewriter* rewriter,
                               clang::TranslationUnitDecl* tuDecl) {
    if (!rewriter || !tuDecl) {
        return;
    }
    clang::SourceLocation insertLoc;
    const clang::SourceManager& SM = rewriter->getSourceMgr();
    for (const clang::Decl* decl : tuDecl->decls()) {
        if (!decl) {
            continue;
        }
        clang::SourceLocation loc = decl->getBeginLoc();
        if (loc.isValid() && SM.isWrittenInMainFile(loc)) {
            insertLoc = loc;
            break;
        }
    }
    if (insertLoc.isValid()) {
        rewriter->InsertText(insertLoc,
                             "static inline bool __dacpp_mpi_is_root_rank();\n");
    }
    RootOnlyPrintRewriteVisitor visitor(rewriter, &tuDecl->getASTContext());
    visitor.TraverseDecl(tuDecl);
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
