#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"

#include "Rewriter_MPI_OutputAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

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

    bool TraverseStmt(clang::Stmt* stmt) {
        if (!stmt) {
            return true;
        }

        const bool isCurrentDacExpr = CurrentDacExpr && stmt == CurrentDacExpr;
        bool isDacExpr = false;
        for (auto* expr : DacExprs) {
            if (stmt == expr) {
                isDacExpr = true;
                break;
            }
        }

        if (isDacExpr) {
            ++InsideDacExpr;
        }
        if (RootRegionStmts.count(stmt) != 0) {
            ++RootRegionDepth;
        }

        bool result =
            clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseStmt(stmt);

        if (RootRegionStmts.count(stmt) != 0) {
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

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (dre->getDecl() && dre->getDecl()->getNameAsString() == TargetName &&
            InsideDacExpr == 0 && SeenCurrentDacExpr) {
            if (WriteDepth == 0 || UpdateReadDepth > 0) {
                HasPostDacRead = true;
                if (RootRegionDepth > 0) {
                    HasReadInsideRootRegion = true;
                } else if (detail::isRootOnlyObservableCall(dre, Context)) {
                    // Observable print/cout is rewritten root-only in MPI codegen.
                } else if (const std::string assignedName =
                               detail::assignmentLhsBaseForRhsRead(dre, Context);
                           !assignedName.empty()) {
                    RootOnlyPropagationTargets.insert(assignedName);
                } else {
                    HasReadOutsideRootRegion = true;
                }
            }
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

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseBinaryOperator(
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

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseUnaryOperator(
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

        return clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseCXXOperatorCallExpr(
            opCall);
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
        tensorName, dacppFile->dacExprs, currentDacExpr, dacppFile->getContext(),
        rootRegionStmts);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(dacppFile->getMainBody()));

    if (visitor.HasReadOutsideRootRegion) {
        visiting.erase(tensorName);
        return false;
    }
    for (const std::string& propagatedName : visitor.RootOnlyPropagationTargets) {
        if (!isRootOnlyObservableAfterDac(dacppFile, propagatedName,
                                          currentDacExpr, rootRegionStmts,
                                          visiting)) {
            visiting.erase(tensorName);
            return false;
        }
    }

    visiting.erase(tensorName);
    return true;
}

Expression* findExprForDacExpr(DacppFile* dacppFile,
                               const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !dacExpr) {
        return nullptr;
    }
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        if (expr && expr->getDacExpr() == dacExpr) {
            return expr;
        }
    }
    return nullptr;
}

}  // namespace

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
    return requirement == OutputSyncRequirement::AllRanksNeeded;
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
    std::set<const clang::Stmt*> rootRegionStmts(rootRegionStmtVec.begin(),
                                                 rootRegionStmtVec.end());

    Expression* siteExpr = findExprForDacExpr(dacppFile, currentDacExpr);
    if (siteExpr && tensorUsesDistributedFollowup(dacppFile, siteExpr->getShell(),
                                                  siteExpr->getCalc(), tensorName,
                                                  currentDacExpr)) {
        return OutputSyncRequirement::DistributedFollowup;
    }

    const std::string actualTensorName =
        detail::resolveActualTensorName(tensorName, currentDacExpr);
    TensorUseVisitor visitor(actualTensorName, dacppFile->dacExprs,
                             currentDacExpr, dacppFile->getContext(),
                             rootRegionStmts);
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
            if (!isRootOnlyObservableAfterDac(dacppFile, propagatedName,
                                              currentDacExpr, rootRegionStmts,
                                              visiting)) {
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

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
