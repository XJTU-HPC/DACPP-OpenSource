#include <unordered_map>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"

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
