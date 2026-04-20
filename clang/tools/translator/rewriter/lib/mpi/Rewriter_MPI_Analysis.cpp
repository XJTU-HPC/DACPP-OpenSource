#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

class TensorUseVisitor : public clang::RecursiveASTVisitor<TensorUseVisitor> {
public:
    std::string TargetName;
    const std::vector<const clang::BinaryOperator*>& DacExprs;
    const clang::BinaryOperator* CurrentDacExpr = nullptr;

    bool NeedsBcast = false;
    int InsideDacExpr = 0;
    bool SeenCurrentDacExpr = false;

    // Read/write tracking — mirrors ParamAccessVisitor logic.
    int WriteDepth = 0;
    int UpdateReadDepth = 0;

    TensorUseVisitor(std::string name,
                     const std::vector<const clang::BinaryOperator*>& dacExprs,
                     const clang::BinaryOperator* currentDacExpr)
        : TargetName(std::move(name)),
          DacExprs(dacExprs),
          CurrentDacExpr(currentDacExpr),
          SeenCurrentDacExpr(currentDacExpr == nullptr) {}

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

        bool result = clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseStmt(S);

        if (isDacExpr) {
            --InsideDacExpr;
        }
        if (isCurrentDacExpr) {
            SeenCurrentDacExpr = true;
        }

        return result;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (NeedsBcast) {
            return true;
        }

        if (DRE->getDecl() && DRE->getDecl()->getNameAsString() == TargetName &&
            InsideDacExpr == 0 && SeenCurrentDacExpr) {
            // Pure read, or compound-assignment read (e.g. +=) → needs bcast.
            // Pure write (e.g. =) with no read → does NOT need bcast.
            if (WriteDepth == 0 || UpdateReadDepth > 0) {
                NeedsBcast = true;
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

bool tensorNeedsBroadcast(DacppFile* dacppFile,
                          const std::string& tensorName,
                          const clang::BinaryOperator* currentDacExpr) {
    if (!dacppFile) {
        return false;
    }

    bool needsBcast = dacppFile->getMPIBroadcastOutputs();
    if (dacppFile->getMainBody()) {
        TensorUseVisitor visitor(tensorName, dacppFile->dacExprs, currentDacExpr);
        visitor.TraverseStmt(const_cast<clang::Stmt*>(dacppFile->getMainBody()));
        needsBcast = visitor.NeedsBcast;
    }
    return needsBcast;
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

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
