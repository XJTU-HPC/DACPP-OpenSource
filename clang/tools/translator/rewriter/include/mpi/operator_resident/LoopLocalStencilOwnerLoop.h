#ifndef DACPP_MPI_OPERATOR_RESIDENT_LOOP_LOCAL_STENCIL_OWNER_LOOP_H
#define DACPP_MPI_OPERATOR_RESIDENT_LOOP_LOCAL_STENCIL_OWNER_LOOP_H

#include <cstdint>
#include <string>

#include "mpi/operator_resident/OperatorResidentPlan.h"

namespace clang {
class BinaryOperator;
class Stmt;
}  // namespace clang

namespace dacppTranslator {

class DacppFile;

namespace mpi_rewriter {
namespace operator_resident {

struct LoopLocalStencilOwnerLoopContract {
    bool enabled = false;
    int exprIndex = -1;
    const clang::Stmt* replacementStmt = nullptr;
    LoopLoweringContract lowering;
    std::string ownerTensorName;
    std::string scalarExpr;
    std::string functionName;
    std::string elementType;
    std::string mpiType;
    bool boundaryReplayLocalFormula = false;
    std::string boundaryReplayReason;
    std::string boundaryTimeArrayName;
    std::string boundaryTimeElementType;
    std::string leftBoundaryExpr;
    std::string rightBoundaryExpr;
    int temporalBlockSize = 0;
    std::string temporalBlockReason;
    bool fixedPostUseRow = false;
    int64_t postUseRow = -1;
    std::string postUseReason;
    bool contractConsistencyCheckPassed = false;
    std::string contractConsistencyCheckReason;
    std::string rejectedReason;
};

LoopLocalStencilOwnerLoopContract detectLoopLocalStencilOwnerLoop(
    DacppFile* dacppFile,
    const ShellPartitionPlan& exprPlan);

std::string buildLoopLocalStencilOwnerLoopCode(
    const LoopLocalStencilOwnerLoopContract& contract,
    const ShellPartitionPlan& exprPlan);

void logLoopLocalStencilOwnerLoopAccepted(
    const LoopLocalStencilOwnerLoopContract& contract);

void logLoopLocalStencilOwnerLoopRejected(
    const LoopLocalStencilOwnerLoopContract& contract);

void logLoopLocalStencilOwnerLoopRewriteEnabled(
    const LoopLocalStencilOwnerLoopContract& contract);

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_MPI_OPERATOR_RESIDENT_LOOP_LOCAL_STENCIL_OWNER_LOOP_H
