#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "StencilAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

bool tryCollectDistributedFollowup(DistributedStencilSitePlan& plan,
                                   DacppFile* dacppFile,
                                   Shell* shell,
                                   const BufferRegionPlan& regionPlan,
                                   const std::vector<IOTYPE>& effectiveModes,
                                   const std::vector<IOTYPE>& transportModes,
                                   const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }

    RouteLoopInfo info;
    if (!extractRouteLoopInfo(
            forStmt, dacppFile->getContext(), regionPlan, info)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectTopLevelAssignments(forStmt->getBody(), assignments)) {
        return false;
    }
    if (assignments.empty()) {
        return false;
    }

    std::vector<DistributedFollowupMapping> routes;
    routes.reserve(assignments.size());
    const auto splitMeta = collectSplitBindMeta(shell);
    for (const RouteAssignment& assignment : assignments) {
        std::string readerTensor;
        std::string writerTensor;
        AffineIndex1D readerIndex;
        AffineIndex1D writerIndex;
        if (!parseAffineVectorAccessAST(assignment.lhs, info,
                                        readerTensor, readerIndex) ||
            !parseAffineVectorAccessAST(assignment.rhs, info,
                                        writerTensor, writerIndex)) {
            return false;
        }
        if (!isEffectiveWriter(writerTensor, shell, transportModes) ||
            !isEffectiveReader(readerTensor, shell, transportModes)) {
            return false;
        }
        const int writerIdx = findShellParamIndex(shell, writerTensor);
        const int readerIdx = findShellParamIndex(shell, readerTensor);
        if (writerIdx < 0 || readerIdx < 0) {
            return false;
        }
        if (!writerRouteCoveredByInputDomain(
                shell, effectiveModes, writerIdx, splitMeta)) {
            return false;
        }

        DistributedFollowupMapping route;
        route.writerTensor = writerTensor;
        route.readerTensor = readerTensor;
        route.writerParamIndex = writerIdx;
        route.readerParamIndex = readerIdx;
        route.writerIndex = writerIndex;
        route.readerIndex = readerIndex;
        route.targetOffset = readerIndex.offset - writerIndex.offset;
        route.stmt = assignment.stmt;
        routes.push_back(route);
        llvm::outs() << "[DACPP][MPI][PhaseC] route detected "
                     << writerTensor << "->" << readerTensor
                     << " offset=" << route.targetOffset << "\n";
    }

    plan.followupMappings.insert(plan.followupMappings.end(),
                                 routes.begin(), routes.end());
    plan.distributedFollowupStmts.push_back(stmt);
    return true;
}

bool tryCollectDistributedFollowup2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& effectiveModes,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }

    RouteLoopInfo2D info;
    if (!extractRouteLoopInfo2D(
            forStmt, dacppFile->getContext(), regionPlan, info)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectNestedLoopAssignments2D(forStmt, assignments) ||
        assignments.empty()) {
        return false;
    }

    std::vector<DistributedFollowupMapping> routes;
    routes.reserve(assignments.size());
    const auto splitMeta = collectSplitBindMeta(shell);
    for (const RouteAssignment& assignment : assignments) {
        std::string readerTensor;
        std::string writerTensor;
        AffineIndex2D readerIndex;
        AffineIndex2D writerIndex;
        if (!parseAffineMatrixAccessAST(assignment.lhs, info,
                                        readerTensor, readerIndex) ||
            !parseAffineMatrixAccessAST(assignment.rhs, info,
                                        writerTensor, writerIndex)) {
            return false;
        }
        if (!isEffectiveWriter(writerTensor, shell, transportModes) ||
            !isEffectiveReader(readerTensor, shell, transportModes)) {
            return false;
        }
        const int writerIdx = findShellParamIndex(shell, writerTensor);
        const int readerIdx = findShellParamIndex(shell, readerTensor);
        if (writerIdx < 0 || readerIdx < 0) {
            return false;
        }
        if (!writerRouteCoveredByInputDomain2D(
                shell, effectiveModes, writerIdx, splitMeta)) {
            return false;
        }

        DistributedFollowupMapping route;
        route.writerTensor = writerTensor;
        route.readerTensor = readerTensor;
        route.writerParamIndex = writerIdx;
        route.readerParamIndex = readerIdx;
        route.rank = 2;
        route.writerIndex2D = writerIndex;
        route.readerIndex2D = readerIndex;
        route.targetRowOffset = readerIndex.rowOffset - writerIndex.rowOffset;
        route.targetColOffset = readerIndex.colOffset - writerIndex.colOffset;
        route.stmt = assignment.stmt;
        routes.push_back(route);
        llvm::outs() << "[DACPP][MPI][PhaseC] route detected "
                     << writerTensor << "->" << readerTensor
                     << " offset=(" << route.targetRowOffset << ","
                     << route.targetColOffset << ")\n";
    }

    plan.followupMappings.insert(plan.followupMappings.end(),
                                 routes.begin(), routes.end());
    plan.distributedFollowupStmts.push_back(stmt);
    return true;
}

bool tryCollectReadCacheTransition2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }

    RouteLoopInfo2D info;
    if (!extractRouteLoopInfo2D(
            forStmt, dacppFile->getContext(), regionPlan, info)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectNestedLoopAssignments2D(forStmt, assignments) ||
        assignments.empty()) {
        return false;
    }

    std::vector<ReadCacheStateTransition> transitions;
    transitions.reserve(assignments.size());
    for (const RouteAssignment& assignment : assignments) {
        std::string readerTensor;
        std::string writerTensor;
        AffineIndex2D readerIndex;
        AffineIndex2D writerIndex;
        if (!parseAffineMatrixAccessAST(assignment.lhs, info,
                                        readerTensor, readerIndex) ||
            !parseAffineMatrixAccessAST(assignment.rhs, info,
                                        writerTensor, writerIndex)) {
            return false;
        }
        if (!isEffectiveReader(writerTensor, shell, transportModes) ||
            !isEffectiveReader(readerTensor, shell, transportModes)) {
            return false;
        }
        const int writerIdx = findShellParamIndex(shell, writerTensor);
        const int readerIdx = findShellParamIndex(shell, readerTensor);
        if (writerIdx < 0 || readerIdx < 0) {
            return false;
        }

        ReadCacheStateTransition transition;
        transition.writerTensor = writerTensor;
        transition.readerTensor = readerTensor;
        transition.writerParamIndex = writerIdx;
        transition.readerParamIndex = readerIdx;
        transition.rank = 2;
        transition.writerIndex2D = writerIndex;
        transition.readerIndex2D = readerIndex;
        transition.targetRowOffset = readerIndex.rowOffset - writerIndex.rowOffset;
        transition.targetColOffset = readerIndex.colOffset - writerIndex.colOffset;
        transition.stmt = assignment.stmt;
        transitions.push_back(transition);
        llvm::outs() << "[DACPP][MPI][PhaseC] read-cache transition detected "
                     << writerTensor << "->" << readerTensor
                     << " offset=(" << transition.targetRowOffset << ","
                     << transition.targetColOffset << ")\n";
    }

    plan.readCacheTransitions.insert(plan.readCacheTransitions.end(),
                                     transitions.begin(), transitions.end());
    plan.distributedFollowupStmts.push_back(stmt);
    return true;
}

bool tryCollectBoundaryLocalUpdate2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt || !dacppFile || !dacppFile->getContext()) {
        return false;
    }

    RouteLoopInfo loopInfo;
    if (!extractRouteLoopInfo(forStmt, dacppFile->getContext(),
                              regionPlan, loopInfo)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectTopLevelAssignments(forStmt->getBody(), assignments) ||
        assignments.empty()) {
        return false;
    }

    std::vector<BoundaryLocalUpdate> updates;
    updates.reserve(assignments.size());
    for (const RouteAssignment& assignment : assignments) {
        std::string targetTensor;
        bool targetRowUsesLoop = false;
        bool targetColUsesLoop = false;
        int targetRow = 0;
        int targetCol = 0;
        std::string targetRowExpr;
        std::string targetColExpr;
        if (!parseBoundaryMatrixAccessAST(assignment.lhs, loopInfo,
                                          dacppFile->getContext(), targetTensor,
                                          targetRowUsesLoop, targetRow,
                                          targetRowExpr,
                                          targetColUsesLoop, targetCol,
                                          targetColExpr)) {
            return false;
        }

        std::string sourceTensor;
        bool sourceRowUsesLoop = false;
        bool sourceColUsesLoop = false;
        int sourceRow = 0;
        int sourceCol = 0;
        std::string sourceRowExpr;
        std::string sourceColExpr;
        bool constantRhs = false;
        std::string constantValue;
        if (parseBoundaryMatrixAccessAST(assignment.rhs, loopInfo,
                                         dacppFile->getContext(), sourceTensor,
                                         sourceRowUsesLoop, sourceRow,
                                         sourceRowExpr,
                                         sourceColUsesLoop, sourceCol,
                                         sourceColExpr)) {
            if (sourceTensor != targetTensor ||
                sourceRowUsesLoop != targetRowUsesLoop ||
                sourceColUsesLoop != targetColUsesLoop) {
                return false;
            }
            const bool rowOk =
                isBoundaryCopyPair(targetRowUsesLoop, targetRow, targetRowExpr,
                                   sourceRowUsesLoop, sourceRow, sourceRowExpr);
            const bool colOk =
                isBoundaryCopyPair(targetColUsesLoop, targetCol, targetColExpr,
                                   sourceColUsesLoop, sourceCol, sourceColExpr);
            const bool rowBoundary =
                !targetRowUsesLoop && !sourceRowUsesLoop;
            const bool colBoundary =
                !targetColUsesLoop && !sourceColUsesLoop;
            if (!((rowBoundary && rowOk && colOk) ||
                  (colBoundary && rowOk && colOk))) {
                return false;
            }
        } else if (int literal = 0; parseIntegerLiteralExpr(assignment.rhs, literal)) {
            sourceTensor = targetTensor;
            sourceRowUsesLoop = targetRowUsesLoop;
            sourceColUsesLoop = targetColUsesLoop;
            sourceRow = targetRow;
            sourceCol = targetCol;
            sourceRowExpr = targetRowExpr;
            sourceColExpr = targetColExpr;
            constantRhs = true;
            constantValue = std::to_string(literal);
            const bool targetIsBoundary =
                isBoundaryConstantTarget(targetRowUsesLoop, targetRow,
                                         targetRowExpr, targetColUsesLoop) ||
                isBoundaryConstantTarget(targetColUsesLoop, targetCol,
                                         targetColExpr, targetRowUsesLoop);
            if (!targetIsBoundary) {
                return false;
            }
        } else {
            return false;
        }

        const int paramIdx = findShellParamIndex(shell, targetTensor);
        if (paramIdx < 0 || paramIdx >= static_cast<int>(transportModes.size()) ||
            transportModes[paramIdx] == IOTYPE::WRITE) {
            return false;
        }

        BoundaryLocalUpdate update;
        update.tensorName = targetTensor;
        update.sourceTensorName = sourceTensor;
        update.paramIndex = paramIdx;
        update.sourceParamIndex = findShellParamIndex(shell, sourceTensor);
        update.rank = 2;
        update.targetRow = targetRow;
        update.targetCol = targetCol;
        update.sourceRow = sourceRow;
        update.sourceCol = sourceCol;
        update.targetRowExpr = targetRowExpr;
        update.targetColExpr = targetColExpr;
        update.sourceRowExpr = sourceRowExpr;
        update.sourceColExpr = sourceColExpr;
        update.targetRowUsesLoop = targetRowUsesLoop;
        update.targetColUsesLoop = targetColUsesLoop;
        update.sourceRowUsesLoop = sourceRowUsesLoop;
        update.sourceColUsesLoop = sourceColUsesLoop;
        update.loopLowerExpr = loopInfo.lowerExpr;
        update.loopUpperExpr = loopInfo.upperExpr;
        update.loopUpperInclusive = loopInfo.upperInclusive;
        update.constantRhs = constantRhs;
        update.constantValue = constantValue;
        update.stmt = assignment.stmt;
        updates.push_back(update);
    }

    plan.boundaryLocalUpdates.insert(plan.boundaryLocalUpdates.end(),
                                     updates.begin(), updates.end());
    plan.boundaryLocalStmts.push_back(stmt);
    llvm::outs() << "[DACPP][MPI][PhaseC] boundary-local update detected stmt\n";
    return true;
}

bool tryCollectBoundaryLocalUpdate1D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt || !dacppFile || !dacppFile->getContext()) {
        return false;
    }

    RouteLoopInfo loopInfo;
    if (!extractRouteLoopInfo(forStmt, dacppFile->getContext(),
                              regionPlan, loopInfo)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectTopLevelAssignments(forStmt->getBody(), assignments) ||
        assignments.empty()) {
        return false;
    }

    std::vector<BoundaryLocalUpdate> updates;
    updates.reserve(assignments.size());
    for (const RouteAssignment& assignment : assignments) {
        std::string targetTensor;
        bool targetUsesLoop = false;
        int targetOffset = 0;
        std::string targetExpr;
        if (!parseBoundaryVectorAccessAST(assignment.lhs, loopInfo,
                                          dacppFile->getContext(),
                                          targetTensor, targetUsesLoop,
                                          targetOffset, targetExpr)) {
            return false;
        }

        std::string sourceTensor;
        bool sourceUsesLoop = false;
        int sourceOffset = 0;
        std::string sourceExpr;
        bool constantRhs = false;
        std::string constantValue;
        if (parseBoundaryVectorAccessAST(assignment.rhs, loopInfo,
                                         dacppFile->getContext(),
                                         sourceTensor, sourceUsesLoop,
                                         sourceOffset, sourceExpr)) {
            if (targetUsesLoop || sourceUsesLoop) {
                return false;
            }
        } else if (int literal = 0; parseIntegerLiteralExpr(assignment.rhs, literal)) {
            sourceTensor = targetTensor;
            sourceUsesLoop = targetUsesLoop;
            sourceOffset = targetOffset;
            sourceExpr = targetExpr;
            constantRhs = true;
            constantValue = std::to_string(literal);
            if (targetUsesLoop) {
                return false;
            }
        } else {
            return false;
        }

        const int targetParamIdx = findShellParamIndex(shell, targetTensor);
        if (targetParamIdx < 0 ||
            targetParamIdx >= static_cast<int>(transportModes.size()) ||
            transportModes[targetParamIdx] == IOTYPE::WRITE) {
            return false;
        }
        const int sourceParamIdx = findShellParamIndex(shell, sourceTensor);
        if (!constantRhs &&
            (sourceParamIdx < 0 ||
             sourceParamIdx >= static_cast<int>(transportModes.size()))) {
            return false;
        }

        BoundaryLocalUpdate update;
        update.tensorName = targetTensor;
        update.sourceTensorName = sourceTensor;
        update.paramIndex = targetParamIdx;
        update.sourceParamIndex = sourceParamIdx;
        update.rank = 1;
        update.targetRow = targetOffset;
        update.sourceRow = sourceOffset;
        update.targetRowExpr = targetExpr;
        update.sourceRowExpr = sourceExpr;
        update.targetRowUsesLoop = targetUsesLoop;
        update.sourceRowUsesLoop = sourceUsesLoop;
        update.loopLowerExpr = loopInfo.lowerExpr;
        update.loopUpperExpr = loopInfo.upperExpr;
        update.loopUpperInclusive = loopInfo.upperInclusive;
        update.constantRhs = constantRhs;
        update.constantValue = constantValue;
        update.stmt = assignment.stmt;
        updates.push_back(update);
    }

    plan.boundaryLocalUpdates.insert(plan.boundaryLocalUpdates.end(),
                                     updates.begin(), updates.end());
    plan.boundaryLocalStmts.push_back(stmt);
    llvm::outs() << "[DACPP][MPI][PhaseC] boundary-local update detected stmt\n";
    return true;
}

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
