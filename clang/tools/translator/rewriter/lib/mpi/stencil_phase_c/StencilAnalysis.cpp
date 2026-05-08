#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "../shared/PostRegion_Internal.h"
#include "StencilAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {

DistributedStencilSitePlan analyzeDistributedStencilSite(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    DistributedStencilSitePlan plan;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        plan.disableReason = "missing shell/calc site";
        return plan;
    }

    BufferRegionPlan regionPlan;
    if (!buildBufferRegionPlanForDacExpr(dacppFile, shell, dacExpr, regionPlan,
                                         &plan.disableReason) ||
        !regionPlan.enabled || regionPlan.dacExpr != dacExpr) {
        if (plan.disableReason.empty()) {
            plan.disableReason = "phase-c requires rewriteMPIStencil loop lowering";
        }
        return plan;
    }

    const auto effectiveModes = inferEffectiveParamModes(shell, calc);
    const auto transportModes = inferPhaseCTransportParamModes(shell, calc);
    if (effectiveModes.size() != static_cast<std::size_t>(shell->getNumShellParams()) ||
        transportModes.size() != static_cast<std::size_t>(shell->getNumShellParams())) {
        plan.disableReason = "failed to infer phase-c parameter modes";
        return plan;
    }

    bool hasVectorParam = false;
    bool hasMatrixParam = false;
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const bool vectorParam = detail::isVectorParam(shell, calc, paramIdx);
        const bool matrixParam = detail::isMatrixParam(shell, calc, paramIdx);
        hasVectorParam = hasVectorParam || vectorParam;
        hasMatrixParam = hasMatrixParam || matrixParam;
        if (!vectorParam && !matrixParam) {
            plan.disableReason =
                "phase-c only supports dacpp::Vector or dacpp::Matrix tensors";
            return plan;
        }
        if (effectiveModes[paramIdx] == IOTYPE::READ_WRITE &&
            transportModes[paramIdx] == IOTYPE::WRITE) {
            llvm::outs() << "[DACPP][MPI][PhaseC] param "
                         << shell->getParam(paramIdx)->getName()
                         << " transport=write after write-before-read analysis\n";
        }
        if (transportModes[paramIdx] == IOTYPE::READ_WRITE) {
            plan.disableReason =
                "phase-c does not support READ_WRITE kernel params";
            return plan;
        }
        plan.distributedTensors.insert(detail::resolveActualTensorName(
            shell->getParam(paramIdx)->getName(), dacExpr));
    }
    if (hasVectorParam && hasMatrixParam) {
        plan.disableReason = "phase-c does not support mixed Vector/Matrix sites";
        return plan;
    }

    std::vector<RootCentricPostRegion> rootRegions;
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        if (hasVectorParam &&
            detail::tryCollectDistributedFollowup(
                plan, dacppFile, shell, regionPlan, effectiveModes,
                transportModes, stmt)) {
            continue;
        }
        if (hasVectorParam && rootRegions.empty() &&
            !plan.followupMappings.empty() &&
            detail::tryCollectBoundaryLocalUpdate1D(
                plan, dacppFile, shell, regionPlan, transportModes, stmt)) {
            continue;
        }
        if (hasMatrixParam &&
            detail::tryCollectDistributedFollowup2D(
                plan, dacppFile, shell, regionPlan, effectiveModes,
                transportModes, stmt)) {
            continue;
        }
        if (hasMatrixParam &&
            detail::tryCollectReadCacheTransition2D(
                plan, dacppFile, shell, regionPlan, transportModes, stmt)) {
            continue;
        }
        if (hasMatrixParam && rootRegions.empty() &&
            plan.followupMappings.size() == 1 &&
            detail::tryCollectBoundaryLocalUpdate2D(
                plan, dacppFile, shell, regionPlan, transportModes, stmt)) {
            continue;
        }
        if (detail::isRootCentricRegionSupported(dacppFile, shell, regionPlan,
                                                 stmt)) {
            rootRegions.push_back({stmt, ""});
            continue;
        }
        if (hasMatrixParam) {
            rootRegions.push_back({stmt, ""});
            continue;
        }
        plan.disableReason =
            "phase-c requires post-shell statements to lower as distributed followup or root-centric helpers";
        return plan;
    }

    if (!rootRegions.empty()) {
        plan.hasRootBridge = true;
        detail::collectRootBridgeTensors(plan, dacppFile, shell, regionPlan);
    }

    plan.supported = true;
    return plan;
}

std::vector<DistributedFollowupRegion> collectDistributedFollowupRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    std::vector<DistributedFollowupRegion> regions;
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return regions;
    }
    for (const clang::Stmt* stmt : plan.distributedFollowupStmts) {
        regions.push_back({stmt});
    }
    return regions;
}

bool tensorUsesDistributedFollowup(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const std::string& tensorName,
    const clang::BinaryOperator* dacExpr) {
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return false;
    }
    const std::string actualName =
        detail::resolveActualTensorName(tensorName, dacExpr);
    return plan.distributedTensors.count(actualName) != 0;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
