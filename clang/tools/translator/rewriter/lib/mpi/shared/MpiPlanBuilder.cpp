#include "Rewriter_MPI_Plan.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "DacppStructure.h"
#include "Rewriter.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

/// Build a DacExprNode from a flat expression index.
DacExprNode buildExprNode(DacppFile *dacppFile, int exprIdx) {
    DacExprNode node;
    node.exprIndex = exprIdx;
    Expression *expr = dacppFile->getExpression(exprIdx);
    node.expr = expr;
    if (expr) {
        node.shell = expr->getShell();
        node.calc = expr->getCalc();
        node.dacExpr = expr->getDacExpr();
    }
    return node;
}

/// Check whether a given expression index belongs to a stencil site.
bool isStencilSiteExpr(DacppFile *dacppFile, int exprIdx) {
    if (!dacppFile->hasMPIStencilSites()) {
        return false;
    }
    // MpiStencilSite stores the exprIndex of the <-> inside the loop.
    for (const auto &site : dacppFile->getMPIStencilSites()) {
        if (site.exprIndex == exprIdx) {
            return true;
        }
    }
    return false;
}

std::string operatorResidentReason(const OperatorResidentChainPlan& chain,
                                   int exprIndex) {
    for (const auto& exprPlan : chain.exprPlans) {
        if (exprPlan.exprIndex != exprIndex) {
            continue;
        }
        return std::string("shell-derived operator-resident ") +
               localLayoutKindName(exprPlan.signature.layout);
    }
    return "shell-derived operator-resident";
}

} // anonymous namespace

MpiLoweringPlan buildMpiLoweringPlan(DacppFile *dacppFile) {
    MpiLoweringPlan plan;
    if (!dacppFile) {
        return plan;
    }

    // Build expression nodes.
    for (int i = 0; i < dacppFile->getNumExpression(); ++i) {
        plan.exprNodes.push_back(buildExprNode(dacppFile, i));
    }

    plan.operatorResidentChainByExpr.assign(plan.exprNodes.size(), -1);

    for (const auto &node : plan.exprNodes) {
        plan.shellPartitionPlans.push_back(analyzeShellPartition(node));
    }

    plan.residentChains = buildOperatorResidentChains(
        dacppFile, plan.exprNodes, plan.shellPartitionPlans);
    for (std::size_t chainIdx = 0; chainIdx < plan.residentChains.size();
         ++chainIdx) {
        const auto& chain = plan.residentChains[chainIdx];
        if (!chain.supported) {
            continue;
        }
        for (const auto& exprPlan : chain.exprPlans) {
            if (exprPlan.exprIndex >= 0 &&
                exprPlan.exprIndex <
                    static_cast<int>(plan.operatorResidentChainByExpr.size())) {
                plan.operatorResidentChainByExpr[exprPlan.exprIndex] =
                    static_cast<int>(chainIdx);
            }
        }
    }

    bool hasStencilSiteNeedingPhaseC = false;
    if (dacppFile->hasMPIStencilSites()) {
        for (const auto& site : dacppFile->getMPIStencilSites()) {
            const int exprIdx = site.exprIndex;
            if (exprIdx < 0 ||
                exprIdx >=
                    static_cast<int>(plan.operatorResidentChainByExpr.size()) ||
                plan.operatorResidentChainByExpr[exprIdx] < 0) {
                hasStencilSiteNeedingPhaseC = true;
                break;
            }
        }
    }

    // Determine overall plan kind.
    if (hasStencilSiteNeedingPhaseC) {
        plan.overallKind = MpiPlanKind::StencilPhaseC;
    } else if (!plan.residentChains.empty() &&
               plan.residentChains.size() == plan.exprNodes.size()) {
        plan.overallKind = MpiPlanKind::OperatorResident;
    } else {
        plan.overallKind = MpiPlanKind::LegacyAccessPattern;
    }

    // Build per-expression results.
    for (const auto &node : plan.exprNodes) {
        MpiPlanResult result;
        result.exprIndex = node.exprIndex;

        if (node.exprIndex >= 0 &&
                   node.exprIndex <
                       static_cast<int>(plan.operatorResidentChainByExpr.size()) &&
                   plan.operatorResidentChainByExpr[node.exprIndex] >= 0) {
            result.kind = MpiPlanKind::OperatorResident;
            result.operatorResidentChainId =
                plan.operatorResidentChainByExpr[node.exprIndex];
            result.reason = operatorResidentReason(
                plan.residentChains[result.operatorResidentChainId],
                node.exprIndex);
        } else if (isStencilSiteExpr(dacppFile, node.exprIndex)) {
            result.kind = MpiPlanKind::StencilPhaseC;
            result.reason = "stencil phase-c";
        } else {
            result.kind = MpiPlanKind::LegacyAccessPattern;
            if (node.exprIndex >= 0 &&
                node.exprIndex <
                    static_cast<int>(plan.shellPartitionPlans.size())) {
                result.reason =
                    plan.shellPartitionPlans[node.exprIndex].rejectReason;
                if (result.reason.empty()) {
                    result.reason = "legacy access-pattern fallback";
                }
            }
        }

        plan.exprResults.push_back(result);
    }

    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
