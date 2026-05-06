#include "Rewriter_MPI_Plan.h"
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

    // Determine overall plan kind.
    if (dacppFile->hasMPIStencilSites()) {
        plan.overallKind = MpiPlanKind::StencilPhaseC;
    } else {
        plan.overallKind = MpiPlanKind::LegacyAccessPattern;
    }

    // Build per-expression results.
    for (const auto &node : plan.exprNodes) {
        MpiPlanResult result;
        result.exprIndex = node.exprIndex;

        if (isStencilSiteExpr(dacppFile, node.exprIndex)) {
            result.kind = MpiPlanKind::StencilPhaseC;
        } else {
            result.kind = MpiPlanKind::LegacyAccessPattern;
        }

        plan.exprResults.push_back(result);
    }

    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
