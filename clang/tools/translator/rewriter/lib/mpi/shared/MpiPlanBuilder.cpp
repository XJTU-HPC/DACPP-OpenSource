#include "Rewriter_MPI_Plan.h"
#include "DacppStructure.h"
#include "Rewriter.h"

namespace dacppTranslator {
namespace mpi_rewriter {

MpiLoweringPlan buildMpiLoweringPlan(DacppFile *dacppFile) {
    MpiLoweringPlan plan;
    if (!dacppFile) {
        return plan;
    }

    // Build expression nodes.
    for (int i = 0; i < dacppFile->getNumExpression(); ++i) {
        Expression *expr = dacppFile->getExpression(i);
        DacExprNode node;
        node.exprIndex = i;
        node.expr = expr;
        if (expr) {
            node.shell = expr->getShell();
            node.calc = expr->getCalc();
            node.dacExpr = expr->getDacExpr();
        }
        plan.exprNodes.push_back(node);
    }

    // Determine overall plan kind using the same logic as before.
    if (dacppFile->hasMPIStencilSites()) {
        plan.overallKind = MpiPlanKind::StencilPhaseC;
    } else {
        plan.overallKind = MpiPlanKind::LegacyAccessPattern;
    }

    // Build per-expression results.
    for (const auto &node : plan.exprNodes) {
        MpiPlanResult result;
        result.exprIndex = node.exprIndex;
        result.kind = plan.overallKind;
        plan.exprResults.push_back(result);
    }

    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
