#ifndef DACPP_REWRITER_MPI_PLAN_H
#define DACPP_REWRITER_MPI_PLAN_H

#include <string>
#include <vector>

#include "mpi/operator_resident/OperatorResidentPlan.h"
#include "mpi/shared/MpiPlanBase.h"

namespace dacppTranslator {

class DacppFile;

namespace mpi_rewriter {

enum class MpiPlanKind {
    LegacyAccessPattern,
    StencilPhaseC,
    OperatorResident,
    Unsupported
};

struct MpiPlanResult {
    MpiPlanKind kind = MpiPlanKind::Unsupported;
    int exprIndex = -1;
    std::string reason;
    int operatorResidentChainId = -1;
};

// Legacy AccessPattern wrapper path.
struct LegacyWrapperPlan : MpiPlanResult {
    DacExprNode exprNode;
};

// Stencil Phase-C path.
struct StencilPhaseCPlan : MpiPlanResult {
    DacExprNode exprNode;
};

struct MpiLoweringPlan {
    MpiPlanKind overallKind = MpiPlanKind::Unsupported;
    std::vector<DacExprNode> exprNodes;
    std::vector<MpiPlanResult> exprResults;
    std::vector<ShellPartitionPlan> shellPartitionPlans;
    std::vector<OperatorResidentChainPlan> residentChains;
    std::vector<int> operatorResidentChainByExpr;
};

MpiLoweringPlan buildMpiLoweringPlan(DacppFile *dacppFile);

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
