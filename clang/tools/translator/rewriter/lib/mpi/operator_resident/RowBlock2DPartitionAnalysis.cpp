#include "DacppStructure.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

bool assignRowBlock2DLayout(ShellPartitionPlan& plan,
                            bool sawScalarParam,
                            std::string& rejectReason) {
    if (sawScalarParam) {
        rejectReason = "unsupported bind rank for phase 1/2";
        return false;
    }

    for (const auto& param : plan.params) {
        if (param.access != ParamAccessKind::DirectMapped &&
            param.access != ParamAccessKind::OutputDirect) {
            rejectReason = "2D row-block layout requires direct-only params";
            return false;
        }
        if (param.bindOrder.size() != 2 ||
            !sameOrder(param.bindOrder, plan.signature.bindOrder) ||
            param.tensorDims.size() != 2 ||
            param.tensorDims[0] != 0 ||
            param.tensorDims[1] != 1) {
            rejectReason = "2D parameter is not row-major direct";
            return false;
        }
    }
    plan.signature.layout = LocalLayoutKind::RowBlock2D;
    plan.signature.linearization = "2d-row-major";
    return true;
}

bool assignPhaseLayout(DacppFile* dacppFile,
                       ShellPartitionPlan& plan,
                       bool sawScalarParam,
                       std::string& rejectReason) {
    // Check if any parameter uses Phase 3 access kinds
    bool hasReplicatedFullTensor = false;
    bool hasRowPartitionFullRow = false;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedFullTensor) {
            hasReplicatedFullTensor = true;
        }
        if (param.access == ParamAccessKind::RowPartitionFullRow) {
            hasRowPartitionFullRow = true;
        }
    }

    // Phase 3: Full payload layouts
    if (hasReplicatedFullTensor || hasRowPartitionFullRow) {
        // Determine overall layout based on parameter composition
        if (hasRowPartitionFullRow) {
            return assignRowPartitionFullRowLayout(plan, sawScalarParam, rejectReason);
        }
        return assignReplicatedFullTensorLayout(plan, sawScalarParam, rejectReason);
    }

    if (assignStencilWindow1DLayout(dacppFile, plan, rejectReason)) {
        return true;
    }

    if (assignStencilWindow2DLayout(dacppFile, plan, rejectReason)) {
        return true;
    }

    // Phase 1/2: Legacy layouts
    const std::size_t bindRank = plan.signature.bindOrder.size();
    if (bindRank == 1) {
        return assignContiguous1DLayout(plan, rejectReason);
    }
    if (bindRank == 2) {
        return assignRowBlock2DLayout(plan, sawScalarParam, rejectReason);
    }
    rejectReason = "unsupported bind rank for phase 1/2";
    return false;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
