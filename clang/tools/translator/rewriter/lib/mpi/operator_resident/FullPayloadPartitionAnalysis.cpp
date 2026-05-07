#include "DacppStructure.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

bool bindOrderIsPrefixOf(const std::vector<int>& prefix,
                         const std::vector<int>& full) {
    if (prefix.size() > full.size()) {
        return false;
    }
    for (std::size_t idx = 0; idx < prefix.size(); ++idx) {
        if (prefix[idx] != full[idx]) {
            return false;
        }
    }
    return true;
}

bool bindOrderIsSubsetOf(const std::vector<int>& subset,
                         const std::vector<int>& full) {
    for (int bindId : subset) {
        bool found = false;
        for (int outputBindId : full) {
            if (bindId == outputBindId) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

} // namespace

bool assignRowPartitionFullRowLayout(ShellPartitionPlan& plan,
                                     bool sawScalarParam,
                                     std::string& rejectReason) {
    // For RowPartitionFullRow: one bind domain (the index split),
    // and one or more void splits representing full dimensions
    // Pattern: tensor[{}][idx1] or tensor[idx1][{}]
    //
    // The bindOrder already contains only the index splits.
    // We need to verify consistency and calculate full sizes for codegen.

    if (sawScalarParam) {
        rejectReason = "RowPartitionFullRow scalar payload not supported";
        return false;
    }
    if (plan.signature.bindOrder.empty()) {
        rejectReason = "RowPartitionFullRow missing output ownership bind";
        return false;
    }
    if (plan.signature.bindOrder.size() != 2) {
        rejectReason = "RowPartitionFullRow currently supports 2D output ownership";
        return false;
    }

    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped ||
            param.access == ParamAccessKind::OutputDirect) {
            if (param.bindOrder != plan.signature.bindOrder) {
                rejectReason = "RowPartitionFullRow parameter bind order mismatch";
                return false;
            }
        }

        if (param.access == ParamAccessKind::RowPartitionFullRow) {
            if (!param.reads || param.writes) {
                rejectReason = "RowPartitionFullRow parameter must be read-only";
                return false;
            }
            if (param.bindOrder.size() != 1 || param.tensorDims.size() != 1 ||
                param.indexDim == -1) {
                rejectReason = "RowPartitionFullRow parameter needs exactly one indexed dim";
                return false;
            }
            if (param.voidDims.empty()) {
                rejectReason = "RowPartitionFullRow parameter needs void split";
                return false;
            }
            if (param.voidDims.size() != 1) {
                rejectReason = "RowPartitionFullRow currently supports one void dim";
                return false;
            }
            if (!((param.indexDim == 0 && param.voidDims[0] == 1) ||
                  (param.indexDim == 1 && param.voidDims[0] == 0))) {
                rejectReason = "RowPartitionFullRow currently supports 2D full payload";
                return false;
            }
            if (param.payloadDirection == PayloadDirection::Unknown) {
                rejectReason = "RowPartitionFullRow payload direction unknown";
                return false;
            }
            if (!bindOrderIsPrefixOf(param.bindOrder, plan.signature.bindOrder) &&
                !bindOrderIsSubsetOf(param.bindOrder, plan.signature.bindOrder)) {
                rejectReason = "RowPartitionFullRow indexed bind not in output ownership";
                return false;
            }
            if (param.voidDimSizes.empty() || param.voidDimSizes[0] == 0) {
                rejectReason = "RowPartitionFullRow void dim size invalid";
                return false;
            }
        }
        if (param.access == ParamAccessKind::ReplicatedFullTensor) {
            rejectReason = "RowPartitionFullRow with ReplicatedFullTensor not supported";
            return false;
        }
    }

    plan.signature.layout = LocalLayoutKind::RowPartitionFullRow;
    plan.signature.linearization = "row-partition-full-row";
    return true;
}

bool assignReplicatedFullTensorLayout(ShellPartitionPlan& plan,
                                      bool sawScalarParam,
                                      std::string& rejectReason) {
    // ReplicatedFullTensor: all ranks hold the full tensor
    // Used for {} parameters that are not scalar (size > 1)
    //
    // Strict validation:
    // 1. All void parameters must be READ-only (writes not supported)
    // 2. Calc usage must match full tensor semantics (no partial indexing)
    // 3. Tensor dimension must be reasonable (not too many dims for simple broadcast)

    int64_t totalSize = 1;
    if (plan.signature.bindOrder.size() != 1) {
        rejectReason = "ReplicatedFullTensor currently supports 1D output ownership";
        return false;
    }
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped ||
            param.access == ParamAccessKind::OutputDirect) {
            if (param.bindOrder != plan.signature.bindOrder) {
                rejectReason = "ReplicatedFullTensor direct bind order mismatch";
                return false;
            }
        }
        if (param.access == ParamAccessKind::ReplicatedFullTensor) {
            // Check 1: Must be READ-only
            if (param.writes) {
                rejectReason = "ReplicatedFullTensor write not supported";
                return false;
            }
            if (!param.reads) {
                rejectReason = "ReplicatedFullTensor parameter must be read-only";
                return false;
            }
            if (param.indexDim != -1 || !param.bindOrder.empty()) {
                rejectReason = "ReplicatedFullTensor cannot have indexed bind";
                return false;
            }

            // Check 2: Tensor dimension should be reasonable (1D or 2D for now)
            if (static_cast<int>(param.voidDims.size()) > 2) {
                rejectReason = "ReplicatedFullTensor too many dimensions (>2)";
                return false;
            }

            // Check 3: Total tensor size should not be excessive
            int64_t paramTotalSize = 1;
            bool hasUnknownDimSize = false;
            for (int64_t dimSize : param.voidDimSizes) {
                if (dimSize > 0) {
                    paramTotalSize *= dimSize;
                } else {
                    hasUnknownDimSize = true;
                }
            }
            // Reject if total size exceeds reasonable broadcast threshold
            if (!hasUnknownDimSize && paramTotalSize > 10000000) {  // 10M elements as safety limit
                rejectReason = "ReplicatedFullTensor too large for broadcast (>10M elements)";
                return false;
            }
            totalSize = std::max(totalSize, paramTotalSize);
        }
    }

    plan.signature.layout = LocalLayoutKind::ReplicatedFullTensor;
    plan.signature.linearization = "replicated-full-tensor";
    return true;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
