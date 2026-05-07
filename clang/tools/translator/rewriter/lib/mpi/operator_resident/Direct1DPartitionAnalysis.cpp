#include "DacppStructure.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

bool assignContiguous1DLayout(ShellPartitionPlan& plan,
                              std::string& rejectReason) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            continue;
        }
        if (param.bindOrder.size() != 1 ||
            !sameOrder(param.bindOrder, plan.signature.bindOrder)) {
            rejectReason = "1D direct parameter bind mismatch";
            return false;
        }
    }
    plan.signature.layout = LocalLayoutKind::Contiguous1D;
    plan.signature.linearization = "1d-linear";
    return true;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
