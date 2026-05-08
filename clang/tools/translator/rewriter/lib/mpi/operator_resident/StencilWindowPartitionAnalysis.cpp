#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

bool isTwoDimStencilWindowReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::StencilWindow &&
           param.reads &&
           !param.writes &&
           param.tensorDims.size() == 2 &&
           param.tensorDims[0] == 0 &&
           param.tensorDims[1] == 1 &&
           param.bindOrder.size() == 2;
}

bool isTwoDimDirectWriter(const ParamAccessPlan& param,
                          const PartitionSignature& signature) {
    return param.access == ParamAccessKind::OutputDirect &&
           param.writes &&
           !param.reads &&
           param.tensorDims.size() == 2 &&
           param.tensorDims[0] == 0 &&
           param.tensorDims[1] == 1 &&
           sameOrder(param.bindOrder, signature.bindOrder);
}

}  // namespace

bool assignStencilWindow2DLayout(DacppFile* dacppFile,
                                 ShellPartitionPlan& plan,
                                 std::string& rejectReason) {
    if (!dacppFile || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.exprNode.dacExpr) {
        rejectReason = "stencil window requires dacppFile context";
        return false;
    }

    int windowReaderCount = 0;
    int directWriterCount = 0;
    if (plan.signature.bindOrder.size() != 2) {
        rejectReason = "stencil window first cut only supports 2D ownership";
        return false;
    }
    for (const auto& param : plan.params) {
        if (isTwoDimStencilWindowReader(param)) {
            ++windowReaderCount;
            continue;
        }
        if (isTwoDimDirectWriter(param, plan.signature)) {
            ++directWriterCount;
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            param.reads) {
            rejectReason =
                "stencil window direct writer must be WRITE-only until read-write old-value support exists";
            return false;
        }
        rejectReason =
            "stencil window requires one 2D window reader and one 2D WRITE-only direct writer";
        return false;
    }

    if (windowReaderCount != 1 || directWriterCount != 1) {
        rejectReason =
            "stencil window requires exactly one 2D window reader and one 2D WRITE-only direct writer";
        return false;
    }

    const mpi_rewriter::DistributedStencilSitePlan sitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc, plan.exprNode.dacExpr);
    if (!sitePlan.supported) {
        rejectReason = "stencil window requires distributed site analysis";
        return false;
    }
    if (sitePlan.hasRootBridge) {
        rejectReason = "stencil window root-bridge sites not yet supported";
        return false;
    }
    if (!sitePlan.readCacheTransitions.empty()) {
        rejectReason = "stencil window read-cache transitions not yet supported";
        return false;
    }
    if (sitePlan.followupMappings.size() != 1) {
        rejectReason = "stencil window currently requires one distributed followup mapping";
        return false;
    }
    const auto& mapping = sitePlan.followupMappings.front();
    if (mapping.rank != 2 ||
        mapping.targetRowOffset != 1 ||
        mapping.targetColOffset != 1) {
        rejectReason = "stencil window only supports +1,+1 2D followup route";
        return false;
    }
    if (sitePlan.boundaryLocalUpdates.empty()) {
        rejectReason = "stencil window requires boundary-local updates";
        return false;
    }

    plan.signature.layout = LocalLayoutKind::StencilWindow2D;
    plan.signature.linearization = "2d-row-major-window";
    return true;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
