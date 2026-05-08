#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

bool isOneDimStencilWindowReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::StencilWindow &&
           param.reads &&
           !param.writes &&
           param.tensorDims.size() == 1 &&
           param.tensorDims[0] == 0 &&
           param.bindOrder.size() == 1;
}

bool isOneDimDirectWriter(const ParamAccessPlan& param,
                          const PartitionSignature& signature) {
    return param.access == ParamAccessKind::OutputDirect &&
           param.writes &&
           !param.reads &&
           param.tensorDims.size() == 1 &&
           param.tensorDims[0] == 0 &&
           sameOrder(param.bindOrder, signature.bindOrder);
}

bool isOneDimDirectReader(const ParamAccessPlan& param,
                          const PartitionSignature& signature) {
    return param.access == ParamAccessKind::DirectMapped &&
           param.reads &&
           !param.writes &&
           param.tensorDims.size() == 1 &&
           param.tensorDims[0] == 0 &&
           sameOrder(param.bindOrder, signature.bindOrder);
}

bool isReplicatedScalarReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::ReplicatedScalar &&
           param.reads &&
           !param.writes;
}

bool declaredReadWrite(Shell* shell, int paramIndex) {
    if (!shell || paramIndex < 0 || paramIndex >= shell->getNumShellParams()) {
        return false;
    }
    ShellParam* shellParam = shell->getShellParam(paramIndex);
    return shellParam && shellParam->getRw() == IOTYPE::READ_WRITE;
}

}  // namespace

bool assignStencilWindow1DLayout(DacppFile* dacppFile,
                                 ShellPartitionPlan& plan,
                                 std::string& rejectReason) {
    if (!dacppFile || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.exprNode.dacExpr) {
        rejectReason = "stencil window requires dacppFile context";
        return false;
    }

    if (plan.signature.bindOrder.size() != 1) {
        rejectReason = "stencil window 1d requires 1D ownership";
        return false;
    }

    int windowReaderCount = 0;
    int directReaderCount = 0;
    int directWriterCount = 0;
    int scalarReaderCount = 0;
    bool windowReaderDeclaredReadWrite = false;
    bool directWriterDeclaredReadWrite = false;
    for (const auto& param : plan.params) {
        if (isOneDimStencilWindowReader(param)) {
            ++windowReaderCount;
            windowReaderDeclaredReadWrite =
                declaredReadWrite(plan.exprNode.shell, param.paramIndex);
            continue;
        }
        if (isOneDimDirectWriter(param, plan.signature)) {
            ++directWriterCount;
            directWriterDeclaredReadWrite =
                declaredReadWrite(plan.exprNode.shell, param.paramIndex);
            continue;
        }
        if (isOneDimDirectReader(param, plan.signature)) {
            ++directReaderCount;
            continue;
        }
        if (isReplicatedScalarReader(param)) {
            ++scalarReaderCount;
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
            "stencil window 1d requires one 1D window reader plus supported direct/scalar params";
        return false;
    }

    if (windowReaderCount != 1 || directWriterCount != 1 ||
        directReaderCount > 0) {
        rejectReason =
            "stencil window 1d currently requires exactly one 1D window reader, no direct readers, and one 1D WRITE-only direct writer";
        return false;
    }

    std::string writerTensorName;
    for (const auto& param : plan.params) {
        if (isOneDimDirectWriter(param, plan.signature)) {
            writerTensorName = param.actualTensorName;
            break;
        }
    }
    const OutputSyncRequirement outputSync =
        classifyOutputSyncRequirement(dacppFile, writerTensorName,
                                      plan.exprNode.dacExpr);

    const mpi_rewriter::DistributedStencilSitePlan sitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(
            dacppFile, plan.exprNode.shell, plan.exprNode.calc,
            plan.exprNode.dacExpr);
    const bool isScalarRootMaterializeShape =
        scalarReaderCount == 1 &&
        outputSync == OutputSyncRequirement::RootOnly &&
        !windowReaderDeclaredReadWrite &&
        !directWriterDeclaredReadWrite;
    const bool isReadWriteRouteShape =
        scalarReaderCount == 0 &&
        windowReaderDeclaredReadWrite &&
        directWriterDeclaredReadWrite;
    if (!sitePlan.supported) {
        if (!isScalarRootMaterializeShape) {
            rejectReason =
                "stencil window 1d requires distributed site analysis";
            return false;
        }
        plan.signature.layout = LocalLayoutKind::StencilWindow1D;
        plan.signature.linearization = "1d-window-root-materialize";
        return true;
    }
    if (!isReadWriteRouteShape) {
        rejectReason =
            "stencil window 1d currently supports only READ_WRITE route shapes or scalar root-materialize shapes";
        return false;
    }
    if (sitePlan.hasRootBridge) {
        rejectReason = "stencil window 1d root-bridge sites not yet supported";
        return false;
    }
    if (!sitePlan.readCacheTransitions.empty()) {
        rejectReason =
            "stencil window 1d read-cache transitions not yet supported";
        return false;
    }
    if (sitePlan.followupMappings.size() != 1) {
        rejectReason =
            "stencil window 1d currently requires one distributed followup mapping";
        return false;
    }
    const auto& mapping = sitePlan.followupMappings.front();
    if (mapping.rank != 1 || mapping.targetOffset != 1) {
        rejectReason =
            "stencil window 1d only supports +1 distributed followup route";
        return false;
    }

    plan.signature.layout = LocalLayoutKind::StencilWindow1D;
    plan.signature.linearization = "1d-window";
    return true;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
