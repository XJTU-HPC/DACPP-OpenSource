#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

bool isFixedBlockCandidate(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::FixedBlock ||
           param.access == ParamAccessKind::StencilWindow;
}

bool isFlat1DWrapperParam(Param* param) {
    if (!param) {
        return false;
    }
    const std::string type = param->getType();
    if (type.find("Matrix<") != std::string::npos) {
        return false;
    }
    if (type.find("Vector<") != std::string::npos) {
        return true;
    }
    if (type.find("Tensor<") != std::string::npos) {
        return type.find(", 1>") != std::string::npos ||
               type.find(",1>") != std::string::npos;
    }
    return param->getDimension() == 1;
}

} // namespace

bool assignFixedBlockLayout(ShellPartitionPlan& plan,
                            std::string& rejectReason) {
    if (!plan.exprNode.calc || plan.signature.bindOrder.size() != 1) {
        rejectReason = "fixed block currently requires 1D ownership";
        return false;
    }

    int readerCount = 0;
    int writerCount = 0;
    int blockSize = 0;
    int blockStride = 0;
    for (auto& param : plan.params) {
        if (!isFixedBlockCandidate(param)) {
            rejectReason = "fixed block requires only fixed-block parameters";
            return false;
        }
        if (!param.voidDims.empty()) {
            rejectReason =
                "fixed block first slice rejects payload dimensions";
            return false;
        }
        if (!plan.exprNode.shell || param.paramIndex < 0 ||
            param.paramIndex >= plan.exprNode.shell->getNumShellParams()) {
            rejectReason = "fixed block shell parameter metadata is incomplete";
            return false;
        }
        ShellParam* shellParam =
            plan.exprNode.shell->getShellParam(param.paramIndex);
        if (!shellParam || shellParam->getNumSplit() != 1) {
            rejectReason =
                "fixed block first slice requires one split per parameter";
            return false;
        }
        Param* shellWrapperParam =
            plan.exprNode.shell->getParam(param.paramIndex);
        Param* calcParam = plan.exprNode.calc->getParam(param.paramIndex);
        if (!isFlat1DWrapperParam(shellWrapperParam)) {
            rejectReason =
                "fixed block first slice requires 1D wrapper tensors";
            return false;
        }
        if (inferViewRank(shellParam, calcParam) != 1) {
            rejectReason =
                "fixed block first slice requires 1D calc views";
            return false;
        }
        if (param.tensorDims.size() != 1 || param.tensorDims[0] != 0 ||
            param.bindOrder.size() != 1 ||
            !sameOrder(param.bindOrder, plan.signature.bindOrder)) {
            rejectReason = "fixed block currently requires matching 1D regular splits";
            return false;
        }
        if (param.fixedBlockSize <= 0 || param.fixedBlockStride <= 0) {
            rejectReason = "fixed block requires positive regular split metadata";
            return false;
        }
        if (param.fixedBlockSize != param.fixedBlockStride) {
            rejectReason = "fixed block requires split size to equal split stride";
            return false;
        }
        if (param.fixedBlockSize != 2) {
            rejectReason = "fixed block first slice only accepts block size 2";
            return false;
        }
        if (blockSize == 0) {
            blockSize = param.fixedBlockSize;
            blockStride = param.fixedBlockStride;
        } else if (blockSize != param.fixedBlockSize ||
                   blockStride != param.fixedBlockStride) {
            rejectReason = "fixed block parameters must share split metadata";
            return false;
        }
        if (param.reads && param.writes) {
            rejectReason = "fixed block READ_WRITE parameters are outside P5 first slice";
            return false;
        }
        if (param.reads) {
            ++readerCount;
        }
        if (param.writes) {
            ++writerCount;
        }
        const std::string elemType = calcParam->getBasicType();
        if (usesByteTransport(elemType)) {
            rejectReason = "fixed block first slice requires native MPI element datatypes";
            return false;
        }
        param.access = ParamAccessKind::FixedBlock;
    }

    if (readerCount != 1 || writerCount != 1) {
        rejectReason =
            "fixed block first slice requires one READ block and one WRITE block";
        return false;
    }

    plan.signature.layout = LocalLayoutKind::FixedBlock;
    plan.signature.linearization = "1d-fixed-block";
    return true;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
