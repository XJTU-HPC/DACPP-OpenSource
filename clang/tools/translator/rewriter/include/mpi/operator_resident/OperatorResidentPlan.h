#ifndef DACPP_REWRITER_MPI_OPERATOR_RESIDENT_PLAN_H
#define DACPP_REWRITER_MPI_OPERATOR_RESIDENT_PLAN_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mpi/shared/MpiPlanBase.h"

namespace dacppTranslator {
namespace mpi_rewriter {

enum class ShellDimKind {
    Index,
    Void,
    Split
};

enum class LocalLayoutKind {
    Contiguous1D,
    RowBlock2D,
    RowPartitionFullRow,
    ReplicatedScalar,
    ReplicatedFullTensor,
    StencilWindow1D,
    StencilWindow2D,
    FixedBlock,
    Unsupported
};

enum class ParamAccessKind {
    DirectMapped,
    OutputDirect,
    ReplicatedScalar,
    ReplicatedFullTensor,
    RowPartitionFullRow,
    StencilWindow,
    FixedBlock,
    Unsupported
};

enum class ResidencyKind {
    RootOnly,
    DistributedClean,
    DistributedDirty,
    ReplicatedScalar,
    MaterializedRoot,
    Unknown
};

struct BindDomain {
    int bindId = -1;
    std::string representative;
    std::string offsetExpr = "0";
    int64_t runtimeSizeParam = -1;
    int dimId = -1;
};

struct TensorDimMapping {
    std::string tensorName;
    int shellParamIndex = -1;
    int tensorDim = -1;
    ShellDimKind kind = ShellDimKind::Void;
    int bindId = -1;
    std::string splitName;
};

struct PartitionSignature {
    std::vector<int64_t> bindSizes;
    std::vector<int> bindOrder;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
    std::string linearization;
};

inline bool isCompatibleForChain(const PartitionSignature& lhs,
                                 const PartitionSignature& rhs) {
    return lhs.bindSizes == rhs.bindSizes &&
           lhs.bindOrder == rhs.bindOrder;
}

struct ParamAccessPlan {
    int paramIndex = -1;
    std::string shellParamName;
    std::string calcParamName;
    std::string actualTensorName;
    ParamAccessKind access = ParamAccessKind::Unsupported;
    bool reads = false;
    bool writes = false;
    std::vector<int> bindOrder;
    std::vector<int> tensorDims;
    bool readFromResident = false;
    bool writeToResident = false;
    bool materializeAfterWrite = false;
};

struct ShellPartitionPlan {
    bool supported = false;
    std::string rejectReason;
    int exprIndex = -1;
    DacExprNode exprNode;
    std::vector<BindDomain> bindDomains;
    std::vector<TensorDimMapping> mappings;
    PartitionSignature signature;
    std::vector<ParamAccessPlan> params;
};

struct TensorResidencyState {
    std::string tensorName;
    ResidencyKind kind = ResidencyKind::Unknown;
    PartitionSignature partition;
    LocalLayoutKind layout = LocalLayoutKind::Unsupported;
    bool rootValid = true;
    bool localValid = false;
};

struct OperatorResidentChainPlan {
    bool supported = false;
    std::string rejectReason;
    int chainId = -1;
    std::vector<DacExprNode> exprs;
    std::vector<ShellPartitionPlan> exprPlans;
    PartitionSignature signature;
    std::unordered_map<std::string, TensorResidencyState> residency;
    std::vector<std::string> materializeTensors;
};

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
