#include <map>
#include <string>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

void reject(ShellPartitionPlan& plan, const std::string& reason) {
    plan.supported = false;
    plan.rejectReason = reason;
    const std::string shellName =
        plan.exprNode.shell ? plan.exprNode.shell->getName() : "<null>";
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shellName
                 << " partition=rejected reason=" << reason << "\n";
}

} // namespace

const char* localLayoutKindName(LocalLayoutKind kind) {
    switch (kind) {
    case LocalLayoutKind::Contiguous1D:
        return "Contiguous1D";
    case LocalLayoutKind::RowBlock2D:
        return "RowBlock2D";
    case LocalLayoutKind::RowPartitionFullRow:
        return "RowPartitionFullRow";
    case LocalLayoutKind::ReplicatedScalar:
        return "ReplicatedScalar";
    case LocalLayoutKind::ReplicatedFullTensor:
        return "ReplicatedFullTensor";
    case LocalLayoutKind::StencilWindow1D:
        return "StencilWindow1D";
    case LocalLayoutKind::StencilWindow2D:
        return "StencilWindow2D";
    case LocalLayoutKind::FixedBlock:
        return "FixedBlock";
    case LocalLayoutKind::Unsupported:
        return "Unsupported";
    }
    return "Unsupported";
}

const char* paramAccessKindName(ParamAccessKind kind) {
    switch (kind) {
    case ParamAccessKind::DirectMapped:
        return "DirectMapped";
    case ParamAccessKind::OutputDirect:
        return "OutputDirect";
    case ParamAccessKind::ReplicatedScalar:
        return "ReplicatedScalar";
    case ParamAccessKind::ReplicatedFullTensor:
        return "ReplicatedFullTensor";
    case ParamAccessKind::RowPartitionFullRow:
        return "RowPartitionFullRow";
    case ParamAccessKind::StencilWindow:
        return "StencilWindow";
    case ParamAccessKind::FixedBlock:
        return "FixedBlock";
    case ParamAccessKind::Unsupported:
        return "Unsupported";
    }
    return "Unsupported";
}

const char* payloadDirectionName(PayloadDirection dir) {
    switch (dir) {
    case PayloadDirection::FullRow:
        return "FullRow";
    case PayloadDirection::FullColumn:
        return "FullColumn";
    case PayloadDirection::IndexedRowFullCols:
        return "IndexedRowFullCols";
    case PayloadDirection::IndexedColFullRows:
        return "IndexedColFullRows";
    case PayloadDirection::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

ShellPartitionPlan analyzeShellPartition(const DacExprNode& node) {
    return analyzeShellPartition(nullptr, node);
}

ShellPartitionPlan analyzeShellPartition(DacppFile* dacppFile,
                                         const DacExprNode& node) {
    ShellPartitionPlan plan;
    plan.exprIndex = node.exprIndex;
    plan.exprNode = node;

    Shell* shell = node.shell;
    Calc* calc = node.calc;
    if (!shell || !calc) {
        reject(plan, "missing shell or calc");
        return plan;
    }
    if (shell->getNumShellParams() != calc->getNumParams() ||
        shell->getNumShellParams() != shell->getNumParams()) {
        reject(plan, "shell/calc parameter count mismatch");
        return plan;
    }

    const auto paramModes = inferPhaseCTransportParamModes(shell, calc);
    const auto splitMeta = collectSplitBindMeta(shell);

    std::map<int, BindDomain> bindDomainsById;
    std::vector<int> outputDirectOrder;
    bool sawDirectParam = false;
    bool sawWriteParam = false;
    bool sawScalarParam = false;
    bool sawRegularSplit = false;
    std::unordered_set<int> regularBindIds;

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        if (!shellParam || !shellWrapperParam || !calcParam) {
            reject(plan, "null parameter metadata");
            return plan;
        }

        ParamAccessPlan paramPlan;
        paramPlan.paramIndex = paramIdx;
        paramPlan.shellParamName = shellWrapperParam->getName();
        paramPlan.calcParamName = calcParam->getName();
        paramPlan.actualTensorName = shellWrapperParam->getName();
        const IOTYPE transportMode = paramModes[paramIdx];
        paramPlan.reads = transportMode == IOTYPE::READ ||
                          transportMode == IOTYPE::READ_WRITE;
        paramPlan.writes = transportMode == IOTYPE::WRITE ||
                           transportMode == IOTYPE::READ_WRITE;

        bool onlyVoid = shellParam->getNumSplit() > 0;
        bool hasIndex = false;
        bool hasRegularWindow = false;
        std::vector<int> localVoidDims;  // Collect void dims for this param
        std::vector<int64_t> localVoidDimSizes;
        int localIndexDim = -1;
        int64_t localIndexDimSize = 1;

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (operator_resident::splitIsRegular(split)) {
                sawRegularSplit = true;
                hasRegularWindow = true;
                onlyVoid = false;
                auto metaIt = splitMeta.find(split->getId());
                if (metaIt == splitMeta.end()) {
                    reject(plan, "missing bind metadata");
                    return plan;
                }
                if (metaIt->second.offset != "0" && !metaIt->second.offset.empty()) {
                    reject(plan, "nonzero bind offset");
                    return plan;
                }
                const int bindId = metaIt->second.bindId;
                regularBindIds.insert(bindId);
                paramPlan.bindOrder.push_back(bindId);
                paramPlan.tensorDims.push_back(split->getDimIdx());
                if (!paramPlan.reads || paramPlan.writes) {
                    reject(plan, "regular split only supported for read-only window");
                    return plan;
                }
                paramPlan.access = ParamAccessKind::StencilWindow;
                TensorDimMapping mapping;
                mapping.tensorName = shellWrapperParam->getName();
                mapping.shellParamIndex = paramIdx;
                mapping.tensorDim = split->getDimIdx();
                mapping.kind = ShellDimKind::Split;
                mapping.bindId = bindId;
                mapping.splitName = split->getId();
                plan.mappings.push_back(mapping);

                auto& domain = bindDomainsById[bindId];
                if (domain.bindId == -1) {
                    domain.bindId = bindId;
                    domain.representative = split->getId();
                    domain.offsetExpr = "0";
                    domain.runtimeSizeParam = paramIdx;
                    domain.dimId = split->getDimIdx();
                }
                continue;
            }
            if (operator_resident::splitIsVoid(split)) {
                const int voidDimId = split ? split->getDimIdx() : splitIdx;
                localVoidDims.push_back(voidDimId);
                localVoidDimSizes.push_back(
                    operator_resident::shapeValueFor(shell, paramIdx, voidDimId));
                TensorDimMapping mapping;
                mapping.tensorName = shellWrapperParam->getName();
                mapping.shellParamIndex = paramIdx;
                mapping.tensorDim = voidDimId;
                mapping.kind = ShellDimKind::Void;
                mapping.splitName = "void";
                plan.mappings.push_back(mapping);
                continue;
            }
            onlyVoid = false;
            if (!operator_resident::splitIsIndex(split)) {
                reject(plan, "contains unsupported split kind");
                return plan;
            }

            auto metaIt = splitMeta.find(split->getId());
            if (metaIt == splitMeta.end()) {
                reject(plan, "missing bind metadata");
                return plan;
            }
            if (metaIt->second.offset != "0" && !metaIt->second.offset.empty()) {
                reject(plan, "nonzero bind offset");
                return plan;
            }

            const int bindId = metaIt->second.bindId;
            const int dimId = split->getDimIdx();
            paramPlan.bindOrder.push_back(bindId);
            paramPlan.tensorDims.push_back(dimId);
            hasIndex = true;
            localIndexDim = dimId;
            localIndexDimSize = operator_resident::shapeValueFor(shell, paramIdx, dimId);

            TensorDimMapping mapping;
            mapping.tensorName = shellWrapperParam->getName();
            mapping.shellParamIndex = paramIdx;
            mapping.tensorDim = dimId;
            mapping.kind = ShellDimKind::Index;
            mapping.bindId = bindId;
            mapping.splitName = split->getId();
            plan.mappings.push_back(mapping);

            auto& domain = bindDomainsById[bindId];
            if (domain.bindId == -1) {
                domain.bindId = bindId;
                domain.representative = split->getId();
                domain.offsetExpr = "0";
                domain.runtimeSizeParam = paramIdx;
                domain.dimId = dimId;
            }
        }

        if (hasRegularWindow) {
            // Stencil window params are already classified while walking splits.
        } else if (onlyVoid && !hasIndex) {
            if (!paramPlan.reads || paramPlan.writes) {
                reject(plan, "void output unsupported");
                return plan;
            }
            // Check if it's a scalar void parameter
            if (shellParam->getNumSplit() == 1 &&
                operator_resident::isScalarVoidParam(shell, paramIdx) &&
                operator_resident::calcUsesParamAsScalar(calc, paramIdx)) {
                paramPlan.access = ParamAccessKind::ReplicatedScalar;
                sawScalarParam = true;
            } else {
                // Full tensor void parameter (e.g., input[{}] in DFT)
                paramPlan.access = ParamAccessKind::ReplicatedFullTensor;
                paramPlan.voidDims = localVoidDims;
                paramPlan.voidDimSizes = localVoidDimSizes;
                paramPlan.indexDim = -1;
                paramPlan.indexDimSize = 1;
                paramPlan.payloadDirection = PayloadDirection::Unknown;
            }
        } else if (hasIndex) {
            // Set payload metadata before determining access kind
            paramPlan.voidDims = localVoidDims;
            paramPlan.voidDimSizes = localVoidDimSizes;
            paramPlan.indexDim = localIndexDim;
            paramPlan.indexDimSize = localIndexDimSize;

            if (static_cast<int>(paramPlan.bindOrder.size()) !=
                shellParam->getNumSplit()) {
                // This case handles mixed void + index splits (RowPartitionFullRow)
                // All void splits are skipped from bindOrder, so bindOrder.size()
                // will be less than NumSplit
                paramPlan.access = ParamAccessKind::RowPartitionFullRow;

                // Determine payload direction based on tensor dimension layout
                // For 2D: tensor[idx][{}] means indexed row with full columns
                //          tensor[{}][idx] means indexed col with full rows
                if (localVoidDims.size() == 1 && localIndexDim != -1) {
                    if (localVoidDims[0] < localIndexDim) {
                        // Void dim comes before index dim: tensor[{}][idx]
                        paramPlan.payloadDirection = PayloadDirection::IndexedColFullRows;
                    } else {
                        // Index dim comes before void dim: tensor[idx][{}]
                        paramPlan.payloadDirection = PayloadDirection::IndexedRowFullCols;
                    }
                } else {
                    paramPlan.payloadDirection = PayloadDirection::Unknown;
                }

                sawDirectParam = true;
                if (paramPlan.writes) {
                    sawWriteParam = true;
                }
            } else {
                // Pure index parameter
                sawDirectParam = true;
                if (paramPlan.writes) {
                    paramPlan.access = ParamAccessKind::OutputDirect;
                    sawWriteParam = true;

                    if (outputDirectOrder.empty()) {
                        outputDirectOrder = paramPlan.bindOrder;
                    } else if (outputDirectOrder != paramPlan.bindOrder) {
                        reject(plan, "output direct bind order mismatch");
                        return plan;
                    }
                    for (std::size_t bindIdx = 0;
                         bindIdx < paramPlan.bindOrder.size() &&
                         bindIdx < paramPlan.tensorDims.size();
                         ++bindIdx) {
                        auto& domain = bindDomainsById[paramPlan.bindOrder[bindIdx]];
                        domain.bindId = paramPlan.bindOrder[bindIdx];
                        if (domain.representative.empty()) {
                            domain.representative =
                                std::to_string(paramPlan.bindOrder[bindIdx]);
                        }
                        domain.offsetExpr = "0";
                        domain.runtimeSizeParam = paramIdx;
                        domain.dimId = paramPlan.tensorDims[bindIdx];
                    }
                } else {
                    paramPlan.access = ParamAccessKind::DirectMapped;
                }
            }
        } else {
            reject(plan, "parameter has no supported split");
            return plan;
        }

        if (transportMode == IOTYPE::READ_WRITE &&
            paramPlan.access != ParamAccessKind::OutputDirect) {
            reject(plan, "read_write parameter unsupported");
            return plan;
        }

        plan.params.push_back(paramPlan);
    }

    if (!sawDirectParam) {
        reject(plan, "no direct mapped parameter");
        return plan;
    }
    if (!sawWriteParam) {
        reject(plan, "no direct output parameter");
        return plan;
    }
    if (outputDirectOrder.empty()) {
        reject(plan, "missing output ownership domain");
        return plan;
    }

    plan.signature.bindOrder =
        operator_resident::uniquePreserveOrder(outputDirectOrder);
    for (int bindId : plan.signature.bindOrder) {
        auto it = bindDomainsById.find(bindId);
        if (it == bindDomainsById.end()) {
            reject(plan, "bind domain missing from signature");
            return plan;
        }
        plan.bindDomains.push_back(it->second);
        plan.signature.bindSizes.push_back(
            operator_resident::shapeValueFor(
                shell, static_cast<int>(it->second.runtimeSizeParam),
                it->second.dimId));
    }

    std::string layoutRejectReason;
    if (!operator_resident::assignPhaseLayout(dacppFile, plan, sawScalarParam,
                                              layoutRejectReason)) {
        reject(plan, layoutRejectReason);
        return plan;
    }

    plan.supported = true;
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shell->getName()
                 << " layout=" << localLayoutKindName(plan.signature.layout)
                 << " partition=accepted\n";
    for (const auto& param : plan.params) {
        llvm::outs() << "[DACPP][MPI][OR]   param=" << param.shellParamName
                     << " access=" << paramAccessKindName(param.access)
                     << " direction=" << payloadDirectionName(param.payloadDirection)
                     << " reads=" << (param.reads ? "1" : "0")
                     << " writes=" << (param.writes ? "1" : "0");
        if (!param.voidDims.empty()) {
            llvm::outs() << " voidDims=[";
            for (std::size_t i = 0; i < param.voidDims.size(); ++i) {
                if (i > 0) llvm::outs() << ",";
                llvm::outs() << param.voidDims[i];
                if (i < param.voidDimSizes.size() && param.voidDimSizes[i] > 0) {
                    llvm::outs() << "x" << param.voidDimSizes[i];
                } else if (i < param.voidDimSizes.size()) {
                    llvm::outs() << "xruntime";
                }
            }
            llvm::outs() << "]";
        }
        if (param.indexDim != -1) {
            llvm::outs() << " indexDim=" << param.indexDim;
            if (param.indexDimSize > 0) {
                llvm::outs() << "x" << param.indexDimSize;
            }
        }
        llvm::outs() << "\n";
    }
    for (const auto& mapping : plan.mappings) {
        llvm::outs() << "[DACPP][MPI][OR]   mapping tensor="
                     << mapping.tensorName << " dim=" << mapping.tensorDim
                     << " kind="
                     << operator_resident::shellDimKindName(mapping.kind)
                     << " bind=" << mapping.bindId << "\n";
    }
    return plan;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
