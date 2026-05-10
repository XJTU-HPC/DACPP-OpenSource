#include <set>
#include <string>

#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

ResidencyKind initialReadResidency(ParamAccessKind access) {
    if (access == ParamAccessKind::ReplicatedScalar) {
        return ResidencyKind::ReplicatedScalar;
    }
    return ResidencyKind::RootOnly;
}

const char* residencyKindName(ResidencyKind kind) {
    switch (kind) {
    case ResidencyKind::RootOnly:
        return "RootOnly";
    case ResidencyKind::DistributedClean:
        return "DistributedClean";
    case ResidencyKind::DistributedDirty:
        return "DistributedDirty";
    case ResidencyKind::ReplicatedScalar:
        return "ReplicatedScalar";
    case ResidencyKind::MaterializedRoot:
        return "MaterializedRoot";
    case ResidencyKind::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

} // namespace

void analyzeResidency(OperatorResidentChainPlan& chain) {
    chain.residency.clear();
    chain.materializeTensors.clear();
    if (!chain.supported) {
        return;
    }

    std::set<std::string> finalOutputs;
    if (!chain.exprPlans.empty()) {
        const auto& last = chain.exprPlans.back();
        for (const auto& param : last.params) {
            if (param.writes &&
                (param.access == ParamAccessKind::OutputDirect ||
                 param.access == ParamAccessKind::FixedBlock)) {
                finalOutputs.insert(param.actualTensorName);
            }
        }
    }

    for (std::size_t exprPos = 0; exprPos < chain.exprPlans.size(); ++exprPos) {
        auto& exprPlan = chain.exprPlans[exprPos];
        const bool isLast = exprPos + 1 == chain.exprPlans.size();
        for (auto& param : exprPlan.params) {
            if (param.reads) {
                auto& state = chain.residency[param.actualTensorName];
                if (state.tensorName.empty()) {
                    state.tensorName = param.actualTensorName;
                    state.kind = initialReadResidency(param.access);
                    state.partition = exprPlan.signature;
                    state.layout = exprPlan.signature.layout;
                    state.rootValid = true;
                    state.localValid = param.access == ParamAccessKind::ReplicatedScalar;
                }
                if (param.access != ParamAccessKind::ReplicatedScalar &&
                    state.kind == ResidencyKind::DistributedDirty &&
                    isCompatibleForChain(state.partition, exprPlan.signature)) {
                    param.readFromResident = true;
                }
            }
            if (param.writes &&
                (param.access == ParamAccessKind::OutputDirect ||
                 param.access == ParamAccessKind::FixedBlock)) {
                auto& state = chain.residency[param.actualTensorName];
                state.tensorName = param.actualTensorName;
                state.kind = ResidencyKind::DistributedDirty;
                state.partition = exprPlan.signature;
                state.layout = exprPlan.signature.layout;
                state.rootValid = false;
                state.localValid = true;
                param.writeToResident = true;
                if (isLast || finalOutputs.count(param.actualTensorName) != 0) {
                    param.materializeAfterWrite = true;
                }
            }
        }
    }

    for (const std::string& output : finalOutputs) {
        chain.materializeTensors.push_back(output);
        auto it = chain.residency.find(output);
        if (it != chain.residency.end()) {
            it->second.kind = ResidencyKind::MaterializedRoot;
            it->second.rootValid = true;
            it->second.localValid = true;
        }
    }

    for (const auto& entry : chain.residency) {
        llvm::outs() << "[DACPP][MPI][OR]   residency tensor="
                     << entry.first
                     << " state=" << residencyKindName(entry.second.kind)
                     << " layout=" << localLayoutKindName(entry.second.layout)
                     << "\n";
    }
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
