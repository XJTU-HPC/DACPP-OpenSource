#ifndef DACPP_REWRITER_MPI_SHARED_LOWERING_CONTRACT_H
#define DACPP_REWRITER_MPI_SHARED_LOWERING_CONTRACT_H

#include <set>
#include <string>
#include <vector>

namespace clang {
class Stmt;
}  // namespace clang

namespace dacppTranslator {
namespace mpi_rewriter {

enum class LoweringContractStmtAction {
    Replace,
    Remove
};

enum class LoweringContractMaterializeTiming {
    None,
    BeforeLoop,
    EveryRun,
    LoopExit
};

enum class LoweringContractGuardDisposition {
    CompileTimeFallback,
    RuntimeAbort
};

struct LoweringContractStatement {
    const clang::Stmt* stmt = nullptr;
    LoweringContractStmtAction action = LoweringContractStmtAction::Remove;
    std::string role;
    std::string reason;
};

struct LoweringContractResidentTensor {
    std::string tensorName;
    std::string role;
};

struct LoweringContractMaterialization {
    std::string tensorName;
    LoweringContractMaterializeTiming timing =
        LoweringContractMaterializeTiming::None;
    std::string reason;
};

struct LoweringContractGuard {
    LoweringContractGuardDisposition disposition =
        LoweringContractGuardDisposition::CompileTimeFallback;
    std::string reason;
};

struct LoopLoweringContract {
    bool enabled = false;
    std::string loweringName;
    std::string acceptedReason;
    std::string rejectedReason;
    std::vector<LoweringContractStatement> statements;
    std::vector<LoweringContractResidentTensor> residentTensors;
    std::vector<LoweringContractMaterialization> materializations;
    std::vector<LoweringContractGuard> guards;
};

inline std::set<const clang::Stmt*> loweringContractRemoveStmtSet(
    const LoopLoweringContract& contract) {
    std::set<const clang::Stmt*> result;
    for (const auto& statement : contract.statements) {
        if (statement.action == LoweringContractStmtAction::Remove &&
            statement.stmt) {
            result.insert(statement.stmt);
        }
    }
    return result;
}

inline const char* loweringContractStmtActionName(
    LoweringContractStmtAction action) {
    switch (action) {
    case LoweringContractStmtAction::Replace:
        return "replace";
    case LoweringContractStmtAction::Remove:
        return "remove";
    }
    return "remove";
}

inline const char* loweringContractMaterializeTimingName(
    LoweringContractMaterializeTiming timing) {
    switch (timing) {
    case LoweringContractMaterializeTiming::None:
        return "none";
    case LoweringContractMaterializeTiming::BeforeLoop:
        return "before-loop";
    case LoweringContractMaterializeTiming::EveryRun:
        return "every-run";
    case LoweringContractMaterializeTiming::LoopExit:
        return "loop-exit";
    }
    return "none";
}

inline const char* loweringContractGuardDispositionName(
    LoweringContractGuardDisposition disposition) {
    switch (disposition) {
    case LoweringContractGuardDisposition::CompileTimeFallback:
        return "compile-time-fallback";
    case LoweringContractGuardDisposition::RuntimeAbort:
        return "runtime-abort";
    }
    return "compile-time-fallback";
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_REWRITER_MPI_SHARED_LOWERING_CONTRACT_H
